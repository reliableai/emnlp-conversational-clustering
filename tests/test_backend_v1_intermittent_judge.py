"""Unit tests for backend_v1 intermittent clustering tool-calling behavior."""

import json
from pathlib import Path

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from conversational_clustering.clustering_agent.backends.backend_v1 import (
    intermittent_clustering_judge as judge_module,
)
from conversational_clustering.clustering_agent.state import AgentState
from conversational_clustering.data_processing.data_loader import DataLoader
from conversational_clustering.models import ModelConfig


class FakeLoader(DataLoader):
    def __init__(self, items: list[str]) -> None:
        self._items = items

    def load(self) -> pd.DataFrame:
        return pd.DataFrame({"item": self._items})

    def to_context(self) -> list[str]:
        return list(self._items)

    def get_item_ids(self) -> list[str]:
        return list(self._items)

    def has_embeddings(self) -> bool:
        return False

    def get_embeddings(self) -> dict[str, list[float]]:
        raise ValueError("embeddings unavailable")


class FakeJudgeApp:
    def __init__(
        self,
        *,
        save_path: str,
        clustering_payload: dict[str, object] | None,
        final_messages: list[BaseMessage],
        seen_states: list[AgentState],
    ) -> None:
        self._save_path = save_path
        self._clustering_payload = clustering_payload
        self._final_messages = final_messages
        self._seen_states = seen_states

    def invoke(self, state: AgentState) -> AgentState:
        self._seen_states.append(state)
        if self._clustering_payload is not None:
            output_path = Path(self._save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(self._clustering_payload), encoding="utf-8")
        return {
            "messages": list(self._final_messages),
            "dataset_metadata": {"size": 2},
            "iteration_count": 0,
            "human_thought_trace": [],
            "judge_clustering_trace": [],
        }


def test_intermittent_clustering_judge_returns_saved_clustering(monkeypatch) -> None:
    seen_states: list[AgentState] = []

    def fake_build_backend_v1_tool_calling_app(
        *,
        data_loader,
        system_prompt: str,
        model_config: ModelConfig | None,
        save_path: str | None = None,
    ) -> FakeJudgeApp:
        assert data_loader is not None
        assert "human: Group by topic." in system_prompt
        assert model_config == ModelConfig(model="openai/gpt-5.4")
        assert save_path is not None
        return FakeJudgeApp(
            save_path=save_path,
            clustering_payload={
                "clusters": [
                    {"name": "Topic A", "members": ["alpha"]},
                    {"name": "Topic B", "members": ["beta"]},
                ]
            },
            final_messages=[
                ToolMessage(
                    name="save_clustering",
                    content="Clustering saved to test.json with 2 clusters and 2 items.",
                    tool_call_id="2",
                )
            ],
            seen_states=seen_states,
        )

    monkeypatch.setattr(
        judge_module,
        "build_backend_v1_tool_calling_app",
        fake_build_backend_v1_tool_calling_app,
    )
    monkeypatch.setattr(
        judge_module,
        "get_backend_v1_intermittent_clustering_judge_prompt",
        lambda _prompt_id: "Conversation:\n{conversation}",
    )

    payload = judge_module.infer_backend_v1_intermittent_clustering(
        messages=[HumanMessage(content="Group by topic.")],
        data_loader=FakeLoader(["alpha", "beta"]),
        model_config=ModelConfig(model="openai/gpt-5.4"),
    )

    assert payload == {
        "clusters": [
            {"name": "Topic A", "members": ["alpha"]},
            {"name": "Topic B", "members": ["beta"]},
        ]
    }
    assert len(seen_states) == 1
    assert seen_states[0]["messages"] == []


def test_intermittent_clustering_judge_result_accumulates_token_usage(monkeypatch) -> None:
    seen_states: list[AgentState] = []

    def fake_build_backend_v1_tool_calling_app(
        *,
        data_loader,
        system_prompt: str,
        model_config: ModelConfig | None,
        save_path: str | None = None,
    ) -> FakeJudgeApp:
        assert data_loader is not None
        assert "human: Group by topic." in system_prompt
        assert model_config == ModelConfig(model="openai/gpt-5.4")
        assert save_path is not None
        return FakeJudgeApp(
            save_path=save_path,
            clustering_payload={
                "clusters": [
                    {"name": "Topic A", "members": ["alpha"]},
                    {"name": "Topic B", "members": ["beta"]},
                ]
            },
            final_messages=[
                AIMessage(
                    content="Draft clustering",
                    additional_kwargs={"token_usage": {"input_tokens": 3, "output_tokens": 4}},
                ),
                AIMessage(
                    content="Final clustering",
                    additional_kwargs={"token_usage": {"input_tokens": 5, "output_tokens": 6}},
                ),
                ToolMessage(
                    name="save_clustering",
                    content="Clustering saved to test.json with 2 clusters and 2 items.",
                    tool_call_id="2",
                ),
            ],
            seen_states=seen_states,
        )

    monkeypatch.setattr(
        judge_module,
        "build_backend_v1_tool_calling_app",
        fake_build_backend_v1_tool_calling_app,
    )
    monkeypatch.setattr(
        judge_module,
        "get_backend_v1_intermittent_clustering_judge_prompt",
        lambda _prompt_id: "Conversation:\n{conversation}",
    )

    result = judge_module.infer_backend_v1_intermittent_clustering_result(
        messages=[HumanMessage(content="Group by topic.")],
        data_loader=FakeLoader(["alpha", "beta"]),
        model_config=ModelConfig(model="openai/gpt-5.4"),
    )

    assert result.clustering == {
        "clusters": [
            {"name": "Topic A", "members": ["alpha"]},
            {"name": "Topic B", "members": ["beta"]},
        ]
    }
    assert result.token_usage == {
        "input_tokens": 8,
        "output_tokens": 10,
        "total_tokens": 18,
    }


def test_intermittent_clustering_judge_surfaces_save_clustering_error(monkeypatch) -> None:
    seen_states: list[AgentState] = []

    def fake_build_backend_v1_tool_calling_app(
        *,
        data_loader,
        system_prompt: str,
        model_config: ModelConfig | None,
        save_path: str | None = None,
    ) -> FakeJudgeApp:
        assert data_loader is not None
        assert "human: Group by topic." in system_prompt
        assert model_config == ModelConfig(model="openai/gpt-5.4")
        assert save_path is not None
        return FakeJudgeApp(
            save_path=save_path,
            clustering_payload=None,
            final_messages=[
                ToolMessage(
                    name="save_clustering",
                    content="Could not save clustering. Reason: Missing UUIDs: u2",
                    tool_call_id="1",
                )
            ],
            seen_states=seen_states,
        )

    monkeypatch.setattr(
        judge_module,
        "build_backend_v1_tool_calling_app",
        fake_build_backend_v1_tool_calling_app,
    )
    monkeypatch.setattr(
        judge_module,
        "get_backend_v1_intermittent_clustering_judge_prompt",
        lambda _prompt_id: "Conversation:\n{conversation}",
    )

    try:
        judge_module.infer_backend_v1_intermittent_clustering(
            messages=[HumanMessage(content="Group by topic.")],
            data_loader=FakeLoader(["alpha", "beta"]),
            model_config=ModelConfig(model="openai/gpt-5.4"),
        )
    except ValueError as exc:
        assert "Missing UUIDs" in str(exc)
    else:
        raise AssertionError("Expected judge inference to fail when save_clustering fails.")

    assert len(seen_states) == 1


def test_intermittent_clustering_judge_surfaces_final_assistant_message(monkeypatch) -> None:
    seen_states: list[AgentState] = []

    def fake_build_backend_v1_tool_calling_app(
        *,
        data_loader,
        system_prompt: str,
        model_config: ModelConfig | None,
        save_path: str | None = None,
    ) -> FakeJudgeApp:
        assert data_loader is not None
        assert "human: Group by topic." in system_prompt
        assert model_config == ModelConfig(model="openai/gpt-5.4")
        assert save_path is not None
        return FakeJudgeApp(
            save_path=save_path,
            clustering_payload=None,
            final_messages=[AIMessage(content="I could not determine a complete clustering.")],
            seen_states=seen_states,
        )

    monkeypatch.setattr(
        judge_module,
        "build_backend_v1_tool_calling_app",
        fake_build_backend_v1_tool_calling_app,
    )
    monkeypatch.setattr(
        judge_module,
        "get_backend_v1_intermittent_clustering_judge_prompt",
        lambda _prompt_id: "Conversation:\n{conversation}",
    )

    try:
        judge_module.infer_backend_v1_intermittent_clustering(
            messages=[HumanMessage(content="Group by topic.")],
            data_loader=FakeLoader(["alpha", "beta"]),
            model_config=ModelConfig(model="openai/gpt-5.4"),
        )
    except ValueError as exc:
        assert "I could not determine a complete clustering." in str(exc)
    else:
        raise AssertionError("Expected judge inference to fail when no clustering is saved.")


def test_intermittent_clustering_judge_uses_baseline_v2_by_default(monkeypatch) -> None:
    seen_prompt_ids: list[str] = []
    seen_states: list[AgentState] = []

    def fake_build_backend_v1_tool_calling_app(
        *,
        data_loader,
        system_prompt: str,
        model_config: ModelConfig | None,
        save_path: str | None = None,
    ) -> FakeJudgeApp:
        assert data_loader is not None
        assert system_prompt == "Conversation:\nhuman: Group by topic."
        assert model_config == ModelConfig(model="openai/gpt-5.4")
        assert save_path is not None
        return FakeJudgeApp(
            save_path=save_path,
            clustering_payload={
                "clusters": [
                    {"name": "Topic A", "members": ["alpha"]},
                    {"name": "Topic B", "members": ["beta"]},
                ]
            },
            final_messages=[
                ToolMessage(
                    name="save_clustering",
                    content="Clustering saved to test.json with 2 clusters and 2 items.",
                    tool_call_id="2",
                )
            ],
            seen_states=seen_states,
        )

    def fake_get_prompt(prompt_id: str) -> str:
        seen_prompt_ids.append(prompt_id)
        return "Conversation:\n{conversation}"

    monkeypatch.setattr(
        judge_module,
        "build_backend_v1_tool_calling_app",
        fake_build_backend_v1_tool_calling_app,
    )
    monkeypatch.setattr(
        judge_module,
        "get_backend_v1_intermittent_clustering_judge_prompt",
        fake_get_prompt,
    )

    payload = judge_module.infer_backend_v1_intermittent_clustering(
        messages=[HumanMessage(content="Group by topic.")],
        data_loader=FakeLoader(["alpha", "beta"]),
        model_config=ModelConfig(model="openai/gpt-5.4"),
    )

    assert payload == {
        "clusters": [
            {"name": "Topic A", "members": ["alpha"]},
            {"name": "Topic B", "members": ["beta"]},
        ]
    }
    assert len(seen_states) == 1
    assert seen_prompt_ids == ["baseline_v2"]
