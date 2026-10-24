"""Unit tests for dataset and clustering tool validation and serialization behavior."""

import copy
import json
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from conversational_clustering.clustering_agent.backends.backend_v1.method import (
    build_backend_v1_tool_calling_app,
)
from conversational_clustering.clustering_agent.backends.backend_v1.tools import (
    BackendV1RuntimeState,
    ClusterSpec,
    _dataset_to_text,
    _validate_clustering,
    build_tools,
    make_append_clustering_povs_tool,
    make_save_clustering,
)
from conversational_clustering.clustering_agent.state import AgentState
from conversational_clustering.data_processing.data_loader import DataLoader


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


class FakeToolCallingLLM:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.invocations: list[list[BaseMessage]] = []
        self.bound_tool_names: list[str] = []

    def bind_tools(self, tools: list[object], **kwargs: object) -> "FakeToolCallingLLM":
        _ = kwargs
        self.bound_tool_names = [str(getattr(tool, "name", "")) for tool in tools]
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.invocations.append(list(messages))
        if not self._responses:
            raise AssertionError("No fake responses remaining")
        return self._responses.pop(0)


class InvokableApp(Protocol):
    def invoke(self, state: AgentState) -> AgentState: ...


def test_dataset_to_text_returns_joined_items() -> None:
    assert _dataset_to_text(None) == ""
    assert _dataset_to_text(FakeLoader(["alpha", "beta"])) == (
        '{"uuid":"u1","value":"alpha"}\n{"uuid":"u2","value":"beta"}'
    )


def test_validate_clustering_rejects_items_in_multiple_clusters() -> None:
    loader = FakeLoader(["alpha", "beta"])
    clusters = [
        ClusterSpec(name="A", member_uuids=["u1"]),
        ClusterSpec(name="B", member_uuids=["u1", "u2"]),
    ]

    with pytest.raises(ValueError, match="duplicated"):
        _validate_clustering(loader, clusters)


def test_validate_clustering_rejects_missing_and_extra_items() -> None:
    loader = FakeLoader(["alpha", "beta"])
    missing_clusters = [ClusterSpec(name="A", member_uuids=["u1"])]
    extra_clusters = [ClusterSpec(name="A", member_uuids=["u1", "u3"])]

    with pytest.raises(ValueError, match="Missing UUIDs"):
        _validate_clustering(loader, missing_clusters)

    with pytest.raises(ValueError, match="Extra UUIDs not in dataset"):
        _validate_clustering(loader, extra_clusters)


def test_make_save_clustering_writes_file_and_duplicate_note(tmp_path: Path) -> None:
    loader = FakeLoader(["alpha", "alpha", "beta"])

    output_path = tmp_path / "clustering.json"
    save_tool = make_save_clustering(loader, default_output_path=str(output_path))

    invoke_payload: Any = {
        "clusters": [
            {"name": "A", "member_uuids": ["u1", "u2"]},
            {"name": "B", "member_uuids": ["u3"]},
        ]
    }
    response = save_tool.invoke(invoke_payload)

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert "Clustering saved" in response
    assert "dataset contains duplicates" in response
    assert payload["clusters"][0]["name"] == "A"
    assert payload["clusters"][0]["members"] == ["alpha", "alpha"]
    assert payload["clusters"][1]["members"] == ["beta"]


def test_make_save_clustering_returns_validation_error_message() -> None:
    save_tool = make_save_clustering(FakeLoader(["alpha"]))

    invoke_payload: Any = {"clusters": []}
    response = save_tool.invoke(invoke_payload)

    assert "Could not save clustering" in response
    assert "non-empty list" in response


def test_append_clustering_povs_stores_initial_batch() -> None:
    runtime_state = BackendV1RuntimeState()
    tool = make_append_clustering_povs_tool(runtime_state=runtime_state)

    payload = json.loads(cast(Any, tool).invoke({"povs": ["genre", "age rating"]}))

    assert payload == {
        "added_povs": ["genre", "age rating"],
        "num_added": 2,
        "total_povs": 2,
    }
    assert runtime_state.clustering_pov_bank == ["genre", "age rating"]


def test_append_clustering_povs_appends_across_calls() -> None:
    runtime_state = BackendV1RuntimeState()
    tool = make_append_clustering_povs_tool(runtime_state=runtime_state)

    first_payload = json.loads(cast(Any, tool).invoke({"povs": ["genre", "age rating"]}))
    second_payload = json.loads(
        cast(Any, tool).invoke({"povs": ["length of movie title", "release decade"]})
    )

    assert first_payload == {
        "added_povs": ["genre", "age rating"],
        "num_added": 2,
        "total_povs": 2,
    }
    assert second_payload == {
        "added_povs": ["length of movie title", "release decade"],
        "num_added": 2,
        "total_povs": 4,
    }
    assert runtime_state.clustering_pov_bank == [
        "genre",
        "age rating",
        "length of movie title",
        "release decade",
    ]


def test_append_clustering_povs_deduplicates_normalized_values() -> None:
    runtime_state = BackendV1RuntimeState()
    tool = make_append_clustering_povs_tool(runtime_state=runtime_state)

    first_payload = json.loads(cast(Any, tool).invoke({"povs": ["Genre", "age   rating"]}))
    second_payload = json.loads(
        cast(Any, tool).invoke({"povs": ["genre", "Age rating", "director"]})
    )

    assert first_payload == {
        "added_povs": ["Genre", "age rating"],
        "num_added": 2,
        "total_povs": 2,
    }
    assert second_payload == {
        "added_povs": ["director"],
        "num_added": 1,
        "total_povs": 3,
    }
    assert runtime_state.clustering_pov_bank == ["Genre", "age rating", "director"]


def test_append_clustering_povs_rejects_blank_inputs() -> None:
    runtime_state = BackendV1RuntimeState()
    tool = make_append_clustering_povs_tool(runtime_state=runtime_state)

    response = cast(Any, tool).invoke({"povs": ["   ", "\n\t"]})

    assert response == (
        "Could not append clustering POVs. Reason: "
        "At least one non-empty clustering POV is required."
    )


def test_build_tools_returns_dataset_and_save_tools() -> None:
    tools = build_tools(FakeLoader(["alpha"]))

    assert [tool.name for tool in tools] == [
        "read_dataset",
        "append_clustering_povs",
        "save_clustering",
    ]


def test_seeded_app_restores_pov_bank_from_serialized_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_llm = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "append_clustering_povs",
                        "args": {"povs": ["genre", "age rating"]},
                        "id": "call-pov-1",
                    }
                ],
            ),
            AIMessage(content="Stored the POV bank."),
        ]
    )
    second_llm = FakeToolCallingLLM([AIMessage(content="Reused the stored POV bank.")])
    llms = iter([first_llm, second_llm])
    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.backend_v1.method.build_clustering_agent_llm",
        lambda model_config: next(llms),
    )

    first_app = cast(
        InvokableApp,
        build_backend_v1_tool_calling_app(
            data_loader=FakeLoader(["alpha", "beta"]),
            system_prompt="System prompt",
        ),
    )
    first_state = first_app.invoke(
        {
            "messages": [],
            "dataset_metadata": {"size": 2},
            "iteration_count": 0,
            "human_thought_trace": [],
            "judge_clustering_trace": [],
        }
    )

    second_app = cast(
        InvokableApp,
        build_backend_v1_tool_calling_app(
            data_loader=FakeLoader(["alpha", "beta"]),
            system_prompt="System prompt",
        ),
    )
    seeded_state = copy.deepcopy(first_state)
    seeded_state["iteration_count"] = 1
    second_state = second_app.invoke(seeded_state)

    serialized_runtime_state = second_state["backend_v1_runtime_state"]
    assert serialized_runtime_state["clustering_pov_bank"] == ["genre", "age rating"]


def test_app_injects_clustering_pov_bank_into_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeToolCallingLLM([AIMessage(content="Compared options.")])
    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.backend_v1.method.build_clustering_agent_llm",
        lambda model_config: llm,
    )

    app = cast(
        InvokableApp,
        build_backend_v1_tool_calling_app(
            data_loader=FakeLoader(["alpha", "beta"]),
            system_prompt="System prompt",
        ),
    )
    app.invoke(
        {
            "messages": [HumanMessage(content="Compare options.")],
            "dataset_metadata": {"size": 2},
            "iteration_count": 0,
            "human_thought_trace": [],
            "judge_clustering_trace": [],
            "backend_v1_runtime_state": {
                "clustering_pov_bank": [
                    "genre",
                    "age rating",
                    "length of movie title",
                ]
            },
        }
    )

    prompt_messages = llm.invocations[0]
    assert len(prompt_messages) >= 4
    assert isinstance(prompt_messages[0], BaseMessage)
    assert cast(str, prompt_messages[1].content) == "Dataset metadata: size=2 items."
    assert cast(str, prompt_messages[2].content) == (
        "Internal clustering POV bank so far "
        "(dataset-grounded possibilities, not necessarily user-desired):\n"
        "1. genre\n"
        "2. age rating\n"
        "3. length of movie title"
    )
    assert cast(str, prompt_messages[3].content) == "Compare options."


def test_app_records_single_turn_clustering_pov_history_with_latest_bank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "append_clustering_povs",
                        "args": {"povs": ["genre"]},
                        "id": "call-pov-1",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "append_clustering_povs",
                        "args": {"povs": ["tone"]},
                        "id": "call-pov-2",
                    }
                ],
            ),
            AIMessage(content="Stored POVs."),
        ]
    )
    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.backend_v1.method.build_clustering_agent_llm",
        lambda model_config: llm,
    )

    app = cast(
        InvokableApp,
        build_backend_v1_tool_calling_app(
            data_loader=FakeLoader(["alpha", "beta"]),
            system_prompt="System prompt",
        ),
    )
    state = app.invoke(
        {
            "messages": [],
            "dataset_metadata": {"size": 2},
            "iteration_count": 0,
            "human_thought_trace": [],
            "judge_clustering_trace": [],
        }
    )

    assert state["clustering_pov_history"] == [
        {"turn_index": 1, "povs": ["genre", "tone"]},
    ]


def test_app_records_later_turn_clustering_pov_history_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_llm = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "append_clustering_povs",
                        "args": {"povs": ["genre"]},
                        "id": "call-pov-1",
                    }
                ],
            ),
            AIMessage(content="Stored initial POV."),
        ]
    )
    second_llm = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "append_clustering_povs",
                        "args": {"povs": ["audience"]},
                        "id": "call-pov-2",
                    }
                ],
            ),
            AIMessage(content="Stored second POV."),
        ]
    )
    llms = iter([first_llm, second_llm])
    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.backend_v1.method.build_clustering_agent_llm",
        lambda model_config: next(llms),
    )

    first_app = cast(
        InvokableApp,
        build_backend_v1_tool_calling_app(
            data_loader=FakeLoader(["alpha", "beta"]),
            system_prompt="System prompt",
        ),
    )
    first_state = first_app.invoke(
        {
            "messages": [],
            "dataset_metadata": {"size": 2},
            "iteration_count": 0,
            "human_thought_trace": [],
            "judge_clustering_trace": [],
        }
    )

    second_app = cast(
        InvokableApp,
        build_backend_v1_tool_calling_app(
            data_loader=FakeLoader(["alpha", "beta"]),
            system_prompt="System prompt",
        ),
    )
    seeded_state = copy.deepcopy(first_state)
    seeded_state["iteration_count"] = 1
    second_state = second_app.invoke(seeded_state)

    assert second_state["clustering_pov_history"] == [
        {"turn_index": 1, "povs": ["genre"]},
        {"turn_index": 2, "povs": ["genre", "audience"]},
    ]


def test_app_does_not_record_clustering_pov_history_when_bank_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "append_clustering_povs",
                        "args": {"povs": ["genre"]},
                        "id": "call-pov-1",
                    }
                ],
            ),
            AIMessage(content="No new POVs."),
        ]
    )
    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.backend_v1.method.build_clustering_agent_llm",
        lambda model_config: llm,
    )

    app = cast(
        InvokableApp,
        build_backend_v1_tool_calling_app(
            data_loader=FakeLoader(["alpha", "beta"]),
            system_prompt="System prompt",
        ),
    )
    state = app.invoke(
        {
            "messages": [],
            "dataset_metadata": {"size": 2},
            "iteration_count": 1,
            "human_thought_trace": [],
            "judge_clustering_trace": [],
            "clustering_pov_history": [{"turn_index": 1, "povs": ["genre"]}],
            "backend_v1_runtime_state": {"clustering_pov_bank": ["genre"]},
        }
    )

    assert state["clustering_pov_history"] == [{"turn_index": 1, "povs": ["genre"]}]
