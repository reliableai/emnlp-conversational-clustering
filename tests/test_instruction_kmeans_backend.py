"""Unit tests for the instruction_kmeans_v1 backend tools and intermittent behavior."""

import copy
import json
import logging
import random
from pathlib import Path
from typing import Any, Protocol, cast

import httpx
import pandas as pd
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from conversational_clustering.clustering_agent.backends.instruction_kmeans_v1 import (
    intermittent_clustering_judge as judge_module,
)
from conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.app import (
    build_instruction_kmeans_v1_tool_calling_app,
)
from conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.config import (
    InstructionKMeansV1ClusteringBackendConfig,
)
from conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.method import (
    InstructionKMeansV1ConversationalClusteringMethod,
)
from conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.tools import (
    ClusterSummaryDraft,
    InstructionKMeansRuntimeState,
    build_tools,
    format_instruction_embedding_input,
    make_append_clustering_povs_tool,
    make_propose_clustering_tool,
    make_sample_dataset_tool,
    make_save_clustering_tool,
)
from conversational_clustering.clustering_agent.state import AgentState
from conversational_clustering.data_processing.data_loader import DataLoader
from conversational_clustering.models import EmbeddingModelConfig, ModelConfig


class FakeLoader(DataLoader):
    def __init__(self, items: list[str], item_ids: list[str] | None = None) -> None:
        self._items = items
        self._item_ids = item_ids or items

    def load(self) -> pd.DataFrame:
        return pd.DataFrame({"item": self._items})

    def to_context(self) -> list[str]:
        return list(self._items)

    def get_item_ids(self) -> list[str]:
        return list(self._item_ids)

    def has_embeddings(self) -> bool:
        return False

    def get_embeddings(self) -> dict[str, list[float]]:
        raise ValueError("embeddings unavailable")


class FakeStructuredSummaryLLM:
    def __init__(self, drafts: list[ClusterSummaryDraft], *, include_raw: bool = False) -> None:
        self._drafts = drafts
        self._include_raw = include_raw

    def invoke(self, prompt: str) -> ClusterSummaryDraft | dict[str, object]:
        assert "Representative points" in prompt
        if not self._drafts:
            raise AssertionError("No summary drafts remaining")
        draft = self._drafts.pop(0)
        if not self._include_raw:
            return draft
        return {
            "raw": None,
            "parsed": draft,
            "parsing_error": None,
        }


class FakeSummaryLLM:
    def __init__(self, drafts: list[ClusterSummaryDraft]) -> None:
        self._drafts = drafts

    def with_structured_output(self, schema, *, include_raw: bool = False):
        assert schema is ClusterSummaryDraft
        return FakeStructuredSummaryLLM(self._drafts, include_raw=include_raw)


class FakeToolCallingLLM:
    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = list(responses)
        self.invocations: list[list[BaseMessage]] = []
        self.bound_tool_names: list[str] = []
        self.bind_tools_kwargs: dict[str, object] = {}

    def bind_tools(self, tools: list[object], **kwargs: object) -> "FakeToolCallingLLM":
        self.bound_tool_names = [str(getattr(tool, "name", "")) for tool in tools]
        self.bind_tools_kwargs = dict(kwargs)
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.invocations.append(list(messages))
        if not self._responses:
            raise AssertionError("No fake responses remaining")
        return self._responses.pop(0)


class InvokableApp(Protocol):
    def invoke(self, state: AgentState) -> AgentState: ...


class FakeInstructionJudgeApp:
    def __init__(
        self,
        *,
        runtime_state: InstructionKMeansRuntimeState,
        proposal_payload: dict[str, object] | None,
        final_messages: list[BaseMessage],
        seen_states: list[AgentState],
    ) -> None:
        self._runtime_state = runtime_state
        self._proposal_payload = proposal_payload
        self._final_messages = final_messages
        self._seen_states = seen_states

    def invoke(self, state: AgentState) -> AgentState:
        self._seen_states.append(state)
        self._runtime_state.latest_proposal_artifacts = self._proposal_payload
        return {
            "messages": list(self._final_messages),
            "dataset_metadata": {"size": 2},
            "iteration_count": 0,
            "human_thought_trace": [],
            "judge_clustering_trace": [],
        }


def _patch_cluster_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    drafts = [
        ClusterSummaryDraft(
            name="Near zero requests",
            summary="Small magnitude items focused on the same topic.",
        ),
        ClusterSummaryDraft(
            name="Large magnitude requests",
            summary="Higher magnitude items covering a second theme.",
        ),
        ClusterSummaryDraft(
            name="Near zero requests",
            summary="Small magnitude items focused on the same topic.",
        ),
        ClusterSummaryDraft(
            name="Large magnitude requests",
            summary="Higher magnitude items covering a second theme.",
        ),
    ]
    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.tools.generate_embeddings_for_texts",
        lambda *, texts, model_config, batch_size, usage_collector=None: [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.2, 9.9],
        ][: len(texts)],
    )
    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.tools.build_llm",
        lambda model_config: FakeSummaryLLM(drafts),
    )


def test_format_instruction_embedding_input_uses_qwen_query_template() -> None:
    assert format_instruction_embedding_input(
        embedding_instruction="billing issue",
        item_text="Need refund for duplicate charge",
    ) == ("Instruct: billing issue\nQuery:Need refund for duplicate charge")


def test_sample_dataset_returns_deterministic_shuffled_window() -> None:
    items = ["alpha", "beta", "gamma", "delta"]
    tool = make_sample_dataset_tool(
        FakeLoader(items),
        shuffle_seed=7,
        sample_window_size=2,
    )

    payload = json.loads(cast(Any, tool).invoke({"start_index": 1, "limit": 5}))
    expected = [{"index": index, "value": value} for index, value in enumerate(items)]
    random.Random(7).shuffle(expected)
    for shuffled_index, row in enumerate(expected):
        row["index"] = shuffled_index

    assert payload == {
        "total_items": 4,
        "start_index": 1,
        "end_index_exclusive": 3,
        "items": expected[1:3],
    }


def test_append_clustering_povs_stores_initial_batch() -> None:
    runtime_state = InstructionKMeansRuntimeState()
    tool = make_append_clustering_povs_tool(runtime_state=runtime_state)

    payload = json.loads(cast(Any, tool).invoke({"povs": ["genre", "age rating"]}))

    assert payload == {
        "added_povs": ["genre", "age rating"],
        "num_added": 2,
        "total_povs": 2,
    }
    assert runtime_state.clustering_pov_bank == ["genre", "age rating"]


def test_append_clustering_povs_appends_across_calls() -> None:
    runtime_state = InstructionKMeansRuntimeState()
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
    runtime_state = InstructionKMeansRuntimeState()
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
    runtime_state = InstructionKMeansRuntimeState()
    tool = make_append_clustering_povs_tool(runtime_state=runtime_state)

    response = cast(Any, tool).invoke({"povs": ["   ", "\n"]})

    assert response == (
        "Could not append clustering POVs. Reason: "
        "At least one non-empty clustering POV is required."
    )


def test_propose_clustering_returns_summaries_and_records_latest_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cluster_execution(monkeypatch)
    runtime_state = InstructionKMeansRuntimeState()
    tool = make_propose_clustering_tool(
        FakeLoader(["alpha", "beta", "gamma", "delta"]),
        runtime_state=runtime_state,
        embedding_model_config=EmbeddingModelConfig(
            model="qwen/qwen3-embedding-8b",
            backend="openrouter",
        ),
        summary_model_config=ModelConfig(model="openai/gpt-5.4-nano"),
        cluster_summary_example_limit=5,
        embedding_batch_size=32,
    )

    payload = json.loads(
        cast(Any, tool).invoke(
            {
                "embedding_instruction": "semantic issue type",
                "num_clusters": 2,
            }
        )
    )

    assert payload["embedding_instruction"] == "semantic issue type"
    assert payload["num_clusters"] == 2
    assert payload["proposal_id"] == "proposal_1"
    assert [cluster["name"] for cluster in payload["clusters"]] == [
        "Near zero requests",
        "Large magnitude requests",
    ]
    assert runtime_state.latest_proposal == {
        "embedding_instruction": "semantic issue type",
        "num_clusters": 2,
        "cluster_summaries": payload["clusters"],
        "clusters": [
            {"name": "Near zero requests", "members": ["alpha", "beta"]},
            {"name": "Large magnitude requests", "members": ["gamma", "delta"]},
        ],
    }
    assert runtime_state.latest_proposal_artifacts is not None
    assert runtime_state.proposals_by_id is not None
    assert (
        runtime_state.proposals_by_id["proposal_1"]["embedding_instruction"]
        == "semantic issue type"
    )


def test_propose_clustering_can_disable_summary_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.tools.generate_embeddings_for_texts",
        lambda *, texts, model_config, batch_size, usage_collector=None: [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.2, 9.9],
        ][: len(texts)],
    )
    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.tools.build_llm",
        lambda model_config: (_ for _ in ()).throw(
            AssertionError("summary model should be disabled")
        ),
    )
    runtime_state = InstructionKMeansRuntimeState()
    tool = make_propose_clustering_tool(
        FakeLoader(["alpha", "beta", "gamma", "delta"]),
        runtime_state=runtime_state,
        embedding_model_config=EmbeddingModelConfig(
            model="qwen/qwen3-embedding-8b",
            backend="openrouter",
        ),
        summary_model_config=ModelConfig(model="openai/gpt-5.4-nano"),
        enable_cluster_summary_model=False,
        cluster_summary_example_limit=5,
        embedding_batch_size=32,
    )

    payload = json.loads(
        cast(Any, tool).invoke(
            {
                "embedding_instruction": "semantic issue type",
                "num_clusters": 2,
            }
        )
    )

    assert [cluster["name"] for cluster in payload["clusters"]] == [
        "Cluster 1",
        "Cluster 2",
    ]
    assert payload["clusters"][0]["summary"].startswith("Representative items include:")
    assert payload["clusters"][1]["summary"].startswith("Representative items include:")


def test_propose_clustering_returns_timeout_message_when_embedding_request_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.tools.generate_embeddings_for_texts",
        lambda *, texts, model_config, batch_size, usage_collector=None: (_ for _ in ()).throw(
            httpx.ReadTimeout("timed out")
        ),
    )
    runtime_state = InstructionKMeansRuntimeState()
    tool = make_propose_clustering_tool(
        FakeLoader(["alpha", "beta"]),
        runtime_state=runtime_state,
        embedding_model_config=EmbeddingModelConfig(
            model="qwen/qwen3-embedding-8b",
            backend="openrouter",
        ),
        summary_model_config=ModelConfig(model="openai/gpt-5.4-nano"),
        cluster_summary_example_limit=5,
        embedding_batch_size=32,
    )

    payload = cast(
        Any,
        tool,
    ).invoke(
        {
            "embedding_instruction": "semantic issue type",
            "num_clusters": 2,
        }
    )

    assert payload == "Could not propose clustering. Reason: request timed out."
    assert runtime_state.latest_proposal is None
    assert runtime_state.latest_proposal_artifacts is None
    assert runtime_state.proposals_by_id is None


def test_propose_clustering_surfaces_unexpected_error_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.tools.generate_embeddings_for_texts",
        lambda *, texts, model_config, batch_size, usage_collector=None: (_ for _ in ()).throw(
            RuntimeError("Cannot send a request, as the client has been closed.")
        ),
    )
    runtime_state = InstructionKMeansRuntimeState()
    tool = make_propose_clustering_tool(
        FakeLoader(["alpha", "beta"]),
        runtime_state=runtime_state,
        embedding_model_config=EmbeddingModelConfig(
            model="qwen/qwen3-embedding-8b",
            backend="openrouter",
        ),
        summary_model_config=ModelConfig(model="openai/gpt-5.4-nano"),
        cluster_summary_example_limit=5,
        embedding_batch_size=32,
    )

    payload = cast(
        Any,
        tool,
    ).invoke(
        {
            "embedding_instruction": "semantic issue type",
            "num_clusters": 2,
        }
    )

    assert (
        payload == "Could not propose clustering. Reason: unexpected RuntimeError: "
        "Cannot send a request, as the client has been closed."
    )
    assert runtime_state.latest_proposal is None
    assert runtime_state.latest_proposal_artifacts is None
    assert runtime_state.proposals_by_id is None


def test_propose_clustering_uses_local_embedding_defaults_without_openrouter_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def fake_generate_embeddings_for_texts(
        *,
        texts: list[str],
        model_config: EmbeddingModelConfig,
        batch_size: int,
        usage_collector: object | None = None,
    ) -> list[list[float]]:
        _ = usage_collector
        assert texts[0].startswith("Instruct: semantic issue type\nQuery:")
        assert model_config == EmbeddingModelConfig(
            model="Qwen/Qwen3-Embedding-8B",
            backend="huggingface_local",
        )
        assert batch_size == 64
        return [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.2, 9.9],
        ][: len(texts)]

    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.tools.generate_embeddings_for_texts",
        fake_generate_embeddings_for_texts,
    )
    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.tools.build_llm",
        lambda model_config: (_ for _ in ()).throw(
            AssertionError("summary model should be disabled")
        ),
    )
    tool = make_propose_clustering_tool(
        FakeLoader(["alpha", "beta", "gamma", "delta"]),
        runtime_state=InstructionKMeansRuntimeState(),
        embedding_model_config=InstructionKMeansV1ClusteringBackendConfig().embedding_model_config,
        summary_model_config=ModelConfig(model="openai/gpt-5.4-nano"),
        enable_cluster_summary_model=False,
        cluster_summary_example_limit=5,
        embedding_batch_size=64,
    )

    payload = json.loads(
        cast(Any, tool).invoke(
            {
                "embedding_instruction": "semantic issue type",
                "num_clusters": 2,
            }
        )
    )

    assert [cluster["name"] for cluster in payload["clusters"]] == ["Cluster 1", "Cluster 2"]


def test_save_clustering_writes_selected_proposed_clustering_without_recomputing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_cluster_execution(monkeypatch)
    runtime_state = InstructionKMeansRuntimeState()
    output_path = tmp_path / "clustering.json"
    loader = FakeLoader(["alpha", "beta", "gamma", "delta"])
    propose_tool = make_propose_clustering_tool(
        loader,
        runtime_state=runtime_state,
        embedding_model_config=EmbeddingModelConfig(
            model="qwen/qwen3-embedding-8b",
            backend="openrouter",
        ),
        summary_model_config=ModelConfig(model="openai/gpt-5.4-nano"),
        cluster_summary_example_limit=5,
        embedding_batch_size=32,
    )
    save_tool = make_save_clustering_tool(
        loader,
        runtime_state=runtime_state,
        default_output_path=str(output_path),
    )
    proposal_response = json.loads(
        cast(Any, propose_tool).invoke(
            {
                "embedding_instruction": "semantic issue type",
                "num_clusters": 2,
            }
        )
    )
    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.tools.generate_embeddings_for_texts",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("save should not recompute")),
    )

    response = json.loads(
        cast(Any, save_tool).invoke({"proposal_id": proposal_response["proposal_id"]})
    )
    saved_payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert response["proposal_id"] == "proposal_1"
    assert "Clustering saved" in response["save_message"]
    assert saved_payload == {
        "clusters": [
            {"name": "Near zero requests", "members": ["alpha", "beta"]},
            {"name": "Large magnitude requests", "members": ["gamma", "delta"]},
        ]
    }


def test_save_clustering_can_save_older_proposal_by_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_cluster_execution(monkeypatch)
    runtime_state = InstructionKMeansRuntimeState()
    output_path = tmp_path / "clustering.json"
    loader = FakeLoader(["alpha", "beta", "gamma", "delta"])
    propose_tool = make_propose_clustering_tool(
        loader,
        runtime_state=runtime_state,
        embedding_model_config=EmbeddingModelConfig(
            model="qwen/qwen3-embedding-8b",
            backend="openrouter",
        ),
        summary_model_config=ModelConfig(model="openai/gpt-5.4-nano"),
        cluster_summary_example_limit=5,
        embedding_batch_size=32,
    )
    save_tool = make_save_clustering_tool(
        loader,
        runtime_state=runtime_state,
        default_output_path=str(output_path),
    )
    first_response = json.loads(
        cast(Any, propose_tool).invoke(
            {
                "embedding_instruction": "semantic issue type",
                "num_clusters": 2,
            }
        )
    )
    second_response = json.loads(
        cast(Any, propose_tool).invoke(
            {
                "embedding_instruction": "urgency",
                "num_clusters": 2,
            }
        )
    )

    response = json.loads(
        cast(Any, save_tool).invoke({"proposal_id": first_response["proposal_id"]})
    )

    assert first_response["proposal_id"] == "proposal_1"
    assert second_response["proposal_id"] == "proposal_2"
    assert response["proposal_id"] == "proposal_1"
    assert response["embedding_instruction"] == "semantic issue type"


def test_save_clustering_requires_existing_proposal(tmp_path: Path) -> None:
    tool = make_save_clustering_tool(
        FakeLoader(["alpha", "beta"]),
        runtime_state=InstructionKMeansRuntimeState(),
        default_output_path=str(tmp_path / "clustering.json"),
    )

    response = cast(Any, tool).invoke({})

    assert response == (
        "Could not save clustering. Reason: proposal_id is required. "
        "Call propose_clustering and pass one of its returned IDs."
    )


def test_save_clustering_rejects_unknown_proposal_id(tmp_path: Path) -> None:
    tool = make_save_clustering_tool(
        FakeLoader(["alpha", "beta"]),
        runtime_state=InstructionKMeansRuntimeState(),
        default_output_path=str(tmp_path / "clustering.json"),
    )

    response = cast(Any, tool).invoke({"proposal_id": "proposal_999"})

    assert response == (
        "Could not save clustering. Reason: Unknown proposal_id 'proposal_999'. "
        "Call propose_clustering to create a proposal first."
    )


def test_build_tools_returns_sample_propose_and_save_tools() -> None:
    tools = build_tools(
        FakeLoader(["alpha"]),
        runtime_state=InstructionKMeansRuntimeState(),
        embedding_model_config=EmbeddingModelConfig(
            model="qwen/qwen3-embedding-8b",
            backend="openrouter",
        ),
        summary_model_config=ModelConfig(model="openai/gpt-5.4-nano"),
        sample_window_size=200,
        shuffle_seed=42,
        cluster_summary_example_limit=15,
        embedding_batch_size=64,
    )

    assert [tool.name for tool in tools] == [
        "sample_dataset",
        "append_clustering_povs",
        "propose_clustering",
        "save_clustering",
    ]
    propose_tool = next(tool for tool in tools if tool.name == "propose_clustering")
    propose_schema = cast(Any, propose_tool.args_schema).model_json_schema()
    assert "embedding_instruction" in propose_schema["properties"]
    assert "instruction" not in propose_schema["properties"]
    description = propose_schema["properties"]["embedding_instruction"]["description"]
    assert "embedding lens" in description
    assert "not a clustering rule" in description
    assert "cluster count" in description


def test_build_tools_can_exclude_save_tool() -> None:
    tools = build_tools(
        FakeLoader(["alpha"]),
        runtime_state=InstructionKMeansRuntimeState(),
        embedding_model_config=EmbeddingModelConfig(
            model="qwen/qwen3-embedding-8b",
            backend="openrouter",
        ),
        summary_model_config=ModelConfig(model="openai/gpt-5.4-nano"),
        sample_window_size=200,
        shuffle_seed=42,
        cluster_summary_example_limit=15,
        embedding_batch_size=64,
        include_save_tool=False,
    )

    assert [tool.name for tool in tools] == [
        "sample_dataset",
        "append_clustering_povs",
        "propose_clustering",
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
        "conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.app.build_llm",
        lambda model_config: next(llms),
    )

    config = InstructionKMeansV1ClusteringBackendConfig()
    first_app = cast(
        InvokableApp,
        build_instruction_kmeans_v1_tool_calling_app(
            data_loader=FakeLoader(["alpha", "beta"]),
            system_prompt="System prompt",
            config=config,
            runtime_state=InstructionKMeansRuntimeState(),
            enable_cluster_summary_model=False,
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
        build_instruction_kmeans_v1_tool_calling_app(
            data_loader=FakeLoader(["alpha", "beta"]),
            system_prompt="System prompt",
            config=config,
            runtime_state=InstructionKMeansRuntimeState(),
            enable_cluster_summary_model=False,
        ),
    )
    seeded_state = copy.deepcopy(first_state)
    seeded_state["iteration_count"] = 1
    second_state = second_app.invoke(seeded_state)

    serialized_runtime_state = second_state["instruction_kmeans_runtime_state"]
    assert serialized_runtime_state["clustering_pov_bank"] == ["genre", "age rating"]


def test_app_injects_clustering_pov_bank_into_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = FakeToolCallingLLM([AIMessage(content="Compared options.")])
    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.app.build_llm",
        lambda model_config: llm,
    )

    app = cast(
        InvokableApp,
        build_instruction_kmeans_v1_tool_calling_app(
            data_loader=FakeLoader(["alpha", "beta"]),
            system_prompt="System prompt",
            config=InstructionKMeansV1ClusteringBackendConfig(),
            runtime_state=InstructionKMeansRuntimeState(),
            enable_cluster_summary_model=False,
        ),
    )
    app.invoke(
        {
            "messages": [HumanMessage(content="Compare options.")],
            "dataset_metadata": {"size": 2},
            "iteration_count": 0,
            "human_thought_trace": [],
            "judge_clustering_trace": [],
            "instruction_kmeans_runtime_state": {
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
    assert cast(str, prompt_messages[1].content).startswith("Dataset metadata: size=2 items.")
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
        "conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.app.build_llm",
        lambda model_config: llm,
    )

    app = cast(
        InvokableApp,
        build_instruction_kmeans_v1_tool_calling_app(
            data_loader=FakeLoader(["alpha", "beta"]),
            system_prompt="System prompt",
            config=InstructionKMeansV1ClusteringBackendConfig(),
            runtime_state=InstructionKMeansRuntimeState(),
            enable_cluster_summary_model=False,
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
        "conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.app.build_llm",
        lambda model_config: next(llms),
    )

    config = InstructionKMeansV1ClusteringBackendConfig()
    first_app = cast(
        InvokableApp,
        build_instruction_kmeans_v1_tool_calling_app(
            data_loader=FakeLoader(["alpha", "beta"]),
            system_prompt="System prompt",
            config=config,
            runtime_state=InstructionKMeansRuntimeState(),
            enable_cluster_summary_model=False,
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
        build_instruction_kmeans_v1_tool_calling_app(
            data_loader=FakeLoader(["alpha", "beta"]),
            system_prompt="System prompt",
            config=config,
            runtime_state=InstructionKMeansRuntimeState(),
            enable_cluster_summary_model=False,
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
        "conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.app.build_llm",
        lambda model_config: llm,
    )

    app = cast(
        InvokableApp,
        build_instruction_kmeans_v1_tool_calling_app(
            data_loader=FakeLoader(["alpha", "beta"]),
            system_prompt="System prompt",
            config=InstructionKMeansV1ClusteringBackendConfig(),
            runtime_state=InstructionKMeansRuntimeState(),
            enable_cluster_summary_model=False,
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
            "instruction_kmeans_runtime_state": {"clustering_pov_bank": ["genre"]},
        }
    )

    assert state["clustering_pov_history"] == [{"turn_index": 1, "povs": ["genre"]}]


def test_seeded_app_can_save_proposal_created_in_prior_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.tools.generate_embeddings_for_texts",
        lambda *, texts, model_config, batch_size, usage_collector=None: [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.2, 9.9],
        ][: len(texts)],
    )
    first_llm = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "propose_clustering",
                        "args": {
                            "embedding_instruction": "semantic issue type",
                            "num_clusters": 2,
                        },
                        "id": "call-propose-1",
                    }
                ],
            ),
            AIMessage(content="Proposal ready."),
        ]
    )
    second_llm = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "save_clustering",
                        "args": {"proposal_id": "proposal_1"},
                        "id": "call-save-1",
                    }
                ],
            ),
            AIMessage(content="Saved."),
        ]
    )
    llms = iter([first_llm, second_llm])
    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.app.build_llm",
        lambda model_config: next(llms),
    )

    loader = FakeLoader(["alpha", "beta", "gamma", "delta"])
    config = InstructionKMeansV1ClusteringBackendConfig()
    shared_app = cast(
        InvokableApp,
        build_instruction_kmeans_v1_tool_calling_app(
            data_loader=loader,
            system_prompt="System prompt",
            config=config,
            runtime_state=InstructionKMeansRuntimeState(),
            save_path=str(tmp_path / "shared.json"),
            enable_cluster_summary_model=False,
        ),
    )
    proposed_state = shared_app.invoke(
        {
            "messages": [],
            "dataset_metadata": {"size": 4},
            "iteration_count": 0,
            "human_thought_trace": [],
            "judge_clustering_trace": [],
        }
    )

    seeded_state = copy.deepcopy(proposed_state)
    branched_app = cast(
        InvokableApp,
        build_instruction_kmeans_v1_tool_calling_app(
            data_loader=loader,
            system_prompt="System prompt",
            config=config,
            runtime_state=InstructionKMeansRuntimeState(),
            save_path=str(tmp_path / "branched.json"),
            enable_cluster_summary_model=False,
        ),
    )
    final_state = branched_app.invoke(seeded_state)

    tool_messages = [
        message for message in final_state["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 2
    save_payload = json.loads(cast(str, tool_messages[-1].content))
    assert save_payload["proposal_id"] == "proposal_1"
    assert json.loads((tmp_path / "branched.json").read_text(encoding="utf-8")) == {
        "clusters": [
            {"name": "Cluster 1", "members": ["alpha", "beta"]},
            {"name": "Cluster 2", "members": ["gamma", "delta"]},
        ]
    }
    assert first_llm.bind_tools_kwargs == {"parallel_tool_calls": False}
    assert second_llm.bind_tools_kwargs == {"parallel_tool_calls": False}


def test_seeded_app_continues_proposal_ids_from_serialized_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.tools.generate_embeddings_for_texts",
        lambda *, texts, model_config, batch_size, usage_collector=None: [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.2, 9.9],
        ][: len(texts)],
    )
    first_llm = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "propose_clustering",
                        "args": {
                            "embedding_instruction": "semantic issue type",
                            "num_clusters": 2,
                        },
                        "id": "call-propose-1",
                    }
                ],
            ),
            AIMessage(content="First proposal ready."),
        ]
    )
    second_llm = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "propose_clustering",
                        "args": {
                            "embedding_instruction": "urgency",
                            "num_clusters": 2,
                        },
                        "id": "call-propose-2",
                    }
                ],
            ),
            AIMessage(content="Second proposal ready."),
        ]
    )
    llms = iter([first_llm, second_llm])
    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.app.build_llm",
        lambda model_config: next(llms),
    )

    loader = FakeLoader(["alpha", "beta", "gamma", "delta"])
    config = InstructionKMeansV1ClusteringBackendConfig()
    first_app = cast(
        InvokableApp,
        build_instruction_kmeans_v1_tool_calling_app(
            data_loader=loader,
            system_prompt="System prompt",
            config=config,
            runtime_state=InstructionKMeansRuntimeState(),
            enable_cluster_summary_model=False,
        ),
    )
    first_state = first_app.invoke(
        {
            "messages": [],
            "dataset_metadata": {"size": 4},
            "iteration_count": 0,
            "human_thought_trace": [],
            "judge_clustering_trace": [],
        }
    )

    second_app = cast(
        InvokableApp,
        build_instruction_kmeans_v1_tool_calling_app(
            data_loader=loader,
            system_prompt="System prompt",
            config=config,
            runtime_state=InstructionKMeansRuntimeState(),
            enable_cluster_summary_model=False,
        ),
    )
    second_state = second_app.invoke(copy.deepcopy(first_state))

    tool_messages = [
        message for message in second_state["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 2
    second_proposal_payload = json.loads(cast(str, tool_messages[-1].content))
    assert second_proposal_payload["proposal_id"] == "proposal_2"
    serialized_runtime_state = second_state["instruction_kmeans_runtime_state"]
    assert serialized_runtime_state["next_proposal_index"] == 3
    assert set(serialized_runtime_state["proposals_by_id"]) == {"proposal_1", "proposal_2"}


def test_intermittent_clustering_judge_returns_latest_proposal_without_save_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_states: list[AgentState] = []

    def fake_build_instruction_kmeans_v1_tool_calling_app(
        *,
        data_loader: DataLoader | None,
        system_prompt: str,
        config: InstructionKMeansV1ClusteringBackendConfig,
        runtime_state: InstructionKMeansRuntimeState,
        save_path: str | None = None,
        include_save_tool: bool = True,
        include_pov_bank_tool: bool = True,
        enable_cluster_summary_model: bool = True,
    ) -> FakeInstructionJudgeApp:
        assert data_loader is not None
        assert '{"content":"Group by topic.","role":"human"}' in system_prompt
        assert config.model_config == config.intermittent_clustering_judge_model
        assert save_path is None
        assert include_save_tool is False
        assert include_pov_bank_tool is False
        assert enable_cluster_summary_model is False
        return FakeInstructionJudgeApp(
            runtime_state=runtime_state,
            proposal_payload={
                "embedding_instruction": "topic",
                "num_clusters": 2,
                "clusters": [],
                "clustering": {
                    "clusters": [
                        {"name": "Topic A", "members": ["alpha"]},
                        {"name": "Topic B", "members": ["beta"]},
                    ]
                },
                "intermittent_clustering": {
                    "embedding_instruction": "topic",
                    "num_clusters": 2,
                    "cluster_summaries": [],
                    "clusters": [
                        {"name": "Topic A", "members": ["alpha"]},
                        {"name": "Topic B", "members": ["beta"]},
                    ],
                },
                "save_clusters": [],
            },
            final_messages=[
                ToolMessage(
                    name="propose_clustering",
                    content="Proposed clustering ready.",
                    tool_call_id="1",
                )
            ],
            seen_states=seen_states,
        )

    monkeypatch.setattr(
        judge_module,
        "build_instruction_kmeans_v1_tool_calling_app",
        fake_build_instruction_kmeans_v1_tool_calling_app,
    )
    monkeypatch.setattr(
        judge_module,
        "get_instruction_kmeans_v1_intermittent_clustering_judge_prompt",
        lambda _prompt_id: "Conversation:\n{conversation}",
    )

    result = judge_module.infer_instruction_kmeans_v1_intermittent_clustering_result(
        messages=[HumanMessage(content="Group by topic.")],
        data_loader=FakeLoader(["alpha", "beta"]),
        config=InstructionKMeansV1ClusteringBackendConfig(),
    )

    assert result.clustering == {
        "clusters": [
            {"name": "Topic A", "members": ["alpha"]},
            {"name": "Topic B", "members": ["beta"]},
        ]
    }
    assert len(seen_states) == 1
    assert seen_states[0]["messages"] == []


def test_intermittent_clustering_judge_prompt_preserves_tool_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_build_instruction_kmeans_v1_tool_calling_app(
        *,
        data_loader: DataLoader | None,
        system_prompt: str,
        config: InstructionKMeansV1ClusteringBackendConfig,
        runtime_state: InstructionKMeansRuntimeState,
        save_path: str | None = None,
        include_save_tool: bool = True,
        include_pov_bank_tool: bool = True,
        enable_cluster_summary_model: bool = True,
    ) -> FakeInstructionJudgeApp:
        assert data_loader is not None
        assert config.model_config == config.intermittent_clustering_judge_model
        assert save_path is None
        assert include_save_tool is False
        assert include_pov_bank_tool is False
        assert enable_cluster_summary_model is False
        assert (
            '{"content":"Compare two proposals, then save the second one.","role":"human"}'
            in system_prompt
        )
        assert '"role":"assistant","tool_calls":' in system_prompt
        assert '"name":"propose_clustering"' in system_prompt
        assert '"tool_call_id":"call-propose-2"' in system_prompt
        assert '"proposal_2"' in system_prompt
        assert '{"content":"Save proposal_2.","role":"human"}' in system_prompt
        return FakeInstructionJudgeApp(
            runtime_state=runtime_state,
            proposal_payload={
                "clustering": {"clusters": [{"name": "Billing", "members": ["refund request"]}]}
            },
            final_messages=[
                ToolMessage(
                    name="propose_clustering",
                    content='{"proposal_id":"proposal_3"}',
                    tool_call_id="call-propose-3",
                )
            ],
            seen_states=[],
        )

    monkeypatch.setattr(
        judge_module,
        "build_instruction_kmeans_v1_tool_calling_app",
        fake_build_instruction_kmeans_v1_tool_calling_app,
    )
    monkeypatch.setattr(
        judge_module,
        "get_instruction_kmeans_v1_intermittent_clustering_judge_prompt",
        lambda _prompt_id: "Conversation:\n{conversation}",
    )

    result = judge_module.infer_instruction_kmeans_v1_intermittent_clustering_result(
        messages=[
            HumanMessage(content="Compare two proposals, then save the second one."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "propose_clustering",
                        "args": {
                            "embedding_instruction": "topic",
                            "num_clusters": 2,
                        },
                        "id": "call-propose-2",
                    }
                ],
            ),
            ToolMessage(
                name="propose_clustering",
                content=(
                    '{"proposal_id":"proposal_2","clusters":'
                    '[{"name":"Billing","summary":"Invoices and refunds"}]}'
                ),
                tool_call_id="call-propose-2",
            ),
            HumanMessage(content="Save proposal_2."),
        ],
        data_loader=FakeLoader(["refund request", "invoice question"]),
        config=InstructionKMeansV1ClusteringBackendConfig(),
    )

    assert result.clustering == {"clusters": [{"name": "Billing", "members": ["refund request"]}]}


def test_format_conversation_serializes_tool_calls_and_tool_responses_as_jsonl() -> None:
    transcript = judge_module._format_conversation(
        [
            HumanMessage(content="Compare proposals."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "propose_clustering",
                        "args": {"embedding_instruction": "topic", "num_clusters": 2},
                        "id": "call-propose-2",
                    }
                ],
            ),
            ToolMessage(
                name="propose_clustering",
                content=(
                    '{"proposal_id":"proposal_2","clusters":'
                    '[{"name":"Billing","summary":"Invoices and refunds"}]}'
                ),
                tool_call_id="call-propose-2",
            ),
        ]
    )

    entries = [json.loads(line) for line in transcript.splitlines()]
    assert entries == [
        {
            "content": "Compare proposals.",
            "role": "human",
        },
        {
            "content": "",
            "role": "assistant",
            "tool_calls": [
                {
                    "args": {
                        "embedding_instruction": "topic",
                        "num_clusters": 2,
                    },
                    "id": "call-propose-2",
                    "name": "propose_clustering",
                    "type": "tool_call",
                }
            ],
        },
        {
            "content": {
                "clusters": [{"name": "Billing", "summary": "Invoices and refunds"}],
                "proposal_id": "proposal_2",
            },
            "name": "propose_clustering",
            "role": "tool",
            "tool_call_id": "call-propose-2",
        },
    ]


def test_format_conversation_extracts_text_from_block_content() -> None:
    transcript = judge_module._format_conversation(
        [
            HumanMessage(content=[{"type": "text", "text": "Group by topic."}]),
            AIMessage(
                content=[
                    {"type": "text", "text": "I will compare two options."},
                    {"type": "metadata", "source": "provider"},
                ]
            ),
        ]
    )

    entries = [json.loads(line) for line in transcript.splitlines()]
    assert entries == [
        {
            "content": "Group by topic.",
            "role": "human",
        },
        {
            "content": 'I will compare two options.\n{"source":"provider","type":"metadata"}',
            "role": "assistant",
        },
    ]


def test_intermittent_clustering_judge_retries_after_failed_latest_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_states: list[AgentState] = []

    class FakeRetryingInstructionJudgeApp:
        def __init__(self, *, runtime_state: InstructionKMeansRuntimeState) -> None:
            self._runtime_state = runtime_state
            self._call_count = 0

        def invoke(self, state: AgentState) -> AgentState:
            self._call_count += 1
            seen_states.append(copy.deepcopy(state))
            if self._call_count == 1:
                self._runtime_state.latest_proposal_artifacts = {
                    "clustering": {
                        "clusters": [{"name": "Stale cluster", "members": ["alpha", "beta"]}]
                    }
                }
                return {
                    "messages": [
                        ToolMessage(
                            name="propose_clustering",
                            content='{"proposal_id":"proposal_1"}',
                            tool_call_id="1",
                        ),
                        ToolMessage(
                            name="propose_clustering",
                            content="Could not propose clustering. Reason: Invalid cluster count.",
                            tool_call_id="2",
                        ),
                    ],
                    "dataset_metadata": {"size": 2},
                    "iteration_count": 0,
                    "human_thought_trace": [],
                    "judge_clustering_trace": [],
                }
            self._runtime_state.latest_proposal_artifacts = {
                "clustering": {
                    "clusters": [{"name": "Recovered cluster", "members": ["beta", "alpha"]}]
                }
            }
            next_state = cast(AgentState, dict(state))
            next_state["messages"] = [
                *state["messages"],
                ToolMessage(
                    name="propose_clustering",
                    content="Proposed clustering ready.",
                    tool_call_id="3",
                ),
            ]
            return next_state

    def fake_build_instruction_kmeans_v1_tool_calling_app(
        *,
        data_loader: DataLoader | None,
        system_prompt: str,
        config: InstructionKMeansV1ClusteringBackendConfig,
        runtime_state: InstructionKMeansRuntimeState,
        save_path: str | None = None,
        include_save_tool: bool = True,
        include_pov_bank_tool: bool = True,
        enable_cluster_summary_model: bool = True,
    ) -> FakeRetryingInstructionJudgeApp:
        assert data_loader is not None
        assert '{"content":"Group by topic.","role":"human"}' in system_prompt
        assert config.model_config == config.intermittent_clustering_judge_model
        assert save_path is None
        assert include_save_tool is False
        assert include_pov_bank_tool is False
        assert enable_cluster_summary_model is False
        return FakeRetryingInstructionJudgeApp(runtime_state=runtime_state)

    monkeypatch.setattr(
        judge_module,
        "build_instruction_kmeans_v1_tool_calling_app",
        fake_build_instruction_kmeans_v1_tool_calling_app,
    )
    monkeypatch.setattr(
        judge_module,
        "get_instruction_kmeans_v1_intermittent_clustering_judge_prompt",
        lambda _prompt_id: "Conversation:\n{conversation}",
    )

    result = judge_module.infer_instruction_kmeans_v1_intermittent_clustering_result(
        messages=[HumanMessage(content="Group by topic.")],
        data_loader=FakeLoader(["alpha", "beta"]),
        config=InstructionKMeansV1ClusteringBackendConfig(),
    )

    assert result.clustering == {
        "clusters": [{"name": "Recovered cluster", "members": ["beta", "alpha"]}]
    }
    assert len(seen_states) == 2
    retry_message = seen_states[1]["messages"][-1]
    assert isinstance(retry_message, HumanMessage)
    assert (
        retry_message.content
        == "The last propose_clustering call failed. Continue the clustering process and "
        "call propose_clustering again with corrected arguments. Do not stop until you "
        "have produced a valid clustering."
    )


def test_intermittent_clustering_judge_surfaces_propose_clustering_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_states: list[AgentState] = []

    def fake_build_instruction_kmeans_v1_tool_calling_app(
        *,
        data_loader: DataLoader | None,
        system_prompt: str,
        config: InstructionKMeansV1ClusteringBackendConfig,
        runtime_state: InstructionKMeansRuntimeState,
        save_path: str | None = None,
        include_save_tool: bool = True,
        include_pov_bank_tool: bool = True,
        enable_cluster_summary_model: bool = True,
    ) -> FakeInstructionJudgeApp:
        assert data_loader is not None
        assert '{"content":"Group by topic.","role":"human"}' in system_prompt
        assert config.model_config == config.intermittent_clustering_judge_model
        assert save_path is None
        assert include_save_tool is False
        assert include_pov_bank_tool is False
        assert enable_cluster_summary_model is False
        return FakeInstructionJudgeApp(
            runtime_state=runtime_state,
            proposal_payload=None,
            final_messages=[
                ToolMessage(
                    name="propose_clustering",
                    content="Could not propose clustering. Reason: Dataset is empty.",
                    tool_call_id="1",
                )
            ],
            seen_states=seen_states,
        )

    monkeypatch.setattr(
        judge_module,
        "build_instruction_kmeans_v1_tool_calling_app",
        fake_build_instruction_kmeans_v1_tool_calling_app,
    )
    monkeypatch.setattr(
        judge_module,
        "get_instruction_kmeans_v1_intermittent_clustering_judge_prompt",
        lambda _prompt_id: "Conversation:\n{conversation}",
    )

    with pytest.raises(ValueError, match="Could not propose clustering. Reason: Dataset is empty."):
        judge_module.infer_instruction_kmeans_v1_intermittent_clustering_result(
            messages=[HumanMessage(content="Group by topic.")],
            data_loader=FakeLoader(["alpha", "beta"]),
            config=InstructionKMeansV1ClusteringBackendConfig(),
        )

    assert len(seen_states) == judge_module._INTERMITTENT_JUDGE_MAX_RECOVERY_ATTEMPTS + 1
    assert isinstance(seen_states[-1]["messages"][-1], HumanMessage)


def test_capture_intermittent_clustering_uses_hidden_judge_even_with_cached_proposal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    method = InstructionKMeansV1ConversationalClusteringMethod(
        InstructionKMeansV1ClusteringBackendConfig()
    )
    method._runtime_state.latest_proposal = {
        "embedding_instruction": "semantic issue type",
        "num_clusters": 2,
        "cluster_summaries": [
            {"name": "A", "size": 2, "summary": "topic", "representative_points": ["alpha"]}
        ],
        "clusters": [{"name": "A", "members": ["alpha", "beta"]}],
    }
    judge_result = {
        "clusters": [{"name": "Judge cluster", "members": ["beta", "alpha"]}],
    }
    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.intermittent_clustering_judge.infer_instruction_kmeans_v1_intermittent_clustering_result",
        lambda **kwargs: type(
            "Result",
            (),
            {
                "clustering": judge_result,
                "token_usage": {"input_tokens": 5, "output_tokens": 6, "total_tokens": 11},
            },
        )(),
    )

    state = cast(
        Any,
        {
            "messages": [HumanMessage(content="Show me the latest grouping.")],
            "dataset_metadata": {"size": 2},
            "iteration_count": 0,
        },
    )
    output_path = tmp_path / "judge.json"
    resolved_state = method.capture_intermittent_clustering(
        state=state,
        data_loader=FakeLoader(["alpha", "beta"]),
        logger=logging.getLogger("test.instruction_kmeans"),
        output_path=str(output_path),
    )

    assert resolved_state["latest_proposed_clustering"] == judge_result
    assert resolved_state["judge_clustering_trace"] == [
        {
            "turn_index": 1,
            "clustering": judge_result,
        }
    ]
    assert resolved_state["intermittent_clustering_token_usage"] == {
        "input_tokens": 5,
        "output_tokens": 6,
        "total_tokens": 11,
    }
    snapshot_path = output_path.with_name("judge.0001.json")
    assert json.loads(snapshot_path.read_text(encoding="utf-8")) == judge_result


def test_capture_intermittent_clustering_uses_hidden_judge_before_first_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method = InstructionKMeansV1ConversationalClusteringMethod(
        InstructionKMeansV1ClusteringBackendConfig()
    )
    monkeypatch.setattr(
        "conversational_clustering.clustering_agent.backends.instruction_kmeans_v1.intermittent_clustering_judge.infer_instruction_kmeans_v1_intermittent_clustering_result",
        lambda **kwargs: type(
            "Result",
            (),
            {
                "clustering": {"clusters": [{"name": "A", "members": ["alpha", "beta"]}]},
                "token_usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
            },
        )(),
    )

    state = cast(
        Any,
        {
            "messages": [HumanMessage(content="Cluster these by topic.")],
            "dataset_metadata": {"size": 2},
            "iteration_count": 0,
        },
    )
    resolved_state = method.capture_intermittent_clustering(
        state=state,
        data_loader=FakeLoader(["alpha", "beta"]),
        logger=logging.getLogger("test.instruction_kmeans"),
        output_path=None,
    )

    assert resolved_state["latest_proposed_clustering"] == {
        "clusters": [{"name": "A", "members": ["alpha", "beta"]}]
    }
    assert resolved_state["judge_clustering_trace"] == [
        {
            "turn_index": 1,
            "clustering": {"clusters": [{"name": "A", "members": ["alpha", "beta"]}]},
        }
    ]
    assert resolved_state["intermittent_clustering_token_usage"] == {
        "input_tokens": 3,
        "output_tokens": 4,
        "total_tokens": 7,
    }
