"""LangGraph app construction and runtime state hydration for instruction_kmeans_v1."""

import copy
from typing import Any, cast

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from ....accounting import UsageCollector, extract_response_model, extract_token_usage
from ....data_processing.data_loader import DataLoader
from ....models import build_llm
from ...state import AgentState
from ..pov_bank import (
    build_clustering_pov_bank_message,
    coerce_clustering_pov_bank,
    upsert_clustering_pov_history,
)
from .config import InstructionKMeansV1ClusteringBackendConfig
from .tools import InstructionKMeansRuntimeState, build_tools


def _serialize_runtime_state(
    runtime_state: InstructionKMeansRuntimeState,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "next_proposal_index": runtime_state.next_proposal_index,
    }
    if runtime_state.clustering_pov_bank is not None:
        payload["clustering_pov_bank"] = copy.deepcopy(runtime_state.clustering_pov_bank)
    if runtime_state.latest_proposal is not None:
        payload["latest_proposal"] = copy.deepcopy(runtime_state.latest_proposal)
    if runtime_state.latest_proposal_artifacts is not None:
        payload["latest_proposal_artifacts"] = copy.deepcopy(
            runtime_state.latest_proposal_artifacts
        )
    if runtime_state.proposals_by_id:
        payload["proposals_by_id"] = copy.deepcopy(runtime_state.proposals_by_id)
    return payload


def _hydrate_runtime_state(
    runtime_state: InstructionKMeansRuntimeState,
    *,
    state: AgentState,
) -> None:
    serialized = state.get("instruction_kmeans_runtime_state")
    runtime_state.latest_proposal = None
    runtime_state.latest_proposal_artifacts = None
    runtime_state.proposals_by_id = None
    runtime_state.clustering_pov_bank = None
    runtime_state.next_proposal_index = 1
    if not isinstance(serialized, dict):
        return

    clustering_pov_bank = serialized.get("clustering_pov_bank")
    if isinstance(clustering_pov_bank, list):
        normalized_pov_bank = coerce_clustering_pov_bank(
            [pov for pov in clustering_pov_bank if isinstance(pov, str)]
        )
        runtime_state.clustering_pov_bank = copy.deepcopy(normalized_pov_bank)

    latest_proposal = serialized.get("latest_proposal")
    if isinstance(latest_proposal, dict):
        runtime_state.latest_proposal = copy.deepcopy(latest_proposal)

    latest_proposal_artifacts = serialized.get("latest_proposal_artifacts")
    if isinstance(latest_proposal_artifacts, dict):
        runtime_state.latest_proposal_artifacts = copy.deepcopy(latest_proposal_artifacts)

    proposals_by_id = serialized.get("proposals_by_id")
    if isinstance(proposals_by_id, dict):
        normalized_proposals_by_id: dict[str, dict[str, object]] = {}
        for proposal_id, proposal in proposals_by_id.items():
            if not isinstance(proposal_id, str) or not isinstance(proposal, dict):
                continue
            normalized_proposals_by_id[proposal_id] = copy.deepcopy(proposal)
        if normalized_proposals_by_id:
            runtime_state.proposals_by_id = normalized_proposals_by_id

    next_proposal_index = serialized.get("next_proposal_index")
    if isinstance(next_proposal_index, int) and not isinstance(next_proposal_index, bool):
        runtime_state.next_proposal_index = max(1, next_proposal_index)


def _state_updates_from_runtime_state(
    runtime_state: InstructionKMeansRuntimeState,
) -> dict[str, object]:
    updates: dict[str, object] = {
        "instruction_kmeans_runtime_state": _serialize_runtime_state(runtime_state),
    }
    if runtime_state.latest_proposal is not None:
        updates["latest_proposed_clustering"] = copy.deepcopy(runtime_state.latest_proposal)
    return updates


def build_instruction_kmeans_v1_tool_calling_app(
    *,
    data_loader: DataLoader | None,
    system_prompt: str,
    config: InstructionKMeansV1ClusteringBackendConfig,
    runtime_state: InstructionKMeansRuntimeState,
    save_path: str | None = None,
    usage_collector: UsageCollector | None = None,
    include_save_tool: bool = True,
    include_pov_bank_tool: bool = True,
    enable_cluster_summary_model: bool = True,
) -> object:
    usage_collector = usage_collector or UsageCollector()
    llm = build_llm(config.model_config)
    tools = build_tools(
        data_loader,
        runtime_state=runtime_state,
        embedding_model_config=config.embedding_model_config,
        summary_model_config=config.summary_model_config,
        enable_cluster_summary_model=enable_cluster_summary_model,
        sample_window_size=config.sample_window_size,
        shuffle_seed=config.shuffle_seed,
        cluster_summary_example_limit=config.cluster_summary_example_limit,
        embedding_batch_size=config.embedding_batch_size,
        save_path=save_path,
        usage_collector=usage_collector,
        include_save_tool=include_save_tool,
        include_pov_bank_tool=include_pov_bank_tool,
    )
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

    def agent_node(state: AgentState) -> dict[str, object]:
        _hydrate_runtime_state(runtime_state, state=state)
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt), *messages]
        dataset_metadata = state.get("dataset_metadata") or {}
        dataset_size = dataset_metadata.get("size")
        if dataset_size is not None:
            metadata_message = SystemMessage(
                content=(
                    f"Dataset metadata: size={dataset_size} items. "
                    "Dataset sampling with `sample_dataset` is limited to "
                    f"{config.sample_window_size} items per tool call."
                )
            )
            messages = [messages[0], metadata_message, *messages[1:]]
        if runtime_state.clustering_pov_bank:
            pov_bank_message = build_clustering_pov_bank_message(runtime_state.clustering_pov_bank)
            pov_message_index = 2 if dataset_size is not None else 1
            messages = [
                *messages[:pov_message_index],
                pov_bank_message,
                *messages[pov_message_index:],
            ]
        response = llm_with_tools.invoke(messages)
        usage_collector.record(
            extract_response_model(response, fallback_model=config.model_config.model),
            extract_token_usage(response),
            source="instruction_kmeans_v1_agent",
        )
        updates: dict[str, object] = {
            "messages": [response],
            **_state_updates_from_runtime_state(runtime_state),
        }
        return updates

    tool_node = ToolNode(tools)

    def tools_node(state: AgentState) -> dict[str, object]:
        _hydrate_runtime_state(runtime_state, state=state)
        pre_tool_pov_bank = copy.deepcopy(runtime_state.clustering_pov_bank)
        tool_updates = cast(dict[str, object], tool_node.invoke(state))
        updates = {
            **tool_updates,
            **_state_updates_from_runtime_state(runtime_state),
        }
        post_tool_pov_bank = runtime_state.clustering_pov_bank
        if pre_tool_pov_bank != post_tool_pov_bank:
            updates["clustering_pov_history"] = upsert_clustering_pov_history(
                history=state.get("clustering_pov_history"),
                turn_index=_current_iteration_count(state) + 1,
                povs=post_tool_pov_bank,
            )
        return updates

    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(cast(type[Any], AgentState))
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile()


def _current_iteration_count(state: AgentState) -> int:
    raw_iteration_count = state.get("iteration_count", 0)
    if isinstance(raw_iteration_count, bool) or not isinstance(raw_iteration_count, int):
        return 0
    return max(0, raw_iteration_count)
