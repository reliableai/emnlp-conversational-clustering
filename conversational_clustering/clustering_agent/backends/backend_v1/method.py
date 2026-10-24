"""LangGraph app and intermittent clustering capture for the backend_v1 backend."""

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from ....accounting import UsageCollector, extract_response_model, extract_token_usage
from ....data_processing.data_loader import DataLoader
from ....logging_utils import log_json
from ....models import (
    ModelConfig,
    build_clustering_agent_llm,
    default_clustering_agent_model_config,
)
from ...state import AgentState
from ..base import ConversationalClusteringMethod
from ..pov_bank import (
    build_clustering_pov_bank_message,
    coerce_clustering_pov_bank,
    upsert_clustering_pov_history,
)
from .config import BACKEND_V1_IDENTIFIER, BackendV1ClusteringBackendConfig
from .prompts import get_backend_v1_system_prompt
from .tools import DEFAULT_CLUSTERING_SAVE_PATH, BackendV1RuntimeState, build_tools


def _serialize_runtime_state(
    runtime_state: BackendV1RuntimeState,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    if runtime_state.clustering_pov_bank is not None:
        payload["clustering_pov_bank"] = copy.deepcopy(runtime_state.clustering_pov_bank)
    return payload


def _hydrate_runtime_state(
    runtime_state: BackendV1RuntimeState,
    *,
    state: AgentState,
) -> None:
    serialized = state.get("backend_v1_runtime_state")
    runtime_state.clustering_pov_bank = None
    if not isinstance(serialized, dict):
        return
    clustering_pov_bank = serialized.get("clustering_pov_bank")
    if isinstance(clustering_pov_bank, list):
        normalized_pov_bank = coerce_clustering_pov_bank(
            [pov for pov in clustering_pov_bank if isinstance(pov, str)]
        )
        runtime_state.clustering_pov_bank = copy.deepcopy(normalized_pov_bank)


def _state_updates_from_runtime_state(
    runtime_state: BackendV1RuntimeState,
) -> dict[str, object]:
    return {
        "backend_v1_runtime_state": _serialize_runtime_state(runtime_state),
    }


def build_backend_v1_tool_calling_app(
    *,
    data_loader: DataLoader | None,
    system_prompt: str,
    model_config: ModelConfig | None = None,
    save_path: str | None = None,
    usage_collector: UsageCollector | None = None,
) -> object:
    usage_collector = usage_collector or UsageCollector()
    resolved_model_config = model_config or default_clustering_agent_model_config()
    llm = build_clustering_agent_llm(model_config)
    runtime_state = BackendV1RuntimeState()
    tools = build_tools(
        data_loader,
        save_path=save_path,
        runtime_state=runtime_state,
    )
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict[str, object]:
        _hydrate_runtime_state(runtime_state, state=state)
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt), *messages]
        dataset_metadata = state.get("dataset_metadata") or {}
        dataset_size = dataset_metadata.get("size")
        if dataset_size is not None:
            metadata_message = SystemMessage(
                content=f"Dataset metadata: size={dataset_size} items."
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
            extract_response_model(response, fallback_model=resolved_model_config.model),
            extract_token_usage(response),
            source="backend_v1_agent",
        )
        return {
            "messages": [response],
            **_state_updates_from_runtime_state(runtime_state),
        }

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


class BackendV1ConversationalClusteringMethod(ConversationalClusteringMethod):
    def __init__(self, config: BackendV1ClusteringBackendConfig):
        self._config = config
        self._usage_collector: UsageCollector | None = None

    @property
    def backend(self) -> str:
        return BACKEND_V1_IDENTIFIER

    def resolve_app(
        self,
        *,
        data_loader: DataLoader | None,
        save_path: str | None = None,
        usage_collector: UsageCollector | None = None,
    ) -> object:
        self._usage_collector = usage_collector
        return build_backend_v1_tool_calling_app(
            data_loader=data_loader,
            system_prompt=get_backend_v1_system_prompt(self._config.prompt_id),
            model_config=self._config.model_config,
            save_path=save_path,
            usage_collector=usage_collector,
        )

    def build_intermittent_clustering_save_path(
        self,
        *,
        clustering_save_path: str | None,
    ) -> str:
        target_path = Path(clustering_save_path or DEFAULT_CLUSTERING_SAVE_PATH)
        return str(target_path.with_name(f"{target_path.stem}.judge{target_path.suffix}"))

    def capture_intermittent_clustering(
        self,
        *,
        state: AgentState,
        data_loader: DataLoader | None,
        logger: logging.Logger,
        output_path: str | None,
    ) -> AgentState:
        if data_loader is None:
            return state
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], HumanMessage):
            return state
        try:
            from .intermittent_clustering_judge import (
                infer_backend_v1_intermittent_clustering_result,
            )

            inference_result = infer_backend_v1_intermittent_clustering_result(
                messages=messages,
                data_loader=data_loader,
                model_config=self._config.intermittent_clustering_judge_model,
            )
        except Exception as exc:
            log_json(
                logger,
                "intermittent_clustering_judge_error",
                {
                    "iteration": state.get("iteration_count", 0),
                    "error": str(exc),
                },
            )
            return state

        clustering = inference_result.clustering
        turn_index = _current_iteration_count(state) + 1
        _append_judge_clustering_trace_entry(
            state,
            turn_index=turn_index,
            clustering=clustering,
        )
        if inference_result.token_usage is not None:
            state["intermittent_clustering_token_usage"] = inference_result.token_usage
            if self._usage_collector is not None:
                self._usage_collector.record(
                    self._config.intermittent_clustering_judge_model.model,
                    inference_result.token_usage,
                    source="backend_v1_intermittent_judge",
                )
        try:
            resolved_output_path = _write_numbered_judge_clustering_snapshot(
                clustering=clustering,
                output_path=output_path,
                turn_index=turn_index,
            )
        except Exception as exc:
            log_json(
                logger,
                "intermittent_clustering_judge_write_error",
                {
                    "iteration": state.get("iteration_count", 0),
                    "output_path": output_path,
                    "error": str(exc),
                },
            )
        else:
            log_json(
                logger,
                "intermittent_clustering_judge_snapshot",
                {
                    "iteration": state.get("iteration_count", 0),
                    "turn_index": turn_index,
                    "output_path": resolved_output_path,
                },
            )
        return state


def _append_judge_clustering_trace_entry(
    state: AgentState,
    *,
    turn_index: int,
    clustering: dict[str, object],
) -> None:
    trace = state.setdefault("judge_clustering_trace", [])
    trace.append(
        {
            "turn_index": turn_index,
            "clustering": clustering,
        }
    )


def _build_numbered_judge_snapshot_path(output_path: str, *, turn_index: int) -> str:
    target_path = Path(output_path)
    return str(target_path.with_name(f"{target_path.stem}.{turn_index:04d}{target_path.suffix}"))


def _write_numbered_judge_clustering_snapshot(
    *,
    clustering: dict[str, object],
    output_path: str | None,
    turn_index: int,
) -> str | None:
    if output_path is None:
        return None
    resolved_output_path = _build_numbered_judge_snapshot_path(output_path, turn_index=turn_index)
    output_dir = os.path.dirname(resolved_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(resolved_output_path, "w", encoding="utf-8") as handle:
        json.dump(clustering, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    return resolved_output_path


def _current_iteration_count(state: AgentState) -> int:
    raw_iteration_count = state.get("iteration_count", 0)
    if isinstance(raw_iteration_count, bool) or not isinstance(raw_iteration_count, int):
        return 0
    return max(0, raw_iteration_count)
