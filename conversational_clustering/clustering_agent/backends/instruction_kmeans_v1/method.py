"""Method orchestration and intermittent clustering capture for instruction_kmeans_v1."""

import json
import logging
import os
from pathlib import Path

from langchain_core.messages import HumanMessage

from ....accounting import UsageCollector
from ....data_processing.data_loader import DataLoader
from ....logging_utils import log_json
from ...state import AgentState
from ..backend_v1.tools import DEFAULT_CLUSTERING_SAVE_PATH
from ..base import ConversationalClusteringMethod
from . import intermittent_clustering_judge as judge
from .app import build_instruction_kmeans_v1_tool_calling_app
from .config import INSTRUCTION_KMEANS_V1_IDENTIFIER, InstructionKMeansV1ClusteringBackendConfig
from .prompts import get_instruction_kmeans_v1_system_prompt
from .tools import InstructionKMeansRuntimeState


class InstructionKMeansV1ConversationalClusteringMethod(ConversationalClusteringMethod):
    def __init__(self, config: InstructionKMeansV1ClusteringBackendConfig):
        self._config = config
        self._runtime_state = InstructionKMeansRuntimeState()
        self._usage_collector: UsageCollector | None = None

    @property
    def backend(self) -> str:
        return INSTRUCTION_KMEANS_V1_IDENTIFIER

    def resolve_app(
        self,
        *,
        data_loader: DataLoader | None,
        save_path: str | None = None,
        usage_collector: UsageCollector | None = None,
    ) -> object:
        self._runtime_state = InstructionKMeansRuntimeState()
        self._usage_collector = usage_collector
        return build_instruction_kmeans_v1_tool_calling_app(
            data_loader=data_loader,
            system_prompt=get_instruction_kmeans_v1_system_prompt(self._config.prompt_id),
            config=self._config,
            runtime_state=self._runtime_state,
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
            inference_result = judge.infer_instruction_kmeans_v1_intermittent_clustering_result(
                messages=messages,
                data_loader=data_loader,
                config=self._config,
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
            state["intermittent_clustering_token_usage"] = {}
            return state

        clustering = inference_result.clustering
        turn_index = _current_iteration_count(state) + 1
        _append_judge_clustering_trace_entry(
            state,
            turn_index=turn_index,
            clustering=clustering,
        )
        state["latest_proposed_clustering"] = clustering
        if inference_result.token_usage is not None:
            state["intermittent_clustering_token_usage"] = inference_result.token_usage
            if self._usage_collector is not None:
                self._usage_collector.record(
                    self._config.intermittent_clustering_judge_model.model,
                    inference_result.token_usage,
                    source="instruction_kmeans_v1_intermittent_judge",
                )
        else:
            state["intermittent_clustering_token_usage"] = {}
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
                    "source": "hidden_judge",
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
