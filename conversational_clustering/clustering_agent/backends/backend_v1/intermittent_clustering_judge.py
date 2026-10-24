"""Hidden intermittent clustering inference using the backend_v1 tool-calling loop."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Protocol, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from ....data_processing.data_loader import DataLoader
from ....logging_utils import extract_token_usage
from ....models import ModelConfig
from ...state import AgentState
from .method import build_backend_v1_tool_calling_app
from .prompts import get_backend_v1_intermittent_clustering_judge_prompt


class InvokableApp(Protocol):
    def invoke(self, state: AgentState) -> AgentState: ...


@dataclass(frozen=True)
class IntermittentClusteringInferenceResult:
    clustering: dict[str, object]
    token_usage: dict[str, int] | None


def _format_conversation(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        content = message.content
        text = content if isinstance(content, str) else str(content)
        normalized_text = text.strip()
        if not normalized_text:
            continue
        lines.append(f"{role}: {normalized_text}")
    if not lines:
        return "(no prior conversation)"
    return "\n".join(lines)


def _build_judge_state(
    *,
    dataset_size: int,
) -> AgentState:
    return {
        "messages": [],
        "dataset_metadata": {"size": dataset_size},
        "iteration_count": 0,
        "human_thought_trace": [],
        "judge_clustering_trace": [],
    }


def _extract_judge_error(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, ToolMessage) and message.name == "save_clustering":
            content = message.content
            text = content if isinstance(content, str) else str(content)
            normalized_text = text.strip()
            if normalized_text:
                return normalized_text
        if isinstance(message, AIMessage):
            content = message.content
            text = content if isinstance(content, str) else str(content)
            normalized_text = text.strip()
            if normalized_text:
                return (
                    "Intermittent clustering judge did not save a clustering. "
                    f"Final assistant message: {normalized_text}"
                )
    return "Intermittent clustering judge did not save a clustering."


def _load_saved_clustering_payload(output_path: str) -> dict[str, object]:
    if not os.path.exists(output_path):
        raise ValueError("Intermittent clustering judge did not save a clustering.")
    with open(output_path, encoding="utf-8") as handle:
        raw_payload = cast(object, json.load(handle))
    if not isinstance(raw_payload, dict):
        raise ValueError("Intermittent clustering judge saved a non-dict clustering payload.")
    payload = cast(dict[str, object], raw_payload)
    raw_clusters = payload.get("clusters")
    if not isinstance(raw_clusters, list) or not raw_clusters:
        raise ValueError("Intermittent clustering judge saved an invalid clustering payload.")
    return payload


def _accumulate_token_usage(
    current_usage: dict[str, int] | None,
    new_usage: dict[str, int] | None,
) -> dict[str, int] | None:
    if new_usage is None:
        return current_usage
    if current_usage is None:
        return {
            "input_tokens": new_usage.get("input_tokens", 0),
            "output_tokens": new_usage.get("output_tokens", 0),
            "total_tokens": new_usage.get("total_tokens", 0),
        }
    return {
        "input_tokens": current_usage.get("input_tokens", 0) + new_usage.get("input_tokens", 0),
        "output_tokens": current_usage.get("output_tokens", 0) + new_usage.get("output_tokens", 0),
        "total_tokens": current_usage.get("total_tokens", 0) + new_usage.get("total_tokens", 0),
    }


def _extract_state_token_usage(state: AgentState) -> dict[str, int] | None:
    accumulated_usage: dict[str, int] | None = None
    for message in state.get("messages", []):
        if isinstance(message, AIMessage):
            accumulated_usage = _accumulate_token_usage(
                accumulated_usage,
                extract_token_usage(message),
            )
    return accumulated_usage


def infer_backend_v1_intermittent_clustering_result(
    *,
    messages: list[BaseMessage],
    data_loader: DataLoader,
    model_config: ModelConfig,
    prompt_id: str = "baseline_v2",
) -> IntermittentClusteringInferenceResult:
    system_prompt = get_backend_v1_intermittent_clustering_judge_prompt(prompt_id).format(
        conversation=_format_conversation(messages),
    )
    dataset_size = len(data_loader.load())

    with tempfile.TemporaryDirectory(prefix="intermittent-clustering-judge-") as temp_dir:
        output_path = os.path.join(temp_dir, "judge-clustering.json")
        app = cast(
            InvokableApp,
            build_backend_v1_tool_calling_app(
                data_loader=data_loader,
                system_prompt=system_prompt,
                model_config=model_config,
                save_path=output_path,
            ),
        )
        judge_state = _build_judge_state(dataset_size=dataset_size)
        final_state = app.invoke(judge_state)
        token_usage = _extract_state_token_usage(final_state)
        try:
            return IntermittentClusteringInferenceResult(
                clustering=_load_saved_clustering_payload(output_path),
                token_usage=token_usage,
            )
        except ValueError as exc:
            if str(exc) == "Intermittent clustering judge did not save a clustering.":
                raise ValueError(_extract_judge_error(final_state)) from exc
            raise


def infer_backend_v1_intermittent_clustering(
    *,
    messages: list[BaseMessage],
    data_loader: DataLoader,
    model_config: ModelConfig,
    prompt_id: str = "baseline_v2",
) -> dict[str, object]:
    return infer_backend_v1_intermittent_clustering_result(
        messages=messages,
        data_loader=data_loader,
        model_config=model_config,
        prompt_id=prompt_id,
    ).clustering
