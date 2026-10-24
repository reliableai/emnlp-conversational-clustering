"""Hidden intermittent clustering inference using the instruction_kmeans_v1 tool-calling loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Protocol, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from ....data_processing.data_loader import DataLoader
from ....logging_utils import extract_token_usage
from ...state import AgentState
from .app import build_instruction_kmeans_v1_tool_calling_app
from .config import InstructionKMeansV1ClusteringBackendConfig
from .prompts import get_instruction_kmeans_v1_intermittent_clustering_judge_prompt
from .tools import InstructionKMeansRuntimeState

_INTERMITTENT_JUDGE_MAX_RECOVERY_ATTEMPTS = 2
_INTERMITTENT_JUDGE_RETRY_MESSAGE = (
    "The last propose_clustering call failed. Continue the clustering process and call "
    "propose_clustering again with corrected arguments. Do not stop until you have "
    "produced a valid clustering."
)


class InvokableApp(Protocol):
    def invoke(self, state: AgentState) -> AgentState: ...


@dataclass(frozen=True)
class IntermittentClusteringInferenceResult:
    clustering: dict[str, object]
    token_usage: dict[str, int] | None


def _normalize_prompt_value(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        typed_value = cast(dict[object, object], value)
        return {
            str(key): _normalize_prompt_value(item)
            for key, item in sorted(typed_value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list | tuple):
        return [_normalize_prompt_value(item) for item in value]
    return str(value)


def _extract_textual_message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        blocks: list[str] = []
        for block in content:
            if isinstance(block, str):
                normalized_block = block.strip()
                if normalized_block:
                    blocks.append(normalized_block)
                continue
            if isinstance(block, dict):
                typed_block = cast(dict[str, object], block)
                raw_text = typed_block.get("text")
                if typed_block.get("type") == "text" and isinstance(raw_text, str):
                    normalized_text = raw_text.strip()
                    if normalized_text:
                        blocks.append(normalized_text)
                    continue
                try:
                    blocks.append(
                        json.dumps(
                            _normalize_prompt_value(typed_block),
                            ensure_ascii=True,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    )
                except TypeError:
                    blocks.append(str(block))
                continue
            blocks.append(str(block))
        return "\n".join(blocks)
    return str(content)


def _normalize_message_content(content: object, *, parse_json_strings: bool = False) -> object:
    if parse_json_strings and isinstance(content, str):
        normalized_content = content.strip()
        if normalized_content.startswith("{") or normalized_content.startswith("["):
            try:
                parsed_content = cast(object, json.loads(normalized_content))
            except json.JSONDecodeError:
                return content
            return _normalize_prompt_value(parsed_content)
    return _normalize_prompt_value(content)


def _normalize_tool_call(tool_call: object) -> dict[str, object] | None:
    if not isinstance(tool_call, dict):
        return None
    typed_tool_call = cast(dict[str, object], tool_call)
    normalized_tool_call = cast(dict[str, object], _normalize_prompt_value(typed_tool_call))
    raw_name = normalized_tool_call.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None
    return normalized_tool_call


def _serialize_prompt_message(message: BaseMessage) -> dict[str, object] | None:
    if isinstance(message, HumanMessage):
        return {
            "role": "human",
            "content": _extract_textual_message_content(message.content),
        }
    if isinstance(message, AIMessage):
        entry: dict[str, object] = {
            "role": "assistant",
            "content": _extract_textual_message_content(message.content),
        }
        normalized_tool_calls = [
            normalized_tool_call
            for raw_tool_call in message.tool_calls
            if (normalized_tool_call := _normalize_tool_call(raw_tool_call)) is not None
        ]
        if normalized_tool_calls:
            entry["tool_calls"] = normalized_tool_calls
        return entry
    if isinstance(message, ToolMessage):
        entry = {
            "role": "tool",
            "content": _normalize_message_content(message.content, parse_json_strings=True),
        }
        if isinstance(message.name, str) and message.name.strip():
            entry["name"] = message.name.strip()
        if isinstance(message.tool_call_id, str) and message.tool_call_id.strip():
            entry["tool_call_id"] = message.tool_call_id.strip()
        return entry
    return None


def _format_conversation(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        serialized_message = _serialize_prompt_message(message)
        if serialized_message is None:
            continue
        lines.append(
            json.dumps(
                serialized_message,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    if not lines:
        return "(no prior conversation)"
    return "\n".join(lines)


def _build_judge_state(*, dataset_size: int) -> AgentState:
    return {
        "messages": [],
        "dataset_metadata": {"size": dataset_size},
        "iteration_count": 0,
        "human_thought_trace": [],
        "judge_clustering_trace": [],
    }


def _extract_judge_error(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, ToolMessage) and message.name in {
            "propose_clustering",
            "save_clustering",
        }:
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
                    "Intermittent clustering judge did not produce a clustering. "
                    f"Final assistant message: {normalized_text}"
                )
    return "Intermittent clustering judge did not produce a clustering."


def _latest_propose_clustering_failed(state: AgentState) -> bool:
    for message in reversed(state.get("messages", [])):
        if not isinstance(message, ToolMessage) or message.name != "propose_clustering":
            continue
        content = message.content
        text = content if isinstance(content, str) else str(content)
        return text.strip().lower().startswith("could not propose clustering")
    return False


def _build_retry_state(state: AgentState) -> AgentState:
    retry_state = cast(AgentState, dict(state))
    retry_state["messages"] = [
        *state.get("messages", []),
        HumanMessage(content=_INTERMITTENT_JUDGE_RETRY_MESSAGE),
    ]
    return retry_state


def _load_latest_proposal_clustering(
    runtime_state: InstructionKMeansRuntimeState,
) -> dict[str, object]:
    proposal = runtime_state.latest_proposal_artifacts
    if proposal is None:
        raise ValueError("Intermittent clustering judge did not produce a clustering.")
    raw_payload = proposal.get("clustering")
    if not isinstance(raw_payload, dict):
        raise ValueError("Intermittent clustering judge produced a non-dict clustering payload.")
    payload = cast(dict[str, object], raw_payload)
    raw_clusters = payload.get("clusters")
    if not isinstance(raw_clusters, list) or not raw_clusters:
        raise ValueError("Intermittent clustering judge produced an invalid clustering payload.")
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


def infer_instruction_kmeans_v1_intermittent_clustering_result(
    *,
    messages: list[BaseMessage],
    data_loader: DataLoader,
    config: InstructionKMeansV1ClusteringBackendConfig,
) -> IntermittentClusteringInferenceResult:
    system_prompt = get_instruction_kmeans_v1_intermittent_clustering_judge_prompt(
        config.prompt_id
    ).format(conversation=_format_conversation(messages))
    dataset_size = len(data_loader.load())
    runtime_state = InstructionKMeansRuntimeState()
    app = cast(
        InvokableApp,
        build_instruction_kmeans_v1_tool_calling_app(
            data_loader=data_loader,
            system_prompt=system_prompt,
            config=replace(
                config,
                model_config=config.intermittent_clustering_judge_model,
            ),
            runtime_state=runtime_state,
            save_path=None,
            include_save_tool=False,
            include_pov_bank_tool=False,
            enable_cluster_summary_model=False,
        ),
    )
    judge_state = _build_judge_state(dataset_size=dataset_size)
    final_state = judge_state
    for recovery_attempt in range(_INTERMITTENT_JUDGE_MAX_RECOVERY_ATTEMPTS + 1):
        final_state = app.invoke(judge_state)
        if not _latest_propose_clustering_failed(final_state):
            break
        if recovery_attempt == _INTERMITTENT_JUDGE_MAX_RECOVERY_ATTEMPTS:
            break
        judge_state = _build_retry_state(final_state)
    token_usage = _extract_state_token_usage(final_state)
    if _latest_propose_clustering_failed(final_state):
        raise ValueError(_extract_judge_error(final_state))
    try:
        return IntermittentClusteringInferenceResult(
            clustering=_load_latest_proposal_clustering(runtime_state),
            token_usage=token_usage,
        )
    except ValueError as exc:
        if str(exc) == "Intermittent clustering judge did not produce a clustering.":
            raise ValueError(_extract_judge_error(final_state)) from exc
        raise
