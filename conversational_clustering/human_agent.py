"""LangGraph app that simulates a human collaborator with verifier-based retries."""

import logging
from typing import Any, Literal, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from pydantic import BaseModel, Field, model_validator

from .accounting import UsageCollector, extract_response_model
from .clustering_agent.state import AgentState
from .human_turn_injection import HumanTurnInjectionConfig
from .logging_utils import extract_token_usage, log_json
from .models import (
    ModelConfig,
    build_human_llm,
    build_human_verifier_llm,
    default_human_model_config,
    default_human_verifier_model_config,
)
from .prompts import (
    get_human_agent_system_prompt,
    get_human_verifier_system_prompt,
    normalize_guidance,
)
from .topic_discovery import TopicCandidate, TopicDiscoveryConfig

DEFAULT_PROMPT_ID = "baseline_v3"
DEFAULT_VERIFIER_PROMPT_ID = "baseline_v1"
_TOPIC_TOOL_NAME = "enumerate_topics"
_HUMAN_STRUCTURED_OUTPUT_RETRY_EXHAUSTED_MESSAGE = (
    "Your previous draft did not follow the required structured output format. "
    "Return a valid HumanTurnDraft response and do not leave the message empty."
)
_HUMAN_STRUCTURED_OUTPUT_NODE_RETRY_ATTEMPTS = 3


class VerifierDecision(BaseModel):
    decision: Literal["allow", "retry"]
    feedback: str


class TopicCandidateDraft(BaseModel):
    label: str
    guidance: str
    rationale: str

    @model_validator(mode="after")
    def _normalize_fields(self) -> "TopicCandidateDraft":
        self.label = self.label.strip()
        self.guidance = self.guidance.strip()
        self.rationale = self.rationale.strip()
        if not self.label:
            raise ValueError("label must be non-empty")
        if not self.guidance:
            raise ValueError("guidance must be non-empty")
        if not self.rationale:
            raise ValueError("rationale must be non-empty")
        return self


class HumanTurnDraft(BaseModel):
    action: Literal["message", "enumerate_topics"]
    message: str
    thought: str
    topics: list[TopicCandidateDraft] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_action_payload(self) -> "HumanTurnDraft":
        self.message = self.message.strip()
        self.thought = self.thought.strip()
        if not self.thought:
            raise ValueError("thought must be non-empty")
        if self.action == "message":
            self.topics = []
            return self
        if not self.topics:
            raise ValueError(f"topics must be non-empty when action is {_TOPIC_TOOL_NAME}")
        return self


class HumanAgentState(AgentState, total=False):
    candidate_human_action: str
    candidate_human_message: str
    candidate_human_thought: str
    candidate_topic_candidates: list[dict[str, str]]
    verifier_feedback: str
    verifier_decision: str
    verifier_retry_count: int
    human_turn_token_usage: dict[str, int]
    turn_injection_consumed_turn_index: int
    topic_candidates_accumulated: list[dict[str, str]]
    topic_discovery_round_count: int
    topic_discovery_new_topics_by_round: list[int]


def _sanitize_ai_message(message: AIMessage) -> AIMessage | None:
    disallowed_additional_kwargs = {
        "tool_calls",
        "function_call",
        "reasoning_content",
        "reasoning_details",
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        content = message.content
        if not content:
            return None
        return AIMessage(content=content)
    additional_kwargs = getattr(message, "additional_kwargs", None) or {}
    if any(key in additional_kwargs for key in disallowed_additional_kwargs):
        cleaned_kwargs = {
            key: value
            for key, value in additional_kwargs.items()
            if key not in disallowed_additional_kwargs
        }
        return AIMessage(content=message.content, additional_kwargs=cleaned_kwargs)
    return message


def _filter_user_visible_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    visible_messages: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            visible_messages.append(message)
            continue
        if isinstance(message, AIMessage):
            sanitized = _sanitize_ai_message(message)
            if sanitized is not None:
                visible_messages.append(sanitized)
    return visible_messages


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    return str(content)


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
    input_tokens = current_usage.get("input_tokens", 0) + new_usage.get("input_tokens", 0)
    output_tokens = current_usage.get("output_tokens", 0) + new_usage.get("output_tokens", 0)
    total_tokens = current_usage.get("total_tokens", 0) + new_usage.get("total_tokens", 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _format_conversation_for_verifier(messages: list[BaseMessage]) -> str:
    if not messages:
        return "(no prior messages)"
    lines: list[str] = []
    for message in messages[-8:]:
        role = "human" if isinstance(message, HumanMessage) else "assistant"
        lines.append(f"{role}: {_content_to_text(message.content).strip()}")
    return "\n".join(lines)


def _build_human_message(
    content: str,
    token_usage: dict[str, int] | None,
    thought: str | None = None,
) -> HumanMessage:
    additional_kwargs: dict[str, object] = {}
    if token_usage:
        additional_kwargs["token_usage"] = token_usage
    if thought:
        additional_kwargs["human_thought"] = thought
    if additional_kwargs:
        return HumanMessage(content=content, additional_kwargs=additional_kwargs)
    return HumanMessage(content=content)


def _log_event(
    logger: logging.Logger | None,
    event: str,
    payload: dict[str, object],
) -> None:
    if logger is None:
        return
    log_json(logger, event, payload)


def _describe_structured_raw_response(raw_response: object) -> dict[str, object]:
    content = getattr(raw_response, "content", "")
    text_content = _content_to_text(content).strip()
    tool_calls = getattr(raw_response, "tool_calls", None)
    return {
        "content": text_content,
        "has_content": bool(text_content),
        "tool_call_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
        "response_model": extract_response_model(raw_response),
        "token_usage": extract_token_usage(raw_response),
    }


def _build_retry_response(
    *,
    feedback: str,
    retry_count: int,
    max_verification_retries: int,
    accumulated_usage: dict[str, int] | None,
    logger: logging.Logger | None,
    topic_discovery_force_generation: bool | None = None,
) -> dict[str, object]:
    if retry_count >= max_verification_retries:
        raise ValueError(feedback)
    _log_event(
        logger,
        "human_agent_retry_requested",
        {
            "retry_count": retry_count + 1,
            "feedback": feedback,
        },
    )
    response: dict[str, object] = {
        "candidate_human_action": "",
        "candidate_human_message": "",
        "candidate_human_thought": "",
        "candidate_topic_candidates": [],
        "verifier_feedback": feedback,
        "verifier_decision": "retry",
        "verifier_retry_count": retry_count + 1,
        "human_turn_token_usage": accumulated_usage or {},
    }
    if topic_discovery_force_generation is not None:
        response["topic_discovery_force_generation"] = topic_discovery_force_generation
    return response


def _is_retryable_human_structured_output_error(exc: Exception) -> bool:
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        if (
            isinstance(current, ValueError)
            and str(current) == _HUMAN_STRUCTURED_OUTPUT_RETRY_EXHAUSTED_MESSAGE
        ):
            return True
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return False


def _current_iteration_count(state: HumanAgentState) -> int:
    raw_iteration_count = state.get("iteration_count", 0)
    if isinstance(raw_iteration_count, bool) or not isinstance(raw_iteration_count, int):
        return 0
    return max(0, raw_iteration_count)


def _current_turn_injection_consumed_turn_index(state: HumanAgentState) -> int | None:
    raw_turn_index = state.get("turn_injection_consumed_turn_index")
    if isinstance(raw_turn_index, bool) or not isinstance(raw_turn_index, int):
        return None
    return max(0, raw_turn_index)


def _build_generation_instruction(
    topic_discovery_config: TopicDiscoveryConfig | None,
    *,
    force_topic_generation: bool,
    topics_already_generated: bool,
    accumulated_topic_candidates: list[dict[str, str]],
) -> str:
    if (
        topic_discovery_config is None
        or not topic_discovery_config.enabled
        or topics_already_generated
    ):
        return (
            "Write your next single user message to the clustering agent. "
            "Also provide one short hidden note describing why you chose that message."
        )
    if force_topic_generation:
        if accumulated_topic_candidates:
            existing_topic_labels = ", ".join(
                sorted({candidate["label"] for candidate in accumulated_topic_candidates})
            )
            return (
                f"Call the {_TOPIC_TOOL_NAME} tool now instead of writing a user message. "
                "Generate only novel topic candidates that are meaningfully distinct from prior "
                "rounds. Do not repeat any existing labels or topic IDs. "
                f"Existing topic labels: {existing_topic_labels}. "
                "Each topic must include a short label, concrete downstream guidance, "
                "and a brief rationale grounded in the conversation. "
                "Also provide one short hidden note describing why you chose that action."
            )
        return (
            f"Call the {_TOPIC_TOOL_NAME} tool now instead of writing a user message. "
            "Generate however many distinct topic candidates are justified by the conversation. "
            "Each topic must include a short label, concrete downstream guidance, "
            "and a brief rationale grounded in the conversation. "
            "Also provide one short hidden note describing why you chose that action."
        )
    if topic_discovery_config.trigger_mode == "human_tool":
        return (
            "Choose one action for this turn: either write your next single user message to the "
            f"clustering agent, or call the {_TOPIC_TOOL_NAME} tool if you already have enough "
            "context to propose several viable clustering topics. "
            "If you call the tool, generate however many distinct topic candidates are justified "
            "by the conversation. "
            "Also provide one short hidden note describing why you chose that action."
        )
    return (
        "Write your next single user message to the clustering agent. "
        "Also provide one short hidden note describing why you chose that message."
    )


def _append_generation_instruction(
    prompt_messages: list[BaseMessage],
    topic_discovery_config: TopicDiscoveryConfig | None,
    *,
    force_topic_generation: bool,
    topics_already_generated: bool,
    accumulated_topic_candidates: list[dict[str, str]],
) -> None:
    prompt_messages.append(
        HumanMessage(
            content=_build_generation_instruction(
                topic_discovery_config,
                force_topic_generation=force_topic_generation,
                topics_already_generated=topics_already_generated,
                accumulated_topic_candidates=accumulated_topic_candidates,
            )
        )
    )


def _coerce_topic_candidates_payload(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    payload: list[dict[str, str]] = []
    for raw_topic in value:
        if not isinstance(raw_topic, dict):
            continue
        raw_topic_dict = cast(dict[str, object], raw_topic)
        raw_topic_id = raw_topic_dict.get("topic_id")
        raw_label = raw_topic_dict.get("label")
        raw_guidance = raw_topic_dict.get("guidance")
        raw_rationale = raw_topic_dict.get("rationale")
        if not isinstance(raw_topic_id, str) or not raw_topic_id.strip():
            continue
        if not isinstance(raw_label, str) or not raw_label.strip():
            continue
        if not isinstance(raw_guidance, str) or not raw_guidance.strip():
            continue
        if not isinstance(raw_rationale, str) or not raw_rationale.strip():
            continue
        topic = {
            "topic_id": raw_topic_id.strip(),
            "label": raw_label.strip(),
            "guidance": raw_guidance.strip(),
            "rationale": raw_rationale.strip(),
        }
        payload.append(topic)
    return payload


def _coerce_topic_discovery_round_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _coerce_topics_per_round(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    rounds: list[int] = []
    for raw_round_count in value:
        if isinstance(raw_round_count, bool) or not isinstance(raw_round_count, int):
            continue
        rounds.append(max(0, raw_round_count))
    return rounds


def _append_selected_topic_instruction(
    prompt_messages: list[BaseMessage],
    *,
    selected_topic_label: str | None,
    selected_topic_guidance: str | None,
) -> None:
    normalized_label = selected_topic_label.strip() if selected_topic_label else ""
    normalized_guidance = normalize_guidance(selected_topic_guidance)
    if not normalized_label and normalized_guidance is None:
        return
    if normalized_label and normalized_guidance is not None:
        content = (
            "A clustering topic has already been selected for this continuation. "
            f"Keep steering toward the topic '{normalized_label}'. "
            f"Use this as your working clustering direction: {normalized_guidance}"
        )
    elif normalized_guidance is not None:
        content = (
            "A clustering topic has already been selected for this continuation. "
            f"Use this as your working clustering direction: {normalized_guidance}"
        )
    else:
        content = (
            "A clustering topic has already been selected for this continuation. "
            f"Keep steering toward the topic '{normalized_label}'."
        )
    prompt_messages.append(SystemMessage(content=content))


def _should_append_selected_topic_instruction(
    *,
    turn_injection: HumanTurnInjectionConfig | None,
    current_turn_index: int,
    consumed_turn_index: int | None,
    selected_topic_generated_at_turn: int | None,
) -> bool:
    if turn_injection is None:
        return True
    if turn_injection.edit_primitive != "theme_switch":
        return True
    if current_turn_index < turn_injection.turn_index:
        return True
    if (
        selected_topic_generated_at_turn is None
        or selected_topic_generated_at_turn < turn_injection.turn_index
    ):
        return False
    return consumed_turn_index is not None and consumed_turn_index >= turn_injection.turn_index


def _append_turn_injection_instruction(
    prompt_messages: list[BaseMessage],
    *,
    state: HumanAgentState,
    turn_injection: HumanTurnInjectionConfig | None,
) -> int | None:
    if turn_injection is None:
        return None
    current_turn_index = _current_iteration_count(state) + 1
    consumed_turn_index = _current_turn_injection_consumed_turn_index(state)
    if current_turn_index < turn_injection.turn_index:
        return None
    content = _build_turn_injection_instruction_content(
        turn_injection=turn_injection,
        current_turn_index=current_turn_index,
        already_consumed=consumed_turn_index is not None
        and consumed_turn_index >= turn_injection.turn_index,
    )
    prompt_messages.append(SystemMessage(content=content))
    return current_turn_index


def _build_turn_injection_instruction_content(
    *,
    turn_injection: HumanTurnInjectionConfig,
    current_turn_index: int,
    already_consumed: bool = False,
) -> str:
    resolved_prompt_text = turn_injection.resolved_prompt_text()
    if current_turn_index == turn_injection.turn_index and not already_consumed:
        return (
            "A new instruction takes effect starting this turn.\n\n"
            "This instruction overrides any earlier clustering guidance or direction for this "
            "turn and all later turns unless a newer system instruction replaces it. Pivot "
            "immediately, even if it conflicts with your previous goal.\n\n"
            "Injected instruction:\n"
            f"{resolved_prompt_text}"
        )

    return (
        "An injected instruction changed your clustering direction on an earlier turn.\n\n"
        "Keep following that injected direction on this turn and later turns unless a newer "
        "system instruction replaces it. Do not revert to the original guidance.\n\n"
        "Original injected instruction:\n"
        f"{resolved_prompt_text}"
    )


def _make_enumerate_topics_tool(topic_discovery_config: TopicDiscoveryConfig) -> BaseTool:
    # This is a topic-discovery-only control-flow schema, not a general tool-calling pattern.
    # Do not copy this for real/provider-native/external tools; reevaluate registration separately.
    @tool
    def enumerate_topics(topics: list[TopicCandidate]) -> str:
        """Stores a structured list of viable clustering topics discovered during exploration."""
        _ = topic_discovery_config
        if not topics:
            raise ValueError("At least one topic is required.")
        topic_ids = [topic.topic_id for topic in topics]
        if len(topic_ids) != len(set(topic_ids)):
            raise ValueError("Topic labels must resolve to unique topic IDs.")
        return f"Stored {len(topics)} topic candidates."

    return enumerate_topics


def build_human_app(
    guidance: str | None,
    human_agent_model_config: ModelConfig | None = None,
    system_prompt: str | None = None,
    verifier_model_config: ModelConfig | None = None,
    verifier_system_prompt: str | None = None,
    max_verification_retries: int = 2,
    logger: logging.Logger | None = None,
    usage_collector: UsageCollector | None = None,
    topic_discovery_config: TopicDiscoveryConfig | None = None,
    turn_injection: HumanTurnInjectionConfig | None = None,
    selected_topic_label: str | None = None,
    selected_topic_guidance: str | None = None,
    selected_topic_generated_at_turn: int | None = None,
):
    usage_collector = usage_collector or UsageCollector()
    resolved_human_model_config = human_agent_model_config or default_human_model_config()
    resolved_verifier_model_config = verifier_model_config or default_human_verifier_model_config()
    llm = build_human_llm(human_agent_model_config)
    human_structured_llm = llm.with_structured_output(HumanTurnDraft, strict=True, include_raw=True)
    verifier_llm = build_human_verifier_llm(verifier_model_config)
    verifier_structured_llm = verifier_llm.with_structured_output(
        VerifierDecision, strict=True, include_raw=True
    )
    if system_prompt is None:
        system_prompt = get_human_agent_system_prompt(DEFAULT_PROMPT_ID)
    if verifier_system_prompt is None:
        verifier_system_prompt = get_human_verifier_system_prompt(DEFAULT_VERIFIER_PROMPT_ID)
    max_verification_retries = max(0, max_verification_retries)
    normalized_guidance = normalize_guidance(guidance)
    resolved_system_prompt = system_prompt
    if normalized_guidance is not None:
        resolved_system_prompt = system_prompt.replace("{guidance}", normalized_guidance)
    enumerate_topics_tool = (
        _make_enumerate_topics_tool(topic_discovery_config)
        if topic_discovery_config is not None and topic_discovery_config.enabled
        else None
    )

    def human_node(state: HumanAgentState) -> dict[str, object]:
        messages = _filter_user_visible_messages(state.get("messages", []))
        verifier_feedback = state.get("verifier_feedback", "").strip()
        force_topic_generation = bool(state.get("topic_discovery_force_generation", False))
        previous_turn_usage = (
            state.get("human_turn_token_usage")
            if verifier_feedback or force_topic_generation
            else None
        )
        retry_count = state.get("verifier_retry_count", 0) if verifier_feedback else 0
        topics_already_generated = bool(state.get("topic_candidates"))
        accumulated_topic_candidates = _coerce_topic_candidates_payload(
            state.get("topic_candidates_accumulated", [])
        )
        current_turn_index = _current_iteration_count(state) + 1
        consumed_turn_index = _current_turn_injection_consumed_turn_index(state)
        prompt_messages = [SystemMessage(content=resolved_system_prompt)]
        prompt_messages.extend(messages)
        injected_turn_index = _append_turn_injection_instruction(
            prompt_messages,
            state=state,
            turn_injection=turn_injection,
        )
        if injected_turn_index is not None and turn_injection is not None:
            _log_event(
                logger,
                "human_turn_injection_applied",
                {
                    "turn_index": injected_turn_index,
                    "configured_turn_index": turn_injection.turn_index,
                    "edit_primitive": turn_injection.edit_primitive,
                    "resolved_prompt_text": turn_injection.resolved_prompt_text(),
                    "retry_count": retry_count,
                },
            )
        if verifier_feedback:
            prompt_messages.append(
                SystemMessage(
                    content=(
                        "Your previous draft did not satisfy the human-agent guidelines. "
                        "Rewrite a new message using this verifier feedback:\n"
                        f"{verifier_feedback}"
                    )
                )
            )
        if _should_append_selected_topic_instruction(
            turn_injection=turn_injection,
            current_turn_index=current_turn_index,
            consumed_turn_index=consumed_turn_index,
            selected_topic_generated_at_turn=selected_topic_generated_at_turn,
        ):
            _append_selected_topic_instruction(
                prompt_messages,
                selected_topic_label=selected_topic_label,
                selected_topic_guidance=selected_topic_guidance,
            )
        _append_generation_instruction(
            prompt_messages,
            topic_discovery_config,
            force_topic_generation=force_topic_generation,
            topics_already_generated=topics_already_generated,
            accumulated_topic_candidates=accumulated_topic_candidates,
        )
        response = human_structured_llm.invoke(prompt_messages)
        if not isinstance(response, dict):
            raise ValueError("Human structured output did not return a dict payload.")
        parsing_error = response.get("parsing_error")
        if parsing_error is not None:
            raise ValueError("Human structured output parsing failed.") from parsing_error
        raw_response = response.get("raw")
        accumulated_usage = previous_turn_usage
        if raw_response is not None:
            raw_token_usage = extract_token_usage(raw_response)
            usage_collector.record(
                extract_response_model(
                    raw_response, fallback_model=resolved_human_model_config.model
                ),
                raw_token_usage,
                source="human_generation",
            )
            accumulated_usage = _accumulate_token_usage(previous_turn_usage, raw_token_usage)
        parsed_response = response.get("parsed")
        if not isinstance(parsed_response, HumanTurnDraft):
            retry_count = state.get("verifier_retry_count", 0)
            diagnostic_payload = (
                _describe_structured_raw_response(raw_response) if raw_response is not None else {}
            )
            _log_event(
                logger,
                "human_agent_malformed_structured_output",
                {
                    "retry_count": retry_count,
                    **diagnostic_payload,
                },
            )
            return _build_retry_response(
                feedback=_HUMAN_STRUCTURED_OUTPUT_RETRY_EXHAUSTED_MESSAGE,
                retry_count=retry_count,
                max_verification_retries=max_verification_retries,
                accumulated_usage=accumulated_usage,
                logger=logger,
                topic_discovery_force_generation=force_topic_generation,
            )
        if force_topic_generation and parsed_response.action != _TOPIC_TOOL_NAME:
            retry_count = state.get("verifier_retry_count", 0)
            feedback = (
                f"Topic discovery is required now. Call {_TOPIC_TOOL_NAME} instead of writing "
                "a normal user message."
            )
            return _build_retry_response(
                feedback=feedback,
                retry_count=retry_count,
                max_verification_retries=max_verification_retries,
                accumulated_usage=accumulated_usage,
                logger=logger,
                topic_discovery_force_generation=True,
            )
        if parsed_response.action == _TOPIC_TOOL_NAME and enumerate_topics_tool is None:
            retry_count = state.get("verifier_retry_count", 0)
            feedback = (
                "Topic discovery is unavailable for this run. Write a normal user message instead."
            )
            return _build_retry_response(
                feedback=feedback,
                retry_count=retry_count,
                max_verification_retries=max_verification_retries,
                accumulated_usage=accumulated_usage,
                logger=logger,
            )
        content = parsed_response.message
        thought = parsed_response.thought
        topic_payload = [
            TopicCandidate(
                label=topic.label,
                guidance=topic.guidance,
                rationale=topic.rationale,
            ).model_dump()
            for topic in parsed_response.topics
        ]
        _log_event(
            logger,
            "human_agent_draft",
            {
                "retry_count": retry_count,
                "candidate_action": parsed_response.action,
                "candidate_message": content,
                "candidate_topics": topic_payload,
                "candidate_thought": thought,
                "has_thought": bool(thought),
                "has_verifier_feedback": bool(verifier_feedback),
            },
        )
        return {
            "candidate_human_action": parsed_response.action,
            "candidate_human_message": content,
            "candidate_human_thought": thought,
            "candidate_topic_candidates": topic_payload,
            "verifier_feedback": "",
            "verifier_decision": "",
            "verifier_retry_count": retry_count,
            "human_turn_token_usage": accumulated_usage,
        }

    def verifier_node(state: HumanAgentState) -> dict[str, object]:
        messages = _filter_user_visible_messages(state.get("messages", []))
        candidate_message = state.get("candidate_human_message", "").strip()
        candidate_thought = state.get("candidate_human_thought", "").strip()
        assert candidate_message is not None  # empty human message should not happen
        retry_count = state.get("verifier_retry_count", 0)
        accumulated_usage = state.get("human_turn_token_usage")

        verifier_response = verifier_structured_llm.invoke(
            [
                SystemMessage(content=verifier_system_prompt),
                HumanMessage(
                    content=(
                        "Conversation so far:\n"
                        f"{_format_conversation_for_verifier(messages)}\n\n"
                        "Candidate user message:\n"
                        f"{candidate_message}\n\n"
                        "Evaluate the candidate now."
                    )
                ),
            ]
        )
        assert isinstance(verifier_response, dict)  # langchain guarantees this
        raw_response = verifier_response.get("raw")
        if raw_response is not None:
            raw_token_usage = extract_token_usage(raw_response)
            usage_collector.record(
                extract_response_model(
                    raw_response, fallback_model=resolved_verifier_model_config.model
                ),
                raw_token_usage,
                source="human_verifier",
            )
            accumulated_usage = _accumulate_token_usage(accumulated_usage, raw_token_usage)
        parsing_error = verifier_response.get("parsing_error")
        if parsing_error is not None:
            raise ValueError("Verifier structured output parsing failed.") from parsing_error

        parsed_verifier_response = verifier_response.get("parsed")
        assert isinstance(parsed_verifier_response, VerifierDecision)  # langchain guarantees this
        decision, feedback = (
            parsed_verifier_response.decision,
            parsed_verifier_response.feedback.strip(),
        )
        if decision == "retry" and not feedback:
            raise ValueError("Verifier decision 'retry' requires non-empty feedback.")
        _log_event(
            logger,
            "human_agent_verifier_decision",
            {
                "decision": decision,
                "feedback": feedback,
                "retry_count": retry_count,
                "candidate_message": candidate_message,
            },
        )

        if decision == "allow" or retry_count >= max_verification_retries:
            _log_event(
                logger,
                "human_agent_message_allowed",
                {
                    "retry_count": retry_count,
                    "forced_allow": decision != "allow",
                    "final_message": candidate_message,
                },
            )
            return {
                "messages": [
                    _build_human_message(
                        candidate_message, accumulated_usage, thought=candidate_thought
                    )
                ],
                "candidate_human_action": "",
                "candidate_human_message": "",
                "candidate_human_thought": "",
                "candidate_topic_candidates": [],
                "verifier_feedback": "",
                "verifier_decision": "allow",
                "verifier_retry_count": 0,
                "human_turn_token_usage": {},
                "topic_discovery_force_generation": False,
            }

        _log_event(
            logger,
            "human_agent_retry_requested",
            {
                "retry_count": retry_count + 1,
                "feedback": feedback,
            },
        )
        return {
            "candidate_human_thought": "",
            "verifier_feedback": feedback,
            "verifier_decision": "retry",
            "verifier_retry_count": retry_count + 1,
            "human_turn_token_usage": accumulated_usage or {},
        }

    def enumerate_topics_node(state: HumanAgentState) -> dict[str, object]:
        if enumerate_topics_tool is None or topic_discovery_config is None:
            raise ValueError("Topic discovery tool is unavailable.")
        retry_count = state.get("verifier_retry_count", 0)
        accumulated_usage = state.get("human_turn_token_usage") or {}
        if state.get("topic_candidates") and not state.get("topic_candidates_accumulated"):
            feedback = "Topic candidates have already been stored for this run."
            return _build_retry_response(
                feedback=feedback,
                retry_count=retry_count,
                max_verification_retries=max_verification_retries,
                accumulated_usage=accumulated_usage,
                logger=logger,
                topic_discovery_force_generation=False,
            )

        raw_topic_candidates = state.get("candidate_topic_candidates", [])
        try:
            topics = [TopicCandidate.model_validate(topic) for topic in raw_topic_candidates]
            tool_content = cast(Any, enumerate_topics_tool).invoke({"topics": topics})
            tool_message_content = _content_to_text(tool_content)
        except Exception as exc:
            tool_message_content = f"Could not store topic candidates. Reason: {exc}"
            if retry_count >= max_verification_retries:
                raise ValueError(tool_message_content) from exc
            _log_event(
                logger,
                "human_agent_retry_requested",
                {
                    "retry_count": retry_count + 1,
                    "feedback": tool_message_content,
                },
            )
            return {
                "candidate_human_action": "",
                "candidate_human_message": "",
                "candidate_human_thought": "",
                "candidate_topic_candidates": [],
                "verifier_feedback": tool_message_content,
                "verifier_decision": "retry",
                "verifier_retry_count": retry_count + 1,
                "human_turn_token_usage": accumulated_usage,
                "topic_discovery_force_generation": bool(
                    state.get("topic_discovery_force_generation", False)
                ),
            }

        generated_at_turn = _current_iteration_count(state)
        topic_payload = [topic.model_dump() for topic in topics]
        accumulated_topic_payload = _coerce_topic_candidates_payload(
            state.get("topic_candidates_accumulated", [])
        )
        accumulated_by_topic_id = {topic["topic_id"]: topic for topic in accumulated_topic_payload}
        round_new_topic_count = 0
        for topic in topic_payload:
            topic_id = topic["topic_id"]
            if topic_id in accumulated_by_topic_id:
                continue
            accumulated_by_topic_id[topic_id] = topic
            round_new_topic_count += 1
        merged_topic_payload = list(accumulated_by_topic_id.values())
        round_count = (
            _coerce_topic_discovery_round_count(state.get("topic_discovery_round_count")) + 1
        )
        new_topics_by_round = _coerce_topics_per_round(
            state.get("topic_discovery_new_topics_by_round")
        )
        new_topics_by_round.append(round_new_topic_count)
        should_continue_topic_discovery = (
            round_count < topic_discovery_config.max_rounds
            and round_new_topic_count >= topic_discovery_config.min_new_topics_per_round
        )
        _log_event(
            logger,
            "human_agent_topics_enumerated",
            {
                "generated_at_turn": generated_at_turn,
                "trigger_mode": topic_discovery_config.trigger_mode,
                "round_count": round_count,
                "num_topics": len(topic_payload),
                "num_new_topics": round_new_topic_count,
                "num_accumulated_topics": len(merged_topic_payload),
                "continue_topic_discovery": should_continue_topic_discovery,
                "topics": topic_payload,
            },
        )
        if should_continue_topic_discovery:
            return {
                "topic_candidates_accumulated": merged_topic_payload,
                "topic_discovery_round_count": round_count,
                "topic_discovery_new_topics_by_round": new_topics_by_round,
                "candidate_human_action": "",
                "candidate_human_message": "",
                "candidate_human_thought": "",
                "candidate_topic_candidates": [],
                "verifier_feedback": "",
                "verifier_decision": "allow",
                "verifier_retry_count": 0,
                "human_turn_token_usage": accumulated_usage,
                "topic_discovery_force_generation": True,
            }
        topic_candidates_metadata: dict[str, object] = {
            "trigger_mode": topic_discovery_config.trigger_mode,
            "generated_at_turn": generated_at_turn,
        }
        if round_count > 1:
            topic_candidates_metadata["num_rounds"] = round_count
            topic_candidates_metadata["new_topics_by_round"] = new_topics_by_round
        return {
            "topic_candidates": merged_topic_payload,
            "topic_candidates_metadata": topic_candidates_metadata,
            "topic_candidates_accumulated": [],
            "topic_discovery_round_count": 0,
            "topic_discovery_new_topics_by_round": [],
            "candidate_human_action": "",
            "candidate_human_message": "",
            "candidate_human_thought": "",
            "candidate_topic_candidates": [],
            "verifier_feedback": "",
            "verifier_decision": "allow",
            "verifier_retry_count": 0,
            "human_turn_token_usage": accumulated_usage,
            "topic_discovery_force_generation": False,
        }

    def should_continue_after_human(state: HumanAgentState) -> str:
        if state.get("verifier_decision") == "retry":
            return "human"
        if state.get("candidate_human_action") == _TOPIC_TOOL_NAME:
            return _TOPIC_TOOL_NAME
        return "verifier"

    def should_continue_after_verifier(state: HumanAgentState) -> str:
        if state.get("verifier_decision") == "allow":
            return END
        return "human"

    def should_continue_after_topic_enumeration(state: HumanAgentState) -> str:
        if state.get("verifier_decision") == "retry":
            return "human"
        if state.get("topic_discovery_force_generation"):
            return "human"
        return END

    graph = StateGraph(cast(type[Any], HumanAgentState))
    graph.add_node(
        "human",
        human_node,
        retry_policy=RetryPolicy(
            initial_interval=0.0,
            backoff_factor=1.0,
            max_interval=0.0,
            max_attempts=_HUMAN_STRUCTURED_OUTPUT_NODE_RETRY_ATTEMPTS,
            jitter=False,
            retry_on=_is_retryable_human_structured_output_error,
        ),
    )
    graph.add_node("verifier", verifier_node)
    graph.add_node(_TOPIC_TOOL_NAME, enumerate_topics_node)
    graph.add_edge(START, "human")
    graph.add_conditional_edges("human", should_continue_after_human)
    graph.add_conditional_edges(_TOPIC_TOOL_NAME, should_continue_after_topic_enumeration)
    graph.add_conditional_edges("verifier", should_continue_after_verifier)
    return graph.compile()
