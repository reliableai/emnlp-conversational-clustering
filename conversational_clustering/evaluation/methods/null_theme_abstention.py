"""Evaluator for whether the assistant rejected a configured unsupported POV."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from conversational_clustering.accounting import UsageCollector, build_pricing_client_from_configs
from conversational_clustering.evaluation.base import EvaluationRequest, EvaluationResult, Evaluator
from conversational_clustering.evaluation.methods.semantic_quality_judges import (
    NullThemeAbstentionJudgment,
    NullThemeUnsupportedTopicMatchJudgment,
    build_structured_semantic_judge,
    judge_null_theme_abstention_trace,
    judge_null_theme_unsupported_topic_match,
)
from conversational_clustering.models import (
    ModelConfig,
    default_semantic_consistency_judge_model_config,
    parse_model_config,
)
from conversational_clustering.prompts.registry import (
    get_null_theme_abstention_topic_match_judge_prompt,
    get_null_theme_abstention_trace_judge_prompt,
)


@dataclass(frozen=True)
class UnsupportedTopicSpec:
    topic_id: str
    label: str
    guidance: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "label": self.label,
            "guidance": self.guidance,
        }

    def format_for_prompt(self) -> str:
        return f"topic_id={self.topic_id}\nlabel={self.label}\nguidance={self.guidance}"


@dataclass(frozen=True)
class RequestedTopic:
    topic_id: str | None
    label: str | None
    guidance: str | None

    def to_payload(self) -> dict[str, str | None]:
        return {
            "topic_id": self.topic_id,
            "label": self.label,
            "guidance": self.guidance,
        }

    def format_for_prompt(self) -> str:
        parts: list[str] = []
        if self.topic_id is not None:
            parts.append(f"topic_id={self.topic_id}")
        if self.label is not None:
            parts.append(f"label={self.label}")
        if self.guidance is not None:
            parts.append(f"guidance={self.guidance}")
        return "\n".join(parts)


@dataclass(frozen=True)
class NullThemeAbstentionEvaluator(Evaluator):
    name = "null_theme_abstention"
    resume_version = "v3"
    unsupported_topics: tuple[UnsupportedTopicSpec, ...]
    model_config: ModelConfig | None = None
    match_judge_prompt_id: str = "baseline_v1"
    trace_judge_prompt_id: str = "baseline_v1"

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None = None) -> NullThemeAbstentionEvaluator:
        normalized_params = dict(params or {})
        supported_keys = {
            "unsupported_topics",
            "model_config",
            "match_judge_prompt_id",
            "trace_judge_prompt_id",
        }
        unsupported_keys = sorted(set(normalized_params) - supported_keys)
        if unsupported_keys:
            raise ValueError(
                f"Evaluator '{cls.name}' received unsupported parameter(s): "
                + ", ".join(unsupported_keys)
            )

        unsupported_topics = _parse_unsupported_topics(
            normalized_params.get("unsupported_topics"),
            evaluator_name=cls.name,
        )
        model_config = _parse_optional_model_config(normalized_params.get("model_config"))
        match_judge_prompt_id = str(
            normalized_params.get("match_judge_prompt_id", "baseline_v1")
        ).strip()
        try:
            get_null_theme_abstention_topic_match_judge_prompt(match_judge_prompt_id)
        except ValueError as exc:
            raise ValueError(f"Invalid match_judge_prompt_id: {exc}") from exc

        trace_judge_prompt_id = str(
            normalized_params.get("trace_judge_prompt_id", "baseline_v1")
        ).strip()
        try:
            get_null_theme_abstention_trace_judge_prompt(trace_judge_prompt_id)
        except ValueError as exc:
            raise ValueError(f"Invalid trace_judge_prompt_id: {exc}") from exc

        return cls(
            unsupported_topics=unsupported_topics,
            model_config=model_config,
            match_judge_prompt_id=match_judge_prompt_id,
            trace_judge_prompt_id=trace_judge_prompt_id,
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        usage_collector = UsageCollector()
        requested_topic = _resolve_requested_topic(request.artifacts, evaluator_name=self.name)
        requested_guidance = _resolve_requested_guidance(
            requested_topic=requested_topic,
            evaluator_name=self.name,
        )
        trace_transcript = _format_conversation_trace(
            request.artifacts.get("conversation_trace"),
            evaluator_name=self.name,
        )
        topic_match_judgment = judge_null_theme_unsupported_topic_match(
            requested_guidance=requested_guidance,
            unsupported_topics=_format_unsupported_topics_for_prompt(self.unsupported_topics),
            model_config=self.model_config,
            prompt_id=self.match_judge_prompt_id,
            structured_llm=self._build_topic_match_judge,
            usage_collector=usage_collector,
        )
        matched_unsupported_topic = _resolve_matched_unsupported_topic(
            topic_match_judgment=topic_match_judgment,
            unsupported_topics=self.unsupported_topics,
            evaluator_name=self.name,
        )

        trace_judgment = judge_null_theme_abstention_trace(
            requested_pov=requested_guidance,
            matched_unsupported_topic=matched_unsupported_topic.format_for_prompt(),
            conversation_trace=trace_transcript,
            model_config=self.model_config,
            prompt_id=self.trace_judge_prompt_id,
            structured_llm=self._build_trace_judge,
            usage_collector=usage_collector,
        )

        abstention_success = (
            trace_judgment.rejected_requested_pov
            and not trace_judgment.assistant_agreed_with_requested_pov
        )
        return EvaluationResult(
            method=self.name,
            metrics={
                "abstention_success": 1.0 if abstention_success else 0.0,
                "explicit_rejection_detected": (
                    1.0 if trace_judgment.rejected_requested_pov else 0.0
                ),
                "assistant_agreement_detected": (
                    1.0 if trace_judgment.assistant_agreed_with_requested_pov else 0.0
                ),
            },
            details={
                "resolved_requested_topic": requested_topic.to_payload(),
                "matched_unsupported_topic": matched_unsupported_topic.to_payload(),
                "topic_match_judgment": topic_match_judgment.model_dump(),
                "trace_transcript": trace_transcript,
                "trace_judgment": trace_judgment.model_dump(),
            },
            accounting=usage_collector.snapshot(
                pricer=build_pricing_client_from_configs(
                    [self.model_config]
                    if self.model_config
                    else [default_semantic_consistency_judge_model_config()]
                )
            ).to_dict(),
        )

    def _build_trace_judge(self) -> Any:
        return build_structured_semantic_judge(
            schema=NullThemeAbstentionJudgment,
            model_config=self.model_config,
        )

    def _build_topic_match_judge(self) -> Any:
        return build_structured_semantic_judge(
            schema=NullThemeUnsupportedTopicMatchJudgment,
            model_config=self.model_config,
        )


def _parse_unsupported_topics(
    raw_unsupported_topics: object,
    *,
    evaluator_name: str,
) -> tuple[UnsupportedTopicSpec, ...]:
    if not isinstance(raw_unsupported_topics, Sequence) or isinstance(
        raw_unsupported_topics, str | bytes
    ):
        raise ValueError(
            f"Evaluator '{evaluator_name}' requires 'unsupported_topics' as a non-empty list"
        )
    if not raw_unsupported_topics:
        raise ValueError(
            f"Evaluator '{evaluator_name}' requires 'unsupported_topics' as a non-empty list"
        )

    return tuple(
        _parse_unsupported_topic(raw_topic, topic_index=index, evaluator_name=evaluator_name)
        for index, raw_topic in enumerate(raw_unsupported_topics, start=1)
    )


def _parse_unsupported_topic(
    raw_topic: object,
    *,
    topic_index: int,
    evaluator_name: str,
) -> UnsupportedTopicSpec:
    context = f"Evaluator '{evaluator_name}' unsupported_topics[{topic_index}]"
    if not isinstance(raw_topic, Mapping):
        raise ValueError(f"{context} must be an object")

    normalized_topic = dict(raw_topic)
    supported_keys = {"topic_id", "label", "guidance"}
    unknown_keys = sorted(set(normalized_topic) - supported_keys)
    if unknown_keys:
        raise ValueError(f"{context} has unsupported field(s): {', '.join(unknown_keys)}")

    missing_required_keys = [
        key for key in ("topic_id", "label", "guidance") if key not in normalized_topic
    ]
    if missing_required_keys:
        raise ValueError(
            f"{context} is missing required field(s): {', '.join(missing_required_keys)}"
        )

    return UnsupportedTopicSpec(
        topic_id=_require_non_empty_string(
            normalized_topic.get("topic_id"), context=f"{context}.topic_id"
        ),
        label=_require_non_empty_string(normalized_topic.get("label"), context=f"{context}.label"),
        guidance=_require_non_empty_string(
            normalized_topic.get("guidance"), context=f"{context}.guidance"
        ),
    )


def _require_non_empty_string(raw_value: object, *, context: str) -> str:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return raw_value.strip()


def _parse_optional_model_config(raw_model_config: object) -> ModelConfig | None:
    if raw_model_config is None:
        return None
    if not isinstance(raw_model_config, Mapping):
        raise TypeError("model_config must be a dict")
    try:
        return parse_model_config(cast(Mapping[str, object], raw_model_config))
    except ValueError as exc:
        raise ValueError(f"'model_config' is invalid: {exc}") from exc


def _resolve_requested_topic(
    artifacts: Mapping[str, Any],
    *,
    evaluator_name: str,
) -> RequestedTopic:
    selected_topic = artifacts.get("selected_topic")
    if isinstance(selected_topic, Mapping):
        topic = _build_requested_topic(
            topic_id=selected_topic.get("topic_id"),
            label=selected_topic.get("label"),
            guidance=selected_topic.get("guidance"),
        )
        if topic is not None:
            return topic

    batch_run_config = artifacts.get("batch_run_config")
    if isinstance(batch_run_config, Mapping):
        metadata = batch_run_config.get("metadata")
        if isinstance(metadata, Mapping):
            topic = _build_requested_topic(
                topic_id=metadata.get("topic_branch_topic_id"),
                label=metadata.get("topic_branch_topic_label"),
                guidance=metadata.get("topic_branch_topic_guidance"),
            )
            if topic is not None:
                return topic

        human_agent_config = batch_run_config.get("human_agent_config")
        if isinstance(human_agent_config, Mapping):
            topic = _build_requested_topic(
                topic_id=None,
                label=None,
                guidance=human_agent_config.get("guidance"),
            )
            if topic is not None:
                return topic

    raise ValueError(
        f"Evaluator '{evaluator_name}' could not resolve the requested unsupported POV from "
        "selected_topic or batch_run_config."
    )


def _build_requested_topic(
    *,
    topic_id: object,
    label: object,
    guidance: object,
) -> RequestedTopic | None:
    normalized_topic_id = _normalize_optional_string(topic_id)
    normalized_label = _normalize_optional_string(label)
    normalized_guidance = _normalize_optional_string(guidance)
    if normalized_topic_id is None and normalized_label is None and normalized_guidance is None:
        return None
    return RequestedTopic(
        topic_id=normalized_topic_id,
        label=normalized_label,
        guidance=normalized_guidance,
    )


def _normalize_optional_string(raw_value: object) -> str | None:
    if not isinstance(raw_value, str):
        return None
    normalized_value = raw_value.strip()
    return normalized_value or None


def _format_unsupported_topics_for_prompt(
    unsupported_topics: tuple[UnsupportedTopicSpec, ...],
) -> str:
    return "\n\n".join(
        f"Unsupported theme #{index}\n{unsupported_topic.format_for_prompt()}"
        for index, unsupported_topic in enumerate(unsupported_topics, start=1)
    )


def _resolve_requested_guidance(*, requested_topic: RequestedTopic, evaluator_name: str) -> str:
    if requested_topic.guidance is None:
        raise ValueError(
            f"Evaluator '{evaluator_name}' requires requested POV guidance for null-theme matching."
        )
    return requested_topic.guidance


def _resolve_matched_unsupported_topic(
    *,
    topic_match_judgment: NullThemeUnsupportedTopicMatchJudgment,
    unsupported_topics: tuple[UnsupportedTopicSpec, ...],
    evaluator_name: str,
) -> UnsupportedTopicSpec:
    matched_topic_id = _normalize_optional_string(topic_match_judgment.matched_unsupported_topic_id)
    if matched_topic_id is None:
        raise ValueError(
            f"Evaluator '{evaluator_name}' could not match the initial requested POV to any "
            "configured unsupported topic."
        )

    for unsupported_topic in unsupported_topics:
        if unsupported_topic.topic_id == matched_topic_id:
            return unsupported_topic

    raise ValueError(
        f"Evaluator '{evaluator_name}' topic-match judge returned unknown unsupported topic "
        f"id '{matched_topic_id}'."
    )


def _format_conversation_trace(raw_conversation_trace: object, *, evaluator_name: str) -> str:
    if not isinstance(raw_conversation_trace, Mapping):
        raise ValueError(
            f"Evaluator '{evaluator_name}' requires 'conversation_trace' in request artifacts"
        )
    conversation_trace = cast(Mapping[str, object], raw_conversation_trace)
    raw_messages = conversation_trace.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError(
            f"Evaluator '{evaluator_name}' expected conversation_trace.messages to be a list"
        )

    lines: list[str] = []
    for index, raw_message in enumerate(raw_messages):
        if not isinstance(raw_message, Mapping):
            raise ValueError(
                f"conversation_trace.messages[{index}] must be an object with role/content fields"
            )
        message = cast(Mapping[str, object], raw_message)
        raw_role = message.get("role")
        raw_content = message.get("content")
        if not isinstance(raw_role, str) or not raw_role.strip():
            raise ValueError(
                f"conversation_trace.messages[{index}].role must be a non-empty string"
            )
        if not isinstance(raw_content, str):
            raise ValueError(f"conversation_trace.messages[{index}].content must be a string")
        normalized_role = raw_role.strip().lower()
        if normalized_role not in {"human", "assistant"}:
            continue
        normalized_content = raw_content.strip()
        if not normalized_content:
            continue
        lines.append(f"{normalized_role}: {normalized_content}")

    return "\n".join(lines)
