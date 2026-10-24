"""Clarification-policy evaluator that judges ambiguity handling over conversation turns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from conversational_clustering.accounting import UsageCollector, build_pricing_client_from_configs
from conversational_clustering.evaluation.base import EvaluationRequest, EvaluationResult, Evaluator
from conversational_clustering.evaluation.benchmark_spec import SyntheticBenchmarkSpec
from conversational_clustering.evaluation.methods.benchmark_utils import (
    compare_clusterings,
    require_benchmark_spec,
    resolve_active_theme,
)
from conversational_clustering.evaluation.methods.semantic_quality_judges import (
    ClarificationOpportunityJudgment,
    judge_clarification_opportunity,
)
from conversational_clustering.models import (
    ModelConfig,
    default_semantic_consistency_judge_model_config,
    parse_model_config,
)
from conversational_clustering.prompts.registry import get_clarification_policy_prompt


@dataclass(frozen=True)
class _TraceMessage:
    role: str
    content: str
    turn_index: int | None


@dataclass(frozen=True)
class _ClarificationCandidate:
    request_turn_index: int
    request_text: str
    conversation_trace: str


@dataclass(frozen=True)
class ClarificationPolicyEvaluator(Evaluator):
    name = "clarification_policy"
    resume_version = "v1"
    model_config: ModelConfig | None = None
    prompt_id: str = "baseline_v1"

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None = None) -> ClarificationPolicyEvaluator:
        normalized_params = dict(params or {})
        supported_keys = {"model_config", "prompt_id"}
        unsupported_keys = sorted(set(normalized_params) - supported_keys)
        if unsupported_keys:
            raise ValueError(
                f"Evaluator '{cls.name}' received unsupported parameter(s): "
                + ", ".join(unsupported_keys)
            )

        model_config: ModelConfig | None = None
        raw_model_config = normalized_params.get("model_config")
        if raw_model_config is not None:
            if not isinstance(raw_model_config, Mapping):
                raise TypeError("model_config must be a dict")
            model_config = parse_model_config(raw_model_config)

        prompt_id = str(normalized_params.get("prompt_id", "baseline_v1")).strip()
        try:
            get_clarification_policy_prompt(prompt_id)
        except ValueError as exc:
            raise ValueError(f"Invalid prompt_id: {exc}") from exc

        return cls(
            model_config=model_config,
            prompt_id=prompt_id,
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        conversation_trace = request.artifacts.get("conversation_trace")
        if not isinstance(conversation_trace, Mapping):
            raise ValueError(
                f"Evaluator '{self.name}' requires 'conversation_trace' in request artifacts"
            )

        benchmark_spec = require_benchmark_spec(request.artifacts, evaluator_name=self.name)
        trace_messages = _parse_trace_messages(conversation_trace)
        candidates = _build_clarification_candidates(trace_messages=trace_messages)

        benchmark_context = _build_benchmark_context(
            request=request,
            benchmark_spec=benchmark_spec,
        )
        final_clustering = _format_clustering(request.predictions)
        usage_collector = UsageCollector()

        num_needed_clarifications = 0
        num_asked_clarifications = 0
        num_useful_clarifications = 0
        num_unnecessary_clarifications = 0
        num_missed_clarifications = 0
        num_opportunities = 0
        sum_downstream_gain = 0.0
        opportunity_records: list[dict[str, Any]] = []

        for candidate in candidates:
            judgment = judge_clarification_opportunity(
                request_turn_index=candidate.request_turn_index,
                request_text=candidate.request_text,
                conversation_trace=candidate.conversation_trace,
                benchmark_context=benchmark_context,
                final_clustering=final_clustering,
                model_config=self.model_config,
                prompt_id=self.prompt_id,
                usage_collector=usage_collector,
            )
            normalized_judgment = _normalize_judgment(judgment=judgment)

            if normalized_judgment.clarification_needed:
                num_needed_clarifications += 1
            if normalized_judgment.asked_clarification:
                num_asked_clarifications += 1
                sum_downstream_gain += normalized_judgment.downstream_gain
            if normalized_judgment.useful_clarification:
                num_useful_clarifications += 1
            if normalized_judgment.unnecessary_clarification:
                num_unnecessary_clarifications += 1
            if normalized_judgment.missed_clarification:
                num_missed_clarifications += 1

            if normalized_judgment.opportunity_exists:
                num_opportunities += 1
                opportunity_records.append(
                    {
                        "request_turn_index": candidate.request_turn_index,
                        "request_text": candidate.request_text,
                        "conversation_trace": candidate.conversation_trace,
                        "judgment": normalized_judgment.model_dump(),
                    }
                )

        mean_downstream_gain_for_asked = _safe_ratio(
            sum_downstream_gain,
            num_asked_clarifications,
            default=0.0,
        )

        accounting = usage_collector.snapshot(
            pricer=build_pricing_client_from_configs(
                [self.model_config]
                if self.model_config is not None
                else [default_semantic_consistency_judge_model_config()]
            )
        ).to_dict()

        return EvaluationResult(
            method=self.name,
            metrics={
                "useful_clarification_rate": _safe_ratio(
                    num_useful_clarifications,
                    num_asked_clarifications,
                    default=0.0,
                ),
                "unnecessary_clarification_rate": _safe_ratio(
                    num_unnecessary_clarifications,
                    num_asked_clarifications,
                    default=0.0,
                ),
                "missed_clarification_rate": _safe_ratio(
                    num_missed_clarifications,
                    num_needed_clarifications,
                    default=0.0,
                ),
                "downstream_gain_from_clarification": mean_downstream_gain_for_asked,
            },
            details={
                "prompt_id": self.prompt_id,
                "num_opportunities": num_opportunities,
                "num_needed_clarifications": num_needed_clarifications,
                "num_asked_clarifications": num_asked_clarifications,
                "num_useful_clarifications": num_useful_clarifications,
                "num_unnecessary_clarifications": num_unnecessary_clarifications,
                "num_missed_clarifications": num_missed_clarifications,
                "sum_downstream_gain": sum_downstream_gain,
                "mean_downstream_gain_for_asked": mean_downstream_gain_for_asked,
                "opportunities": opportunity_records,
            },
            accounting=accounting,
        )


def _parse_trace_messages(conversation_trace: Mapping[str, Any]) -> list[_TraceMessage]:
    raw_messages = conversation_trace.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError(
            "Evaluator 'clarification_policy' expected conversation_trace.messages to be a list"
        )

    trace_messages: list[_TraceMessage] = []
    human_turn_index = 0
    for index, raw_message in enumerate(raw_messages):
        if not isinstance(raw_message, Mapping):
            raise ValueError(
                "conversation_trace.messages entries must be objects with role/content fields; "
                f"received {type(raw_message).__name__} at index {index}"
            )
        raw_role = raw_message.get("role")
        if not isinstance(raw_role, str) or not raw_role.strip():
            raise ValueError(
                f"conversation_trace.messages[{index}].role must be a non-empty string"
            )
        role = raw_role.strip().lower()
        raw_content = raw_message.get("content", "")
        content = raw_content if isinstance(raw_content, str) else str(raw_content)
        if role == "human":
            human_turn_index += 1
            turn_index = human_turn_index
        elif role == "assistant":
            turn_index = human_turn_index if human_turn_index > 0 else None
        else:
            turn_index = None
        trace_messages.append(_TraceMessage(role=role, content=content, turn_index=turn_index))
    return trace_messages


def _build_clarification_candidates(
    *,
    trace_messages: list[_TraceMessage],
) -> list[_ClarificationCandidate]:
    conversation_trace = _format_conversation_trace(trace_messages)
    human_turns = [
        message
        for message in trace_messages
        if message.role == "human" and message.turn_index is not None
    ]

    candidates: list[_ClarificationCandidate] = []
    for message in human_turns:
        request_turn_index = message.turn_index
        assert request_turn_index is not None
        candidates.append(
            _ClarificationCandidate(
                request_turn_index=request_turn_index,
                request_text=message.content,
                conversation_trace=conversation_trace,
            )
        )
    return candidates


def _format_conversation_trace(messages: Sequence[_TraceMessage]) -> str:
    if not messages:
        return "(no messages available)"
    lines: list[str] = []
    for message in messages:
        if message.turn_index is None:
            lines.append(f"Unnumbered {message.role}: {message.content}")
        else:
            lines.append(f"Turn {message.turn_index} {message.role}: {message.content}")
    return "\n".join(lines)


def _build_benchmark_context(
    *,
    request: EvaluationRequest,
    benchmark_spec: SyntheticBenchmarkSpec,
) -> str:
    lines = [f"Dataset: {benchmark_spec.dataset_name}"]
    active_theme = resolve_active_theme(artifacts=request.artifacts, benchmark_spec=benchmark_spec)
    if active_theme is None:
        raise ValueError(
            "Evaluator 'clarification_policy' could not resolve the active benchmark theme"
        )

    lines.append(f"Active theme id: {active_theme.theme_id}")
    lines.append(f"Active theme label: {active_theme.label}")
    lines.append(f"Active theme guidance: {active_theme.guidance}")
    try:
        reference = benchmark_spec.resolve_reference(
            topic_id=active_theme.theme_id,
            granularity=active_theme.granularity,
        )
        metrics, _ = compare_clusterings(
            left=request.predictions,
            right=reference.to_loaded_clustering(),
        )
        lines.append("Final clustering vs active-theme benchmark reference:")
        for metric_name in sorted(metrics):
            lines.append(f"- {metric_name}: {metrics[metric_name]:.4f}")
    except ValueError:
        lines.append("Final clustering could not be compared to an active-theme reference.")
    return "\n".join(lines)


def _format_clustering(predictions: Any) -> str:
    clusters: dict[int, list[str]] = {}
    for item_text, cluster_id in predictions.assignment.items():
        clusters.setdefault(cluster_id, []).append(item_text)
    lines: list[str] = []
    for cluster_id in sorted(clusters):
        cluster_name = (
            predictions.cluster_names[cluster_id]
            if cluster_id < len(predictions.cluster_names)
            else f"cluster_{cluster_id}"
        )
        lines.append(f"{cluster_name}:")
        for member in sorted(clusters[cluster_id]):
            lines.append(f"- {member}")
    return "\n".join(lines)


def _normalize_judgment(
    *,
    judgment: ClarificationOpportunityJudgment,
) -> ClarificationOpportunityJudgment:
    normalized_payload = judgment.model_dump()
    if not normalized_payload["asked_clarification"]:
        normalized_payload["useful_clarification"] = False
        normalized_payload["unnecessary_clarification"] = False
        normalized_payload["downstream_gain"] = 0.0
        normalized_payload["missed_clarification"] = bool(
            normalized_payload["clarification_needed"]
        )
    if not normalized_payload["clarification_needed"]:
        normalized_payload["missed_clarification"] = False
    if normalized_payload["useful_clarification"]:
        normalized_payload["unnecessary_clarification"] = False
    if normalized_payload["unnecessary_clarification"]:
        normalized_payload["useful_clarification"] = False
        normalized_payload["downstream_gain"] = 0.0
    if not normalized_payload["useful_clarification"]:
        normalized_payload["downstream_gain"] = 0.0
    downstream_gain = float(normalized_payload["downstream_gain"])
    normalized_payload["downstream_gain"] = min(1.0, max(0.0, downstream_gain))
    return ClarificationOpportunityJudgment.model_validate(normalized_payload)


def _safe_ratio(numerator: float, denominator: int, *, default: float) -> float:
    if denominator <= 0:
        return default
    return numerator / denominator
