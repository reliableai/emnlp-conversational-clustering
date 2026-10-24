"""Human-effort evaluator that measures conversational turn and message word-count statistics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from conversational_clustering.evaluation.base import (
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
)


class HumanEffortMetricsEvaluator(Evaluator):
    name = "human_effort_metrics"
    resume_version = "v1"

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None = None) -> HumanEffortMetricsEvaluator:
        if params:
            unsupported = ", ".join(sorted(params))
            raise ValueError(
                f"Evaluator '{cls.name}' does not support parameters yet. Got: {unsupported}"
            )
        return cls()

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        conversation_trace = request.artifacts.get("conversation_trace")
        if not isinstance(conversation_trace, dict):
            raise ValueError(
                f"Evaluator '{self.name}' requires 'conversation_trace' in request artifacts"
            )

        raw_messages = conversation_trace.get("messages", [])
        if not isinstance(raw_messages, list):
            raise ValueError(
                f"Evaluator '{self.name}' expected conversation_trace.messages to be a list"
            )

        human_message_lengths: list[int] = []
        agent_message_lengths: list[int] = []
        for index, raw_message in enumerate(raw_messages):
            if not isinstance(raw_message, dict):
                raise ValueError(
                    "conversation_trace.messages entries must be objects with role/content fields; "
                    f"received {type(raw_message).__name__} at index {index}"
                )
            raw_role = raw_message.get("role")
            if not isinstance(raw_role, str) or not raw_role.strip():
                raise ValueError(
                    f"conversation_trace.messages[{index}].role must be a non-empty string"
                )
            raw_content = raw_message.get("content", "")
            content = raw_content if isinstance(raw_content, str) else str(raw_content)
            message_word_count = _count_words(content)

            normalized_role = raw_role.strip().lower()
            if normalized_role == "human":
                human_message_lengths.append(message_word_count)
            elif normalized_role == "assistant":
                agent_message_lengths.append(message_word_count)

        human_turn_count = len(human_message_lengths)
        agent_turn_count = len(agent_message_lengths)
        total_human_words = sum(human_message_lengths)
        total_agent_words = sum(agent_message_lengths)

        return EvaluationResult(
            method=self.name,
            metrics={},
            details={
                "turn_count": human_turn_count + agent_turn_count,
                "human_turn_count": human_turn_count,
                "agent_turn_count": agent_turn_count,
                "avg_human_message_length": _mean(human_message_lengths),
                "avg_agent_message_length": _mean(agent_message_lengths),
                "human_to_agent_word_ratio": _safe_ratio(
                    total_human_words,
                    total_agent_words,
                    default=None,
                ),
            },
        )


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_ratio(numerator: float, denominator: float, *, default: float | None) -> float | None:
    if denominator == 0:
        return default
    return numerator / denominator


def _count_words(content: str) -> int:
    return len(content.split())
