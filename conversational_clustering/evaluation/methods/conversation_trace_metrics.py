"""Conversation trace evaluator that computes basic message and tool-call statistics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from conversational_clustering.evaluation.base import (
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
)


class ConversationTraceMetricsEvaluator(Evaluator):
    name = "conversation_trace_metrics"
    resume_version = "v1"

    @classmethod
    def from_params(
        cls, params: Mapping[str, Any] | None = None
    ) -> ConversationTraceMetricsEvaluator:
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

        messages = conversation_trace.get("messages", [])
        if not isinstance(messages, list):
            raise ValueError(
                f"Evaluator '{self.name}' expected conversation_trace.messages to be a list"
            )

        total_tool_calls_by_type = _parse_tool_call_counts(
            conversation_trace.get("total_tool_calls_by_type"),
            field_name="total_tool_calls_by_type",
        )
        total_failed_tool_calls_by_type = _parse_tool_call_counts(
            conversation_trace.get("total_failed_tool_calls_by_type"),
            field_name="total_failed_tool_calls_by_type",
        )

        num_tool_calls = sum(total_tool_calls_by_type.values())
        num_failed_tool_calls = sum(total_failed_tool_calls_by_type.values())
        if num_failed_tool_calls > num_tool_calls:
            raise ValueError(
                "conversation_trace.total_failed_tool_calls_by_type cannot exceed "
                "conversation_trace.total_tool_calls_by_type"
            )
        num_successful_tool_calls = num_tool_calls - num_failed_tool_calls

        return EvaluationResult(
            method=self.name,
            metrics={},
            details={
                "conversation_length_messages": len(messages),
                "num_tool_calls": num_tool_calls,
                "num_successful_tool_calls": num_successful_tool_calls,
                "num_failed_tool_calls": num_failed_tool_calls,
                "total_tool_calls_by_type": total_tool_calls_by_type,
                "total_failed_tool_calls_by_type": total_failed_tool_calls_by_type,
            },
        )


def _parse_tool_call_counts(payload: Any, *, field_name: str) -> dict[str, int]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(
            f"conversation_trace.{field_name} must be an object mapping tool names to counts"
        )

    counts: dict[str, int] = {}
    for raw_tool_name, raw_count in payload.items():
        if not isinstance(raw_tool_name, str) or not raw_tool_name.strip():
            raise ValueError(
                f"conversation_trace.{field_name} keys must be non-empty tool-name strings"
            )
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            raise ValueError(
                f"conversation_trace.{field_name}.{raw_tool_name} must be a non-negative integer"
            )
        counts[raw_tool_name] = raw_count
    return {tool_name: counts[tool_name] for tool_name in sorted(counts)}
