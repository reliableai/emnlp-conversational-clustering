"""Basic metrics evaluator that computes simple statistics on clustering artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from conversational_clustering.evaluation.base import (
    EvaluationRequest,
    EvaluationResult,
    Evaluator,
)


class BasicMetricsEvaluator(Evaluator):
    name = "basic_metrics"
    resume_version = "v1"

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None = None) -> BasicMetricsEvaluator:
        if params:
            unsupported = ", ".join(sorted(params))
            raise ValueError(
                f"Evaluator '{cls.name}' does not support parameters yet. Got: {unsupported}"
            )
        return cls()

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        return EvaluationResult(
            method=self.name,
            metrics={},
            details={
                "num_items": len(request.predictions.assignment),
                "num_clusters": len(request.predictions.cluster_names),
            },
        )
