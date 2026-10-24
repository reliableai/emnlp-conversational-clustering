"""Core evaluator request/result types and the evaluator protocol interface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..accounting import merge_usage_summaries
from .artifacts import LoadedClustering


@dataclass(frozen=True)
class EvaluationRequest:
    predictions: LoadedClustering
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairwiseEvaluationRequest:
    left_predictions: LoadedClustering
    right_predictions: LoadedClustering
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    method: str
    metrics: dict[str, float]
    details: dict[str, Any]
    accounting: dict[str, Any] | None = None

    @classmethod
    def summarize(cls, results: list[EvaluationResult]) -> EvaluationResult:
        accumulator = EvaluationSummaryAccumulator()
        for result in results:
            accumulator.add_result(result)
        return accumulator.to_summary_result()


@dataclass
class EvaluationSummaryAccumulator:
    metric_totals: dict[str, float] = field(default_factory=dict)
    metric_counts: dict[str, int] = field(default_factory=dict)
    detail_totals: dict[str, float] = field(default_factory=dict)
    detail_counts: dict[str, int] = field(default_factory=dict)
    methods: set[str] = field(default_factory=set)
    num_results: int = 0
    accounting_records: list[dict[str, Any]] = field(default_factory=list)

    def add_result(self, result: EvaluationResult) -> None:
        self.num_results += 1
        self.methods.add(result.method)
        if result.accounting is not None:
            self.accounting_records.append(result.accounting)

        for metric_name, metric_value in result.metrics.items():
            self.metric_totals[metric_name] = (
                self.metric_totals.get(metric_name, 0.0) + metric_value
            )
            self.metric_counts[metric_name] = self.metric_counts.get(metric_name, 0) + 1

        for detail_name, detail_value in result.details.items():
            if isinstance(detail_value, bool):
                continue
            if isinstance(detail_value, int | float):
                self.detail_totals[detail_name] = self.detail_totals.get(detail_name, 0.0) + float(
                    detail_value
                )
                self.detail_counts[detail_name] = self.detail_counts.get(detail_name, 0) + 1

    def to_summary_result(self) -> EvaluationResult:
        mean_metrics = {
            metric_name: self.metric_totals[metric_name] / self.metric_counts[metric_name]
            for metric_name in sorted(self.metric_totals)
        }
        mean_metrics = _normalize_summary_metrics(mean_metrics)
        mean_numeric_details = {
            detail_name: self.detail_totals[detail_name] / self.detail_counts[detail_name]
            for detail_name in sorted(self.detail_totals)
        }
        accounting = (
            merge_usage_summaries(self.accounting_records).to_dict()
            if self.accounting_records
            else None
        )

        return EvaluationResult(
            method="summary",
            metrics=mean_metrics,
            details={
                "num_results": self.num_results,
                "methods": sorted(self.methods),
                "mean_numeric_details": mean_numeric_details,
            },
            accounting=accounting,
        )


class Evaluator(Protocol):
    name: str
    resume_version: str

    @classmethod
    def from_params(
        cls,
        params: Mapping[str, Any] | None = None,
    ) -> Evaluator: ...

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult: ...


class PairwiseEvaluator(Protocol):
    name: str
    resume_version: str

    @classmethod
    def from_params(
        cls,
        params: Mapping[str, Any] | None = None,
    ) -> PairwiseEvaluator: ...

    def evaluate(self, request: PairwiseEvaluationRequest) -> EvaluationResult: ...


def _normalize_summary_metrics(mean_metrics: dict[str, float]) -> dict[str, float]:
    normalized_metrics = dict(mean_metrics)
    _normalize_rate_by_count(
        metrics=normalized_metrics,
        rate_metric="invariance_success_rate",
        count_metric="invariance_pair_count",
    )
    _normalize_rate_by_count(
        metrics=normalized_metrics,
        rate_metric="responsiveness_success_rate",
        count_metric="responsiveness_pair_count",
    )
    return normalized_metrics


def _normalize_rate_by_count(
    *,
    metrics: dict[str, float],
    rate_metric: str,
    count_metric: str,
) -> None:
    rate_value = metrics.get(rate_metric)
    count_value = metrics.get(count_metric)
    if rate_value is None or count_value is None:
        return
    if count_value <= 0.0:
        metrics.pop(rate_metric, None)
        return
    metrics[rate_metric] = rate_value / count_value
