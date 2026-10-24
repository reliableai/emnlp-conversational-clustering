"""Pairwise robustness evaluator for invariance and responsiveness expectations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from conversational_clustering.accounting import UsageCollector, build_pricing_client_from_configs
from conversational_clustering.evaluation.artifacts import LoadedClustering
from conversational_clustering.evaluation.base import (
    EvaluationResult,
    PairwiseEvaluationRequest,
    PairwiseEvaluator,
)
from conversational_clustering.evaluation.benchmark_spec import SyntheticBenchmarkSpec
from conversational_clustering.evaluation.methods.benchmark_theme_resolution import (
    build_theme_resolution_details,
    resolve_theme_for_request,
)
from conversational_clustering.evaluation.methods.benchmark_utils import (
    build_request_from_pairwise_side,
    compare_clusterings,
    require_pairwise_benchmark_spec,
)
from conversational_clustering.evaluation.methods.metrics import overlap_stats, safe_ratio
from conversational_clustering.evaluation.methods.semantic_quality_judges import judge_theme_match
from conversational_clustering.models import default_semantic_consistency_judge_model_config

_SUPPORTED_SIMILARITY_METRICS = frozenset(
    {
        "adjusted_rand_index",
        "normalized_mutual_info",
        "rand_index",
        "coassignment_jaccard",
        "fowlkes_mallows",
    }
)


@dataclass(frozen=True)
class RobustnessExpectationsEvaluator(PairwiseEvaluator):
    name = "robustness_expectations"
    resume_version = "v2"
    invariance_threshold: float | None = None
    responsiveness_threshold: float | None = None
    similarity_metric: str = "adjusted_rand_index"

    @classmethod
    def from_params(
        cls, params: Mapping[str, Any] | None = None
    ) -> RobustnessExpectationsEvaluator:
        normalized_params = dict(params or {})
        supported_keys = {
            "invariance_threshold",
            "responsiveness_threshold",
            "similarity_metric",
        }
        unsupported_keys = sorted(set(normalized_params) - supported_keys)
        if unsupported_keys:
            raise ValueError(
                f"Evaluator '{cls.name}' received unsupported parameter(s): "
                + ", ".join(unsupported_keys)
            )
        invariance_threshold = _parse_optional_probability(
            normalized_params,
            field_name="invariance_threshold",
        )
        responsiveness_threshold = _parse_optional_probability(
            normalized_params,
            field_name="responsiveness_threshold",
        )
        similarity_metric = str(
            normalized_params.get("similarity_metric", "adjusted_rand_index")
        ).strip()
        if similarity_metric not in _SUPPORTED_SIMILARITY_METRICS:
            raise ValueError(
                "'similarity_metric' must be one of: "
                + ", ".join(sorted(_SUPPORTED_SIMILARITY_METRICS))
            )
        return cls(
            invariance_threshold=invariance_threshold,
            responsiveness_threshold=responsiveness_threshold,
            similarity_metric=similarity_metric,
        )

    def evaluate(self, request: PairwiseEvaluationRequest) -> EvaluationResult:
        usage_collector = UsageCollector()
        default_benchmark_spec = require_pairwise_benchmark_spec(
            request.artifacts, evaluator_name=self.name
        )
        left_benchmark_spec = _resolve_benchmark_spec_for_side(
            artifacts=request.artifacts,
            side="left",
            default_benchmark_spec=default_benchmark_spec,
        )
        right_benchmark_spec = _resolve_benchmark_spec_for_side(
            artifacts=request.artifacts,
            side="right",
            default_benchmark_spec=default_benchmark_spec,
        )
        left_request = build_request_from_pairwise_side(request, side="left")
        right_request = build_request_from_pairwise_side(request, side="right")
        left_resolution = resolve_theme_for_request(
            artifacts=left_request.artifacts,
            benchmark_spec=left_benchmark_spec,
            evaluator_name=self.name,
            model_config=None,
            theme_match_prompt_id="baseline_v1",
            theme_matcher=judge_theme_match,
            usage_collector=usage_collector,
            predictions=request.left_predictions,
            granularity=None,
            allow_best_reference_fallback=True,
        )
        right_resolution = resolve_theme_for_request(
            artifacts=right_request.artifacts,
            benchmark_spec=right_benchmark_spec,
            evaluator_name=self.name,
            model_config=None,
            theme_match_prompt_id="baseline_v1",
            theme_matcher=judge_theme_match,
            usage_collector=usage_collector,
            predictions=request.right_predictions,
            granularity=None,
            allow_best_reference_fallback=True,
        )
        comparison_metrics, comparison_details = _compare_clusterings_allow_zero_overlap(
            left=request.left_predictions,
            right=request.right_predictions,
        )
        similarity_value = comparison_metrics[self.similarity_metric]
        expected_pair_type = (
            "invariant"
            if left_resolution.theme.theme_id == right_resolution.theme.theme_id
            else "responsive"
        )
        is_invariant_pair = expected_pair_type == "invariant"

        metrics = {
            **comparison_metrics,
            "invariance_pair_count": 1.0 if is_invariant_pair else 0.0,
            "responsiveness_pair_count": 0.0 if is_invariant_pair else 1.0,
        }
        details: dict[str, Any] = {
            "similarity_metric": self.similarity_metric,
            "expected_pair_type": expected_pair_type,
            "left_theme_id": left_resolution.theme.theme_id,
            "right_theme_id": right_resolution.theme.theme_id,
            **build_theme_resolution_details(
                artifacts=left_request.artifacts,
                resolved_theme=left_resolution.theme,
                resolution_mode=left_resolution.resolution_mode,
                prefix="left_",
            ),
            **build_theme_resolution_details(
                artifacts=right_request.artifacts,
                resolved_theme=right_resolution.theme,
                resolution_mode=right_resolution.resolution_mode,
                prefix="right_",
            ),
            **comparison_details,
        }
        if self.invariance_threshold is not None:
            details["invariance_threshold"] = self.invariance_threshold
            metrics["invariance_success_rate"] = (
                1.0 if is_invariant_pair and similarity_value >= self.invariance_threshold else 0.0
            )
        if self.responsiveness_threshold is not None:
            details["responsiveness_threshold"] = self.responsiveness_threshold
        if is_invariant_pair and self.invariance_threshold is not None:
            success = bool(metrics["invariance_success_rate"])
            details["margin_to_threshold"] = similarity_value - self.invariance_threshold
            details["success"] = success
        if self.responsiveness_threshold is not None:
            metrics["responsiveness_success_rate"] = (
                1.0
                if not is_invariant_pair and similarity_value <= self.responsiveness_threshold
                else 0.0
            )
        if not is_invariant_pair and self.responsiveness_threshold is not None:
            success = bool(metrics["responsiveness_success_rate"])
            details["margin_to_threshold"] = self.responsiveness_threshold - similarity_value
            details["success"] = success
        return EvaluationResult(
            method=self.name,
            metrics=metrics,
            details=details,
            accounting=usage_collector.snapshot(
                pricer=build_pricing_client_from_configs(
                    [default_semantic_consistency_judge_model_config()]
                )
            ).to_dict(),
        )


def _parse_optional_probability(
    params: Mapping[str, Any],
    *,
    field_name: str,
) -> float | None:
    if field_name not in params:
        return None
    value = params[field_name]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"'{field_name}' must be a numeric probability")
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value < -1.0 or parsed_value > 1.0:
        raise ValueError(f"'{field_name}' must be a finite number between -1.0 and 1.0")
    return parsed_value


def _resolve_benchmark_spec_for_side(
    *,
    artifacts: Mapping[str, Any],
    side: str,
    default_benchmark_spec: SyntheticBenchmarkSpec,
) -> SyntheticBenchmarkSpec:
    side_benchmark_spec = artifacts.get(f"{side}_benchmark_spec")
    if isinstance(side_benchmark_spec, SyntheticBenchmarkSpec):
        return side_benchmark_spec
    return default_benchmark_spec


def _compare_clusterings_allow_zero_overlap(
    *,
    left: LoadedClustering,
    right: LoadedClustering,
) -> tuple[dict[str, float], dict[str, Any]]:
    overlap = overlap_stats(left.assignment, right.assignment)
    if overlap.num_common_items == 0:
        return (
            {
                "adjusted_rand_index": 0.0,
                "normalized_mutual_info": 0.0,
                "rand_index": 0.0,
                "coassignment_jaccard": 0.0,
                "fowlkes_mallows": 0.0,
            },
            {
                "num_common_items": 0,
                "num_union_items": overlap.num_union_items,
                "common_item_coverage": safe_ratio(0, overlap.num_union_items, default=0.0),
                "same_item_set": False,
                "num_item_pairs_compared": 0,
                "true_positive_pairs": 0,
                "true_negative_pairs": 0,
                "false_positive_pairs": 0,
                "false_negative_pairs": 0,
            },
        )
    return compare_clusterings(left=left, right=right)
