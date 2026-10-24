"""Pairwise clustering consistency evaluator for comparing two clustering artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from conversational_clustering.evaluation.base import (
    EvaluationResult,
    PairwiseEvaluationRequest,
    PairwiseEvaluator,
)
from conversational_clustering.evaluation.methods.metrics import (
    coassignment_jaccard,
    fowlkes_mallows,
    normalized_mutual_info,
    overlap_stats,
    pairwise_adjusted_rand_index,
    pairwise_counts,
    rand_index,
    safe_ratio,
)


class PairwiseConsistencyEvaluator(PairwiseEvaluator):
    name = "pairwise_consistency"
    resume_version = "v1"

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None = None) -> PairwiseConsistencyEvaluator:
        if params:
            unsupported = ", ".join(sorted(params))
            raise ValueError(
                f"Evaluator '{cls.name}' does not support parameters yet. Got: {unsupported}"
            )
        return cls()

    def evaluate(self, request: PairwiseEvaluationRequest) -> EvaluationResult:
        left_assignment = request.left_predictions.assignment
        right_assignment = request.right_predictions.assignment
        overlap = overlap_stats(left_assignment, right_assignment)
        num_common_items = overlap.num_common_items
        num_union_items = overlap.num_union_items

        if num_common_items == 0:
            return EvaluationResult(
                method=self.name,
                metrics={
                    "adjusted_rand_index": 0.0,
                    "normalized_mutual_info": 0.0,
                    "rand_index": 0.0,
                    "coassignment_jaccard": 0.0,
                    "fowlkes_mallows": 0.0,
                },
                details={
                    "num_common_items": 0,
                    "num_union_items": num_union_items,
                    "common_item_coverage": 0.0,
                    "same_item_set": False,
                    "num_item_pairs_compared": 0,
                    "true_positive_pairs": 0,
                    "true_negative_pairs": 0,
                    "false_positive_pairs": 0,
                    "false_negative_pairs": 0,
                },
            )

        counts = pairwise_counts(
            left_assignment,
            right_assignment,
            overlap.common_items,
        )
        rand_index_value = rand_index(
            true_positive_pairs=counts.true_positive_pairs,
            true_negative_pairs=counts.true_negative_pairs,
            num_item_pairs_compared=counts.num_item_pairs_compared,
        )
        coassignment_jaccard_value = coassignment_jaccard(
            true_positive_pairs=counts.true_positive_pairs,
            false_positive_pairs=counts.false_positive_pairs,
            false_negative_pairs=counts.false_negative_pairs,
        )
        fowlkes_mallows_value = fowlkes_mallows(
            true_positive_pairs=counts.true_positive_pairs,
            false_positive_pairs=counts.false_positive_pairs,
            false_negative_pairs=counts.false_negative_pairs,
        )
        adjusted_rand_index_value = pairwise_adjusted_rand_index(
            true_positive_pairs=counts.true_positive_pairs,
            left_positive_pairs=counts.left_positive_pairs,
            right_positive_pairs=counts.right_positive_pairs,
            num_item_pairs_compared=counts.num_item_pairs_compared,
        )
        normalized_mutual_info_value = normalized_mutual_info(
            contingency=counts.contingency,
            left_cluster_sizes=counts.left_cluster_sizes,
            right_cluster_sizes=counts.right_cluster_sizes,
            num_common_items=num_common_items,
        )

        return EvaluationResult(
            method=self.name,
            metrics={
                "adjusted_rand_index": adjusted_rand_index_value,
                "normalized_mutual_info": normalized_mutual_info_value,
                "rand_index": rand_index_value,
                "coassignment_jaccard": coassignment_jaccard_value,
                "fowlkes_mallows": fowlkes_mallows_value,
            },
            details={
                "num_common_items": num_common_items,
                "num_union_items": num_union_items,
                "common_item_coverage": safe_ratio(
                    num_common_items,
                    num_union_items,
                    default=0.0,
                ),
                "same_item_set": num_common_items == num_union_items,
                "left_num_clusters_on_common_items": len(counts.left_cluster_sizes),
                "right_num_clusters_on_common_items": len(counts.right_cluster_sizes),
                "cluster_count_abs_diff_on_common_items": abs(
                    len(counts.left_cluster_sizes) - len(counts.right_cluster_sizes)
                ),
                "num_item_pairs_compared": counts.num_item_pairs_compared,
                "true_positive_pairs": counts.true_positive_pairs,
                "true_negative_pairs": counts.true_negative_pairs,
                "false_positive_pairs": counts.false_positive_pairs,
                "false_negative_pairs": counts.false_negative_pairs,
            },
        )
