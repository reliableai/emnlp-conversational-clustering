"""Unit tests for shared pairwise clustering metrics in the metrics module."""

from __future__ import annotations

import pytest

from conversational_clustering.evaluation.methods.metrics.pairwise import (
    adjusted_rand_index as pairwise_adjusted_rand_index,
)
from conversational_clustering.evaluation.methods.metrics.pairwise import (
    coassignment_jaccard,
    fowlkes_mallows,
    normalized_mutual_info,
    overlap_stats,
    pairwise_counts,
    rand_index,
)


def test_pairwise_metrics_perfect_for_identical_assignments() -> None:
    left_assignment = {"alpha": 0, "beta": 0, "gamma": 1}
    right_assignment = {"alpha": 0, "beta": 0, "gamma": 1}
    overlap = overlap_stats(left_assignment, right_assignment)

    counts = pairwise_counts(left_assignment, right_assignment, overlap.common_items)

    assert pairwise_adjusted_rand_index(
        true_positive_pairs=counts.true_positive_pairs,
        left_positive_pairs=counts.left_positive_pairs,
        right_positive_pairs=counts.right_positive_pairs,
        num_item_pairs_compared=counts.num_item_pairs_compared,
    ) == pytest.approx(1.0)
    assert normalized_mutual_info(
        contingency=counts.contingency,
        left_cluster_sizes=counts.left_cluster_sizes,
        right_cluster_sizes=counts.right_cluster_sizes,
        num_common_items=overlap.num_common_items,
    ) == pytest.approx(1.0)
    assert rand_index(
        true_positive_pairs=counts.true_positive_pairs,
        true_negative_pairs=counts.true_negative_pairs,
        num_item_pairs_compared=counts.num_item_pairs_compared,
    ) == pytest.approx(1.0)
    assert coassignment_jaccard(
        true_positive_pairs=counts.true_positive_pairs,
        false_positive_pairs=counts.false_positive_pairs,
        false_negative_pairs=counts.false_negative_pairs,
    ) == pytest.approx(1.0)
    assert fowlkes_mallows(
        true_positive_pairs=counts.true_positive_pairs,
        false_positive_pairs=counts.false_positive_pairs,
        false_negative_pairs=counts.false_negative_pairs,
    ) == pytest.approx(1.0)


def test_overlap_stats_returns_common_and_union_counts() -> None:
    left_assignment = {"alpha": 0, "beta": 0}
    right_assignment = {"beta": 1, "gamma": 1}

    overlap = overlap_stats(left_assignment, right_assignment)

    assert overlap.common_items == {"beta"}
    assert overlap.num_common_items == 1
    assert overlap.num_union_items == 3
