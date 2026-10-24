"""Shared pairwise clustering metric computations used by evaluation methods."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log, sqrt


@dataclass(frozen=True)
class PairwiseOverlap:
    common_items: set[str]
    num_common_items: int
    num_union_items: int


@dataclass(frozen=True)
class PairwiseCounts:
    contingency: Counter[tuple[int, int]]
    left_cluster_sizes: Counter[int]
    right_cluster_sizes: Counter[int]
    true_positive_pairs: int
    true_negative_pairs: int
    false_positive_pairs: int
    false_negative_pairs: int
    left_positive_pairs: int
    right_positive_pairs: int
    num_item_pairs_compared: int


def overlap_stats(
    left_assignment: dict[str, int],
    right_assignment: dict[str, int],
) -> PairwiseOverlap:
    left_items = set(left_assignment)
    right_items = set(right_assignment)
    common_items = left_items & right_items
    union_items = left_items | right_items
    return PairwiseOverlap(
        common_items=common_items,
        num_common_items=len(common_items),
        num_union_items=len(union_items),
    )


def pairwise_counts(
    left_assignment: dict[str, int],
    right_assignment: dict[str, int],
    common_items: set[str],
) -> PairwiseCounts:
    contingency: Counter[tuple[int, int]] = Counter()
    left_cluster_sizes: Counter[int] = Counter()
    right_cluster_sizes: Counter[int] = Counter()
    for item in common_items:
        left_cluster_id = left_assignment[item]
        right_cluster_id = right_assignment[item]
        contingency[(left_cluster_id, right_cluster_id)] += 1
        left_cluster_sizes[left_cluster_id] += 1
        right_cluster_sizes[right_cluster_id] += 1

    true_positive_pairs = sum(_comb2(cluster_size) for cluster_size in contingency.values())
    left_positive_pairs = sum(_comb2(cluster_size) for cluster_size in left_cluster_sizes.values())
    right_positive_pairs = sum(
        _comb2(cluster_size) for cluster_size in right_cluster_sizes.values()
    )
    num_item_pairs_compared = _comb2(len(common_items))
    false_positive_pairs = left_positive_pairs - true_positive_pairs
    false_negative_pairs = right_positive_pairs - true_positive_pairs
    true_negative_pairs = (
        num_item_pairs_compared - true_positive_pairs - false_positive_pairs - false_negative_pairs
    )

    return PairwiseCounts(
        contingency=contingency,
        left_cluster_sizes=left_cluster_sizes,
        right_cluster_sizes=right_cluster_sizes,
        true_positive_pairs=true_positive_pairs,
        true_negative_pairs=true_negative_pairs,
        false_positive_pairs=false_positive_pairs,
        false_negative_pairs=false_negative_pairs,
        left_positive_pairs=left_positive_pairs,
        right_positive_pairs=right_positive_pairs,
        num_item_pairs_compared=num_item_pairs_compared,
    )


def rand_index(
    *,
    true_positive_pairs: int,
    true_negative_pairs: int,
    num_item_pairs_compared: int,
) -> float:
    return safe_ratio(
        true_positive_pairs + true_negative_pairs,
        num_item_pairs_compared,
        default=1.0,
    )


def coassignment_jaccard(
    *,
    true_positive_pairs: int,
    false_positive_pairs: int,
    false_negative_pairs: int,
) -> float:
    return safe_ratio(
        true_positive_pairs,
        true_positive_pairs + false_positive_pairs + false_negative_pairs,
        default=1.0,
    )


def fowlkes_mallows(
    *,
    true_positive_pairs: int,
    false_positive_pairs: int,
    false_negative_pairs: int,
) -> float:
    precision_denominator = true_positive_pairs + false_positive_pairs
    recall_denominator = true_positive_pairs + false_negative_pairs
    if precision_denominator == 0 and recall_denominator == 0:
        return 1.0
    if precision_denominator == 0 or recall_denominator == 0:
        return 0.0
    precision = true_positive_pairs / precision_denominator
    recall = true_positive_pairs / recall_denominator
    return sqrt(precision * recall)


def adjusted_rand_index(
    *,
    true_positive_pairs: int,
    left_positive_pairs: int,
    right_positive_pairs: int,
    num_item_pairs_compared: int,
) -> float:
    if num_item_pairs_compared == 0:
        return 1.0

    expected_index = left_positive_pairs * right_positive_pairs / num_item_pairs_compared
    max_index = 0.5 * (left_positive_pairs + right_positive_pairs)
    denominator = max_index - expected_index
    if denominator == 0:
        return 1.0 if true_positive_pairs == expected_index else 0.0
    raw = (true_positive_pairs - expected_index) / denominator
    return min(1.0, max(-1.0, raw))


def normalized_mutual_info(
    *,
    contingency: Counter[tuple[int, int]],
    left_cluster_sizes: Counter[int],
    right_cluster_sizes: Counter[int],
    num_common_items: int,
) -> float:
    if num_common_items == 0:
        return 0.0

    left_entropy = _entropy(left_cluster_sizes, num_items=num_common_items)
    right_entropy = _entropy(right_cluster_sizes, num_items=num_common_items)
    if left_entropy == 0 and right_entropy == 0:
        return 1.0
    if left_entropy == 0 or right_entropy == 0:
        return 0.0

    mutual_information = 0.0
    for (left_cluster_id, right_cluster_id), joint_count in contingency.items():
        if joint_count == 0:
            continue
        joint_probability = joint_count / num_common_items
        left_probability = left_cluster_sizes[left_cluster_id] / num_common_items
        right_probability = right_cluster_sizes[right_cluster_id] / num_common_items
        mutual_information += joint_probability * log(
            joint_probability / (left_probability * right_probability)
        )

    return mutual_information / sqrt(left_entropy * right_entropy)


def variation_of_information(
    *,
    contingency: Counter[tuple[int, int]],
    left_cluster_sizes: Counter[int],
    right_cluster_sizes: Counter[int],
    num_common_items: int,
) -> float:
    if num_common_items == 0:
        return 0.0

    left_entropy = _entropy(left_cluster_sizes, num_items=num_common_items)
    right_entropy = _entropy(right_cluster_sizes, num_items=num_common_items)

    mutual_information = 0.0
    for (left_cluster_id, right_cluster_id), joint_count in contingency.items():
        if joint_count == 0:
            continue
        joint_probability = joint_count / num_common_items
        left_probability = left_cluster_sizes[left_cluster_id] / num_common_items
        right_probability = right_cluster_sizes[right_cluster_id] / num_common_items
        mutual_information += joint_probability * log(
            joint_probability / (left_probability * right_probability)
        )

    return max(0.0, left_entropy + right_entropy - 2.0 * mutual_information)


def _comb2(value: int) -> int:
    return value * (value - 1) // 2


def safe_ratio(numerator: float, denominator: float, *, default: float) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def _entropy(cluster_sizes: Counter[int], *, num_items: int) -> float:
    entropy_value = 0.0
    for cluster_size in cluster_sizes.values():
        if cluster_size == 0:
            continue
        probability = cluster_size / num_items
        entropy_value -= probability * log(probability)
    return entropy_value
