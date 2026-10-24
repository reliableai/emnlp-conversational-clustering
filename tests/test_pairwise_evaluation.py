"""Unit tests for pairwise evaluator metrics and registry lookup behavior."""

import pytest

from conversational_clustering.evaluation.artifacts import LoadedClustering
from conversational_clustering.evaluation.base import PairwiseEvaluationRequest
from conversational_clustering.evaluation.methods.pairwise_consistency import (
    PairwiseConsistencyEvaluator,
)
from conversational_clustering.evaluation.methods.robustness_expectations import (
    RobustnessExpectationsEvaluator,
)
from conversational_clustering.evaluation.pairwise_registry import (
    get_pairwise_evaluator,
    list_pairwise_evaluators,
)


def test_evaluate_identical_clusterings_returns_perfect_scores() -> None:
    evaluator = PairwiseConsistencyEvaluator.from_params()
    predictions = LoadedClustering(
        assignment={"alpha": 0, "beta": 0, "gamma": 1},
        cluster_names=["A", "B"],
    )
    request = PairwiseEvaluationRequest(
        left_predictions=predictions,
        right_predictions=predictions,
    )

    result = evaluator.evaluate(request)

    assert result.method == "pairwise_consistency"
    for metric_name in (
        "adjusted_rand_index",
        "normalized_mutual_info",
        "rand_index",
        "coassignment_jaccard",
        "fowlkes_mallows",
    ):
        assert result.metrics[metric_name] == pytest.approx(1.0)
    assert result.details["num_common_items"] == 3
    assert result.details["num_union_items"] == 3
    assert result.details["same_item_set"] is True
    assert result.details["false_positive_pairs"] == 0
    assert result.details["false_negative_pairs"] == 0


def test_evaluate_disjoint_item_sets_returns_zero_metrics() -> None:
    evaluator = PairwiseConsistencyEvaluator.from_params()
    request = PairwiseEvaluationRequest(
        left_predictions=LoadedClustering(
            assignment={"alpha": 0, "beta": 0},
            cluster_names=["A"],
        ),
        right_predictions=LoadedClustering(
            assignment={"gamma": 0, "delta": 0},
            cluster_names=["B"],
        ),
    )

    result = evaluator.evaluate(request)

    assert result.metrics == {
        "adjusted_rand_index": 0.0,
        "normalized_mutual_info": 0.0,
        "rand_index": 0.0,
        "coassignment_jaccard": 0.0,
        "fowlkes_mallows": 0.0,
    }
    assert result.details["num_common_items"] == 0
    assert result.details["num_union_items"] == 4
    assert result.details["same_item_set"] is False


def test_from_params_rejects_unsupported_settings() -> None:
    with pytest.raises(ValueError, match="does not support parameters yet"):
        PairwiseConsistencyEvaluator.from_params(params={"foo": "bar"})


def test_registry_lists_and_loads_pairwise_consistency() -> None:
    evaluator_names = list_pairwise_evaluators()
    evaluator = get_pairwise_evaluator("pairwise_consistency")
    robustness_evaluator = get_pairwise_evaluator("robustness_expectations")

    assert "pairwise_consistency" in evaluator_names
    assert "robustness_expectations" in evaluator_names
    assert isinstance(evaluator, PairwiseConsistencyEvaluator)
    assert isinstance(robustness_evaluator, RobustnessExpectationsEvaluator)


def test_registry_rejects_unknown_pairwise_evaluator() -> None:
    with pytest.raises(ValueError, match="Unknown pairwise evaluator"):
        get_pairwise_evaluator("does_not_exist")
