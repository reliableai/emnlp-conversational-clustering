"""Tests for ARI evaluator behavior in dataset-ground-truth mode."""

from __future__ import annotations

from pathlib import Path

import pytest

from conversational_clustering.evaluation.artifacts import LoadedClustering
from conversational_clustering.evaluation.base import EvaluationRequest
from conversational_clustering.evaluation.methods.label_groundtruth_consistency import (
    LabelGroundtruthConsistencyEvaluator,
)


def _predictions() -> LoadedClustering:
    return LoadedClustering(
        assignment={"alpha": 10, "beta": 10, "gamma": 99},
        cluster_names=["cluster_a", "cluster_b"],
    )


def test_ari_similarity_requires_ground_truth_label_column() -> None:
    with pytest.raises(ValueError, match="requires 'ground_truth_label_column'"):
        LabelGroundtruthConsistencyEvaluator.from_params(None)


def test_ari_similarity_supports_dataset_ground_truth_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text(
        "item,label\nalpha,A\nbeta,A\ngamma,B\n",
        encoding="utf-8",
    )

    from conversational_clustering.data_processing import datasets

    monkeypatch.setitem(
        datasets.DATASETS,
        "tmp_eval_dataset",
        {"path": str(dataset_path), "item_column": "item"},
    )

    evaluator = LabelGroundtruthConsistencyEvaluator.from_params(
        {"ground_truth_label_column": "label"}
    )
    request = EvaluationRequest(
        predictions=_predictions(),
        artifacts={
            "batch_run_config": {
                "dataset_name": "tmp_eval_dataset",
                "limit": None,
                "random_seed": 42,
            }
        },
    )

    result = evaluator.evaluate(request)

    assert result.metrics["adjusted_rand_index"] == pytest.approx(1.0)
    assert result.details["reference_source"] == "dataset_label_column"
    assert result.details["dataset_name"] == "tmp_eval_dataset"
    assert result.details["ground_truth_label_column"] == "label"


def test_ari_similarity_ground_truth_mode_requires_batch_run_config() -> None:
    evaluator = LabelGroundtruthConsistencyEvaluator.from_params(
        {"ground_truth_label_column": "label"}
    )

    with pytest.raises(ValueError, match=r"artifacts\['batch_run_config'\]"):
        evaluator.evaluate(EvaluationRequest(predictions=_predictions()))


def test_ari_similarity_rejects_unknown_dataset_name() -> None:
    evaluator = LabelGroundtruthConsistencyEvaluator.from_params(
        {"ground_truth_label_column": "label"}
    )
    request = EvaluationRequest(
        predictions=_predictions(),
        artifacts={
            "batch_run_config": {
                "dataset_name": "does_not_exist",
            }
        },
    )

    with pytest.raises(ValueError, match="Unknown dataset_name"):
        evaluator.evaluate(request)


def test_ari_similarity_rejects_missing_ground_truth_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text(
        "item,not_label\nalpha,A\nbeta,A\ngamma,B\n",
        encoding="utf-8",
    )

    from conversational_clustering.data_processing import datasets

    monkeypatch.setitem(
        datasets.DATASETS,
        "tmp_missing_label_dataset",
        {"path": str(dataset_path), "item_column": "item"},
    )

    evaluator = LabelGroundtruthConsistencyEvaluator.from_params(
        {"ground_truth_label_column": "label"}
    )
    request = EvaluationRequest(
        predictions=_predictions(),
        artifacts={
            "batch_run_config": {
                "dataset_name": "tmp_missing_label_dataset",
            }
        },
    )

    with pytest.raises(ValueError, match="missing ground-truth column"):
        evaluator.evaluate(request)


def test_ari_similarity_rejects_missing_label_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text(
        "item,label\nalpha,A\nbeta,\ngamma,B\n",
        encoding="utf-8",
    )

    from conversational_clustering.data_processing import datasets

    monkeypatch.setitem(
        datasets.DATASETS,
        "tmp_missing_label_values",
        {"path": str(dataset_path), "item_column": "item"},
    )

    evaluator = LabelGroundtruthConsistencyEvaluator.from_params(
        {"ground_truth_label_column": "label"}
    )
    request = EvaluationRequest(
        predictions=_predictions(),
        artifacts={
            "batch_run_config": {
                "dataset_name": "tmp_missing_label_values",
            }
        },
    )

    with pytest.raises(ValueError, match="missing values in label column"):
        evaluator.evaluate(request)


def test_ari_similarity_rejects_missing_item_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text(
        "item,label\nalpha,A\n,B\ngamma,B\n",
        encoding="utf-8",
    )

    from conversational_clustering.data_processing import datasets

    monkeypatch.setitem(
        datasets.DATASETS,
        "tmp_missing_item_values",
        {"path": str(dataset_path), "item_column": "item"},
    )

    evaluator = LabelGroundtruthConsistencyEvaluator.from_params(
        {"ground_truth_label_column": "label"}
    )
    request = EvaluationRequest(
        predictions=_predictions(),
        artifacts={
            "batch_run_config": {
                "dataset_name": "tmp_missing_item_values",
            }
        },
    )

    with pytest.raises(ValueError, match="missing values in item column"):
        evaluator.evaluate(request)


def test_ari_similarity_rejects_inconsistent_duplicate_item_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text(
        "item,label\nalpha,A\nalpha,B\ngamma,B\n",
        encoding="utf-8",
    )

    from conversational_clustering.data_processing import datasets

    monkeypatch.setitem(
        datasets.DATASETS,
        "tmp_inconsistent_labels",
        {"path": str(dataset_path), "item_column": "item"},
    )

    evaluator = LabelGroundtruthConsistencyEvaluator.from_params(
        {"ground_truth_label_column": "label"}
    )
    request = EvaluationRequest(
        predictions=LoadedClustering(
            assignment={"alpha": 0, "gamma": 1},
            cluster_names=["left", "right"],
        ),
        artifacts={
            "batch_run_config": {
                "dataset_name": "tmp_inconsistent_labels",
            }
        },
    )

    with pytest.raises(ValueError, match="inconsistent"):
        evaluator.evaluate(request)


def test_ari_similarity_with_disjoint_items_returns_zero_similarity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text(
        "item,label\ndelta,A\nepsilon,A\nzeta,B\n",
        encoding="utf-8",
    )

    from conversational_clustering.data_processing import datasets

    monkeypatch.setitem(
        datasets.DATASETS,
        "tmp_disjoint_dataset",
        {"path": str(dataset_path), "item_column": "item"},
    )

    evaluator = LabelGroundtruthConsistencyEvaluator.from_params(
        {"ground_truth_label_column": "label"}
    )
    request = EvaluationRequest(
        predictions=LoadedClustering(
            assignment={"other-1": 10, "other-2": 10, "other-3": 99},
            cluster_names=["cluster_a", "cluster_b"],
        ),
        artifacts={
            "batch_run_config": {
                "dataset_name": "tmp_disjoint_dataset",
                "limit": None,
                "random_seed": 42,
            }
        },
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
    assert result.details["same_item_set"] is False
