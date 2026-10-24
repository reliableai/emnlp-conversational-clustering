"""Evaluator that measures clustering consistency against a configured label-based reference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from conversational_clustering.data_processing import DATASETS
from conversational_clustering.data_processing.csv_loader import CsvDataLoader
from conversational_clustering.evaluation.base import EvaluationRequest, EvaluationResult, Evaluator
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


@dataclass(frozen=True)
class LabelGroundtruthConsistencyEvaluator(Evaluator):
    name = "label_groundtruth_consistency"
    resume_version = "v1"
    ground_truth_label_column: str

    @classmethod
    def from_params(
        cls,
        params: Mapping[str, Any] | None = None,
    ) -> LabelGroundtruthConsistencyEvaluator:
        normalized_params = dict(params or {})
        supported_keys = {
            "ground_truth_label_column",
        }

        unsupported_keys = sorted(set(normalized_params) - supported_keys)
        if unsupported_keys:
            unsupported = ", ".join(unsupported_keys)
            raise ValueError(
                f"Evaluator '{cls.name}' received unsupported parameter(s): {unsupported}"
            )

        raw_ground_truth_label_column = normalized_params.get("ground_truth_label_column")
        if (
            raw_ground_truth_label_column is None
            or (
                isinstance(raw_ground_truth_label_column, str)
                and not raw_ground_truth_label_column.strip()
            )
            or not isinstance(raw_ground_truth_label_column, str)
        ):
            raise ValueError(
                f"Evaluator '{cls.name}' requires 'ground_truth_label_column' "
                "to be a non-empty string."
            )
        return cls(ground_truth_label_column=raw_ground_truth_label_column.strip())

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        batch_run_config = request.artifacts.get("batch_run_config")
        if not isinstance(batch_run_config, dict):
            raise ValueError(
                f"Evaluator '{self.name}' with 'ground_truth_label_column' requires "
                "artifacts['batch_run_config']."
            )
        dataset_name = batch_run_config.get("dataset_name")
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise ValueError(
                f"Evaluator '{self.name}' with 'ground_truth_label_column' requires "
                "batch_run_config['dataset_name']."
            )

        (
            reference_assignment,
            reference_num_clusters,
            reference_name_by_cluster_id,
        ) = _load_ground_truth_assignment(
            dataset_name=dataset_name,
            label_column=self.ground_truth_label_column,
            limit=batch_run_config.get("limit"),
            random_seed=batch_run_config.get("random_seed"),
        )
        reference_details: dict[str, Any] = {
            "reference_source": "dataset_label_column",
            "dataset_name": dataset_name,
            "ground_truth_label_column": self.ground_truth_label_column,
            "num_reference_clusters": reference_num_clusters,
        }

        overlap = overlap_stats(request.predictions.assignment, reference_assignment)
        if overlap.num_common_items == 0:
            metrics = {
                "adjusted_rand_index": 0.0,
                "normalized_mutual_info": 0.0,
                "rand_index": 0.0,
                "coassignment_jaccard": 0.0,
                "fowlkes_mallows": 0.0,
            }
            comparison_details = {
                "num_common_items": 0,
                "num_union_items": overlap.num_union_items,
                "common_item_coverage": 0.0,
                "same_item_set": False,
                "left_num_clusters_on_common_items": 0,
                "right_num_clusters_on_common_items": 0,
                "cluster_count_abs_diff_on_common_items": 0,
                "num_item_pairs_compared": 0,
                "true_positive_pairs": 0,
                "true_negative_pairs": 0,
                "false_positive_pairs": 0,
                "false_negative_pairs": 0,
            }
        else:
            counts = pairwise_counts(
                request.predictions.assignment,
                reference_assignment,
                overlap.common_items,
            )
            metrics = {
                "adjusted_rand_index": pairwise_adjusted_rand_index(
                    true_positive_pairs=counts.true_positive_pairs,
                    left_positive_pairs=counts.left_positive_pairs,
                    right_positive_pairs=counts.right_positive_pairs,
                    num_item_pairs_compared=counts.num_item_pairs_compared,
                ),
                "normalized_mutual_info": normalized_mutual_info(
                    contingency=counts.contingency,
                    left_cluster_sizes=counts.left_cluster_sizes,
                    right_cluster_sizes=counts.right_cluster_sizes,
                    num_common_items=overlap.num_common_items,
                ),
                "rand_index": rand_index(
                    true_positive_pairs=counts.true_positive_pairs,
                    true_negative_pairs=counts.true_negative_pairs,
                    num_item_pairs_compared=counts.num_item_pairs_compared,
                ),
                "coassignment_jaccard": coassignment_jaccard(
                    true_positive_pairs=counts.true_positive_pairs,
                    false_positive_pairs=counts.false_positive_pairs,
                    false_negative_pairs=counts.false_negative_pairs,
                ),
                "fowlkes_mallows": fowlkes_mallows(
                    true_positive_pairs=counts.true_positive_pairs,
                    false_positive_pairs=counts.false_positive_pairs,
                    false_negative_pairs=counts.false_negative_pairs,
                ),
            }
            comparison_details = {
                "num_common_items": overlap.num_common_items,
                "num_union_items": overlap.num_union_items,
                "common_item_coverage": safe_ratio(
                    overlap.num_common_items,
                    overlap.num_union_items,
                    default=0.0,
                ),
                "same_item_set": overlap.num_common_items == overlap.num_union_items,
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
            }

        return EvaluationResult(
            method=self.name,
            metrics=metrics,
            details={**reference_details, **comparison_details},
        )


def _load_ground_truth_assignment(
    *,
    dataset_name: str,
    label_column: str,
    limit: Any,
    random_seed: Any,
) -> tuple[dict[str, int], int, dict[int, str]]:
    dataset_config = DATASETS.get(dataset_name)
    if dataset_config is None:
        available = ", ".join(sorted(DATASETS))
        raise ValueError(
            f"Unknown dataset_name in batch_run_config: '{dataset_name}' (available: {available})"
        )

    dataset_path = dataset_config.get("path")
    item_column = dataset_config.get("item_column")
    if not isinstance(dataset_path, str) or not dataset_path.strip():
        raise ValueError(f"Dataset '{dataset_name}' is missing a valid 'path'.")
    if not isinstance(item_column, str) or not item_column.strip():
        raise ValueError(f"Dataset '{dataset_name}' is missing a valid 'item_column'.")

    parsed_limit: int | None
    if limit is None:
        parsed_limit = None
    elif isinstance(limit, int) and limit > 0:
        parsed_limit = limit
    else:
        raise ValueError(
            "batch_run_config['limit'] must be an integer >= 1 when provided for "
            "ground-truth ARI evaluation."
        )

    parsed_random_seed = 42
    if random_seed is not None:
        if not isinstance(random_seed, int) or random_seed < 0:
            raise ValueError(
                "batch_run_config['random_seed'] must be an integer >= 0 when provided for "
                "ground-truth ARI evaluation."
            )
        parsed_random_seed = random_seed

    dataframe = CsvDataLoader(
        dataset_path,
        item_column=item_column,
        limit=parsed_limit,
        random_seed=parsed_random_seed,
    ).load()

    if item_column not in dataframe.columns:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing configured item column '{item_column}'."
        )
    if label_column not in dataframe.columns:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing ground-truth column '{label_column}'."
        )

    label_to_cluster_id: dict[str, int] = {}
    assignment: dict[str, int] = {}
    for _, row in dataframe.iterrows():
        raw_item_value = row[item_column]
        raw_label_value = row[label_column]
        if pd.isna(raw_item_value):
            raise ValueError(
                f"Dataset '{dataset_name}' contains missing values in item column '{item_column}'."
            )
        if pd.isna(raw_label_value):
            raise ValueError(
                "Dataset "
                f"'{dataset_name}' contains missing values in label column '{label_column}'."
            )
        item_value = str(raw_item_value).strip()
        label_value = str(raw_label_value).strip()
        if not item_value:
            raise ValueError(
                f"Dataset '{dataset_name}' contains blank values in item column '{item_column}'."
            )
        if not label_value:
            raise ValueError(
                f"Dataset '{dataset_name}' contains blank values in label column '{label_column}'."
            )

        if label_value not in label_to_cluster_id:
            label_to_cluster_id[label_value] = len(label_to_cluster_id)
        cluster_id = label_to_cluster_id[label_value]

        existing_cluster_id = assignment.get(item_value)
        if existing_cluster_id is not None and existing_cluster_id != cluster_id:
            raise ValueError(
                "Ground-truth labels are inconsistent for duplicated item "
                f"'{item_value}' in dataset '{dataset_name}'."
            )
        assignment[item_value] = cluster_id

    reference_name_by_cluster_id = {
        cluster_id: label_name for label_name, cluster_id in label_to_cluster_id.items()
    }
    return assignment, len(label_to_cluster_id), reference_name_by_cluster_id
