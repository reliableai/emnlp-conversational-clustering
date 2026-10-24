"""Pairwise experiment config parsing, artifact loading, and request generation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import combinations
from os import PathLike
from pathlib import Path
from typing import Any

from .artifacts import LoadedClustering, load_saved_clustering
from .base import PairwiseEvaluationRequest
from .batch_run_artifacts import (
    iter_batch_run_artifact_contexts,
)
from .benchmark_spec import SyntheticBenchmarkSpec

_EXPERIMENT_IGNORED_CONFIG_KEYS = {"name"}
_VALID_PAIRWISE_SCOPES = {"inter", "intra", "branch_inter", "sweep_inter"}
_VALID_RECORD_VALIDATION_SCOPES = {"inter", "intra"}


@dataclass(frozen=True)
class PairwiseRunRecord:
    config_slug: str
    run_id: str
    batch_run_config: dict[str, Any]
    clustering_path: Path
    conversation_trace_path: Path
    judge_clustering_trace_path: Path | None
    clustering_pov_history_path: Path | None
    topic_candidates_path: Path | None
    selected_topic_path: Path | None
    conversation_trace: dict[str, Any]
    judge_clustering_trace: dict[str, Any] | None
    clustering_pov_history: dict[str, Any] | None
    topic_candidates: dict[str, Any] | None
    selected_topic: dict[str, Any] | None
    benchmark_spec: SyntheticBenchmarkSpec | None
    predictions: LoadedClustering


@dataclass(frozen=True)
class PairwiseExperimentSummary:
    dataset_name: str
    num_configs: int
    vary_values: list[Any]


def _require_dataset_name(*, batch_run_config: Mapping[str, Any], context: str) -> str:
    raw_dataset_name = batch_run_config.get("dataset_name")
    if not isinstance(raw_dataset_name, str) or not raw_dataset_name.strip():
        raise ValueError(f"Missing non-empty 'dataset_name' in {context}")
    return raw_dataset_name.strip()


def parse_pairwise_scope(config_data: Mapping[str, Any]) -> str:
    raw_pairwise_scope = config_data.get("pairwise_scope", "inter")
    if not isinstance(raw_pairwise_scope, str) or not raw_pairwise_scope.strip():
        raise ValueError("'pairwise_scope' must be a non-empty string when provided")
    pairwise_scope = raw_pairwise_scope.strip()
    if pairwise_scope not in _VALID_PAIRWISE_SCOPES:
        valid = ", ".join(sorted(_VALID_PAIRWISE_SCOPES))
        raise ValueError(f"'pairwise_scope' must be one of: {valid}")
    return pairwise_scope


def load_pairwise_inputs(
    input_dir: str | PathLike[str],
    *,
    load_benchmark_specs: bool = True,
) -> list[PairwiseRunRecord]:
    input_dir_path = Path(input_dir)
    shared_configs_by_config_slug: dict[str, dict[str, Any]] = {}

    records: list[PairwiseRunRecord] = []
    for artifact in iter_batch_run_artifact_contexts(
        input_dir_path,
        load_benchmark_specs=load_benchmark_specs,
        require_clustering=True,
    ):
        config_slug = artifact.config_slug
        run_id = artifact.run_id
        loaded_batch_run_config = artifact.batch_run_config

        prior_config = shared_configs_by_config_slug.get(config_slug)
        if prior_config is None:
            shared_configs_by_config_slug[config_slug] = loaded_batch_run_config
        elif prior_config != loaded_batch_run_config:
            raise ValueError(
                "Batch-run config mismatch across artifacts for "
                f"config_slug='{config_slug}' in '{artifact.results_path}'."
            )

        records.append(
            PairwiseRunRecord(
                config_slug=config_slug,
                run_id=run_id,
                batch_run_config=loaded_batch_run_config,
                clustering_path=artifact.clustering_path,
                conversation_trace_path=artifact.conversation_trace_path,
                judge_clustering_trace_path=artifact.judge_clustering_trace_path,
                clustering_pov_history_path=artifact.clustering_pov_history_path,
                topic_candidates_path=artifact.topic_candidates_path,
                selected_topic_path=artifact.selected_topic_path,
                conversation_trace=artifact.conversation_trace,
                judge_clustering_trace=artifact.judge_clustering_trace,
                clustering_pov_history=artifact.clustering_pov_history,
                topic_candidates=artifact.topic_candidates,
                selected_topic=artifact.selected_topic,
                benchmark_spec=artifact.benchmark_spec,
                predictions=load_saved_clustering(artifact.clustering_path),
            )
        )

    return sorted(records, key=lambda record: (record.config_slug, record.clustering_path))


def validate_vary_by_key(*, records: list[PairwiseRunRecord], vary_by: str) -> None:
    missing_key_records = [
        f"config_slug='{record.config_slug}', run_id='{record.run_id}'"
        for record in records
        if vary_by not in record.batch_run_config
    ]
    if not missing_key_records:
        return

    preview = ", ".join(missing_key_records[:3])
    if len(missing_key_records) > 3:
        preview = f"{preview}, ..."
    raise ValueError(
        f"'vary_by' key '{vary_by}' is missing from "
        f"{len(missing_key_records)} batch-run config(s): {preview}"
    )


def validate_intra_experiment_records_without_vary(
    *,
    records: list[PairwiseRunRecord],
) -> PairwiseExperimentSummary:
    if not records:
        raise ValueError("No pairwise records loaded from experiment input_dir")

    config_by_slug: dict[str, dict[str, Any]] = {}
    records_by_config_slug: dict[str, list[PairwiseRunRecord]] = {}
    for record in records:
        config_by_slug.setdefault(record.config_slug, record.batch_run_config)
        records_by_config_slug.setdefault(record.config_slug, []).append(record)

    dataset_names = {
        _require_dataset_name(
            batch_run_config=record.batch_run_config,
            context=f"run_id='{record.run_id}', config_slug='{record.config_slug}'",
        )
        for record in records
    }
    if len(dataset_names) != 1:
        raise ValueError(
            "pairwise experiment input_dir must contain exactly one dataset_name across configs; "
            f"found {sorted(dataset_names)}"
        )
    dataset_name = next(iter(dataset_names))

    has_intra_pairs = any(
        len(records_for_config_slug) >= 2
        for records_for_config_slug in records_by_config_slug.values()
    )
    if not has_intra_pairs:
        raise ValueError(
            "pairwise intra scope requires at least 2 artifacts in a shared config group"
        )

    return PairwiseExperimentSummary(
        dataset_name=dataset_name,
        num_configs=len(config_by_slug),
        vary_values=[],
    )


def _freeze_for_grouping(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            (key, _freeze_for_grouping(nested_value)) for key, nested_value in sorted(value.items())
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_for_grouping(nested_value) for nested_value in value)
    if isinstance(value, set):
        return tuple(
            sorted((_freeze_for_grouping(nested_value) for nested_value in value), key=repr)
        )
    return value


def _config_without_ignored_keys(*, config: Mapping[str, Any], vary_by: str) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key != vary_by and key not in _EXPERIMENT_IGNORED_CONFIG_KEYS
    }


def _group_records_by_fixed_config_and_vary_value(
    *,
    records: list[PairwiseRunRecord],
    vary_by: str,
) -> dict[Any, dict[Any, list[PairwiseRunRecord]]]:
    records_by_fixed_config_and_vary_value: dict[tuple[Any, Any], list[PairwiseRunRecord]] = {}
    for record in records:
        fixed_config_key = _freeze_for_grouping(
            _config_without_ignored_keys(config=record.batch_run_config, vary_by=vary_by)
        )
        vary_value_key = _freeze_for_grouping(record.batch_run_config[vary_by])
        records_by_fixed_config_and_vary_value.setdefault(
            (fixed_config_key, vary_value_key), []
        ).append(record)

    grouped_vary_values_by_fixed_config: dict[Any, dict[Any, list[PairwiseRunRecord]]] = {}
    for (
        fixed_config_key,
        vary_value_key,
    ), grouped_records in records_by_fixed_config_and_vary_value.items():
        grouped_vary_values_by_fixed_config.setdefault(fixed_config_key, {})[vary_value_key] = (
            grouped_records
        )
    return grouped_vary_values_by_fixed_config


def validate_experiment_records(
    *,
    records: list[PairwiseRunRecord],
    vary_by: str,
    pairwise_scope: str,
) -> PairwiseExperimentSummary:
    if not records:
        raise ValueError("No pairwise records loaded from experiment input_dir")
    if pairwise_scope not in _VALID_RECORD_VALIDATION_SCOPES:
        valid = ", ".join(sorted(_VALID_RECORD_VALIDATION_SCOPES))
        raise ValueError(f"Unexpected pairwise_scope='{pairwise_scope}'. Expected one of: {valid}")

    config_by_slug: dict[str, dict[str, Any]] = {}
    for record in records:
        config_by_slug.setdefault(record.config_slug, record.batch_run_config)

    dataset_names = {
        _require_dataset_name(
            batch_run_config=record.batch_run_config,
            context=f"run_id='{record.run_id}', config_slug='{record.config_slug}'",
        )
        for record in records
    }
    if len(dataset_names) != 1:
        raise ValueError(
            "pairwise experiment input_dir must contain exactly one dataset_name across configs; "
            f"found {sorted(dataset_names)}"
        )
    dataset_name = next(iter(dataset_names))

    config_slugs = sorted(config_by_slug)
    vary_values = [config_by_slug[config_slug][vary_by] for config_slug in config_slugs]

    unique_vary_values = sorted(
        {_freeze_for_grouping(value): value for value in vary_values}.values(), key=repr
    )
    if pairwise_scope == "inter":
        if len(config_slugs) < 2:
            raise ValueError("pairwise inter scope requires at least 2 configs")

        baseline_slug = config_slugs[0]
        baseline_config = config_by_slug[baseline_slug]
        baseline_comparable = _config_without_ignored_keys(config=baseline_config, vary_by=vary_by)

        for config_slug in config_slugs:
            config = config_by_slug[config_slug]
            comparable_config = _config_without_ignored_keys(config=config, vary_by=vary_by)
            if comparable_config != baseline_comparable:
                raise ValueError(
                    "pairwise inter scope configs must match for all keys except "
                    f"'{vary_by}' (and ignored keys {sorted(_EXPERIMENT_IGNORED_CONFIG_KEYS)}). "
                    "Mismatch between "
                    f"config_slug='{baseline_slug}' and config_slug='{config_slug}'."
                )

        if len(unique_vary_values) < 2:
            raise ValueError(
                "pairwise inter scope requires at least 2 distinct vary_by values; "
                f"found {len(unique_vary_values)} for vary_by='{vary_by}'"
            )
    elif pairwise_scope == "intra":
        grouped_records = _group_records_by_fixed_config_and_vary_value(
            records=records,
            vary_by=vary_by,
        )
        has_intra_pairs = any(
            len(records_by_vary_value[vary_value_key]) >= 2
            for records_by_vary_value in grouped_records.values()
            for vary_value_key in records_by_vary_value
        )
        if not has_intra_pairs:
            raise ValueError(
                "pairwise intra scope requires at least 2 artifacts in a shared "
                f"config group for vary_by='{vary_by}'"
            )
    return PairwiseExperimentSummary(
        dataset_name=dataset_name,
        num_configs=len(config_slugs),
        vary_values=unique_vary_values,
    )


def _build_pairwise_request(
    *,
    left_record: PairwiseRunRecord,
    right_record: PairwiseRunRecord,
    vary_by: str | None,
    pairwise_scope: str,
) -> PairwiseEvaluationRequest:
    artifacts = {
        "left_config_slug": left_record.config_slug,
        "right_config_slug": right_record.config_slug,
        "left_batch_run_config": left_record.batch_run_config,
        "right_batch_run_config": right_record.batch_run_config,
        "left_clustering_path": str(left_record.clustering_path),
        "right_clustering_path": str(right_record.clustering_path),
        "left_conversation_trace_path": str(left_record.conversation_trace_path),
        "right_conversation_trace_path": str(right_record.conversation_trace_path),
        "left_conversation_trace": left_record.conversation_trace,
        "right_conversation_trace": right_record.conversation_trace,
        "left_judge_clustering_trace_path": (
            str(left_record.judge_clustering_trace_path)
            if left_record.judge_clustering_trace_path is not None
            else None
        ),
        "right_judge_clustering_trace_path": (
            str(right_record.judge_clustering_trace_path)
            if right_record.judge_clustering_trace_path is not None
            else None
        ),
        "left_judge_clustering_trace": left_record.judge_clustering_trace,
        "right_judge_clustering_trace": right_record.judge_clustering_trace,
        "left_clustering_pov_history_path": (
            str(left_record.clustering_pov_history_path)
            if left_record.clustering_pov_history_path is not None
            else None
        ),
        "right_clustering_pov_history_path": (
            str(right_record.clustering_pov_history_path)
            if right_record.clustering_pov_history_path is not None
            else None
        ),
        "left_clustering_pov_history": left_record.clustering_pov_history,
        "right_clustering_pov_history": right_record.clustering_pov_history,
        "left_topic_candidates_path": (
            str(left_record.topic_candidates_path) if left_record.topic_candidates_path else None
        ),
        "right_topic_candidates_path": (
            str(right_record.topic_candidates_path) if right_record.topic_candidates_path else None
        ),
        "left_selected_topic_path": (
            str(left_record.selected_topic_path) if left_record.selected_topic_path else None
        ),
        "right_selected_topic_path": (
            str(right_record.selected_topic_path) if right_record.selected_topic_path else None
        ),
        "left_topic_candidates": left_record.topic_candidates,
        "right_topic_candidates": right_record.topic_candidates,
        "left_selected_topic": left_record.selected_topic,
        "right_selected_topic": right_record.selected_topic,
        "left_benchmark_spec": left_record.benchmark_spec,
        "right_benchmark_spec": right_record.benchmark_spec,
        "benchmark_spec": left_record.benchmark_spec or right_record.benchmark_spec,
        "left_run_id": left_record.run_id,
        "right_run_id": right_record.run_id,
    }
    metadata: dict[str, Any] = {
        "pairwise_scope": pairwise_scope,
    }
    if vary_by is not None:
        artifacts["left_vary_value"] = left_record.batch_run_config[vary_by]
        artifacts["right_vary_value"] = right_record.batch_run_config[vary_by]
        metadata["vary_by"] = vary_by

    return PairwiseEvaluationRequest(
        left_predictions=left_record.predictions,
        right_predictions=right_record.predictions,
        artifacts=artifacts,
        metadata=metadata,
    )


def iter_pairwise_requests(
    *,
    records: list[PairwiseRunRecord],
    vary_by: str,
    pairwise_scope: str,
) -> Iterator[PairwiseEvaluationRequest]:
    if pairwise_scope not in _VALID_RECORD_VALIDATION_SCOPES:
        valid = ", ".join(sorted(_VALID_RECORD_VALIDATION_SCOPES))
        raise ValueError(f"Unexpected pairwise_scope='{pairwise_scope}'. Expected one of: {valid}")

    grouped_vary_values_by_fixed_config = _group_records_by_fixed_config_and_vary_value(
        records=records,
        vary_by=vary_by,
    )
    for fixed_config_key in sorted(grouped_vary_values_by_fixed_config, key=repr):
        records_by_vary_value = grouped_vary_values_by_fixed_config[fixed_config_key]
        vary_value_keys = sorted(records_by_vary_value, key=repr)
        if pairwise_scope == "inter":
            for left_vary_value_key, right_vary_value_key in combinations(vary_value_keys, 2):
                for left_record in records_by_vary_value[left_vary_value_key]:
                    for right_record in records_by_vary_value[right_vary_value_key]:
                        yield _build_pairwise_request(
                            left_record=left_record,
                            right_record=right_record,
                            vary_by=vary_by,
                            pairwise_scope=pairwise_scope,
                        )
        elif pairwise_scope == "intra":
            for vary_value_key in vary_value_keys:
                for left_record, right_record in combinations(
                    records_by_vary_value[vary_value_key],
                    2,
                ):
                    yield _build_pairwise_request(
                        left_record=left_record,
                        right_record=right_record,
                        vary_by=vary_by,
                        pairwise_scope=pairwise_scope,
                    )


def iter_pairwise_intra_requests_without_vary(
    *,
    records: list[PairwiseRunRecord],
) -> Iterator[PairwiseEvaluationRequest]:
    records_by_config_slug: dict[str, list[PairwiseRunRecord]] = {}
    for record in records:
        records_by_config_slug.setdefault(record.config_slug, []).append(record)

    for config_slug in sorted(records_by_config_slug):
        records_for_config_slug = records_by_config_slug[config_slug]
        for left_record, right_record in combinations(records_for_config_slug, 2):
            yield _build_pairwise_request(
                left_record=left_record,
                right_record=right_record,
                vary_by=None,
                pairwise_scope="intra",
            )
