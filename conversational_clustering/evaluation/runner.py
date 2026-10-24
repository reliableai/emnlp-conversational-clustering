"""TOML-driven multi-evaluator runner for single and pairwise clustering evaluation."""

from __future__ import annotations

import hashlib
import json
import logging
import tomllib
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass, is_dataclass
from itertools import combinations
from os import PathLike
from pathlib import Path
from time import perf_counter
from typing import Any, TextIO
from uuid import uuid4

from .artifacts import LoadedClustering, load_saved_clustering
from .base import (
    EvaluationRequest,
    EvaluationResult,
    EvaluationSummaryAccumulator,
    Evaluator,
    PairwiseEvaluationRequest,
    PairwiseEvaluator,
)
from .batch_run_artifacts import (
    iter_batch_run_artifact_contexts,
    load_configs_by_run_id,
)
from .pairwise import (
    PairwiseRunRecord,
    iter_pairwise_intra_requests_without_vary,
    iter_pairwise_requests,
    load_pairwise_inputs,
    parse_pairwise_scope,
    validate_experiment_records,
    validate_intra_experiment_records_without_vary,
    validate_vary_by_key,
)
from .pairwise_registry import get_pairwise_evaluator, list_pairwise_evaluators
from .registry import get_evaluator, list_evaluators

_INFERRED_VARY_BY_IGNORED_KEYS = {"name", "dataset_name"}
_SINGLE_EVALUATORS_REQUIRING_BENCHMARK_SPEC = {
    "clarification_policy",
    "clustering_pov_semantic_quality",
    "controllability",
    "discovery_semantic_quality",
    "efficiency_utility",
    "inferred_theme_efficiency_utility",
    "topic_conditioned_quality",
}
_PAIRWISE_EVALUATORS_REQUIRING_BENCHMARK_SPEC = {"robustness_expectations"}
_SHARED_RUN_SINGLE_EVALUATORS = {"discovery_semantic_quality"}
_SINGLE_EVALUATORS_WITHOUT_CLUSTERING_REQUIREMENT = {
    "clustering_pov_semantic_quality",
    "conversation_trace_metrics",
    "discovery_semantic_quality",
    "human_effort_metrics",
    "null_theme_abstention",
}
logger = logging.getLogger("conversational_clustering")


def iter_batch_run_requests(
    input_dir: str | PathLike[str],
    *,
    load_benchmark_specs: bool = True,
    coalesce_auto_topic_shared_runs: bool = False,
    require_clustering: bool = True,
) -> Iterator[EvaluationRequest]:
    seen_shared_request_keys: set[str] = set()
    for artifact in iter_batch_run_artifact_contexts(
        input_dir,
        load_benchmark_specs=load_benchmark_specs,
        require_clustering=require_clustering,
    ):
        summary_config_slug = artifact.config_slug
        shared_request_key = None
        if coalesce_auto_topic_shared_runs:
            shared_request_identity = _resolve_shared_topic_candidates_request_identity(
                artifact=artifact
            )
            if shared_request_identity is not None:
                shared_request_key, summary_config_slug, _ = shared_request_identity
                if shared_request_key in seen_shared_request_keys:
                    continue
                seen_shared_request_keys.add(shared_request_key)
        loaded_clustering = (
            load_saved_clustering(artifact.clustering_path)
            if artifact.clustering_path.is_file()
            else LoadedClustering(assignment={}, cluster_names=[])
        )
        clustering_path = (
            str(artifact.clustering_path) if artifact.clustering_path.is_file() else None
        )

        yield EvaluationRequest(
            predictions=loaded_clustering,
            artifacts={
                "clustering_path": clustering_path,
                "conversation_trace_path": str(artifact.conversation_trace_path),
                "conversation_trace": artifact.conversation_trace,
                "judge_clustering_trace_path": (
                    str(artifact.judge_clustering_trace_path)
                    if artifact.judge_clustering_trace_path is not None
                    else None
                ),
                "judge_clustering_trace": artifact.judge_clustering_trace,
                "clustering_pov_history_path": (
                    str(artifact.clustering_pov_history_path)
                    if artifact.clustering_pov_history_path is not None
                    else None
                ),
                "clustering_pov_history": artifact.clustering_pov_history,
                "human_thought_trace_path": (
                    str(artifact.human_thought_trace_path)
                    if artifact.human_thought_trace_path is not None
                    else None
                ),
                "human_thought_trace": artifact.human_thought_trace,
                "topic_candidates_path": (
                    str(artifact.topic_candidates_path)
                    if artifact.topic_candidates_path is not None
                    else None
                ),
                "topic_candidates": artifact.topic_candidates,
                "selected_topic_path": (
                    str(artifact.selected_topic_path)
                    if artifact.selected_topic_path is not None
                    else None
                ),
                "selected_topic": artifact.selected_topic,
                "benchmark_spec": artifact.benchmark_spec,
                "batch_run_config_path": str(artifact.results_path),
                "batch_run_config": artifact.batch_run_config,
                "summary_config_slug": summary_config_slug,
                "batch_run_result": artifact.batch_run_result,
                "turns_run": artifact.turns_run,
            },
            metadata={
                "input_dir": str(artifact.input_dir),
                "config_slug": artifact.config_slug,
                "run_id": artifact.run_id,
            },
        )


def _resolve_shared_topic_candidates_request_identity(
    *,
    artifact: Any,
) -> tuple[str, str, str] | None:
    topic_candidates_path = artifact.topic_candidates_path
    if topic_candidates_path is None or "_shared" not in topic_candidates_path.parts:
        return None
    batch_run_config = artifact.batch_run_config
    if not isinstance(batch_run_config, Mapping):
        return None
    metadata = batch_run_config.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    raw_source_config_slug = metadata.get("topic_branch_source_config_slug")
    if not isinstance(raw_source_config_slug, str) or not raw_source_config_slug.strip():
        return None
    source_config_slug = raw_source_config_slug.strip()
    raw_shared_run_id = metadata.get("topic_branch_source_run_id")
    if isinstance(raw_shared_run_id, str) and raw_shared_run_id.strip():
        shared_run_id = raw_shared_run_id.strip()
    else:
        shared_run_id = topic_candidates_path.parent.name
    if not shared_run_id:
        return None
    return (
        str(topic_candidates_path.resolve()),
        source_config_slug,
        shared_run_id,
    )


@dataclass(frozen=True)
class EvaluationRunConfig:
    evaluation_mode: str
    input_dir: str
    evaluators: list[str]
    evaluator_params: dict[str, dict[str, Any]]
    pairwise_scope: str | None
    report_path: str | None
    resume: bool


@dataclass(frozen=True)
class SingleEvaluationRollup:
    num_artifacts_by_evaluator: dict[str, int]
    num_artifacts_by_config_by_evaluator: dict[str, dict[str, int]]
    accumulators_by_config_by_evaluator: dict[
        str, dict[str, dict[str, EvaluationSummaryAccumulator]]
    ]
    num_failures_by_evaluator: dict[str, int]
    num_failures_by_config_by_evaluator: dict[str, dict[str, dict[str, int]]]
    vary_by_by_evaluator: dict[str, str | None]
    shared_run_request_mode_by_evaluator: dict[str, bool]


@dataclass(frozen=True)
class PairwiseEvaluationRollup:
    num_pairwise_requests: int
    num_pairwise_requests_by_vary_value_pair: dict[str, int]
    accumulators_by_vary_value_pair: dict[str, dict[str, EvaluationSummaryAccumulator]]
    num_pairwise_requests_by_intra_vary_value: dict[str, int]
    intra_accumulators_by_vary_value: dict[str, dict[str, EvaluationSummaryAccumulator]]
    num_failures_by_evaluator: dict[str, int]
    num_failures_by_vary_value_pair: dict[str, dict[str, int]]
    num_failures_by_intra_vary_value: dict[str, dict[str, int]]


@dataclass(frozen=True)
class PairwiseIntraNoVaryEvaluationRollup:
    num_pairwise_requests: int
    num_pairwise_requests_by_config_slug: dict[str, int]
    accumulators_by_config_slug: dict[str, dict[str, EvaluationSummaryAccumulator]]
    num_failures_by_evaluator: dict[str, int]
    num_failures_by_config_slug: dict[str, dict[str, int]]


@dataclass(frozen=True)
class BranchScopePairwisePlan:
    dataset_name: str
    num_configs: int
    vary_by: str
    vary_values: list[Any]
    requests: Iterator[PairwiseEvaluationRequest]


@dataclass(frozen=True)
class ExistingResultRecord:
    key: tuple[str, ...]
    status: str
    result: EvaluationResult | None
    raw_record: dict[str, Any]


class EvaluationRunFailed(RuntimeError):
    def __init__(self, *, report_path: Path, total_failures: int) -> None:
        self.report_path = report_path
        self.total_failures = total_failures
        super().__init__(
            f"Evaluation completed with {total_failures} failure(s). Report: {report_path}"
        )


def _normalize_resume_fingerprint_value(value: Any) -> Any:
    if isinstance(value, LoadedClustering):
        return {
            "assignment": {
                item_text: value.assignment[item_text] for item_text in sorted(value.assignment)
            },
            "cluster_names": list(value.cluster_names),
        }
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _normalize_resume_fingerprint_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_resume_fingerprint_value(nested_value)
            for key, nested_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list | tuple):
        return [_normalize_resume_fingerprint_value(nested_value) for nested_value in value]
    if isinstance(value, set):
        normalized_values = [
            _normalize_resume_fingerprint_value(nested_value) for nested_value in value
        ]
        return sorted(
            normalized_values,
            key=lambda normalized_value: json.dumps(
                normalized_value, sort_keys=True, separators=(",", ":")
            ),
        )
    return value


def _build_resume_fingerprint(*, payload: Mapping[str, Any]) -> str:
    normalized_payload = _normalize_resume_fingerprint_value(dict(payload))
    serialized_payload = json.dumps(
        normalized_payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()


def _resolve_evaluator_resume_version(*, evaluator: Evaluator | PairwiseEvaluator) -> str:
    raw_resume_version = getattr(evaluator, "resume_version", None)
    if not isinstance(raw_resume_version, str) or not raw_resume_version.strip():
        evaluator_name = getattr(evaluator, "name", type(evaluator).__name__)
        raise ValueError(
            f"Evaluator '{evaluator_name}' must define a non-empty string resume_version"
        )
    return raw_resume_version.strip()


def _build_single_resume_fingerprint(
    *,
    request: EvaluationRequest,
    evaluator: Evaluator,
    evaluator_name: str,
    evaluator_params: Mapping[str, Any] | None,
) -> str:
    return _build_resume_fingerprint(
        payload={
            "evaluation_mode": "single",
            "evaluator_name": evaluator_name,
            "evaluator_resume_version": _resolve_evaluator_resume_version(evaluator=evaluator),
            "evaluator_params": dict(evaluator_params or {}),
            "metadata": request.metadata,
            "predictions": request.predictions,
            "artifacts": request.artifacts,
        }
    )


def _build_pairwise_resume_fingerprint(
    *,
    request: PairwiseEvaluationRequest,
    evaluator: PairwiseEvaluator,
    evaluator_name: str,
    evaluator_params: Mapping[str, Any] | None,
) -> str:
    return _build_resume_fingerprint(
        payload={
            "evaluation_mode": "pairwise",
            "evaluator_name": evaluator_name,
            "evaluator_resume_version": _resolve_evaluator_resume_version(evaluator=evaluator),
            "evaluator_params": dict(evaluator_params or {}),
            "metadata": request.metadata,
            "left_predictions": request.left_predictions,
            "right_predictions": request.right_predictions,
            "artifacts": request.artifacts,
        }
    )


def _result_record_matches_resume_fingerprint(
    *,
    record: ExistingResultRecord,
    expected_resume_fingerprint: str,
) -> bool:
    return record.raw_record.get("resume_fingerprint") == expected_resume_fingerprint


def _evaluation_requires_benchmark_specs(config: EvaluationRunConfig) -> bool:
    required_evaluators = (
        _SINGLE_EVALUATORS_REQUIRING_BENCHMARK_SPEC
        if config.evaluation_mode == "single"
        else _PAIRWISE_EVALUATORS_REQUIRING_BENCHMARK_SPEC
    )
    return any(evaluator_name in required_evaluators for evaluator_name in config.evaluators)


def _single_evaluator_uses_shared_run_requests(*, evaluator_name: str) -> bool:
    return evaluator_name in _SHARED_RUN_SINGLE_EVALUATORS


def _single_evaluator_requires_clustering(*, evaluator_name: str) -> bool:
    return evaluator_name not in _SINGLE_EVALUATORS_WITHOUT_CLUSTERING_REQUIREMENT


def _parse_input_dir(config_data: dict[str, Any]) -> str:
    raw_input_dir = config_data.get("input_dir")
    if not isinstance(raw_input_dir, str) or not raw_input_dir.strip():
        raise ValueError("evaluation config must define a non-empty 'input_dir' string")
    return raw_input_dir.strip()


def _parse_evaluators(config_data: dict[str, Any]) -> list[str]:
    raw_evaluators = config_data.get("evaluators")
    if not isinstance(raw_evaluators, list) or not raw_evaluators:
        raise ValueError("evaluation config must define a non-empty 'evaluators' list")

    evaluators: list[str] = []
    seen_evaluators: set[str] = set()
    for index, evaluator_name in enumerate(raw_evaluators, start=1):
        if not isinstance(evaluator_name, str) or not evaluator_name.strip():
            raise ValueError(
                f"'evaluators' entry #{index} must be a non-empty string evaluator name"
            )
        normalized_evaluator_name = evaluator_name.strip()
        if normalized_evaluator_name in seen_evaluators:
            raise ValueError(
                f"'evaluators' contains duplicate evaluator name: {normalized_evaluator_name}"
            )
        seen_evaluators.add(normalized_evaluator_name)
        evaluators.append(normalized_evaluator_name)
    return evaluators


def _infer_evaluation_mode_from_evaluators(evaluators: list[str]) -> str:
    single_evaluators = set(list_evaluators())
    pairwise_evaluators = set(list_pairwise_evaluators())
    evaluator_modes_by_name: dict[str, str] = {}
    for evaluator_name in evaluators:
        is_single = evaluator_name in single_evaluators
        is_pairwise = evaluator_name in pairwise_evaluators
        if is_single and is_pairwise:
            raise ValueError(
                "Ambiguous evaluator registration for "
                f"'{evaluator_name}': registered as both single and pairwise"
            )
        if not is_single and not is_pairwise:
            available = sorted(single_evaluators | pairwise_evaluators)
            valid = ", ".join(available) if available else "(none registered)"
            raise ValueError(f"Unknown evaluator '{evaluator_name}'. Valid: {valid}")
        evaluator_modes_by_name[evaluator_name] = "single" if is_single else "pairwise"

    modes = set(evaluator_modes_by_name.values())
    if len(modes) != 1:
        single = sorted(
            evaluator_name
            for evaluator_name, mode in evaluator_modes_by_name.items()
            if mode == "single"
        )
        pairwise = sorted(
            evaluator_name
            for evaluator_name, mode in evaluator_modes_by_name.items()
            if mode == "pairwise"
        )
        raise ValueError(
            "All evaluators in an evaluation config must be the same type. "
            f"single={single}, pairwise={pairwise}"
        )

    return next(iter(modes))


def _parse_evaluator_params(
    config_data: dict[str, Any],
    *,
    evaluators: list[str],
) -> dict[str, dict[str, Any]]:
    evaluator_params_data = config_data.get("evaluator_params", {})
    if not isinstance(evaluator_params_data, dict):
        raise ValueError("'evaluator_params' must be a table when provided")

    evaluator_params: dict[str, dict[str, Any]] = {}
    for evaluator_name, raw_params in evaluator_params_data.items():
        if not isinstance(evaluator_name, str) or not evaluator_name.strip():
            raise ValueError("'evaluator_params' keys must be non-empty evaluator names")
        normalized_evaluator_name = evaluator_name.strip()
        if not isinstance(raw_params, dict):
            raise ValueError(
                f"'evaluator_params.{evaluator_name}' must be a table of evaluator settings"
            )
        evaluator_params[normalized_evaluator_name] = dict(raw_params)

    unknown_param_evaluators = sorted(set(evaluator_params) - set(evaluators))
    if unknown_param_evaluators:
        raise ValueError(
            "'evaluator_params' contains evaluator(s) not listed in 'evaluators': "
            + ", ".join(unknown_param_evaluators)
        )
    return evaluator_params


def _parse_report_path(config_data: dict[str, Any]) -> str | None:
    report_path = config_data.get("report_path")
    if report_path is not None and (not isinstance(report_path, str) or not report_path.strip()):
        raise ValueError("'report_path' must be a non-empty string when provided")
    return report_path.strip() if isinstance(report_path, str) else None


def _parse_resume(config_data: dict[str, Any]) -> bool:
    raw_resume = config_data.get("resume", False)
    if isinstance(raw_resume, bool):
        return raw_resume
    raise ValueError("'resume' must be a boolean when provided")


def _validate_allowed_top_level_keys(
    *,
    config_data: Mapping[str, Any],
    allowed_keys: set[str],
    evaluation_mode: str,
) -> None:
    unknown_keys = sorted(set(config_data) - allowed_keys)
    if not unknown_keys:
        return

    allowed_keys_text = ", ".join(sorted(allowed_keys))
    unknown_keys_text = ", ".join(unknown_keys)
    raise ValueError(
        f"{evaluation_mode} evaluation config has unsupported top-level key(s): "
        f"{unknown_keys_text}. Allowed keys: {allowed_keys_text}"
    )


def parse_evaluation_run_config(config_file: str | PathLike[str]) -> EvaluationRunConfig:
    with open(config_file, "rb") as handle:
        config_data = tomllib.load(handle)

    evaluators = _parse_evaluators(config_data)
    inferred_evaluation_mode = _infer_evaluation_mode_from_evaluators(evaluators)
    input_dir = _parse_input_dir(config_data)
    evaluator_params = _parse_evaluator_params(config_data, evaluators=evaluators)
    normalized_report_path = _parse_report_path(config_data)
    resume = _parse_resume(config_data)
    if resume and normalized_report_path is None:
        raise ValueError("'resume = true' requires an explicit 'report_path'")

    pairwise_scope: str | None = None
    if inferred_evaluation_mode == "single":
        _validate_allowed_top_level_keys(
            config_data=config_data,
            allowed_keys={"input_dir", "evaluators", "evaluator_params", "report_path", "resume"},
            evaluation_mode="single",
        )
    elif inferred_evaluation_mode == "pairwise":
        _validate_allowed_top_level_keys(
            config_data=config_data,
            allowed_keys={
                "input_dir",
                "evaluators",
                "evaluator_params",
                "report_path",
                "pairwise_scope",
                "resume",
            },
            evaluation_mode="pairwise",
        )
        pairwise_scope = parse_pairwise_scope(config_data)

    return EvaluationRunConfig(
        evaluation_mode=inferred_evaluation_mode,
        input_dir=input_dir,
        evaluators=evaluators,
        evaluator_params=evaluator_params,
        pairwise_scope=pairwise_scope,
        report_path=normalized_report_path,
        resume=resume,
    )


def _default_report_source_name(config: EvaluationRunConfig) -> str:
    source_name = Path(config.input_dir).name or "batch_run_output"
    if config.evaluation_mode == "single":
        return source_name
    return f"pairwise_{source_name}"


def _default_report_path(*, source_name: str) -> Path:
    run_id = uuid4().hex[:8]
    return Path("data") / "evaluation_results" / source_name / f"evaluation_report_{run_id}.json"


def _details_dir_for_report(report_path: Path) -> Path:
    return report_path.parent / f"{report_path.stem}_details"


def _resolve_report_and_details_paths(
    *,
    default_report_source_name: str,
    report_path: str | None,
) -> tuple[Path, Path]:
    resolved_report_path = (
        Path(report_path)
        if report_path
        else _default_report_path(source_name=default_report_source_name)
    )
    return resolved_report_path, _details_dir_for_report(resolved_report_path)


def _create_evaluators(config: EvaluationRunConfig) -> dict[str, Evaluator]:
    return {
        evaluator_name: get_evaluator(
            evaluator_name,
            params=config.evaluator_params.get(evaluator_name),
        )
        for evaluator_name in config.evaluators
    }


def _create_pairwise_evaluators(config: EvaluationRunConfig) -> dict[str, PairwiseEvaluator]:
    return {
        evaluator_name: get_pairwise_evaluator(
            evaluator_name,
            params=config.evaluator_params.get(evaluator_name),
        )
        for evaluator_name in config.evaluators
    }


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


def _infer_vary_by_from_configs(
    *,
    configs_by_slug: Mapping[str, Mapping[str, Any]],
    ignored_keys: set[str],
) -> str | None:
    if len(configs_by_slug) < 2:
        return None

    configs = list(configs_by_slug.values())
    baseline_keys = set(configs[0])
    for batch_run_config in configs[1:]:
        if set(batch_run_config) != baseline_keys:
            return None

    common_keys = set(configs[0])
    for batch_run_config in configs[1:]:
        common_keys &= set(batch_run_config)

    varying_keys: list[str] = []
    for key in sorted(common_keys):
        if key in ignored_keys:
            continue
        unique_frozen_values = {
            _freeze_for_grouping(batch_run_config[key]) for batch_run_config in configs
        }
        if len(unique_frozen_values) > 1:
            varying_keys.append(key)

    if len(varying_keys) == 1:
        return varying_keys[0]
    return None


def _resolve_single_vary_by(*, input_dir: Path) -> str | None:
    configs_by_run_id, _ = load_configs_by_run_id(input_dir)
    batch_run_configs_by_slug: dict[str, dict[str, Any]] = {}
    for config_record in configs_by_run_id.values():
        batch_run_config = config_record.batch_run_config
        config_slug = config_record.config_slug
        prior_config = batch_run_configs_by_slug.get(config_slug)
        if prior_config is None:
            batch_run_configs_by_slug[config_slug] = batch_run_config
            continue
        if prior_config != batch_run_config:
            raise ValueError(
                "Batch-run config mismatch across run_ids for "
                f"config_slug='{config_slug}' in '{input_dir / 'results.jsonl'}'."
            )
    return _infer_vary_by_from_configs(
        configs_by_slug=batch_run_configs_by_slug,
        ignored_keys=_INFERRED_VARY_BY_IGNORED_KEYS,
    )


def _resolve_pairwise_vary_by(
    *,
    records: list[PairwiseRunRecord],
    pairwise_scope: str,
) -> str | None:
    configs_by_slug: dict[str, dict[str, Any]] = {}
    for record in records:
        configs_by_slug.setdefault(record.config_slug, record.batch_run_config)

    if pairwise_scope == "inter" and len(configs_by_slug) < 2:
        raise ValueError("pairwise inter scope requires at least 2 configs with clustering outputs")

    inferred_vary_by = _infer_vary_by_from_configs(
        configs_by_slug=configs_by_slug,
        ignored_keys=_INFERRED_VARY_BY_IGNORED_KEYS,
    )
    if inferred_vary_by is None and pairwise_scope == "intra":
        return None
    if inferred_vary_by is None:
        raise ValueError(
            "Could not infer a single vary-by key for pairwise evaluation from batch-run configs."
        )
    return inferred_vary_by


def _resolve_pairwise_dataset_name(*, records: list[PairwiseRunRecord]) -> str:
    if not records:
        raise ValueError("No pairwise records loaded from experiment input_dir")
    dataset_names: set[str] = set()
    for record in records:
        raw_dataset_name = record.batch_run_config.get("dataset_name")
        if not isinstance(raw_dataset_name, str) or not raw_dataset_name.strip():
            raise ValueError(
                "Missing non-empty 'dataset_name' in pairwise experiment records for "
                f"config_slug='{record.config_slug}', run_id='{record.run_id}'"
            )
        dataset_names.add(raw_dataset_name.strip())
    if len(dataset_names) != 1:
        raise ValueError(
            "pairwise experiment input_dir must contain exactly one dataset_name across configs; "
            f"found {sorted(dataset_names)}"
        )
    return next(iter(dataset_names))


def _extract_branch_context_from_config(
    *,
    config_slug: str,
    batch_run_config: Mapping[str, Any],
) -> tuple[str, str, Any]:
    metadata = batch_run_config.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(
            "branch-aware pairwise scope requires 'metadata' in batch-run config for "
            f"config_slug='{config_slug}'"
        )
    branch_group = metadata.get("branch_group")
    if not isinstance(branch_group, str) or not branch_group.strip():
        raise ValueError(
            "branch-aware pairwise scope requires non-empty metadata['branch_group'] for "
            f"config_slug='{config_slug}'"
        )
    branch_vary_by = metadata.get("branch_vary_by")
    if not isinstance(branch_vary_by, str) or not branch_vary_by.strip():
        raise ValueError(
            "branch-aware pairwise scope requires non-empty metadata['branch_vary_by'] for "
            f"config_slug='{config_slug}'"
        )
    normalized_branch_vary_by = branch_vary_by.strip()
    if normalized_branch_vary_by not in batch_run_config:
        raise ValueError(
            "branch-aware pairwise scope requires branch vary-by key "
            f"'{normalized_branch_vary_by}' in config for config_slug='{config_slug}'"
        )
    return (
        branch_group.strip(),
        normalized_branch_vary_by,
        batch_run_config[normalized_branch_vary_by],
    )


def _build_pairwise_request_with_explicit_vary_values(
    *,
    left_record: PairwiseRunRecord,
    right_record: PairwiseRunRecord,
    pairwise_scope: str,
    vary_by: str,
    left_vary_value: Any,
    right_vary_value: Any,
) -> PairwiseEvaluationRequest:
    return PairwiseEvaluationRequest(
        left_predictions=left_record.predictions,
        right_predictions=right_record.predictions,
        artifacts={
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
                str(left_record.topic_candidates_path)
                if left_record.topic_candidates_path
                else None
            ),
            "right_topic_candidates_path": (
                str(right_record.topic_candidates_path)
                if right_record.topic_candidates_path
                else None
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
            "left_vary_value": left_vary_value,
            "right_vary_value": right_vary_value,
        },
        metadata={
            "pairwise_scope": pairwise_scope,
            "vary_by": vary_by,
        },
    )


def _build_branch_inter_pairwise_plan(
    *,
    records: list[PairwiseRunRecord],
) -> BranchScopePairwisePlan:
    dataset_name = _resolve_pairwise_dataset_name(records=records)
    records_by_config_slug: dict[str, list[PairwiseRunRecord]] = {}
    for record in records:
        records_by_config_slug.setdefault(record.config_slug, []).append(record)

    branch_vary_by_values: set[str] = set()
    grouped_records: dict[str, dict[Any, list[PairwiseRunRecord]]] = {}
    branch_values_by_key: dict[Any, Any] = {}
    for config_slug, records_for_config in records_by_config_slug.items():
        branch_group, branch_vary_by, branch_value = _extract_branch_context_from_config(
            config_slug=config_slug,
            batch_run_config=records_for_config[0].batch_run_config,
        )
        branch_vary_by_values.add(branch_vary_by)
        branch_value_key = _freeze_for_grouping(branch_value)
        branch_values_by_key.setdefault(branch_value_key, branch_value)
        grouped_records.setdefault(branch_group, {}).setdefault(branch_value_key, []).extend(
            records_for_config
        )

    if len(branch_vary_by_values) != 1:
        raise ValueError(
            "branch_inter scope requires all configs to share a single metadata['branch_vary_by']"
        )
    branch_vary_by = next(iter(branch_vary_by_values))

    vary_values_by_key: dict[Any, Any] = {}
    eligible_branch_groups: list[tuple[str, list[Any]]] = []
    for branch_group in sorted(grouped_records):
        records_by_branch_value = grouped_records[branch_group]
        branch_value_keys = sorted(records_by_branch_value, key=repr)
        if len(branch_value_keys) < 2:
            continue
        eligible_branch_groups.append((branch_group, branch_value_keys))
        for branch_value_key in branch_value_keys:
            vary_values_by_key.setdefault(
                branch_value_key,
                branch_values_by_key[branch_value_key],
            )

    if not eligible_branch_groups:
        raise ValueError(
            "branch_inter scope requires at least one branch_group with 2+ branch values "
            "and corresponding clustering artifacts"
        )

    def _iter_requests() -> Iterator[PairwiseEvaluationRequest]:
        for branch_group, branch_value_keys in eligible_branch_groups:
            records_by_branch_value = grouped_records[branch_group]
            for left_value_key, right_value_key in combinations(branch_value_keys, 2):
                left_value = {
                    "branch_group": branch_group,
                    "branch_value": branch_values_by_key[left_value_key],
                }
                right_value = {
                    "branch_group": branch_group,
                    "branch_value": branch_values_by_key[right_value_key],
                }
                for left_record in records_by_branch_value[left_value_key]:
                    for right_record in records_by_branch_value[right_value_key]:
                        yield _build_pairwise_request_with_explicit_vary_values(
                            left_record=left_record,
                            right_record=right_record,
                            pairwise_scope="branch_inter",
                            vary_by=branch_vary_by,
                            left_vary_value=left_value,
                            right_vary_value=right_value,
                        )

    return BranchScopePairwisePlan(
        dataset_name=dataset_name,
        num_configs=len(records_by_config_slug),
        vary_by=branch_vary_by,
        vary_values=sorted(vary_values_by_key.values(), key=repr),
        requests=_iter_requests(),
    )


def _build_sweep_inter_pairwise_plan(
    *,
    records: list[PairwiseRunRecord],
) -> BranchScopePairwisePlan:
    dataset_name = _resolve_pairwise_dataset_name(records=records)
    records_by_config_slug: dict[str, list[PairwiseRunRecord]] = {}
    for record in records:
        records_by_config_slug.setdefault(record.config_slug, []).append(record)

    grouped_records: dict[Any, dict[str, list[PairwiseRunRecord]]] = {}
    branch_vary_by_values: set[str] = set()
    branch_values_by_key: dict[Any, Any] = {}
    sweep_values: set[str] = set()
    for config_slug, records_for_config in records_by_config_slug.items():
        branch_group, branch_vary_by, branch_value = _extract_branch_context_from_config(
            config_slug=config_slug,
            batch_run_config=records_for_config[0].batch_run_config,
        )
        branch_vary_by_values.add(branch_vary_by)
        branch_value_key = _freeze_for_grouping(branch_value)
        branch_values_by_key.setdefault(branch_value_key, branch_value)
        sweep_values.add(branch_group)
        grouped_records.setdefault(branch_value_key, {}).setdefault(branch_group, []).extend(
            records_for_config
        )

    if len(branch_vary_by_values) != 1:
        raise ValueError(
            "sweep_inter scope requires all configs to share a single metadata['branch_vary_by']"
        )

    eligible_branch_values: list[tuple[Any, list[str]]] = []
    for branch_value_key in sorted(grouped_records, key=repr):
        records_by_branch_group = grouped_records[branch_value_key]
        branch_groups = sorted(records_by_branch_group)
        if len(branch_groups) < 2:
            continue
        eligible_branch_values.append((branch_value_key, branch_groups))

    if not eligible_branch_values:
        raise ValueError(
            "sweep_inter scope requires at least one branch value with 2+ sweep groups "
            "and corresponding clustering artifacts"
        )

    def _iter_requests() -> Iterator[PairwiseEvaluationRequest]:
        for branch_value_key, branch_groups in eligible_branch_values:
            records_by_branch_group = grouped_records[branch_value_key]
            for left_group, right_group in combinations(branch_groups, 2):
                left_value = {
                    "branch_value": branch_values_by_key[branch_value_key],
                    "branch_group": left_group,
                }
                right_value = {
                    "branch_value": branch_values_by_key[branch_value_key],
                    "branch_group": right_group,
                }
                for left_record in records_by_branch_group[left_group]:
                    for right_record in records_by_branch_group[right_group]:
                        yield _build_pairwise_request_with_explicit_vary_values(
                            left_record=left_record,
                            right_record=right_record,
                            pairwise_scope="sweep_inter",
                            vary_by="branch_group",
                            left_vary_value=left_value,
                            right_vary_value=right_value,
                        )

    return BranchScopePairwisePlan(
        dataset_name=dataset_name,
        num_configs=len(records_by_config_slug),
        vary_by="branch_group",
        vary_values=sorted(sweep_values),
        requests=_iter_requests(),
    )


def _evaluate_single_requests_and_write_details(
    *,
    input_dir: Path,
    load_benchmark_specs: bool,
    evaluator_names: list[str],
    evaluators_by_name: dict[str, Evaluator],
    evaluator_params_by_name: dict[str, dict[str, Any]],
    accumulators_by_evaluator: dict[str, EvaluationSummaryAccumulator],
    handles_by_evaluator: dict[str, TextIO],
    existing_records_by_evaluator: dict[str, dict[tuple[str, ...], ExistingResultRecord]],
    current_request_keys_by_evaluator: dict[str, set[tuple[str, ...]]],
) -> SingleEvaluationRollup:
    inferred_vary_by = _resolve_single_vary_by(input_dir=input_dir)
    num_artifacts_by_evaluator: dict[str, int] = {}
    num_artifacts_by_config_by_evaluator: dict[str, dict[str, int]] = {}
    accumulators_by_config_by_evaluator: dict[
        str, dict[str, dict[str, EvaluationSummaryAccumulator]]
    ] = {}
    num_failures_by_evaluator: dict[str, int] = {}
    num_failures_by_config_by_evaluator: dict[str, dict[str, dict[str, int]]] = {}
    vary_by_by_evaluator: dict[str, str | None] = {}
    shared_run_request_mode_by_evaluator: dict[str, bool] = {}

    for evaluator_name in evaluator_names:
        coalesce_auto_topic_shared_runs = _single_evaluator_uses_shared_run_requests(
            evaluator_name=evaluator_name
        )
        vary_by = None if coalesce_auto_topic_shared_runs else inferred_vary_by
        shared_run_request_mode_by_evaluator[evaluator_name] = coalesce_auto_topic_shared_runs
        vary_by_by_evaluator[evaluator_name] = vary_by

        num_artifacts = 0
        num_artifacts_by_config: dict[str, int] = {}
        accumulators_by_config: dict[str, dict[str, EvaluationSummaryAccumulator]] = {}
        evaluator_failure_counts = {evaluator_name: 0}
        num_failures_by_config: dict[str, dict[str, int]] = {}
        evaluator = evaluators_by_name[evaluator_name]

        for request in iter_batch_run_requests(
            input_dir,
            load_benchmark_specs=load_benchmark_specs,
            coalesce_auto_topic_shared_runs=coalesce_auto_topic_shared_runs,
            require_clustering=_single_evaluator_requires_clustering(evaluator_name=evaluator_name),
        ):
            num_artifacts += 1
            request_label = _format_single_request_label(request)
            _emit_progress(
                f"[evaluation] single artifact {num_artifacts} evaluator '{evaluator_name}' "
                f"queued: {request_label}"
            )
            config_slug_raw = request.metadata.get("config_slug")
            config_slug = (
                config_slug_raw if isinstance(config_slug_raw, str) else str(config_slug_raw)
            )
            summary_config_slug_raw = request.artifacts.get("summary_config_slug")
            summary_group_key = (
                summary_config_slug_raw if isinstance(summary_config_slug_raw, str) else config_slug
            )
            if vary_by is not None:
                batch_run_config = request.artifacts.get("batch_run_config")
                if not isinstance(batch_run_config, dict) or vary_by not in batch_run_config:
                    run_id = request.metadata.get("run_id")
                    raise ValueError(
                        f"'vary_by' key '{vary_by}' is missing from single-evaluation artifact "
                        f"for config_slug='{config_slug}', run_id='{run_id}'"
                    )
                summary_group_key = _format_vary_value_key(value=batch_run_config[vary_by])

            _increment_group_count(
                group_key=summary_group_key,
                counts_by_group=num_artifacts_by_config,
            )
            accumulators_by_evaluator_for_config = _get_or_create_accumulators_for_group(
                group_key=summary_group_key,
                evaluator_names=[evaluator_name],
                accumulators_by_group=accumulators_by_config,
            )

            request_key = _single_request_key(request)
            current_request_keys_by_evaluator[evaluator_name].add(request_key)
            resume_fingerprint = _build_single_resume_fingerprint(
                request=request,
                evaluator=evaluator,
                evaluator_name=evaluator_name,
                evaluator_params=evaluator_params_by_name.get(evaluator_name),
            )
            existing_record = existing_records_by_evaluator[evaluator_name].get(request_key)
            if (
                existing_record is not None
                and existing_record.status == "ok"
                and _result_record_matches_resume_fingerprint(
                    record=existing_record,
                    expected_resume_fingerprint=resume_fingerprint,
                )
            ):
                assert existing_record.result is not None
                accumulators_by_evaluator[evaluator_name].add_result(existing_record.result)
                accumulators_by_evaluator_for_config[evaluator_name].add_result(
                    existing_record.result
                )
                _emit_progress(
                    f"[evaluation] single artifact {num_artifacts} evaluator '{evaluator_name}' "
                    f"skipped via resume: {request_label}"
                )
                continue

            started_at = perf_counter()
            _emit_progress(
                f"[evaluation] single artifact {num_artifacts} evaluator '{evaluator_name}' "
                f"started: {request_label}"
            )
            try:
                result = evaluator.evaluate(request)
            except Exception as exc:
                _increment_failure_count(
                    evaluator_name=evaluator_name,
                    counts_by_evaluator=evaluator_failure_counts,
                )
                _increment_group_failure_count(
                    group_key=summary_group_key,
                    evaluator_name=evaluator_name,
                    counts_by_group=num_failures_by_config,
                )
                error_record = _single_error_to_record(
                    request=request,
                    evaluator_name=evaluator_name,
                    error=exc,
                )
                error_record["resume_fingerprint"] = resume_fingerprint
                _store_result_record(
                    records_by_evaluator=existing_records_by_evaluator,
                    evaluator_name=evaluator_name,
                    key=request_key,
                    record=error_record,
                    result=None,
                )
                _write_result_record(handles_by_evaluator[evaluator_name], error_record)
                elapsed_seconds = perf_counter() - started_at
                _emit_evaluation_error(
                    "[evaluation] single artifact "
                    f"{num_artifacts} evaluator '{evaluator_name}' failed in "
                    f"{elapsed_seconds:.1f}s: {request_label}",
                    exc,
                )
                continue
            accumulators_by_evaluator[evaluator_name].add_result(result)
            accumulators_by_evaluator_for_config[evaluator_name].add_result(result)

            result_record = _single_result_to_record(
                request=request,
                result=result,
            )
            result_record["resume_fingerprint"] = resume_fingerprint
            _store_result_record(
                records_by_evaluator=existing_records_by_evaluator,
                evaluator_name=evaluator_name,
                key=request_key,
                record=result_record,
                result=result,
            )
            _write_result_record(handles_by_evaluator[evaluator_name], result_record)
            elapsed_seconds = perf_counter() - started_at
            _emit_progress(
                f"[evaluation] single artifact {num_artifacts} evaluator '{evaluator_name}' "
                f"finished in {elapsed_seconds:.1f}s: {request_label}"
            )

        num_artifacts_by_evaluator[evaluator_name] = num_artifacts
        num_artifacts_by_config_by_evaluator[evaluator_name] = num_artifacts_by_config
        accumulators_by_config_by_evaluator[evaluator_name] = accumulators_by_config
        num_failures_by_evaluator[evaluator_name] = evaluator_failure_counts[evaluator_name]
        num_failures_by_config_by_evaluator[evaluator_name] = num_failures_by_config

    return SingleEvaluationRollup(
        num_artifacts_by_evaluator=num_artifacts_by_evaluator,
        num_artifacts_by_config_by_evaluator=num_artifacts_by_config_by_evaluator,
        accumulators_by_config_by_evaluator=accumulators_by_config_by_evaluator,
        num_failures_by_evaluator=num_failures_by_evaluator,
        num_failures_by_config_by_evaluator=num_failures_by_config_by_evaluator,
        vary_by_by_evaluator=vary_by_by_evaluator,
        shared_run_request_mode_by_evaluator=shared_run_request_mode_by_evaluator,
    )


def _format_vary_pair_key(*, left: Any, right: Any) -> str:
    return f"{left!r}->{right!r}"


def _format_vary_value_key(*, value: Any) -> str:
    return f"{value!r}"


def _build_empty_accumulators_by_evaluator(
    *,
    evaluator_names: list[str],
) -> dict[str, EvaluationSummaryAccumulator]:
    return {evaluator_name: EvaluationSummaryAccumulator() for evaluator_name in evaluator_names}


def _get_or_create_accumulators_for_group(
    *,
    group_key: str,
    evaluator_names: list[str],
    accumulators_by_group: dict[str, dict[str, EvaluationSummaryAccumulator]],
) -> dict[str, EvaluationSummaryAccumulator]:
    accumulators_by_evaluator_for_group = accumulators_by_group.get(group_key)
    if accumulators_by_evaluator_for_group is None:
        accumulators_by_evaluator_for_group = _build_empty_accumulators_by_evaluator(
            evaluator_names=evaluator_names
        )
        accumulators_by_group[group_key] = accumulators_by_evaluator_for_group
    return accumulators_by_evaluator_for_group


def _increment_group_count(
    *,
    group_key: str,
    counts_by_group: dict[str, int],
) -> None:
    counts_by_group[group_key] = counts_by_group.get(group_key, 0) + 1


def _increment_failure_count(
    *,
    evaluator_name: str,
    counts_by_evaluator: dict[str, int],
) -> None:
    counts_by_evaluator[evaluator_name] = counts_by_evaluator.get(evaluator_name, 0) + 1


def _increment_group_failure_count(
    *,
    group_key: str,
    evaluator_name: str,
    counts_by_group: dict[str, dict[str, int]],
) -> None:
    counts_by_group.setdefault(group_key, {})
    _increment_failure_count(
        evaluator_name=evaluator_name,
        counts_by_evaluator=counts_by_group[group_key],
    )


def _emit_progress(message: str) -> None:
    logger.info(message)


def _emit_evaluation_error(message: str, exc: Exception) -> None:
    logger.exception(message, exc_info=exc)


def _single_request_key(request: EvaluationRequest) -> tuple[str, ...]:
    return (
        str(request.metadata.get("config_slug")),
        str(request.metadata.get("run_id")),
    )


def _pairwise_request_key(request: PairwiseEvaluationRequest) -> tuple[str, ...]:
    return (
        str(request.artifacts.get("left_config_slug")),
        str(request.artifacts.get("left_run_id")),
        str(request.artifacts.get("right_config_slug")),
        str(request.artifacts.get("right_run_id")),
    )


def _load_existing_result_records(
    *,
    path: Path,
    evaluation_mode: str,
    evaluator_name: str,
) -> dict[tuple[str, ...], ExistingResultRecord]:
    if not path.exists():
        return {}

    records_by_key: dict[tuple[str, ...], ExistingResultRecord] = {}
    with open(path, encoding="utf-8") as handle:
        for line_index, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in existing evaluator details '{path}' at line "
                    f"{line_index}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Invalid evaluator details payload in '{path}' at line {line_index}: "
                    "expected object"
                )
            if payload.get("method") != evaluator_name:
                raise ValueError(
                    f"Existing evaluator details '{path}' at line {line_index} has method "
                    f"{payload.get('method')!r}; expected {evaluator_name!r}"
                )
            if evaluation_mode == "single":
                key = (
                    str(payload.get("config_slug")),
                    str(payload.get("run_id")),
                )
            else:
                key = (
                    str(payload.get("left_config_slug")),
                    str(payload.get("left_run_id")),
                    str(payload.get("right_config_slug")),
                    str(payload.get("right_run_id")),
                )
            status = payload.get("status")
            normalized_status = (
                status.strip() if isinstance(status, str) and status.strip() else "ok"
            )
            if normalized_status == "error":
                records_by_key[key] = ExistingResultRecord(
                    key=key,
                    status="error",
                    result=None,
                    raw_record=dict(payload),
                )
                continue
            metrics = payload.get("metrics")
            details = payload.get("details")
            if not isinstance(metrics, dict) or not isinstance(details, dict):
                raise ValueError(
                    f"Existing evaluator details '{path}' at line {line_index} must include "
                    "object 'metrics' and 'details' fields"
                )
            records_by_key[key] = ExistingResultRecord(
                key=key,
                status="ok",
                result=EvaluationResult(
                    method=evaluator_name,
                    metrics={str(name): float(value) for name, value in metrics.items()},
                    details=dict(details),
                ),
                raw_record=dict(payload),
            )
    return records_by_key


def _format_single_request_label(request: EvaluationRequest) -> str:
    return (
        f"config_slug={request.metadata.get('config_slug')}, "
        f"run_id={request.metadata.get('run_id')}"
    )


def _format_pairwise_request_label(request: PairwiseEvaluationRequest) -> str:
    return (
        f"left_config_slug={request.artifacts.get('left_config_slug')}, "
        f"left_run_id={request.artifacts.get('left_run_id')}, "
        f"right_config_slug={request.artifacts.get('right_config_slug')}, "
        f"right_run_id={request.artifacts.get('right_run_id')}"
    )


def _write_result_record(handle: TextIO, record: Mapping[str, Any]) -> None:
    json.dump(record, handle)
    handle.write("\n")
    handle.flush()


def _store_result_record(
    *,
    records_by_evaluator: dict[str, dict[tuple[str, ...], ExistingResultRecord]],
    evaluator_name: str,
    key: tuple[str, ...],
    record: dict[str, Any],
    result: EvaluationResult | None,
) -> None:
    status = record.get("status")
    normalized_status = status if isinstance(status, str) and status == "error" else "ok"
    records_by_evaluator[evaluator_name][key] = ExistingResultRecord(
        key=key,
        status=normalized_status,
        result=result,
        raw_record=record,
    )


def _rewrite_result_records(
    *,
    path: Path,
    records_by_key: dict[tuple[str, ...], ExistingResultRecord],
) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records_by_key.values():
            json.dump(record.raw_record, handle)
            handle.write("\n")


def _evaluate_pairwise_requests_and_write_details(
    *,
    pairwise_requests: Iterator[PairwiseEvaluationRequest],
    pairwise_scope: str,
    evaluator_names: list[str],
    evaluators_by_name: dict[str, PairwiseEvaluator],
    evaluator_params_by_name: dict[str, dict[str, Any]],
    accumulators_by_evaluator: dict[str, EvaluationSummaryAccumulator],
    handles_by_evaluator: dict[str, TextIO],
    existing_records_by_evaluator: dict[str, dict[tuple[str, ...], ExistingResultRecord]],
    current_request_keys_by_evaluator: dict[str, set[tuple[str, ...]]],
) -> PairwiseEvaluationRollup:
    num_pairwise_requests = 0
    num_pairwise_requests_by_vary_value_pair: dict[str, int] = {}
    accumulators_by_vary_value_pair: dict[str, dict[str, EvaluationSummaryAccumulator]] = {}
    num_pairwise_requests_by_intra_vary_value: dict[str, int] = {}
    intra_accumulators_by_vary_value: dict[str, dict[str, EvaluationSummaryAccumulator]] = {}
    num_failures_by_evaluator = {evaluator_name: 0 for evaluator_name in evaluator_names}
    num_failures_by_vary_value_pair: dict[str, dict[str, int]] = {}
    num_failures_by_intra_vary_value: dict[str, dict[str, int]] = {}
    for request in pairwise_requests:
        num_pairwise_requests += 1
        request_label = _format_pairwise_request_label(request)
        _emit_progress(
            f"[evaluation] pairwise request {num_pairwise_requests} queued: {request_label}"
        )
        accumulators_by_evaluator_for_vary_value: dict[str, EvaluationSummaryAccumulator] | None = (
            None
        )
        accumulators_by_evaluator_for_vary_value_pair: dict[str, EvaluationSummaryAccumulator]
        if pairwise_scope == "intra":
            intra_vary_value_key = _format_vary_value_key(
                value=request.artifacts.get("left_vary_value")
            )
            _increment_group_count(
                group_key=intra_vary_value_key,
                counts_by_group=num_pairwise_requests_by_intra_vary_value,
            )
            accumulators_by_evaluator_for_vary_value = _get_or_create_accumulators_for_group(
                group_key=intra_vary_value_key,
                evaluator_names=evaluator_names,
                accumulators_by_group=intra_accumulators_by_vary_value,
            )

        vary_value_pair = _format_vary_pair_key(
            left=request.artifacts.get("left_vary_value"),
            right=request.artifacts.get("right_vary_value"),
        )
        _increment_group_count(
            group_key=vary_value_pair,
            counts_by_group=num_pairwise_requests_by_vary_value_pair,
        )
        accumulators_by_evaluator_for_vary_value_pair = _get_or_create_accumulators_for_group(
            group_key=vary_value_pair,
            evaluator_names=evaluator_names,
            accumulators_by_group=accumulators_by_vary_value_pair,
        )

        for evaluator_name in evaluator_names:
            evaluator = evaluators_by_name[evaluator_name]
            request_key = _pairwise_request_key(request)
            current_request_keys_by_evaluator[evaluator_name].add(request_key)
            resume_fingerprint = _build_pairwise_resume_fingerprint(
                request=request,
                evaluator=evaluator,
                evaluator_name=evaluator_name,
                evaluator_params=evaluator_params_by_name.get(evaluator_name),
            )
            existing_record = existing_records_by_evaluator[evaluator_name].get(request_key)
            if (
                existing_record is not None
                and existing_record.status == "ok"
                and _result_record_matches_resume_fingerprint(
                    record=existing_record,
                    expected_resume_fingerprint=resume_fingerprint,
                )
            ):
                assert existing_record.result is not None
                accumulators_by_evaluator[evaluator_name].add_result(existing_record.result)
                accumulators_by_evaluator_for_vary_value_pair[evaluator_name].add_result(
                    existing_record.result
                )
                if accumulators_by_evaluator_for_vary_value is not None:
                    accumulators_by_evaluator_for_vary_value[evaluator_name].add_result(
                        existing_record.result
                    )
                _emit_progress(
                    f"[evaluation] pairwise request {num_pairwise_requests} evaluator "
                    f"'{evaluator_name}' skipped via resume: {request_label}"
                )
                continue
            started_at = perf_counter()
            _emit_progress(
                f"[evaluation] pairwise request {num_pairwise_requests} evaluator "
                f"'{evaluator_name}' started: {request_label}"
            )
            try:
                result = evaluator.evaluate(request)
            except Exception as exc:
                _increment_failure_count(
                    evaluator_name=evaluator_name,
                    counts_by_evaluator=num_failures_by_evaluator,
                )
                _increment_group_failure_count(
                    group_key=vary_value_pair,
                    evaluator_name=evaluator_name,
                    counts_by_group=num_failures_by_vary_value_pair,
                )
                if accumulators_by_evaluator_for_vary_value is not None:
                    assert pairwise_scope == "intra"
                    _increment_group_failure_count(
                        group_key=intra_vary_value_key,
                        evaluator_name=evaluator_name,
                        counts_by_group=num_failures_by_intra_vary_value,
                    )
                error_record = _pairwise_error_to_record(
                    request=request,
                    evaluator_name=evaluator_name,
                    error=exc,
                )
                error_record["resume_fingerprint"] = resume_fingerprint
                _store_result_record(
                    records_by_evaluator=existing_records_by_evaluator,
                    evaluator_name=evaluator_name,
                    key=request_key,
                    record=error_record,
                    result=None,
                )
                _write_result_record(handles_by_evaluator[evaluator_name], error_record)
                elapsed_seconds = perf_counter() - started_at
                _emit_evaluation_error(
                    "[evaluation] pairwise request "
                    f"{num_pairwise_requests} evaluator '{evaluator_name}' failed in "
                    f"{elapsed_seconds:.1f}s: {request_label}",
                    exc,
                )
                continue
            accumulators_by_evaluator[evaluator_name].add_result(result)
            accumulators_by_evaluator_for_vary_value_pair[evaluator_name].add_result(result)
            if accumulators_by_evaluator_for_vary_value is not None:
                accumulators_by_evaluator_for_vary_value[evaluator_name].add_result(result)

            result_record = _pairwise_result_to_record(
                request=request,
                result=result,
            )
            result_record["resume_fingerprint"] = resume_fingerprint
            _store_result_record(
                records_by_evaluator=existing_records_by_evaluator,
                evaluator_name=evaluator_name,
                key=request_key,
                record=result_record,
                result=result,
            )
            _write_result_record(handles_by_evaluator[evaluator_name], result_record)
            elapsed_seconds = perf_counter() - started_at
            _emit_progress(
                f"[evaluation] pairwise request {num_pairwise_requests} evaluator "
                f"'{evaluator_name}' finished in {elapsed_seconds:.1f}s: {request_label}"
            )

    return PairwiseEvaluationRollup(
        num_pairwise_requests=num_pairwise_requests,
        num_pairwise_requests_by_vary_value_pair=num_pairwise_requests_by_vary_value_pair,
        accumulators_by_vary_value_pair=accumulators_by_vary_value_pair,
        num_pairwise_requests_by_intra_vary_value=num_pairwise_requests_by_intra_vary_value,
        intra_accumulators_by_vary_value=intra_accumulators_by_vary_value,
        num_failures_by_evaluator=num_failures_by_evaluator,
        num_failures_by_vary_value_pair=num_failures_by_vary_value_pair,
        num_failures_by_intra_vary_value=num_failures_by_intra_vary_value,
    )


def _evaluate_pairwise_intra_requests_without_vary_and_write_details(
    *,
    pairwise_requests: Iterator[PairwiseEvaluationRequest],
    evaluator_names: list[str],
    evaluators_by_name: dict[str, PairwiseEvaluator],
    evaluator_params_by_name: dict[str, dict[str, Any]],
    accumulators_by_evaluator: dict[str, EvaluationSummaryAccumulator],
    handles_by_evaluator: dict[str, TextIO],
    existing_records_by_evaluator: dict[str, dict[tuple[str, ...], ExistingResultRecord]],
    current_request_keys_by_evaluator: dict[str, set[tuple[str, ...]]],
) -> PairwiseIntraNoVaryEvaluationRollup:
    num_pairwise_requests = 0
    num_pairwise_requests_by_config_slug: dict[str, int] = {}
    accumulators_by_config_slug: dict[str, dict[str, EvaluationSummaryAccumulator]] = {}
    num_failures_by_evaluator = {evaluator_name: 0 for evaluator_name in evaluator_names}
    num_failures_by_config_slug: dict[str, dict[str, int]] = {}
    for request in pairwise_requests:
        num_pairwise_requests += 1
        request_label = _format_pairwise_request_label(request)
        _emit_progress(
            f"[evaluation] pairwise request {num_pairwise_requests} queued: {request_label}"
        )
        config_slug = str(request.artifacts.get("left_config_slug"))
        _increment_group_count(
            group_key=config_slug,
            counts_by_group=num_pairwise_requests_by_config_slug,
        )
        accumulators_by_evaluator_for_config = _get_or_create_accumulators_for_group(
            group_key=config_slug,
            evaluator_names=evaluator_names,
            accumulators_by_group=accumulators_by_config_slug,
        )
        for evaluator_name in evaluator_names:
            evaluator = evaluators_by_name[evaluator_name]
            request_key = _pairwise_request_key(request)
            current_request_keys_by_evaluator[evaluator_name].add(request_key)
            resume_fingerprint = _build_pairwise_resume_fingerprint(
                request=request,
                evaluator=evaluator,
                evaluator_name=evaluator_name,
                evaluator_params=evaluator_params_by_name.get(evaluator_name),
            )
            existing_record = existing_records_by_evaluator[evaluator_name].get(request_key)
            if (
                existing_record is not None
                and existing_record.status == "ok"
                and _result_record_matches_resume_fingerprint(
                    record=existing_record,
                    expected_resume_fingerprint=resume_fingerprint,
                )
            ):
                assert existing_record.result is not None
                accumulators_by_evaluator[evaluator_name].add_result(existing_record.result)
                accumulators_by_evaluator_for_config[evaluator_name].add_result(
                    existing_record.result
                )
                _emit_progress(
                    f"[evaluation] pairwise request {num_pairwise_requests} evaluator "
                    f"'{evaluator_name}' skipped via resume: {request_label}"
                )
                continue
            started_at = perf_counter()
            _emit_progress(
                f"[evaluation] pairwise request {num_pairwise_requests} evaluator "
                f"'{evaluator_name}' started: {request_label}"
            )
            try:
                result = evaluator.evaluate(request)
            except Exception as exc:
                _increment_failure_count(
                    evaluator_name=evaluator_name,
                    counts_by_evaluator=num_failures_by_evaluator,
                )
                _increment_group_failure_count(
                    group_key=config_slug,
                    evaluator_name=evaluator_name,
                    counts_by_group=num_failures_by_config_slug,
                )
                error_record = _pairwise_error_to_record(
                    request=request,
                    evaluator_name=evaluator_name,
                    error=exc,
                )
                error_record["resume_fingerprint"] = resume_fingerprint
                _store_result_record(
                    records_by_evaluator=existing_records_by_evaluator,
                    evaluator_name=evaluator_name,
                    key=request_key,
                    record=error_record,
                    result=None,
                )
                _write_result_record(handles_by_evaluator[evaluator_name], error_record)
                elapsed_seconds = perf_counter() - started_at
                _emit_evaluation_error(
                    "[evaluation] pairwise request "
                    f"{num_pairwise_requests} evaluator '{evaluator_name}' failed in "
                    f"{elapsed_seconds:.1f}s: {request_label}",
                    exc,
                )
                continue
            accumulators_by_evaluator[evaluator_name].add_result(result)
            accumulators_by_evaluator_for_config[evaluator_name].add_result(result)

            result_record = _pairwise_result_to_record(
                request=request,
                result=result,
            )
            result_record["resume_fingerprint"] = resume_fingerprint
            _store_result_record(
                records_by_evaluator=existing_records_by_evaluator,
                evaluator_name=evaluator_name,
                key=request_key,
                record=result_record,
                result=result,
            )
            _write_result_record(handles_by_evaluator[evaluator_name], result_record)
            elapsed_seconds = perf_counter() - started_at
            _emit_progress(
                f"[evaluation] pairwise request {num_pairwise_requests} evaluator "
                f"'{evaluator_name}' finished in {elapsed_seconds:.1f}s: {request_label}"
            )
    return PairwiseIntraNoVaryEvaluationRollup(
        num_pairwise_requests=num_pairwise_requests,
        num_pairwise_requests_by_config_slug=num_pairwise_requests_by_config_slug,
        accumulators_by_config_slug=accumulators_by_config_slug,
        num_failures_by_evaluator=num_failures_by_evaluator,
        num_failures_by_config_slug=num_failures_by_config_slug,
    )


def _single_result_to_record(
    *, request: EvaluationRequest, result: EvaluationResult
) -> dict[str, Any]:
    return {
        "clustering_path": request.artifacts.get("clustering_path"),
        "conversation_trace_path": request.artifacts.get("conversation_trace_path"),
        "batch_run_config_path": request.artifacts.get("batch_run_config_path"),
        "config_slug": request.metadata.get("config_slug"),
        "run_id": request.metadata.get("run_id"),
        "method": result.method,
        "metrics": result.metrics,
        "details": result.details,
        "accounting": result.accounting,
    }


def _single_error_to_record(
    *,
    request: EvaluationRequest,
    evaluator_name: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "clustering_path": request.artifacts.get("clustering_path"),
        "conversation_trace_path": request.artifacts.get("conversation_trace_path"),
        "batch_run_config_path": request.artifacts.get("batch_run_config_path"),
        "config_slug": request.metadata.get("config_slug"),
        "run_id": request.metadata.get("run_id"),
        "method": evaluator_name,
        "status": "error",
        "metrics": {},
        "details": {
            "error_type": type(error).__name__,
            "error_message": str(error),
        },
    }


def _pairwise_result_to_record(
    *,
    request: PairwiseEvaluationRequest,
    result: EvaluationResult,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "pairwise_scope": request.metadata.get("pairwise_scope"),
        "left_config_slug": request.artifacts.get("left_config_slug"),
        "right_config_slug": request.artifacts.get("right_config_slug"),
        "left_run_id": request.artifacts.get("left_run_id"),
        "right_run_id": request.artifacts.get("right_run_id"),
        "left_clustering_path": request.artifacts.get("left_clustering_path"),
        "right_clustering_path": request.artifacts.get("right_clustering_path"),
        "left_conversation_trace_path": request.artifacts.get("left_conversation_trace_path"),
        "right_conversation_trace_path": request.artifacts.get("right_conversation_trace_path"),
        "method": result.method,
        "metrics": result.metrics,
        "details": result.details,
        "accounting": result.accounting,
    }
    if "vary_by" in request.metadata:
        record["vary_by"] = request.metadata.get("vary_by")
    if "left_vary_value" in request.artifacts:
        record["left_vary_value"] = request.artifacts.get("left_vary_value")
    if "right_vary_value" in request.artifacts:
        record["right_vary_value"] = request.artifacts.get("right_vary_value")
    return record


def _pairwise_error_to_record(
    *,
    request: PairwiseEvaluationRequest,
    evaluator_name: str,
    error: Exception,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "pairwise_scope": request.metadata.get("pairwise_scope"),
        "left_config_slug": request.artifacts.get("left_config_slug"),
        "right_config_slug": request.artifacts.get("right_config_slug"),
        "left_run_id": request.artifacts.get("left_run_id"),
        "right_run_id": request.artifacts.get("right_run_id"),
        "left_clustering_path": request.artifacts.get("left_clustering_path"),
        "right_clustering_path": request.artifacts.get("right_clustering_path"),
        "left_conversation_trace_path": request.artifacts.get("left_conversation_trace_path"),
        "right_conversation_trace_path": request.artifacts.get("right_conversation_trace_path"),
        "method": evaluator_name,
        "status": "error",
        "metrics": {},
        "details": {
            "error_type": type(error).__name__,
            "error_message": str(error),
        },
    }
    if "vary_by" in request.metadata:
        record["vary_by"] = request.metadata.get("vary_by")
    if "left_vary_value" in request.artifacts:
        record["left_vary_value"] = request.artifacts.get("left_vary_value")
    if "right_vary_value" in request.artifacts:
        record["right_vary_value"] = request.artifacts.get("right_vary_value")
    return record


def _build_summary_by_evaluator(
    accumulators_by_evaluator: dict[str, EvaluationSummaryAccumulator],
) -> dict[str, dict[str, Any]]:
    summary_by_evaluator: dict[str, dict[str, Any]] = {}
    for evaluator_name, accumulator in accumulators_by_evaluator.items():
        summary_result = accumulator.to_summary_result()
        summary_record: dict[str, Any] = {
            "metrics": summary_result.metrics,
            "details": summary_result.details,
        }
        if summary_result.accounting is not None:
            summary_record["accounting"] = summary_result.accounting
        summary_by_evaluator[evaluator_name] = summary_record
    return summary_by_evaluator


def _build_summary_by_group(
    *,
    evaluator_names: list[str],
    accumulators_by_group: dict[str, dict[str, EvaluationSummaryAccumulator]],
    counts_by_group: dict[str, int],
    failure_counts_by_group: dict[str, dict[str, int]] | None,
    count_field_name: str,
    expected_group_keys: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    summary_by_group: dict[str, dict[str, Any]] = {}
    group_keys = sorted(
        set(expected_group_keys or []) | set(accumulators_by_group) | set(counts_by_group)
    )
    for group_key in group_keys:
        accumulators_by_evaluator_for_group = accumulators_by_group.get(group_key)
        if accumulators_by_evaluator_for_group is None:
            accumulators_by_evaluator_for_group = _build_empty_accumulators_by_evaluator(
                evaluator_names=evaluator_names
            )
        summary_by_group[group_key] = {
            count_field_name: counts_by_group.get(group_key, 0),
            "num_failures_by_evaluator": {
                evaluator_name: failure_counts_by_group.get(group_key, {}).get(evaluator_name, 0)
                for evaluator_name in evaluator_names
            }
            if failure_counts_by_group is not None
            else {evaluator_name: 0 for evaluator_name in evaluator_names},
            "summary_by_evaluator": _build_summary_by_evaluator(
                accumulators_by_evaluator_for_group
            ),
        }
    return summary_by_group


def _build_single_summary_by_config(
    *,
    evaluator_names: list[str],
    accumulators_by_config: dict[str, dict[str, EvaluationSummaryAccumulator]],
    num_artifacts_by_config: dict[str, int],
    num_failures_by_config: dict[str, dict[str, int]] | None = None,
) -> dict[str, dict[str, Any]]:
    return _build_summary_by_group(
        evaluator_names=evaluator_names,
        accumulators_by_group=accumulators_by_config,
        counts_by_group=num_artifacts_by_config,
        failure_counts_by_group=num_failures_by_config,
        count_field_name="num_artifacts",
    )


def _build_single_summary_by_config_by_evaluator(
    *,
    evaluator_names: list[str],
    accumulators_by_config_by_evaluator: dict[
        str, dict[str, dict[str, EvaluationSummaryAccumulator]]
    ],
    num_artifacts_by_config_by_evaluator: dict[str, dict[str, int]],
    num_failures_by_config_by_evaluator: dict[str, dict[str, dict[str, int]]] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    summary_by_config_by_evaluator: dict[str, dict[str, dict[str, Any]]] = {}
    for evaluator_name in evaluator_names:
        summary_by_config_by_evaluator[evaluator_name] = _build_summary_by_group(
            evaluator_names=[evaluator_name],
            accumulators_by_group=accumulators_by_config_by_evaluator.get(evaluator_name, {}),
            counts_by_group=num_artifacts_by_config_by_evaluator.get(evaluator_name, {}),
            failure_counts_by_group=(
                None
                if num_failures_by_config_by_evaluator is None
                else num_failures_by_config_by_evaluator.get(evaluator_name, {})
            ),
            count_field_name="num_artifacts",
        )
    return summary_by_config_by_evaluator


def _build_intra_summary_by_vary_value(
    *,
    vary_values: list[Any],
    evaluator_names: list[str],
    intra_accumulators_by_vary_value: dict[str, dict[str, EvaluationSummaryAccumulator]],
    num_pairwise_requests_by_intra_vary_value: dict[str, int],
    num_failures_by_intra_vary_value: dict[str, dict[str, int]],
) -> dict[str, dict[str, Any]]:
    return _build_summary_by_group(
        evaluator_names=evaluator_names,
        accumulators_by_group=intra_accumulators_by_vary_value,
        counts_by_group=num_pairwise_requests_by_intra_vary_value,
        failure_counts_by_group=num_failures_by_intra_vary_value,
        count_field_name="num_pairwise_requests",
        expected_group_keys=[
            _format_vary_value_key(value=vary_value) for vary_value in vary_values
        ],
    )


def _expected_vary_value_pair_keys(
    *,
    pairwise_scope: str,
    vary_values: list[Any],
) -> list[str]:
    if pairwise_scope == "inter":
        return [
            _format_vary_pair_key(left=left_vary_value, right=right_vary_value)
            for left_vary_value, right_vary_value in combinations(vary_values, 2)
        ]
    if pairwise_scope == "intra":
        return [
            _format_vary_pair_key(left=vary_value, right=vary_value) for vary_value in vary_values
        ]
    valid = ", ".join(sorted(["inter", "intra"]))
    raise ValueError(f"Unexpected pairwise_scope='{pairwise_scope}'. Expected one of: {valid}")


def _build_summary_by_vary_value_pair(
    *,
    pairwise_scope: str,
    vary_values: list[Any],
    evaluator_names: list[str],
    accumulators_by_vary_value_pair: dict[str, dict[str, EvaluationSummaryAccumulator]],
    num_pairwise_requests_by_vary_value_pair: dict[str, int],
    num_failures_by_vary_value_pair: dict[str, dict[str, int]],
) -> dict[str, dict[str, Any]]:
    expected_pair_keys = _expected_vary_value_pair_keys(
        pairwise_scope=pairwise_scope, vary_values=vary_values
    )
    return _build_summary_by_group(
        evaluator_names=evaluator_names,
        accumulators_by_group=accumulators_by_vary_value_pair,
        counts_by_group=num_pairwise_requests_by_vary_value_pair,
        failure_counts_by_group=num_failures_by_vary_value_pair,
        count_field_name="num_pairwise_requests",
        expected_group_keys=expected_pair_keys,
    )


def _write_report(report_path: Path, report: dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


def run_evaluation_config(config_file: str | PathLike[str]) -> str:
    config = parse_evaluation_run_config(config_file)
    load_benchmark_specs = _evaluation_requires_benchmark_specs(config)
    default_report_source_name = _default_report_source_name(config)
    report_path, details_dir = _resolve_report_and_details_paths(
        default_report_source_name=default_report_source_name,
        report_path=config.report_path,
    )
    accumulators_by_evaluator = {
        evaluator_name: EvaluationSummaryAccumulator() for evaluator_name in config.evaluators
    }
    results_files_by_evaluator = {
        evaluator_name: details_dir / f"{evaluator_name}.jsonl"
        for evaluator_name in config.evaluators
    }
    current_request_keys_by_evaluator = {
        evaluator_name: set() for evaluator_name in config.evaluators
    }
    existing_records_by_evaluator = {
        evaluator_name: (
            _load_existing_result_records(
                path=path,
                evaluation_mode=config.evaluation_mode,
                evaluator_name=evaluator_name,
            )
            if config.resume
            else {}
        )
        for evaluator_name, path in results_files_by_evaluator.items()
    }

    details_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        handles_by_evaluator = {
            evaluator_name: stack.enter_context(
                open(path, "a" if config.resume else "w", encoding="utf-8")
            )
            for evaluator_name, path in results_files_by_evaluator.items()
        }
        if config.evaluation_mode == "single":
            input_dir = Path(config.input_dir)
            evaluators_by_name = _create_evaluators(config)
            single_rollup = _evaluate_single_requests_and_write_details(
                input_dir=input_dir,
                load_benchmark_specs=load_benchmark_specs,
                evaluator_names=config.evaluators,
                evaluators_by_name=evaluators_by_name,
                evaluator_params_by_name=config.evaluator_params,
                accumulators_by_evaluator=accumulators_by_evaluator,
                handles_by_evaluator=handles_by_evaluator,
                existing_records_by_evaluator=existing_records_by_evaluator,
                current_request_keys_by_evaluator=current_request_keys_by_evaluator,
            )
            mode_result_fields = {
                "num_artifacts_by_evaluator": {
                    evaluator_name: single_rollup.num_artifacts_by_evaluator[evaluator_name]
                    for evaluator_name in config.evaluators
                },
                "num_artifacts_by_config_by_evaluator": {
                    evaluator_name: {
                        config_slug: single_rollup.num_artifacts_by_config_by_evaluator[
                            evaluator_name
                        ][config_slug]
                        for config_slug in sorted(
                            single_rollup.num_artifacts_by_config_by_evaluator[evaluator_name]
                        )
                    }
                    for evaluator_name in config.evaluators
                },
                "summary_by_config_by_evaluator": _build_single_summary_by_config_by_evaluator(
                    evaluator_names=config.evaluators,
                    accumulators_by_config_by_evaluator=(
                        single_rollup.accumulators_by_config_by_evaluator
                    ),
                    num_artifacts_by_config_by_evaluator=(
                        single_rollup.num_artifacts_by_config_by_evaluator
                    ),
                    num_failures_by_config_by_evaluator=(
                        single_rollup.num_failures_by_config_by_evaluator
                    ),
                ),
            }
            mode_source_fields: dict[str, Any] = {"input_dir": str(input_dir)}
            shared_run_modes = set(single_rollup.shared_run_request_mode_by_evaluator.values())
            vary_by_values = set(single_rollup.vary_by_by_evaluator.values())
            if len(shared_run_modes) == 1 and len(vary_by_values) == 1:
                shared_vary_by = next(iter(vary_by_values))
                first_evaluator_name = config.evaluators[0]
                mode_result_fields["num_artifacts"] = single_rollup.num_artifacts_by_evaluator[
                    first_evaluator_name
                ]
                mode_result_fields["num_artifacts_by_config"] = {
                    config_slug: single_rollup.num_artifacts_by_config_by_evaluator[
                        first_evaluator_name
                    ][config_slug]
                    for config_slug in sorted(
                        single_rollup.num_artifacts_by_config_by_evaluator[first_evaluator_name]
                    )
                }
                mode_result_fields["summary_by_config"] = _build_single_summary_by_config(
                    evaluator_names=config.evaluators,
                    accumulators_by_config=single_rollup.accumulators_by_config_by_evaluator[
                        first_evaluator_name
                    ],
                    num_artifacts_by_config=single_rollup.num_artifacts_by_config_by_evaluator[
                        first_evaluator_name
                    ],
                    num_failures_by_config=single_rollup.num_failures_by_config_by_evaluator[
                        first_evaluator_name
                    ],
                )
                if shared_vary_by is not None:
                    mode_source_fields["vary_by"] = shared_vary_by
            else:
                mode_source_fields["vary_by_by_evaluator"] = single_rollup.vary_by_by_evaluator
                mode_source_fields["shared_run_request_mode_by_evaluator"] = (
                    single_rollup.shared_run_request_mode_by_evaluator
                )
        elif config.evaluation_mode == "pairwise":
            if config.pairwise_scope is None:
                raise ValueError(
                    "pairwise evaluation config is missing required field 'pairwise_scope'"
                )
            input_dir = Path(config.input_dir)

            pairwise_records = load_pairwise_inputs(
                input_dir,
                load_benchmark_specs=load_benchmark_specs,
            )
            num_artifacts_by_config: dict[str, int] = {}
            for record in pairwise_records:
                num_artifacts_by_config[record.config_slug] = (
                    num_artifacts_by_config.get(record.config_slug, 0) + 1
                )
            pairwise_evaluators_by_name = _create_pairwise_evaluators(config)
            if config.pairwise_scope in {"branch_inter", "sweep_inter"}:
                branch_scope_plan = (
                    _build_branch_inter_pairwise_plan(records=pairwise_records)
                    if config.pairwise_scope == "branch_inter"
                    else _build_sweep_inter_pairwise_plan(records=pairwise_records)
                )
                pairwise_rollup = _evaluate_pairwise_requests_and_write_details(
                    pairwise_requests=iter(branch_scope_plan.requests),
                    pairwise_scope=config.pairwise_scope,
                    evaluator_names=config.evaluators,
                    evaluators_by_name=pairwise_evaluators_by_name,
                    evaluator_params_by_name=config.evaluator_params,
                    accumulators_by_evaluator=accumulators_by_evaluator,
                    handles_by_evaluator=handles_by_evaluator,
                    existing_records_by_evaluator=existing_records_by_evaluator,
                    current_request_keys_by_evaluator=current_request_keys_by_evaluator,
                )
                mode_result_fields = {
                    "num_artifacts": len(pairwise_records),
                    "num_configs": branch_scope_plan.num_configs,
                    "vary_values": branch_scope_plan.vary_values,
                    "num_artifacts_by_config": {
                        config_slug: num_artifacts_by_config[config_slug]
                        for config_slug in sorted(num_artifacts_by_config)
                    },
                    "num_pairwise_requests": pairwise_rollup.num_pairwise_requests,
                    "num_pairwise_requests_by_vary_value_pair": {
                        vary_value_pair: pairwise_rollup.num_pairwise_requests_by_vary_value_pair[
                            vary_value_pair
                        ]
                        for vary_value_pair in sorted(
                            pairwise_rollup.num_pairwise_requests_by_vary_value_pair
                        )
                    },
                    "summary_by_vary_value_pair": _build_summary_by_group(
                        evaluator_names=config.evaluators,
                        accumulators_by_group=pairwise_rollup.accumulators_by_vary_value_pair,
                        counts_by_group=pairwise_rollup.num_pairwise_requests_by_vary_value_pair,
                        failure_counts_by_group=pairwise_rollup.num_failures_by_vary_value_pair,
                        count_field_name="num_pairwise_requests",
                    ),
                }
                mode_source_fields = {
                    "input_dir": str(input_dir),
                    "dataset_name": branch_scope_plan.dataset_name,
                    "pairwise_scope": config.pairwise_scope,
                    "vary_by": branch_scope_plan.vary_by,
                }
            else:
                pairwise_vary_by = _resolve_pairwise_vary_by(
                    records=pairwise_records,
                    pairwise_scope=config.pairwise_scope,
                )
                if pairwise_vary_by is None:
                    if config.pairwise_scope != "intra":
                        raise ValueError(
                            "Could not infer a single vary-by key for pairwise evaluation from "
                            "batch-run configs."
                        )
                    experiment_summary = validate_intra_experiment_records_without_vary(
                        records=pairwise_records,
                    )
                    pairwise_rollup_no_vary = (
                        _evaluate_pairwise_intra_requests_without_vary_and_write_details(
                            pairwise_requests=iter_pairwise_intra_requests_without_vary(
                                records=pairwise_records
                            ),
                            evaluator_names=config.evaluators,
                            evaluators_by_name=pairwise_evaluators_by_name,
                            evaluator_params_by_name=config.evaluator_params,
                            accumulators_by_evaluator=accumulators_by_evaluator,
                            handles_by_evaluator=handles_by_evaluator,
                            existing_records_by_evaluator=existing_records_by_evaluator,
                            current_request_keys_by_evaluator=current_request_keys_by_evaluator,
                        )
                    )
                    mode_result_fields = {
                        "num_artifacts": len(pairwise_records),
                        "num_configs": experiment_summary.num_configs,
                        "num_artifacts_by_config": {
                            config_slug: num_artifacts_by_config[config_slug]
                            for config_slug in sorted(num_artifacts_by_config)
                        },
                        "num_pairwise_requests": pairwise_rollup_no_vary.num_pairwise_requests,
                        "num_pairwise_requests_by_config_slug": {
                            config_slug: (
                                pairwise_rollup_no_vary.num_pairwise_requests_by_config_slug[
                                    config_slug
                                ]
                            )
                            for config_slug in sorted(
                                pairwise_rollup_no_vary.num_pairwise_requests_by_config_slug
                            )
                        },
                        "summary_by_config_slug": _build_summary_by_group(
                            evaluator_names=config.evaluators,
                            accumulators_by_group=(
                                pairwise_rollup_no_vary.accumulators_by_config_slug
                            ),
                            counts_by_group=(
                                pairwise_rollup_no_vary.num_pairwise_requests_by_config_slug
                            ),
                            failure_counts_by_group=(
                                pairwise_rollup_no_vary.num_failures_by_config_slug
                            ),
                            count_field_name="num_pairwise_requests",
                        ),
                    }
                    mode_source_fields = {
                        "input_dir": str(input_dir),
                        "dataset_name": experiment_summary.dataset_name,
                        "pairwise_scope": config.pairwise_scope,
                    }
                else:
                    validate_vary_by_key(records=pairwise_records, vary_by=pairwise_vary_by)
                    experiment_summary = validate_experiment_records(
                        records=pairwise_records,
                        vary_by=pairwise_vary_by,
                        pairwise_scope=config.pairwise_scope,
                    )
                    pairwise_rollup = _evaluate_pairwise_requests_and_write_details(
                        pairwise_requests=iter_pairwise_requests(
                            records=pairwise_records,
                            vary_by=pairwise_vary_by,
                            pairwise_scope=config.pairwise_scope,
                        ),
                        pairwise_scope=config.pairwise_scope,
                        evaluator_names=config.evaluators,
                        evaluators_by_name=pairwise_evaluators_by_name,
                        evaluator_params_by_name=config.evaluator_params,
                        accumulators_by_evaluator=accumulators_by_evaluator,
                        handles_by_evaluator=handles_by_evaluator,
                        existing_records_by_evaluator=existing_records_by_evaluator,
                        current_request_keys_by_evaluator=current_request_keys_by_evaluator,
                    )
                    mode_result_fields = {
                        "num_artifacts": len(pairwise_records),
                        "num_configs": experiment_summary.num_configs,
                        "vary_values": experiment_summary.vary_values,
                        "num_artifacts_by_config": {
                            config_slug: num_artifacts_by_config[config_slug]
                            for config_slug in sorted(num_artifacts_by_config)
                        },
                        "num_pairwise_requests": pairwise_rollup.num_pairwise_requests,
                        "num_pairwise_requests_by_vary_value_pair": {
                            vary_value_pair: (
                                pairwise_rollup.num_pairwise_requests_by_vary_value_pair[
                                    vary_value_pair
                                ]
                            )
                            for vary_value_pair in sorted(
                                pairwise_rollup.num_pairwise_requests_by_vary_value_pair
                            )
                        },
                        "summary_by_vary_value_pair": _build_summary_by_vary_value_pair(
                            pairwise_scope=config.pairwise_scope,
                            vary_values=experiment_summary.vary_values,
                            evaluator_names=config.evaluators,
                            accumulators_by_vary_value_pair=(
                                pairwise_rollup.accumulators_by_vary_value_pair
                            ),
                            num_pairwise_requests_by_vary_value_pair=(
                                pairwise_rollup.num_pairwise_requests_by_vary_value_pair
                            ),
                            num_failures_by_vary_value_pair=(
                                pairwise_rollup.num_failures_by_vary_value_pair
                            ),
                        ),
                    }
                    if config.pairwise_scope == "intra":
                        mode_result_fields["summary_by_vary_value"] = (
                            _build_intra_summary_by_vary_value(
                                vary_values=experiment_summary.vary_values,
                                evaluator_names=config.evaluators,
                                intra_accumulators_by_vary_value=(
                                    pairwise_rollup.intra_accumulators_by_vary_value
                                ),
                                num_pairwise_requests_by_intra_vary_value=(
                                    pairwise_rollup.num_pairwise_requests_by_intra_vary_value
                                ),
                                num_failures_by_intra_vary_value=(
                                    pairwise_rollup.num_failures_by_intra_vary_value
                                ),
                            )
                        )
                    mode_source_fields = {
                        "input_dir": str(input_dir),
                        "dataset_name": experiment_summary.dataset_name,
                        "pairwise_scope": config.pairwise_scope,
                        "vary_by": pairwise_vary_by,
                    }
        else:
            raise ValueError(
                (
                    f"Found evaluation mode {config.evaluation_mode} is unknown.",
                    'Expected one of: ["single", "pairwise"]',
                )
            )

    summary_by_evaluator = _build_summary_by_evaluator(accumulators_by_evaluator)
    if config.evaluation_mode == "single":
        num_failures_by_evaluator = single_rollup.num_failures_by_evaluator
    elif config.pairwise_scope in {"branch_inter", "sweep_inter"} or (
        config.evaluation_mode == "pairwise" and "pairwise_rollup" in locals()
    ):
        num_failures_by_evaluator = pairwise_rollup.num_failures_by_evaluator
    else:
        num_failures_by_evaluator = pairwise_rollup_no_vary.num_failures_by_evaluator

    report = {
        "evaluation_mode": config.evaluation_mode,
        **mode_source_fields,
        "report_path": str(report_path),
        "evaluators": config.evaluators,
        "evaluator_params": config.evaluator_params,
        "resume": config.resume,
        **mode_result_fields,
        "num_failures_by_evaluator": num_failures_by_evaluator,
        "summary_by_evaluator": summary_by_evaluator,
        "results_file_format": "jsonl",
        "results_files_by_evaluator": {
            evaluator_name: str(path) for evaluator_name, path in results_files_by_evaluator.items()
        },
    }

    if config.resume:
        for evaluator_name, path in results_files_by_evaluator.items():
            active_records_by_key = {
                key: record
                for key, record in existing_records_by_evaluator[evaluator_name].items()
                if key in current_request_keys_by_evaluator[evaluator_name]
            }
            existing_records_by_evaluator[evaluator_name] = active_records_by_key
            _rewrite_result_records(
                path=path,
                records_by_key=active_records_by_key,
            )

    _write_report(report_path, report)

    if config.evaluation_mode == "single":
        if "num_artifacts" in mode_result_fields:
            evaluation_count_text = f"artifacts={mode_result_fields['num_artifacts']}"
        else:
            evaluation_count_text = (
                f"artifacts_by_evaluator={mode_result_fields['num_artifacts_by_evaluator']}"
            )
    elif config.evaluation_mode == "pairwise":
        evaluation_count_text = (
            f"artifacts={mode_result_fields['num_artifacts']}, "
            f"pairwise_requests={mode_result_fields['num_pairwise_requests']}"
        )
    total_failures = sum(num_failures_by_evaluator.values())

    _emit_progress(
        f"Evaluation complete: mode={config.evaluation_mode}, "
        f"evaluators={config.evaluators}, {evaluation_count_text}, "
        f"failures={total_failures}, report={report_path}"
    )
    if total_failures > 0:
        raise EvaluationRunFailed(report_path=report_path, total_failures=total_failures)
    return str(report_path)
