"""Utilities for loading batch-run artifacts and resolving run-to-config mappings."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

from conversational_clustering.batch_runs.config import GENERATED_BENCHMARK_SPEC_METADATA_KEY

from .benchmark_spec import (
    SyntheticBenchmarkSpec,
    load_benchmark_spec_for_dataset,
    load_synthetic_benchmark_spec,
)


@dataclass(frozen=True)
class BatchRunArtifactContext:
    input_dir: Path
    results_path: Path
    run_dir: Path
    clustering_path: Path
    conversation_trace_path: Path
    judge_clustering_trace_path: Path | None
    clustering_pov_history_path: Path | None
    human_thought_trace_path: Path | None
    topic_candidates_path: Path | None
    selected_topic_path: Path | None
    config_slug: str
    run_id: str
    batch_run_config: dict[str, Any]
    batch_run_result: dict[str, Any] | None
    turns_run: int | None
    conversation_trace: dict[str, Any]
    judge_clustering_trace: dict[str, Any] | None
    clustering_pov_history: dict[str, Any] | None
    human_thought_trace: dict[str, Any] | None
    topic_candidates: dict[str, Any] | None
    selected_topic: dict[str, Any] | None
    benchmark_spec: SyntheticBenchmarkSpec | None


@dataclass(frozen=True)
class BatchRunConfigRecord:
    batch_run_config: dict[str, Any]
    config_slug: str
    batch_run_result: dict[str, Any] | None


@dataclass(frozen=True)
class BatchRunManifestRecord:
    config_slug: str
    run_id: str
    run_dir: Path
    clustering_path: Path
    batch_run_config: dict[str, Any]
    batch_run_result: dict[str, Any] | None


def _is_failed_batch_run_result(batch_run_result: dict[str, Any] | None) -> bool:
    if batch_run_result is None:
        return False
    stopped_reason = batch_run_result.get("stopped_reason")
    return isinstance(stopped_reason, str) and stopped_reason.startswith("run_error_")


def _has_saved_clustering_result(batch_run_result: dict[str, Any] | None) -> bool:
    if batch_run_result is None:
        return True
    stopped_reason = batch_run_result.get("stopped_reason")
    if stopped_reason is None:
        return True
    return stopped_reason == "clustering_saved"


def list_batch_run_artifact_paths(input_dir: str | PathLike[str]) -> list[Path]:
    return [
        record.clustering_path
        for record in list_batch_run_manifest_records(input_dir, require_clustering=True)
    ]


def _load_conversation_trace(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(
            f"Missing required conversation trace artifact '{path}'. "
            "Expected a JSON file generated from non-tool AgentState messages."
        )
    with open(path, encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in conversation trace '{path}': {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid conversation trace '{path}': expected top-level object")
    return payload


def _load_optional_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in '{path}': {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid JSON artifact '{path}': expected top-level object")
    return payload


def _resolve_shared_topic_candidates_path(
    *,
    input_dir: Path,
    config_slug: str,
    run_id: str,
    batch_run_config: dict[str, Any],
) -> Path | None:
    metadata = batch_run_config.get("metadata")
    if not isinstance(metadata, dict):
        return None
    raw_source_config_slug = metadata.get("topic_branch_source_config_slug")
    if not isinstance(raw_source_config_slug, str) or not raw_source_config_slug.strip():
        return None
    source_config_slug = raw_source_config_slug.strip()
    shared_root = input_dir / "_shared" / source_config_slug
    if not shared_root.is_dir():
        return None

    for mapping_path in sorted(shared_root.glob("*/topic_branch_mapping.json")):
        mapping_payload = _load_optional_json_object(mapping_path)
        if mapping_payload is None:
            continue
        branches = mapping_payload.get("branches")
        if not isinstance(branches, list):
            continue
        for branch in branches:
            if not isinstance(branch, dict):
                continue
            branch_config_slug = branch.get("config_slug")
            branch_run_id = branch.get("run_id")
            if branch_config_slug == config_slug and branch_run_id == run_id:
                candidate_path = mapping_path.parent / "topic_candidates.json"
                if candidate_path.is_file():
                    return candidate_path
    return None


def _resolve_generated_benchmark_spec_path(
    *,
    input_dir: Path,
    batch_run_config: dict[str, Any],
) -> Path | None:
    metadata = batch_run_config.get("metadata")
    if not isinstance(metadata, dict):
        return None
    raw_path = metadata.get(GENERATED_BENCHMARK_SPEC_METADATA_KEY)
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    configured_path = Path(raw_path.strip())
    if configured_path.is_absolute():
        return configured_path
    return input_dir / configured_path


def _load_benchmark_spec_for_batch_run(
    *,
    input_dir: Path,
    batch_run_config: dict[str, Any],
    benchmark_spec_cache: dict[str, SyntheticBenchmarkSpec | None],
) -> SyntheticBenchmarkSpec | None:
    raw_dataset_name = batch_run_config.get("dataset_name")
    dataset_name = raw_dataset_name.strip() if isinstance(raw_dataset_name, str) else None

    generated_benchmark_path = _resolve_generated_benchmark_spec_path(
        input_dir=input_dir,
        batch_run_config=batch_run_config,
    )
    if generated_benchmark_path is not None:
        resolved_benchmark_path = generated_benchmark_path.resolve()
        cache_key = f"path:{resolved_benchmark_path}"
        if cache_key not in benchmark_spec_cache:
            if not resolved_benchmark_path.is_file():
                raise ValueError(
                    "Generated benchmark spec configured in batch-run metadata does not exist: "
                    f"'{resolved_benchmark_path}'"
                )
            benchmark_spec = load_synthetic_benchmark_spec(resolved_benchmark_path)
            if dataset_name is not None and benchmark_spec.dataset_name != dataset_name:
                raise ValueError(
                    "Generated benchmark spec configured in batch-run metadata does not match "
                    f"dataset_name='{dataset_name}': '{resolved_benchmark_path}' declares "
                    f"dataset_name='{benchmark_spec.dataset_name}'"
                )
            benchmark_spec_cache[cache_key] = benchmark_spec
        return benchmark_spec_cache[cache_key]

    if dataset_name is None:
        return None

    cache_key = f"dataset:{dataset_name}"
    if cache_key not in benchmark_spec_cache:
        benchmark_spec_cache[cache_key] = load_benchmark_spec_for_dataset(dataset_name)
    return benchmark_spec_cache[cache_key]


def _resolve_request_paths(clustering_path: Path) -> tuple[str, str]:
    run_id = clustering_path.parent.name
    config_slug = clustering_path.parent.parent.name
    if not run_id or not config_slug:
        raise ValueError(
            "Expected clustering artifact path "
            "'<input_dir>/<config_slug>/<run_id>/clustering.json', "
            f"got '{clustering_path}'"
        )
    return config_slug, run_id


def _load_manifest_records(
    input_dir: Path,
    *,
    require_clustering: bool,
) -> tuple[list[BatchRunManifestRecord], dict[str, BatchRunConfigRecord], Path]:
    results_path = input_dir / "results.jsonl"
    if not results_path.exists():
        raise ValueError(
            f"Missing '{results_path}'. Evaluation needs this file to map each run to its config."
        )

    configs_by_run_id: dict[str, BatchRunConfigRecord] = {}
    records: list[BatchRunManifestRecord] = []
    with open(results_path, encoding="utf-8") as handle:
        for line_idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in '{results_path}' at line {line_idx}: {exc.msg}"
                ) from exc

            if not isinstance(payload, dict):
                raise ValueError(
                    f"Invalid results payload in '{results_path}' at line {line_idx}: "
                    "expected object"
                )

            config_payload = payload.get("config")
            if not isinstance(config_payload, dict):
                raise ValueError(
                    f"Invalid results payload in '{results_path}' at line {line_idx}: "
                    "missing object 'config'"
                )

            config_slug_payload = payload.get("config_slug")
            if not isinstance(config_slug_payload, str) or not config_slug_payload.strip():
                raise ValueError(
                    f"Invalid results payload in '{results_path}' at line {line_idx}: "
                    "missing non-empty string 'config_slug'"
                )
            config_slug = config_slug_payload.strip()

            run_ids_payload = payload.get("run_ids")
            if not isinstance(run_ids_payload, list):
                raise ValueError(
                    f"Invalid results payload in '{results_path}' at line {line_idx}: "
                    "missing list 'run_ids'"
                )
            raw_results_payload = payload.get("results")
            results_payload: list[dict[str, Any]] | None = None
            if raw_results_payload is not None:
                if not isinstance(raw_results_payload, list):
                    raise ValueError(
                        f"Invalid results payload in '{results_path}' at line {line_idx}: "
                        "'results' must be a list when provided"
                    )
                if len(raw_results_payload) != len(run_ids_payload):
                    raise ValueError(
                        f"Invalid results payload in '{results_path}' at line {line_idx}: "
                        "'results' must align 1:1 with 'run_ids'"
                    )
                results_payload = []
                for result_idx, raw_result in enumerate(raw_results_payload):
                    if not isinstance(raw_result, dict):
                        raise ValueError(
                            f"Invalid results payload in '{results_path}' at line {line_idx}: "
                            f"results[{result_idx}] must be an object"
                        )
                    results_payload.append(raw_result)

            for run_id_idx, raw_run_id in enumerate(run_ids_payload):
                if not isinstance(raw_run_id, str) or not raw_run_id.strip():
                    raise ValueError(
                        f"Invalid results payload in '{results_path}' at line {line_idx}: "
                        f"run_ids[{run_id_idx}] must be a non-empty string"
                    )
                run_id = raw_run_id.strip()
                if run_id in configs_by_run_id:
                    raise ValueError(f"Duplicate run_id '{run_id}' found in '{results_path}'.")
                batch_run_result = (
                    results_payload[run_id_idx] if results_payload is not None else None
                )
                configs_by_run_id[run_id] = BatchRunConfigRecord(
                    batch_run_config=config_payload,
                    config_slug=config_slug,
                    batch_run_result=batch_run_result,
                )
                if _is_failed_batch_run_result(batch_run_result):
                    continue
                run_dir = input_dir / config_slug / run_id
                clustering_path = run_dir / "clustering.json"
                if require_clustering and not _has_saved_clustering_result(batch_run_result):
                    continue
                if require_clustering and not clustering_path.is_file():
                    raise ValueError(
                        "Missing required clustering artifact "
                        f"'{clustering_path}' for run_id '{run_id}' listed in "
                        f"'{results_path}'."
                    )
                records.append(
                    BatchRunManifestRecord(
                        config_slug=config_slug,
                        run_id=run_id,
                        run_dir=run_dir,
                        clustering_path=clustering_path,
                        batch_run_config=config_payload,
                        batch_run_result=batch_run_result,
                    )
                )

    if not configs_by_run_id:
        raise ValueError(f"No run_id entries found in '{results_path}'.")
    if not records:
        raise ValueError(f"No evaluable batch-run artifacts found in '{results_path}'.")

    return records, configs_by_run_id, results_path


def list_batch_run_manifest_records(
    input_dir: str | PathLike[str],
    *,
    require_clustering: bool = True,
) -> list[BatchRunManifestRecord]:
    input_dir_path = Path(input_dir)
    records, _, _ = _load_manifest_records(
        input_dir_path,
        require_clustering=require_clustering,
    )
    return records


def load_configs_by_run_id(
    input_dir: Path,
) -> tuple[dict[str, BatchRunConfigRecord], Path]:
    records, _, results_path = _load_manifest_records(
        input_dir,
        require_clustering=False,
    )
    configs_by_run_id = {
        record.run_id: BatchRunConfigRecord(
            batch_run_config=record.batch_run_config,
            config_slug=record.config_slug,
            batch_run_result=record.batch_run_result,
        )
        for record in records
    }

    return configs_by_run_id, results_path


def resolve_batch_run_context_for_artifact(
    *,
    clustering_path: Path,
    configs_by_run_id: dict[str, BatchRunConfigRecord],
    results_path: Path,
) -> tuple[str, str, dict[str, Any], dict[str, Any] | None]:
    config_slug, run_id = _resolve_request_paths(clustering_path)
    run_mapping = configs_by_run_id.get(run_id)
    if run_mapping is None:
        raise ValueError(
            "Unable to load matching batch-run config for "
            f"'{clustering_path}': run_id '{run_id}' missing from '{results_path}'."
        )
    loaded_batch_run_config = run_mapping.batch_run_config
    config_slug_from_results = run_mapping.config_slug
    if config_slug_from_results != config_slug:
        raise ValueError(
            "Batch-run config mismatch for "
            f"'{clustering_path}': path config_slug='{config_slug}' but "
            f"results config_slug='{config_slug_from_results}' for run_id '{run_id}'."
        )
    return config_slug, run_id, loaded_batch_run_config, run_mapping.batch_run_result


def _extract_turns_run(batch_run_result: dict[str, Any] | None) -> int | None:
    if batch_run_result is None:
        return None
    raw_turns_run = batch_run_result.get("turns_run")
    if raw_turns_run is None:
        return None
    if isinstance(raw_turns_run, bool) or not isinstance(raw_turns_run, int) or raw_turns_run < 0:
        raise ValueError("batch-run result 'turns_run' must be an integer >= 0 when provided")
    return raw_turns_run


def iter_batch_run_artifact_contexts(
    input_dir: str | PathLike[str],
    *,
    load_benchmark_specs: bool = True,
    require_clustering: bool = True,
) -> Iterator[BatchRunArtifactContext]:
    input_dir_path = Path(input_dir)
    manifest_records, _, results_path = _load_manifest_records(
        input_dir_path,
        require_clustering=require_clustering,
    )
    benchmark_spec_cache: dict[str, SyntheticBenchmarkSpec | None] = {}
    for manifest_record in manifest_records:
        config_slug = manifest_record.config_slug
        run_id = manifest_record.run_id
        batch_run_config = manifest_record.batch_run_config
        batch_run_result = manifest_record.batch_run_result
        run_dir = manifest_record.run_dir
        clustering_path = manifest_record.clustering_path
        conversation_trace_path = clustering_path.with_name("conversation_trace.json")
        conversation_trace = _load_conversation_trace(conversation_trace_path)
        judge_clustering_trace_path = clustering_path.with_name("judge_clustering_trace.json")
        judge_clustering_trace = _load_optional_json_object(judge_clustering_trace_path)
        clustering_pov_history_path = clustering_path.with_name("clustering_pov_history.json")
        clustering_pov_history = _load_optional_json_object(clustering_pov_history_path)
        human_thought_trace_path = clustering_path.with_name("human_thought_trace.json")
        human_thought_trace = _load_optional_json_object(human_thought_trace_path)
        topic_candidates_path = clustering_path.with_name("topic_candidates.json")
        topic_candidates = _load_optional_json_object(topic_candidates_path)
        if topic_candidates is None:
            shared_topic_candidates_path = _resolve_shared_topic_candidates_path(
                input_dir=input_dir_path,
                config_slug=config_slug,
                run_id=run_id,
                batch_run_config=batch_run_config,
            )
            if shared_topic_candidates_path is not None:
                topic_candidates_path = shared_topic_candidates_path
                topic_candidates = _load_optional_json_object(topic_candidates_path)
        selected_topic_path = clustering_path.with_name("selected_topic.json")
        selected_topic = _load_optional_json_object(selected_topic_path)
        benchmark_spec = None
        if load_benchmark_specs:
            benchmark_spec = _load_benchmark_spec_for_batch_run(
                input_dir=input_dir_path,
                batch_run_config=batch_run_config,
                benchmark_spec_cache=benchmark_spec_cache,
            )
        yield BatchRunArtifactContext(
            input_dir=input_dir_path,
            results_path=results_path,
            run_dir=run_dir,
            clustering_path=clustering_path,
            conversation_trace_path=conversation_trace_path,
            judge_clustering_trace_path=(
                judge_clustering_trace_path if judge_clustering_trace is not None else None
            ),
            clustering_pov_history_path=(
                clustering_pov_history_path if clustering_pov_history is not None else None
            ),
            human_thought_trace_path=(
                human_thought_trace_path if human_thought_trace is not None else None
            ),
            topic_candidates_path=(topic_candidates_path if topic_candidates is not None else None),
            selected_topic_path=(selected_topic_path if selected_topic is not None else None),
            config_slug=config_slug,
            run_id=run_id,
            batch_run_config=batch_run_config,
            batch_run_result=batch_run_result,
            turns_run=_extract_turns_run(batch_run_result),
            conversation_trace=conversation_trace,
            judge_clustering_trace=judge_clustering_trace,
            clustering_pov_history=clustering_pov_history,
            human_thought_trace=human_thought_trace,
            topic_candidates=topic_candidates,
            selected_topic=selected_topic,
            benchmark_spec=benchmark_spec,
        )
