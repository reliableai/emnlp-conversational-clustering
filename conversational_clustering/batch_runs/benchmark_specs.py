"""Helpers for generating sampled benchmark specs that match batch-run dataset windows."""

from __future__ import annotations

import csv
from dataclasses import replace
from os import PathLike
from pathlib import Path

from conversational_clustering.data_processing.datasets import DATASETS
from conversational_clustering.evaluation.benchmark_builders import (
    DEFAULT_ENTERPRISE_TICKET_THEME_DEFINITIONS,
    build_enterprise_ticket_benchmark_spec,
    load_labeled_csv_rows,
    write_benchmark_spec,
)

from .config import GENERATED_BENCHMARK_SPEC_METADATA_KEY, BatchRunConfig


def attach_generated_benchmark_specs(
    *,
    output_dir: str | PathLike[str],
    configs: list[BatchRunConfig],
) -> list[BatchRunConfig]:
    resolved_output_dir = Path(output_dir)
    generated_paths_by_key: dict[tuple[str, int | None, int], str | None] = {}
    updated_configs: list[BatchRunConfig] = []
    for config in configs:
        cache_key = (config.dataset_name, config.limit, config.random_seed)
        generated_rel_path = generated_paths_by_key.get(cache_key)
        if generated_rel_path is None:
            generated_rel_path = _maybe_generate_benchmark_spec(
                output_dir=resolved_output_dir,
                config=config,
            )
            generated_paths_by_key[cache_key] = generated_rel_path
        if generated_rel_path is None:
            updated_configs.append(config)
            continue
        metadata = dict(config.metadata)
        metadata[GENERATED_BENCHMARK_SPEC_METADATA_KEY] = generated_rel_path
        updated_configs.append(replace(config, metadata=metadata))
    return updated_configs


def _maybe_generate_benchmark_spec(
    *,
    output_dir: Path,
    config: BatchRunConfig,
) -> str | None:
    if config.dataset_name != "enterprise_ticket":
        return None
    dataset_config = DATASETS.get(config.dataset_name)
    if dataset_config is None:
        raise ValueError(f"Unknown dataset '{config.dataset_name}'")
    dataset_path = dataset_config.get("path")
    if not isinstance(dataset_path, str) or not dataset_path.strip():
        raise ValueError(f"Dataset '{config.dataset_name}' has no valid CSV path configured")
    item_column = dataset_config.get("item_column", "item")
    if not isinstance(item_column, str) or not item_column.strip():
        raise ValueError(f"Dataset '{config.dataset_name}' has no valid item_column configured")
    if not _has_enterprise_ticket_benchmark_columns(
        dataset_path=dataset_path,
        item_column=item_column,
    ):
        return None

    sampled_rows = load_labeled_csv_rows(
        path=dataset_path,
        limit=config.limit,
        random_seed=config.random_seed,
    )
    payload = build_enterprise_ticket_benchmark_spec(
        sampled_rows,
        dataset_name=config.dataset_name,
        item_column=item_column,
    )
    relative_output_path = _build_generated_benchmark_spec_relpath(config)
    write_benchmark_spec(output_dir / relative_output_path, payload)
    return relative_output_path.as_posix()


def _build_generated_benchmark_spec_relpath(config: BatchRunConfig) -> Path:
    path_stem = config.dataset_name
    if config.limit is None:
        path_stem += "__all"
    else:
        path_stem += f"__limit-{config.limit}__seed-{config.random_seed}"
    return Path("_benchmark_specs") / f"{path_stem}.benchmark.json"


def _has_enterprise_ticket_benchmark_columns(
    *,
    dataset_path: str | PathLike[str],
    item_column: str,
) -> bool:
    required_columns = {
        item_column,
        *(theme.theme_id for theme in DEFAULT_ENTERPRISE_TICKET_THEME_DEFINITIONS),
    }
    with open(dataset_path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    if header is None:
        raise ValueError(f"CSV '{dataset_path}' is missing a header row")
    normalized_header = {
        column.strip() for column in header if isinstance(column, str) and column.strip()
    }
    return required_columns.issubset(normalized_header)
