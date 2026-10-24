"""Generic orchestration for preset-backed structured dataset generation."""

from __future__ import annotations

import random

from .io import default_summary_path, write_csv, write_text
from .llm import generate_texts_with_openrouter
from .types import OpenRouterConfig, StructuredDatasetPreset


def _format_schema_support_report(report: dict[str, list[str]]) -> str:
    if not report:
        return "Schema support audit: no unsupported declared labels found."
    lines = ["Schema support audit:"]
    for issue_name in sorted(report):
        lines.append(f"  {issue_name}:")
        for value in report[issue_name]:
            lines.append(f"    - {value}")
    return "\n".join(lines)


def handle_schema_support_report(report: dict[str, list[str]], mode: str) -> None:
    if mode not in {"warn", "error", "ignore"}:
        raise ValueError(f"Unsupported schema audit mode: {mode}")
    if mode == "ignore" or not report:
        return

    message = _format_schema_support_report(report)
    if mode == "error":
        raise ValueError(message)
    print(message)


def generate_dataset_rows(
    preset: StructuredDatasetPreset,
    n: int,
    rng: random.Random | None = None,
    *,
    mode: str,
    difficulty: str,
    min_coverage_count: int | None,
    entity_mention_probability: float | None,
    openrouter_config: OpenRouterConfig,
) -> list[dict[str, str]]:
    prng = rng or random.Random()
    records = preset.generate_records(
        n,
        prng,
        mode=mode,
        difficulty=difficulty,
        min_coverage_count=min_coverage_count,
    )
    return _build_rows_from_records(
        preset,
        records,
        prng,
        difficulty=difficulty,
        entity_mention_probability=entity_mention_probability,
        openrouter_config=openrouter_config,
    )


def _build_rows_from_records(
    preset: StructuredDatasetPreset,
    records: list[dict[str, str]],
    prng: random.Random,
    *,
    difficulty: str,
    entity_mention_probability: float | None,
    openrouter_config: OpenRouterConfig,
) -> list[dict[str, str]]:
    row_payloads = [
        preset.build_row_payload(
            record,
            row_id,
            prng,
            difficulty,
            entity_mention_probability,
        )
        for row_id, record in enumerate(records)
    ]
    texts = generate_texts_with_openrouter(
        preset,
        records,
        row_payloads,
        prng,
        difficulty,
        config=openrouter_config,
    )

    rows: list[dict[str, str]] = []
    for index, record in enumerate(records):
        row = {preset.text_column_name: texts[index]}
        for column in preset.schema.keys():
            row[column] = record[column]
        rows.append(row)
    return rows


def generate_dataset_rows_for_coverage_targets(
    preset: StructuredDatasetPreset,
    coverage_targets: list[str],
    rng: random.Random | None = None,
    *,
    difficulty: str,
    entity_mention_probability: float | None,
    openrouter_config: OpenRouterConfig,
) -> list[dict[str, str]]:
    prng = rng or random.Random()
    records = preset.generate_records_for_coverage_targets(
        coverage_targets,
        prng,
        difficulty=difficulty,
    )
    return _build_rows_from_records(
        preset,
        records,
        prng,
        difficulty=difficulty,
        entity_mention_probability=entity_mention_probability,
        openrouter_config=openrouter_config,
    )


def write_dataset_csv(
    preset: StructuredDatasetPreset,
    path: str,
    rows: list[dict[str, str]],
) -> None:
    expected_columns = set(preset.output_columns)
    for row in rows:
        if set(row.keys()) != expected_columns:
            raise ValueError(f"Row keys must exactly match output columns: {sorted(row.keys())}")
        preset.validate_record({field: row[field] for field in preset.schema.keys()})
    write_csv(path, rows, preset.output_columns)


def run_generation(
    preset: StructuredDatasetPreset,
    *,
    rows: int,
    output: str,
    mode: str,
    difficulty: str,
    min_coverage_count: int | None,
    schema_audit: str,
    summary_path: str | None,
    entity_mention_probability: float | None,
    openrouter_config: OpenRouterConfig,
    seed: int,
) -> int:
    schema_support = preset.audit_schema_support()
    handle_schema_support_report(schema_support, schema_audit)
    rng = random.Random(seed)
    generated_rows = generate_dataset_rows(
        preset,
        rows,
        rng,
        mode=mode,
        difficulty=difficulty,
        min_coverage_count=min_coverage_count,
        entity_mention_probability=entity_mention_probability,
        openrouter_config=openrouter_config,
    )
    write_dataset_csv(preset, output, generated_rows)
    summary_text = preset.build_summary(
        generated_rows,
        mode=mode,
        difficulty=difficulty,
        schema_support_report=schema_support,
        entity_mention_probability=entity_mention_probability,
        openrouter_model=openrouter_config.model,
    )
    resolved_summary_path = summary_path or default_summary_path(output)
    write_text(resolved_summary_path, summary_text)
    print(f"Wrote {len(generated_rows)} rows to {output}")
    print(f"Wrote summary report to {resolved_summary_path}")
    return 0
