"""Preset-driven dataset evaluation with deterministic and LLM checks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .io import default_summary_path, write_csv, write_text
from .presets import PRESETS
from .progress import render_generation_progress, write_generation_progress
from .types import EvaluationRating, OpenRouterConfig, StructuredDatasetPreset

EVALUATION_COLUMNS = (
    "evaluation_row_id",
    "evaluation_final_rating",
    "evaluation_llm_rating",
    "evaluation_deterministic_rating",
    "evaluation_llm_reason",
    "evaluation_deterministic_reasons",
    "evaluation_deterministic_details",
)
RATING_RANK: dict[EvaluationRating, int] = {"pass": 0, "need_attention": 1, "not_pass": 2}


class RowEvaluationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_id: int
    llm_rating: EvaluationRating
    llm_reason: str

    @field_validator("llm_reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        normalized = " ".join(value.split()).strip()
        if not normalized:
            raise ValueError("llm_reason must be non-empty.")
        return normalized


class RowEvaluationBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[RowEvaluationItem] = Field(default_factory=list)


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


def worse_rating(*ratings: EvaluationRating) -> EvaluationRating:
    return max(ratings, key=RATING_RANK.__getitem__)


def evaluation_response_schema() -> dict[str, Any]:
    return json.loads(json.dumps(RowEvaluationBatch.model_json_schema()))


def extract_structured_output_payload(response: object) -> RowEvaluationBatch:
    if isinstance(response, RowEvaluationBatch):
        return response
    if isinstance(response, dict):
        response_dict = cast(dict[str, object], response)
        parsing_error = response_dict.get("parsing_error")
        if parsing_error is not None:
            if isinstance(parsing_error, BaseException):
                raise ValueError("OpenRouter structured output parsing failed.") from parsing_error
            raise ValueError("OpenRouter structured output parsing failed.")
        parsed = response_dict.get("parsed")
        if isinstance(parsed, RowEvaluationBatch):
            return parsed
        raise TypeError(
            "OpenRouter structured output response did not include a parsed RowEvaluationBatch."
        )
    raise TypeError(
        "Expected OpenRouter structured output response to be "
        "a RowEvaluationBatch or response dict."
    )


def parse_row_evaluation_batch(
    payload: RowEvaluationBatch,
    expected_row_ids: set[int],
) -> dict[int, RowEvaluationItem]:
    parsed: dict[int, RowEvaluationItem] = {}
    for item in payload.items:
        if item.row_id in parsed:
            raise ValueError(f"Duplicate row_id in OpenRouter response: {item.row_id}")
        parsed[item.row_id] = item

    if set(parsed.keys()) != expected_row_ids:
        raise ValueError(
            "OpenRouter response row_ids do not match the requested batch. "
            f"expected={sorted(expected_row_ids)}, got={sorted(parsed.keys())}"
        )
    return parsed


def _fallback_llm_result(row_id: int, reason: str) -> RowEvaluationItem:
    return RowEvaluationItem(
        row_id=row_id,
        llm_rating="need_attention",
        llm_reason=reason,
    )


def evaluate_rows_with_openrouter(
    preset: StructuredDatasetPreset,
    row_payloads: list[dict[str, Any]],
    *,
    config: OpenRouterConfig,
) -> dict[int, RowEvaluationItem]:
    try:
        from dotenv import load_dotenv
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openrouter import ChatOpenRouter
        from pydantic import SecretStr
    except ImportError as exc:
        raise RuntimeError(
            "OpenRouter row evaluation requires python-dotenv, langchain-core, "
            "langchain-openrouter, and pydantic."
        ) from exc

    load_dotenv()
    api_key_env = os.getenv("OPENROUTER_API_KEY")
    if not api_key_env:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    llm = ChatOpenRouter(
        openrouter_api_base="https://openrouter.ai/api/v1",
        openrouter_api_key=SecretStr(api_key_env),
        model=config.model,
        temperature=config.temperature,
        max_retries=0,
    )
    structured_llm = llm.with_structured_output(
        RowEvaluationBatch,
        strict=True,
        include_raw=True,
    )

    if not row_payloads:
        return {}

    total_rows = len(row_payloads)
    batch_total = (total_rows + config.batch_size - 1) // config.batch_size
    completed_rows = 0
    write_generation_progress(
        render_generation_progress(completed_rows, total_rows, status="Starting evaluation")
    )

    results: dict[int, RowEvaluationItem] = {}
    for batch_number, batch_start in enumerate(
        range(0, len(row_payloads), config.batch_size), start=1
    ):
        batch_end = min(batch_start + config.batch_size, len(row_payloads))
        batch_rows = row_payloads[batch_start:batch_end]
        expected_row_ids = {row["row_id"] for row in batch_rows}
        prompt = preset.build_evaluation_prompt(batch_rows)
        retry_prefix: str | None = None
        last_error: Exception | None = None
        parsed_batch: dict[int, RowEvaluationItem] | None = None

        for attempt in range(config.max_retries + 1):
            status = "Evaluating" if attempt == 0 else f"Retrying ({attempt}/{config.max_retries})"
            write_generation_progress(
                render_generation_progress(
                    completed_rows,
                    total_rows,
                    status=status,
                    batch_index=batch_number,
                    batch_total=batch_total,
                )
            )
            attempt_prompt = prompt if retry_prefix is None else retry_prefix + prompt
            try:
                response = structured_llm.invoke(
                    [
                        SystemMessage(content=preset.evaluation_system_prompt),
                        HumanMessage(content=attempt_prompt),
                    ]
                )
                payload = extract_structured_output_payload(response)
                parsed_batch = parse_row_evaluation_batch(payload, expected_row_ids)
                break
            except Exception as exc:
                last_error = exc
                retry_prefix = (
                    "The previous response did not match the required schema exactly. "
                    "Retry and return valid JSON only.\n\n"
                )

        if parsed_batch is None:
            fallback_reason = (
                "llm_evaluation_unavailable_after_retries"
                if last_error is not None
                else "llm_evaluation_unavailable"
            )
            for row_id in sorted(expected_row_ids):
                results[row_id] = _fallback_llm_result(row_id, fallback_reason)
            completed_rows = batch_end
            write_generation_progress(
                render_generation_progress(
                    completed_rows,
                    total_rows,
                    status="Evaluated with fallback",
                    batch_index=batch_number,
                    batch_total=batch_total,
                ),
                final=completed_rows == total_rows,
            )
            continue

        results.update(parsed_batch)
        completed_rows = batch_end
        write_generation_progress(
            render_generation_progress(
                completed_rows,
                total_rows,
                status="Evaluated",
                batch_index=batch_number,
                batch_total=batch_total,
            ),
            final=completed_rows == total_rows,
        )

    return results


def _required_input_columns(preset: StructuredDatasetPreset) -> tuple[str, ...]:
    return (preset.text_column_name, *tuple(preset.schema.keys()))


def read_dataset_rows(
    preset: StructuredDatasetPreset,
    path: str,
) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        required_columns = _required_input_columns(preset)
        missing_columns = [column for column in required_columns if column not in columns]
        if missing_columns:
            raise ValueError(
                f"Input dataset is missing required columns: {', '.join(sorted(missing_columns))}"
            )
        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError("Input dataset contains no rows.")

    return rows


def evaluate_dataset_rows(
    preset: StructuredDatasetPreset,
    rows: list[dict[str, str]],
    *,
    config: OpenRouterConfig,
) -> list[dict[str, str]]:
    payloads = [preset.build_evaluation_row_payload(row_id, row) for row_id, row in enumerate(rows)]
    llm_results = evaluate_rows_with_openrouter(preset, payloads, config=config)

    evaluated_rows: list[dict[str, str]] = []
    for row_id, row in enumerate(rows):
        record = {field: row[field] for field in preset.schema.keys()}
        deterministic = preset.evaluate_row_deterministically(record, row[preset.text_column_name])
        llm_result = llm_results.get(
            row_id,
            _fallback_llm_result(row_id, "llm_evaluation_missing_row_result"),
        )
        final_rating = worse_rating(deterministic.rating, llm_result.llm_rating)
        deterministic_reasons = ", ".join(deterministic.reasons) if deterministic.reasons else ""

        evaluated_row = dict(row)
        evaluated_row["evaluation_row_id"] = str(row_id)
        evaluated_row["evaluation_final_rating"] = final_rating
        evaluated_row["evaluation_llm_rating"] = llm_result.llm_rating
        evaluated_row["evaluation_deterministic_rating"] = deterministic.rating
        evaluated_row["evaluation_llm_reason"] = llm_result.llm_reason
        evaluated_row["evaluation_deterministic_reasons"] = deterministic_reasons
        evaluated_row["evaluation_deterministic_details"] = json.dumps(
            deterministic.details,
            sort_keys=True,
        )
        evaluated_rows.append(evaluated_row)

    return evaluated_rows


def build_evaluation_summary(
    evaluated_rows: list[dict[str, str]],
    *,
    preset_name: str,
    input_path: str,
    output_path: str,
    config: OpenRouterConfig,
) -> str:
    final_rating_counts = Counter(row["evaluation_final_rating"] for row in evaluated_rows)
    llm_rating_counts = Counter(row["evaluation_llm_rating"] for row in evaluated_rows)
    deterministic_rating_counts = Counter(
        row["evaluation_deterministic_rating"] for row in evaluated_rows
    )
    deterministic_reason_counts: Counter[str] = Counter()
    for row in evaluated_rows:
        reasons = [reason.strip() for reason in row["evaluation_deterministic_reasons"].split(",")]
        deterministic_reason_counts.update(reason for reason in reasons if reason)

    lines = [
        "Structured dataset evaluation summary",
        f"Preset: {preset_name}",
        "",
        f"Input path: {input_path}",
        f"Output path: {output_path}",
        f"Rows evaluated: {len(evaluated_rows)}",
        f"Model: {config.model}",
        f"Temperature: {config.temperature}",
        f"Batch size: {config.batch_size}",
        f"Max retries: {config.max_retries}",
        "",
        "Final rating counts:",
    ]
    for rating in ("pass", "need_attention", "not_pass"):
        lines.append(f"  {rating}: {final_rating_counts.get(rating, 0)}")

    lines.extend(["", "LLM rating counts:"])
    for rating in ("pass", "need_attention", "not_pass"):
        lines.append(f"  {rating}: {llm_rating_counts.get(rating, 0)}")

    lines.extend(["", "Deterministic rating counts:"])
    for rating in ("pass", "need_attention", "not_pass"):
        lines.append(f"  {rating}: {deterministic_rating_counts.get(rating, 0)}")

    lines.extend(["", "Top deterministic reasons:"])
    if not deterministic_reason_counts:
        lines.append("  (none)")
    else:
        for reason, count in deterministic_reason_counts.most_common(12):
            lines.append(f"  {reason}: {count}")

    return "\n".join(lines).rstrip() + "\n"


def write_evaluation_csv(path: str, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Cannot write evaluation CSV for an empty row set.")
    fieldnames = tuple(rows[0].keys())
    write_csv(path, rows, fieldnames)


def _default_evaluation_output_path(input_path: str) -> str:
    path = Path(input_path)
    if path.suffix:
        return str(path.with_name(f"{path.stem}_evaluation{path.suffix}"))
    return f"{input_path}_evaluation.csv"


def _build_bootstrap_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        required=True,
        help="Registered dataset preset to evaluate.",
    )
    return parser


def build_parser(
    default_preset: str | None = None,
    *,
    selected_preset: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a structured synthetic dataset with deterministic and LLM checks."
    )
    active_preset_name = default_preset or selected_preset
    preset = PRESETS[active_preset_name] if active_preset_name is not None else None
    if default_preset is None:
        parser.add_argument(
            "--preset",
            choices=sorted(PRESETS),
            required=True,
            help="Registered dataset preset to evaluate.",
        )
    input_default = preset.default_output_path if preset is not None else "data/structured.csv"
    output_default = _default_evaluation_output_path(input_default)
    parser.add_argument(
        "--input",
        default=input_default,
        help="Input dataset CSV path.",
    )
    parser.add_argument(
        "--output",
        default=output_default,
        help="Output CSV path containing per-row evaluation ratings.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional summary output path.",
    )
    parser.add_argument(
        "--model",
        default="google/gemini-3-flash-preview",
        help="OpenRouter model name used for row evaluation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="OpenRouter sampling temperature for evaluation responses.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Rows per OpenRouter evaluation request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum additional retries after the initial OpenRouter attempt.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    default_preset: str | None = None,
) -> int:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    preset_name = default_preset
    if preset_name is None:
        if argv_list is not None and any(flag in argv_list for flag in ("-h", "--help")):
            parser = build_parser()
            parser.parse_args(argv_list)
            return 0
        bootstrap_parser = _build_bootstrap_parser()
        bootstrap_args, _ = bootstrap_parser.parse_known_args(argv_list)
        preset_name = bootstrap_args.preset
    parser = build_parser(default_preset, selected_preset=preset_name)
    args = parser.parse_args(argv_list)
    preset_name = default_preset or args.preset
    preset = PRESETS[preset_name]
    try:
        config = OpenRouterConfig(
            model=args.model,
            temperature=args.temperature,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
        )
    except ValidationError as exc:
        parser.error(exc.errors()[0]["msg"])

    rows = read_dataset_rows(preset, args.input)
    evaluated_rows = evaluate_dataset_rows(preset, rows, config=config)
    write_evaluation_csv(args.output, evaluated_rows)
    summary_path = args.summary_path or default_summary_path(args.output)
    summary_text = build_evaluation_summary(
        evaluated_rows,
        preset_name=preset.name,
        input_path=args.input,
        output_path=args.output,
        config=config,
    )
    write_text(summary_path, summary_text)
    print(f"Wrote {len(evaluated_rows)} evaluation rows to {args.output}")
    print(f"Wrote evaluation summary report to {summary_path}")
    return 0


def run_structured_dataset_evaluation_cli() -> int:
    return main()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
