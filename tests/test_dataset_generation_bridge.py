"""Tests for round-based generation and evaluation bridging."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import pytest

from dataset_generation.evaluation_bridge import (
    EvaluationBridgeRoundResult,
    EvaluationBridgeRunResult,
    build_evaluation_bridge_summary,
    generate_dataset_rows_with_evaluation_bridge,
    run_generation_with_evaluation_bridge,
)
from dataset_generation.types import (
    DeterministicEvaluationResult,
    EvaluationBridgeConfig,
    OpenRouterConfig,
    StructuredDatasetPreset,
)


class _StubPreset(StructuredDatasetPreset):
    name = "stub"
    default_rows = 1
    default_output_path = "unused.csv"
    default_mode = "balanced"
    default_difficulty = "medium"
    default_evaluate_and_regenerate = False
    schema = {"label": {"type": "string"}, "issue_type": {"type": "string"}}
    output_columns = ("text", "label", "issue_type")
    system_prompt = "stub system prompt"
    evaluation_system_prompt = "stub evaluation system prompt"
    difficulty_choices = ("medium",)
    mode_choices = ("balanced",)
    coverage_field = "issue_type"

    def audit_schema_support(self) -> dict[str, list[str]]:
        return {}

    def generate_records(
        self,
        n: int,
        rng: random.Random | None = None,
        *,
        mode: str,
        difficulty: str,
        min_coverage_count: int | None,
    ) -> list[dict[str, str]]:
        _ = rng, mode, difficulty, min_coverage_count
        return [{"label": f"record-{row_id}", "issue_type": "type-a"} for row_id in range(n)]

    def reachable_coverage_targets(self) -> list[str]:
        return ["type-a", "type-b"]

    def generate_records_for_coverage_targets(
        self,
        coverage_targets: list[str],
        rng: random.Random | None = None,
        *,
        difficulty: str,
    ) -> list[dict[str, str]]:
        _ = rng, difficulty
        return [
            {"label": coverage_target, "issue_type": coverage_target}
            for coverage_target in coverage_targets
        ]

    def build_row_payload(
        self,
        record: dict[str, str],
        row_id: int,
        rng: random.Random,
        difficulty: str,
        entity_mention_probability: float | None,
    ) -> dict[str, Any]:
        _ = rng, difficulty, entity_mention_probability
        return {"row_id": row_id, "label": record["label"]}

    def build_prompt(self, batch_rows: list[dict[str, Any]], difficulty: str) -> str:
        _ = difficulty
        return str(batch_rows)

    def collect_alignment_failures(
        self,
        records: list[dict[str, str]],
        parsed: dict[int, str],
    ) -> tuple[dict[int, ValueError], dict[int, ValueError]]:
        _ = records, parsed
        return {}, {}

    def build_alignment_retry_prefix(
        self,
        records: list[dict[str, str]],
        parsed: dict[int, str],
    ) -> str | None:
        _ = records, parsed
        return None

    def repair_generated_text(self, record: dict[str, str], text: str) -> str:
        _ = record
        return text

    def build_emergency_fallback_text(self, record: dict[str, str]) -> str:
        return f"fallback:{record['label']}"

    def build_evaluation_row_payload(self, row_id: int, row: dict[str, str]) -> dict[str, Any]:
        return {"row_id": row_id, "text": row["text"], "metadata": row}

    def build_evaluation_prompt(self, batch_rows: list[dict[str, Any]]) -> str:
        return str(batch_rows)

    def evaluate_row_deterministically(
        self,
        record: dict[str, str],
        text: str,
    ) -> DeterministicEvaluationResult:
        _ = record, text
        return DeterministicEvaluationResult(rating="pass", reasons=(), details={})

    def validate_record(self, record: dict[str, str]) -> None:
        _ = record

    def build_summary(
        self,
        rows: list[dict[str, str]],
        *,
        mode: str,
        difficulty: str,
        schema_support_report: dict[str, list[str]],
        entity_mention_probability: float | None = None,
        openrouter_model: str | None = None,
    ) -> str:
        _ = mode, difficulty, schema_support_report, entity_mention_probability, openrouter_model
        return f"Dataset summary\nRows: {len(rows)}\n"


def _openrouter_config(*, model: str = "demo/model") -> OpenRouterConfig:
    return OpenRouterConfig(
        model=model,
        temperature=0.0,
        batch_size=10,
        max_retries=0,
    )


def _evaluated_row(
    text: str,
    label: str,
    issue_type: str,
    final_rating: str,
    *,
    deterministic_reasons: str = "",
) -> dict[str, str]:
    return {
        "text": text,
        "label": label,
        "issue_type": issue_type,
        "evaluation_final_rating": final_rating,
        "evaluation_llm_rating": final_rating,
        "evaluation_deterministic_rating": final_rating,
        "evaluation_llm_reason": "ok" if final_rating == "pass" else "review",
        "evaluation_deterministic_reasons": deterministic_reasons,
        "evaluation_deterministic_details": "{}",
    }


def test_generate_dataset_rows_with_evaluation_bridge_regenerates_rejected_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_batches = [
        [
            {"text": "row-a", "label": "a", "issue_type": "type-a"},
            {"text": "row-b", "label": "b", "issue_type": "type-a"},
            {"text": "row-c", "label": "c", "issue_type": "type-a"},
        ],
        [
            {"text": "row-d", "label": "d", "issue_type": "type-a"},
            {"text": "row-e", "label": "e", "issue_type": "type-b"},
        ],
    ]
    evaluated_batches = [
        [
            _evaluated_row("row-a", "a", "type-a", "pass"),
            _evaluated_row(
                "row-b",
                "b",
                "type-a",
                "need_attention",
                deterministic_reasons="very_short_text",
            ),
            _evaluated_row(
                "row-c",
                "c",
                "type-a",
                "not_pass",
                deterministic_reasons="issue_type_mismatch",
            ),
        ],
        [
            _evaluated_row("row-d", "d", "type-a", "pass"),
            _evaluated_row("row-e", "e", "type-b", "pass"),
        ],
    ]

    monkeypatch.setattr(
        "dataset_generation.evaluation_bridge.generate_dataset_rows",
        lambda *args, **kwargs: generated_batches.pop(0),
    )
    monkeypatch.setattr(
        "dataset_generation.evaluation_bridge.evaluate_dataset_rows",
        lambda *args, **kwargs: evaluated_batches.pop(0),
    )

    result = generate_dataset_rows_with_evaluation_bridge(
        _StubPreset(),
        3,
        random.Random(7),
        mode="balanced",
        difficulty="medium",
        min_coverage_count=None,
        entity_mention_probability=None,
        openrouter_config=_openrouter_config(model="demo/generator"),
        evaluation_openrouter_config=_openrouter_config(model="demo/evaluator"),
        bridge_config=EvaluationBridgeConfig(max_rounds=4),
    )

    assert result.accepted_rows == [
        {"text": "row-a", "label": "a", "issue_type": "type-a"},
        {"text": "row-d", "label": "d", "issue_type": "type-a"},
        {"text": "row-e", "label": "e", "issue_type": "type-b"},
    ]
    assert [(row.round_index, row.requested_count) for row in result.round_results] == [
        (1, 3),
        (2, 2),
    ]
    assert result.total_need_attention_count == 1
    assert result.total_not_pass_count == 1


def test_generate_dataset_rows_with_evaluation_bridge_refills_missing_coverage_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_min_coverage_counts: list[int | None] = []
    captured_coverage_targets: list[list[str]] = []
    generated_batches = [
        [
            {"text": "row-a", "label": "a", "issue_type": "type-a"},
            {"text": "row-b", "label": "b", "issue_type": "type-b"},
        ]
    ]
    targeted_batches = [[{"text": "row-c", "label": "c", "issue_type": "type-b"}]]
    evaluated_batches = [
        [
            _evaluated_row("row-a", "a", "type-a", "pass"),
            _evaluated_row(
                "row-b",
                "b",
                "type-b",
                "not_pass",
                deterministic_reasons="issue_type_mismatch",
            ),
        ],
        [_evaluated_row("row-c", "c", "type-b", "pass")],
    ]

    def _fake_generate_rows(*args: Any, **kwargs: Any) -> list[dict[str, str]]:
        captured_min_coverage_counts.append(kwargs["min_coverage_count"])
        return generated_batches.pop(0)

    def _fake_generate_rows_for_coverage_targets(
        *args: Any,
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        _ = kwargs
        captured_coverage_targets.append(list(args[1]))
        return targeted_batches.pop(0)

    monkeypatch.setattr(
        "dataset_generation.evaluation_bridge.generate_dataset_rows",
        _fake_generate_rows,
    )
    monkeypatch.setattr(
        "dataset_generation.evaluation_bridge.generate_dataset_rows_for_coverage_targets",
        _fake_generate_rows_for_coverage_targets,
    )
    monkeypatch.setattr(
        "dataset_generation.evaluation_bridge.evaluate_dataset_rows",
        lambda *args, **kwargs: evaluated_batches.pop(0),
    )

    result = generate_dataset_rows_with_evaluation_bridge(
        _StubPreset(),
        2,
        random.Random(7),
        mode="balanced",
        difficulty="medium",
        min_coverage_count=1,
        entity_mention_probability=None,
        openrouter_config=_openrouter_config(),
        evaluation_openrouter_config=_openrouter_config(model="demo/evaluator"),
        bridge_config=EvaluationBridgeConfig(max_rounds=3),
    )

    assert result.total_accepted_count == 2
    assert captured_min_coverage_counts == [1]
    assert captured_coverage_targets == [["type-b"]]


def test_generate_dataset_rows_with_evaluation_bridge_tracks_actual_generated_rows_per_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_requested_counts: list[int] = []
    captured_coverage_targets: list[list[str]] = []
    generated_batches = [
        [
            {"text": "row-a", "label": "a", "issue_type": "type-a"},
            {"text": "row-b", "label": "b", "issue_type": "type-a"},
            {"text": "row-c", "label": "c", "issue_type": "type-a"},
            {"text": "row-d", "label": "d", "issue_type": "type-a"},
        ],
        [
            {"text": "row-f", "label": "f", "issue_type": "type-a"},
            {"text": "row-g", "label": "g", "issue_type": "type-b"},
        ],
    ]
    targeted_batches = [[{"text": "row-e", "label": "e", "issue_type": "type-b"}]]
    evaluated_batches = [
        [
            _evaluated_row("row-a", "a", "type-a", "pass"),
            _evaluated_row(
                "row-b", "b", "type-a", "not_pass", deterministic_reasons="issue_type_mismatch"
            ),
            _evaluated_row(
                "row-c", "c", "type-a", "not_pass", deterministic_reasons="issue_type_mismatch"
            ),
            _evaluated_row(
                "row-d", "d", "type-a", "not_pass", deterministic_reasons="issue_type_mismatch"
            ),
        ],
        [_evaluated_row("row-e", "e", "type-b", "pass")],
        [
            _evaluated_row("row-f", "f", "type-a", "pass"),
            _evaluated_row("row-g", "g", "type-b", "pass"),
        ],
    ]

    def _fake_generate_rows(*args: Any, **kwargs: Any) -> list[dict[str, str]]:
        captured_requested_counts.append(kwargs["n"] if "n" in kwargs else args[1])
        return generated_batches.pop(0)

    def _fake_generate_rows_for_coverage_targets(
        *args: Any,
        **kwargs: Any,
    ) -> list[dict[str, str]]:
        _ = kwargs
        captured_coverage_targets.append(list(args[1]))
        return targeted_batches.pop(0)

    monkeypatch.setattr(
        "dataset_generation.evaluation_bridge.generate_dataset_rows",
        _fake_generate_rows,
    )
    monkeypatch.setattr(
        "dataset_generation.evaluation_bridge.generate_dataset_rows_for_coverage_targets",
        _fake_generate_rows_for_coverage_targets,
    )
    monkeypatch.setattr(
        "dataset_generation.evaluation_bridge.evaluate_dataset_rows",
        lambda *args, **kwargs: evaluated_batches.pop(0),
    )

    result = generate_dataset_rows_with_evaluation_bridge(
        _StubPreset(),
        4,
        random.Random(7),
        mode="balanced",
        difficulty="medium",
        min_coverage_count=1,
        entity_mention_probability=None,
        openrouter_config=_openrouter_config(),
        evaluation_openrouter_config=_openrouter_config(model="demo/evaluator"),
        bridge_config=EvaluationBridgeConfig(max_rounds=4),
    )

    assert captured_requested_counts == [4, 2]
    assert captured_coverage_targets == [["type-b"]]
    assert [
        (row.round_index, row.requested_count, row.generated_count, row.accepted_count)
        for row in result.round_results
    ] == [
        (1, 4, 4, 1),
        (2, 3, 1, 1),
        (3, 2, 2, 2),
    ]
    assert result.total_generated_count == 7


def test_generate_dataset_rows_with_evaluation_bridge_raises_when_missing_coverage_overflows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targeted_calls: list[list[str]] = []
    generated_batches = [
        [
            {"text": "row-a", "label": "a", "issue_type": "type-a"},
            {"text": "row-b", "label": "b", "issue_type": "type-a"},
            {"text": "row-c", "label": "c", "issue_type": "type-a"},
            {"text": "row-d", "label": "d", "issue_type": "type-a"},
        ]
    ]
    evaluated_batches = [
        [
            _evaluated_row("row-a", "a", "type-a", "pass"),
            _evaluated_row("row-b", "b", "type-a", "pass"),
            _evaluated_row("row-c", "c", "type-a", "pass"),
            _evaluated_row(
                "row-d",
                "d",
                "type-a",
                "not_pass",
                deterministic_reasons="issue_type_mismatch",
            ),
        ]
    ]

    monkeypatch.setattr(
        "dataset_generation.evaluation_bridge.generate_dataset_rows",
        lambda *args, **kwargs: generated_batches.pop(0),
    )
    monkeypatch.setattr(
        "dataset_generation.evaluation_bridge.generate_dataset_rows_for_coverage_targets",
        lambda *args, **kwargs: targeted_calls.append(list(args[1])) or [],
    )
    monkeypatch.setattr(
        "dataset_generation.evaluation_bridge.evaluate_dataset_rows",
        lambda *args, **kwargs: evaluated_batches.pop(0),
    )

    with pytest.raises(
        RuntimeError,
        match="cannot satisfy min_coverage_count without exceeding the requested row count",
    ):
        generate_dataset_rows_with_evaluation_bridge(
            _StubPreset(),
            4,
            random.Random(7),
            mode="balanced",
            difficulty="medium",
            min_coverage_count=2,
            entity_mention_probability=None,
            openrouter_config=_openrouter_config(),
            evaluation_openrouter_config=_openrouter_config(model="demo/evaluator"),
            bridge_config=EvaluationBridgeConfig(max_rounds=3),
        )

    assert targeted_calls == []


def test_run_generation_with_evaluation_bridge_writes_dataset_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_rows = [
        {"text": "row-a", "label": "a", "issue_type": "type-a"},
        {"text": "row-b", "label": "b", "issue_type": "type-b"},
    ]

    monkeypatch.setattr(
        "dataset_generation.evaluation_bridge.generate_dataset_rows_with_evaluation_bridge",
        lambda *args, **kwargs: EvaluationBridgeRunResult(
            target_count=2,
            accepted_rows=result_rows,
            round_results=[
                EvaluationBridgeRoundResult(
                    round_index=1,
                    requested_count=2,
                    generated_count=2,
                    accepted_count=2,
                    need_attention_count=0,
                    not_pass_count=0,
                )
            ],
        ),
    )

    output_path = tmp_path / "output.csv"
    summary_path = tmp_path / "summary.txt"
    exit_code = run_generation_with_evaluation_bridge(
        _StubPreset(),
        rows=2,
        output=str(output_path),
        mode="balanced",
        difficulty="medium",
        min_coverage_count=None,
        schema_audit="warn",
        summary_path=str(summary_path),
        entity_mention_probability=None,
        openrouter_config=_openrouter_config(model="demo/generator"),
        evaluation_openrouter_config=_openrouter_config(model="demo/evaluator"),
        bridge_config=EvaluationBridgeConfig(max_rounds=4),
        seed=7,
    )

    assert exit_code == 0
    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert rows == result_rows
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "Evaluation bridge summary" in summary_text
    assert "Evaluator model: demo/evaluator" in summary_text
    assert "Dataset summary" in summary_text


def test_build_evaluation_bridge_summary_lists_round_counts() -> None:
    summary = build_evaluation_bridge_summary(
        EvaluationBridgeRunResult(
            target_count=4,
            accepted_rows=[
                {"text": "row-a", "label": "a", "issue_type": "type-a"},
                {"text": "row-d", "label": "d", "issue_type": "type-a"},
                {"text": "row-e", "label": "e", "issue_type": "type-b"},
                {"text": "row-f", "label": "f", "issue_type": "type-a"},
            ],
            round_results=[
                EvaluationBridgeRoundResult(
                    round_index=1,
                    requested_count=4,
                    generated_count=4,
                    accepted_count=2,
                    need_attention_count=1,
                    not_pass_count=1,
                ),
                EvaluationBridgeRoundResult(
                    round_index=2,
                    requested_count=2,
                    generated_count=2,
                    accepted_count=2,
                    need_attention_count=0,
                    not_pass_count=0,
                ),
            ],
        ),
        openrouter_config=_openrouter_config(model="demo/generator"),
        evaluation_openrouter_config=_openrouter_config(model="demo/evaluator"),
        bridge_config=EvaluationBridgeConfig(max_rounds=5),
    )

    assert "Requested rows: 4" in summary
    assert "Generation model: demo/generator" in summary
    assert "round=1 requested=4 generated=4 accepted=2 need_attention=1 not_pass=1" in summary
