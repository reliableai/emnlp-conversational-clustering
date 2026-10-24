"""Shared delegation helpers for structured dataset preset implementations."""

from __future__ import annotations

from typing import Any


class DelegatingStructuredDatasetPreset:
    sampling_module: Any
    text_generation_module: Any
    summary_probability_kwarg = "entity_mention_probability"
    text_column_name = "text"
    schema: dict[str, dict[str, Any]]

    def audit_schema_support(self) -> dict[str, list[str]]:
        return self.sampling_module.audit_schema_support(self.schema)

    def build_prompt(self, batch_rows: list[dict[str, Any]], difficulty: str) -> str:
        return self.text_generation_module.build_openrouter_batch_prompt(batch_rows, difficulty)

    def collect_alignment_failures(
        self,
        records: list[dict[str, str]],
        parsed: dict[int, str],
    ) -> tuple[dict[int, ValueError], dict[int, ValueError]]:
        return self.text_generation_module.collect_alignment_failures(records, parsed)

    def build_alignment_retry_prefix(
        self,
        records: list[dict[str, str]],
        parsed: dict[int, str],
    ) -> str | None:
        return self.text_generation_module.build_alignment_retry_prefix(records, parsed)

    def build_evaluation_row_payload(self, row_id: int, row: dict[str, str]) -> dict[str, Any]:
        return self.text_generation_module.build_evaluation_row_payload(row_id, row)

    def build_evaluation_prompt(self, batch_rows: list[dict[str, Any]]) -> str:
        return self.text_generation_module.build_openrouter_evaluation_prompt(batch_rows)

    def validate_record(self, record: dict[str, str]) -> None:
        self.sampling_module.validate_record(record)

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
        summary_kwargs = {
            "mode": mode,
            "difficulty": difficulty,
            "schema_support_report": schema_support_report,
            "openrouter_model": openrouter_model,
        }
        summary_kwargs[self.summary_probability_kwarg] = entity_mention_probability
        return self.sampling_module.build_dataset_summary(rows, **summary_kwargs)
