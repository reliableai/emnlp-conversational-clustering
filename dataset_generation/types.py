"""Typed models and preset protocol for structured dataset generation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

Record: TypeAlias = dict[str, str]
SchemaDefinition: TypeAlias = dict[str, dict[str, Any]]
EvaluationRating: TypeAlias = Literal["pass", "need_attention", "not_pass"]


class OpenRouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float
    batch_size: int
    max_retries: int

    @field_validator("temperature", "batch_size", "max_retries", mode="before")
    @classmethod
    def _reject_boolean_numeric_values(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("boolean values are not allowed")
        return value

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0 or value > 2.0:
            raise ValueError("'temperature' must be a number between 0.0 and 2.0")
        return value

    @field_validator("batch_size")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @field_validator("max_retries")
    @classmethod
    def _validate_non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be non-negative")
        return value


class EvaluationBridgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_rounds: int

    @field_validator("max_rounds", mode="before")
    @classmethod
    def _reject_boolean_round_values(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("boolean values are not allowed")
        return value

    @field_validator("max_rounds")
    @classmethod
    def _validate_positive_rounds(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value


@dataclass(frozen=True)
class DeterministicEvaluationResult:
    rating: EvaluationRating
    reasons: tuple[str, ...]
    details: dict[str, str] = field(default_factory=dict)


class GeneratedTextItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_id: int
    text: str

    @field_validator("text")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        normalized = " ".join(value.split()).strip()
        if not normalized:
            raise ValueError("text must be a non-empty string.")
        return normalized


class GeneratedTextBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[GeneratedTextItem] = Field(default_factory=list)


class StructuredDatasetPreset(Protocol):
    name: str
    default_rows: int
    default_output_path: str
    default_mode: str
    default_difficulty: str
    default_evaluate_and_regenerate: bool
    schema: SchemaDefinition
    text_column_name: str
    output_columns: tuple[str, ...]
    system_prompt: str
    evaluation_system_prompt: str
    difficulty_choices: tuple[str, ...]
    mode_choices: tuple[str, ...]
    coverage_field: str

    def audit_schema_support(self) -> dict[str, list[str]]: ...

    def generate_records(
        self,
        n: int,
        rng: random.Random | None = None,
        *,
        mode: str,
        difficulty: str,
        min_coverage_count: int | None,
    ) -> list[Record]: ...

    def reachable_coverage_targets(self) -> list[str]: ...

    def generate_records_for_coverage_targets(
        self,
        coverage_targets: list[str],
        rng: random.Random | None = None,
        *,
        difficulty: str,
    ) -> list[Record]: ...

    def build_row_payload(
        self,
        record: Record,
        row_id: int,
        rng: random.Random,
        difficulty: str,
        entity_mention_probability: float | None,
    ) -> dict[str, Any]: ...

    def build_prompt(self, batch_rows: list[dict[str, Any]], difficulty: str) -> str: ...

    def collect_alignment_failures(
        self,
        records: list[Record],
        parsed: dict[int, str],
    ) -> tuple[dict[int, ValueError], dict[int, ValueError]]: ...

    def build_alignment_retry_prefix(
        self,
        records: list[Record],
        parsed: dict[int, str],
    ) -> str | None: ...

    def repair_generated_text(self, record: Record, text: str) -> str: ...

    def build_emergency_fallback_text(self, record: Record) -> str: ...

    def build_evaluation_row_payload(self, row_id: int, row: dict[str, str]) -> dict[str, Any]: ...

    def build_evaluation_prompt(self, batch_rows: list[dict[str, Any]]) -> str: ...

    def evaluate_row_deterministically(
        self,
        record: Record,
        text: str,
    ) -> DeterministicEvaluationResult: ...

    def validate_record(self, record: Record) -> None: ...

    def build_summary(
        self,
        rows: list[dict[str, str]],
        *,
        mode: str,
        difficulty: str,
        schema_support_report: dict[str, list[str]],
        entity_mention_probability: float | None = None,
        openrouter_model: str | None = None,
    ) -> str: ...
