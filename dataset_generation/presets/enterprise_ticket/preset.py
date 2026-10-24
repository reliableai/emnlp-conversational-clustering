"""Enterprise ticket preset implementation for the generic dataset engine."""

from __future__ import annotations

import random
from typing import Any

from dataset_generation.preset_base import DelegatingStructuredDatasetPreset
from dataset_generation.types import DeterministicEvaluationResult

from . import sampling, spec, text_generation


class EnterpriseTicketPreset(DelegatingStructuredDatasetPreset):
    sampling_module = sampling
    text_generation_module = text_generation
    summary_probability_kwarg = "system_mention_probability"
    text_column_name = "ticket_text"
    name = "enterprise_ticket"
    default_rows = 210
    default_output_path = spec.DEFAULT_OUTPUT_PATH
    default_mode = "balanced"
    default_difficulty = "medium"
    default_evaluate_and_regenerate = True
    schema: dict[str, dict[str, Any]] = spec.SCHEMA
    output_columns = tuple(spec.CSV_COLUMNS)
    system_prompt = spec.OPENROUTER_SYSTEM_PROMPT
    evaluation_system_prompt = (
        "You evaluate enterprise support tickets for metadata alignment and quality. "
        "Return structured JSON only."
    )
    difficulty_choices = ("easy", "medium", "hard")
    mode_choices = ("balanced", "random")
    coverage_field = "issue_type"

    def generate_records(
        self,
        n: int,
        rng: random.Random | None = None,
        *,
        mode: str,
        difficulty: str,
        min_coverage_count: int | None,
    ) -> list[spec.Record]:
        return sampling.generate_records(
            n,
            rng,
            mode=mode,
            difficulty=difficulty,
            min_coverage_target_count=min_coverage_count,
        )

    def reachable_coverage_targets(self) -> list[str]:
        return sampling.reachable_issue_targets()

    def generate_records_for_coverage_targets(
        self,
        coverage_targets: list[str],
        rng: random.Random | None = None,
        *,
        difficulty: str,
    ) -> list[spec.Record]:
        return sampling.generate_records_for_issue_targets(
            coverage_targets,
            rng,
            difficulty=difficulty,
        )

    def build_row_payload(
        self,
        record: spec.Record,
        row_id: int,
        rng: random.Random,
        difficulty: str,
        entity_mention_probability: float | None,
    ) -> dict[str, Any]:
        variation_profile = text_generation.sample_text_variation_profile(
            record,
            rng,
            difficulty,
            system_mention_probability=entity_mention_probability,
        )
        return {
            "row_id": row_id,
            "surface_metadata": text_generation.build_surface_metadata(record),
            "hidden_generation_guidance": text_generation.build_hidden_generation_guidance(
                record,
                rng,
                variation_profile,
            ),
            "text_generation_directives": text_generation.build_llm_generation_directives(
                record,
                rng,
                difficulty,
                system_mention_probability=entity_mention_probability,
                variation_profile=variation_profile,
            ),
        }

    def repair_generated_text(self, record: spec.Record, text: str) -> str:
        return text_generation.repair_ticket_text_alignment(record, text)

    def build_emergency_fallback_text(self, record: spec.Record) -> str:
        return text_generation.build_aligned_fallback_ticket_text(record)

    def evaluate_row_deterministically(
        self,
        record: spec.Record,
        text: str,
    ) -> DeterministicEvaluationResult:
        return text_generation.evaluate_ticket_text_deterministically(record, text)


PRESET = EnterpriseTicketPreset()
