"""Round-based orchestration that filters generated rows through evaluation
until a target size is reached."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from .evaluation import EVALUATION_COLUMNS, evaluate_dataset_rows
from .io import default_summary_path, write_text
from .runtime import (
    generate_dataset_rows,
    generate_dataset_rows_for_coverage_targets,
    handle_schema_support_report,
    write_dataset_csv,
)
from .types import EvaluationBridgeConfig, OpenRouterConfig, StructuredDatasetPreset


@dataclass(frozen=True)
class EvaluationBridgeRoundResult:
    round_index: int
    requested_count: int
    generated_count: int
    accepted_count: int
    need_attention_count: int
    not_pass_count: int


@dataclass(frozen=True)
class EvaluationBridgeRunResult:
    target_count: int
    accepted_rows: list[dict[str, str]]
    round_results: list[EvaluationBridgeRoundResult]

    @property
    def total_generated_count(self) -> int:
        return sum(result.generated_count for result in self.round_results)

    @property
    def total_accepted_count(self) -> int:
        return len(self.accepted_rows)

    @property
    def total_need_attention_count(self) -> int:
        return sum(result.need_attention_count for result in self.round_results)

    @property
    def total_not_pass_count(self) -> int:
        return sum(result.not_pass_count for result in self.round_results)


def _strip_evaluation_columns(row: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in row.items() if key not in EVALUATION_COLUMNS}


def _remaining_coverage_targets(
    preset: StructuredDatasetPreset,
    accepted_rows: list[dict[str, str]],
    min_coverage_count: int | None,
) -> list[str]:
    if min_coverage_count is None:
        return []

    accepted_counts = Counter(row[preset.coverage_field] for row in accepted_rows)
    coverage_targets: list[str] = []
    for coverage_target in preset.reachable_coverage_targets():
        remaining_count = max(0, min_coverage_count - accepted_counts.get(coverage_target, 0))
        coverage_targets.extend([coverage_target] * remaining_count)
    return coverage_targets


def _format_missing_coverage_targets(coverage_targets: list[str]) -> str:
    counts = Counter(coverage_targets)
    return ", ".join(
        f"{coverage_target}={count}" for coverage_target, count in sorted(counts.items())
    )


def generate_dataset_rows_with_evaluation_bridge(
    preset: StructuredDatasetPreset,
    target_count: int,
    rng: random.Random | None = None,
    *,
    mode: str,
    difficulty: str,
    min_coverage_count: int | None,
    entity_mention_probability: float | None,
    openrouter_config: OpenRouterConfig,
    evaluation_openrouter_config: OpenRouterConfig,
    bridge_config: EvaluationBridgeConfig,
) -> EvaluationBridgeRunResult:
    prng = rng or random.Random()
    accepted_rows: list[dict[str, str]] = []
    round_results: list[EvaluationBridgeRoundResult] = []
    remaining_count = target_count

    for round_index in range(1, bridge_config.max_rounds + 1):
        coverage_targets = (
            _remaining_coverage_targets(preset, accepted_rows, min_coverage_count)
            if mode == "balanced" and round_index > 1
            else []
        )
        if coverage_targets:
            if len(coverage_targets) > remaining_count:
                raise RuntimeError(
                    "Evaluation bridge cannot satisfy min_coverage_count without "
                    "exceeding the requested row count: "
                    f"accepted={len(accepted_rows)}, requested={target_count}, "
                    f"remaining={remaining_count}, coverage_field={preset.coverage_field}, "
                    f"missing_coverage_targets={_format_missing_coverage_targets(coverage_targets)}"
                )
            prng.shuffle(coverage_targets)
            generated_rows = generate_dataset_rows_for_coverage_targets(
                preset,
                coverage_targets,
                prng,
                difficulty=difficulty,
                entity_mention_probability=entity_mention_probability,
                openrouter_config=openrouter_config,
            )
        else:
            generation_min_coverage_count = min_coverage_count if round_index == 1 else None
            generated_rows = generate_dataset_rows(
                preset,
                remaining_count,
                prng,
                mode=mode,
                difficulty=difficulty,
                min_coverage_count=generation_min_coverage_count,
                entity_mention_probability=entity_mention_probability,
                openrouter_config=openrouter_config,
            )
        evaluated_rows = evaluate_dataset_rows(
            preset,
            generated_rows,
            config=evaluation_openrouter_config,
        )

        accepted_count = 0
        need_attention_count = 0
        not_pass_count = 0
        for row in evaluated_rows:
            final_rating = row["evaluation_final_rating"]
            if final_rating == "pass":
                accepted_rows.append(_strip_evaluation_columns(row))
                accepted_count += 1
            elif final_rating == "need_attention":
                need_attention_count += 1
            elif final_rating == "not_pass":
                not_pass_count += 1
            else:
                raise ValueError(f"Unsupported evaluation_final_rating: {final_rating}")

        round_results.append(
            EvaluationBridgeRoundResult(
                round_index=round_index,
                requested_count=remaining_count,
                generated_count=len(generated_rows),
                accepted_count=accepted_count,
                need_attention_count=need_attention_count,
                not_pass_count=not_pass_count,
            )
        )
        remaining_count = target_count - len(accepted_rows)
        if remaining_count == 0:
            return EvaluationBridgeRunResult(
                target_count=target_count,
                accepted_rows=accepted_rows,
                round_results=round_results,
            )

    raise RuntimeError(
        "Evaluation bridge exhausted max_rounds before reaching the requested row count: "
        f"accepted={len(accepted_rows)}, requested={target_count}, remaining={remaining_count}"
    )


def build_evaluation_bridge_summary(
    result: EvaluationBridgeRunResult,
    *,
    openrouter_config: OpenRouterConfig,
    evaluation_openrouter_config: OpenRouterConfig,
    bridge_config: EvaluationBridgeConfig,
) -> str:
    lines = [
        "Evaluation bridge summary",
        f"Requested rows: {result.target_count}",
        f"Accepted rows: {result.total_accepted_count}",
        f"Total generated rows: {result.total_generated_count}",
        f"Rejected need_attention rows: {result.total_need_attention_count}",
        f"Rejected not_pass rows: {result.total_not_pass_count}",
        f"Generation model: {openrouter_config.model}",
        f"Evaluator model: {evaluation_openrouter_config.model}",
        f"Generation rounds used: {len(result.round_results)}",
        f"Generation round limit: {bridge_config.max_rounds}",
        "",
        "Round results:",
    ]
    for round_result in result.round_results:
        lines.append(
            "  "
            f"round={round_result.round_index} "
            f"requested={round_result.requested_count} "
            f"generated={round_result.generated_count} "
            f"accepted={round_result.accepted_count} "
            f"need_attention={round_result.need_attention_count} "
            f"not_pass={round_result.not_pass_count}"
        )
    return "\n".join(lines).rstrip() + "\n"


def run_generation_with_evaluation_bridge(
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
    evaluation_openrouter_config: OpenRouterConfig,
    bridge_config: EvaluationBridgeConfig,
    seed: int,
) -> int:
    schema_support = preset.audit_schema_support()
    handle_schema_support_report(schema_support, schema_audit)
    rng = random.Random(seed)
    bridge_result = generate_dataset_rows_with_evaluation_bridge(
        preset,
        rows,
        rng,
        mode=mode,
        difficulty=difficulty,
        min_coverage_count=min_coverage_count,
        entity_mention_probability=entity_mention_probability,
        openrouter_config=openrouter_config,
        evaluation_openrouter_config=evaluation_openrouter_config,
        bridge_config=bridge_config,
    )
    write_dataset_csv(preset, output, bridge_result.accepted_rows)
    summary_text = (
        "\n\n".join(
            (
                build_evaluation_bridge_summary(
                    bridge_result,
                    openrouter_config=openrouter_config,
                    evaluation_openrouter_config=evaluation_openrouter_config,
                    bridge_config=bridge_config,
                ).rstrip(),
                preset.build_summary(
                    bridge_result.accepted_rows,
                    mode=mode,
                    difficulty=difficulty,
                    schema_support_report=schema_support,
                    entity_mention_probability=entity_mention_probability,
                    openrouter_model=openrouter_config.model,
                ).rstrip(),
            )
        )
        + "\n"
    )
    resolved_summary_path = summary_path or default_summary_path(output)
    write_text(resolved_summary_path, summary_text)
    print(f"Wrote {len(bridge_result.accepted_rows)} rows to {output}")
    print(
        "Evaluation bridge results: "
        f"generated={bridge_result.total_generated_count}, "
        f"accepted={bridge_result.total_accepted_count}, "
        f"need_attention={bridge_result.total_need_attention_count}, "
        f"not_pass={bridge_result.total_not_pass_count}"
    )
    print(f"Wrote summary report to {resolved_summary_path}")
    return 0
