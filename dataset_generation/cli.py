"""CLI entrypoints for preset-backed structured dataset generation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from .evaluation_bridge import run_generation_with_evaluation_bridge
from .presets import PRESETS
from .runtime import run_generation
from .sampling import validate_probability
from .types import EvaluationBridgeConfig, OpenRouterConfig


def _build_bootstrap_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        required=True,
        help="Registered dataset preset to generate.",
    )
    return parser


def build_parser(
    default_preset: str | None = None,
    *,
    selected_preset: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a structured synthetic dataset from a registered preset."
    )
    active_preset_name = default_preset or selected_preset
    preset = PRESETS[active_preset_name] if active_preset_name is not None else None
    if default_preset is None:
        parser.add_argument(
            "--preset",
            choices=sorted(PRESETS),
            required=True,
            help="Registered dataset preset to generate.",
        )
    rows_default = preset.default_rows if preset is not None else 210
    output_default = preset.default_output_path if preset is not None else "data/structured.csv"
    mode_choices = preset.mode_choices if preset is not None else ("balanced", "random")
    mode_default = preset.default_mode if preset is not None else "balanced"
    difficulty_choices = (
        preset.difficulty_choices if preset is not None else ("easy", "medium", "hard")
    )
    difficulty_default = preset.default_difficulty if preset is not None else "medium"
    coverage_field = preset.coverage_field if preset is not None else "coverage target"
    evaluate_and_regenerate_default = (
        preset.default_evaluate_and_regenerate if preset is not None else False
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=rows_default,
        help="Number of rows to generate.",
    )
    parser.add_argument("--output", default=output_default, help="Output CSV path.")
    parser.add_argument(
        "--mode",
        choices=mode_choices,
        default=mode_default,
        help="Record generation mode.",
    )
    parser.add_argument(
        "--difficulty",
        choices=difficulty_choices,
        default=difficulty_default,
        help="Difficulty preset for language variation.",
    )
    parser.add_argument(
        "--min-coverage-count",
        type=int,
        default=None,
        help=f"Optional minimum count per reachable {coverage_field} in balanced mode.",
    )
    parser.add_argument(
        "--schema-audit",
        choices=["warn", "error", "ignore"],
        default="warn",
        help="How to handle declared schema labels that are not fully generatable.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional path for a dataset summary report.",
    )
    parser.add_argument(
        "--entity-mention-probability",
        type=float,
        default=None,
        help="Optional explicit named-entity mention probability between 0 and 1.",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-5.4-mini",
        help="OpenRouter model name used for text generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="OpenRouter sampling temperature.",
    )
    parser.add_argument("--batch-size", type=int, default=20, help="Rows per OpenRouter request.")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum additional retries after the initial OpenRouter attempt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--evaluate-and-regenerate",
        action=argparse.BooleanOptionalAction,
        default=evaluate_and_regenerate_default,
        help=(
            "Evaluate generated rows, keep only passes, and regenerate rejected "
            "rows until the target size is reached."
        ),
    )
    parser.add_argument(
        "--evaluator-model",
        default=None,
        help="Optional OpenRouter model name used for row evaluation in bridge mode.",
    )
    parser.add_argument(
        "--evaluator-temperature",
        type=float,
        default=0.0,
        help="OpenRouter sampling temperature for bridge-mode evaluation.",
    )
    parser.add_argument(
        "--evaluator-batch-size",
        type=int,
        default=None,
        help="Rows per OpenRouter evaluation request in bridge mode.",
    )
    parser.add_argument(
        "--evaluator-max-retries",
        type=int,
        default=2,
        help="Maximum additional retries after the initial bridge-mode evaluation attempt.",
    )
    parser.add_argument(
        "--max-evaluation-rounds",
        type=int,
        default=8,
        help="Maximum generate/evaluate refill rounds when bridge mode is enabled.",
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
        entity_mention_probability = validate_probability(
            "entity_mention_probability",
            args.entity_mention_probability,
        )
        openrouter_config = OpenRouterConfig(
            model=args.model,
            temperature=args.temperature,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
        )
    except ValueError as exc:
        parser.error(str(exc))
    except ValidationError as exc:
        parser.error(exc.errors()[0]["msg"])
    if args.evaluate_and_regenerate:
        try:
            evaluation_openrouter_config = OpenRouterConfig(
                model=args.evaluator_model or args.model,
                temperature=args.evaluator_temperature,
                batch_size=(
                    args.evaluator_batch_size
                    if args.evaluator_batch_size is not None
                    else args.batch_size
                ),
                max_retries=args.evaluator_max_retries,
            )
            bridge_config = EvaluationBridgeConfig(max_rounds=args.max_evaluation_rounds)
        except ValueError as exc:
            parser.error(str(exc))
        except ValidationError as exc:
            parser.error(exc.errors()[0]["msg"])
        return run_generation_with_evaluation_bridge(
            preset,
            rows=args.rows,
            output=args.output,
            mode=args.mode,
            difficulty=args.difficulty,
            min_coverage_count=args.min_coverage_count,
            schema_audit=args.schema_audit,
            summary_path=args.summary_path,
            entity_mention_probability=entity_mention_probability,
            openrouter_config=openrouter_config,
            evaluation_openrouter_config=evaluation_openrouter_config,
            bridge_config=bridge_config,
            seed=args.seed,
        )
    return run_generation(
        preset,
        rows=args.rows,
        output=args.output,
        mode=args.mode,
        difficulty=args.difficulty,
        min_coverage_count=args.min_coverage_count,
        schema_audit=args.schema_audit,
        summary_path=args.summary_path,
        entity_mention_probability=entity_mention_probability,
        openrouter_config=openrouter_config,
        seed=args.seed,
    )


def run_structured_dataset_cli() -> int:
    return main()


def run_enterprise_ticket_dataset_cli() -> int:
    return main(default_preset="enterprise_ticket")


def run_social_media_dataset_cli() -> int:
    return main(default_preset="social_media")


def run_restaurant_review_dataset_cli() -> int:
    return main(default_preset="restaurant_review")
