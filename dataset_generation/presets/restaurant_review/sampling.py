"""Sampling and summary helpers for the restaurant review dataset preset."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from dataset_generation.sampling import (
    apply_weight_adjustments,
    copy_weights,
    format_ratio,
    weighted_choice,
)
from dataset_generation.schema import validate_record_against_schema

from . import spec

_FOCUS_LABEL_TO_VALUE = {
    spec.build_granularity_label(primary_focus): primary_focus
    for primary_focus in spec.PRIMARY_FOCUS_VALUES
}


def _allowed(schema: dict[str, dict[str, Any]], field: str, parent_value: str) -> list[str]:
    return list(schema[field]["mapping"][parent_value])


def _allowed_pair(
    schema: dict[str, dict[str, Any]],
    field: str,
    parent_values: tuple[str, str],
) -> list[str]:
    return list(schema[field]["mapping"][parent_values])


def build_label_suites(record: spec.RecordDraft) -> dict[str, dict[str, str]]:
    suites: dict[str, dict[str, str]] = {}
    for family, components_by_level in spec.GRANULARITY_LABEL_COMPONENTS.items():
        suites[family] = {}
        for level, components in components_by_level.items():
            suites[family][level] = spec.build_granularity_label(
                *(record[field] for field in components)
            )
    return suites


def apply_granularity_labels(record: spec.RecordDraft) -> spec.Record:
    completed = dict(record)
    label_suites = build_label_suites(completed)
    for family, fields in spec.GRANULARITY_LABEL_FIELDS.items():
        for level, field_name in fields.items():
            completed[field_name] = label_suites[family][level]
    return completed


def audit_schema_support(schema: dict[str, Any]) -> dict[str, list[str]]:
    if schema is not spec.SCHEMA:
        raise ValueError("audit_schema_support expects the fixed SCHEMA defined in this module.")

    report: dict[str, list[str]] = {}
    for primary_focus in spec.PRIMARY_FOCUS_VALUES:
        for sentiment_polarity in spec.SENTIMENT_POLARITY_VALUES:
            key = (primary_focus, sentiment_polarity)
            if key not in schema["review_scenario"]["mapping"]:
                report.setdefault("missing_review_scenario_mapping", []).append(
                    f"{primary_focus} -> {sentiment_polarity}"
                )

    for family, fields in spec.GRANULARITY_LABEL_FIELDS.items():
        for level, field_name in fields.items():
            declared = set(schema[field_name]["values"])
            recomputed = {
                spec.build_granularity_label(*parts)
                for parts in _iter_granularity_parts(family, level)
            }
            if declared != recomputed:
                report.setdefault("granularity_label_mismatch", []).append(f"{family}:{level}")
    return report


def _iter_granularity_parts(family: str, level: str) -> list[tuple[str, ...]]:
    if family != "review":
        raise ValueError(f"Unknown granularity family: {family}")
    if level == "coarse":
        return [(primary_focus,) for primary_focus in spec.PRIMARY_FOCUS_VALUES]
    if level == "normal":
        return [
            (primary_focus, sentiment_polarity)
            for primary_focus in spec.PRIMARY_FOCUS_VALUES
            for sentiment_polarity in spec.SENTIMENT_POLARITY_VALUES
        ]
    return [
        (primary_focus, sentiment_polarity, review_scenario)
        for primary_focus in spec.PRIMARY_FOCUS_VALUES
        for sentiment_polarity in spec.SENTIMENT_POLARITY_VALUES
        for review_scenario in spec.REVIEW_SCENARIO_MAPPING[(primary_focus, sentiment_polarity)]
    ]


def _focus_target_order() -> list[str]:
    return [
        spec.build_granularity_label(primary_focus) for primary_focus in spec.PRIMARY_FOCUS_VALUES
    ]


def reachable_focus_targets() -> list[str]:
    return _focus_target_order()


def _build_balanced_focus_plan(
    n: int,
    rng: random.Random,
    *,
    min_coverage_count: int | None,
) -> list[str]:
    focus_targets = _focus_target_order()
    if min_coverage_count is not None and min_coverage_count < 0:
        raise ValueError("min_coverage_count must be non-negative.")
    required = len(focus_targets) * (min_coverage_count or 0)
    if required > n:
        raise ValueError(
            "rows is too small to satisfy the requested min_coverage_count across all review "
            "focus targets."
        )

    plan: list[str] = []
    if min_coverage_count:
        for _ in range(min_coverage_count):
            shuffled = list(focus_targets)
            rng.shuffle(shuffled)
            plan.extend(shuffled)
    elif n >= len(focus_targets):
        shuffled = list(focus_targets)
        rng.shuffle(shuffled)
        plan.extend(shuffled)

    while len(plan) < n:
        shuffled = list(focus_targets)
        rng.shuffle(shuffled)
        for focus_target in shuffled:
            if len(plan) >= n:
                break
            plan.append(focus_target)
    return plan[:n]


def _sample_focus_target(rng: random.Random) -> str:
    primary_focus = weighted_choice(rng, dict(spec.PRIMARY_FOCUS_WEIGHTS))
    return spec.build_granularity_label(primary_focus)


def _sample_rating_score(primary_focus: str, rng: random.Random) -> str:
    weights = dict(spec.RATING_SCORE_WEIGHTS)
    if primary_focus == "food_quality":
        weights = apply_weight_adjustments(weights, {"8": 0.08, "9": 0.05, "3": -0.02})
    elif primary_focus == "speed":
        weights = apply_weight_adjustments(weights, {"4": 0.06, "5": 0.04, "9": -0.04})
    elif primary_focus == "value":
        weights = apply_weight_adjustments(weights, {"6": 0.05, "7": 0.05, "10": -0.03})
    elif primary_focus == "ambience":
        weights = apply_weight_adjustments(weights, {"7": 0.05, "8": 0.05})
    return weighted_choice(rng, weights)


def _sample_restaurant_category(primary_focus: str, rng: random.Random) -> str:
    weights = dict(spec.RESTAURANT_CATEGORY_WEIGHTS)
    if primary_focus == "food_quality":
        weights = apply_weight_adjustments(
            weights,
            {"fine_dining": 0.15, "sushi_spot": 0.12, "fast_food": -0.08},
        )
    elif primary_focus == "service":
        weights = apply_weight_adjustments(
            weights,
            {"fine_dining": 0.15, "casual_dining": 0.1, "fast_food": -0.1},
        )
    elif primary_focus == "ambience":
        weights = apply_weight_adjustments(
            weights,
            {"fine_dining": 0.18, "cafe_bakery": 0.1, "fast_food": -0.12},
        )
    elif primary_focus == "value":
        weights = apply_weight_adjustments(
            weights,
            {"fast_food": 0.1, "pizza_place": 0.08, "fine_dining": -0.12},
        )
    elif primary_focus == "speed":
        weights = apply_weight_adjustments(
            weights,
            {"fast_food": 0.16, "pizza_place": 0.1, "fine_dining": -0.15},
        )
    elif primary_focus == "cleanliness":
        weights = apply_weight_adjustments(
            weights,
            {"cafe_bakery": 0.08, "fast_food": 0.06, "fine_dining": -0.04},
        )
    return weighted_choice(rng, weights)


def _sample_visit_mode(restaurant_category: str, primary_focus: str, rng: random.Random) -> str:
    weights = copy_weights(_allowed(spec.SCHEMA, "visit_mode", restaurant_category), 1.0)
    if restaurant_category == "fine_dining":
        return "dine_in"
    if primary_focus == "speed":
        for mode in ("takeout", "delivery", "drive_through"):
            if mode in weights:
                weights[mode] += 0.25
    if primary_focus == "ambience" and "dine_in" in weights:
        weights["dine_in"] += 0.35
    if primary_focus == "value" and "takeout" in weights:
        weights["takeout"] += 0.15
    return weighted_choice(rng, weights)


def _sample_review_scenario(
    primary_focus: str,
    sentiment_polarity: str,
    rng: random.Random,
) -> str:
    weights = copy_weights(
        _allowed_pair(spec.SCHEMA, "review_scenario", (primary_focus, sentiment_polarity)),
        1.0,
    )
    return weighted_choice(rng, weights)


def _sample_length_style(
    primary_focus: str,
    rating_score: str,
    visit_mode: str,
    difficulty: str,
    rng: random.Random,
) -> str:
    weights = copy_weights(list(spec.LENGTH_STYLE_VALUES), 1.0)
    if difficulty == "easy":
        weights = apply_weight_adjustments(
            weights, {"short": 0.25, "medium": 0.15, "detailed": -0.15}
        )
    elif difficulty == "hard":
        weights = apply_weight_adjustments(weights, {"medium": 0.1, "detailed": 0.25})

    if primary_focus in {"food_quality", "service"}:
        weights = apply_weight_adjustments(weights, {"medium": 0.2, "detailed": 0.15})
    elif primary_focus == "speed":
        weights = apply_weight_adjustments(weights, {"short": 0.25, "medium": 0.1})
    elif primary_focus == "ambience":
        weights = apply_weight_adjustments(weights, {"medium": 0.15, "detailed": 0.1})

    if visit_mode in {"delivery", "drive_through"}:
        weights = apply_weight_adjustments(weights, {"short": 0.2, "medium": 0.1})
    if int(rating_score) in {1, 2, 9, 10}:
        weights = apply_weight_adjustments(weights, {"medium": 0.15, "detailed": 0.15})

    return weighted_choice(rng, weights)


def sample_record_for_focus_target(
    focus_target: str,
    rng: random.Random,
    *,
    difficulty: str,
) -> spec.Record:
    _ = difficulty
    primary_focus = _FOCUS_LABEL_TO_VALUE.get(focus_target)
    if primary_focus is None:
        raise ValueError(f"Unknown review coverage target: {focus_target}")

    rating_score = _sample_rating_score(primary_focus, rng)
    sentiment_polarity = spec.RATING_TO_SENTIMENT[rating_score]
    review_scenario = _sample_review_scenario(primary_focus, sentiment_polarity, rng)
    restaurant_category = _sample_restaurant_category(primary_focus, rng)
    visit_mode = _sample_visit_mode(restaurant_category, primary_focus, rng)
    length_style = _sample_length_style(primary_focus, rating_score, visit_mode, difficulty, rng)

    record = apply_granularity_labels(
        {
            "restaurant_category": restaurant_category,
            "visit_mode": visit_mode,
            "primary_focus": primary_focus,
            "rating_score": rating_score,
            "sentiment_polarity": sentiment_polarity,
            "review_scenario": review_scenario,
            "length_style": length_style,
        }
    )
    validate_record(record)
    return record


def generate_records(
    n: int,
    rng: random.Random | None = None,
    *,
    mode: str,
    difficulty: str,
    min_coverage_count: int | None,
) -> list[spec.Record]:
    prng = rng or random.Random()
    if n < 0:
        raise ValueError("n must be non-negative")
    if mode not in {"balanced", "random"}:
        raise ValueError(f"Unsupported restaurant_review generation mode: {mode}")

    if mode == "balanced":
        focus_targets = _build_balanced_focus_plan(
            n,
            prng,
            min_coverage_count=min_coverage_count,
        )
    else:
        focus_targets = [_sample_focus_target(prng) for _ in range(n)]

    return [
        sample_record_for_focus_target(focus_target, prng, difficulty=difficulty)
        for focus_target in focus_targets
    ]


def generate_records_for_focus_targets(
    focus_targets: list[str],
    rng: random.Random | None = None,
    *,
    difficulty: str,
) -> list[spec.Record]:
    prng = rng or random.Random()
    return [
        sample_record_for_focus_target(focus_target, prng, difficulty=difficulty)
        for focus_target in focus_targets
    ]


def validate_record(record: spec.Record) -> None:
    validate_record_against_schema(spec.SCHEMA, record)


def build_dataset_summary(
    rows: list[dict[str, str]],
    *,
    mode: str,
    difficulty: str,
    schema_support_report: dict[str, list[str]],
    entity_mention_probability: float | None = None,
    openrouter_model: str | None = None,
) -> str:
    _ = entity_mention_probability
    focus_counts = Counter(row["primary_focus"] for row in rows)
    restaurant_counts = Counter(row["restaurant_category"] for row in rows)
    rating_counts = Counter(row["rating_score"] for row in rows)
    length_style_counts = Counter(row["length_style"] for row in rows)
    fine_label_counts = Counter(row["review_label_fine"] for row in rows)

    lines = [
        "Restaurant Review Dataset Summary",
        f"Rows: {len(rows)}",
        f"Mode: {mode}",
        f"Difficulty: {difficulty}",
        f"OpenRouter model: {openrouter_model or 'n/a'}",
        "",
        "Primary focus distribution:",
    ]
    for primary_focus in spec.PRIMARY_FOCUS_VALUES:
        count = focus_counts[primary_focus]
        lines.append(f"  {primary_focus}: {count} ({format_ratio(count, len(rows))})")

    lines.extend(["", "Restaurant category distribution:"])
    for restaurant_category in spec.RESTAURANT_CATEGORY_VALUES:
        count = restaurant_counts[restaurant_category]
        lines.append(f"  {restaurant_category}: {count} ({format_ratio(count, len(rows))})")

    lines.extend(["", "Rating distribution:"])
    for rating_score in spec.RATING_SCORE_VALUES:
        count = rating_counts[rating_score]
        lines.append(f"  {rating_score}: {count} ({format_ratio(count, len(rows))})")

    lines.extend(["", "Length-style distribution:"])
    for length_style in spec.LENGTH_STYLE_VALUES:
        count = length_style_counts[length_style]
        lines.append(f"  {length_style}: {count} ({format_ratio(count, len(rows))})")

    lines.extend(["", "Top fine-grained review labels:"])
    for label, count in fine_label_counts.most_common(12):
        lines.append(f"  {label}: {count}")

    lines.extend(["", "Schema support audit:"])
    if not schema_support_report:
        lines.append("  no unsupported declared labels found")
    else:
        for issue_name, values in sorted(schema_support_report.items()):
            lines.append(f"  {issue_name}:")
            for value in values:
                lines.append(f"    - {value}")
    return "\n".join(lines)
