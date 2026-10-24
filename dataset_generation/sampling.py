"""Shared sampling utilities for structured dataset generation."""

from __future__ import annotations

import random


def weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    filtered = {value: weight for value, weight in weights.items() if weight > 0}
    if not filtered:
        raise ValueError("weighted_choice requires at least one positive weight.")
    values = list(filtered.keys())
    return rng.choices(values, weights=list(filtered.values()), k=1)[0]


def apply_weight_adjustments(
    base_weights: dict[str, float],
    adjustments: dict[str, float],
) -> dict[str, float]:
    adjusted = dict(base_weights)
    for value, delta in adjustments.items():
        if value in adjusted:
            adjusted[value] = max(0.0, adjusted[value] + delta)
    return adjusted


def copy_weights(values: list[str], default_weight: float = 1.0) -> dict[str, float]:
    return {value: default_weight for value in values}


def pick_one(rng: random.Random, options: list[str]) -> str:
    return options[rng.randrange(len(options))]


def clamp_probability(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def validate_probability(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1 inclusive.")
    return value


def format_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator / denominator:.3f}"


def build_balanced_coverage_plan(
    targets: list[str],
    n: int,
    rng: random.Random,
    *,
    min_count: int | None = None,
    coverage_label: str = "coverage target",
) -> list[str]:
    if min_count is not None and min_count < 0:
        raise ValueError("min_coverage_target_count must be non-negative")

    shuffled_targets = list(targets)
    rng.shuffle(shuffled_targets)

    if min_count is None:
        base_count = n // len(shuffled_targets)
        remainder = n % len(shuffled_targets)
    else:
        required_rows = len(shuffled_targets) * min_count
        if required_rows > n:
            raise ValueError(
                f"rows is too small for the requested minimum {coverage_label} coverage: "
                f"{n} < {required_rows}"
            )
        base_count = min_count
        remainder = n - required_rows

    plan: list[str] = []
    for target in shuffled_targets:
        for _ in range(base_count):
            plan.append(target)
    for index in range(remainder):
        plan.append(shuffled_targets[index % len(shuffled_targets)])
    rng.shuffle(plan)
    return plan
