"""Pairwise evaluator registry for mapping evaluator names to implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from .base import PairwiseEvaluator
from .methods import PairwiseConsistencyEvaluator, RobustnessExpectationsEvaluator


class PairwiseEvaluatorFactory(Protocol):
    """Class-like pairwise evaluator factory used by the registry."""

    name: str

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None = None) -> PairwiseEvaluator: ...


PAIRWISE_EVALUATORS: dict[str, PairwiseEvaluatorFactory] = {
    PairwiseConsistencyEvaluator.name: PairwiseConsistencyEvaluator,
    RobustnessExpectationsEvaluator.name: RobustnessExpectationsEvaluator,
}


def list_pairwise_evaluators() -> list[str]:
    return sorted(PAIRWISE_EVALUATORS)


def get_pairwise_evaluator(
    name: str,
    *,
    params: Mapping[str, Any] | None = None,
) -> PairwiseEvaluator:
    normalized = name.strip()
    if normalized not in PAIRWISE_EVALUATORS:
        available = list_pairwise_evaluators()
        valid = ", ".join(available) if available else "(none registered)"
        raise ValueError(f"Unknown pairwise evaluator '{name}'. Valid: {valid}")
    return PAIRWISE_EVALUATORS[normalized].from_params(params=params)
