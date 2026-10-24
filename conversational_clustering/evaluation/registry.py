"""Evaluator registry lookup for mapping evaluator names to implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from .base import Evaluator
from .methods import (
    BasicMetricsEvaluator,
    ClarificationPolicyEvaluator,
    ClusteringPovSemanticQualityEvaluator,
    ClusterSemanticQualityEvaluator,
    ControllabilityEvaluator,
    ConversationTraceMetricsEvaluator,
    DiscoverySemanticQualityEvaluator,
    EfficiencyUtilityEvaluator,
    HumanEffortMetricsEvaluator,
    InferredThemeEfficiencyUtilityEvaluator,
    LabelGroundtruthConsistencyEvaluator,
    NullThemeAbstentionEvaluator,
    SemanticConsistencyEvaluator,
    TopicConditionedQualityEvaluator,
)


class EvaluatorFactory(Protocol):
    """Class-like evaluator factory used by the registry."""

    name: str

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None = None) -> Evaluator: ...


EVALUATORS: dict[str, EvaluatorFactory] = {
    LabelGroundtruthConsistencyEvaluator.name: LabelGroundtruthConsistencyEvaluator,
    BasicMetricsEvaluator.name: BasicMetricsEvaluator,
    ClarificationPolicyEvaluator.name: ClarificationPolicyEvaluator,
    ClusteringPovSemanticQualityEvaluator.name: ClusteringPovSemanticQualityEvaluator,
    ClusterSemanticQualityEvaluator.name: ClusterSemanticQualityEvaluator,
    ConversationTraceMetricsEvaluator.name: ConversationTraceMetricsEvaluator,
    ControllabilityEvaluator.name: ControllabilityEvaluator,
    DiscoverySemanticQualityEvaluator.name: DiscoverySemanticQualityEvaluator,
    EfficiencyUtilityEvaluator.name: EfficiencyUtilityEvaluator,
    HumanEffortMetricsEvaluator.name: HumanEffortMetricsEvaluator,
    InferredThemeEfficiencyUtilityEvaluator.name: InferredThemeEfficiencyUtilityEvaluator,
    SemanticConsistencyEvaluator.name: SemanticConsistencyEvaluator,
    TopicConditionedQualityEvaluator.name: TopicConditionedQualityEvaluator,
    NullThemeAbstentionEvaluator.name: NullThemeAbstentionEvaluator,
}


def list_evaluators() -> list[str]:
    return sorted(EVALUATORS)


def get_evaluator(
    name: str,
    *,
    params: Mapping[str, Any] | None = None,
) -> Evaluator:
    normalized = name.strip()
    if normalized not in EVALUATORS:
        available = list_evaluators()
        valid = ", ".join(available) if available else "(none registered)"
        raise ValueError(f"Unknown evaluator '{name}'. Valid: {valid}")
    return EVALUATORS[normalized].from_params(params=params)
