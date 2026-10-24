"""Controllability evaluator for intent-changing turn-injection runs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any, cast

from conversational_clustering.evaluation.artifacts import LoadedClustering
from conversational_clustering.evaluation.base import EvaluationRequest, EvaluationResult, Evaluator
from conversational_clustering.evaluation.benchmark_spec import (
    ReferenceClusteringSpec,
    SyntheticBenchmarkSpec,
    ThemeSpec,
)
from conversational_clustering.evaluation.methods.benchmark_utils import (
    collect_snapshot_clusterings,
    compare_clusterings,
    require_benchmark_spec,
    resolve_active_theme,
    resolve_final_turn_index,
)
from conversational_clustering.human_turn_injection import HumanTurnInjectionConfig

_GRANULARITY_ORDER = ("coarse", "normal", "fine")


@dataclass(frozen=True)
class _GranularityShiftBoundaryError(ValueError):
    edit_primitive: str
    theme_id: str
    source_granularity: str

    def __str__(self) -> str:
        return (
            f"Evaluator 'controllability' cannot {self.edit_primitive} benchmark theme "
            f"'{self.theme_id}' from source granularity '{self.source_granularity}'"
        )


@dataclass(frozen=True)
class ControllabilityEvaluator(Evaluator):
    name = "controllability"
    resume_version = "v2"
    alpha: float = 0.8
    beta: float = 0.1

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None = None) -> ControllabilityEvaluator:
        normalized_params = dict(params or {})
        supported_keys = {"alpha", "beta"}
        unsupported_keys = sorted(set(normalized_params) - supported_keys)
        if unsupported_keys:
            raise ValueError(
                f"Evaluator '{cls.name}' received unsupported parameter(s): "
                + ", ".join(unsupported_keys)
            )
        return cls(
            alpha=_parse_probability(normalized_params, field_name="alpha", default=0.8),
            beta=_parse_probability(normalized_params, field_name="beta", default=0.1),
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        benchmark_spec = require_benchmark_spec(request.artifacts, evaluator_name=self.name)
        turn_injection = _resolve_turn_injection(request)
        snapshots = collect_snapshot_clusterings(request)

        pre_request_candidates = [
            (turn_index, clustering)
            for turn_index, clustering in snapshots
            if turn_index < turn_injection.turn_index
        ]
        pre_request_turn_index: int
        pre_request_clustering: LoadedClustering | None
        if pre_request_candidates:
            pre_request_turn_index, pre_request_clustering = pre_request_candidates[-1]
        else:
            pre_request_turn_index = 0
            pre_request_clustering = None

        later_states = [
            (turn_index, clustering)
            for turn_index, clustering in snapshots
            if turn_index >= turn_injection.turn_index
        ]
        if not later_states:
            raise ValueError(
                f"Evaluator '{self.name}' found no judge snapshots at or after "
                f"turn_injection.turn_index={turn_injection.turn_index}; the run likely ended "
                "before the injected edit request executed"
            )

        try:
            resolved_targets = _resolve_controllability_targets(
                benchmark_spec=benchmark_spec,
                turn_injection=turn_injection,
                pre_request_clustering=pre_request_clustering,
                final_clustering=request.predictions,
                artifacts=request.artifacts,
            )
        except _GranularityShiftBoundaryError as exc:
            return EvaluationResult(
                method=self.name,
                metrics={},
                details={
                    "alpha": self.alpha,
                    "beta": self.beta,
                    "request_turn": turn_injection.turn_index,
                    "pre_request_turn": pre_request_turn_index,
                    "edit_primitive": turn_injection.edit_primitive,
                    "source_theme_id": exc.theme_id,
                    "source_granularity": exc.source_granularity,
                    "invalid_episode": True,
                    "invalid_reason": "granularity_shift_at_boundary",
                },
            )

        if pre_request_clustering is None:
            pre_request_clustering = resolved_targets.source_reference.to_loaded_clustering()

        final_turn_index = resolve_final_turn_index(request, snapshots=snapshots)
        if _assignments_differ(request.predictions, later_states[-1][1]):
            if later_states[-1][0] == final_turn_index:
                later_states[-1] = (final_turn_index, request.predictions)
            else:
                later_states.append((final_turn_index, request.predictions))

        target_clustering = resolved_targets.target_reference.to_loaded_clustering()
        trajectory = [
            _evaluate_state(
                current_turn_index=turn_index,
                current_clustering=clustering,
                pre_request_clustering=pre_request_clustering,
                target_clustering=target_clustering,
            )
            for turn_index, clustering in later_states
        ]
        max_change_pairs_state = max(trajectory, key=lambda state: state.num_change_pairs)
        if max_change_pairs_state.num_change_pairs == 0:
            return EvaluationResult(
                method=self.name,
                metrics={},
                details={
                    "alpha": self.alpha,
                    "beta": self.beta,
                    "request_turn": turn_injection.turn_index,
                    "pre_request_turn": pre_request_turn_index,
                    "edit_primitive": turn_injection.edit_primitive,
                    "source_theme_id": resolved_targets.source_theme.theme_id,
                    "target_theme_id": resolved_targets.target_theme.theme_id,
                    "source_granularity": resolved_targets.source_reference.granularity,
                    "target_granularity": resolved_targets.target_reference.granularity,
                    "source_reference_id": resolved_targets.source_reference.reference_id,
                    "target_reference_id": resolved_targets.target_reference.reference_id,
                    "num_common_items": max_change_pairs_state.num_common_items,
                    "num_change_pairs": 0,
                    "num_keep_pairs": max_change_pairs_state.num_keep_pairs,
                    "invalid_episode": True,
                    "invalid_reason": "no_change_pairs",
                },
            )

        lag: int | None = None
        acceptable_turn_index: int | None = None
        for state in trajectory:
            if state.target_change_success >= self.alpha and state.collateral_churn <= self.beta:
                lag = state.turn_index - turn_injection.turn_index
                acceptable_turn_index = state.turn_index
                break

        final_state = trajectory[-1]
        return EvaluationResult(
            method=self.name,
            metrics={
                "target_change_success": final_state.target_change_success,
                "collateral_churn": final_state.collateral_churn,
                "lag_success_rate": 1.0 if lag is not None else 0.0,
            },
            details={
                "alpha": self.alpha,
                "beta": self.beta,
                "request_turn": turn_injection.turn_index,
                "pre_request_turn": pre_request_turn_index,
                "acceptable_turn": acceptable_turn_index,
                "lag": lag,
                "edit_primitive": turn_injection.edit_primitive,
                "source_theme_id": resolved_targets.source_theme.theme_id,
                "target_theme_id": resolved_targets.target_theme.theme_id,
                "source_granularity": resolved_targets.source_reference.granularity,
                "target_granularity": resolved_targets.target_reference.granularity,
                "source_reference_id": resolved_targets.source_reference.reference_id,
                "target_reference_id": resolved_targets.target_reference.reference_id,
                "num_common_items": final_state.num_common_items,
                "num_change_pairs": final_state.num_change_pairs,
                "num_keep_pairs": final_state.num_keep_pairs,
                "best_target_change_success": max(
                    state.target_change_success for state in trajectory
                ),
                "min_collateral_churn": min(state.collateral_churn for state in trajectory),
                "final_target_change_success": final_state.target_change_success,
                "final_collateral_churn": final_state.collateral_churn,
            },
        )


@dataclass(frozen=True)
class _ResolvedControllabilityTargets:
    source_theme: ThemeSpec
    target_theme: ThemeSpec
    source_reference: ReferenceClusteringSpec
    target_reference: ReferenceClusteringSpec


@dataclass(frozen=True)
class _ControllabilityStateEvaluation:
    turn_index: int
    num_common_items: int
    num_change_pairs: int
    num_keep_pairs: int
    target_change_success: float
    collateral_churn: float


def _parse_probability(
    params: Mapping[str, Any],
    *,
    field_name: str,
    default: float,
) -> float:
    value = params.get(field_name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"'{field_name}' must be a numeric probability")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"'{field_name}' must be a finite number between 0.0 and 1.0")
    return parsed


def _resolve_turn_injection(request: EvaluationRequest) -> HumanTurnInjectionConfig:
    batch_run_config = request.artifacts.get("batch_run_config")
    if not isinstance(batch_run_config, Mapping):
        raise ValueError(
            "Evaluator 'controllability' requires batch_run_config in request artifacts"
        )
    human_agent_config = batch_run_config.get("human_agent_config")
    if not isinstance(human_agent_config, Mapping):
        raise ValueError("Evaluator 'controllability' requires batch_run_config.human_agent_config")
    raw_turn_injection = human_agent_config.get("turn_injection")
    if not isinstance(raw_turn_injection, Mapping):
        raise ValueError(
            "Evaluator 'controllability' requires an explicit "
            "batch_run_config.human_agent_config.turn_injection"
        )
    try:
        return HumanTurnInjectionConfig.from_dict(cast(Mapping[str, object], raw_turn_injection))
    except ValueError as exc:
        raise ValueError(
            f"Evaluator 'controllability' received invalid turn-injection metadata: {exc}"
        ) from exc


def _resolve_controllability_targets(
    *,
    benchmark_spec: SyntheticBenchmarkSpec,
    turn_injection: HumanTurnInjectionConfig,
    pre_request_clustering: LoadedClustering | None,
    final_clustering: LoadedClustering,
    artifacts: Mapping[str, Any],
) -> _ResolvedControllabilityTargets:
    if turn_injection.edit_primitive == "theme_switch":
        return _resolve_theme_switch_targets(
            benchmark_spec=benchmark_spec,
            turn_injection=turn_injection,
            pre_request_clustering=pre_request_clustering,
            final_clustering=final_clustering,
        )
    if turn_injection.edit_primitive in {"coarsen", "refine"}:
        return _resolve_granularity_shift_targets(
            benchmark_spec=benchmark_spec,
            turn_injection=turn_injection,
            pre_request_clustering=pre_request_clustering,
            final_clustering=final_clustering,
            artifacts=artifacts,
        )
    raise ValueError(
        "Evaluator 'controllability' does not support edit_primitive="
        f"'{turn_injection.edit_primitive}'"
    )


def _resolve_theme_switch_targets(
    *,
    benchmark_spec: SyntheticBenchmarkSpec,
    turn_injection: HumanTurnInjectionConfig,
    pre_request_clustering: LoadedClustering | None,
    final_clustering: LoadedClustering,
) -> _ResolvedControllabilityTargets:
    source_theme = _resolve_configured_theme(
        benchmark_spec=benchmark_spec,
        theme_id=turn_injection.source_theme_id,
        field_name="source_theme_id",
    )
    target_theme = _resolve_configured_theme(
        benchmark_spec=benchmark_spec,
        theme_id=turn_injection.target_theme_id,
        field_name="target_theme_id",
    )

    source_reference = _resolve_source_reference(
        benchmark_spec=benchmark_spec,
        source_theme=source_theme,
        pre_request_clustering=pre_request_clustering,
    )
    target_reference = _resolve_target_reference(
        benchmark_spec=benchmark_spec,
        target_theme=target_theme,
        final_clustering=final_clustering,
    )

    return _ResolvedControllabilityTargets(
        source_theme=source_theme,
        target_theme=target_theme,
        source_reference=source_reference,
        target_reference=target_reference,
    )


def _resolve_granularity_shift_targets(
    *,
    benchmark_spec: SyntheticBenchmarkSpec,
    turn_injection: HumanTurnInjectionConfig,
    pre_request_clustering: LoadedClustering | None,
    final_clustering: LoadedClustering,
    artifacts: Mapping[str, Any],
) -> _ResolvedControllabilityTargets:
    active_theme = resolve_active_theme(artifacts=artifacts, benchmark_spec=benchmark_spec)
    if active_theme is None:
        raise ValueError(
            "Evaluator 'controllability' could not resolve the active benchmark theme for "
            f"edit_primitive='{turn_injection.edit_primitive}'"
        )

    candidate_references = _references_for_theme(
        benchmark_spec=benchmark_spec,
        theme_id=active_theme.theme_id,
    )
    if not candidate_references:
        raise ValueError(
            "Evaluator 'controllability' found no benchmark references for active theme "
            f"'{active_theme.theme_id}'"
        )

    if pre_request_clustering is None:
        source_reference = _resolve_theme_granularity_reference(
            benchmark_spec=benchmark_spec,
            theme=active_theme,
        )
    else:
        source_reference = _resolve_source_reference(
            benchmark_spec=benchmark_spec,
            source_theme=active_theme,
            pre_request_clustering=pre_request_clustering,
        )
    target_reference = _resolve_shift_target_reference(
        references=candidate_references,
        source_reference=source_reference,
        edit_primitive=turn_injection.edit_primitive,
        theme_id=active_theme.theme_id,
        final_clustering=final_clustering,
    )
    target_theme = benchmark_spec.theme_by_id.get(target_reference.topic_id)
    if target_theme is None:
        raise ValueError(
            "Evaluator 'controllability' resolved target reference "
            f"'{target_reference.reference_id}', but its topic_id '{target_reference.topic_id}' "
            "is not defined in the benchmark spec"
        )

    return _ResolvedControllabilityTargets(
        source_theme=active_theme,
        target_theme=target_theme,
        source_reference=source_reference,
        target_reference=target_reference,
    )


def _resolve_theme_granularity_reference(
    *,
    benchmark_spec: SyntheticBenchmarkSpec,
    theme: ThemeSpec,
) -> ReferenceClusteringSpec:
    try:
        return benchmark_spec.resolve_reference(
            topic_id=theme.theme_id,
            granularity=theme.granularity,
        )
    except ValueError as exc:
        raise ValueError(
            "Evaluator 'controllability' found no benchmark reference for active theme "
            f"'{theme.theme_id}' at granularity '{theme.granularity}'"
        ) from exc


def _resolve_configured_theme(
    *,
    benchmark_spec: SyntheticBenchmarkSpec,
    theme_id: str | None,
    field_name: str,
) -> ThemeSpec:
    if theme_id is None:
        raise ValueError(
            f"Evaluator 'controllability' requires turn_injection.{field_name} for theme_switch"
        )
    configured_theme = benchmark_spec.theme_by_id.get(theme_id)
    if configured_theme is None:
        raise ValueError(
            "Evaluator 'controllability' received "
            f"{field_name}='{theme_id}', but that theme is not "
            "defined in the dataset benchmark spec"
        )
    return configured_theme


def _resolve_source_reference(
    *,
    benchmark_spec: SyntheticBenchmarkSpec,
    source_theme: ThemeSpec,
    pre_request_clustering: LoadedClustering | None,
) -> ReferenceClusteringSpec:
    candidate_references = _references_for_theme(
        benchmark_spec=benchmark_spec,
        theme_id=source_theme.theme_id,
    )
    if not candidate_references:
        raise ValueError(
            "Evaluator 'controllability' found no benchmark references for "
            f"source_theme_id='{source_theme.theme_id}'"
        )
    if pre_request_clustering is not None:
        return _best_matching_reference(
            clustering=pre_request_clustering,
            references=candidate_references,
        )
    default_reference = _default_reference(candidate_references)
    if default_reference is None:
        raise ValueError(
            "Evaluator 'controllability' found no default source reference for "
            f"source_theme_id='{source_theme.theme_id}'"
        )
    return default_reference


def _resolve_shift_target_reference(
    *,
    references: list[ReferenceClusteringSpec],
    source_reference: ReferenceClusteringSpec,
    edit_primitive: str,
    theme_id: str,
    final_clustering: LoadedClustering,
) -> ReferenceClusteringSpec:
    source_granularity = source_reference.granularity
    if source_granularity not in _GRANULARITY_ORDER:
        raise ValueError(
            "Evaluator 'controllability' requires benchmark references with granularity in "
            f"{_GRANULARITY_ORDER} for edit_primitive='{edit_primitive}', but source reference "
            f"'{source_reference.reference_id}' used unsupported granularity "
            f"'{source_granularity}'"
        )

    source_index = _GRANULARITY_ORDER.index(source_granularity)
    if edit_primitive == "coarsen":
        if source_index == 0:
            raise _GranularityShiftBoundaryError(
                edit_primitive=edit_primitive,
                theme_id=theme_id,
                source_granularity=source_granularity,
            )
        target_granularity = _GRANULARITY_ORDER[source_index - 1]
    elif edit_primitive == "refine":
        if source_index == len(_GRANULARITY_ORDER) - 1:
            raise _GranularityShiftBoundaryError(
                edit_primitive=edit_primitive,
                theme_id=theme_id,
                source_granularity=source_granularity,
            )
        target_granularity = _GRANULARITY_ORDER[source_index + 1]
    else:
        raise ValueError(
            f"Evaluator 'controllability' does not support edit_primitive='{edit_primitive}'"
        )

    target_references = [
        reference for reference in references if reference.granularity == target_granularity
    ]
    if target_references:
        return _best_matching_reference(
            clustering=final_clustering,
            references=target_references,
        )
    raise ValueError(
        "Evaluator 'controllability' found no benchmark reference for "
        f"theme '{theme_id}' at target granularity '{target_granularity}'"
    )


def _resolve_target_reference(
    *,
    benchmark_spec: SyntheticBenchmarkSpec,
    target_theme: ThemeSpec,
    final_clustering: LoadedClustering,
) -> ReferenceClusteringSpec:
    target_references = _references_for_theme(
        benchmark_spec=benchmark_spec,
        theme_id=target_theme.theme_id,
    )
    if not target_references:
        raise ValueError(
            "Evaluator 'controllability' found no benchmark references for "
            f"target_theme_id='{target_theme.theme_id}'"
        )
    return _best_matching_reference(
        clustering=final_clustering,
        references=target_references,
    )


def _references_for_theme(
    *,
    benchmark_spec: SyntheticBenchmarkSpec,
    theme_id: str,
) -> list[ReferenceClusteringSpec]:
    theme = benchmark_spec.theme_by_id.get(theme_id)
    if theme is None:
        return []
    references = list(benchmark_spec.references_for_theme_or_group(theme.theme_id))
    return sorted(
        references,
        key=lambda reference: (
            _reference_preference_rank(reference),
            reference.granularity != "normal",
            reference.reference_id,
        ),
    )


def _default_reference(
    references: list[ReferenceClusteringSpec],
) -> ReferenceClusteringSpec | None:
    for preferred_granularity in ("normal", "coarse", "fine"):
        for reference in references:
            if reference.granularity == preferred_granularity:
                return reference
    return references[0] if references else None


def _best_matching_reference(
    *,
    clustering: LoadedClustering,
    references: list[ReferenceClusteringSpec],
) -> ReferenceClusteringSpec:
    best_reference = references[0]
    best_score: tuple[float, float, int, int, str] | None = None
    granularity_priority = {
        "normal": 0,
        "coarse": 1,
        "fine": 2,
    }
    for reference in references:
        metrics, details = compare_clusterings(
            left=clustering,
            right=reference.to_loaded_clustering(),
        )
        score = (
            metrics["adjusted_rand_index"],
            metrics["coassignment_jaccard"],
            -int(details["cluster_count_abs_diff_on_common_items"]),
            -granularity_priority.get(reference.granularity, 4),
            reference.reference_id,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_reference = reference
    return best_reference


def _reference_preference_rank(reference: ReferenceClusteringSpec) -> tuple[int, int]:
    granularity_priority = {
        "normal": 0,
        "coarse": 1,
        "fine": 2,
    }.get(reference.granularity, 4)
    return (granularity_priority, _reference_num_clusters(reference))


def _reference_num_clusters(reference: ReferenceClusteringSpec) -> int:
    return len(set(reference.assignment.values()))


def _evaluate_state(
    *,
    current_turn_index: int,
    current_clustering: LoadedClustering,
    pre_request_clustering: LoadedClustering,
    target_clustering: LoadedClustering,
) -> _ControllabilityStateEvaluation:
    common_items = (
        set(pre_request_clustering.assignment)
        & set(target_clustering.assignment)
        & set(current_clustering.assignment)
    )
    num_change_pairs = 0
    num_keep_pairs = 0
    corrected_changed_pairs = 0
    disturbed_keep_pairs = 0
    for left_item, right_item in combinations(sorted(common_items), 2):
        pre_relation = (
            pre_request_clustering.assignment[left_item]
            == pre_request_clustering.assignment[right_item]
        )
        target_relation = (
            target_clustering.assignment[left_item] == target_clustering.assignment[right_item]
        )
        current_relation = (
            current_clustering.assignment[left_item] == current_clustering.assignment[right_item]
        )
        if pre_relation == target_relation:
            num_keep_pairs += 1
            if current_relation != target_relation:
                disturbed_keep_pairs += 1
            continue
        num_change_pairs += 1
        if current_relation == target_relation:
            corrected_changed_pairs += 1

    target_change_success = (
        corrected_changed_pairs / num_change_pairs if num_change_pairs > 0 else 0.0
    )
    collateral_churn = disturbed_keep_pairs / num_keep_pairs if num_keep_pairs > 0 else 0.0
    return _ControllabilityStateEvaluation(
        turn_index=current_turn_index,
        num_common_items=len(common_items),
        num_change_pairs=num_change_pairs,
        num_keep_pairs=num_keep_pairs,
        target_change_success=target_change_success,
        collateral_churn=collateral_churn,
    )


def _assignments_differ(left: LoadedClustering, right: LoadedClustering) -> bool:
    return left.assignment != right.assignment or left.cluster_names != right.cluster_names
