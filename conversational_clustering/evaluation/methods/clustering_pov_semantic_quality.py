"""LLM-judged semantic quality evaluator for clustering-agent POV-bank artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from conversational_clustering.accounting import UsageCollector, build_pricing_client_from_configs
from conversational_clustering.evaluation.base import EvaluationRequest, EvaluationResult, Evaluator
from conversational_clustering.evaluation.benchmark_spec import SyntheticBenchmarkSpec
from conversational_clustering.evaluation.methods.benchmark_utils import require_benchmark_spec
from conversational_clustering.evaluation.methods.discovery_semantic_quality_shared import (
    DiscoverySemanticQualityConfig,
    clamp_probability,
    evaluate_surfaced_topics_semantic_quality,
    format_benchmark_themes,
    parse_discovery_semantic_quality_config,
)
from conversational_clustering.evaluation.methods.semantic_quality_judges import (
    judge_duplicate_topics,
    judge_theme_match,
)
from conversational_clustering.models import (
    ModelConfig,
    default_semantic_consistency_judge_model_config,
)


@dataclass(frozen=True)
class _PovSnapshot:
    turn_index: int
    povs: list[str]


@dataclass(frozen=True)
class _HistoryDiscoveryOutcome:
    discovery_utility_auc: float
    total_turns_evaluated: int
    discovery_utility_curve: list[dict[str, float | int]]
    first_valid_discovery_turn_by_theme: dict[str, int]
    history_judged_povs: list[dict[str, Any]]


@dataclass(frozen=True)
class ClusteringPovSemanticQualityEvaluator(Evaluator):
    name = "clustering_pov_semantic_quality"
    resume_version = "v2"
    model_config: ModelConfig | None = None
    theme_match_prompt_id: str = "baseline_v1"
    duplicate_prompt_id: str = "baseline_v1"

    @classmethod
    def from_params(
        cls, params: Mapping[str, Any] | None = None
    ) -> ClusteringPovSemanticQualityEvaluator:
        config = parse_discovery_semantic_quality_config(evaluator_name=cls.name, params=params)
        return cls(
            model_config=config.model_config,
            theme_match_prompt_id=config.theme_match_prompt_id,
            duplicate_prompt_id=config.duplicate_prompt_id,
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        usage_collector = UsageCollector()
        benchmark_spec = require_benchmark_spec(request.artifacts, evaluator_name=self.name)
        pov_history = request.artifacts.get("clustering_pov_history")
        if not isinstance(pov_history, Mapping):
            raise ValueError(
                f"Evaluator '{self.name}' requires a clustering_pov_history artifact in the request"
            )

        normalized_snapshots = _normalize_pov_history(pov_history.get("snapshots"))
        if not normalized_snapshots:
            return EvaluationResult(
                method=self.name,
                metrics={
                    "non_redundancy": 0.0,
                    "supported_theme_precision": 0.0,
                    "distinct_valid_discovery_precision": 0.0,
                    "valid_theme_coverage": 0.0,
                    "num_supported_topics": 0.0,
                    "num_distinct_valid_topics": 0.0,
                    "discovery_utility_auc": 0.0,
                },
                details={
                    "num_surfaced_topics": 0,
                    "num_supported_topics": 0,
                    "num_distinct_valid_topics": 0,
                    "judged_topics": [],
                    "duplicate_pairs": [],
                    "num_history_snapshots": 0,
                    "total_turns_evaluated": 0,
                    "discovery_utility_curve": [],
                    "first_valid_discovery_turn_by_theme": {},
                    "history_judged_povs": [],
                },
            )

        final_snapshot = normalized_snapshots[-1]
        semantic_outcome = evaluate_surfaced_topics_semantic_quality(
            surfaced_topics=final_snapshot.povs,
            benchmark_spec=benchmark_spec,
            config=self._config,
            theme_match_judge=lambda **kwargs: judge_theme_match(
                **kwargs, usage_collector=usage_collector
            ),
            duplicate_judge=lambda **kwargs: judge_duplicate_topics(
                **kwargs, usage_collector=usage_collector
            ),
        )
        history_outcome = _evaluate_pov_history_discovery(
            snapshots=normalized_snapshots,
            benchmark_spec=benchmark_spec,
            config=self._config,
            turns_run=request.artifacts.get("turns_run"),
            usage_collector=usage_collector,
        )

        return EvaluationResult(
            method=self.name,
            metrics={
                **semantic_outcome.metrics,
                "discovery_utility_auc": history_outcome.discovery_utility_auc,
            },
            details={
                **semantic_outcome.details,
                "num_history_snapshots": len(normalized_snapshots),
                "total_turns_evaluated": history_outcome.total_turns_evaluated,
                "discovery_utility_curve": history_outcome.discovery_utility_curve,
                "first_valid_discovery_turn_by_theme": (
                    history_outcome.first_valid_discovery_turn_by_theme
                ),
                "history_judged_povs": history_outcome.history_judged_povs,
            },
            accounting=usage_collector.snapshot(
                pricer=build_pricing_client_from_configs(
                    [self.model_config]
                    if self.model_config
                    else [default_semantic_consistency_judge_model_config()]
                )
            ).to_dict(),
        )

    @property
    def _config(self) -> DiscoverySemanticQualityConfig:
        return DiscoverySemanticQualityConfig(
            model_config=self.model_config,
            theme_match_prompt_id=self.theme_match_prompt_id,
            duplicate_prompt_id=self.duplicate_prompt_id,
        )


def _normalize_pov_history(raw_snapshots: object) -> list[_PovSnapshot]:
    if not isinstance(raw_snapshots, list):
        return []

    normalized_snapshots: list[_PovSnapshot] = []
    for raw_snapshot in raw_snapshots:
        if not isinstance(raw_snapshot, Mapping):
            continue
        snapshot_mapping = cast(Mapping[str, object], raw_snapshot)
        raw_turn_index = snapshot_mapping.get("turn_index")
        raw_povs = snapshot_mapping.get("povs")
        if isinstance(raw_turn_index, bool) or not isinstance(raw_turn_index, int):
            continue
        if raw_turn_index < 1 or not isinstance(raw_povs, list):
            continue
        normalized_povs = _coerce_povs(cast(list[object], raw_povs))
        if not normalized_povs:
            continue
        normalized_snapshots.append(_PovSnapshot(turn_index=raw_turn_index, povs=normalized_povs))
    return sorted(normalized_snapshots, key=lambda snapshot: snapshot.turn_index)


def _coerce_povs(raw_povs: list[object]) -> list[str]:
    normalized_povs: list[str] = []
    seen_keys: set[str] = set()
    for raw_pov in raw_povs:
        if not isinstance(raw_pov, str):
            continue
        normalized_pov = " ".join(raw_pov.split())
        if not normalized_pov:
            continue
        dedupe_key = normalized_pov.lower()
        if dedupe_key in seen_keys:
            continue
        normalized_povs.append(normalized_pov)
        seen_keys.add(dedupe_key)
    return normalized_povs


def _evaluate_pov_history_discovery(
    *,
    snapshots: list[_PovSnapshot],
    benchmark_spec: SyntheticBenchmarkSpec,
    config: DiscoverySemanticQualityConfig,
    turns_run: object,
    usage_collector: UsageCollector | None = None,
) -> _HistoryDiscoveryOutcome:
    benchmark_themes_text = format_benchmark_themes(benchmark_spec)
    first_seen_turn_by_pov: dict[str, int] = {}
    first_valid_discovery_turn_by_theme: dict[str, int] = {}
    history_judged_povs: list[dict[str, Any]] = []

    for snapshot in snapshots:
        for pov in snapshot.povs:
            pov_key = pov.lower()
            if pov_key in first_seen_turn_by_pov:
                continue
            first_seen_turn_by_pov[pov_key] = snapshot.turn_index
            judgment = judge_theme_match(
                surfaced_topic=pov,
                benchmark_themes=benchmark_themes_text,
                model_config=config.model_config,
                prompt_id=config.theme_match_prompt_id,
                usage_collector=usage_collector,
            )
            normalized_theme_id = (
                judgment.benchmark_theme_id.strip()
                if isinstance(judgment.benchmark_theme_id, str)
                and judgment.benchmark_theme_id.strip()
                else None
            )
            matched_theme = (
                benchmark_spec.theme_by_id.get(normalized_theme_id)
                if normalized_theme_id is not None
                else None
            )
            supported = (
                matched_theme is not None and judgment.supported and not matched_theme.is_foil
            )
            if matched_theme is not None and supported:
                prior_turn = first_valid_discovery_turn_by_theme.get(matched_theme.theme_id)
                if prior_turn is None or snapshot.turn_index < prior_turn:
                    first_valid_discovery_turn_by_theme[matched_theme.theme_id] = (
                        snapshot.turn_index
                    )
            history_judged_povs.append(
                {
                    "pov": pov,
                    "first_seen_turn": snapshot.turn_index,
                    "supported": supported,
                    "benchmark_theme_id": matched_theme.theme_id
                    if matched_theme is not None
                    else None,
                    "confidence": clamp_probability(judgment.confidence),
                    "reasoning": judgment.reasoning,
                }
            )

    max_snapshot_turn = max((snapshot.turn_index for snapshot in snapshots), default=0)
    total_turns_evaluated = max_snapshot_turn
    if (
        isinstance(turns_run, int)
        and not isinstance(turns_run, bool)
        and turns_run >= max_snapshot_turn
    ):
        total_turns_evaluated = turns_run

    num_valid_themes = len(benchmark_spec.valid_themes)
    if total_turns_evaluated == 0 or num_valid_themes == 0:
        return _HistoryDiscoveryOutcome(
            discovery_utility_auc=0.0,
            total_turns_evaluated=total_turns_evaluated,
            discovery_utility_curve=[],
            first_valid_discovery_turn_by_theme=first_valid_discovery_turn_by_theme,
            history_judged_povs=history_judged_povs,
        )

    discovery_utility_curve: list[dict[str, float | int]] = []
    cumulative_utility = 0.0
    for turn_index in range(1, total_turns_evaluated + 1):
        num_discovered_themes = sum(
            1
            for first_turn in first_valid_discovery_turn_by_theme.values()
            if first_turn <= turn_index
        )
        utility = num_discovered_themes / num_valid_themes
        cumulative_utility += utility
        discovery_utility_curve.append({"turn_index": turn_index, "utility": utility})

    return _HistoryDiscoveryOutcome(
        discovery_utility_auc=cumulative_utility / total_turns_evaluated,
        total_turns_evaluated=total_turns_evaluated,
        discovery_utility_curve=discovery_utility_curve,
        first_valid_discovery_turn_by_theme=first_valid_discovery_turn_by_theme,
        history_judged_povs=history_judged_povs,
    )
