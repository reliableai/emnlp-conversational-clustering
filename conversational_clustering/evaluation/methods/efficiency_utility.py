"""Efficiency evaluator over best-so-far structural and semantic utility across turns."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from conversational_clustering.accounting import UsageCollector, build_pricing_client_from_configs
from conversational_clustering.evaluation.base import EvaluationRequest, EvaluationResult, Evaluator
from conversational_clustering.evaluation.benchmark_spec import ReferenceClusteringSpec
from conversational_clustering.evaluation.methods.benchmark_theme_resolution import (
    build_theme_resolution_details,
)
from conversational_clustering.evaluation.methods.benchmark_utils import (
    collect_snapshot_clusterings,
    compare_clusterings,
    require_active_theme,
    require_benchmark_spec,
    resolve_final_turn_index,
)
from conversational_clustering.evaluation.methods.cluster_semantic_quality import (
    ClusterSemanticQualityEvaluator,
)
from conversational_clustering.evaluation.methods.topic_conditioned_quality import (
    _resolve_reference_for_request,
)
from conversational_clustering.models import (
    ModelConfig,
    default_semantic_consistency_judge_model_config,
    parse_model_config,
)
from conversational_clustering.prompts.registry import get_discovery_semantic_theme_match_prompt

_SUPPORTED_SEMANTIC_SNAPSHOT_MODES = frozenset({"all", "final_only"})
logger = logging.getLogger("conversational_clustering")


def _emit_progress(message: str) -> None:
    logger.info(message)


@dataclass(frozen=True)
class EfficiencyUtilityEvaluator(Evaluator):
    name = "efficiency_utility"
    resume_version = "v2"
    granularity: str | None = None
    threshold: float | None = None
    include_semantic_utility: bool = False
    model_config: ModelConfig | None = None
    coherence_prompt_id: str = "baseline_v1"
    distinctiveness_prompt_id: str = "baseline_v1"
    semantic_snapshot_mode: str = "all"

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None = None) -> EfficiencyUtilityEvaluator:
        normalized_params = dict(params or {})
        supported_keys = {
            "granularity",
            "threshold",
            "include_semantic_utility",
            "model_config",
            "coherence_prompt_id",
            "distinctiveness_prompt_id",
            "semantic_snapshot_mode",
        }
        unsupported_keys = sorted(set(normalized_params) - supported_keys)
        if unsupported_keys:
            raise ValueError(
                f"Evaluator '{cls.name}' received unsupported parameter(s): "
                + ", ".join(unsupported_keys)
            )
        raw_granularity = normalized_params.get("granularity")
        if raw_granularity is not None and (
            not isinstance(raw_granularity, str) or not raw_granularity.strip()
        ):
            raise ValueError("'granularity' must be a non-empty string")
        threshold: float | None = None
        if "threshold" in normalized_params:
            raw_threshold = normalized_params["threshold"]
            if isinstance(raw_threshold, bool) or not isinstance(raw_threshold, int | float):
                raise ValueError("'threshold' must be a numeric score threshold")
            threshold = float(raw_threshold)
            if not math.isfinite(threshold) or threshold < -1.0 or threshold > 1.0:
                raise ValueError("'threshold' must be a finite number between -1.0 and 1.0")
        raw_include_semantic = normalized_params.get("include_semantic_utility", False)
        if not isinstance(raw_include_semantic, bool):
            raise ValueError("'include_semantic_utility' must be a boolean")
        raw_model_config = normalized_params.get("model_config")
        model_config: ModelConfig | None = None
        if raw_model_config is not None:
            if not isinstance(raw_model_config, Mapping):
                raise TypeError("model_config must be a dict")
            model_config = parse_model_config(raw_model_config)
        if raw_include_semantic:
            ClusterSemanticQualityEvaluator.from_params(
                {
                    "model_config": raw_model_config,
                    "coherence_prompt_id": str(
                        normalized_params.get("coherence_prompt_id", "baseline_v1")
                    ).strip(),
                    "distinctiveness_prompt_id": str(
                        normalized_params.get("distinctiveness_prompt_id", "baseline_v1")
                    ).strip(),
                }
            )
        return cls(
            granularity=raw_granularity.strip() if isinstance(raw_granularity, str) else None,
            threshold=threshold,
            include_semantic_utility=raw_include_semantic,
            model_config=model_config,
            coherence_prompt_id=str(
                normalized_params.get("coherence_prompt_id", "baseline_v1")
            ).strip(),
            distinctiveness_prompt_id=str(
                normalized_params.get("distinctiveness_prompt_id", "baseline_v1")
            ).strip(),
            semantic_snapshot_mode=_parse_semantic_snapshot_mode(normalized_params),
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        usage_collector = UsageCollector()
        benchmark_spec = require_benchmark_spec(request.artifacts, evaluator_name=self.name)
        _emit_progress(f"[evaluation] {self.name}: resolving active theme")
        active_theme = require_active_theme(
            artifacts=request.artifacts,
            benchmark_spec=benchmark_spec,
            evaluator_name=self.name,
        )
        _emit_progress(f"[evaluation] {self.name}: resolved active theme '{active_theme.theme_id}'")
        return _evaluate_efficiency_utility(
            request=request,
            evaluator_name=self.name,
            granularity=self.granularity,
            threshold=self.threshold,
            include_semantic_utility=self.include_semantic_utility,
            model_config=self.model_config,
            coherence_prompt_id=self.coherence_prompt_id,
            distinctiveness_prompt_id=self.distinctiveness_prompt_id,
            semantic_snapshot_mode=self.semantic_snapshot_mode,
            active_theme=active_theme,
            resolution_details=None,
            usage_collector=usage_collector,
        )


@dataclass(frozen=True)
class InferredThemeEfficiencyUtilityEvaluator(Evaluator):
    name = "inferred_theme_efficiency_utility"
    resume_version = "v2"
    granularity: str | None = None
    threshold: float | None = None
    include_semantic_utility: bool = False
    model_config: ModelConfig | None = None
    coherence_prompt_id: str = "baseline_v1"
    distinctiveness_prompt_id: str = "baseline_v1"
    theme_match_prompt_id: str = "baseline_v1"
    semantic_snapshot_mode: str = "all"

    @classmethod
    def from_params(
        cls, params: Mapping[str, Any] | None = None
    ) -> InferredThemeEfficiencyUtilityEvaluator:
        normalized_params = dict(params or {})
        supported_keys = {
            "granularity",
            "threshold",
            "include_semantic_utility",
            "model_config",
            "coherence_prompt_id",
            "distinctiveness_prompt_id",
            "theme_match_prompt_id",
            "semantic_snapshot_mode",
        }
        unsupported_keys = sorted(set(normalized_params) - supported_keys)
        if unsupported_keys:
            raise ValueError(
                f"Evaluator '{cls.name}' received unsupported parameter(s): "
                + ", ".join(unsupported_keys)
            )
        raw_granularity = normalized_params.get("granularity")
        if raw_granularity is not None and (
            not isinstance(raw_granularity, str) or not raw_granularity.strip()
        ):
            raise ValueError("'granularity' must be a non-empty string")
        threshold: float | None = None
        if "threshold" in normalized_params:
            raw_threshold = normalized_params["threshold"]
            if isinstance(raw_threshold, bool) or not isinstance(raw_threshold, int | float):
                raise ValueError("'threshold' must be a numeric score threshold")
            threshold = float(raw_threshold)
            if not math.isfinite(threshold) or threshold < -1.0 or threshold > 1.0:
                raise ValueError("'threshold' must be a finite number between -1.0 and 1.0")
        raw_include_semantic = normalized_params.get("include_semantic_utility", False)
        if not isinstance(raw_include_semantic, bool):
            raise ValueError("'include_semantic_utility' must be a boolean")
        raw_model_config = normalized_params.get("model_config")
        model_config: ModelConfig | None = None
        if raw_model_config is not None:
            if not isinstance(raw_model_config, Mapping):
                raise TypeError("model_config must be a dict")
            model_config = parse_model_config(raw_model_config)
        theme_match_prompt_id = str(
            normalized_params.get("theme_match_prompt_id", "baseline_v1")
        ).strip()
        try:
            get_discovery_semantic_theme_match_prompt(theme_match_prompt_id)
        except ValueError as exc:
            raise ValueError(f"Invalid theme_match_prompt_id: {exc}") from exc
        if raw_include_semantic:
            ClusterSemanticQualityEvaluator.from_params(
                {
                    "model_config": raw_model_config,
                    "coherence_prompt_id": str(
                        normalized_params.get("coherence_prompt_id", "baseline_v1")
                    ).strip(),
                    "distinctiveness_prompt_id": str(
                        normalized_params.get("distinctiveness_prompt_id", "baseline_v1")
                    ).strip(),
                }
            )
        return cls(
            granularity=raw_granularity.strip() if isinstance(raw_granularity, str) else None,
            threshold=threshold,
            include_semantic_utility=raw_include_semantic,
            model_config=model_config,
            coherence_prompt_id=str(
                normalized_params.get("coherence_prompt_id", "baseline_v1")
            ).strip(),
            distinctiveness_prompt_id=str(
                normalized_params.get("distinctiveness_prompt_id", "baseline_v1")
            ).strip(),
            theme_match_prompt_id=theme_match_prompt_id,
            semantic_snapshot_mode=_parse_semantic_snapshot_mode(normalized_params),
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        usage_collector = UsageCollector()
        benchmark_spec = require_benchmark_spec(request.artifacts, evaluator_name=self.name)
        _emit_progress(f"[evaluation] {self.name}: starting inferred theme resolution")
        resolution = _resolve_reference_for_request(
            predictions=request.predictions,
            artifacts=request.artifacts,
            benchmark_spec=benchmark_spec,
            granularity=self.granularity,
            evaluator_name=self.name,
            model_config=self.model_config,
            theme_match_prompt_id=self.theme_match_prompt_id,
            usage_collector=usage_collector,
        )
        _emit_progress(
            "[evaluation] "
            f"{self.name}: resolved theme '{resolution.theme.theme_id}' via "
            f"{resolution.resolution_mode}"
        )
        return _evaluate_efficiency_utility(
            request=request,
            evaluator_name=self.name,
            granularity=self.granularity,
            threshold=self.threshold,
            include_semantic_utility=self.include_semantic_utility,
            model_config=self.model_config,
            coherence_prompt_id=self.coherence_prompt_id,
            distinctiveness_prompt_id=self.distinctiveness_prompt_id,
            semantic_snapshot_mode=self.semantic_snapshot_mode,
            active_theme=resolution.theme,
            reference=resolution.reference,
            resolution_details=build_theme_resolution_details(
                artifacts=request.artifacts,
                resolved_theme=resolution.theme,
                resolution_mode=resolution.resolution_mode,
            ),
            usage_collector=usage_collector,
        )


def _evaluate_efficiency_utility(
    *,
    request: EvaluationRequest,
    evaluator_name: str,
    granularity: str | None,
    threshold: float | None,
    include_semantic_utility: bool,
    model_config: ModelConfig | None,
    coherence_prompt_id: str,
    distinctiveness_prompt_id: str,
    semantic_snapshot_mode: str,
    active_theme: Any,
    resolution_details: Mapping[str, Any] | None,
    usage_collector: UsageCollector,
    reference: ReferenceClusteringSpec | None = None,
) -> EvaluationResult:
    benchmark_spec = require_benchmark_spec(request.artifacts, evaluator_name=evaluator_name)
    if reference is not None:
        resolved_reference = reference
    elif active_theme.pov_group_id is not None and granularity is None:
        grouped_references = list(
            benchmark_spec.references_for_pov_group(active_theme.pov_group_id)
        )
        if not grouped_references:
            raise ValueError(
                f"Evaluator '{evaluator_name}' found no benchmark references for "
                f"pov_group_id='{active_theme.pov_group_id}'"
            )
        resolved_reference = max(
            grouped_references,
            key=lambda candidate: compare_clusterings(
                left=request.predictions,
                right=candidate.to_loaded_clustering(),
            )[0]["adjusted_rand_index"],
        )
    else:
        resolved_reference = benchmark_spec.resolve_reference(
            topic_id=active_theme.theme_id,
            granularity=granularity or active_theme.granularity,
        )
    snapshots = collect_snapshot_clusterings(request)
    final_turn_index = resolve_final_turn_index(request, snapshots=snapshots)
    snapshots.append((final_turn_index, request.predictions))
    snapshots = _deduplicate_snapshots(snapshots)

    best_structural_seen: float | None = None
    best_structural_by_turn: list[dict[str, float | int]] = []
    turn_to_threshold_structural: float | None = None
    for turn_index, clustering in snapshots:
        comparison_metrics, _ = compare_clusterings(
            left=clustering,
            right=resolved_reference.to_loaded_clustering(),
        )
        current_score = comparison_metrics["adjusted_rand_index"]
        best_structural_seen = (
            current_score
            if best_structural_seen is None
            else max(best_structural_seen, current_score)
        )
        best_structural_by_turn.append(
            {
                "turn_index": turn_index,
                "current_score": current_score,
                "best_score": best_structural_seen,
            }
        )
        if (
            threshold is not None
            and turn_to_threshold_structural is None
            and best_structural_seen >= threshold
        ):
            turn_to_threshold_structural = float(turn_index)

    if best_structural_seen is None:
        raise ValueError(f"Evaluator '{evaluator_name}' requires at least one clustering snapshot")

    metrics = {
        "area_under_utility_curve_structural": _area_under_best_score_curve(
            best_structural_by_turn
        ),
        "best_score_seen_structural": best_structural_seen,
    }
    details: dict[str, Any] = {
        "topic_id": active_theme.theme_id,
        "reference_id": resolved_reference.reference_id,
        "granularity": resolved_reference.granularity,
        "requested_granularity": granularity,
        "best_so_far_structural_by_turn": best_structural_by_turn,
    }
    if resolution_details is not None:
        details.update(dict(resolution_details))
    if threshold is not None:
        details["threshold"] = threshold
        details["structural_threshold_reached"] = turn_to_threshold_structural is not None
        metrics["structural_threshold_reached_rate"] = (
            1.0 if turn_to_threshold_structural is not None else 0.0
        )
        if turn_to_threshold_structural is not None:
            metrics["turn_to_threshold_structural"] = turn_to_threshold_structural

    if include_semantic_utility:
        semantic_evaluator = ClusterSemanticQualityEvaluator(
            model_config=model_config,
            coherence_prompt_id=coherence_prompt_id,
            distinctiveness_prompt_id=distinctiveness_prompt_id,
        )
        semantic_artifacts = _with_resolved_theme_guidance(
            artifacts=request.artifacts,
            active_theme=active_theme,
        )
        semantic_snapshots = _select_semantic_snapshots(
            snapshots=snapshots,
            semantic_snapshot_mode=semantic_snapshot_mode,
        )
        _emit_progress(
            "[evaluation] "
            f"{evaluator_name}: semantic utility starting with "
            f"{len(semantic_snapshots)} snapshot(s) using mode={semantic_snapshot_mode}"
        )
        best_semantic_seen = 0.0
        best_semantic_by_turn: list[dict[str, float | int]] = []
        turn_to_threshold_semantic: float | None = None
        for semantic_snapshot_index, (turn_index, clustering) in enumerate(
            semantic_snapshots, start=1
        ):
            num_clusters = len(clustering.cluster_names)
            _emit_progress(
                "[evaluation] "
                f"{evaluator_name}: semantic snapshot {semantic_snapshot_index}/"
                f"{len(semantic_snapshots)} turn_index={turn_index} "
                f"clusters={num_clusters} started"
            )
            semantic_result = semantic_evaluator.evaluate_with_progress(
                EvaluationRequest(
                    predictions=clustering,
                    artifacts=semantic_artifacts,
                    metadata=request.metadata,
                ),
                progress_callback=lambda message, snapshot_index=semantic_snapshot_index: (
                    _emit_progress(
                        f"[evaluation] {evaluator_name}: snapshot "
                        f"{snapshot_index}/{len(semantic_snapshots)} {message}"
                    )
                ),
                usage_collector=usage_collector,
            )
            current_score = semantic_result.metrics["average_cluster_coherence"]
            best_semantic_seen = max(best_semantic_seen, current_score)
            best_semantic_by_turn.append(
                {
                    "turn_index": turn_index,
                    "current_score": current_score,
                    "best_score": best_semantic_seen,
                }
            )
            if (
                threshold is not None
                and turn_to_threshold_semantic is None
                and best_semantic_seen >= threshold
            ):
                turn_to_threshold_semantic = float(turn_index)
            _emit_progress(
                "[evaluation] "
                f"{evaluator_name}: semantic snapshot {semantic_snapshot_index}/"
                f"{len(semantic_snapshots)} turn_index={turn_index} "
                f"finished with average_cluster_coherence={current_score:.3f}"
            )
        if semantic_snapshot_mode != "final_only":
            metrics["area_under_utility_curve_semantic"] = _area_under_best_score_curve(
                best_semantic_by_turn
            )
        metrics["best_score_seen_semantic"] = best_semantic_seen
        details["best_so_far_semantic_by_turn"] = best_semantic_by_turn
        details["semantic_snapshot_mode"] = semantic_snapshot_mode
        if threshold is not None:
            metrics["semantic_threshold_reached_rate"] = (
                1.0 if turn_to_threshold_semantic is not None else 0.0
            )
            if turn_to_threshold_semantic is not None:
                metrics["turn_to_threshold_semantic"] = turn_to_threshold_semantic
            details["semantic_threshold_reached"] = turn_to_threshold_semantic is not None

    return EvaluationResult(
        method=evaluator_name,
        metrics=metrics,
        details=details,
        accounting=usage_collector.snapshot(
            pricer=build_pricing_client_from_configs(
                [model_config]
                if model_config
                else [default_semantic_consistency_judge_model_config()]
            )
        ).to_dict(),
    )


def _parse_semantic_snapshot_mode(params: Mapping[str, Any]) -> str:
    raw_snapshot_mode = params.get("semantic_snapshot_mode", "all")
    if not isinstance(raw_snapshot_mode, str) or not raw_snapshot_mode.strip():
        raise ValueError("'semantic_snapshot_mode' must be a non-empty string")
    semantic_snapshot_mode = raw_snapshot_mode.strip()
    if semantic_snapshot_mode not in _SUPPORTED_SEMANTIC_SNAPSHOT_MODES:
        valid = ", ".join(sorted(_SUPPORTED_SEMANTIC_SNAPSHOT_MODES))
        raise ValueError(f"'semantic_snapshot_mode' must be one of: {valid}")
    return semantic_snapshot_mode


def _select_semantic_snapshots(
    *,
    snapshots: list[tuple[int, Any]],
    semantic_snapshot_mode: str,
) -> list[tuple[int, Any]]:
    if semantic_snapshot_mode == "all":
        return snapshots
    if semantic_snapshot_mode == "final_only":
        return [snapshots[-1]]
    valid = ", ".join(sorted(_SUPPORTED_SEMANTIC_SNAPSHOT_MODES))
    raise ValueError(
        f"Unexpected semantic_snapshot_mode='{semantic_snapshot_mode}'. Expected one of: {valid}"
    )


def _deduplicate_snapshots(
    snapshots: list[tuple[int, Any]],
) -> list[tuple[int, Any]]:
    deduped: dict[int, Any] = {}
    for turn_index, clustering in snapshots:
        deduped[turn_index] = clustering
    return sorted(deduped.items(), key=lambda item: item[0])


def _area_under_best_score_curve(best_so_far_by_turn: list[dict[str, float | int]]) -> float:
    if not best_so_far_by_turn:
        return 0.0
    sorted_points = sorted(best_so_far_by_turn, key=lambda point: int(point["turn_index"]))
    last_turn = int(sorted_points[-1]["turn_index"])
    if last_turn <= 0:
        return float(sorted_points[-1]["best_score"])
    area = 0.0
    previous_turn = 0
    previous_score = 0.0
    for point in sorted_points:
        turn_index = int(point["turn_index"])
        best_score = float(point["best_score"])
        if turn_index > previous_turn + 1:
            area += previous_score * (turn_index - previous_turn - 1)
        area += best_score
        previous_turn = turn_index
        previous_score = best_score
    return area / last_turn


def _with_resolved_theme_guidance(
    *,
    artifacts: Mapping[str, Any],
    active_theme: Any,
) -> dict[str, Any]:
    semantic_artifacts = dict(artifacts)

    selected_topic = artifacts.get("selected_topic")
    selected_topic_payload = dict(selected_topic) if isinstance(selected_topic, Mapping) else {}
    selected_topic_payload.setdefault("topic_id", active_theme.theme_id)
    selected_topic_payload.setdefault("label", active_theme.label)
    selected_topic_payload["guidance"] = active_theme.guidance
    semantic_artifacts["selected_topic"] = selected_topic_payload

    batch_run_config = artifacts.get("batch_run_config")
    if isinstance(batch_run_config, Mapping):
        batch_run_config_payload = dict(batch_run_config)
        human_agent_config = batch_run_config.get("human_agent_config")
        human_agent_config_payload = (
            dict(human_agent_config) if isinstance(human_agent_config, Mapping) else {}
        )
        human_agent_config_payload["guidance"] = active_theme.guidance
        batch_run_config_payload["human_agent_config"] = human_agent_config_payload
        semantic_artifacts["batch_run_config"] = batch_run_config_payload

    return semantic_artifacts
