"""Shared benchmark-theme resolution helpers for benchmark-aware evaluators."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from conversational_clustering.accounting import UsageCollector
from conversational_clustering.evaluation.benchmark_spec import (
    ReferenceClusteringSpec,
    SyntheticBenchmarkSpec,
    ThemeSpec,
)
from conversational_clustering.evaluation.methods.benchmark_utils import (
    compare_clusterings,
    is_auto_topic_discovery_branch,
    resolve_active_theme,
)
from conversational_clustering.models import ModelConfig


@dataclass(frozen=True)
class ResolvedBenchmarkTheme:
    theme: ThemeSpec
    resolution_mode: str


@dataclass(frozen=True)
class ResolvedBenchmarkReference:
    theme: ThemeSpec
    reference: ReferenceClusteringSpec
    comparison_metrics: dict[str, float]
    comparison_details: dict[str, Any]
    resolution_mode: str


def build_theme_resolution_details(
    *,
    artifacts: Mapping[str, Any],
    resolved_theme: ThemeSpec,
    resolution_mode: str,
    prefix: str = "",
) -> dict[str, Any]:
    details: dict[str, Any] = {
        f"{prefix}resolved_theme_id": resolved_theme.theme_id,
        f"{prefix}resolved_theme_label": resolved_theme.label,
        f"{prefix}resolved_theme_guidance": resolved_theme.guidance,
        f"{prefix}topic_resolution_mode": resolution_mode,
    }
    original_topic = _extract_original_topic(artifacts)
    if original_topic["topic_id"] is not None:
        details[f"{prefix}original_topic_id"] = original_topic["topic_id"]
    if original_topic["label"] is not None:
        details[f"{prefix}original_topic_label"] = original_topic["label"]
    if original_topic["guidance"] is not None:
        details[f"{prefix}original_topic_guidance"] = original_topic["guidance"]
    return details


def resolve_theme_for_request(
    *,
    artifacts: Mapping[str, Any],
    benchmark_spec: SyntheticBenchmarkSpec,
    evaluator_name: str,
    model_config: ModelConfig | None,
    theme_match_prompt_id: str,
    theme_matcher: Callable[..., Any],
    usage_collector: UsageCollector | None = None,
    predictions: Any | None = None,
    granularity: str | None = None,
    allow_best_reference_fallback: bool = False,
) -> ResolvedBenchmarkTheme:
    active_theme = resolve_active_theme(artifacts=artifacts, benchmark_spec=benchmark_spec)
    if active_theme is not None:
        return ResolvedBenchmarkTheme(
            theme=active_theme,
            resolution_mode="explicit_or_lexical",
        )

    semantic_theme = _resolve_semantic_active_theme(
        artifacts=artifacts,
        benchmark_spec=benchmark_spec,
        model_config=model_config,
        prompt_id=theme_match_prompt_id,
        theme_matcher=theme_matcher,
        usage_collector=usage_collector,
    )
    if semantic_theme is not None:
        return ResolvedBenchmarkTheme(
            theme=semantic_theme,
            resolution_mode="semantic_judge",
        )

    if (
        allow_best_reference_fallback
        and is_auto_topic_discovery_branch(artifacts)
        and predictions is not None
    ):
        best_resolution = _select_best_matching_reference(
            predictions=predictions,
            benchmark_spec=benchmark_spec,
            granularity=granularity,
            evaluator_name=evaluator_name,
        )
        return ResolvedBenchmarkTheme(
            theme=best_resolution.theme,
            resolution_mode=best_resolution.resolution_mode,
        )

    raise ValueError(
        f"Evaluator '{evaluator_name}' could not resolve the requested benchmark theme from "
        "selected_topic, batch-run metadata, explicit benchmark theme ids, "
        "or semantic theme matching. "
        "For guided benchmarked runs, provide 'benchmark_theme_id' or "
        "'topic_conditioned_theme_id' in metadata."
    )


def resolve_reference_for_request(
    *,
    predictions: Any,
    artifacts: Mapping[str, Any],
    benchmark_spec: SyntheticBenchmarkSpec,
    granularity: str | None,
    evaluator_name: str,
    model_config: ModelConfig | None,
    theme_match_prompt_id: str,
    theme_matcher: Callable[..., Any],
    usage_collector: UsageCollector | None = None,
) -> ResolvedBenchmarkReference:
    try:
        resolved_theme = resolve_theme_for_request(
            artifacts=artifacts,
            benchmark_spec=benchmark_spec,
            evaluator_name=evaluator_name,
            model_config=model_config,
            theme_match_prompt_id=theme_match_prompt_id,
            theme_matcher=theme_matcher,
            usage_collector=usage_collector,
        )
    except ValueError:
        if not is_auto_topic_discovery_branch(artifacts):
            raise
        return _select_best_matching_reference(
            predictions=predictions,
            benchmark_spec=benchmark_spec,
            granularity=granularity,
            evaluator_name=evaluator_name,
        )
    return _resolve_theme_reference(
        theme=resolved_theme.theme,
        predictions=predictions,
        benchmark_spec=benchmark_spec,
        granularity=granularity,
        resolution_mode=resolved_theme.resolution_mode,
    )


def _select_best_matching_reference(
    *,
    predictions: Any,
    benchmark_spec: SyntheticBenchmarkSpec,
    granularity: str | None,
    evaluator_name: str,
    pov_group_id: str | None = None,
) -> ResolvedBenchmarkReference:
    candidate_references = [
        reference
        for reference in benchmark_spec.references
        if (granularity is None or reference.granularity == granularity)
        and (pov_group_id is None or reference.pov_group_id == pov_group_id)
        and not benchmark_spec.theme_by_id[reference.topic_id].is_foil
    ]
    if not candidate_references:
        granularity_label = granularity if granularity is not None else "any"
        raise ValueError(
            f"Evaluator '{evaluator_name}' found no benchmark references for granularity "
            f"'{granularity_label}'."
        )

    best_resolution: ResolvedBenchmarkReference | None = None
    best_sort_key: tuple[float, float, float, int, int, str, str] | None = None
    saw_overlap = False
    for reference in candidate_references:
        try:
            comparison_metrics, comparison_details = compare_clusterings(
                left=predictions,
                right=reference.to_loaded_clustering(),
            )
        except ValueError as exc:
            if "no overlapping items" not in str(exc):
                raise
            continue
        saw_overlap = True
        theme = benchmark_spec.theme_by_id[reference.topic_id]
        requested_granularity_preference = int(
            granularity is not None and reference.granularity == granularity
        )
        normal_granularity_preference = int(reference.granularity == "normal")
        sort_key = (
            comparison_metrics["adjusted_rand_index"],
            comparison_metrics["normalized_mutual_info"],
            -comparison_metrics["variation_of_information"],
            requested_granularity_preference,
            normal_granularity_preference,
            theme.theme_id,
            reference.reference_id,
        )
        if best_sort_key is None or sort_key > best_sort_key:
            best_sort_key = sort_key
            best_resolution = ResolvedBenchmarkReference(
                theme=theme,
                reference=reference,
                comparison_metrics=comparison_metrics,
                comparison_details=comparison_details,
                resolution_mode="best_reference_fallback",
            )

    if best_resolution is not None:
        return best_resolution

    if not saw_overlap:
        raise ValueError(
            f"Evaluator '{evaluator_name}' could not compare predictions against any benchmark "
            "reference because there were no overlapping items."
        )

    raise ValueError(
        f"Evaluator '{evaluator_name}' could not resolve a benchmark reference for "
        f"granularity '{granularity}'."
    )


def _resolve_theme_reference(
    *,
    theme: ThemeSpec,
    predictions: Any,
    benchmark_spec: SyntheticBenchmarkSpec,
    granularity: str | None,
    resolution_mode: str,
) -> ResolvedBenchmarkReference:
    if theme.pov_group_id is not None:
        grouped_references = benchmark_spec.references_for_pov_group(theme.pov_group_id)
        if grouped_references:
            best_group_match = _select_best_matching_reference(
                predictions=predictions,
                benchmark_spec=benchmark_spec,
                granularity=granularity,
                evaluator_name="benchmark_theme_resolution",
                pov_group_id=theme.pov_group_id,
            )
            return ResolvedBenchmarkReference(
                theme=theme,
                reference=best_group_match.reference,
                comparison_metrics=best_group_match.comparison_metrics,
                comparison_details=best_group_match.comparison_details,
                resolution_mode=resolution_mode,
            )

    resolved_granularity = granularity or theme.granularity
    reference = benchmark_spec.resolve_reference(
        topic_id=theme.theme_id,
        granularity=resolved_granularity,
    )
    comparison_metrics, comparison_details = compare_clusterings(
        left=predictions,
        right=reference.to_loaded_clustering(),
    )
    return ResolvedBenchmarkReference(
        theme=theme,
        reference=reference,
        comparison_metrics=comparison_metrics,
        comparison_details=comparison_details,
        resolution_mode=resolution_mode,
    )


def _resolve_semantic_active_theme(
    *,
    artifacts: Mapping[str, Any],
    benchmark_spec: SyntheticBenchmarkSpec,
    model_config: ModelConfig | None,
    prompt_id: str,
    theme_matcher: Callable[..., Any],
    usage_collector: UsageCollector | None = None,
) -> ThemeSpec | None:
    surfaced_topic = _format_active_topic_for_semantic_match(artifacts)
    if surfaced_topic is None:
        return None

    judgment = theme_matcher(
        surfaced_topic=surfaced_topic,
        benchmark_themes=_format_benchmark_themes(benchmark_spec),
        model_config=model_config,
        prompt_id=prompt_id,
        usage_collector=usage_collector,
    )
    if judgment is None:
        return None
    raw_theme_id = getattr(judgment, "benchmark_theme_id", None)
    normalized_theme_id = (
        raw_theme_id.strip() if isinstance(raw_theme_id, str) and raw_theme_id.strip() else None
    )
    matched_theme = (
        benchmark_spec.theme_by_id.get(normalized_theme_id)
        if normalized_theme_id is not None
        else None
    )
    if (
        bool(getattr(judgment, "supported", False))
        and matched_theme is not None
        and not matched_theme.is_foil
    ):
        return matched_theme
    return None


def _format_active_topic_for_semantic_match(artifacts: Mapping[str, Any]) -> str | None:
    auto_topic_branch = is_auto_topic_discovery_branch(artifacts)
    candidates: list[str] = []

    selected_topic = artifacts.get("selected_topic")
    if isinstance(selected_topic, Mapping):
        formatted = _format_topic_mapping(
            selected_topic,
            id_key="topic_id",
            label_key="label",
            guidance_key="guidance",
        )
        if formatted is not None:
            candidates.append(formatted)

    batch_run_config = artifacts.get("batch_run_config")
    if isinstance(batch_run_config, Mapping):
        metadata = batch_run_config.get("metadata")
        if isinstance(metadata, Mapping):
            formatted = _format_topic_mapping(
                metadata,
                id_key="topic_branch_topic_id",
                label_key="topic_branch_topic_label",
                guidance_key="topic_branch_topic_guidance",
            )
            if formatted is not None:
                candidates.append(formatted)
        if not auto_topic_branch:
            human_agent_config = batch_run_config.get("human_agent_config")
            if isinstance(human_agent_config, Mapping):
                raw_guidance = human_agent_config.get("guidance")
                if isinstance(raw_guidance, str) and raw_guidance.strip():
                    candidates.append(f"guidance={raw_guidance.strip()}")

    deduped_candidates = list(dict.fromkeys(candidates))
    if not deduped_candidates:
        return None
    return "\n".join(deduped_candidates)


def _extract_original_topic(artifacts: Mapping[str, Any]) -> dict[str, str | None]:
    selected_topic = artifacts.get("selected_topic")
    if isinstance(selected_topic, Mapping):
        topic_id = _get_non_empty_str(selected_topic, "topic_id")
        label = _get_non_empty_str(selected_topic, "label")
        guidance = _get_non_empty_str(selected_topic, "guidance")
        if topic_id is not None or label is not None or guidance is not None:
            return {
                "topic_id": topic_id,
                "label": label,
                "guidance": guidance,
            }

    batch_run_config = artifacts.get("batch_run_config")
    if isinstance(batch_run_config, Mapping):
        metadata = batch_run_config.get("metadata")
        if isinstance(metadata, Mapping):
            topic_id = _get_non_empty_str(metadata, "topic_branch_topic_id")
            if topic_id is None:
                topic_id = _get_non_empty_str(metadata, "benchmark_theme_id")
            if topic_id is None:
                topic_id = _get_non_empty_str(metadata, "topic_conditioned_theme_id")
            label = _get_non_empty_str(metadata, "topic_branch_topic_label")
            guidance = _get_non_empty_str(metadata, "topic_branch_topic_guidance")
            if topic_id is not None or label is not None or guidance is not None:
                return {
                    "topic_id": topic_id,
                    "label": label,
                    "guidance": guidance,
                }
        human_agent_config = batch_run_config.get("human_agent_config")
        if isinstance(human_agent_config, Mapping):
            guidance = _get_non_empty_str(human_agent_config, "guidance")
            if guidance is not None:
                return {
                    "topic_id": None,
                    "label": None,
                    "guidance": guidance,
                }

    return {
        "topic_id": None,
        "label": None,
        "guidance": None,
    }


def _format_topic_mapping(
    payload: Mapping[str, Any],
    *,
    id_key: str,
    label_key: str,
    guidance_key: str,
) -> str | None:
    parts: list[str] = []
    raw_id = payload.get(id_key)
    raw_label = payload.get(label_key)
    raw_guidance = payload.get(guidance_key)
    if isinstance(raw_id, str) and raw_id.strip():
        parts.append(f"id={raw_id.strip()}")
    if isinstance(raw_label, str) and raw_label.strip():
        parts.append(f"label={raw_label.strip()}")
    if isinstance(raw_guidance, str) and raw_guidance.strip():
        parts.append(f"guidance={raw_guidance.strip()}")
    if not parts:
        return None
    return " | ".join(parts)


def _get_non_empty_str(payload: Mapping[str, Any], key: str) -> str | None:
    raw_value = payload.get(key)
    if isinstance(raw_value, str) and raw_value.strip():
        return raw_value.strip()
    return None


def _format_benchmark_themes(benchmark_spec: SyntheticBenchmarkSpec) -> str:
    lines: list[str] = []
    for theme in benchmark_spec.themes:
        lines.append(
            f"- theme_id={theme.theme_id}; label={theme.label}; guidance={theme.guidance}; "
            f"is_foil={theme.is_foil}"
        )
    return "\n".join(lines)
