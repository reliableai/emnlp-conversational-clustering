"""Shared benchmark-aware helpers for evaluation methods."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from conversational_clustering.evaluation.artifacts import (
    LoadedClustering,
    parse_clustering_payload,
)
from conversational_clustering.evaluation.base import EvaluationRequest, PairwiseEvaluationRequest
from conversational_clustering.evaluation.benchmark_spec import (
    ReferenceClusteringSpec,
    SyntheticBenchmarkSpec,
    ThemeSpec,
    normalize_benchmark_text,
)
from conversational_clustering.evaluation.methods.metrics import (
    coassignment_jaccard,
    fowlkes_mallows,
    normalized_mutual_info,
    overlap_stats,
    pairwise_adjusted_rand_index,
    pairwise_counts,
    rand_index,
    safe_ratio,
    variation_of_information,
)

_EXPLICIT_THEME_ID_KEYS = ("benchmark_theme_id", "topic_conditioned_theme_id")


def _match_theme_by_guidance(
    *,
    raw_guidance: object,
    benchmark_spec: SyntheticBenchmarkSpec,
) -> ThemeSpec | None:
    if not isinstance(raw_guidance, str):
        return None
    normalized_guidance = normalize_benchmark_text(raw_guidance)
    if not normalized_guidance:
        return None
    for theme in benchmark_spec.themes:
        if normalize_benchmark_text(theme.guidance) == normalized_guidance:
            return theme
    return None


def is_auto_topic_discovery_branch(artifacts: Mapping[str, Any]) -> bool:
    selected_topic = artifacts.get("selected_topic")
    if isinstance(selected_topic, Mapping):
        raw_generated_at_turn = selected_topic.get("generated_at_turn")
        if isinstance(raw_generated_at_turn, int) and not isinstance(raw_generated_at_turn, bool):
            return True

    batch_run_config = artifacts.get("batch_run_config")
    if isinstance(batch_run_config, Mapping):
        metadata = batch_run_config.get("metadata")
        if isinstance(metadata, Mapping):
            raw_topic_branch_mode = metadata.get("topic_branch_mode")
            if (
                isinstance(raw_topic_branch_mode, str)
                and raw_topic_branch_mode.strip() == "auto_topic_discovery"
            ):
                return True

    return False


def _resolve_theme_by_explicit_id(
    *, payload: Mapping[str, Any], benchmark_spec: SyntheticBenchmarkSpec
) -> ThemeSpec | None:
    for key in _EXPLICIT_THEME_ID_KEYS:
        raw_theme_id = payload.get(key)
        if not isinstance(raw_theme_id, str) or not raw_theme_id.strip():
            continue
        theme = benchmark_spec.theme_by_id.get(raw_theme_id.strip())
        if theme is not None:
            return theme
    return None


def require_benchmark_spec(
    artifacts: Mapping[str, Any], *, evaluator_name: str
) -> SyntheticBenchmarkSpec:
    benchmark_spec = artifacts.get("benchmark_spec")
    if not isinstance(benchmark_spec, SyntheticBenchmarkSpec):
        raise ValueError(
            f"Evaluator '{evaluator_name}' requires a dataset benchmark spec. "
            "Configure benchmark_spec_path for the dataset and rerun batch evaluation."
        )
    return benchmark_spec


def require_pairwise_benchmark_spec(
    artifacts: Mapping[str, Any], *, evaluator_name: str
) -> SyntheticBenchmarkSpec:
    benchmark_spec = artifacts.get("benchmark_spec")
    if not isinstance(benchmark_spec, SyntheticBenchmarkSpec):
        benchmark_spec = artifacts.get("left_benchmark_spec")
    if not isinstance(benchmark_spec, SyntheticBenchmarkSpec):
        raise ValueError(
            f"Evaluator '{evaluator_name}' requires a dataset benchmark spec on pairwise artifacts."
        )
    return benchmark_spec


def resolve_active_theme(
    *,
    artifacts: Mapping[str, Any],
    benchmark_spec: SyntheticBenchmarkSpec,
) -> ThemeSpec | None:
    auto_topic_branch = is_auto_topic_discovery_branch(artifacts)
    selected_topic = artifacts.get("selected_topic")
    if isinstance(selected_topic, Mapping):
        explicit_theme = _resolve_theme_by_explicit_id(
            payload=selected_topic,
            benchmark_spec=benchmark_spec,
        )
        if explicit_theme is not None:
            return explicit_theme
        raw_topic_id = selected_topic.get("topic_id")
        if isinstance(raw_topic_id, str) and raw_topic_id.strip():
            theme = benchmark_spec.theme_by_id.get(raw_topic_id.strip())
            if theme is not None:
                return theme
        for key in ("label",):
            raw_value = selected_topic.get(key)
            if isinstance(raw_value, str):
                matched = benchmark_spec.match_theme(raw_value)
                if matched is not None:
                    return matched
        matched_by_guidance = _match_theme_by_guidance(
            raw_guidance=selected_topic.get("guidance"),
            benchmark_spec=benchmark_spec,
        )
        if matched_by_guidance is not None:
            return matched_by_guidance

    batch_run_config = artifacts.get("batch_run_config")
    if isinstance(batch_run_config, Mapping):
        metadata = batch_run_config.get("metadata")
        if isinstance(metadata, Mapping):
            explicit_theme = _resolve_theme_by_explicit_id(
                payload=metadata,
                benchmark_spec=benchmark_spec,
            )
            if explicit_theme is not None:
                return explicit_theme
            raw_topic_id = metadata.get("topic_branch_topic_id")
            if isinstance(raw_topic_id, str) and raw_topic_id.strip():
                theme = benchmark_spec.theme_by_id.get(raw_topic_id.strip())
                if theme is not None:
                    return theme
            for key in ("topic_branch_topic_label",):
                raw_value = metadata.get(key)
                if isinstance(raw_value, str):
                    matched = benchmark_spec.match_theme(raw_value)
                    if matched is not None:
                        return matched
            matched_by_guidance = _match_theme_by_guidance(
                raw_guidance=metadata.get("topic_branch_topic_guidance"),
                benchmark_spec=benchmark_spec,
            )
            if matched_by_guidance is not None:
                return matched_by_guidance
        human_agent_config = batch_run_config.get("human_agent_config")
        if isinstance(human_agent_config, Mapping) and not auto_topic_branch:
            explicit_theme = _resolve_theme_by_explicit_id(
                payload=human_agent_config,
                benchmark_spec=benchmark_spec,
            )
            if explicit_theme is not None:
                return explicit_theme
            matched_by_guidance = _match_theme_by_guidance(
                raw_guidance=human_agent_config.get("guidance"),
                benchmark_spec=benchmark_spec,
            )
            if matched_by_guidance is not None:
                return matched_by_guidance
    return None


def require_active_theme(
    *,
    artifacts: Mapping[str, Any],
    benchmark_spec: SyntheticBenchmarkSpec,
    evaluator_name: str,
) -> ThemeSpec:
    theme = resolve_active_theme(artifacts=artifacts, benchmark_spec=benchmark_spec)
    if theme is None:
        raise ValueError(
            f"Evaluator '{evaluator_name}' could not resolve the active topic from selected_topic "
            "artifact or batch-run config metadata."
        )
    return theme


def compare_clusterings(
    *,
    left: LoadedClustering,
    right: LoadedClustering,
) -> tuple[dict[str, float], dict[str, Any]]:
    overlap = overlap_stats(left.assignment, right.assignment)
    if overlap.num_common_items == 0:
        raise ValueError(
            "Cannot compare clusterings with no overlapping items. "
            "Predictions and benchmark reference must cover at least one shared item."
        )

    counts = pairwise_counts(left.assignment, right.assignment, overlap.common_items)
    return (
        {
            "adjusted_rand_index": pairwise_adjusted_rand_index(
                true_positive_pairs=counts.true_positive_pairs,
                left_positive_pairs=counts.left_positive_pairs,
                right_positive_pairs=counts.right_positive_pairs,
                num_item_pairs_compared=counts.num_item_pairs_compared,
            ),
            "normalized_mutual_info": normalized_mutual_info(
                contingency=counts.contingency,
                left_cluster_sizes=counts.left_cluster_sizes,
                right_cluster_sizes=counts.right_cluster_sizes,
                num_common_items=overlap.num_common_items,
            ),
            "variation_of_information": variation_of_information(
                contingency=counts.contingency,
                left_cluster_sizes=counts.left_cluster_sizes,
                right_cluster_sizes=counts.right_cluster_sizes,
                num_common_items=overlap.num_common_items,
            ),
            "rand_index": rand_index(
                true_positive_pairs=counts.true_positive_pairs,
                true_negative_pairs=counts.true_negative_pairs,
                num_item_pairs_compared=counts.num_item_pairs_compared,
            ),
            "coassignment_jaccard": coassignment_jaccard(
                true_positive_pairs=counts.true_positive_pairs,
                false_positive_pairs=counts.false_positive_pairs,
                false_negative_pairs=counts.false_negative_pairs,
            ),
            "fowlkes_mallows": fowlkes_mallows(
                true_positive_pairs=counts.true_positive_pairs,
                false_positive_pairs=counts.false_positive_pairs,
                false_negative_pairs=counts.false_negative_pairs,
            ),
        },
        {
            "num_common_items": overlap.num_common_items,
            "num_union_items": overlap.num_union_items,
            "common_item_coverage": safe_ratio(
                overlap.num_common_items,
                overlap.num_union_items,
                default=0.0,
            ),
            "same_item_set": overlap.num_common_items == overlap.num_union_items,
            "left_num_clusters_on_common_items": len(counts.left_cluster_sizes),
            "right_num_clusters_on_common_items": len(counts.right_cluster_sizes),
            "cluster_count_abs_diff_on_common_items": abs(
                len(counts.left_cluster_sizes) - len(counts.right_cluster_sizes)
            ),
            "num_item_pairs_compared": counts.num_item_pairs_compared,
            "true_positive_pairs": counts.true_positive_pairs,
            "true_negative_pairs": counts.true_negative_pairs,
            "false_positive_pairs": counts.false_positive_pairs,
            "false_negative_pairs": counts.false_negative_pairs,
        },
    )


def parse_judge_snapshot_clustering(raw_payload: object) -> LoadedClustering:
    if not isinstance(raw_payload, Mapping):
        raise ValueError("judge snapshot clustering must be an object")
    return parse_clustering_payload(cast(Mapping[str, Any], raw_payload))


def collect_snapshot_clusterings(request: EvaluationRequest) -> list[tuple[int, LoadedClustering]]:
    judge_clustering_trace = request.artifacts.get("judge_clustering_trace")
    if not isinstance(judge_clustering_trace, Mapping):
        raise ValueError(
            "This evaluator requires 'judge_clustering_trace' in evaluation request artifacts"
        )
    raw_clusterings = judge_clustering_trace.get("clusterings")
    if not isinstance(raw_clusterings, list) or not raw_clusterings:
        raise ValueError("judge_clustering_trace.clusterings must be a non-empty list")
    snapshots: list[tuple[int, LoadedClustering]] = []
    for raw_snapshot in raw_clusterings:
        if not isinstance(raw_snapshot, Mapping):
            continue
        raw_turn_index = raw_snapshot.get("turn_index")
        raw_clustering = raw_snapshot.get("clustering")
        if isinstance(raw_turn_index, bool) or not isinstance(raw_turn_index, int):
            continue
        try:
            clustering = parse_judge_snapshot_clustering(raw_clustering)
        except ValueError:
            continue
        snapshots.append((raw_turn_index, clustering))
    if not snapshots:
        raise ValueError("judge_clustering_trace did not contain any valid clustering snapshots")
    return sorted(snapshots, key=lambda item: item[0])


def resolve_final_turn_index(
    request: EvaluationRequest,
    *,
    snapshots: list[tuple[int, LoadedClustering]],
) -> int:
    conversation_trace = request.artifacts.get("conversation_trace")
    if isinstance(conversation_trace, Mapping):
        raw_messages = conversation_trace.get("messages")
        if isinstance(raw_messages, list):
            human_turn_count = sum(
                1
                for raw_message in raw_messages
                if isinstance(raw_message, Mapping) and raw_message.get("role") == "human"
            )
            if human_turn_count > 0:
                return human_turn_count
    if snapshots:
        return max(turn_index for turn_index, _ in snapshots) + 1
    return 1


def resolve_pairwise_active_theme(
    *,
    artifacts: Mapping[str, Any],
    benchmark_spec: SyntheticBenchmarkSpec,
    side: str,
) -> ThemeSpec | None:
    side_artifacts = {
        "selected_topic": artifacts.get(f"{side}_selected_topic"),
        "batch_run_config": artifacts.get(f"{side}_batch_run_config"),
    }
    return resolve_active_theme(artifacts=side_artifacts, benchmark_spec=benchmark_spec)


def require_pairwise_active_theme(
    *,
    artifacts: Mapping[str, Any],
    benchmark_spec: SyntheticBenchmarkSpec,
    side: str,
    evaluator_name: str,
) -> ThemeSpec:
    theme = resolve_pairwise_active_theme(
        artifacts=artifacts,
        benchmark_spec=benchmark_spec,
        side=side,
    )
    if theme is None:
        raise ValueError(f"Evaluator '{evaluator_name}' could not resolve the {side} active topic")
    return theme


def build_request_from_pairwise_side(
    request: PairwiseEvaluationRequest,
    *,
    side: str,
) -> EvaluationRequest:
    predictions = request.left_predictions if side == "left" else request.right_predictions
    artifacts = {
        "batch_run_config": request.artifacts.get(f"{side}_batch_run_config"),
        "selected_topic": request.artifacts.get(f"{side}_selected_topic"),
        "benchmark_spec": request.artifacts.get(f"{side}_benchmark_spec")
        or request.artifacts.get("benchmark_spec"),
        "judge_clustering_trace": request.artifacts.get(f"{side}_judge_clustering_trace"),
        "conversation_trace": request.artifacts.get(f"{side}_conversation_trace"),
        "topic_candidates": request.artifacts.get(f"{side}_topic_candidates"),
    }
    metadata = {
        "config_slug": request.artifacts.get(f"{side}_config_slug"),
        "run_id": request.artifacts.get(f"{side}_run_id"),
    }
    return EvaluationRequest(predictions=predictions, artifacts=artifacts, metadata=metadata)


def resolve_reference_for_active_theme(
    *,
    artifacts: Mapping[str, Any],
    benchmark_spec: SyntheticBenchmarkSpec,
    evaluator_name: str,
    granularity: str = "normal",
) -> ReferenceClusteringSpec:
    active_theme = require_active_theme(
        artifacts=artifacts,
        benchmark_spec=benchmark_spec,
        evaluator_name=evaluator_name,
    )
    if active_theme.pov_group_id is not None and granularity == "normal":
        grouped_references = benchmark_spec.references_for_pov_group(active_theme.pov_group_id)
        if grouped_references:
            for preferred_granularity in ("normal", "coarse", "fine"):
                for reference in grouped_references:
                    if reference.granularity == preferred_granularity:
                        return reference
    return benchmark_spec.resolve_reference(topic_id=active_theme.theme_id, granularity=granularity)
