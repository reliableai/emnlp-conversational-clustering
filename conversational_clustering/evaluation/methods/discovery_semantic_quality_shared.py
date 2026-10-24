"""Shared helpers for benchmark-aware semantic discovery quality evaluators."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from conversational_clustering.evaluation.benchmark_spec import SyntheticBenchmarkSpec
from conversational_clustering.evaluation.methods.semantic_quality_judges import (
    DuplicateTopicJudgment,
    ThemeMatchJudgment,
)
from conversational_clustering.models import ModelConfig, parse_model_config
from conversational_clustering.prompts.registry import (
    get_discovery_semantic_duplicate_prompt,
    get_discovery_semantic_theme_match_prompt,
)


@dataclass(frozen=True)
class DiscoverySemanticQualityConfig:
    model_config: ModelConfig | None = None
    theme_match_prompt_id: str = "baseline_v1"
    duplicate_prompt_id: str = "baseline_v1"


@dataclass(frozen=True)
class SemanticQualityOutcome:
    metrics: dict[str, float]
    details: dict[str, Any]


def parse_discovery_semantic_quality_config(
    *,
    evaluator_name: str,
    params: Mapping[str, Any] | None,
) -> DiscoverySemanticQualityConfig:
    normalized_params = dict(params or {})
    supported_keys = {"model_config", "theme_match_prompt_id", "duplicate_prompt_id"}
    unsupported_keys = sorted(set(normalized_params) - supported_keys)
    if unsupported_keys:
        raise ValueError(
            f"Evaluator '{evaluator_name}' received unsupported parameter(s): "
            + ", ".join(unsupported_keys)
        )

    model_config = _parse_optional_model_config(normalized_params.get("model_config"))
    theme_match_prompt_id = str(
        normalized_params.get("theme_match_prompt_id", "baseline_v1")
    ).strip()
    duplicate_prompt_id = str(normalized_params.get("duplicate_prompt_id", "baseline_v1")).strip()
    try:
        get_discovery_semantic_theme_match_prompt(theme_match_prompt_id)
    except ValueError as exc:
        raise ValueError(f"Invalid theme_match_prompt_id: {exc}") from exc
    try:
        get_discovery_semantic_duplicate_prompt(duplicate_prompt_id)
    except ValueError as exc:
        raise ValueError(f"Invalid duplicate_prompt_id: {exc}") from exc

    return DiscoverySemanticQualityConfig(
        model_config=model_config,
        theme_match_prompt_id=theme_match_prompt_id,
        duplicate_prompt_id=duplicate_prompt_id,
    )


def format_discovery_topic(raw_topic: object) -> str | None:
    if not isinstance(raw_topic, Mapping):
        return None
    topic_mapping = cast(Mapping[str, object], raw_topic)
    topic_id = topic_mapping.get("topic_id")
    label = topic_mapping.get("label")
    guidance = topic_mapping.get("guidance")
    parts: list[str] = []
    if isinstance(topic_id, str) and topic_id.strip():
        parts.append(f"id={topic_id.strip()}")
    if isinstance(label, str) and label.strip():
        parts.append(f"label={label.strip()}")
    if isinstance(guidance, str) and guidance.strip():
        parts.append(f"guidance={guidance.strip()}")
    if not parts:
        return None
    return " | ".join(parts)


def evaluate_surfaced_topics_semantic_quality(
    *,
    surfaced_topics: Sequence[str],
    benchmark_spec: SyntheticBenchmarkSpec,
    config: DiscoverySemanticQualityConfig,
    theme_match_judge: Callable[..., ThemeMatchJudgment],
    duplicate_judge: Callable[..., DuplicateTopicJudgment],
) -> SemanticQualityOutcome:
    if not surfaced_topics:
        return SemanticQualityOutcome(
            metrics={
                "non_redundancy": 0.0,
                "supported_theme_precision": 0.0,
                "distinct_valid_discovery_precision": 0.0,
                "valid_theme_coverage": 0.0,
                "num_supported_topics": 0.0,
                "num_distinct_valid_topics": 0.0,
            },
            details={
                "num_surfaced_topics": 0,
                "num_supported_topics": 0,
                "num_distinct_valid_topics": 0,
                "judged_topics": [],
                "duplicate_pairs": [],
            },
        )

    benchmark_themes_text = format_benchmark_themes(benchmark_spec)
    judged_topics: list[dict[str, Any]] = []
    supported_topic_indices: list[int] = []
    supported_theme_ids: set[str] = set()
    for index, surfaced_topic in enumerate(surfaced_topics):
        judgment = theme_match_judge(
            surfaced_topic=surfaced_topic,
            benchmark_themes=benchmark_themes_text,
            model_config=config.model_config,
            prompt_id=config.theme_match_prompt_id,
        )
        normalized_theme_id = (
            judgment.benchmark_theme_id.strip()
            if isinstance(judgment.benchmark_theme_id, str) and judgment.benchmark_theme_id.strip()
            else None
        )
        matched_theme = (
            benchmark_spec.theme_by_id.get(normalized_theme_id)
            if normalized_theme_id is not None
            else None
        )
        supported = matched_theme is not None and judgment.supported and not matched_theme.is_foil
        if matched_theme is not None and supported:
            supported_topic_indices.append(index)
            supported_theme_ids.add(matched_theme.theme_id)
        judged_topics.append(
            {
                "topic": surfaced_topic,
                "supported": supported,
                "benchmark_theme_id": matched_theme.theme_id if matched_theme is not None else None,
                "confidence": clamp_probability(judgment.confidence),
                "reasoning": judgment.reasoning,
            }
        )

    duplicate_pairs: list[dict[str, Any]] = []
    for left_index in range(len(surfaced_topics)):
        for right_index in range(left_index + 1, len(surfaced_topics)):
            judgment = duplicate_judge(
                left_topic=surfaced_topics[left_index],
                right_topic=surfaced_topics[right_index],
                model_config=config.model_config,
                prompt_id=config.duplicate_prompt_id,
            )
            duplicate_pairs.append(
                {
                    "left_index": left_index,
                    "right_index": right_index,
                    "duplicate": judgment.duplicate,
                    "confidence": clamp_probability(judgment.confidence),
                    "reasoning": judgment.reasoning,
                }
            )

    duplicate_components = group_indices_by_duplicate_component(
        num_topics=len(surfaced_topics),
        duplicate_pairs=duplicate_pairs,
    )
    supported_topic_index_set = set(supported_topic_indices)
    supported_count = len(supported_topic_indices)
    distinct_valid_count = sum(
        1 for component in duplicate_components if component <= supported_topic_index_set
    )
    distinct_topic_group_count = len(duplicate_components)

    return SemanticQualityOutcome(
        metrics={
            "non_redundancy": safe_ratio(
                distinct_topic_group_count,
                len(surfaced_topics),
                default=0.0,
            ),
            "supported_theme_precision": safe_ratio(
                supported_count,
                len(surfaced_topics),
                default=0.0,
            ),
            "distinct_valid_discovery_precision": safe_ratio(
                distinct_valid_count,
                len(surfaced_topics),
                default=0.0,
            ),
            "valid_theme_coverage": safe_ratio(
                len(supported_theme_ids),
                len(benchmark_spec.valid_themes),
                default=0.0,
            ),
            "num_supported_topics": float(supported_count),
            "num_distinct_valid_topics": float(distinct_valid_count),
        },
        details={
            "num_surfaced_topics": len(surfaced_topics),
            "num_supported_topics": supported_count,
            "num_distinct_valid_topics": distinct_valid_count,
            "judged_topics": judged_topics,
            "duplicate_pairs": duplicate_pairs,
        },
    )


def format_benchmark_themes(benchmark_spec: SyntheticBenchmarkSpec) -> str:
    lines: list[str] = []
    for theme in benchmark_spec.themes:
        lines.append(
            f"- theme_id={theme.theme_id}; label={theme.label}; guidance={theme.guidance}; "
            f"is_foil={theme.is_foil}"
        )
    return "\n".join(lines)


def clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def group_indices_by_duplicate_component(
    *, num_topics: int, duplicate_pairs: Sequence[Mapping[str, Any]]
) -> list[set[int]]:
    if num_topics <= 0:
        return []

    parents = list(range(num_topics))

    def find(index: int) -> int:
        root = index
        while parents[root] != root:
            root = parents[root]
        while parents[index] != index:
            next_index = parents[index]
            parents[index] = root
            index = next_index
        return root

    def union(left_index: int, right_index: int) -> None:
        left_root = find(left_index)
        right_root = find(right_index)
        if left_root != right_root:
            parents[right_root] = left_root

    for pair in duplicate_pairs:
        if not pair.get("duplicate"):
            continue
        left_index = pair.get("left_index")
        right_index = pair.get("right_index")
        if isinstance(left_index, int) and isinstance(right_index, int):
            union(left_index, right_index)

    groups_by_root: dict[int, set[int]] = {}
    for index in range(num_topics):
        groups_by_root.setdefault(find(index), set()).add(index)
    return list(groups_by_root.values())


def safe_ratio(numerator: float, denominator: float, *, default: float) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def _parse_optional_model_config(raw_model_config: object) -> ModelConfig | None:
    if raw_model_config is None:
        return None
    if not isinstance(raw_model_config, Mapping):
        raise TypeError("model_config must be a dict")
    try:
        return parse_model_config(cast(Mapping[str, object], raw_model_config))
    except ValueError as exc:
        raise ValueError(f"'model_config' is invalid: {exc}") from exc
