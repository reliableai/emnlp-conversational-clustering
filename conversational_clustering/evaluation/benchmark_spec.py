"""Typed benchmark specifications and lookup helpers for evaluation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, cast

from conversational_clustering.data_processing.datasets import DATASETS
from conversational_clustering.evaluation.artifacts import LoadedClustering

_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")


def normalize_benchmark_text(value: str) -> str:
    normalized = _NORMALIZE_PATTERN.sub(" ", value.strip().lower()).strip()
    return normalized


@dataclass(frozen=True)
class ThemeSpec:
    theme_id: str
    label: str
    guidance: str
    weight: float = 1.0
    is_foil: bool = False
    pov_group_id: str | None = None
    granularity: str = "normal"

    def __post_init__(self) -> None:
        if not isinstance(self.theme_id, str) or not self.theme_id.strip():
            raise ValueError("ThemeSpec.theme_id must be a non-empty string")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("ThemeSpec.label must be a non-empty string")
        if not isinstance(self.guidance, str) or not self.guidance.strip():
            raise ValueError("ThemeSpec.guidance must be a non-empty string")
        if isinstance(self.weight, bool) or not isinstance(self.weight, int | float):
            raise ValueError("ThemeSpec.weight must be a positive number")
        if float(self.weight) <= 0:
            raise ValueError("ThemeSpec.weight must be a positive number")
        object.__setattr__(self, "theme_id", self.theme_id.strip())
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "guidance", self.guidance.strip())
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(
            self,
            "pov_group_id",
            self.pov_group_id.strip()
            if isinstance(self.pov_group_id, str) and self.pov_group_id.strip()
            else None,
        )
        object.__setattr__(
            self,
            "granularity",
            self.granularity.strip()
            if isinstance(self.granularity, str) and self.granularity.strip()
            else "normal",
        )

    @property
    def normalized_match_texts(self) -> set[str]:
        return {
            normalize_benchmark_text(value)
            for value in (self.theme_id, self.label)
            if normalize_benchmark_text(value)
        }


@dataclass(frozen=True)
class ReferenceClusteringSpec:
    reference_id: str
    topic_id: str
    granularity: str
    assignment: dict[str, int]
    cluster_names: list[str] = field(default_factory=list)
    pov_group_id: str | None = None
    granularity_level: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reference_id, str) or not self.reference_id.strip():
            raise ValueError("ReferenceClusteringSpec.reference_id must be a non-empty string")
        if not isinstance(self.topic_id, str) or not self.topic_id.strip():
            raise ValueError("ReferenceClusteringSpec.topic_id must be a non-empty string")
        if not isinstance(self.granularity, str) or not self.granularity.strip():
            raise ValueError("ReferenceClusteringSpec.granularity must be a non-empty string")
        if not isinstance(self.assignment, dict) or not self.assignment:
            raise ValueError("ReferenceClusteringSpec.assignment must be a non-empty mapping")
        normalized_assignment: dict[str, int] = {}
        max_cluster_id = -1
        for raw_item_text, raw_cluster_id in self.assignment.items():
            if not isinstance(raw_item_text, str) or not raw_item_text.strip():
                raise ValueError(
                    "ReferenceClusteringSpec.assignment keys must be item-text strings"
                )
            if isinstance(raw_cluster_id, bool) or not isinstance(raw_cluster_id, int):
                raise ValueError(
                    "ReferenceClusteringSpec.assignment values must be integer cluster IDs"
                )
            if raw_cluster_id < 0:
                raise ValueError("ReferenceClusteringSpec.assignment cluster IDs must be >= 0")
            item_text = raw_item_text.strip()
            normalized_assignment[item_text] = raw_cluster_id
            max_cluster_id = max(max_cluster_id, raw_cluster_id)
        normalized_cluster_names = [
            str(cluster_name).strip()
            for cluster_name in self.cluster_names
            if isinstance(cluster_name, str) and str(cluster_name).strip()
        ]
        if normalized_cluster_names and len(normalized_cluster_names) <= max_cluster_id:
            raise ValueError(
                "ReferenceClusteringSpec.cluster_names must cover every referenced cluster ID"
            )
        if not normalized_cluster_names:
            normalized_cluster_names = [f"cluster_{index}" for index in range(max_cluster_id + 1)]
        object.__setattr__(self, "reference_id", self.reference_id.strip())
        object.__setattr__(self, "topic_id", self.topic_id.strip())
        object.__setattr__(self, "granularity", self.granularity.strip())
        object.__setattr__(self, "assignment", normalized_assignment)
        object.__setattr__(self, "cluster_names", normalized_cluster_names)
        object.__setattr__(
            self,
            "pov_group_id",
            self.pov_group_id.strip()
            if isinstance(self.pov_group_id, str) and self.pov_group_id.strip()
            else None,
        )
        object.__setattr__(
            self,
            "granularity_level",
            self.granularity_level.strip()
            if isinstance(self.granularity_level, str) and self.granularity_level.strip()
            else None,
        )

    def to_loaded_clustering(self) -> LoadedClustering:
        return LoadedClustering(
            assignment=dict(self.assignment),
            cluster_names=list(self.cluster_names),
        )


@dataclass(frozen=True)
class SyntheticBenchmarkSpec:
    dataset_name: str
    item_texts: tuple[str, ...]
    themes: tuple[ThemeSpec, ...]
    references: tuple[ReferenceClusteringSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_name, str) or not self.dataset_name.strip():
            raise ValueError("SyntheticBenchmarkSpec.dataset_name must be a non-empty string")
        if not self.item_texts:
            raise ValueError("SyntheticBenchmarkSpec.item_texts must be non-empty")
        normalized_item_texts = tuple(item_text.strip() for item_text in self.item_texts)
        if any(not item_text for item_text in normalized_item_texts):
            raise ValueError("SyntheticBenchmarkSpec.item_texts must not contain blanks")
        if len(set(normalized_item_texts)) != len(normalized_item_texts):
            raise ValueError(
                "SyntheticBenchmarkSpec.item_texts must contain unique item texts for evaluation"
            )
        object.__setattr__(self, "dataset_name", self.dataset_name.strip())
        object.__setattr__(self, "item_texts", normalized_item_texts)

        theme_ids = [theme.theme_id for theme in self.themes]
        if len(set(theme_ids)) != len(theme_ids):
            raise ValueError("SyntheticBenchmarkSpec.theme_id values must be unique")

        reference_ids = [reference.reference_id for reference in self.references]
        if len(set(reference_ids)) != len(reference_ids):
            raise ValueError("SyntheticBenchmarkSpec.reference_id values must be unique")

        item_text_set = set(normalized_item_texts)
        valid_theme_ids = set(theme_ids)
        for reference in self.references:
            if reference.topic_id not in valid_theme_ids:
                raise ValueError(
                    f"ReferenceClusteringSpec '{reference.reference_id}' refers to unknown "
                    f"topic_id '{reference.topic_id}'"
                )
            if set(reference.assignment) != item_text_set:
                raise ValueError(
                    f"ReferenceClusteringSpec '{reference.reference_id}' must cover the benchmark "
                    "item set exactly."
                )

    @property
    def theme_by_id(self) -> dict[str, ThemeSpec]:
        return {theme.theme_id: theme for theme in self.themes}

    @property
    def reference_by_id(self) -> dict[str, ReferenceClusteringSpec]:
        return {reference.reference_id: reference for reference in self.references}

    @property
    def valid_themes(self) -> tuple[ThemeSpec, ...]:
        return tuple(theme for theme in self.themes if not theme.is_foil)

    @property
    def foil_themes(self) -> tuple[ThemeSpec, ...]:
        return tuple(theme for theme in self.themes if theme.is_foil)

    def match_theme(self, value: str | None) -> ThemeSpec | None:
        if value is None:
            return None
        normalized_value = normalize_benchmark_text(value)
        if not normalized_value:
            return None
        for theme in self.themes:
            if normalized_value in theme.normalized_match_texts:
                return theme
        return None

    def grouped_themes(self, pov_group_id: str) -> tuple[ThemeSpec, ...]:
        normalized_group_id = pov_group_id.strip()
        return tuple(theme for theme in self.themes if theme.pov_group_id == normalized_group_id)

    def grouped_theme_siblings(self, theme_id: str) -> tuple[ThemeSpec, ...]:
        theme = self.theme_by_id.get(theme_id.strip())
        if theme is None or theme.pov_group_id is None:
            return ()
        return self.grouped_themes(theme.pov_group_id)

    def references_for_pov_group(self, pov_group_id: str) -> tuple[ReferenceClusteringSpec, ...]:
        normalized_group_id = pov_group_id.strip()
        return tuple(
            reference
            for reference in self.references
            if reference.pov_group_id == normalized_group_id
        )

    def references_for_theme(self, theme_id: str) -> tuple[ReferenceClusteringSpec, ...]:
        normalized_theme_id = theme_id.strip()
        return tuple(
            reference for reference in self.references if reference.topic_id == normalized_theme_id
        )

    def references_for_theme_or_group(self, theme_id: str) -> tuple[ReferenceClusteringSpec, ...]:
        theme = self.theme_by_id.get(theme_id.strip())
        if theme is None:
            return ()
        if theme.pov_group_id is not None:
            grouped_references = self.references_for_pov_group(theme.pov_group_id)
            if grouped_references:
                return grouped_references
        return self.references_for_theme(theme.theme_id)

    def resolve_reference(
        self,
        *,
        topic_id: str,
        granularity: str = "normal",
    ) -> ReferenceClusteringSpec:
        normalized_topic_id = topic_id.strip()
        normalized_granularity = granularity.strip()
        theme = self.theme_by_id.get(normalized_topic_id)
        for reference in self.references:
            if (
                reference.topic_id == normalized_topic_id
                and reference.granularity == normalized_granularity
            ):
                return reference
        if theme is not None and theme.pov_group_id is not None:
            return self.resolve_group_reference(
                pov_group_id=theme.pov_group_id,
                granularity=normalized_granularity,
            )
        available = [
            f"{reference.topic_id}:{reference.granularity}" for reference in self.references
        ]
        raise ValueError(
            "No benchmark reference clustering found for "
            f"topic_id='{normalized_topic_id}', granularity='{normalized_granularity}'. "
            f"Available: {sorted(available)}"
        )

    def resolve_group_reference(
        self,
        *,
        pov_group_id: str,
        granularity: str,
    ) -> ReferenceClusteringSpec:
        normalized_group_id = pov_group_id.strip()
        normalized_granularity = granularity.strip()
        for reference in self.references:
            if (
                reference.pov_group_id == normalized_group_id
                and reference.granularity == normalized_granularity
            ):
                return reference
        available = [
            f"{reference.pov_group_id}:{reference.granularity}"
            for reference in self.references
            if reference.pov_group_id is not None
        ]
        raise ValueError(
            "No benchmark reference clustering found for "
            f"pov_group_id='{normalized_group_id}', granularity='{normalized_granularity}'. "
            f"Available: {sorted(available)}"
        )


def _require_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, Any], value)


def _require_string_field(
    value: object,
    *,
    context: str,
    allow_blank: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    normalized_value = value.strip()
    if not allow_blank and not normalized_value:
        raise ValueError(f"{context} must be a non-empty string")
    return normalized_value


def _require_optional_string_field(
    value: object,
    *,
    context: str,
    allow_blank: bool = False,
) -> str | None:
    if value is None:
        return None
    return _require_string_field(value, context=context, allow_blank=allow_blank)


def _require_string_list(value: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of strings")
    normalized_values: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{context}[{index}] must be a string")
        stripped_item = item.strip()
        if stripped_item:
            normalized_values.append(stripped_item)
    return tuple(normalized_values)


def _require_bool_field(value: object, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean")
    return value


def _parse_theme_specs(payload: object) -> tuple[ThemeSpec, ...]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("Synthetic benchmark spec must define a non-empty 'themes' list")
    themes: list[ThemeSpec] = []
    for index, raw_theme in enumerate(payload, start=1):
        theme = _require_mapping(raw_theme, context=f"themes[{index}]")
        themes.append(
            ThemeSpec(
                theme_id=_require_string_field(
                    theme.get("theme_id"), context=f"themes[{index}].theme_id"
                ),
                label=_require_string_field(theme.get("label"), context=f"themes[{index}].label"),
                guidance=_require_string_field(
                    theme.get("guidance"), context=f"themes[{index}].guidance"
                ),
                weight=theme.get("weight", 1.0),
                is_foil=_require_bool_field(
                    theme.get("is_foil", False), context=f"themes[{index}].is_foil"
                ),
                pov_group_id=_require_optional_string_field(
                    theme.get("pov_group_id", ""),
                    context=f"themes[{index}].pov_group_id",
                    allow_blank=True,
                )
                or None,
                granularity=_require_optional_string_field(
                    theme.get("granularity", "normal"),
                    context=f"themes[{index}].granularity",
                    allow_blank=True,
                )
                or "normal",
            )
        )
    return tuple(themes)


def _parse_reference_specs(payload: object) -> tuple[ReferenceClusteringSpec, ...]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("Synthetic benchmark spec must define a non-empty 'references' list")
    references: list[ReferenceClusteringSpec] = []
    for index, raw_reference in enumerate(payload, start=1):
        reference = _require_mapping(raw_reference, context=f"references[{index}]")
        granularity = str(reference.get("granularity", "normal")).strip() or "normal"
        raw_assignment = _require_mapping(
            reference.get("assignment"),
            context=f"references[{index}].assignment",
        )
        assignment = {
            str(item_text): cluster_id for item_text, cluster_id in raw_assignment.items()
        }
        raw_cluster_names = reference.get("cluster_names", [])
        if raw_cluster_names is None:
            cluster_names: list[str] = []
        elif isinstance(raw_cluster_names, list):
            cluster_names = [str(cluster_name) for cluster_name in raw_cluster_names]
        else:
            raise ValueError(f"references[{index}].cluster_names must be a list of strings")
        references.append(
            ReferenceClusteringSpec(
                reference_id=str(reference.get("reference_id", "")).strip(),
                topic_id=str(reference.get("topic_id", "")).strip(),
                granularity=granularity,
                assignment=assignment,
                cluster_names=cluster_names,
                pov_group_id=_require_optional_string_field(
                    reference.get("pov_group_id", ""),
                    context=f"references[{index}].pov_group_id",
                    allow_blank=True,
                )
                or None,
                granularity_level=_require_optional_string_field(
                    reference.get("granularity_level", granularity),
                    context=f"references[{index}].granularity_level",
                    allow_blank=True,
                )
                or granularity,
            )
        )
    return tuple(references)


def parse_synthetic_benchmark_spec(payload: Mapping[str, Any]) -> SyntheticBenchmarkSpec:
    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("Synthetic benchmark spec must define a non-empty 'items' list")
    item_texts = tuple(str(item).strip() for item in raw_items)
    if any(not item_text for item_text in item_texts):
        raise ValueError("Synthetic benchmark spec 'items' must not contain blank values")

    return SyntheticBenchmarkSpec(
        dataset_name=str(payload.get("dataset_name", "")).strip(),
        item_texts=item_texts,
        themes=_parse_theme_specs(payload.get("themes")),
        references=_parse_reference_specs(payload.get("references")),
    )


def load_synthetic_benchmark_spec(path: str | PathLike[str]) -> SyntheticBenchmarkSpec:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Synthetic benchmark spec must be a top-level object")
    return parse_synthetic_benchmark_spec(payload)


def load_benchmark_spec_for_dataset(dataset_name: str) -> SyntheticBenchmarkSpec | None:
    dataset_config = DATASETS.get(dataset_name)
    if dataset_config is None:
        return None
    benchmark_spec_path = dataset_config.get("benchmark_spec_path")
    if benchmark_spec_path is None:
        return None
    if not isinstance(benchmark_spec_path, str) or not benchmark_spec_path.strip():
        raise ValueError(
            f"Dataset '{dataset_name}' has invalid benchmark_spec_path={benchmark_spec_path!r}"
        )
    benchmark_path = Path(benchmark_spec_path)
    if not benchmark_path.is_file():
        raise ValueError(
            f"Dataset '{dataset_name}' benchmark spec was configured at '{benchmark_path}' "
            "but the file does not exist"
        )
    benchmark_spec = load_synthetic_benchmark_spec(benchmark_path)
    if benchmark_spec.dataset_name != dataset_name:
        raise ValueError(
            f"Dataset '{dataset_name}' benchmark spec at '{benchmark_path}' declares "
            f"dataset_name='{benchmark_spec.dataset_name}'"
        )
    return benchmark_spec
