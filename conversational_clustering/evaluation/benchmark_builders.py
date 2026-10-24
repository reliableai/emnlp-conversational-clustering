"""Utilities for building evaluation benchmark specs from labeled dataset CSVs."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TypedDict

import pandas as pd


class BenchmarkThemePayload(TypedDict):
    theme_id: str
    label: str
    guidance: str
    weight: float
    pov_group_id: str | None
    granularity: str


class BenchmarkReferencePayload(TypedDict):
    reference_id: str
    topic_id: str
    granularity: str
    assignment: dict[str, int]
    cluster_names: list[str]
    pov_group_id: str | None
    granularity_level: str | None


class BenchmarkSpecPayload(TypedDict):
    dataset_name: str
    items: list[str]
    themes: list[BenchmarkThemePayload]
    references: list[BenchmarkReferencePayload]


@dataclass(frozen=True)
class BenchmarkThemeDefinition:
    theme_id: str
    label: str
    granularity: str = "normal"
    weight: float = 1.0
    pov_group_id: str | None = None


DEFAULT_ENTERPRISE_TICKET_THEME_DEFINITIONS: tuple[BenchmarkThemeDefinition, ...] = (
    BenchmarkThemeDefinition(
        theme_id="department",
        label="Department",
    ),
    BenchmarkThemeDefinition(
        theme_id="ticket_category",
        label="Ticket Category",
    ),
    BenchmarkThemeDefinition(
        theme_id="intent_group",
        label="Intent Group",
    ),
    BenchmarkThemeDefinition(
        theme_id="issue_type",
        label="Issue Type",
    ),
    BenchmarkThemeDefinition(
        theme_id="sub_intent",
        label="Sub-Intent",
    ),
    BenchmarkThemeDefinition(
        theme_id="system_name",
        label="System Name",
    ),
    BenchmarkThemeDefinition(
        theme_id="organizational_label_coarse",
        label="Organizational (Coarse)",
        pov_group_id="organizational",
        granularity="coarse",
    ),
    BenchmarkThemeDefinition(
        theme_id="organizational_label_normal",
        label="Organizational (Normal)",
        pov_group_id="organizational",
    ),
    BenchmarkThemeDefinition(
        theme_id="organizational_label_fine",
        label="Organizational (Fine)",
        pov_group_id="organizational",
        granularity="fine",
    ),
    BenchmarkThemeDefinition(
        theme_id="intent_label_coarse",
        label="Intent (Coarse)",
        pov_group_id="intent",
        granularity="coarse",
    ),
    BenchmarkThemeDefinition(
        theme_id="intent_label_normal",
        label="Intent (Normal)",
        pov_group_id="intent",
    ),
    BenchmarkThemeDefinition(
        theme_id="intent_label_fine",
        label="Intent (Fine)",
        pov_group_id="intent",
        granularity="fine",
    ),
)

ENTERPRISE_TICKET_THEME_GUIDANCE_BY_ID: dict[str, str] = {
    "department": (
        "Group tickets by the primary department involved in the request, "
        "such as Finance, IT, or Facilities."
    ),
    "ticket_category": (
        "Group tickets by the broad operational category of work, such as "
        "account issues, badge issues, or customer communications."
    ),
    "intent_group": (
        "Group tickets by the broad workflow or intent family, such as "
        "access identity, finance procurement, customer operations, or "
        "engineering operations."
    ),
    "issue_type": (
        "Group tickets by the specific problem category being reported, "
        "such as MFA problems, badge failures, payment delays, or "
        "release-operation issues."
    ),
    "sub_intent": (
        "Group tickets by the most specific task or issue subtype described in the ticket."
    ),
    "system_name": (
        "Group tickets by the primary software system, platform, or tool mentioned in the request."
    ),
    "organizational_label_coarse": (
        "Group tickets by the broad organizational area responsible for handling the request."
    ),
    "organizational_label_normal": (
        "Group tickets by the primary team area responsible for handling "
        "the request, such as identity access, badge operations, "
        "accounts payable, or network operations."
    ),
    "organizational_label_fine": (
        "Group tickets by the most specific responsible organizational "
        "unit or handling team described by the dataset labels."
    ),
    "intent_label_coarse": (
        "Group tickets by the broad user intent or workflow family behind the request."
    ),
    "intent_label_normal": (
        "Group tickets by the primary request type or problem category expressed by the ticket."
    ),
    "intent_label_fine": (
        "Group tickets by the most specific request or issue subtype, "
        "separating distinct problems such as MFA delivery failure, "
        "badge malfunction, invoice mismatch, or VPN outage."
    ),
}


def load_labeled_csv_rows(
    path: str | PathLike[str],
    *,
    limit: int | None = None,
    random_seed: int = 42,
) -> list[dict[str, str | None]]:
    if limit is None:
        with open(path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV '{path}' is missing a header row")
            rows = [{str(key): value for key, value in row.items()} for row in reader]
        if not rows:
            raise ValueError(f"CSV '{path}' did not contain any data rows")
        return rows

    dataframe = pd.read_csv(path, low_memory=False).drop_duplicates().reset_index(drop=True)
    if dataframe.empty:
        raise ValueError(f"CSV '{path}' did not contain any data rows")
    sampled = dataframe.sample(limit, random_state=random_seed).reset_index(drop=True)
    return [
        {str(key): None if pd.isna(value) else str(value) for key, value in row.items()}
        for row in sampled.to_dict(orient="records")
    ]


def build_enterprise_ticket_benchmark_spec(
    rows: Sequence[Mapping[str, str | None]],
    *,
    dataset_name: str = "enterprise_ticket",
    item_column: str = "ticket_text",
    theme_columns: Sequence[str] | None = None,
) -> BenchmarkSpecPayload:
    if not rows:
        raise ValueError("rows must be non-empty")
    normalized_dataset_name = dataset_name.strip()
    if not normalized_dataset_name:
        raise ValueError("dataset_name must be a non-empty string")
    normalized_item_column = item_column.strip()
    if not normalized_item_column:
        raise ValueError("item_column must be a non-empty string")

    theme_definitions = _resolve_theme_definitions(theme_columns)
    _validate_required_columns(
        rows,
        item_column=normalized_item_column,
        theme_definitions=theme_definitions,
    )

    item_texts: list[str] = []
    seen_item_texts: set[str] = set()
    for row_index, row in enumerate(rows, start=1):
        item_text = _require_cell(
            row,
            column=normalized_item_column,
            row_index=row_index,
        )
        if item_text in seen_item_texts:
            raise ValueError(
                "Duplicate item text found in column "
                f"'{normalized_item_column}' at row {row_index}: "
                f"{item_text!r}. Benchmark evaluation requires unique item texts."
            )
        seen_item_texts.add(item_text)
        item_texts.append(item_text)

    references = [
        _build_reference_spec(
            rows,
            item_column=normalized_item_column,
            theme_definition=theme_definition,
        )
        for theme_definition in theme_definitions
    ]
    return {
        "dataset_name": normalized_dataset_name,
        "items": item_texts,
        "themes": [
            {
                "theme_id": theme_definition.theme_id,
                "label": theme_definition.label,
                "guidance": _theme_guidance(theme_definition),
                "weight": theme_definition.weight,
                "pov_group_id": theme_definition.pov_group_id,
                "granularity": theme_definition.granularity,
            }
            for theme_definition in theme_definitions
        ],
        "references": references,
    }


def write_benchmark_spec(path: str | PathLike[str], payload: BenchmarkSpecPayload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def _resolve_theme_definitions(
    theme_columns: Sequence[str] | None,
) -> tuple[BenchmarkThemeDefinition, ...]:
    if theme_columns is None:
        return DEFAULT_ENTERPRISE_TICKET_THEME_DEFINITIONS
    normalized_columns: list[str] = []
    for raw_column in theme_columns:
        if not isinstance(raw_column, str) or not raw_column.strip():
            raise ValueError("theme_columns entries must be non-empty strings")
        normalized_column = raw_column.strip()
        if normalized_column in normalized_columns:
            raise ValueError(f"Duplicate theme column requested: {normalized_column!r}")
        normalized_columns.append(normalized_column)
    if not normalized_columns:
        raise ValueError("theme_columns must contain at least one column name")
    defaults_by_id = {
        definition.theme_id: definition
        for definition in DEFAULT_ENTERPRISE_TICKET_THEME_DEFINITIONS
    }
    return tuple(
        defaults_by_id.get(column, _build_generic_theme_definition(column))
        for column in normalized_columns
    )


def _build_generic_theme_definition(column_name: str) -> BenchmarkThemeDefinition:
    label = column_name.replace("_", " ").title()
    pov_group_id: str | None = None
    granularity = "normal"
    grouped_theme_prefixes = {
        "intent_label_": "intent",
        "organizational_label_": "organizational",
    }
    for prefix, resolved_group_id in grouped_theme_prefixes.items():
        if column_name.startswith(prefix):
            raw_level = column_name.removeprefix(prefix)
            if raw_level in {"coarse", "normal", "fine"}:
                granularity = raw_level
                pov_group_id = resolved_group_id
                group_label = resolved_group_id.replace("_", " ").title()
                label = f"{group_label} ({raw_level.title()})"
                break
    return BenchmarkThemeDefinition(
        theme_id=column_name,
        label=label,
        granularity=granularity,
        pov_group_id=pov_group_id,
    )


def _theme_guidance(theme_definition: BenchmarkThemeDefinition) -> str:
    return ENTERPRISE_TICKET_THEME_GUIDANCE_BY_ID.get(
        theme_definition.theme_id,
        f"Group tickets by {theme_definition.label.lower()}.",
    )


def _validate_required_columns(
    rows: Sequence[Mapping[str, str | None]],
    *,
    item_column: str,
    theme_definitions: Iterable[BenchmarkThemeDefinition],
) -> None:
    required_columns = {
        item_column,
        *(theme_definition.theme_id for theme_definition in theme_definitions),
    }
    header = set(rows[0])
    missing_columns = sorted(required_columns - header)
    if missing_columns:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing_columns)}")


def _require_cell(
    row: Mapping[str, str | None],
    *,
    column: str,
    row_index: int,
) -> str:
    raw_value = row.get(column)
    if raw_value is None:
        raise ValueError(f"Row {row_index} is missing required column '{column}'")
    value = raw_value.strip()
    if not value:
        raise ValueError(f"Row {row_index} has blank value for required column '{column}'")
    return value


def _build_reference_spec(
    rows: Sequence[Mapping[str, str | None]],
    *,
    item_column: str,
    theme_definition: BenchmarkThemeDefinition,
) -> BenchmarkReferencePayload:
    theme_value_to_items: dict[str, list[str]] = {}
    for row_index, row in enumerate(rows, start=1):
        item_text = _require_cell(row, column=item_column, row_index=row_index)
        theme_value = _require_cell(row, column=theme_definition.theme_id, row_index=row_index)
        theme_value_to_items.setdefault(theme_value, []).append(item_text)

    cluster_names = sorted(theme_value_to_items)
    cluster_ids = {cluster_name: index for index, cluster_name in enumerate(cluster_names)}
    assignment = {
        item_text: cluster_ids[theme_value]
        for theme_value, grouped_items in theme_value_to_items.items()
        for item_text in grouped_items
    }
    return {
        "reference_id": (f"{theme_definition.theme_id}__{theme_definition.granularity}"),
        "topic_id": theme_definition.theme_id,
        "granularity": theme_definition.granularity,
        "assignment": assignment,
        "cluster_names": cluster_names,
        "pov_group_id": theme_definition.pov_group_id,
        "granularity_level": theme_definition.granularity,
    }
