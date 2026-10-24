"""Shared clustering POV-bank helpers for conversational clustering backends."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field


class ClusteringPovBankState(Protocol):
    clustering_pov_bank: list[str] | None


class AppendClusteringPovsInput(BaseModel):
    povs: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Dataset-grounded clustering POV labels to append to the internal POV bank. "
            "Only include the POV itself, not parenthetical example datapoints."
        ),
    )


@dataclass(frozen=True)
class AppendClusteringPovsResult:
    added_povs: list[str]
    merged_pov_bank: list[str]

    @property
    def num_added(self) -> int:
        return len(self.added_povs)

    @property
    def total_povs(self) -> int:
        return len(self.merged_pov_bank)


def normalize_clustering_pov(pov: str) -> str:
    return " ".join(str(pov).split())


def coerce_clustering_pov_bank(povs: list[str] | None) -> list[str]:
    if povs is None:
        return []
    normalized_povs: list[str] = []
    seen_keys: set[str] = set()
    for pov in povs:
        normalized_pov = normalize_clustering_pov(pov)
        if not normalized_pov:
            continue
        dedupe_key = _clustering_pov_dedupe_key(normalized_pov)
        if dedupe_key in seen_keys:
            continue
        normalized_povs.append(normalized_pov)
        seen_keys.add(dedupe_key)
    return normalized_povs


def coerce_clustering_pov_history(
    history: list[dict[str, object]] | None,
) -> list[dict[str, object]]:
    if history is None:
        return []

    normalized_history: list[dict[str, object]] = []
    history_index_by_turn: dict[int, int] = {}
    for raw_entry in history:
        if not isinstance(raw_entry, dict):
            continue
        raw_turn_index = raw_entry.get("turn_index")
        raw_povs = raw_entry.get("povs")
        if isinstance(raw_turn_index, bool) or not isinstance(raw_turn_index, int):
            continue
        if raw_turn_index < 1:
            continue
        if not isinstance(raw_povs, list):
            continue

        normalized_povs = coerce_clustering_pov_bank(
            [pov for pov in raw_povs if isinstance(pov, str)]
        )
        if not normalized_povs:
            continue

        normalized_entry: dict[str, object] = {
            "turn_index": raw_turn_index,
            "povs": normalized_povs,
        }
        existing_index = history_index_by_turn.get(raw_turn_index)
        if existing_index is None:
            history_index_by_turn[raw_turn_index] = len(normalized_history)
            normalized_history.append(normalized_entry)
            continue
        normalized_history[existing_index] = normalized_entry

    return normalized_history


def upsert_clustering_pov_history(
    *,
    history: list[dict[str, object]] | None,
    turn_index: int,
    povs: list[str] | None,
) -> list[dict[str, object]]:
    normalized_history = coerce_clustering_pov_history(history)
    if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 1:
        return normalized_history

    normalized_povs = coerce_clustering_pov_bank(povs)
    if not normalized_povs:
        return normalized_history

    normalized_entry: dict[str, object] = {
        "turn_index": turn_index,
        "povs": normalized_povs,
    }
    for index, existing_entry in enumerate(normalized_history):
        if existing_entry.get("turn_index") == turn_index:
            normalized_history[index] = normalized_entry
            return normalized_history

    normalized_history.append(normalized_entry)
    return normalized_history


def append_clustering_povs_to_bank(
    *,
    existing_pov_bank: list[str] | None,
    povs: list[str],
) -> AppendClusteringPovsResult:
    normalized_inputs = [normalize_clustering_pov(pov) for pov in povs]
    normalized_inputs = [pov for pov in normalized_inputs if pov]
    if not normalized_inputs:
        raise ValueError("At least one non-empty clustering POV is required.")

    resolved_existing_bank = coerce_clustering_pov_bank(existing_pov_bank)
    seen_keys = {_clustering_pov_dedupe_key(pov) for pov in resolved_existing_bank}
    added_povs: list[str] = []
    merged_bank = list(resolved_existing_bank)
    for pov in normalized_inputs:
        dedupe_key = _clustering_pov_dedupe_key(pov)
        if dedupe_key in seen_keys:
            continue
        merged_bank.append(pov)
        added_povs.append(pov)
        seen_keys.add(dedupe_key)

    return AppendClusteringPovsResult(
        added_povs=added_povs,
        merged_pov_bank=merged_bank,
    )


def make_append_clustering_povs_tool(
    *,
    runtime_state: ClusteringPovBankState,
) -> BaseTool:
    @tool("append_clustering_povs", args_schema=AppendClusteringPovsInput)
    def append_clustering_povs(povs: list[str]) -> str:
        """Appends dataset-grounded clustering POV labels to a persistent internal bank."""
        try:
            result = append_clustering_povs_to_bank(
                existing_pov_bank=runtime_state.clustering_pov_bank,
                povs=povs,
            )
            runtime_state.clustering_pov_bank = result.merged_pov_bank
            return json.dumps(
                {
                    "added_povs": result.added_povs,
                    "num_added": result.num_added,
                    "total_povs": result.total_povs,
                },
                ensure_ascii=True,
            )
        except ValueError as exc:
            return f"Could not append clustering POVs. Reason: {exc}"
        except Exception:
            return "Could not append clustering POVs due to an unexpected error."

    return append_clustering_povs


def build_clustering_pov_bank_message(clustering_pov_bank: list[str]) -> SystemMessage:
    numbered_povs = "\n".join(
        f"{index}. {pov}" for index, pov in enumerate(clustering_pov_bank, start=1)
    )
    return SystemMessage(
        content=(
            "Internal clustering POV bank so far "
            "(dataset-grounded possibilities, not necessarily user-desired):\n"
            f"{numbered_povs}"
        )
    )


def _clustering_pov_dedupe_key(pov: str) -> str:
    return normalize_clustering_pov(pov).lower()
