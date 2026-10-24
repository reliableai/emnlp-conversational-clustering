"""Shared text and retry helpers for structured dataset presets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


def humanize_slug(value: str, replacements: Mapping[str, str] | None = None) -> str:
    replacement_map = replacements or {}
    parts = [replacement_map.get(part, part) for part in value.split("_")]
    return " ".join(parts)


def build_surface_metadata(record: Mapping[str, str], fields: Sequence[str]) -> dict[str, str]:
    return {field: record[field] for field in fields}


def build_alignment_retry_prefix(
    failures: Sequence[str],
    *,
    intro_lines: Sequence[str],
    max_failures: int = 4,
) -> str | None:
    if not failures:
        return None
    return "\n".join([*intro_lines, *failures[:max_failures], ""])
