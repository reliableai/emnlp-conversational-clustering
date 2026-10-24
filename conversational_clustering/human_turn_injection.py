"""Typed configuration for injecting registered human edit primitives and metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .prompts import get_human_turn_injection_prompt


@dataclass(frozen=True)
class HumanTurnInjectionConfig:
    turn_index: int
    edit_primitive: str
    source_theme_id: str | None = None
    target_theme_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.turn_index, bool) or not isinstance(self.turn_index, int):
            raise ValueError("turn_index must be an integer >= 1")
        if self.turn_index < 1:
            raise ValueError("turn_index must be an integer >= 1")
        normalized_edit_primitive = _normalize_required_string(
            self.edit_primitive,
            field_name="edit_primitive",
        )
        get_human_turn_injection_prompt(normalized_edit_primitive)
        object.__setattr__(self, "edit_primitive", normalized_edit_primitive)
        normalized_source_theme_id = _normalize_optional_string(
            self.source_theme_id,
            field_name="source_theme_id",
        )
        normalized_target_theme_id = _normalize_optional_string(
            self.target_theme_id,
            field_name="target_theme_id",
        )
        if normalized_edit_primitive != "theme_switch":
            if normalized_source_theme_id is not None:
                raise ValueError(
                    "source_theme_id is only supported when edit_primitive='theme_switch'"
                )
            if normalized_target_theme_id is not None:
                raise ValueError(
                    "target_theme_id is only supported when edit_primitive='theme_switch'"
                )
        else:
            if normalized_source_theme_id is None:
                raise ValueError("source_theme_id is required when edit_primitive='theme_switch'")
            if normalized_target_theme_id is None:
                raise ValueError("target_theme_id is required when edit_primitive='theme_switch'")
            if normalized_source_theme_id == normalized_target_theme_id:
                raise ValueError(
                    "source_theme_id and target_theme_id must differ when "
                    "edit_primitive='theme_switch'"
                )
        object.__setattr__(self, "source_theme_id", normalized_source_theme_id)
        object.__setattr__(self, "target_theme_id", normalized_target_theme_id)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> HumanTurnInjectionConfig:
        unknown_fields = sorted(
            set(payload)
            - {
                "turn_index",
                "edit_primitive",
                "source_theme_id",
                "target_theme_id",
            }
        )
        if unknown_fields:
            raise ValueError(
                "turn injection payload has unsupported field(s): " + ", ".join(unknown_fields)
            )
        raw_turn_index = payload.get("turn_index")
        if raw_turn_index is None:
            raise ValueError("turn_index is required")
        if isinstance(raw_turn_index, bool) or not isinstance(raw_turn_index, int):
            raise ValueError("turn_index must be an integer >= 1")
        raw_edit_primitive = payload.get("edit_primitive")
        if raw_edit_primitive is None:
            raise ValueError("edit_primitive is required")
        if not isinstance(raw_edit_primitive, str):
            raise ValueError("edit_primitive must be a non-empty string")
        raw_source_theme_id = payload.get("source_theme_id")
        if raw_source_theme_id is not None and not isinstance(raw_source_theme_id, str):
            raise ValueError("source_theme_id must be a non-empty string when provided")
        raw_target_theme_id = payload.get("target_theme_id")
        if raw_target_theme_id is not None and not isinstance(raw_target_theme_id, str):
            raise ValueError("target_theme_id must be a non-empty string when provided")
        return cls(
            turn_index=raw_turn_index,
            edit_primitive=raw_edit_primitive,
            source_theme_id=raw_source_theme_id,
            target_theme_id=raw_target_theme_id,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "turn_index": self.turn_index,
            "edit_primitive": self.edit_primitive,
        }
        if self.source_theme_id is not None:
            payload["source_theme_id"] = self.source_theme_id
        if self.target_theme_id is not None:
            payload["target_theme_id"] = self.target_theme_id
        return payload

    def resolved_prompt_text(self) -> str:
        base_prompt = get_human_turn_injection_prompt(self.edit_primitive)
        if self.source_theme_id is None or self.target_theme_id is None:
            return base_prompt
        return (
            f"{base_prompt} Specifically switch from the benchmark-supported theme "
            f"'{self.source_theme_id}' ({_humanize_theme_id(self.source_theme_id)}) "
            f"to the benchmark-supported theme '{self.target_theme_id}' "
            f"({_humanize_theme_id(self.target_theme_id)})."
        )


def _normalize_required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string")
    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized_value


def _normalize_optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string when provided")
    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{field_name} must be a non-empty string when provided")
    return normalized_value


def _humanize_theme_id(theme_id: str) -> str:
    return theme_id.replace("_", " ")
