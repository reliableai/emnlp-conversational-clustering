"""Topic discovery schemas and validation helpers for simulated human runs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, model_validator

_TOPIC_ID_FALLBACK = "topic"
_TRIGGER_MODES = {"turn_budget", "human_tool"}


def slugify_topic_label(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", label.strip().lower()).strip("-")
    if normalized:
        return normalized
    return _TOPIC_ID_FALLBACK


class TopicCandidate(BaseModel):
    topic_id: str = ""
    label: str
    guidance: str
    rationale: str

    @model_validator(mode="after")
    def _normalize_fields(self) -> TopicCandidate:
        label = self.label.strip()
        guidance = self.guidance.strip()
        rationale = self.rationale.strip()
        if not label:
            raise ValueError("label must be non-empty")
        if not guidance:
            raise ValueError("guidance must be non-empty")
        if not rationale:
            raise ValueError("rationale must be non-empty")
        self.label = label
        self.guidance = guidance
        self.rationale = rationale
        self.topic_id = slugify_topic_label(label)
        return self


@dataclass(frozen=True)
class TopicDiscoveryConfig:
    enabled: bool = False
    trigger_mode: Literal["turn_budget", "human_tool"] = "turn_budget"
    turn_budget: int | None = None
    max_rounds: int = 3
    min_new_topics_per_round: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        if self.trigger_mode not in _TRIGGER_MODES:
            supported = ", ".join(sorted(_TRIGGER_MODES))
            raise ValueError(f"trigger_mode must be one of: {supported}")
        if isinstance(self.max_rounds, bool) or not isinstance(self.max_rounds, int):
            raise ValueError("max_rounds must be an integer >= 1")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be an integer >= 1")
        if isinstance(self.min_new_topics_per_round, bool) or not isinstance(
            self.min_new_topics_per_round, int
        ):
            raise ValueError("min_new_topics_per_round must be an integer >= 0")
        if self.min_new_topics_per_round < 0:
            raise ValueError("min_new_topics_per_round must be an integer >= 0")
        if not self.enabled:
            return
        if self.trigger_mode == "turn_budget":
            if isinstance(self.turn_budget, bool) or not isinstance(self.turn_budget, int):
                raise ValueError(
                    "turn_budget must be an integer >= 1 when trigger_mode is turn_budget"
                )
            if self.turn_budget < 1:
                raise ValueError(
                    "turn_budget must be an integer >= 1 when trigger_mode is turn_budget"
                )
            return
        if self.turn_budget is not None:
            raise ValueError("turn_budget must be null when trigger_mode is human_tool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "trigger_mode": self.trigger_mode,
            "turn_budget": self.turn_budget,
            "max_rounds": self.max_rounds,
            "min_new_topics_per_round": self.min_new_topics_per_round,
        }
