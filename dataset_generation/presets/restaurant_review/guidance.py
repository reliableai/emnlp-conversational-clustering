"""Loaded prompt-guidance data for the restaurant review preset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
GUIDANCE_DATA: dict[str, Any] = json.loads((RESOURCE_DIR / "guidance.json").read_text("utf-8"))


def _copy_mapping(name: str) -> dict[str, Any]:
    return dict(GUIDANCE_DATA[name])


PRIMARY_FOCUS_GUIDANCE: dict[str, str] = _copy_mapping("PRIMARY_FOCUS_GUIDANCE")
SENTIMENT_GUIDANCE: dict[str, str] = _copy_mapping("SENTIMENT_GUIDANCE")
VISIT_MODE_GUIDANCE: dict[str, str] = _copy_mapping("VISIT_MODE_GUIDANCE")
OPENING_STYLE_GUIDANCE: dict[str, str] = _copy_mapping("OPENING_STYLE_GUIDANCE")
LENGTH_STYLE_GUIDANCE: dict[str, str] = _copy_mapping("LENGTH_STYLE_GUIDANCE")
DETAIL_STYLE_GUIDANCE: dict[str, str] = _copy_mapping("DETAIL_STYLE_GUIDANCE")
CLOSING_STYLE_GUIDANCE: dict[str, str] = _copy_mapping("CLOSING_STYLE_GUIDANCE")
REPETITION_AVOIDANCE_GUIDANCE: list[str] = list(GUIDANCE_DATA["REPETITION_AVOIDANCE_GUIDANCE"])
BANNED_TEMPLATE_SCAFFOLDS: tuple[str, ...] = tuple(GUIDANCE_DATA["BANNED_TEMPLATE_SCAFFOLDS"])
