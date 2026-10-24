"""Loaded prompt-guidance data for the social media preset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
GUIDANCE_DATA: dict[str, Any] = json.loads((RESOURCE_DIR / "guidance.json").read_text("utf-8"))


def _copy_mapping(name: str) -> dict[str, Any]:
    return dict(GUIDANCE_DATA[name])


POST_ACT_GUIDANCE: dict[str, str] = _copy_mapping("POST_ACT_GUIDANCE")
STANCE_GUIDANCE: dict[str, str] = _copy_mapping("STANCE_GUIDANCE")
SENTIMENT_GUIDANCE: dict[str, str] = _copy_mapping("SENTIMENT_GUIDANCE")
TONE_GUIDANCE: dict[str, str] = _copy_mapping("TONE_GUIDANCE")
AUDIENCE_GUIDANCE: dict[str, str] = _copy_mapping("AUDIENCE_GUIDANCE")
FORMALITY_GUIDANCE: dict[str, str] = _copy_mapping("FORMALITY_GUIDANCE")
CERTAINTY_GUIDANCE: dict[str, str] = _copy_mapping("CERTAINTY_GUIDANCE")
MODALITY_GUIDANCE: dict[str, str] = _copy_mapping("MODALITY_GUIDANCE")
TEMPORAL_GUIDANCE: dict[str, str] = _copy_mapping("TEMPORAL_GUIDANCE")
OPENING_STYLE_GUIDANCE: dict[str, str] = _copy_mapping("OPENING_STYLE_GUIDANCE")
SENTENCE_STYLE_GUIDANCE: dict[str, str] = _copy_mapping("SENTENCE_STYLE_GUIDANCE")
INFORMATION_DENSITY_GUIDANCE: dict[str, str] = _copy_mapping("INFORMATION_DENSITY_GUIDANCE")
COGNITIVE_FRAMING_GUIDANCE: dict[str, str] = _copy_mapping("COGNITIVE_FRAMING_GUIDANCE")
ENGAGEMENT_GUIDANCE: dict[str, str] = _copy_mapping("ENGAGEMENT_GUIDANCE")
LENGTH_STYLE_GUIDANCE: dict[str, str] = _copy_mapping("LENGTH_STYLE_GUIDANCE")
EMOJI_USAGE_GUIDANCE: dict[str, str] = _copy_mapping("EMOJI_USAGE_GUIDANCE")
HASHTAG_USAGE_GUIDANCE: dict[str, str] = _copy_mapping("HASHTAG_USAGE_GUIDANCE")
ENTITY_REFERENCE_GUIDANCE: dict[str, str] = _copy_mapping("ENTITY_REFERENCE_GUIDANCE")
REPLY_CONTEXT_GUIDANCE: dict[str, str] = _copy_mapping("REPLY_CONTEXT_GUIDANCE")
SARCASM_GUIDANCE: dict[str, str] = _copy_mapping("SARCASM_GUIDANCE")
HUMOR_GUIDANCE: dict[str, str] = _copy_mapping("HUMOR_GUIDANCE")
IDENTITY_SIGNALING_GUIDANCE: dict[str, str] = _copy_mapping("IDENTITY_SIGNALING_GUIDANCE")
IDEOLOGY_GUIDANCE: dict[str, str] = _copy_mapping("IDEOLOGY_GUIDANCE")
CODE_SWITCHING_GUIDANCE: dict[str, str] = _copy_mapping("CODE_SWITCHING_GUIDANCE")
VIRALITY_GUIDANCE: dict[str, str] = _copy_mapping("VIRALITY_GUIDANCE")
REPETITION_AVOIDANCE_GUIDANCE: list[str] = list(GUIDANCE_DATA["REPETITION_AVOIDANCE_GUIDANCE"])
BANNED_TEMPLATE_SCAFFOLDS: tuple[str, ...] = tuple(GUIDANCE_DATA["BANNED_TEMPLATE_SCAFFOLDS"])
