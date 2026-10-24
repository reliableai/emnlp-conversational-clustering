"""Loaded prompt-guidance data for the enterprise ticket preset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
_RAW_GUIDANCE = json.loads((RESOURCE_DIR / "guidance.json").read_text(encoding="utf-8"))


def _copy_mapping(name: str) -> dict[str, Any]:
    return dict(_RAW_GUIDANCE[name])


SEVERITY_RANKS: dict[str, int] = _copy_mapping("SEVERITY_RANKS")
URGENCY_RANKS: dict[str, int] = _copy_mapping("URGENCY_RANKS")
TEMPORAL_DIRECTIVE_MAP: dict[str, str] = _copy_mapping("TEMPORAL_DIRECTIVE_MAP")
INTERACTION_DIRECTIVE_MAP: dict[str, str] = _copy_mapping("INTERACTION_DIRECTIVE_MAP")
STRUCTURE_GUIDANCE: dict[str, str] = _copy_mapping("STRUCTURE_GUIDANCE")
KNOWLEDGE_GUIDANCE: dict[str, str] = _copy_mapping("KNOWLEDGE_GUIDANCE")
INTENT_GUIDANCE: dict[str, str] = _copy_mapping("INTENT_GUIDANCE")
EMOTION_GUIDANCE: dict[str, str] = _copy_mapping("EMOTION_GUIDANCE")
COMPLETENESS_GUIDANCE: dict[str, str] = _copy_mapping("COMPLETENESS_GUIDANCE")
TEMPORAL_GUIDANCE: dict[str, str] = _copy_mapping("TEMPORAL_GUIDANCE")
INTERACTION_GUIDANCE: dict[str, str] = _copy_mapping("INTERACTION_GUIDANCE")
WORKFLOW_GUIDANCE: dict[str, list[str]] = _copy_mapping("WORKFLOW_GUIDANCE")
APPROVAL_GUIDANCE: dict[str, list[str]] = _copy_mapping("APPROVAL_GUIDANCE")
DEPENDENCY_GUIDANCE: dict[str, list[str]] = _copy_mapping("DEPENDENCY_GUIDANCE")
DEPENDENCY_GUIDANCE_PROBABILITY: dict[str, float] = _copy_mapping("DEPENDENCY_GUIDANCE_PROBABILITY")
SEVERITY_GUIDANCE: dict[str, list[str]] = _copy_mapping("SEVERITY_GUIDANCE")
SEVERITY_GUIDANCE_PROBABILITY: dict[str, float] = _copy_mapping("SEVERITY_GUIDANCE_PROBABILITY")
IMPACT_GUIDANCE: dict[str, list[str]] = _copy_mapping("IMPACT_GUIDANCE")
IMPACT_GUIDANCE_PROBABILITY: dict[str, float] = _copy_mapping("IMPACT_GUIDANCE_PROBABILITY")
