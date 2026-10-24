"""Tests that presets load prompt resources from their checked-in resource files."""

from __future__ import annotations

import importlib


def test_social_media_imports_load_from_resource_files() -> None:
    guidance = importlib.import_module("dataset_generation.presets.social_media.guidance")
    spec = importlib.import_module("dataset_generation.presets.social_media.spec")

    importlib.reload(guidance)
    importlib.reload(spec)

    assert spec.OPENROUTER_SYSTEM_PROMPT.startswith("You generate synthetic social media posts")
    assert guidance.POST_ACT_GUIDANCE["ask_question"]


def test_restaurant_review_imports_load_from_resource_files() -> None:
    guidance = importlib.import_module("dataset_generation.presets.restaurant_review.guidance")
    spec = importlib.import_module("dataset_generation.presets.restaurant_review.spec")

    importlib.reload(guidance)
    importlib.reload(spec)

    assert spec.OPENROUTER_SYSTEM_PROMPT.startswith("You generate synthetic restaurant reviews")
    assert guidance.PRIMARY_FOCUS_GUIDANCE["food_quality"]
