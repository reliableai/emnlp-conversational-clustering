"""Static resources and typed profiles for the restaurant review preset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from . import spec_data


@dataclass(frozen=True)
class TextVariationProfile:
    opening_style: str
    length_style: str
    detail_style: str
    closing_style: str


SCHEMA = spec_data.SCHEMA
RecordDraft: TypeAlias = dict[str, str]
Record: TypeAlias = dict[str, str]

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"


def _load_text_resource(filename: str) -> str:
    return (RESOURCE_DIR / filename).read_text(encoding="utf-8").rstrip("\n")


OPENROUTER_SYSTEM_PROMPT = _load_text_resource("restaurant_review_system_prompt.txt")

DEFAULT_OUTPUT_PATH = "data/restaurant_review_dataset.csv"
CSV_COLUMNS = ["text", *SCHEMA.keys()]

SURFACE_METADATA_FIELDS = spec_data.SURFACE_METADATA_FIELDS
GRANULARITY_LABEL_COMPONENTS = spec_data.GRANULARITY_LABEL_COMPONENTS
GRANULARITY_LABEL_FIELDS = spec_data.GRANULARITY_LABEL_FIELDS
build_granularity_label = spec_data.build_granularity_label

RESTAURANT_CATEGORY_VALUES = tuple(spec_data.RESTAURANT_CATEGORY_VALUES)
PRIMARY_FOCUS_VALUES = tuple(spec_data.PRIMARY_FOCUS_VALUES)
SENTIMENT_POLARITY_VALUES = tuple(spec_data.SENTIMENT_POLARITY_VALUES)
RATING_SCORE_VALUES = tuple(spec_data.RATING_SCORE_VALUES)
LENGTH_STYLE_VALUES = tuple(spec_data.LENGTH_STYLE_VALUES)
VISIT_MODE_MAPPING = spec_data.VISIT_MODE_MAPPING
RATING_TO_SENTIMENT = spec_data.RATING_TO_SENTIMENT
REVIEW_SCENARIO_MAPPING = spec_data.REVIEW_SCENARIO_MAPPING
REVIEW_SCENARIO_HINT_OVERRIDES = spec_data.REVIEW_SCENARIO_HINT_OVERRIDES
RESTAURANT_CATEGORY_WEIGHTS = spec_data.RESTAURANT_CATEGORY_WEIGHTS
PRIMARY_FOCUS_WEIGHTS = spec_data.PRIMARY_FOCUS_WEIGHTS
RATING_SCORE_WEIGHTS = spec_data.RATING_SCORE_WEIGHTS
