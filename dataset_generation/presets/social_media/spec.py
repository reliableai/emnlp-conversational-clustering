"""Static resources and typed profiles for the social media preset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Required, TypeAlias, TypedDict

from . import spec_data


@dataclass(frozen=True)
class TextVariationProfile:
    opening_style: str
    sentence_style: str
    information_density: str
    cognitive_framing: str
    engagement_strategy: str
    length_style: str
    emoji_usage_pattern: str
    hashtag_usage_pattern: str
    entity_reference_style: str
    reply_context_style: str
    sarcasm_irony_level: str
    humor_type: str
    identity_signaling: str
    political_ideology_latent: str
    code_switching_mode: str
    virality_potential_pattern: str


class _SocialMediaRecordFields(TypedDict, total=False):
    platform: str
    topic_domain: str
    post_act: str
    content_scenario: str
    length_style: str
    stance: str
    sentiment_polarity: str
    voice_tone: str
    audience_targeting: str
    formality_style: str
    certainty_confidence: str
    toxicity_civility: str
    content_modality: str
    temporal_orientation: str
    content_label_coarse: str
    content_label_normal: str
    content_label_fine: str
    channel_label_coarse: str
    channel_label_normal: str
    channel_label_fine: str


class SocialMediaRecordDraft(_SocialMediaRecordFields):
    pass


class SocialMediaRecord(_SocialMediaRecordFields):
    platform: Required[str]
    topic_domain: Required[str]
    post_act: Required[str]
    content_scenario: Required[str]
    length_style: Required[str]
    stance: Required[str]
    sentiment_polarity: Required[str]
    voice_tone: Required[str]
    audience_targeting: Required[str]
    formality_style: Required[str]
    certainty_confidence: Required[str]
    toxicity_civility: Required[str]
    content_modality: Required[str]
    temporal_orientation: Required[str]
    content_label_coarse: Required[str]
    content_label_normal: Required[str]
    content_label_fine: Required[str]
    channel_label_coarse: Required[str]
    channel_label_normal: Required[str]
    channel_label_fine: Required[str]


SCHEMA = spec_data.SCHEMA
RecordDraft: TypeAlias = dict[str, str]
Record: TypeAlias = dict[str, str]

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"


def _load_text_resource(filename: str) -> str:
    return (RESOURCE_DIR / filename).read_text(encoding="utf-8").rstrip("\n")


OPENROUTER_SYSTEM_PROMPT = _load_text_resource("social_media_system_prompt.txt")

DEFAULT_OUTPUT_PATH = "data/social_media_dataset.csv"
CSV_COLUMNS = ["text", *SCHEMA.keys()]

SURFACE_METADATA_FIELDS = spec_data.SURFACE_METADATA_FIELDS
GRANULARITY_LABEL_COMPONENTS = spec_data.GRANULARITY_LABEL_COMPONENTS
GRANULARITY_LABEL_FIELDS = spec_data.GRANULARITY_LABEL_FIELDS
build_granularity_label = spec_data.build_granularity_label

TOPIC_DOMAIN_VALUES = tuple(spec_data.TOPIC_DOMAIN_VALUES)
PLATFORM_VALUES = tuple(spec_data.PLATFORM_VALUES)
LENGTH_STYLE_VALUES = tuple(spec_data.LENGTH_STYLE_VALUES)
TOPIC_POST_ACT_MAPPING = spec_data.TOPIC_POST_ACT_MAPPING
CONTENT_SCENARIO_MAPPING = spec_data.CONTENT_SCENARIO_MAPPING
POST_ACT_STANCE_MAPPING = spec_data.POST_ACT_STANCE_MAPPING
PLATFORM_AUDIENCE_MAPPING = spec_data.PLATFORM_AUDIENCE_MAPPING
PLATFORM_FORMALITY_MAPPING = spec_data.PLATFORM_FORMALITY_MAPPING
PLATFORM_MODALITY_MAPPING = spec_data.PLATFORM_MODALITY_MAPPING
SCENARIO_HINT_OVERRIDES = spec_data.SCENARIO_HINT_OVERRIDES
TOPIC_DOMAIN_WEIGHTS = spec_data.TOPIC_DOMAIN_WEIGHTS
PLATFORM_WEIGHTS = spec_data.PLATFORM_WEIGHTS
