"""Prompt construction and alignment checks for the social media preset."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from dataset_generation.llm import generated_text_response_schema
from dataset_generation.sampling import clamp_probability, pick_one
from dataset_generation.schema import prompt_safe_schema
from dataset_generation.types import DeterministicEvaluationResult, EvaluationRating
from dataset_generation.utils import (
    build_alignment_retry_prefix as build_shared_alignment_retry_prefix,
)
from dataset_generation.utils import (
    build_surface_metadata as build_shared_surface_metadata,
)
from dataset_generation.utils import (
    humanize_slug,
    normalize_text,
)

from . import guidance, spec

OPENING_STYLE_OPTIONS = tuple(guidance.OPENING_STYLE_GUIDANCE)
SENTENCE_STYLE_OPTIONS = tuple(guidance.SENTENCE_STYLE_GUIDANCE)
INFORMATION_DENSITY_OPTIONS = tuple(guidance.INFORMATION_DENSITY_GUIDANCE)
COGNITIVE_FRAMING_OPTIONS = tuple(guidance.COGNITIVE_FRAMING_GUIDANCE)
ENGAGEMENT_STRATEGY_OPTIONS = tuple(guidance.ENGAGEMENT_GUIDANCE)
LENGTH_STYLE_OPTIONS = tuple(guidance.LENGTH_STYLE_GUIDANCE)
EMOJI_USAGE_OPTIONS = tuple(guidance.EMOJI_USAGE_GUIDANCE)
HASHTAG_USAGE_OPTIONS = tuple(guidance.HASHTAG_USAGE_GUIDANCE)
REPLY_CONTEXT_OPTIONS = tuple(guidance.REPLY_CONTEXT_GUIDANCE)
SARCASM_OPTIONS = tuple(guidance.SARCASM_GUIDANCE)
HUMOR_OPTIONS = tuple(guidance.HUMOR_GUIDANCE)
IDENTITY_SIGNALING_OPTIONS = tuple(guidance.IDENTITY_SIGNALING_GUIDANCE)
IDEOLOGY_OPTIONS = tuple(guidance.IDEOLOGY_GUIDANCE)
CODE_SWITCHING_OPTIONS = tuple(guidance.CODE_SWITCHING_GUIDANCE)
VIRALITY_OPTIONS = tuple(guidance.VIRALITY_GUIDANCE)

_ABBREVIATION_REPLACEMENTS = {
    "ai": "AI",
    "pmf": "product-market fit",
    "ev": "EV",
    "saas": "SaaS",
    "pr": "PR",
}

_QUESTION_LEADS = (
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "anyone",
    "does anyone",
    "is anyone",
)

_ACTION_HINTS = (
    "join",
    "sign",
    "register",
    "show up",
    "call",
    "email",
    "reply",
    "share",
    "vote",
    "come through",
)

_PROMO_HINTS = (
    "launch",
    "available",
    "open",
    "signup",
    "sign up",
    "free trial",
    "demo",
    "invite",
    "discount",
)


@dataclass(frozen=True)
class AlignmentAssessment:
    scenario_match: bool
    modality_match: bool
    act_match: bool
    length_match: bool
    banned_scaffold: str


def _normalize_text(value: str) -> str:
    return normalize_text(value)


def _scenario_tokens(content_scenario: str) -> tuple[str, ...]:
    override_terms = spec.SCENARIO_HINT_OVERRIDES.get(content_scenario, ())
    terms: list[str] = []
    for token in content_scenario.split("_"):
        normalized = _ABBREVIATION_REPLACEMENTS.get(token, token)
        normalized = normalized.lower().strip()
        if len(normalized) >= 4:
            terms.append(normalized)
    for override in override_terms:
        terms.append(override.lower())
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term not in seen:
            seen.add(term)
            deduped.append(term)
    return tuple(deduped)


def _humanize_slug(value: str) -> str:
    return humanize_slug(value, _ABBREVIATION_REPLACEMENTS)


def preferred_content_scenario_phrase(content_scenario: str) -> str:
    phrase = _humanize_slug(content_scenario)
    return phrase.replace("  ", " ").strip()


def build_surface_metadata(record: spec.Record) -> dict[str, str]:
    return build_shared_surface_metadata(record, spec.SURFACE_METADATA_FIELDS)


def _length_word_bounds(length_style: str) -> tuple[int, int]:
    return {
        "very_short": (3, 12),
        "short": (8, 24),
        "medium": (18, 45),
        "long": (35, 75),
        "threadish": (45, 110),
    }[length_style]


def _length_target_guidance(length_style: str) -> str:
    minimum, maximum = _length_word_bounds(length_style)
    return f"target roughly {minimum}-{maximum} words so the post clearly reads as {length_style}"


def build_cluster_labels(record: spec.Record) -> dict[str, str]:
    labels: dict[str, str] = {}
    for family, levels in spec.GRANULARITY_LABEL_FIELDS.items():
        for level, field_name in levels.items():
            labels[f"{family}_{level}"] = record[field_name]
    return labels


def _sample_entity_reference_style(
    record: spec.Record,
    rng: random.Random,
    entity_mention_probability: float | None,
) -> str:
    if entity_mention_probability is not None:
        if rng.random() < clamp_probability(entity_mention_probability):
            if record["topic_domain"] == "politics":
                return "named_public_figure"
            if record["topic_domain"] in {"education", "environment_climate"}:
                return "named_local_institution"
            return "named_org_or_product"
        return "generic_reference"

    if record["topic_domain"] == "politics":
        return pick_one(rng, ["named_public_figure", "generic_reference", "generic_reference"])
    if record["topic_domain"] in {"technology", "business_startups"}:
        return pick_one(rng, ["named_org_or_product", "generic_reference", "named_org_or_product"])
    return pick_one(rng, ["generic_reference", "generic_reference", "named_local_institution"])


def sample_text_variation_profile(
    record: spec.Record,
    rng: random.Random,
    difficulty: str,
    *,
    entity_mention_probability: float | None,
) -> spec.TextVariationProfile:
    opening_style = pick_one(rng, list(OPENING_STYLE_OPTIONS))
    if record["post_act"] == "ask_question":
        opening_style = pick_one(rng, ["question_open", "direct", "context_first"])
    elif record["post_act"] == "narrate_personal_experience":
        opening_style = pick_one(rng, ["anecdote_open", "status_update", "observation"])
    elif record["content_modality"] == "reply":
        opening_style = pick_one(rng, ["direct", "question_open", "context_first"])

    sentence_style = pick_one(rng, list(SENTENCE_STYLE_OPTIONS))
    if difficulty == "easy":
        sentence_style = pick_one(rng, ["single", "short_plus_followup", "two_clause"])
    elif difficulty == "hard":
        sentence_style = pick_one(
            rng,
            ["narrative", "contrast", "two_clause", "short_plus_followup"],
        )

    information_density = pick_one(rng, list(INFORMATION_DENSITY_OPTIONS))
    if record["certainty_confidence"] == "data_backed":
        information_density = pick_one(rng, ["high", "very_high", "data_heavy"])
    elif record["content_modality"] == "text_plus_link":
        information_density = pick_one(rng, ["medium", "high", "link_heavy"])

    cognitive_framing = pick_one(rng, list(COGNITIVE_FRAMING_OPTIONS))
    if record["post_act"] == "narrate_personal_experience":
        cognitive_framing = pick_one(
            rng,
            ["personal_anecdote", "emotional_reaction", "analogy_metaphor"],
        )
    elif record["post_act"] == "debate_argue":
        cognitive_framing = pick_one(
            rng,
            ["logical_argument", "moral_framing", "rhetorical_question"],
        )

    engagement_strategy = pick_one(rng, list(ENGAGEMENT_STRATEGY_OPTIONS))
    if record["post_act"] == "ask_question":
        engagement_strategy = pick_one(rng, ["question_prompt", "direct_call_to_respond", "none"])
    elif record["post_act"] == "mobilize_call_to_action":
        engagement_strategy = pick_one(
            rng,
            ["direct_call_to_respond", "emotional_appeal", "hashtag_driven"],
        )

    length_style = record["length_style"]

    emoji_usage_pattern = pick_one(rng, list(EMOJI_USAGE_OPTIONS))
    if record["platform"] == "linkedin_like":
        emoji_usage_pattern = pick_one(rng, ["none", "light"])
    elif record["formality_style"] in {"slang_heavy", "meme_style"}:
        emoji_usage_pattern = pick_one(rng, ["light", "moderate", "expressive"])

    hashtag_usage_pattern = pick_one(rng, list(HASHTAG_USAGE_OPTIONS))
    if record["content_modality"] == "thread_starter":
        hashtag_usage_pattern = pick_one(rng, ["none", "light"])
    elif record["post_act"] == "mobilize_call_to_action":
        hashtag_usage_pattern = pick_one(rng, ["light", "campaign", "campaign"])
    elif record["audience_targeting"] == "fans_fandom":
        hashtag_usage_pattern = pick_one(rng, ["light", "fandom", "none"])

    entity_reference_style = _sample_entity_reference_style(
        record,
        rng,
        entity_mention_probability,
    )

    reply_context_style = "none"
    if record["content_modality"] == "reply":
        reply_context_style = pick_one(
            rng,
            ["direct_reply", "agree_then_add", "pushback_reply"],
        )
    elif record["content_modality"] == "quote_post":
        reply_context_style = "quote_and_reframe"

    sarcasm_irony_level = "none_literal"
    if record["voice_tone"] == "sarcastic":
        sarcasm_irony_level = pick_one(rng, ["mild_irony", "clear_sarcasm", "ambiguous"])
    elif difficulty == "hard" and record["post_act"] in {
        "debate_argue",
        "vent_rant",
        "entertain_joke",
    }:
        sarcasm_irony_level = pick_one(rng, ["none_literal", "mild_irony", "ambiguous"])

    humor_type = "no_humor"
    if record["post_act"] == "entertain_joke":
        humor_type = pick_one(
            rng,
            ["dry_subtle", "sarcastic_humor", "meme_based", "self_deprecating", "observational"],
        )
    elif record["voice_tone"] == "sarcastic":
        humor_type = pick_one(rng, ["no_humor", "sarcastic_humor", "dry_subtle"])

    identity_signaling = "none"
    if record["audience_targeting"] == "industry_peers":
        identity_signaling = pick_one(
            rng,
            ["professional_identity", "subtle_affiliation", "none"],
        )
    elif record["audience_targeting"] == "fans_fandom":
        identity_signaling = pick_one(
            rng,
            ["fandom_identity", "strong_in_group_signaling", "none"],
        )
    elif record["audience_targeting"] == "political_group":
        identity_signaling = pick_one(
            rng,
            [
                "political_identity_signaling",
                "strong_in_group_signaling",
                "subtle_affiliation",
            ],
        )

    political_ideology_latent = "none"
    if record["topic_domain"] == "politics":
        political_ideology_latent = pick_one(
            rng,
            [
                "strongly_progressive",
                "moderate_progressive",
                "centrist",
                "moderate_conservative",
                "strongly_conservative",
                "libertarian",
                "apolitical_unclear",
            ],
        )

    code_switching_mode = "none"
    if record["formality_style"] in {"slang_heavy", "casual_conversational", "meme_style"}:
        code_switching_mode = pick_one(
            rng,
            ["none", "light_internet_slang", "light_internet_slang", "regional_informal"],
        )

    virality_potential_pattern = pick_one(rng, list(VIRALITY_OPTIONS))
    if record["post_act"] in {"inform_share_news", "debate_argue"}:
        virality_potential_pattern = pick_one(
            rng,
            ["moderate", "news_amplification", "high_relatable", "very_high_controversial"],
        )
    elif record["post_act"] == "narrate_personal_experience":
        virality_potential_pattern = pick_one(
            rng,
            ["low_personal_note", "moderate", "high_relatable"],
        )

    return spec.TextVariationProfile(
        opening_style=opening_style,
        sentence_style=sentence_style,
        information_density=information_density,
        cognitive_framing=cognitive_framing,
        engagement_strategy=engagement_strategy,
        length_style=length_style,
        emoji_usage_pattern=emoji_usage_pattern,
        hashtag_usage_pattern=hashtag_usage_pattern,
        entity_reference_style=entity_reference_style,
        reply_context_style=reply_context_style,
        sarcasm_irony_level=sarcasm_irony_level,
        humor_type=humor_type,
        identity_signaling=identity_signaling,
        political_ideology_latent=political_ideology_latent,
        code_switching_mode=code_switching_mode,
        virality_potential_pattern=virality_potential_pattern,
    )


def build_hidden_generation_guidance(
    record: spec.Record,
    variation_profile: spec.TextVariationProfile,
) -> list[str]:
    guidance_items = [
        guidance.POST_ACT_GUIDANCE[record["post_act"]],
        guidance.STANCE_GUIDANCE[record["stance"]],
        guidance.SENTIMENT_GUIDANCE[record["sentiment_polarity"]],
        guidance.TONE_GUIDANCE[record["voice_tone"]],
        guidance.AUDIENCE_GUIDANCE[record["audience_targeting"]],
        guidance.FORMALITY_GUIDANCE[record["formality_style"]],
        guidance.CERTAINTY_GUIDANCE[record["certainty_confidence"]],
        guidance.MODALITY_GUIDANCE[record["content_modality"]],
        guidance.TEMPORAL_GUIDANCE[record["temporal_orientation"]],
        guidance.OPENING_STYLE_GUIDANCE[variation_profile.opening_style],
        guidance.SENTENCE_STYLE_GUIDANCE[variation_profile.sentence_style],
        guidance.INFORMATION_DENSITY_GUIDANCE[variation_profile.information_density],
        guidance.COGNITIVE_FRAMING_GUIDANCE[variation_profile.cognitive_framing],
        guidance.ENGAGEMENT_GUIDANCE[variation_profile.engagement_strategy],
        guidance.LENGTH_STYLE_GUIDANCE[variation_profile.length_style],
        _length_target_guidance(variation_profile.length_style),
        guidance.EMOJI_USAGE_GUIDANCE[variation_profile.emoji_usage_pattern],
        guidance.HASHTAG_USAGE_GUIDANCE[variation_profile.hashtag_usage_pattern],
        guidance.ENTITY_REFERENCE_GUIDANCE[variation_profile.entity_reference_style],
        guidance.REPLY_CONTEXT_GUIDANCE[variation_profile.reply_context_style],
    ]
    if variation_profile.sarcasm_irony_level != "none_literal":
        guidance_items.append(guidance.SARCASM_GUIDANCE[variation_profile.sarcasm_irony_level])
    if variation_profile.humor_type != "no_humor":
        guidance_items.append(guidance.HUMOR_GUIDANCE[variation_profile.humor_type])
    if variation_profile.identity_signaling != "none":
        guidance_items.append(
            guidance.IDENTITY_SIGNALING_GUIDANCE[variation_profile.identity_signaling]
        )
    if variation_profile.political_ideology_latent != "none":
        guidance_items.append(
            guidance.IDEOLOGY_GUIDANCE[variation_profile.political_ideology_latent]
        )
    if variation_profile.code_switching_mode != "none":
        guidance_items.append(
            guidance.CODE_SWITCHING_GUIDANCE[variation_profile.code_switching_mode]
        )
    guidance_items.append(guidance.VIRALITY_GUIDANCE[variation_profile.virality_potential_pattern])
    return guidance_items


def build_generation_controls(variation_profile: spec.TextVariationProfile) -> dict[str, str]:
    return {
        "opening_style": variation_profile.opening_style,
        "sentence_style": variation_profile.sentence_style,
        "information_density": variation_profile.information_density,
        "cognitive_framing": variation_profile.cognitive_framing,
        "engagement_strategy": variation_profile.engagement_strategy,
        "length_style": variation_profile.length_style,
        "emoji_usage_pattern": variation_profile.emoji_usage_pattern,
        "hashtag_usage_pattern": variation_profile.hashtag_usage_pattern,
        "entity_reference_style": variation_profile.entity_reference_style,
        "reply_context_style": variation_profile.reply_context_style,
        "sarcasm_irony_level": variation_profile.sarcasm_irony_level,
        "humor_type": variation_profile.humor_type,
        "identity_signaling": variation_profile.identity_signaling,
        "political_ideology_latent": variation_profile.political_ideology_latent,
        "code_switching_mode": variation_profile.code_switching_mode,
        "virality_potential_pattern": variation_profile.virality_potential_pattern,
        "target_word_range": "-".join(
            str(bound) for bound in _length_word_bounds(variation_profile.length_style)
        ),
    }


def build_openrouter_batch_prompt(
    batch_rows: list[dict[str, Any]],
    difficulty: str,
) -> str:
    schema_for_prompt = prompt_safe_schema(spec.SCHEMA, spec.SURFACE_METADATA_FIELDS)
    repetition_rules = "\n- ".join(guidance.REPETITION_AVOIDANCE_GUIDANCE)
    return (
        "Generate text values for the social media rows below.\n"
        "The surface_metadata block is authoritative ground truth for the public-facing post.\n"
        "Treat topic_domain as the broad cluster, post_act as the communicative action, and "
        "content_scenario as the concrete leaf scenario that should be obvious in the text.\n"
        "Treat length_style as authoritative for how compact or expanded the final post "
        "should be.\n"
        "The cluster_labels block contains derived coarse/normal/fine labels for downstream "
        "clustering; do not copy those labels into the post.\n"
        "The hidden_generation_guidance list gives private style and realism hints. Translate "
        "those hints into natural writing rather than copying them.\n"
        "The generation_controls block is binding for opening style, target length, sentence "
        "rhythm, density, engagement shape, modality behavior, and optional irony or humor.\n"
        "Write only the natural-language post text for each row_id.\n"
        "Do not use headings, bullets, or JSON-like wording inside the post.\n"
        "Do not copy snake_case labels verbatim.\n"
        "Keep posts varied even when multiple rows share the same topic_domain.\n"
        "For content_modality requirements: '[image attached]' and '[video attached]' must appear "
        "exactly when requested, a plausible link token must appear for text_plus_link, and "
        "thread_starter should feel like the opening of a longer thread.\n"
        "Question posts should actually sound like questions, call-to-action posts should ask for "
        "a real action, and promotional posts should still sound like real social content.\n"
        "Avoid stale row openers such as: "
        f"{', '.join(repr(value) for value in guidance.BANNED_TEMPLATE_SCAFFOLDS)}.\n"
        "Repetition-avoidance rules:\n"
        f"- {repetition_rules}\n"
        f"Difficulty preset: {difficulty}\n"
        "Return JSON that matches this schema exactly:\n"
        f"{json.dumps(generated_text_response_schema(), indent=2)}\n\n"
        "Surface metadata schema:\n"
        f"{json.dumps(schema_for_prompt, indent=2)}\n\n"
        "Rows:\n"
        f"{json.dumps(batch_rows, indent=2)}"
    )


def build_evaluation_row_payload(row_id: int, row: dict[str, str]) -> dict[str, Any]:
    metadata = {field: row[field] for field in spec.SCHEMA.keys()}
    return {
        "row_id": row_id,
        "text": _normalize_text(row.get("text", "")),
        "metadata": metadata,
    }


def evaluation_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "row_id": {"type": "integer"},
                        "llm_rating": {
                            "type": "string",
                            "enum": ["pass", "need_attention", "not_pass"],
                        },
                        "llm_reason": {"type": "string"},
                    },
                    "required": ["row_id", "llm_rating", "llm_reason"],
                },
            }
        },
        "required": ["items"],
    }


def build_openrouter_evaluation_prompt(batch_rows: list[dict[str, Any]]) -> str:
    return (
        "Evaluate each social-media dataset row for label-to-text alignment and realism.\n"
        "The most important fields are topic_domain, post_act, content_scenario, platform, and "
        "content_modality.\n"
        "Do not require every metadata field to be stated literally.\n"
        "A row can still pass if the post paraphrases the leaf scenario naturally.\n"
        "Treat content_scenario as a concrete underlying scenario, not a phrase that must appear "
        "verbatim.\n"
        "Ratings:\n"
        "  pass: metadata and text align clearly and the post sounds realistic.\n"
        "  need_attention: mostly plausible but vague, templated, or partially misaligned.\n"
        "  not_pass: the text contradicts the leaf scenario or misses the scenario badly.\n"
        "Return JSON only, matching this schema exactly:\n"
        f"{json.dumps(evaluation_response_schema(), indent=2)}\n\n"
        "Rows:\n"
        f"{json.dumps(batch_rows, indent=2)}"
    )


def _required_modality_marker(content_modality: str) -> str | None:
    if content_modality == "text_plus_image":
        return "[image attached]"
    if content_modality == "text_plus_video":
        return "[video attached]"
    return None


def _matches_modality(content_modality: str, text: str) -> bool:
    normalized = text.lower()
    marker = _required_modality_marker(content_modality)
    if marker is not None:
        return marker in normalized
    if content_modality == "text_plus_link":
        return any(token in normalized for token in ("http://", "https://", "www.", ".com/"))
    if content_modality == "poll":
        return "?" in text and any(
            token in normalized for token in ("vote", "poll", "which", "pick one")
        )
    if content_modality == "thread_starter":
        return any(
            token in normalized for token in ("thread", "1/", "more below", "more in thread")
        )
    if content_modality == "quote_post":
        return any(
            token in normalized
            for token in ("that post", "this take", "seeing people", "watching people", "quote")
        )
    return True


def _matches_generation_modality(content_modality: str, text: str) -> bool:
    if content_modality in {"text_plus_image", "text_plus_video", "text_plus_link"}:
        return _matches_modality(content_modality, text)
    return True


def _matches_post_act(post_act: str, text: str) -> bool:
    normalized = text.lower()
    if post_act == "ask_question":
        return "?" in text or normalized.startswith(_QUESTION_LEADS)
    if post_act == "mobilize_call_to_action":
        return any(token in normalized for token in _ACTION_HINTS)
    if post_act == "promote_advertise":
        return any(token in normalized for token in _PROMO_HINTS)
    return True


def _find_banned_scaffold(text: str) -> str:
    normalized = _normalize_text(text).lower()
    for scaffold in guidance.BANNED_TEMPLATE_SCAFFOLDS:
        if normalized.startswith(scaffold):
            return scaffold
    return ""


def assess_social_post_alignment(record: spec.Record, text: str) -> AlignmentAssessment:
    normalized = _normalize_text(text).lower()
    scenario_terms = _scenario_tokens(record["content_scenario"])
    scenario_match = any(term in normalized for term in scenario_terms)
    word_count = len(_normalize_text(text).split())
    minimum_words, maximum_words = _length_word_bounds(record["length_style"])
    return AlignmentAssessment(
        scenario_match=scenario_match,
        modality_match=_matches_modality(record["content_modality"], normalized),
        act_match=_matches_post_act(record["post_act"], normalized),
        length_match=minimum_words <= word_count <= maximum_words,
        banned_scaffold=_find_banned_scaffold(text),
    )


def evaluate_social_post_deterministically(
    record: spec.Record,
    text: str,
) -> DeterministicEvaluationResult:
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return DeterministicEvaluationResult(
            rating="not_pass",
            reasons=("empty_text",),
            details={
                "scenario_match": "false",
                "modality_match": "false",
                "act_match": "false",
                "length_match": "false",
                "banned_scaffold": "",
            },
        )

    assessment = assess_social_post_alignment(record, normalized_text)
    rating: EvaluationRating = "pass"
    reasons: list[str] = []
    if not assessment.scenario_match:
        rating = "need_attention"
        reasons.append("content_scenario_mismatch")
    if not assessment.modality_match:
        if rating != "not_pass":
            rating = "need_attention"
        reasons.append("content_modality_mismatch")
    if not assessment.act_match:
        if rating != "not_pass":
            rating = "need_attention"
        reasons.append("post_act_shape_mismatch")
    if not assessment.length_match:
        if rating != "not_pass":
            rating = "need_attention"
        reasons.append("length_style_mismatch")
    if assessment.banned_scaffold:
        if rating != "not_pass":
            rating = "need_attention"
        reasons.append(f"banned_template_scaffold:{assessment.banned_scaffold}")
    if len(normalized_text.split()) < 6:
        if rating != "not_pass":
            rating = "need_attention"
        reasons.append("very_short_text")

    return DeterministicEvaluationResult(
        rating=rating,
        reasons=tuple(reasons),
        details={
            "scenario_match": str(assessment.scenario_match).lower(),
            "modality_match": str(assessment.modality_match).lower(),
            "act_match": str(assessment.act_match).lower(),
            "length_match": str(assessment.length_match).lower(),
            "banned_scaffold": assessment.banned_scaffold,
        },
    )


def _alignment_terms_preview(record: spec.Record, limit: int = 8) -> str:
    return ", ".join(_scenario_tokens(record["content_scenario"])[:limit])


def collect_alignment_failures(
    records: list[spec.Record],
    parsed: dict[int, str],
) -> tuple[dict[int, ValueError], dict[int, ValueError]]:
    hard_failures: dict[int, ValueError] = {}
    soft_failures: dict[int, ValueError] = {}
    for row_id, text in parsed.items():
        record = records[row_id]
        assessment = assess_social_post_alignment(record, text)
        if not _matches_generation_modality(record["content_modality"], text):
            soft_failures[row_id] = ValueError(
                "Generated post did not satisfy the requested content_modality "
                f"({record['content_modality']})."
            )
            continue
        if not assessment.scenario_match:
            soft_failures[row_id] = ValueError(
                "Generated post matched the broad social-media framing but did not make the "
                "sampled leaf scenario explicit for "
                f"content_scenario={record['content_scenario']!r}. "
                f"Helpful concepts included: {_alignment_terms_preview(record, limit=12)}."
            )
            continue
        if not assessment.act_match:
            soft_failures[row_id] = ValueError(
                f"Generated post did not clearly match the communicative act {record['post_act']}."
            )
            continue
        if not assessment.length_match:
            soft_failures[row_id] = ValueError(
                "Generated post did not match the requested length_style strongly enough."
            )
            continue
        if assessment.banned_scaffold:
            soft_failures[row_id] = ValueError(
                "Generated post opened with a discouraged template scaffold "
                f"('{assessment.banned_scaffold}')."
            )
    return hard_failures, soft_failures


def build_alignment_retry_prefix(
    records: list[spec.Record],
    parsed: dict[int, str],
) -> str | None:
    hard_failures, soft_failures = collect_alignment_failures(records, parsed)
    messages: list[str] = []
    for row_id in sorted({*hard_failures, *soft_failures}):
        record = records[row_id]
        failure = hard_failures.get(row_id) or soft_failures.get(row_id)
        messages.append(
            f"Row {row_id}: keep the same metadata but rewrite the post so the leaf scenario "
            f"'{record['content_scenario']}' is obvious from natural wording.\n"
            f"Rewrite reason: {failure}\n"
            f"Helpful scenario terms: {_alignment_terms_preview(record, limit=12)}\n"
            f"Failed text: {parsed[row_id]}"
        )

    return build_shared_alignment_retry_prefix(
        messages,
        intro_lines=(
            "The previous response violated alignment or modality constraints for some rows.",
            "Retry only the rows listed below and return valid JSON only.",
            "Keep every row_id exactly the same.",
        ),
    )


def _append_modality_suffix(text: str, content_modality: str) -> str:
    if content_modality == "text_plus_image":
        return f"{text} [image attached]"
    if content_modality == "text_plus_video":
        return f"{text} [video attached]"
    if content_modality == "text_plus_link":
        return f"{text} https://example.com/story"
    if content_modality == "poll":
        return f"{text} Vote below?"
    if content_modality == "thread_starter":
        return f"{text} More in thread."
    if content_modality == "quote_post":
        return f"Seeing people say the opposite, but {text}"
    return text


def _ensure_question_shape(text: str) -> str:
    stripped = text.rstrip()
    if stripped.endswith("?"):
        return stripped
    return stripped.rstrip(".!") + "?"


def _ensure_action_shape(text: str) -> str:
    normalized = _normalize_text(text)
    if any(token in normalized.lower() for token in _ACTION_HINTS):
        return normalized
    return f"{normalized} Please show up, share it, or reply if you're in."


def _ensure_promo_shape(text: str) -> str:
    normalized = _normalize_text(text)
    if any(token in normalized.lower() for token in _PROMO_HINTS):
        return normalized
    return f"{normalized} It's live now if you want to check it out."


def _ensure_leaf_scenario_visibility(record: spec.Record, text: str) -> str:
    if assess_social_post_alignment(record, text).scenario_match:
        return text
    scenario_phrase = preferred_content_scenario_phrase(record["content_scenario"])
    return f"{text} Specifically, this is about {scenario_phrase}."


def _expand_to_length_style(record: spec.Record, text: str) -> str:
    minimum_words, _ = _length_word_bounds(record["length_style"])
    expanded = _normalize_text(text)
    scenario_phrase = preferred_content_scenario_phrase(record["content_scenario"])
    padding_sentences = [
        f"The part people keep circling back to is {scenario_phrase}.",
        (
            "It keeps landing differently once you look at the actual context around "
            f"{scenario_phrase}."
        ),
        "That is the piece making this feel more concrete than a generic timeline update.",
    ]
    for sentence in padding_sentences:
        if len(expanded.split()) >= minimum_words:
            break
        expanded = f"{expanded} {sentence}"
    return expanded


def _repair_from_generated_text(record: spec.Record, text: str) -> str:
    repaired = _normalize_text(text)
    if not repaired:
        return build_aligned_fallback_social_post(record)

    banned_scaffold = _find_banned_scaffold(repaired)
    if banned_scaffold:
        repaired = repaired[len(banned_scaffold) :].lstrip(" ,:-")
        repaired = _normalize_text(repaired)
    if not repaired:
        return build_aligned_fallback_social_post(record)

    repaired = _ensure_leaf_scenario_visibility(record, repaired)
    if record["post_act"] == "ask_question":
        repaired = _ensure_question_shape(repaired)
    elif record["post_act"] == "mobilize_call_to_action":
        repaired = _ensure_action_shape(repaired)
    elif record["post_act"] == "promote_advertise":
        repaired = _ensure_promo_shape(repaired)

    if not _matches_modality(record["content_modality"], repaired):
        repaired = _append_modality_suffix(repaired, record["content_modality"])

    repaired = _expand_to_length_style(record, repaired)

    final_assessment = assess_social_post_alignment(record, repaired)
    if (
        not final_assessment.scenario_match
        or not final_assessment.modality_match
        or not final_assessment.act_match
        or not final_assessment.length_match
    ):
        return build_aligned_fallback_social_post(record)
    return repaired


def build_aligned_fallback_social_post(record: spec.Record) -> str:
    scenario_phrase = preferred_content_scenario_phrase(record["content_scenario"])
    post_act = record["post_act"]
    if post_act == "inform_share_news":
        text = f"New update on {scenario_phrase}: this is moving faster than I expected."
    elif post_act == "express_opinion":
        text = f"My read on {scenario_phrase} is that people are missing the bigger point."
    elif post_act == "ask_question":
        text = f"Anyone else have a good read on {scenario_phrase}?"
    elif post_act == "promote_advertise":
        text = f"We just opened {scenario_phrase} and would love early feedback."
    elif post_act == "entertain_joke":
        text = f"At this point {scenario_phrase} feels like its own genre."
    elif post_act == "vent_rant":
        text = f"Still annoyed about {scenario_phrase}. This is getting old fast."
    elif post_act == "mobilize_call_to_action":
        text = f"If you care about {scenario_phrase}, please show up and do something today."
    elif post_act == "debate_argue":
        text = f"Hot take: {scenario_phrase} is where the real argument actually is."
    else:
        text = f"Today basically turned into {scenario_phrase}, and I am still processing it."
    text = _append_modality_suffix(text, record["content_modality"])
    return _expand_to_length_style(record, text)


def repair_social_post_alignment(record: spec.Record, text: str) -> str:
    return _repair_from_generated_text(record, text)
