"""Prompt construction and alignment checks for the restaurant review preset."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from dataset_generation.llm import generated_text_response_schema
from dataset_generation.sampling import pick_one
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
LENGTH_STYLE_OPTIONS = tuple(guidance.LENGTH_STYLE_GUIDANCE)
DETAIL_STYLE_OPTIONS = tuple(guidance.DETAIL_STYLE_GUIDANCE)
CLOSING_STYLE_OPTIONS = tuple(guidance.CLOSING_STYLE_GUIDANCE)

_POSITIVE_CUE_WORDS = (
    "love",
    "amazing",
    "excellent",
    "fantastic",
    "perfect",
    "great",
)
_NEGATIVE_CUE_WORDS = (
    "awful",
    "terrible",
    "disappointing",
    "worst",
    "bland",
    "rude",
    "dirty",
)


@dataclass(frozen=True)
class AlignmentAssessment:
    scenario_match: bool
    length_match: bool
    banned_scaffold: str
    extreme_rating_contradiction: bool


def _normalize_text(value: str) -> str:
    return normalize_text(value)


def _scenario_tokens(review_scenario: str) -> tuple[str, ...]:
    override_terms = spec.REVIEW_SCENARIO_HINT_OVERRIDES.get(review_scenario, ())
    terms: list[str] = []
    for token in review_scenario.split("_"):
        normalized = token.lower().strip()
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
    return humanize_slug(value)


def preferred_review_scenario_phrase(review_scenario: str) -> str:
    return _humanize_slug(review_scenario).replace("  ", " ").strip()


def build_surface_metadata(record: spec.Record) -> dict[str, str]:
    return build_shared_surface_metadata(record, spec.SURFACE_METADATA_FIELDS)


def _length_word_bounds(length_style: str) -> tuple[int, int]:
    return {
        "short": (4, 24),
        "medium": (18, 50),
        "detailed": (40, 100),
    }[length_style]


def _length_target_guidance(length_style: str) -> str:
    minimum, maximum = _length_word_bounds(length_style)
    return f"target roughly {minimum}-{maximum} words so the review clearly reads as {length_style}"


def build_cluster_labels(record: spec.Record) -> dict[str, str]:
    return {
        "review_coarse": record["review_label_coarse"],
        "review_normal": record["review_label_normal"],
        "review_fine": record["review_label_fine"],
    }


def _rating_score_guidance(rating_score: str) -> str:
    score = int(rating_score)
    if score <= 2:
        return "overall tone should feel clearly bad enough that the reviewer would not return"
    if score <= 4:
        return "overall tone should feel mostly negative with a concrete complaint"
    if score <= 6:
        return "overall tone should feel mixed rather than fully positive or fully negative"
    if score <= 8:
        return "overall tone should feel clearly positive but still realistic"
    return "overall tone should feel excellent and easy to recommend"


def sample_text_variation_profile(
    record: spec.Record,
    rng: random.Random,
    difficulty: str,
    *,
    entity_mention_probability: float | None,
) -> spec.TextVariationProfile:
    _ = entity_mention_probability
    opening_style = pick_one(rng, list(OPENING_STYLE_OPTIONS))
    if record["rating_score"] in {"1", "2", "9", "10"}:
        opening_style = pick_one(rng, ["verdict_first", "detail_first"])

    length_style = record["length_style"]

    detail_style = pick_one(rng, list(DETAIL_STYLE_OPTIONS))
    if record["sentiment_polarity"] == "mixed":
        detail_style = pick_one(rng, ["balanced_note", "specific_example"])
    elif record["rating_score"] in {"1", "10"}:
        detail_style = pick_one(rng, ["specific_example", "single_observation"])

    closing_style = pick_one(rng, list(CLOSING_STYLE_OPTIONS))
    if int(record["rating_score"]) >= 8:
        closing_style = pick_one(rng, ["recommend", "recommend", "none"])
    elif int(record["rating_score"]) <= 3:
        closing_style = pick_one(rng, ["skip_returning", "conditional", "none"])
    elif record["sentiment_polarity"] == "mixed":
        closing_style = pick_one(rng, ["conditional", "none"])

    return spec.TextVariationProfile(
        opening_style=opening_style,
        length_style=length_style,
        detail_style=detail_style,
        closing_style=closing_style,
    )


def build_hidden_generation_guidance(
    record: spec.Record,
    variation_profile: spec.TextVariationProfile,
) -> list[str]:
    return [
        guidance.PRIMARY_FOCUS_GUIDANCE[record["primary_focus"]],
        guidance.SENTIMENT_GUIDANCE[record["sentiment_polarity"]],
        guidance.VISIT_MODE_GUIDANCE[record["visit_mode"]],
        _rating_score_guidance(record["rating_score"]),
        guidance.OPENING_STYLE_GUIDANCE[variation_profile.opening_style],
        guidance.LENGTH_STYLE_GUIDANCE[variation_profile.length_style],
        _length_target_guidance(variation_profile.length_style),
        guidance.DETAIL_STYLE_GUIDANCE[variation_profile.detail_style],
        guidance.CLOSING_STYLE_GUIDANCE[variation_profile.closing_style],
    ]


def build_generation_controls(variation_profile: spec.TextVariationProfile) -> dict[str, str]:
    return {
        "opening_style": variation_profile.opening_style,
        "length_style": variation_profile.length_style,
        "target_word_range": "-".join(
            str(bound) for bound in _length_word_bounds(variation_profile.length_style)
        ),
        "detail_style": variation_profile.detail_style,
        "closing_style": variation_profile.closing_style,
    }


def build_openrouter_batch_prompt(
    batch_rows: list[dict[str, Any]],
    difficulty: str,
) -> str:
    schema_for_prompt = prompt_safe_schema(spec.SCHEMA, spec.SURFACE_METADATA_FIELDS)
    repetition_rules = "\n- ".join(guidance.REPETITION_AVOIDANCE_GUIDANCE)
    return (
        "Generate text values for the restaurant review rows below.\n"
        "The surface_metadata block is authoritative ground truth for the review.\n"
        "Treat primary_focus as the broad aspect being judged and review_scenario as the concrete "
        "leaf detail that should be obvious in the review.\n"
        "The cluster_labels block contains derived coarse/normal/fine labels for downstream "
        "clustering; do not copy those labels into the review.\n"
        "The hidden_generation_guidance list gives private realism hints. Translate those hints "
        "into natural review language rather than copying them.\n"
        "The generation_controls block is binding for opening shape, length, detail density, and "
        "closing style.\n"
        "Write only the natural-language review text for each row_id.\n"
        "Do not copy snake_case labels verbatim.\n"
        "Reflect the numeric rating naturally, but mentioning the exact score is optional.\n"
        "Avoid stale openers such as: "
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
        "Evaluate each restaurant-review row for metadata alignment and realism.\n"
        "The most important fields are rating_score, primary_focus, and review_scenario.\n"
        "Do not require the numeric score to appear literally in the text.\n"
        "A row can still pass if the overall tone clearly fits the score band and the review "
        "naturally expresses the scenario.\n"
        "Ratings:\n"
        "  pass: metadata and review align clearly and the review sounds realistic.\n"
        "  need_attention: mostly plausible but vague, generic, or only partly aligned.\n"
        "  not_pass: the overall tone clearly conflicts with the score or the scenario is badly "
        "missed.\n"
        "Return JSON only, matching this schema exactly:\n"
        f"{json.dumps(evaluation_response_schema(), indent=2)}\n\n"
        "Rows:\n"
        f"{json.dumps(batch_rows, indent=2)}"
    )


def _find_banned_scaffold(text: str) -> str:
    normalized = _normalize_text(text).lower()
    for scaffold in guidance.BANNED_TEMPLATE_SCAFFOLDS:
        if normalized.startswith(scaffold):
            return scaffold
    return ""


def _has_extreme_rating_contradiction(rating_score: str, text: str) -> bool:
    normalized = text.lower()
    score = int(rating_score)
    if score <= 2:
        return any(token in normalized for token in _POSITIVE_CUE_WORDS)
    if score >= 9:
        return any(token in normalized for token in _NEGATIVE_CUE_WORDS)
    return False


def assess_review_alignment(record: spec.Record, text: str) -> AlignmentAssessment:
    normalized = _normalize_text(text).lower()
    scenario_terms = _scenario_tokens(record["review_scenario"])
    scenario_match = any(term in normalized for term in scenario_terms)
    word_count = len(_normalize_text(text).split())
    minimum_words, maximum_words = _length_word_bounds(record["length_style"])
    return AlignmentAssessment(
        scenario_match=scenario_match,
        length_match=minimum_words <= word_count <= maximum_words,
        banned_scaffold=_find_banned_scaffold(text),
        extreme_rating_contradiction=_has_extreme_rating_contradiction(
            record["rating_score"],
            normalized,
        ),
    )


def evaluate_restaurant_review_deterministically(
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
                "length_match": "false",
                "banned_scaffold": "",
                "extreme_rating_contradiction": "false",
            },
        )

    assessment = assess_review_alignment(record, normalized_text)
    rating: EvaluationRating = "pass"
    reasons: list[str] = []
    if not assessment.scenario_match:
        rating = "need_attention"
        reasons.append("review_scenario_mismatch")
    if not assessment.length_match:
        rating = "need_attention"
        reasons.append("length_style_mismatch")
    if assessment.banned_scaffold:
        rating = "need_attention"
        reasons.append(f"banned_template_scaffold:{assessment.banned_scaffold}")
    if assessment.extreme_rating_contradiction:
        rating = "need_attention"
        reasons.append("extreme_rating_contradiction")
    if len(normalized_text.split()) < 4:
        rating = "need_attention"
        reasons.append("very_short_text")

    return DeterministicEvaluationResult(
        rating=rating,
        reasons=tuple(reasons),
        details={
            "scenario_match": str(assessment.scenario_match).lower(),
            "length_match": str(assessment.length_match).lower(),
            "banned_scaffold": assessment.banned_scaffold,
            "extreme_rating_contradiction": str(assessment.extreme_rating_contradiction).lower(),
        },
    )


def collect_alignment_failures(
    records: list[spec.Record],
    parsed: dict[int, str],
) -> tuple[dict[int, ValueError], dict[int, ValueError]]:
    hard_failures: dict[int, ValueError] = {}
    soft_failures: dict[int, ValueError] = {}
    for row_id, text in parsed.items():
        record = records[row_id]
        assessment = assess_review_alignment(record, text)
        if len(_normalize_text(text).split()) < 3:
            soft_failures[row_id] = ValueError("Generated review was too short to be useful.")
            continue
        banned_scaffold = _find_banned_scaffold(text)
        if banned_scaffold:
            soft_failures[row_id] = ValueError(
                "Generated review opened with a discouraged template scaffold "
                f"('{banned_scaffold}')."
            )
            continue
        if not assessment.scenario_match:
            soft_failures[row_id] = ValueError(
                "Generated review matched the broad restaurant-review framing but did not "
                "make the sampled leaf scenario explicit for "
                f"review_scenario={record['review_scenario']!r}."
            )
            continue
        if not assessment.length_match:
            soft_failures[row_id] = ValueError(
                "Generated review did not match the requested length_style strongly enough."
            )
            continue
        if _has_extreme_rating_contradiction(record["rating_score"], text):
            soft_failures[row_id] = ValueError(
                "Generated review used language that strongly contradicts the rating score."
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
            f"Row {row_id}: keep the same metadata but rewrite the review so it sounds like a "
            f"natural {record['rating_score']}/10 experience centered on "
            f"'{record['review_scenario']}'.\n"
            f"Rewrite reason: {failure}\n"
            f"Failed text: {parsed[row_id]}"
        )

    return build_shared_alignment_retry_prefix(
        messages,
        intro_lines=(
            "The previous response violated alignment or quality constraints for some rows.",
            "Retry only the rows listed below and return valid JSON only.",
            "Keep every row_id exactly the same.",
        ),
    )


def _append_scenario_phrase(record: spec.Record, text: str) -> str:
    if assess_review_alignment(record, text).scenario_match:
        return text
    scenario_phrase = preferred_review_scenario_phrase(record["review_scenario"])
    return f"{text} The part that stood out most was {scenario_phrase}."


def _expand_to_length_style(record: spec.Record, text: str) -> str:
    minimum_words, _ = _length_word_bounds(record["length_style"])
    expanded = _normalize_text(text)
    scenario_phrase = preferred_review_scenario_phrase(record["review_scenario"])
    padding_sentences = [
        f"The most memorable detail was the {scenario_phrase}.",
        "That one part did most of the work in shaping the whole experience for me.",
        "It is the main reason I would describe this visit the way I am describing it now.",
    ]
    for sentence in padding_sentences:
        if len(expanded.split()) >= minimum_words:
            break
        expanded = f"{expanded} {sentence}"
    return expanded


def build_aligned_fallback_review(record: spec.Record) -> str:
    score = int(record["rating_score"])
    scenario_phrase = preferred_review_scenario_phrase(record["review_scenario"])
    if score >= 9:
        text = f"{score}/10. The {scenario_phrase} really stood out, and I would gladly come back."
    elif score >= 7:
        text = f"{score}/10. Pretty happy overall, especially with the {scenario_phrase}."
    elif score >= 5:
        text = f"{score}/10. Mixed experience here because the {scenario_phrase} stood out most."
    elif score >= 3:
        text = f"{score}/10. The {scenario_phrase} made this harder to enjoy than it should have."
    else:
        text = f"{score}/10. The {scenario_phrase} was rough enough that I would not return."
    return _expand_to_length_style(record, text)


def repair_restaurant_review_alignment(record: spec.Record, text: str) -> str:
    repaired = _normalize_text(text)
    if not repaired:
        return build_aligned_fallback_review(record)

    banned_scaffold = _find_banned_scaffold(repaired)
    if banned_scaffold:
        repaired = repaired[len(banned_scaffold) :].lstrip(" ,:-")
        repaired = _normalize_text(repaired)
    if not repaired:
        return build_aligned_fallback_review(record)

    repaired = _append_scenario_phrase(record, repaired)
    if len(repaired.split()) < 4:
        repaired = f"{repaired} Overall it landed around {record['rating_score']}/10 for me."
    repaired = _expand_to_length_style(record, repaired)

    if _has_extreme_rating_contradiction(record["rating_score"], repaired):
        return build_aligned_fallback_review(record)
    if not assess_review_alignment(record, repaired).length_match:
        return build_aligned_fallback_review(record)
    return repaired
