"""Sampling and summary helpers for the social media dataset preset."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from dataset_generation.sampling import (
    apply_weight_adjustments,
    copy_weights,
    format_ratio,
    weighted_choice,
)
from dataset_generation.schema import validate_record_against_schema

from . import spec

_TOPIC_LABEL_TO_VALUE = {
    spec.build_granularity_label(topic_domain): topic_domain
    for topic_domain in spec.TOPIC_DOMAIN_VALUES
}


def _allowed(schema: dict[str, dict[str, Any]], field: str, parent_value: str) -> list[str]:
    return list(schema[field]["mapping"][parent_value])


def _allowed_pair(
    schema: dict[str, dict[str, Any]],
    field: str,
    parent_values: tuple[str, str],
) -> list[str]:
    return list(schema[field]["mapping"][parent_values])


def build_label_suites(record: spec.RecordDraft) -> dict[str, dict[str, str]]:
    suites: dict[str, dict[str, str]] = {}
    for family, components_by_level in spec.GRANULARITY_LABEL_COMPONENTS.items():
        suites[family] = {}
        for level, components in components_by_level.items():
            suites[family][level] = spec.build_granularity_label(
                *(record[field] for field in components)
            )
    return suites


def apply_granularity_labels(record: spec.RecordDraft) -> spec.Record:
    completed = dict(record)
    label_suites = build_label_suites(completed)
    for family, fields in spec.GRANULARITY_LABEL_FIELDS.items():
        for level, field_name in fields.items():
            completed[field_name] = label_suites[family][level]
    return completed


def audit_schema_support(schema: dict[str, Any]) -> dict[str, list[str]]:
    if schema is not spec.SCHEMA:
        raise ValueError("audit_schema_support expects the fixed SCHEMA defined in this module.")

    report: dict[str, list[str]] = {}
    for topic_domain in spec.TOPIC_DOMAIN_VALUES:
        if topic_domain not in schema["post_act"]["mapping"]:
            report.setdefault("missing_post_act_mapping", []).append(topic_domain)
            continue
        for post_act in schema["post_act"]["mapping"][topic_domain]:
            if (topic_domain, post_act) not in schema["content_scenario"]["mapping"]:
                report.setdefault("missing_content_scenario_mapping", []).append(
                    f"{topic_domain} -> {post_act}"
                )
    for platform in spec.PLATFORM_VALUES:
        for field in ("audience_targeting", "formality_style", "content_modality"):
            if platform not in schema[field]["mapping"]:
                report.setdefault(f"missing_{field}_mapping", []).append(platform)

    for family, fields in spec.GRANULARITY_LABEL_FIELDS.items():
        for level, field_name in fields.items():
            declared = set(schema[field_name]["values"])
            recomputed = {
                spec.build_granularity_label(*(record_value for record_value in parts))
                for parts in _iter_granularity_parts(family, level)
            }
            if declared != recomputed:
                report.setdefault("granularity_label_mismatch", []).append(f"{family}:{level}")
    return report


def _iter_granularity_parts(family: str, level: str) -> list[tuple[str, ...]]:
    if family == "content":
        if level == "coarse":
            return [(topic_domain,) for topic_domain in spec.TOPIC_DOMAIN_VALUES]
        if level == "normal":
            return [
                (topic_domain, post_act)
                for topic_domain in spec.TOPIC_DOMAIN_VALUES
                for post_act in spec.TOPIC_POST_ACT_MAPPING[topic_domain]
            ]
        return [
            (topic_domain, post_act, content_scenario)
            for topic_domain in spec.TOPIC_DOMAIN_VALUES
            for post_act in spec.TOPIC_POST_ACT_MAPPING[topic_domain]
            for content_scenario in spec.CONTENT_SCENARIO_MAPPING[(topic_domain, post_act)]
        ]
    if level == "coarse":
        return [(platform,) for platform in spec.PLATFORM_VALUES]
    if level == "normal":
        return [
            (platform, audience)
            for platform in spec.PLATFORM_VALUES
            for audience in spec.PLATFORM_AUDIENCE_MAPPING[platform]
        ]
    return [
        (platform, audience, content_modality)
        for platform in spec.PLATFORM_VALUES
        for audience in spec.PLATFORM_AUDIENCE_MAPPING[platform]
        for content_modality in spec.PLATFORM_MODALITY_MAPPING[platform]
    ]


def _topic_target_order() -> list[str]:
    return [spec.build_granularity_label(topic_domain) for topic_domain in spec.TOPIC_DOMAIN_VALUES]


def reachable_topic_targets() -> list[str]:
    return _topic_target_order()


def _build_balanced_topic_plan(
    n: int,
    rng: random.Random,
    *,
    min_coverage_count: int | None,
) -> list[str]:
    topics = _topic_target_order()
    if min_coverage_count is not None and min_coverage_count < 0:
        raise ValueError("min_coverage_count must be non-negative.")
    required = len(topics) * (min_coverage_count or 0)
    if required > n:
        raise ValueError(
            "rows is too small to satisfy the requested min_coverage_count "
            "across all topic domains."
        )

    plan: list[str] = []
    if min_coverage_count:
        for _ in range(min_coverage_count):
            shuffled = list(topics)
            rng.shuffle(shuffled)
            plan.extend(shuffled)
    elif n >= len(topics):
        shuffled = list(topics)
        rng.shuffle(shuffled)
        plan.extend(shuffled)

    while len(plan) < n:
        shuffled = list(topics)
        rng.shuffle(shuffled)
        for topic in shuffled:
            if len(plan) >= n:
                break
            plan.append(topic)
    return plan[:n]


def _sample_topic_target(rng: random.Random) -> str:
    topic_domain = weighted_choice(rng, dict(spec.TOPIC_DOMAIN_WEIGHTS))
    return spec.build_granularity_label(topic_domain)


def _sample_platform(topic_domain: str, post_act: str, rng: random.Random) -> str:
    weights = dict(spec.PLATFORM_WEIGHTS)
    if topic_domain == "politics":
        weights = apply_weight_adjustments(
            weights,
            {"x_like": 0.25, "facebook_like": 0.15, "linkedin_like": -0.2},
        )
    elif topic_domain == "business_startups":
        weights = apply_weight_adjustments(
            weights,
            {"linkedin_like": 0.35, "x_like": 0.1, "facebook_like": -0.15},
        )
    elif topic_domain in {"entertainment_pop_culture", "sports"}:
        weights = apply_weight_adjustments(
            weights,
            {"x_like": 0.1, "instagram_caption": 0.2, "youtube_comment": 0.1},
        )
    elif topic_domain == "personal_life_diary":
        weights = apply_weight_adjustments(
            weights,
            {"instagram_caption": 0.2, "facebook_like": 0.15, "linkedin_like": -0.3},
        )

    if post_act == "promote_advertise":
        weights = apply_weight_adjustments(
            weights,
            {"linkedin_like": 0.25, "instagram_caption": 0.2, "youtube_comment": -0.2},
        )
    elif post_act == "debate_argue":
        weights = apply_weight_adjustments(
            weights,
            {"reddit_like": 0.2, "x_like": 0.2, "instagram_caption": -0.2},
        )
    elif post_act == "narrate_personal_experience":
        weights = apply_weight_adjustments(
            weights,
            {"instagram_caption": 0.2, "facebook_like": 0.15, "linkedin_like": -0.15},
        )

    return weighted_choice(rng, weights)


def _sample_audience(platform: str, topic_domain: str, rng: random.Random) -> str:
    weights = copy_weights(_allowed(spec.SCHEMA, "audience_targeting", platform), 1.0)
    if topic_domain == "politics" and "political_group" in weights:
        weights["political_group"] += 0.5
    if topic_domain in {"sports", "entertainment_pop_culture"} and "fans_fandom" in weights:
        weights["fans_fandom"] += 0.5
    if topic_domain == "business_startups" and "industry_peers" in weights:
        weights["industry_peers"] += 0.5
    if topic_domain == "personal_life_diary" and "friends_followers" in weights:
        weights["friends_followers"] += 0.5
    return weighted_choice(rng, weights)


def _sample_formality(platform: str, topic_domain: str, post_act: str, rng: random.Random) -> str:
    weights = copy_weights(_allowed(spec.SCHEMA, "formality_style", platform), 1.0)
    if topic_domain == "business_startups" and "professional_pr_tone" in weights:
        weights["professional_pr_tone"] += 0.35
    if post_act in {"inform_share_news", "debate_argue"} and "informational_neutral" in weights:
        weights["informational_neutral"] += 0.25
    if post_act == "promote_advertise" and "professional_pr_tone" in weights:
        weights["professional_pr_tone"] += 0.35
    if post_act == "entertain_joke":
        for field in ("slang_heavy", "meme_style", "casual_conversational"):
            if field in weights:
                weights[field] += 0.25
    if post_act == "vent_rant" and "very_formal" in weights:
        weights["very_formal"] = max(0.0, weights["very_formal"] - 0.5)
    return weighted_choice(rng, weights)


def _sample_post_act(topic_domain: str, rng: random.Random) -> str:
    weights = copy_weights(_allowed(spec.SCHEMA, "post_act", topic_domain), 1.0)
    if topic_domain == "politics":
        weights = apply_weight_adjustments(
            weights,
            {"express_opinion": 0.3, "mobilize_call_to_action": 0.2, "debate_argue": 0.15},
        )
    if topic_domain == "technology":
        weights = apply_weight_adjustments(
            weights,
            {"inform_share_news": 0.2, "express_opinion": 0.2, "ask_question": 0.15},
        )
    if topic_domain == "personal_life_diary":
        weights = apply_weight_adjustments(
            weights,
            {"narrate_personal_experience": 0.35, "vent_rant": 0.15},
        )
    return weighted_choice(rng, weights)


def _sample_content_scenario(topic_domain: str, post_act: str, rng: random.Random) -> str:
    weights = copy_weights(
        _allowed_pair(spec.SCHEMA, "content_scenario", (topic_domain, post_act)), 1.0
    )
    return weighted_choice(rng, weights)


def _sample_stance(post_act: str, topic_domain: str, rng: random.Random) -> str:
    weights = copy_weights(_allowed(spec.SCHEMA, "stance", post_act), 1.0)
    if post_act == "promote_advertise" and "promotional" in weights:
        weights["promotional"] += 0.7
    if post_act == "vent_rant" and "critical" in weights:
        weights["critical"] += 0.6
    if post_act == "mobilize_call_to_action" and topic_domain in {
        "politics",
        "environment_climate",
    }:
        if "supportive" in weights:
            weights["supportive"] += 0.25
        if "critical" in weights:
            weights["critical"] += 0.25
    return weighted_choice(rng, weights)


def _sample_sentiment(post_act: str, stance: str, rng: random.Random) -> str:
    weights = {
        "very_negative": 0.05,
        "negative": 0.15,
        "neutral": 0.2,
        "mixed": 0.15,
        "positive": 0.2,
        "very_positive": 0.05,
    }
    if stance == "critical":
        weights = apply_weight_adjustments(
            weights,
            {"very_negative": 0.25, "negative": 0.5, "positive": -0.15, "very_positive": -0.05},
        )
    elif stance == "supportive":
        weights = apply_weight_adjustments(
            weights,
            {"positive": 0.45, "very_positive": 0.2, "negative": -0.15},
        )
    elif stance == "promotional":
        weights = apply_weight_adjustments(
            weights,
            {"positive": 0.55, "very_positive": 0.3, "negative": -0.2, "very_negative": -0.05},
        )
    elif stance == "mixed":
        weights = apply_weight_adjustments(weights, {"mixed": 0.45})

    if post_act == "vent_rant":
        weights = apply_weight_adjustments(
            weights, {"very_negative": 0.2, "negative": 0.35, "positive": -0.2}
        )
    if post_act == "entertain_joke":
        weights = apply_weight_adjustments(weights, {"mixed": 0.2, "positive": 0.1})
    return weighted_choice(rng, weights)


def _sample_voice_tone(post_act: str, stance: str, sentiment: str, rng: random.Random) -> str:
    weights = copy_weights(list(spec.SCHEMA["voice_tone"]["values"].copy()), 1.0)
    if post_act == "inform_share_news":
        for tone in ("informative", "neutral"):
            weights[tone] += 0.35
    if post_act == "vent_rant":
        for tone in ("frustrated", "angry", "sarcastic"):
            weights[tone] += 0.4
    if post_act == "ask_question":
        for tone in ("skeptical", "neutral", "reflective"):
            weights[tone] += 0.25
    if post_act == "promote_advertise":
        for tone in ("optimistic", "informative"):
            weights[tone] += 0.35
    if stance == "supportive":
        weights["hopeful"] += 0.2
    if sentiment in {"very_negative", "negative"}:
        weights["frustrated"] += 0.25
        weights["angry"] += 0.2
    if sentiment in {"positive", "very_positive"}:
        weights["optimistic"] += 0.3
        weights["hopeful"] += 0.15
    return weighted_choice(rng, weights)


def _sample_certainty(post_act: str, topic_domain: str, rng: random.Random) -> str:
    weights = copy_weights(list(spec.SCHEMA["certainty_confidence"]["values"].copy()), 1.0)
    if post_act == "ask_question":
        for value in ("speculative", "uncertain", "mixed_certainty"):
            weights[value] += 0.35
    if post_act in {"inform_share_news", "debate_argue"}:
        for value in ("confident", "data_backed"):
            weights[value] += 0.25
    if topic_domain in {"politics", "economy_finance", "environment_climate"}:
        weights["data_backed"] += 0.1
    return weighted_choice(rng, weights)


def _sample_toxicity(stance: str, voice_tone: str, rng: random.Random) -> str:
    weights = {
        "highly_toxic_abusive": 0.02,
        "aggressive": 0.05,
        "critical_but_civil": 0.2,
        "neutral": 0.28,
        "polite": 0.15,
        "supportive": 0.15,
        "encouraging": 0.1,
    }
    if stance == "critical":
        weights = apply_weight_adjustments(
            weights, {"critical_but_civil": 0.35, "aggressive": 0.15, "supportive": -0.1}
        )
    if voice_tone in {"angry", "sarcastic", "frustrated"}:
        weights = apply_weight_adjustments(
            weights, {"aggressive": 0.12, "critical_but_civil": 0.15}
        )
    if stance in {"supportive", "promotional"}:
        weights = apply_weight_adjustments(
            weights, {"supportive": 0.2, "encouraging": 0.15, "aggressive": -0.04}
        )
    return weighted_choice(rng, weights)


def _sample_modality(platform: str, post_act: str, rng: random.Random) -> str:
    weights = copy_weights(_allowed(spec.SCHEMA, "content_modality", platform), 1.0)
    if post_act == "inform_share_news":
        for value in ("text_plus_link", "text_plus_image", "quote_post"):
            if value in weights:
                weights[value] += 0.25
    if post_act == "ask_question" and "poll" in weights:
        weights["poll"] += 0.2
    if post_act == "debate_argue" and "thread_starter" in weights:
        weights["thread_starter"] += 0.35
    if post_act == "narrate_personal_experience":
        for value in ("pure_text", "text_plus_image"):
            if value in weights:
                weights[value] += 0.25
    if post_act == "promote_advertise":
        for value in ("text_plus_link", "text_plus_image", "text_plus_video"):
            if value in weights:
                weights[value] += 0.3
    return weighted_choice(rng, weights)


def _sample_temporal_orientation(post_act: str, rng: random.Random) -> str:
    weights = {
        "past_reflection": 0.12,
        "present_commentary": 0.34,
        "near_future_prediction": 0.12,
        "long_term_prediction": 0.06,
        "timeless_general_statement": 0.12,
        "real_time_reaction": 0.24,
    }
    if post_act == "inform_share_news":
        weights = apply_weight_adjustments(
            weights, {"present_commentary": 0.15, "real_time_reaction": 0.15}
        )
    if post_act == "narrate_personal_experience":
        weights = apply_weight_adjustments(
            weights, {"past_reflection": 0.2, "present_commentary": 0.1}
        )
    if post_act == "debate_argue":
        weights = apply_weight_adjustments(
            weights, {"timeless_general_statement": 0.15, "long_term_prediction": 0.12}
        )
    if post_act == "mobilize_call_to_action":
        weights = apply_weight_adjustments(
            weights, {"near_future_prediction": 0.18, "present_commentary": 0.1}
        )
    return weighted_choice(rng, weights)


def _sample_length_style(
    topic_domain: str,
    post_act: str,
    content_modality: str,
    difficulty: str,
    rng: random.Random,
) -> str:
    weights = copy_weights(list(spec.LENGTH_STYLE_VALUES), 1.0)
    if difficulty == "easy":
        weights = apply_weight_adjustments(
            weights,
            {"very_short": 0.2, "short": 0.2, "long": -0.1},
        )
    elif difficulty == "hard":
        weights = apply_weight_adjustments(
            weights,
            {"medium": 0.1, "long": 0.15, "threadish": 0.2},
        )

    if content_modality == "thread_starter":
        weights = apply_weight_adjustments(
            weights, {"threadish": 0.8, "long": 0.2, "very_short": -0.3}
        )
    elif content_modality == "reply":
        weights = apply_weight_adjustments(
            weights, {"very_short": 0.25, "short": 0.25, "threadish": -0.3}
        )
    elif content_modality in {"text_plus_link", "quote_post"}:
        weights = apply_weight_adjustments(weights, {"medium": 0.15, "long": 0.1})

    if post_act == "ask_question":
        weights = apply_weight_adjustments(
            weights,
            {"very_short": 0.25, "short": 0.2, "threadish": -0.2},
        )
    elif post_act == "inform_share_news":
        weights = apply_weight_adjustments(weights, {"medium": 0.15, "long": 0.15})
    elif post_act == "debate_argue":
        weights = apply_weight_adjustments(weights, {"long": 0.2, "threadish": 0.25})
    elif post_act == "narrate_personal_experience":
        weights = apply_weight_adjustments(weights, {"medium": 0.2, "long": 0.15})
    elif post_act == "entertain_joke":
        weights = apply_weight_adjustments(weights, {"very_short": 0.15, "short": 0.15})

    if topic_domain == "personal_life_diary":
        weights = apply_weight_adjustments(weights, {"medium": 0.15, "long": 0.1})
    elif topic_domain == "business_startups":
        weights = apply_weight_adjustments(weights, {"medium": 0.15, "long": 0.1})

    return weighted_choice(rng, weights)


def sample_record_for_topic_target(
    topic_target: str,
    rng: random.Random,
    *,
    difficulty: str,
) -> spec.Record:
    _ = difficulty
    topic_domain = _TOPIC_LABEL_TO_VALUE.get(topic_target)
    if topic_domain is None:
        raise ValueError(f"Unknown topic coverage target: {topic_target}")

    post_act = _sample_post_act(topic_domain, rng)
    content_scenario = _sample_content_scenario(topic_domain, post_act, rng)
    stance = _sample_stance(post_act, topic_domain, rng)
    platform = _sample_platform(topic_domain, post_act, rng)
    audience_targeting = _sample_audience(platform, topic_domain, rng)
    formality_style = _sample_formality(platform, topic_domain, post_act, rng)
    sentiment_polarity = _sample_sentiment(post_act, stance, rng)
    voice_tone = _sample_voice_tone(post_act, stance, sentiment_polarity, rng)
    certainty_confidence = _sample_certainty(post_act, topic_domain, rng)
    toxicity_civility = _sample_toxicity(stance, voice_tone, rng)
    content_modality = _sample_modality(platform, post_act, rng)
    temporal_orientation = _sample_temporal_orientation(post_act, rng)
    length_style = _sample_length_style(
        topic_domain,
        post_act,
        content_modality,
        difficulty,
        rng,
    )

    record = apply_granularity_labels(
        {
            "platform": platform,
            "topic_domain": topic_domain,
            "post_act": post_act,
            "content_scenario": content_scenario,
            "length_style": length_style,
            "stance": stance,
            "sentiment_polarity": sentiment_polarity,
            "voice_tone": voice_tone,
            "audience_targeting": audience_targeting,
            "formality_style": formality_style,
            "certainty_confidence": certainty_confidence,
            "toxicity_civility": toxicity_civility,
            "content_modality": content_modality,
            "temporal_orientation": temporal_orientation,
        }
    )
    validate_record(record)
    return record


def generate_records(
    n: int,
    rng: random.Random | None = None,
    *,
    mode: str,
    difficulty: str,
    min_coverage_count: int | None,
) -> list[spec.Record]:
    prng = rng or random.Random()
    if n < 0:
        raise ValueError("n must be non-negative")
    if mode not in {"balanced", "random"}:
        raise ValueError(f"Unsupported social_media generation mode: {mode}")

    if mode == "balanced":
        topic_targets = _build_balanced_topic_plan(
            n,
            prng,
            min_coverage_count=min_coverage_count,
        )
    else:
        topic_targets = [_sample_topic_target(prng) for _ in range(n)]

    return [
        sample_record_for_topic_target(topic_target, prng, difficulty=difficulty)
        for topic_target in topic_targets
    ]


def generate_records_for_topic_targets(
    topic_targets: list[str],
    rng: random.Random | None = None,
    *,
    difficulty: str,
) -> list[spec.Record]:
    prng = rng or random.Random()
    return [
        sample_record_for_topic_target(topic_target, prng, difficulty=difficulty)
        for topic_target in topic_targets
    ]


def validate_record(record: spec.Record) -> None:
    validate_record_against_schema(spec.SCHEMA, record)


def build_dataset_summary(
    rows: list[dict[str, str]],
    *,
    mode: str,
    difficulty: str,
    schema_support_report: dict[str, list[str]],
    entity_mention_probability: float | None = None,
    openrouter_model: str | None = None,
) -> str:
    topic_counts = Counter(row["topic_domain"] for row in rows)
    platform_counts = Counter(row["platform"] for row in rows)
    post_act_counts = Counter(row["post_act"] for row in rows)
    modality_counts = Counter(row["content_modality"] for row in rows)
    length_style_counts = Counter(row["length_style"] for row in rows)
    content_fine_counts = Counter(row["content_label_fine"] for row in rows)
    platform_modality_counts = Counter(
        f"{row['platform']} x {row['content_modality']}" for row in rows
    )

    lines = [
        "Social Media Dataset Summary",
        f"Rows: {len(rows)}",
        f"Mode: {mode}",
        f"Difficulty: {difficulty}",
        f"OpenRouter model: {openrouter_model or 'n/a'}",
        (
            f"Entity mention probability: {entity_mention_probability:.2f}"
            if entity_mention_probability is not None
            else "Entity mention probability: preset-driven"
        ),
        "",
        "Topic domain distribution:",
    ]
    for topic_domain in spec.TOPIC_DOMAIN_VALUES:
        count = topic_counts[topic_domain]
        lines.append(f"  {topic_domain}: {count} ({format_ratio(count, len(rows))})")

    lines.extend(["", "Platform distribution:"])
    for platform in spec.PLATFORM_VALUES:
        count = platform_counts[platform]
        lines.append(f"  {platform}: {count} ({format_ratio(count, len(rows))})")

    lines.extend(["", "Post-act distribution:"])
    for post_act, count in sorted(post_act_counts.items()):
        lines.append(f"  {post_act}: {count} ({format_ratio(count, len(rows))})")

    lines.extend(["", "Modality distribution:"])
    for modality, count in sorted(modality_counts.items()):
        lines.append(f"  {modality}: {count} ({format_ratio(count, len(rows))})")

    lines.extend(["", "Length-style distribution:"])
    for length_style in spec.LENGTH_STYLE_VALUES:
        count = length_style_counts[length_style]
        lines.append(f"  {length_style}: {count} ({format_ratio(count, len(rows))})")

    lines.extend(["", "Top fine-grained content labels:"])
    for label, count in content_fine_counts.most_common(12):
        lines.append(f"  {label}: {count}")

    lines.extend(["", "Platform x modality:"])
    for label, count in platform_modality_counts.most_common(12):
        lines.append(f"  {label}: {count}")

    lines.extend(["", "Schema support audit:"])
    if not schema_support_report:
        lines.append("  no unsupported declared labels found")
    else:
        for issue_name, values in sorted(schema_support_report.items()):
            lines.append(f"  {issue_name}:")
            for value in values:
                lines.append(f"    - {value}")
    return "\n".join(lines)
