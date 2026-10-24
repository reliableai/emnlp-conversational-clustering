"""Enterprise-ticket text-variation, prompt, and alignment-repair helpers."""

from __future__ import annotations

import json
import random
from typing import Any

from dataset_generation.llm import generated_text_response_schema
from dataset_generation.sampling import clamp_probability, validate_probability
from dataset_generation.types import DeterministicEvaluationResult, EvaluationRating
from dataset_generation.utils import (
    build_alignment_retry_prefix as build_shared_alignment_retry_prefix,
)
from dataset_generation.utils import (
    build_surface_metadata as build_shared_surface_metadata,
)
from dataset_generation.utils import (
    normalize_text,
)

from .alignment import (
    AlignmentAssessment,
    assess_ticket_text_alignment,
    build_alignment_terms,
    preferred_sub_intent_phrase,
)
from .guidance import (
    APPROVAL_GUIDANCE,
    COMPLETENESS_GUIDANCE,
    DEPENDENCY_GUIDANCE,
    DEPENDENCY_GUIDANCE_PROBABILITY,
    EMOTION_GUIDANCE,
    IMPACT_GUIDANCE,
    IMPACT_GUIDANCE_PROBABILITY,
    INTENT_GUIDANCE,
    INTERACTION_DIRECTIVE_MAP,
    INTERACTION_GUIDANCE,
    KNOWLEDGE_GUIDANCE,
    SEVERITY_GUIDANCE,
    SEVERITY_GUIDANCE_PROBABILITY,
    STRUCTURE_GUIDANCE,
    TEMPORAL_DIRECTIVE_MAP,
    TEMPORAL_GUIDANCE,
    WORKFLOW_GUIDANCE,
)
from .sampling import apply_weight_adjustments, difficulty_profile, pick_one, weighted_choice
from .spec import (
    BANNED_INTERNAL_PROCESS_PHRASES,
    OPENROUTER_EVALUATION_PROMPT,
    SCHEMA,
    SURFACE_METADATA_FIELDS,
    SYSTEM_SYNC_ISSUES,
    Record,
    TextVariationProfile,
)


def _normalize_text(value: str) -> str:
    return normalize_text(value)


def sample_text_variation_profile(
    record: Record,
    rng: random.Random,
    difficulty: str,
    system_mention_probability: float | None = None,
) -> TextVariationProfile:
    profile = difficulty_profile(difficulty)
    effective_system_probability = validate_probability(
        "system_mention_probability",
        system_mention_probability,
    )
    if effective_system_probability is None:
        effective_system_probability = profile.explicit_system_probability

    message_channel_weights = {
        "portal_form": 0.42,
        "email_note": 0.24,
        "chat_message": 0.34,
    }
    completeness_weights = {"complete": 0.48, "partial": 0.32, "minimal": 0.20}
    intent_weights = {
        "request_fix": 0.32,
        "ask_question": 0.18,
        "ask_status": 0.12,
        "report_issue_only": 0.14,
        "confirm_action": 0.08,
        "escalate": 0.16,
    }
    knowledge_weights = {
        "correct_understanding": 0.40,
        "confused": 0.18,
        "misdiagnosing": 0.14,
        "guessing": 0.16,
        "unsure_if_correct_channel": 0.12,
    }
    emotional_weights = {
        "neutral": 0.26,
        "frustrated": 0.14,
        "confused": 0.18,
        "apologetic": 0.10,
        "urgent": 0.14,
        "passive": 0.08,
        "slightly_annoyed": 0.10,
    }
    temporal_weights = {
        "first_occurrence": 0.30,
        "recurring_issue": 0.18,
        "since_recent_time": 0.18,
        "again_today": 0.12,
        "after_change_or_update": 0.22,
    }
    interaction_weights = {
        "none": 0.30,
        "self_tried": 0.20,
        "redirected_here": 0.10,
        "still_not_fixed": 0.12,
        "following_up": 0.18,
        "asked_for_more_info": 0.10,
    }

    writing_style = record["writing_style"]
    clarity_level = record["clarity_level"]
    workflow_stage = record["workflow_stage"]

    if writing_style == "concise":
        message_channel_weights = apply_weight_adjustments(
            message_channel_weights,
            {"chat_message": 0.10, "portal_form": -0.04},
        )
        completeness_weights = apply_weight_adjustments(
            completeness_weights,
            {"partial": 0.06, "minimal": 0.06},
        )
    elif writing_style == "formal":
        message_channel_weights = apply_weight_adjustments(
            message_channel_weights,
            {"email_note": 0.18},
        )
        completeness_weights = apply_weight_adjustments(
            completeness_weights,
            {"complete": 0.12, "minimal": -0.08},
        )
        emotional_weights = apply_weight_adjustments(
            emotional_weights,
            {"neutral": 0.06, "apologetic": 0.06, "frustrated": -0.06},
        )
    elif writing_style == "informal":
        message_channel_weights = apply_weight_adjustments(
            message_channel_weights,
            {"chat_message": 0.16},
        )
        emotional_weights = apply_weight_adjustments(
            emotional_weights,
            {"slightly_annoyed": 0.06, "neutral": -0.04},
        )
    elif writing_style == "frustrated":
        message_channel_weights = apply_weight_adjustments(
            message_channel_weights,
            {"chat_message": 0.12},
        )
        emotional_weights = apply_weight_adjustments(
            emotional_weights,
            {"frustrated": 0.20, "urgent": 0.08, "passive": -0.04},
        )
        intent_weights = apply_weight_adjustments(
            intent_weights,
            {"escalate": 0.10, "report_issue_only": 0.04},
        )
    elif writing_style == "technical":
        message_channel_weights = apply_weight_adjustments(
            message_channel_weights,
            {"portal_form": 0.12, "email_note": 0.08},
        )
        knowledge_weights = apply_weight_adjustments(
            knowledge_weights,
            {"correct_understanding": 0.12, "guessing": -0.04},
        )
        completeness_weights = apply_weight_adjustments(
            completeness_weights,
            {"complete": 0.08},
        )
    elif writing_style == "non_native_english":
        emotional_weights = apply_weight_adjustments(
            emotional_weights,
            {"apologetic": 0.08, "confused": 0.08},
        )
        knowledge_weights = apply_weight_adjustments(
            knowledge_weights,
            {"guessing": 0.04, "confused": 0.04},
        )
        completeness_weights = apply_weight_adjustments(
            completeness_weights,
            {"partial": 0.06, "minimal": 0.04},
        )

    if clarity_level == "clear":
        completeness_weights = apply_weight_adjustments(
            completeness_weights,
            {"complete": 0.28, "partial": -0.08, "minimal": -0.12},
        )
        knowledge_weights = apply_weight_adjustments(
            knowledge_weights,
            {"correct_understanding": 0.16, "confused": -0.06, "guessing": -0.04},
        )
    elif clarity_level == "moderately_clear":
        completeness_weights = apply_weight_adjustments(
            completeness_weights,
            {"partial": 0.10, "complete": -0.04},
        )
    elif clarity_level == "vague":
        completeness_weights = apply_weight_adjustments(
            completeness_weights,
            {"complete": -0.24, "partial": 0.06, "minimal": 0.28},
        )
        knowledge_weights = apply_weight_adjustments(
            knowledge_weights,
            {
                "correct_understanding": -0.14,
                "confused": 0.10,
                "guessing": 0.12,
                "unsure_if_correct_channel": 0.08,
            },
        )
        intent_weights = apply_weight_adjustments(
            intent_weights,
            {"report_issue_only": 0.08, "ask_question": 0.06, "request_fix": -0.06},
        )
    elif clarity_level == "missing_some_details":
        completeness_weights = apply_weight_adjustments(
            completeness_weights,
            {"complete": -0.28, "partial": 0.08, "minimal": 0.32},
        )
        knowledge_weights = apply_weight_adjustments(
            knowledge_weights,
            {
                "correct_understanding": -0.10,
                "confused": 0.08,
                "guessing": 0.08,
                "unsure_if_correct_channel": 0.08,
            },
        )
        intent_weights = apply_weight_adjustments(
            intent_weights,
            {"report_issue_only": 0.10, "confirm_action": -0.04},
        )

    if record["requester_role"] in {"intern", "new_hire"}:
        knowledge_weights = apply_weight_adjustments(
            knowledge_weights,
            {
                "confused": 0.08,
                "unsure_if_correct_channel": 0.10,
                "correct_understanding": -0.08,
            },
        )
        emotional_weights = apply_weight_adjustments(
            emotional_weights,
            {"apologetic": 0.06, "confused": 0.04},
        )

    if workflow_stage == "new_submission":
        intent_weights = apply_weight_adjustments(
            intent_weights,
            {
                "request_fix": 0.10,
                "ask_question": 0.05,
                "report_issue_only": 0.05,
                "ask_status": -0.10,
            },
        )
        interaction_weights = apply_weight_adjustments(
            interaction_weights,
            {"none": 0.14, "self_tried": 0.10, "following_up": -0.08},
        )
    elif workflow_stage == "triaged":
        intent_weights = apply_weight_adjustments(
            intent_weights,
            {"ask_status": 0.16, "escalate": 0.04},
        )
        interaction_weights = apply_weight_adjustments(
            interaction_weights,
            {"following_up": 0.14, "still_not_fixed": 0.08},
        )
    elif workflow_stage == "pending_requester_info":
        intent_weights = apply_weight_adjustments(
            intent_weights,
            {"confirm_action": 0.18, "ask_question": 0.06},
        )
        interaction_weights = apply_weight_adjustments(
            interaction_weights,
            {"asked_for_more_info": 0.28, "following_up": 0.08},
        )
    elif workflow_stage == "pending_approval":
        intent_weights = apply_weight_adjustments(
            intent_weights,
            {"ask_status": 0.18, "ask_question": 0.08},
        )
        interaction_weights = apply_weight_adjustments(
            interaction_weights,
            {"following_up": 0.16, "still_not_fixed": 0.10},
        )
    elif workflow_stage == "in_progress":
        intent_weights = apply_weight_adjustments(
            intent_weights,
            {"ask_status": 0.16, "escalate": 0.04},
        )
        interaction_weights = apply_weight_adjustments(
            interaction_weights,
            {"following_up": 0.18, "still_not_fixed": 0.12},
        )
    elif workflow_stage == "blocked":
        intent_weights = apply_weight_adjustments(
            intent_weights,
            {"ask_status": 0.12, "escalate": 0.12},
        )
        interaction_weights = apply_weight_adjustments(
            interaction_weights,
            {"following_up": 0.16, "still_not_fixed": 0.16},
        )
    elif workflow_stage == "escalated":
        intent_weights = apply_weight_adjustments(
            intent_weights,
            {"escalate": 0.26, "ask_status": 0.10, "request_fix": 0.06},
        )
        emotional_weights = apply_weight_adjustments(
            emotional_weights,
            {"urgent": 0.14, "frustrated": 0.10, "passive": -0.04},
        )
        interaction_weights = apply_weight_adjustments(
            interaction_weights,
            {"still_not_fixed": 0.20, "following_up": 0.12},
        )

    if record["urgency"] in {"high", "critical"}:
        emotional_weights = apply_weight_adjustments(
            emotional_weights,
            {"urgent": 0.18, "frustrated": 0.06, "passive": -0.06},
        )
        temporal_weights = apply_weight_adjustments(
            temporal_weights,
            {"since_recent_time": 0.10, "again_today": 0.08},
        )
    if record["business_event_context"] != "routine_operation":
        temporal_weights = apply_weight_adjustments(
            temporal_weights,
            {"after_change_or_update": 0.08, "since_recent_time": 0.06},
        )
    if record["business_event_context"] in {
        "new_hire_onboarding",
        "employee_offboarding",
        "org_restructure",
        "product_launch",
        "office_move",
    }:
        temporal_weights = apply_weight_adjustments(
            temporal_weights,
            {"after_change_or_update": 0.12},
        )
    if record["sub_intent"] in SYSTEM_SYNC_ISSUES:
        temporal_weights = apply_weight_adjustments(
            temporal_weights,
            {"after_change_or_update": 0.14},
        )
    if record["department"] in {"IT", "Security", "Facilities", "Engineering", "Data"}:
        interaction_weights = apply_weight_adjustments(
            interaction_weights,
            {"self_tried": 0.12},
        )

    if difficulty == "easy":
        completeness_weights = apply_weight_adjustments(
            completeness_weights,
            {"complete": 0.08, "minimal": -0.08},
        )
    elif difficulty == "hard":
        completeness_weights = apply_weight_adjustments(
            completeness_weights,
            {"partial": 0.06, "minimal": 0.10, "complete": -0.08},
        )
        emotional_weights = apply_weight_adjustments(
            emotional_weights,
            {"confused": 0.06, "passive": 0.04},
        )
        knowledge_weights = apply_weight_adjustments(
            knowledge_weights,
            {"guessing": 0.06, "misdiagnosing": 0.06},
        )

    message_channel = weighted_choice(rng, message_channel_weights)
    completeness_mode = weighted_choice(rng, completeness_weights)
    intent_mode = weighted_choice(rng, intent_weights)
    knowledge_state = weighted_choice(rng, knowledge_weights)
    emotional_tone = weighted_choice(rng, emotional_weights)
    temporal_pov = weighted_choice(rng, temporal_weights)
    interaction_pov = weighted_choice(rng, interaction_weights)

    structural_weights = {
        "symptom_first": 0.22,
        "request_first": 0.18,
        "context_first": 0.14,
        "fragmented": 0.12,
        "follow_up": 0.14,
        "narrative": 0.20,
    }
    if intent_mode in {"ask_status", "confirm_action"}:
        structural_weights = apply_weight_adjustments(
            structural_weights,
            {"follow_up": 0.22, "request_first": 0.04},
        )
    if interaction_pov in {"following_up", "still_not_fixed", "asked_for_more_info"}:
        structural_weights = apply_weight_adjustments(
            structural_weights,
            {"follow_up": 0.26, "fragmented": 0.04, "narrative": -0.08},
        )
    if record["business_event_context"] != "routine_operation":
        structural_weights = apply_weight_adjustments(
            structural_weights,
            {"context_first": 0.12, "narrative": 0.08},
        )
    if completeness_mode == "minimal":
        structural_weights = apply_weight_adjustments(
            structural_weights,
            {"fragmented": 0.24, "symptom_first": 0.10, "narrative": -0.12},
        )
    elif completeness_mode == "partial":
        structural_weights = apply_weight_adjustments(
            structural_weights,
            {"symptom_first": 0.06, "request_first": 0.04},
        )
    if intent_mode == "ask_question":
        structural_weights = apply_weight_adjustments(
            structural_weights,
            {"request_first": 0.12, "context_first": 0.06},
        )
    if intent_mode == "report_issue_only":
        structural_weights = apply_weight_adjustments(
            structural_weights,
            {"symptom_first": 0.10, "fragmented": 0.08, "request_first": -0.08},
        )
    if emotional_tone == "urgent":
        structural_weights = apply_weight_adjustments(
            structural_weights,
            {"symptom_first": 0.12, "request_first": 0.06},
        )
    if message_channel == "chat_message":
        structural_weights = apply_weight_adjustments(
            structural_weights,
            {"fragmented": 0.08, "symptom_first": 0.04},
        )
    structural_mode = weighted_choice(rng, structural_weights)

    length_weights = {
        "very_short": 0.10,
        "short": 0.32,
        "medium": 0.38,
        "long": 0.20,
    }
    if structural_mode == "fragmented":
        length_weights = apply_weight_adjustments(
            length_weights,
            {"very_short": 0.40, "short": 0.18, "medium": -0.10, "long": -0.18},
        )
    elif structural_mode == "narrative":
        length_weights = apply_weight_adjustments(
            length_weights,
            {"long": 0.34, "medium": 0.12, "very_short": -0.08, "short": -0.10},
        )
    elif structural_mode == "context_first":
        length_weights = apply_weight_adjustments(
            length_weights,
            {"medium": 0.10, "long": 0.10},
        )
    elif structural_mode == "follow_up":
        length_weights = apply_weight_adjustments(
            length_weights,
            {"short": 0.18, "medium": 0.08},
        )
    elif structural_mode == "symptom_first":
        length_weights = apply_weight_adjustments(
            length_weights,
            {"very_short": 0.08, "short": 0.10},
        )
    elif structural_mode == "request_first":
        length_weights = apply_weight_adjustments(
            length_weights,
            {"short": 0.12},
        )
    if completeness_mode == "minimal":
        length_weights = apply_weight_adjustments(
            length_weights,
            {"very_short": 0.18, "short": 0.10, "long": -0.12},
        )
    elif completeness_mode == "complete":
        length_weights = apply_weight_adjustments(
            length_weights,
            {"medium": 0.06, "long": 0.06},
        )
    if writing_style == "concise":
        length_weights = apply_weight_adjustments(
            length_weights,
            {"very_short": 0.10, "short": 0.10, "long": -0.10},
        )
    elif writing_style in {"formal", "technical"}:
        length_weights = apply_weight_adjustments(
            length_weights,
            {"medium": 0.06, "long": 0.08},
        )
    length_style = weighted_choice(rng, length_weights)

    messiness_weights = {
        "polished": 0.26,
        "natural": 0.50,
        "slightly_messy": 0.24,
    }
    if message_channel == "chat_message":
        messiness_weights = apply_weight_adjustments(
            messiness_weights,
            {"slightly_messy": 0.10, "polished": -0.06},
        )
    if structural_mode == "fragmented":
        messiness_weights = apply_weight_adjustments(
            messiness_weights,
            {"slightly_messy": 0.18, "polished": -0.10},
        )
    if writing_style == "formal":
        messiness_weights = apply_weight_adjustments(
            messiness_weights,
            {"polished": 0.16, "slightly_messy": -0.08},
        )
    elif writing_style == "frustrated":
        messiness_weights = apply_weight_adjustments(
            messiness_weights,
            {"slightly_messy": 0.14},
        )
    elif writing_style == "non_native_english":
        messiness_weights = apply_weight_adjustments(
            messiness_weights,
            {"slightly_messy": 0.12},
        )
    if completeness_mode == "minimal":
        messiness_weights = apply_weight_adjustments(
            messiness_weights,
            {"slightly_messy": 0.08},
        )
    messiness = weighted_choice(rng, messiness_weights)

    greeting_probability = profile.opener_probability
    if message_channel == "email_note":
        greeting_probability += 0.18
    elif message_channel == "chat_message":
        greeting_probability -= 0.28
    if structural_mode == "fragmented":
        greeting_probability -= 0.20
    elif structural_mode == "narrative":
        greeting_probability += 0.06
    elif structural_mode == "follow_up":
        greeting_probability -= 0.08
    if intent_mode in {"ask_status", "escalate"}:
        greeting_probability -= 0.06
    if length_style == "very_short":
        greeting_probability -= 0.14

    system_probability = effective_system_probability
    if completeness_mode == "partial":
        system_probability -= 0.10
    elif completeness_mode == "minimal":
        system_probability -= 0.22
    if knowledge_state in {"confused", "guessing", "unsure_if_correct_channel"}:
        system_probability -= 0.10
    if structural_mode in {"fragmented", "symptom_first"}:
        system_probability -= 0.08
    if message_channel == "chat_message":
        system_probability -= 0.08
    if temporal_pov == "after_change_or_update":
        system_probability += 0.10

    context_probability = profile.context_probability
    if structural_mode in {"context_first", "narrative"}:
        context_probability += 0.30
    elif structural_mode == "fragmented":
        context_probability -= 0.24
    elif structural_mode == "request_first":
        context_probability -= 0.06
    if record["business_event_context"] != "routine_operation":
        context_probability += 0.18
    if completeness_mode == "minimal":
        context_probability -= 0.22
    elif completeness_mode == "complete":
        context_probability += 0.10
    if intent_mode == "confirm_action":
        context_probability += 0.08

    previous_attempts_probability = profile.progress_probability
    if interaction_pov in {"following_up", "still_not_fixed", "asked_for_more_info"}:
        previous_attempts_probability += 0.34
    elif interaction_pov == "redirected_here":
        previous_attempts_probability += 0.18
    if workflow_stage != "new_submission":
        previous_attempts_probability += 0.14
    if structural_mode == "follow_up":
        previous_attempts_probability += 0.24

    time_reference_probability = {
        "first_occurrence": 0.18,
        "recurring_issue": 0.38,
        "since_recent_time": 0.62,
        "again_today": 0.72,
        "after_change_or_update": 0.66,
    }[temporal_pov]
    if record["urgency"] in {"high", "critical"}:
        time_reference_probability += 0.10
    if structural_mode == "fragmented":
        time_reference_probability -= 0.08

    consequence_probability = 0.12
    if record["urgency"] == "medium":
        consequence_probability += 0.08
    elif record["urgency"] in {"high", "critical"}:
        consequence_probability += 0.22
    if record["severity"] in {"major", "business_critical"}:
        consequence_probability += 0.18
    if record["impact_scope"] != "single_user":
        consequence_probability += 0.10
    if intent_mode == "escalate":
        consequence_probability += 0.12
    elif intent_mode == "report_issue_only":
        consequence_probability -= 0.06

    other_people_probability = {
        "single_user": 0.04,
        "single_team": 0.28,
        "single_department": 0.38,
        "multi_department": 0.48,
        "company_wide": 0.44,
    }[record["impact_scope"]]
    if emotional_tone == "passive":
        other_people_probability -= 0.06

    resolution_probability = {
        "request_fix": 0.86,
        "ask_question": 0.56,
        "ask_status": 0.68,
        "report_issue_only": 0.14,
        "confirm_action": 0.38,
        "escalate": 0.74,
    }[intent_mode]
    if completeness_mode == "minimal":
        resolution_probability -= 0.14
    if structural_mode == "fragmented":
        resolution_probability -= 0.12
    if clarity_level in {"clear", "moderately_clear"}:
        resolution_probability += 0.08
    elif clarity_level in {"vague", "missing_some_details"}:
        resolution_probability -= 0.10

    uncertainty_probability = {
        "correct_understanding": 0.08,
        "confused": 0.72,
        "misdiagnosing": 0.40,
        "guessing": 0.74,
        "unsure_if_correct_channel": 0.62,
    }[knowledge_state]
    if clarity_level in {"vague", "missing_some_details"}:
        uncertainty_probability += 0.12

    self_help_probability = {
        "none": 0.10,
        "self_tried": 0.92,
        "redirected_here": 0.12,
        "still_not_fixed": 0.36,
        "following_up": 0.24,
        "asked_for_more_info": 0.18,
    }[interaction_pov]
    if record["department"] in {"IT", "Security", "Facilities", "Engineering", "Data"}:
        self_help_probability += 0.10

    pattern_weights = dict(profile.sentence_pattern_weights)
    if structural_mode == "symptom_first":
        pattern_weights = apply_weight_adjustments(
            pattern_weights,
            {"pressure_first": 0.18, "compressed": 0.06},
        )
    elif structural_mode == "request_first":
        pattern_weights = apply_weight_adjustments(
            pattern_weights,
            {"standard": 0.18},
        )
    elif structural_mode == "context_first":
        pattern_weights = apply_weight_adjustments(
            pattern_weights,
            {"context_first": 0.34},
        )
    elif structural_mode == "fragmented":
        pattern_weights = apply_weight_adjustments(
            pattern_weights,
            {"compressed": 0.44, "standard": -0.06},
        )
    elif structural_mode == "follow_up":
        pattern_weights = apply_weight_adjustments(
            pattern_weights,
            {"status_first": 0.40, "compressed": 0.06},
        )
    elif structural_mode == "narrative":
        pattern_weights = apply_weight_adjustments(
            pattern_weights,
            {"context_first": 0.20, "standard": 0.10},
        )

    return TextVariationProfile(
        structural_mode=structural_mode,
        knowledge_state=knowledge_state,
        intent_mode=intent_mode,
        emotional_tone=emotional_tone,
        temporal_pov=temporal_pov,
        interaction_pov=interaction_pov,
        message_channel=message_channel,
        length_style=length_style,
        completeness_mode=completeness_mode,
        messiness=messiness,
        mention_system_name=rng.random() < clamp_probability(system_probability),
        include_greeting=rng.random() < clamp_probability(greeting_probability),
        include_context=rng.random() < clamp_probability(context_probability),
        mention_previous_attempts_or_follow_up=rng.random()
        < clamp_probability(previous_attempts_probability),
        mention_time_reference=rng.random() < clamp_probability(time_reference_probability),
        mention_work_consequence=rng.random() < clamp_probability(consequence_probability),
        mention_other_people_affected=(
            record["impact_scope"] != "single_user"
            and rng.random() < clamp_probability(other_people_probability)
        ),
        include_explicit_resolution_request=rng.random()
        < clamp_probability(resolution_probability),
        prefer_indirect_issue_description=(
            knowledge_state in {"confused", "misdiagnosing", "guessing"}
            or structural_mode in {"symptom_first", "fragmented"}
            or rng.random()
            < clamp_probability(
                profile.indirect_issue_probability
                + (0.10 if completeness_mode != "complete" else 0.0)
            )
        ),
        prefer_symptom_before_request=structural_mode in {"symptom_first", "fragmented"}
        or (structural_mode == "follow_up" and intent_mode in {"report_issue_only", "ask_status"}),
        mention_uncertainty=rng.random() < clamp_probability(uncertainty_probability),
        mention_self_help_attempt=rng.random() < clamp_probability(self_help_probability),
        allow_incomplete_details=(
            completeness_mode != "complete" or structural_mode == "fragmented"
        ),
        allow_fragment_style=structural_mode == "fragmented"
        or (message_channel == "chat_message" and length_style in {"very_short", "short"}),
        sentence_pattern=weighted_choice(rng, pattern_weights),
    )


def build_llm_generation_directives(
    record: Record,
    rng: random.Random,
    difficulty: str,
    system_mention_probability: float | None = None,
    variation_profile: TextVariationProfile | None = None,
) -> dict[str, Any]:
    sampled_profile = variation_profile or sample_text_variation_profile(
        record,
        rng,
        difficulty,
        system_mention_probability=system_mention_probability,
    )
    return {
        "structural_pov": sampled_profile.structural_mode,
        "knowledge_pov": sampled_profile.knowledge_state,
        "intent_pov": sampled_profile.intent_mode,
        "emotional_pov": sampled_profile.emotional_tone,
        "temporal_pov": TEMPORAL_DIRECTIVE_MAP[sampled_profile.temporal_pov],
        "interaction_pov": INTERACTION_DIRECTIVE_MAP[sampled_profile.interaction_pov],
        "message_channel": sampled_profile.message_channel,
        "length_style": sampled_profile.length_style,
        "completeness_mode": sampled_profile.completeness_mode,
        "messiness": sampled_profile.messiness,
        "mention_system_name": sampled_profile.mention_system_name,
        "include_greeting": sampled_profile.include_greeting,
        "include_context": sampled_profile.include_context,
        "mention_previous_attempts_or_follow_up": (
            sampled_profile.mention_previous_attempts_or_follow_up
        ),
        "mention_time_reference": sampled_profile.mention_time_reference,
        "mention_work_consequence": sampled_profile.mention_work_consequence,
        "mention_other_people_affected": sampled_profile.mention_other_people_affected,
        "include_explicit_resolution_request": (
            sampled_profile.include_explicit_resolution_request
        ),
        "prefer_indirect_issue_description": (sampled_profile.prefer_indirect_issue_description),
        "prefer_symptom_before_request": sampled_profile.prefer_symptom_before_request,
        "mention_uncertainty": sampled_profile.mention_uncertainty,
        "mention_self_help_attempt": sampled_profile.mention_self_help_attempt,
        "allow_incomplete_details": sampled_profile.allow_incomplete_details,
        "allow_fragment_style": sampled_profile.allow_fragment_style,
        "sentence_pattern": sampled_profile.sentence_pattern,
    }


def build_surface_metadata(record: Record) -> dict[str, str]:
    return build_shared_surface_metadata(record, SURFACE_METADATA_FIELDS)


def maybe_append_guidance(
    guidance: list[str],
    rng: random.Random,
    options: list[str],
    probability: float = 1.0,
) -> None:
    if options and rng.random() < clamp_probability(probability):
        guidance.append(pick_one(rng, options))


def build_hidden_generation_guidance(
    record: Record,
    rng: random.Random,
    variation_profile: TextVariationProfile | None = None,
) -> list[str]:
    sampled_profile = variation_profile or sample_text_variation_profile(
        record,
        rng,
        "medium",
    )
    guidance: list[str] = []

    guidance.append(STRUCTURE_GUIDANCE[sampled_profile.structural_mode])
    guidance.append(KNOWLEDGE_GUIDANCE[sampled_profile.knowledge_state])
    guidance.append(INTENT_GUIDANCE[sampled_profile.intent_mode])
    guidance.append(EMOTION_GUIDANCE[sampled_profile.emotional_tone])
    guidance.append(COMPLETENESS_GUIDANCE[sampled_profile.completeness_mode])

    if sampled_profile.mention_time_reference:
        guidance.append(TEMPORAL_GUIDANCE[sampled_profile.temporal_pov])

    if sampled_profile.interaction_pov != "none":
        guidance.append(INTERACTION_GUIDANCE[sampled_profile.interaction_pov])

    if not sampled_profile.mention_system_name:
        guidance.append(
            "do not force the product name into the ticket if the user would naturally omit it"
        )
    if sampled_profile.mention_work_consequence:
        guidance.append("mention what this is delaying, blocking, or making harder")
    if sampled_profile.mention_other_people_affected:
        guidance.append("hint that other people are affected without using formal scope language")
    if sampled_profile.mention_uncertainty:
        guidance.append(
            "let the requester hedge, guess, or admit uncertainty about the cause or next step"
        )

    maybe_append_guidance(guidance, rng, WORKFLOW_GUIDANCE[record["workflow_stage"]])

    if record["approval_required"] != "none":
        approval_probability = 0.24
        if record["workflow_stage"] in {"pending_approval", "blocked", "escalated"}:
            approval_probability += 0.20
        maybe_append_guidance(
            guidance,
            rng,
            APPROVAL_GUIDANCE[record["approval_required"]],
            probability=approval_probability,
        )

    dependency_probability = DEPENDENCY_GUIDANCE_PROBABILITY[record["dependency_type"]]
    if record["workflow_stage"] in {"blocked", "pending_requester_info"}:
        dependency_probability += 0.10
    maybe_append_guidance(
        guidance,
        rng,
        DEPENDENCY_GUIDANCE[record["dependency_type"]],
        probability=dependency_probability,
    )

    severity_probability = SEVERITY_GUIDANCE_PROBABILITY[record["severity"]]
    if record["urgency"] in {"high", "critical"}:
        severity_probability += 0.10
    maybe_append_guidance(
        guidance,
        rng,
        SEVERITY_GUIDANCE[record["severity"]],
        probability=severity_probability,
    )

    impact_probability = IMPACT_GUIDANCE_PROBABILITY[record["impact_scope"]]
    if record["severity"] in {"major", "business_critical"}:
        impact_probability += 0.08
    maybe_append_guidance(
        guidance,
        rng,
        IMPACT_GUIDANCE[record["impact_scope"]],
        probability=impact_probability,
    )

    return guidance


def prompt_safe_schema(
    schema: dict[str, Any],
    fields: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    prompt_schema: dict[str, Any] = {}
    field_names = fields or tuple(schema.keys())
    for field in field_names:
        config = schema[field]
        prompt_config: dict[str, Any] = {}
        if "values" in config:
            prompt_config["values"] = list(config["values"])
        if "parents" in config:
            prompt_config["parents"] = list(config["parents"])
        if "mapping" in config:
            mapping = config["mapping"]
            parents = config.get("parents", [])
            if parents and any(isinstance(key, tuple) for key in mapping):
                prompt_config["mapping"] = [
                    {
                        "parents": {parents[index]: key[index] for index in range(len(parents))},
                        "values": list(values),
                    }
                    for key, values in mapping.items()
                ]
            else:
                prompt_config["mapping"] = {
                    str(key): list(values) for key, values in mapping.items()
                }
        prompt_schema[field] = prompt_config
    return prompt_schema


def build_openrouter_batch_prompt(
    batch_rows: list[dict[str, Any]],
    difficulty: str,
) -> str:
    schema_for_prompt = prompt_safe_schema(SCHEMA, SURFACE_METADATA_FIELDS)
    banned_phrases = ", ".join(f"'{phrase}'" for phrase in BANNED_INTERNAL_PROCESS_PHRASES)
    return (
        "Generate text values from the row data below.\n"
        "The surface metadata is authoritative ground truth for the user-side scenario.\n"
        "The hidden_generation_guidance entries are private tone and stakes hints derived from "
        "internal labels; translate them into natural requester language rather than process "
        "terms.\n"
        "The text_generation_directives fields control point of view, structure, "
        "knowledge level, intent, "
        "emotional tone, time framing, completeness, and interaction context.\n"
        "Each output item must correspond to exactly one input row_id.\n"
        "Write only the natural-language ticket text.\n"
        "Write like a requester talking to support, not like a support analyst or dashboard.\n"
        "Most metadata values are classification labels. Interpret them semantically; do not copy "
        "snake_case labels into the ticket.\n"
        "Treat issue_type as the broad cluster label and sub_intent as the concrete scenario.\n"
        "Only mention system_name when mention_system_name is true or a product name is genuinely "
        "needed for coherence.\n"
        "Use ordinary business language: symptoms, confusion, prior attempts, blockers, missing "
        "information, deadlines, and work consequences.\n"
        "It is fine for some tickets to be one sentence, incomplete, vague, messy, fragmented, or "
        "simple follow-up nudges.\n"
        "Do not use headings, bullets, or templated sections.\n"
        "For rows that share an issue_type, do not generate simple rewordings of the same "
        "pattern.\n"
        "Rows with the same issue_type may have different sub_intent values; keep those concrete "
        "details aligned.\n"
        "Generate fundamentally different perspectives using the structural_pov, knowledge_pov, "
        "intent_pov, emotional_pov, temporal_pov, and interaction_pov fields.\n"
        "Two rows with the same issue_type should still differ in opening style, sentence order, "
        "confidence, framing, and whether the requester asks, reports, follows up, or escalates.\n"
        "Treat structural_pov as binding: symptom_first, request_first, context_first, "
        "fragmented, follow_up, and narrative should produce clearly different openings and flow.\n"
        "Use knowledge_pov to vary whether the requester understands the problem, is confused, is "
        "guessing, or is misdiagnosing it.\n"
        "Use temporal_pov to vary whether the issue feels new, recurring, again today, or caused "
        "by a recent change.\n"
        "Use interaction_pov to vary whether this is a first report, a follow-up, something they "
        "already tried to solve, or something they were told to submit.\n"
        "Never use or closely paraphrase these phrases: "
        f"{banned_phrases}.\n"
        "Do not invent facts that contradict the surface metadata.\n"
        "Directive reference:\n"
        "  structural_pov: binding overall structure for the ticket opening and flow\n"
        "  knowledge_pov: how well the requester understands the issue\n"
        "  intent_pov: whether the requester wants a fix, answer, status, confirmation, or "
        "escalation\n"
        "  issue_type: broad cluster label for the underlying request family\n"
        "  intent_group: higher-level grouping above issue_type\n"
        "  sub_intent: specific underlying workflow scenario that must stay aligned with the text\n"
        "  emotional_pov: tone such as neutral, urgent, confused, apologetic, or mildly annoyed\n"
        "  temporal_pov: whether it sounds new, recurring, happening again today, or tied to a "
        "recent change\n"
        "  interaction_pov: whether this includes a self-try, redirect, follow-up, "
        "still-not-fixed note, or added details\n"
        "  message_channel: light flavor of portal form, email note, or internal chat\n"
        "  length_style: target approximate length\n"
        "  completeness_mode: complete, partial, or minimal detail level\n"
        "  messiness: polished, natural, or slightly messy writing\n"
        "  mention_system_name: only name the product or system when true\n"
        "  include_greeting: include a short opener only when it fits the mode\n"
        "  include_context: include extra background or setup context when true\n"
        "  mention_previous_attempts_or_follow_up: briefly mention retries, earlier contact, "
        "or ongoing delay if true\n"
        "  mention_time_reference: mention timing such as today, this morning, before Monday, "
        "or last week if true\n"
        "  mention_work_consequence: mention how the issue affects work if true\n"
        "  mention_other_people_affected: hint that others are affected without formal scope "
        "language if true\n"
        "  include_explicit_resolution_request: clearly ask for help or action if true\n"
        "  prefer_indirect_issue_description: describe symptoms before diagnosis when true\n"
        "  prefer_symptom_before_request: open with what is happening before the ask when true\n"
        "  mention_uncertainty: let the requester sound unsure about the cause or next step if "
        "true\n"
        "  mention_self_help_attempt: mention one brief thing already tried if true\n"
        "  allow_incomplete_details: it is acceptable to leave some obvious details unstated\n"
        "  allow_fragment_style: sentence fragments or clipped wording are acceptable\n"
        "  sentence_pattern: overall ordering preference; treat status_first as a follow-up-"
        "first opening and pressure_first as starting with urgency or consequence\n"
        "Follow the row-level directives as closely as possible.\n"
        f"Difficulty preset: {difficulty}\n"
        "Return JSON that matches this schema exactly:\n"
        f"{json.dumps(generated_text_response_schema(), indent=2)}\n\n"
        "Surface metadata schema:\n"
        f"{json.dumps(schema_for_prompt, indent=2)}\n\n"
        "Rows:\n"
        f"{json.dumps(batch_rows, indent=2)}"
    )


def response_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                text_attr = getattr(item, "text", None)
                if isinstance(text_attr, str):
                    parts.append(text_attr)
        combined = "".join(parts).strip()
        if combined:
            return combined
    raise TypeError("Expected OpenRouter response content to be a string or text blocks.")


def build_evaluation_row_payload(row_id: int, row: dict[str, str]) -> dict[str, Any]:
    metadata = {field: row[field] for field in SCHEMA.keys()}
    return {
        "row_id": row_id,
        "text": _normalize_text(row.get("ticket_text", row.get("text", ""))),
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
        f"{OPENROUTER_EVALUATION_PROMPT}\n"
        f"{json.dumps(evaluation_response_schema(), indent=2)}\n\n"
        "Rows:\n"
        f"{json.dumps(batch_rows, indent=2)}"
    )


def find_banned_internal_phrase(ticket_text: str) -> str | None:
    normalized_text = ticket_text.lower()
    for banned_phrase in BANNED_INTERNAL_PROCESS_PHRASES:
        if banned_phrase in normalized_text:
            return banned_phrase
    return None


def _looks_like_template_fallback(text: str) -> bool:
    normalized = _normalize_text(text).lower()
    if not normalized:
        return False
    return normalized.startswith("i need help with") and (
        "this is still affecting my work and needs follow-up." in normalized
        or normalized.endswith("can someone take a look today?")
    )


def evaluate_ticket_text_deterministically(
    record: Record,
    text: str,
) -> DeterministicEvaluationResult:
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return DeterministicEvaluationResult(
            rating="not_pass",
            reasons=("empty_text",),
            details={
                "issue_type_match": "false",
                "sub_intent_match": "false",
                "banned_phrase": "",
            },
        )

    assessment = assess_ticket_text_alignment(record, normalized_text)
    banned_phrase = find_banned_internal_phrase(normalized_text)

    rating: EvaluationRating = "pass"
    reasons: list[str] = []
    if not assessment.issue_type_match:
        rating = "not_pass"
        reasons.append("issue_type_mismatch")
    if banned_phrase is not None:
        rating = "not_pass"
        reasons.append(f"banned_internal_phrase:{banned_phrase}")
    if not assessment.sub_intent_match:
        if rating == "pass":
            rating = "need_attention"
        reasons.append("sub_intent_not_explicit")
    if _looks_like_template_fallback(normalized_text):
        if rating == "pass":
            rating = "need_attention"
        reasons.append("template_like_fallback_text")
    if len(normalized_text.split()) < 8:
        if rating == "pass":
            rating = "need_attention"
        reasons.append("very_short_text")

    return DeterministicEvaluationResult(
        rating=rating,
        reasons=tuple(reasons),
        details={
            "issue_type_match": str(assessment.issue_type_match).lower(),
            "sub_intent_match": str(assessment.sub_intent_match).lower(),
            "banned_phrase": banned_phrase or "",
        },
    )


def alignment_terms_preview(record: Record, limit: int = 8) -> str:
    return ", ".join(build_alignment_terms(record["issue_type"], record["sub_intent"])[:limit])


def validate_ticket_text_alignment(record: Record, ticket_text: str) -> AlignmentAssessment:
    banned_phrase = find_banned_internal_phrase(ticket_text)
    if banned_phrase is not None:
        raise ValueError(
            f"Generated ticket text includes banned internal process phrase: {banned_phrase}"
        )

    assessment = assess_ticket_text_alignment(record, ticket_text)
    if not assessment.issue_type_match:
        raise ValueError(
            "Generated ticket text failed label alignment checks for "
            f"{record['issue_type']=} and {record['sub_intent']=}. "
            f"Expected hints included: {alignment_terms_preview(record)}."
        )
    return assessment


def collect_alignment_failures(
    records: list[Record],
    parsed: dict[int, str],
) -> tuple[dict[int, ValueError], dict[int, ValueError]]:
    hard_failures: dict[int, ValueError] = {}
    soft_failures: dict[int, ValueError] = {}
    for row_id, ticket_text in parsed.items():
        record = records[row_id]
        assessment = assess_ticket_text_alignment(record, ticket_text)
        if not assessment.issue_type_match:
            hard_failures[row_id] = ValueError(
                "Generated ticket text failed label alignment checks for "
                f"{record['issue_type']=} and {record['sub_intent']=}. "
                f"Expected hints included: {alignment_terms_preview(record)}."
            )
            continue

        banned_phrase = find_banned_internal_phrase(ticket_text)
        if banned_phrase is not None:
            soft_failures[row_id] = ValueError(
                "Generated ticket text used banned internal process wording "
                f"('{banned_phrase}'). Rewrite it using natural requester language."
            )
            continue

        if not assessment.sub_intent_match:
            soft_failures[row_id] = ValueError(
                "Generated ticket text matched the broad issue family but did not make the "
                f"sampled leaf scenario explicit for {record['sub_intent']=}. "
                f"Helpful concepts included: {alignment_terms_preview(record, limit=12)}."
            )

    return hard_failures, soft_failures


def build_alignment_retry_prefix(
    records: list[Record],
    parsed: dict[int, str],
) -> str | None:
    hard_failure_items, soft_failure_items = collect_alignment_failures(records, parsed)
    failures: list[str] = []
    for row_id in sorted({*hard_failure_items, *soft_failure_items}):
        ticket_text = parsed[row_id]
        record = records[row_id]
        failure = hard_failure_items.get(row_id) or soft_failure_items.get(row_id)
        failure_reason = str(failure) if failure is not None else "alignment mismatch"
        failures.append(
            f"Row {row_id}: keep the same metadata, but make the scenario more explicit. "
            f"The text should clearly imply {record['sub_intent']} within "
            f"{record['issue_type']} using normal requester wording.\n"
            f"Rewrite reason: {failure_reason}\n"
            f"Failed text: {ticket_text}\n"
            f"Helpful concepts to imply naturally: {alignment_terms_preview(record, limit=12)}"
        )

    return build_shared_alignment_retry_prefix(
        failures,
        intro_lines=(
            "The previous response violated alignment or wording constraints for some rows.",
            "Retry only the rows listed below and return valid JSON only.",
            "Keep every row_id exactly the same.",
            (
                "For the rows below, rewrite the text so the concrete scenario is obvious "
                "from natural requester language without using backend labels."
            ),
        ),
        max_failures=3,
    )


def build_aligned_fallback_ticket_text(record: Record) -> str:
    detail = preferred_sub_intent_phrase(record["sub_intent"])
    system_clause = f" in {record['system_name']}" if record.get("system_name") else ""
    intent_group = record["intent_group"]
    department = record["department"]

    if record["issue_type"] == "security_incident" or intent_group == "security_operations":
        first_sentence = f"I need security help with {detail}{system_clause}."
        second_sentence = "This looks serious and I need guidance on what to do next."
    elif record["issue_type"] in {"access_request", "authentication_issue"}:
        first_sentence = f"I need help with {detail}{system_clause}."
        second_sentence = "I still cannot get the access I need to keep working."
    elif intent_group == "tech_support":
        first_sentence = f"I am running into {detail}{system_clause}."
        second_sentence = "The tool is still not usable and it is blocking my work."
    elif intent_group == "finance_procurement":
        first_sentence = f"I need help with {detail}{system_clause}."
        second_sentence = "This is affecting the workflow and needs to be corrected."
    elif intent_group == "policy_compliance":
        first_sentence = f"I need help with {detail}{system_clause}."
        second_sentence = "This is still waiting on review and is holding up the work."
    elif intent_group == "workplace_operations":
        if department == "Facilities":
            first_sentence = f"I need facilities help with {detail} in the office."
            second_sentence = "This is still affecting the workspace and needs attention."
        else:
            first_sentence = f"I need help with {detail}{system_clause}."
            second_sentence = (
                "This is still affecting an office or workplace task and needs attention."
            )
    elif intent_group == "go_to_market_ops":
        first_sentence = f"I need help with {detail}{system_clause}."
        second_sentence = "This is still affecting our sales or marketing workflow."
    elif intent_group == "customer_operations":
        first_sentence = f"I need help with {detail}{system_clause}."
        second_sentence = (
            "This is still affecting customer support operations and needs a response."
        )
    elif intent_group == "engineering_operations":
        first_sentence = f"I need help with {detail}{system_clause}."
        second_sentence = "This is still blocking engineering work and needs attention."
    elif intent_group == "data_operations":
        first_sentence = f"I need help with {detail}{system_clause}."
        second_sentence = "This is still affecting data or reporting work and needs attention."
    elif intent_group == "travel_operations":
        first_sentence = f"I need help with {detail}{system_clause}."
        second_sentence = "This is still affecting travel planning and needs follow-up."
    elif intent_group == "people_ops":
        if record["issue_type"] == "policy_question":
            first_sentence = f"I need clarification on {detail}."
            second_sentence = "I am still not sure how this should be handled."
        else:
            first_sentence = f"I need help with {detail}{system_clause}."
            second_sentence = (
                "This is still affecting a people or payroll workflow and needs a response."
            )
    else:
        first_sentence = f"I need help with {detail}{system_clause}."
        second_sentence = "This is still blocking the work and needs a response."

    return f"{first_sentence} {second_sentence} Can someone take a look today?"


def repair_ticket_text_alignment(record: Record, ticket_text: str) -> str:
    fallback_text = build_aligned_fallback_ticket_text(record)
    assessment = validate_ticket_text_alignment(record, fallback_text)
    if not assessment.sub_intent_match:
        raise ValueError(
            "Generated fallback ticket text did not make the sampled leaf scenario explicit."
        )
    return fallback_text
