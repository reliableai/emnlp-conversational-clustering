"""Social-media schema and taxonomy constants for structured dataset generation."""

from __future__ import annotations

import re
from typing import Any

PLATFORM_VALUES = [
    "x_like",
    "reddit_like",
    "linkedin_like",
    "instagram_caption",
    "youtube_comment",
    "facebook_like",
]

TOPIC_DOMAIN_VALUES = [
    "politics",
    "economy_finance",
    "technology",
    "entertainment_pop_culture",
    "sports",
    "health_fitness",
    "education",
    "environment_climate",
    "personal_life_diary",
    "business_startups",
]

STANCE_VALUES = [
    "supportive",
    "critical",
    "neutral",
    "mixed",
    "promotional",
]

SENTIMENT_POLARITY_VALUES = [
    "very_negative",
    "negative",
    "neutral",
    "mixed",
    "positive",
    "very_positive",
]

VOICE_TONE_VALUES = [
    "neutral",
    "optimistic",
    "skeptical",
    "frustrated",
    "angry",
    "hopeful",
    "sarcastic",
    "urgent",
    "informative",
    "reflective",
]

CERTAINTY_CONFIDENCE_VALUES = [
    "speculative",
    "uncertain",
    "mixed_certainty",
    "confident",
    "very_confident",
    "data_backed",
    "none",
]

TOXICITY_CIVILITY_VALUES = [
    "highly_toxic_abusive",
    "aggressive",
    "critical_but_civil",
    "neutral",
    "polite",
    "supportive",
    "encouraging",
]

TEMPORAL_ORIENTATION_VALUES = [
    "past_reflection",
    "present_commentary",
    "near_future_prediction",
    "long_term_prediction",
    "timeless_general_statement",
    "real_time_reaction",
]

LENGTH_STYLE_VALUES = [
    "very_short",
    "short",
    "medium",
    "long",
    "threadish",
]

TOPIC_POST_ACT_MAPPING: dict[str, list[str]] = {
    "politics": [
        "inform_share_news",
        "express_opinion",
        "ask_question",
        "mobilize_call_to_action",
        "debate_argue",
    ],
    "economy_finance": [
        "inform_share_news",
        "express_opinion",
        "ask_question",
        "vent_rant",
    ],
    "technology": [
        "inform_share_news",
        "express_opinion",
        "ask_question",
        "promote_advertise",
        "vent_rant",
        "narrate_personal_experience",
    ],
    "entertainment_pop_culture": [
        "inform_share_news",
        "express_opinion",
        "entertain_joke",
        "ask_question",
    ],
    "sports": [
        "inform_share_news",
        "express_opinion",
        "entertain_joke",
        "vent_rant",
        "ask_question",
    ],
    "health_fitness": [
        "narrate_personal_experience",
        "ask_question",
        "inform_share_news",
        "promote_advertise",
        "mobilize_call_to_action",
    ],
    "education": [
        "inform_share_news",
        "express_opinion",
        "ask_question",
        "narrate_personal_experience",
        "mobilize_call_to_action",
    ],
    "environment_climate": [
        "inform_share_news",
        "express_opinion",
        "mobilize_call_to_action",
        "debate_argue",
        "ask_question",
    ],
    "personal_life_diary": [
        "narrate_personal_experience",
        "vent_rant",
        "ask_question",
        "express_opinion",
        "entertain_joke",
    ],
    "business_startups": [
        "inform_share_news",
        "express_opinion",
        "ask_question",
        "promote_advertise",
        "narrate_personal_experience",
        "debate_argue",
    ],
}

CONTENT_SCENARIO_MAPPING: dict[tuple[str, str], list[str]] = {
    ("politics", "inform_share_news"): [
        "local_election_result",
        "ethics_probe_update",
        "debate_night_recap",
    ],
    ("politics", "express_opinion"): [
        "budget_priorities_argument",
        "campaign_messaging_critique",
        "leadership_trust_commentary",
    ],
    ("politics", "ask_question"): [
        "ballot_measure_question",
        "debate_fact_check_question",
        "voting_rule_clarification",
    ],
    ("politics", "mobilize_call_to_action"): [
        "voter_registration_push",
        "contact_representatives_push",
        "rally_turnout_push",
    ],
    ("politics", "debate_argue"): [
        "tax_policy_thread",
        "border_policy_argument",
        "housing_policy_argument",
    ],
    ("economy_finance", "inform_share_news"): [
        "inflation_report_reaction",
        "rate_cut_update",
        "layoff_trend_report",
    ],
    ("economy_finance", "express_opinion"): [
        "rent_cost_pressure_take",
        "gig_economy_fees_critique",
        "wage_growth_commentary",
    ],
    ("economy_finance", "ask_question"): [
        "retirement_allocation_question",
        "credit_score_question",
        "emergency_fund_question",
    ],
    ("economy_finance", "vent_rant"): [
        "overdraft_fee_rant",
        "grocery_price_spike_rant",
        "airline_hidden_fee_rant",
    ],
    ("technology", "inform_share_news"): [
        "ai_feature_launch",
        "security_breach_update",
        "app_store_policy_change",
    ],
    ("technology", "express_opinion"): [
        "subscription_price_increase",
        "app_update_regression",
        "device_launch_reaction",
    ],
    ("technology", "ask_question"): [
        "laptop_buying_advice",
        "privacy_setting_question",
        "cloud_cost_question",
    ],
    ("technology", "promote_advertise"): [
        "indie_app_launch",
        "saas_discount_offer",
        "webinar_signup_pitch",
    ],
    ("technology", "vent_rant"): [
        "notification_overload_rant",
        "customer_support_bot_rant",
        "forced_ai_feature_rant",
    ],
    ("technology", "narrate_personal_experience"): [
        "failed_migration_story",
        "remote_setup_upgrade",
        "burnout_from_alerts",
    ],
    ("entertainment_pop_culture", "inform_share_news"): [
        "casting_announcement",
        "album_release_day",
        "streaming_service_merger",
    ],
    ("entertainment_pop_culture", "express_opinion"): [
        "finale_reaction",
        "celebrity_interview_take",
        "franchise_fatigue_commentary",
    ],
    ("entertainment_pop_culture", "entertain_joke"): [
        "fandom_meme_observation",
        "spoiler_avoidance_joke",
        "tour_ticket_queue_joke",
    ],
    ("entertainment_pop_culture", "ask_question"): [
        "watchlist_recommendation_request",
        "best_album_question",
        "spoiler_free_review_request",
    ],
    ("sports", "inform_share_news"): [
        "injury_update",
        "trade_deadline_news",
        "coaching_change_report",
    ],
    ("sports", "express_opinion"): [
        "playoff_collapse_reaction",
        "breakout_player_praise",
        "season_outlook_take",
    ],
    ("sports", "entertain_joke"): [
        "fantasy_team_self_roast",
        "referee_call_joke",
        "mascot_meme_post",
    ],
    ("sports", "vent_rant"): [
        "referee_call_outrage",
        "blackout_restriction_rant",
        "ticket_price_rant",
    ],
    ("sports", "ask_question"): [
        "starting_lineup_question",
        "trade_target_question",
        "playoff_odds_question",
    ],
    ("health_fitness", "narrate_personal_experience"): [
        "half_marathon_training_update",
        "sleep_reset_attempt",
        "meal_prep_burnout",
    ],
    ("health_fitness", "ask_question"): [
        "workout_split_question",
        "protein_intake_question",
        "knee_pain_form_question",
    ],
    ("health_fitness", "inform_share_news"): [
        "step_goal_study",
        "recall_notice_share",
        "heat_wave_health_alert",
    ],
    ("health_fitness", "promote_advertise"): [
        "coaching_slot_offer",
        "supplement_launch_post",
        "studio_trial_invite",
    ],
    ("health_fitness", "mobilize_call_to_action"): [
        "blood_drive_reminder",
        "community_run_signup",
        "mental_health_checkin_push",
    ],
    ("education", "inform_share_news"): [
        "school_budget_cut_update",
        "admissions_deadline_reminder",
        "district_policy_change",
    ],
    ("education", "express_opinion"): [
        "homework_load_commentary",
        "tuition_value_debate",
        "teacher_support_argument",
    ],
    ("education", "ask_question"): [
        "study_plan_help",
        "course_selection_question",
        "student_loan_question",
    ],
    ("education", "narrate_personal_experience"): [
        "exam_week_update",
        "first_day_teacher_reflection",
        "graduate_school_decision",
    ],
    ("education", "mobilize_call_to_action"): [
        "school_board_meeting_push",
        "teacher_supply_drive",
        "library_funding_petition",
    ],
    ("environment_climate", "inform_share_news"): [
        "wildfire_smoke_alert",
        "local_recycling_change",
        "climate_report_summary",
    ],
    ("environment_climate", "express_opinion"): [
        "transit_policy_support",
        "plastic_ban_argument",
        "corporate_greenwashing_critique",
    ],
    ("environment_climate", "mobilize_call_to_action"): [
        "cleanup_volunteer_push",
        "climate_march_invite",
        "city_council_comment_push",
    ],
    ("environment_climate", "debate_argue"): [
        "nuclear_energy_argument",
        "carbon_tax_thread",
        "ev_infrastructure_debate",
    ],
    ("environment_climate", "ask_question"): [
        "composting_question",
        "home_solar_question",
        "public_transit_question",
    ],
    ("personal_life_diary", "narrate_personal_experience"): [
        "burnout_check_in",
        "moving_day_exhaustion",
        "new_job_update",
    ],
    ("personal_life_diary", "vent_rant"): [
        "apartment_noise_rant",
        "commute_breakdown_rant",
        "family_group_chat_rant",
    ],
    ("personal_life_diary", "ask_question"): [
        "pet_adoption_question",
        "travel_packing_question",
        "dinner_idea_request",
    ],
    ("personal_life_diary", "express_opinion"): [
        "adult_friendship_reflection",
        "work_life_balance_take",
        "weekend_recovery_commentary",
    ],
    ("personal_life_diary", "entertain_joke"): [
        "laundry_pile_joke",
        "calendar_overbooking_joke",
        "group_chat_reaction_meme",
    ],
    ("business_startups", "inform_share_news"): [
        "funding_round_update",
        "product_beta_launch",
        "partnership_announcement",
    ],
    ("business_startups", "express_opinion"): [
        "founder_burnout_commentary",
        "growth_at_all_costs_critique",
        "pricing_experiment_take",
    ],
    ("business_startups", "ask_question"): [
        "first_hire_question",
        "pmf_signal_question",
        "churn_reduction_question",
    ],
    ("business_startups", "promote_advertise"): [
        "demo_day_invite",
        "free_trial_launch",
        "case_study_promo",
    ],
    ("business_startups", "narrate_personal_experience"): [
        "missed_revenue_target_reflection",
        "customer_discovery_lesson",
        "first_enterprise_deal_story",
    ],
    ("business_startups", "debate_argue"): [
        "remote_work_policy_argument",
        "bootstrap_vs_venture_thread",
        "ai_moat_debate",
    ],
}

POST_ACT_STANCE_MAPPING: dict[str, list[str]] = {
    "inform_share_news": ["neutral", "mixed", "supportive", "critical"],
    "express_opinion": ["supportive", "critical", "mixed", "neutral"],
    "ask_question": ["neutral", "mixed", "critical"],
    "promote_advertise": ["promotional", "supportive", "neutral"],
    "entertain_joke": ["supportive", "critical", "neutral"],
    "vent_rant": ["critical", "mixed"],
    "mobilize_call_to_action": ["supportive", "critical", "promotional"],
    "debate_argue": ["critical", "supportive", "mixed"],
    "narrate_personal_experience": ["neutral", "mixed", "supportive", "critical"],
}

PLATFORM_AUDIENCE_MAPPING: dict[str, list[str]] = {
    "x_like": [
        "general_public",
        "niche_community",
        "friends_followers",
        "political_group",
        "industry_peers",
        "fans_fandom",
    ],
    "reddit_like": ["niche_community", "general_public", "fans_fandom", "self_directed"],
    "linkedin_like": ["industry_peers", "customers", "general_public", "self_directed"],
    "instagram_caption": ["friends_followers", "fans_fandom", "customers", "general_public"],
    "youtube_comment": ["fans_fandom", "general_public", "niche_community"],
    "facebook_like": ["friends_followers", "general_public", "customers", "political_group"],
}

PLATFORM_FORMALITY_MAPPING: dict[str, list[str]] = {
    "x_like": ["informational_neutral", "casual_conversational", "slang_heavy", "meme_style"],
    "reddit_like": ["informational_neutral", "casual_conversational", "academic_explanatory"],
    "linkedin_like": [
        "very_formal",
        "informational_neutral",
        "professional_pr_tone",
        "academic_explanatory",
    ],
    "instagram_caption": [
        "casual_conversational",
        "slang_heavy",
        "meme_style",
        "professional_pr_tone",
    ],
    "youtube_comment": [
        "casual_conversational",
        "slang_heavy",
        "meme_style",
        "informational_neutral",
    ],
    "facebook_like": [
        "informational_neutral",
        "casual_conversational",
        "very_formal",
        "slang_heavy",
    ],
}

PLATFORM_MODALITY_MAPPING: dict[str, list[str]] = {
    "x_like": [
        "pure_text",
        "text_plus_link",
        "text_plus_image",
        "text_plus_video",
        "poll",
        "thread_starter",
        "reply",
        "quote_post",
    ],
    "reddit_like": ["pure_text", "text_plus_link", "text_plus_image", "text_plus_video", "reply"],
    "linkedin_like": [
        "pure_text",
        "text_plus_link",
        "text_plus_image",
        "text_plus_video",
        "poll",
        "reply",
    ],
    "instagram_caption": ["pure_text", "text_plus_image", "text_plus_video", "reply"],
    "youtube_comment": ["pure_text", "reply"],
    "facebook_like": [
        "pure_text",
        "text_plus_link",
        "text_plus_image",
        "text_plus_video",
        "poll",
        "reply",
    ],
}

SCENARIO_HINT_OVERRIDES: dict[str, tuple[str, ...]] = {
    "pmf_signal_question": ("product market fit", "pmf", "signal"),
    "ev_infrastructure_debate": ("ev", "electric vehicle", "charging"),
    "ai_feature_launch": ("ai", "feature", "launch"),
    "ai_moat_debate": ("ai", "moat", "defensible"),
    "school_board_meeting_push": ("school board", "meeting"),
    "city_council_comment_push": ("city council", "comment period"),
    "contact_representatives_push": ("representatives", "call your rep", "email your rep"),
    "first_enterprise_deal_story": ("enterprise deal", "big customer", "procurement"),
}

TOPIC_DOMAIN_WEIGHTS = {
    "politics": 0.9,
    "economy_finance": 0.85,
    "technology": 1.0,
    "entertainment_pop_culture": 0.85,
    "sports": 0.9,
    "health_fitness": 0.8,
    "education": 0.75,
    "environment_climate": 0.75,
    "personal_life_diary": 0.8,
    "business_startups": 0.7,
}

PLATFORM_WEIGHTS = {
    "x_like": 1.0,
    "reddit_like": 0.9,
    "linkedin_like": 0.65,
    "instagram_caption": 0.8,
    "youtube_comment": 0.6,
    "facebook_like": 0.55,
}

GRANULARITY_LABEL_COMPONENTS: dict[str, dict[str, tuple[str, ...]]] = {
    "content": {
        "coarse": ("topic_domain",),
        "normal": ("topic_domain", "post_act"),
        "fine": ("topic_domain", "post_act", "content_scenario"),
    },
    "channel": {
        "coarse": ("platform",),
        "normal": ("platform", "audience_targeting"),
        "fine": ("platform", "audience_targeting", "content_modality"),
    },
}

GRANULARITY_LABEL_FIELDS: dict[str, dict[str, str]] = {
    "content": {
        "coarse": "content_label_coarse",
        "normal": "content_label_normal",
        "fine": "content_label_fine",
    },
    "channel": {
        "coarse": "channel_label_coarse",
        "normal": "channel_label_normal",
        "fine": "channel_label_fine",
    },
}

SURFACE_METADATA_FIELDS = (
    "platform",
    "topic_domain",
    "post_act",
    "content_scenario",
    "length_style",
    "stance",
    "sentiment_polarity",
    "voice_tone",
    "audience_targeting",
    "formality_style",
    "certainty_confidence",
    "toxicity_civility",
    "content_modality",
    "temporal_orientation",
)


def _granularity_label_token(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not normalized:
        raise ValueError("Granularity label parts must contain at least one alphanumeric token.")
    return normalized


def build_granularity_label(*parts: str) -> str:
    return "__".join(_granularity_label_token(part) for part in parts)


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _content_granularity_values(level: str) -> list[str]:
    if level == "coarse":
        return [build_granularity_label(value) for value in TOPIC_DOMAIN_VALUES]
    labels: list[str] = []
    for topic_domain in TOPIC_DOMAIN_VALUES:
        for post_act in TOPIC_POST_ACT_MAPPING[topic_domain]:
            if level == "normal":
                labels.append(build_granularity_label(topic_domain, post_act))
                continue
            for content_scenario in CONTENT_SCENARIO_MAPPING[(topic_domain, post_act)]:
                labels.append(build_granularity_label(topic_domain, post_act, content_scenario))
    return _dedupe_preserving_order(labels)


def _channel_granularity_values(level: str) -> list[str]:
    if level == "coarse":
        return [build_granularity_label(value) for value in PLATFORM_VALUES]
    labels: list[str] = []
    for platform in PLATFORM_VALUES:
        for audience in PLATFORM_AUDIENCE_MAPPING[platform]:
            if level == "normal":
                labels.append(build_granularity_label(platform, audience))
                continue
            for content_modality in PLATFORM_MODALITY_MAPPING[platform]:
                labels.append(build_granularity_label(platform, audience, content_modality))
    return _dedupe_preserving_order(labels)


def _granularity_labels_for_family(family: str, level: str) -> list[str]:
    if family == "content":
        return _content_granularity_values(level)
    return _channel_granularity_values(level)


def _granularity_schema_fields() -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    for family, label_fields in GRANULARITY_LABEL_FIELDS.items():
        for level, field_name in label_fields.items():
            fields[field_name] = {"values": _granularity_labels_for_family(family, level)}
    return fields


SCHEMA: dict[str, dict[str, Any]] = {
    "platform": {"values": PLATFORM_VALUES},
    "topic_domain": {"values": TOPIC_DOMAIN_VALUES},
    "post_act": {
        "parents": ["topic_domain"],
        "mapping": TOPIC_POST_ACT_MAPPING,
    },
    "content_scenario": {
        "parents": ["topic_domain", "post_act"],
        "mapping": CONTENT_SCENARIO_MAPPING,
    },
    "length_style": {"values": LENGTH_STYLE_VALUES},
    "stance": {
        "parents": ["post_act"],
        "mapping": POST_ACT_STANCE_MAPPING,
    },
    "sentiment_polarity": {"values": SENTIMENT_POLARITY_VALUES},
    "voice_tone": {"values": VOICE_TONE_VALUES},
    "audience_targeting": {
        "parents": ["platform"],
        "mapping": PLATFORM_AUDIENCE_MAPPING,
    },
    "formality_style": {
        "parents": ["platform"],
        "mapping": PLATFORM_FORMALITY_MAPPING,
    },
    "certainty_confidence": {"values": CERTAINTY_CONFIDENCE_VALUES},
    "toxicity_civility": {"values": TOXICITY_CIVILITY_VALUES},
    "content_modality": {
        "parents": ["platform"],
        "mapping": PLATFORM_MODALITY_MAPPING,
    },
    "temporal_orientation": {"values": TEMPORAL_ORIENTATION_VALUES},
}
SCHEMA.update(_granularity_schema_fields())
