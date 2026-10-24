"""Taxonomy and schema definitions for the restaurant review preset."""

from __future__ import annotations

RESTAURANT_CATEGORY_VALUES = (
    "cafe_bakery",
    "fast_food",
    "casual_dining",
    "fine_dining",
    "pizza_place",
    "sushi_spot",
)

VISIT_MODE_MAPPING = {
    "cafe_bakery": ("dine_in", "takeout", "delivery"),
    "fast_food": ("dine_in", "takeout", "delivery", "drive_through"),
    "casual_dining": ("dine_in", "takeout", "delivery"),
    "fine_dining": ("dine_in",),
    "pizza_place": ("dine_in", "takeout", "delivery"),
    "sushi_spot": ("dine_in", "takeout", "delivery"),
}

PRIMARY_FOCUS_VALUES = (
    "food_quality",
    "service",
    "ambience",
    "value",
    "speed",
    "cleanliness",
)

RATING_SCORE_VALUES = tuple(str(score) for score in range(1, 11))
SENTIMENT_POLARITY_VALUES = (
    "very_negative",
    "negative",
    "mixed",
    "positive",
    "very_positive",
)

LENGTH_STYLE_VALUES = (
    "short",
    "medium",
    "detailed",
)

RATING_TO_SENTIMENT = {
    "1": "very_negative",
    "2": "very_negative",
    "3": "negative",
    "4": "negative",
    "5": "mixed",
    "6": "mixed",
    "7": "positive",
    "8": "positive",
    "9": "very_positive",
    "10": "very_positive",
}

REVIEW_SCENARIO_MAPPING = {
    ("food_quality", "very_negative"): (
        "overcooked_food",
        "bland_food",
        "stale_food",
    ),
    ("food_quality", "negative"): (
        "soggy_food",
        "uneven_flavors",
        "forgettable_food",
    ),
    ("food_quality", "mixed"): (
        "good_main_weak_sides",
        "tasty_but_inconsistent",
        "great_dessert_average_food",
    ),
    ("food_quality", "positive"): (
        "fresh_flavors",
        "well_cooked_meal",
        "standout_dessert",
    ),
    ("food_quality", "very_positive"): (
        "memorable_dish",
        "best_meal_lately",
        "perfectly_cooked_food",
    ),
    ("service", "very_negative"): (
        "rude_staff",
        "forgotten_order",
        "ignored_table",
    ),
    ("service", "negative"): (
        "slow_service",
        "order_mistake",
        "rushed_service",
    ),
    ("service", "mixed"): (
        "friendly_but_slow",
        "good_recovery_after_mistake",
        "kind_but_overwhelmed",
    ),
    ("service", "positive"): (
        "friendly_staff",
        "attentive_service",
        "smooth_service",
    ),
    ("service", "very_positive"): (
        "above_and_beyond_service",
        "warm_welcome",
        "flawless_service",
    ),
    ("ambience", "very_negative"): (
        "too_loud_room",
        "unpleasant_smell",
        "cramped_seating",
    ),
    ("ambience", "negative"): (
        "dated_decor",
        "harsh_lighting",
        "noisy_room",
    ),
    ("ambience", "mixed"): (
        "cozy_but_crowded",
        "stylish_but_loud",
        "clean_but_generic",
    ),
    ("ambience", "positive"): (
        "cozy_atmosphere",
        "comfortable_seating",
        "relaxed_vibe",
    ),
    ("ambience", "very_positive"): (
        "beautiful_space",
        "perfect_date_night_vibe",
        "great_ambience",
    ),
    ("value", "very_negative"): (
        "wildly_overpriced",
        "tiny_portions",
        "hidden_fees",
    ),
    ("value", "negative"): (
        "overpriced_for_quality",
        "small_portions",
        "not_worth_the_price",
    ),
    ("value", "mixed"): (
        "good_food_but_pricey",
        "fair_price_mixed_quality",
        "decent_deal_with_caveats",
    ),
    ("value", "positive"): (
        "worth_the_price",
        "generous_portions",
        "solid_lunch_value",
    ),
    ("value", "very_positive"): (
        "excellent_value",
        "big_portions_for_price",
        "bargain_find",
    ),
    ("speed", "very_negative"): (
        "extremely_long_wait",
        "delivery_took_forever",
        "cold_food_after_wait",
    ),
    ("speed", "negative"): (
        "slow_counter_line",
        "delayed_takeout",
        "long_wait_for_table",
    ),
    ("speed", "mixed"): (
        "quick_order_slow_food",
        "worth_the_wait",
        "fast_start_slow_finish",
    ),
    ("speed", "positive"): (
        "quick_service",
        "fast_takeout_ready",
        "smooth_turnaround",
    ),
    ("speed", "very_positive"): (
        "surprisingly_fast",
        "lightning_fast_delivery",
        "in_and_out_easy",
    ),
    ("cleanliness", "very_negative"): (
        "dirty_tableware",
        "sticky_tables",
        "messy_restroom",
    ),
    ("cleanliness", "negative"): (
        "untidy_counter",
        "neglected_dining_area",
        "dusty_corners",
    ),
    ("cleanliness", "mixed"): (
        "clean_tables_but_restroom_issue",
        "mostly_clean_one_issue",
        "front_clean_back_messy",
    ),
    ("cleanliness", "positive"): (
        "clean_restroom",
        "tidy_space",
        "spotless_counter",
    ),
    ("cleanliness", "very_positive"): (
        "sparkling_clean",
        "immaculate_dining_room",
        "very_well_kept",
    ),
}

REVIEW_SCENARIO_HINT_OVERRIDES = {
    "best_meal_lately": ("best meal", "best thing I have had"),
    "perfect_date_night_vibe": ("date night", "great mood"),
    "not_worth_the_price": ("not worth the price", "not worth it"),
    "delivery_took_forever": ("delivery took forever", "forever to arrive"),
    "in_and_out_easy": ("in and out", "super quick"),
    "mostly_clean_one_issue": ("mostly clean", "one issue"),
    "front_clean_back_messy": ("front looked clean", "back felt messy"),
}

RESTAURANT_CATEGORY_WEIGHTS = {
    "cafe_bakery": 0.16,
    "fast_food": 0.16,
    "casual_dining": 0.22,
    "fine_dining": 0.1,
    "pizza_place": 0.18,
    "sushi_spot": 0.18,
}

PRIMARY_FOCUS_WEIGHTS = {
    "food_quality": 0.26,
    "service": 0.18,
    "ambience": 0.12,
    "value": 0.18,
    "speed": 0.14,
    "cleanliness": 0.12,
}

RATING_SCORE_WEIGHTS = {
    "1": 0.03,
    "2": 0.04,
    "3": 0.06,
    "4": 0.08,
    "5": 0.1,
    "6": 0.14,
    "7": 0.18,
    "8": 0.18,
    "9": 0.12,
    "10": 0.07,
}

SURFACE_METADATA_FIELDS = (
    "restaurant_category",
    "visit_mode",
    "primary_focus",
    "rating_score",
    "sentiment_polarity",
    "review_scenario",
    "length_style",
)

GRANULARITY_LABEL_COMPONENTS = {
    "review": {
        "coarse": ("primary_focus",),
        "normal": ("primary_focus", "sentiment_polarity"),
        "fine": ("primary_focus", "sentiment_polarity", "review_scenario"),
    }
}

GRANULARITY_LABEL_FIELDS = {
    "review": {
        "coarse": "review_label_coarse",
        "normal": "review_label_normal",
        "fine": "review_label_fine",
    }
}


def build_granularity_label(*parts: str) -> str:
    return "__".join(parts)


REVIEW_LABEL_COARSE_VALUES = tuple(
    build_granularity_label(primary_focus) for primary_focus in PRIMARY_FOCUS_VALUES
)
REVIEW_LABEL_NORMAL_VALUES = tuple(
    build_granularity_label(primary_focus, sentiment_polarity)
    for primary_focus in PRIMARY_FOCUS_VALUES
    for sentiment_polarity in SENTIMENT_POLARITY_VALUES
)
REVIEW_LABEL_FINE_VALUES = tuple(
    build_granularity_label(primary_focus, sentiment_polarity, review_scenario)
    for primary_focus in PRIMARY_FOCUS_VALUES
    for sentiment_polarity in SENTIMENT_POLARITY_VALUES
    for review_scenario in REVIEW_SCENARIO_MAPPING[(primary_focus, sentiment_polarity)]
)

SCHEMA = {
    "restaurant_category": {"values": RESTAURANT_CATEGORY_VALUES},
    "visit_mode": {
        "parents": ["restaurant_category"],
        "mapping": VISIT_MODE_MAPPING,
    },
    "primary_focus": {"values": PRIMARY_FOCUS_VALUES},
    "rating_score": {"values": RATING_SCORE_VALUES},
    "sentiment_polarity": {
        "parents": ["rating_score"],
        "mapping": {
            rating_score: (sentiment_polarity,)
            for rating_score, sentiment_polarity in RATING_TO_SENTIMENT.items()
        },
    },
    "review_scenario": {
        "parents": ["primary_focus", "sentiment_polarity"],
        "mapping": REVIEW_SCENARIO_MAPPING,
    },
    "length_style": {"values": LENGTH_STYLE_VALUES},
    "review_label_coarse": {"values": REVIEW_LABEL_COARSE_VALUES},
    "review_label_normal": {"values": REVIEW_LABEL_NORMAL_VALUES},
    "review_label_fine": {"values": REVIEW_LABEL_FINE_VALUES},
}
