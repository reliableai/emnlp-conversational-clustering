"""Tests for the restaurant review dataset preset."""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

from dataset_generation.cli import build_parser
from dataset_generation.presets.restaurant_review import PRESET, sampling, spec, text_generation
from dataset_generation.schema import prompt_safe_schema


def _sample_record(seed: int = 7) -> dict[str, str]:
    rng = random.Random(seed)
    target = PRESET.reachable_coverage_targets()[0]
    record = sampling.sample_record_for_focus_target(target, rng, difficulty="easy")
    sampling.validate_record(record)
    return record


def test_restaurant_review_schema_exposes_hierarchical_label_fields() -> None:
    assert "review_scenario" in spec.SCHEMA
    assert "length_style" in spec.SCHEMA
    assert "review_label_coarse" in spec.SCHEMA
    assert "review_label_fine" in spec.SCHEMA
    assert isinstance(next(iter(spec.SCHEMA["review_scenario"]["mapping"])), tuple)
    assert spec.GRANULARITY_LABEL_FIELDS["review"]["fine"] == "review_label_fine"


def test_prompt_safe_schema_serializes_tuple_key_mappings_for_review_scenario() -> None:
    prompt_schema = prompt_safe_schema(spec.SCHEMA, ("review_scenario",))
    mapping = prompt_schema["review_scenario"]["mapping"]

    assert isinstance(mapping, list)
    assert set(mapping[0]["parents"]) == {"primary_focus", "sentiment_polarity"}
    assert isinstance(mapping[0]["values"], list)
    assert mapping[0]["values"]


def test_sampling_applies_granularity_labels_consistently() -> None:
    record = _sample_record()
    label_suites = sampling.build_label_suites(record)

    assert record["review_label_coarse"] == label_suites["review"]["coarse"]
    assert record["review_label_normal"] == label_suites["review"]["normal"]
    assert record["review_label_fine"] == label_suites["review"]["fine"]


def test_generate_records_balanced_covers_each_focus_once_when_enough_rows() -> None:
    records = sampling.generate_records(
        len(spec.PRIMARY_FOCUS_VALUES),
        random.Random(7),
        mode="balanced",
        difficulty="easy",
        min_coverage_count=None,
    )

    observed = {record["review_label_coarse"] for record in records}
    assert observed == set(PRESET.reachable_coverage_targets())


def test_generate_records_for_focus_targets_respects_requested_targets() -> None:
    targets = PRESET.reachable_coverage_targets()[:3]
    records = sampling.generate_records_for_focus_targets(
        targets,
        random.Random(11),
        difficulty="easy",
    )

    assert [record["review_label_coarse"] for record in records] == targets


def test_build_row_payload_includes_surface_metadata_cluster_labels_and_controls() -> None:
    record = _sample_record()
    payload = PRESET.build_row_payload(
        record,
        0,
        random.Random(7),
        "easy",
        entity_mention_probability=None,
    )

    assert payload["row_id"] == 0
    assert payload["surface_metadata"]["review_scenario"] == record["review_scenario"]
    assert payload["surface_metadata"]["length_style"] == record["length_style"]
    assert payload["cluster_labels"]["review_fine"] == record["review_label_fine"]
    assert payload["hidden_generation_guidance"]
    assert payload["generation_controls"]["opening_style"]


def test_build_prompt_mentions_review_scenario_and_rating_score() -> None:
    record = _sample_record()
    payload = PRESET.build_row_payload(
        record,
        0,
        random.Random(7),
        "easy",
        entity_mention_probability=None,
    )
    prompt = PRESET.build_prompt([payload], "easy")

    assert "review_scenario" in prompt
    assert "rating_score" in prompt
    assert "cluster_labels" in prompt


def test_deterministic_evaluation_flags_scenario_mismatch_as_need_attention() -> None:
    record = _sample_record()
    result = PRESET.evaluate_row_deterministically(
        record,
        "I stopped by yesterday, and I do not have any specific standout detail to mention.",
    )

    assert result.rating == "need_attention"
    assert "review_scenario_mismatch" in result.reasons


def test_collect_alignment_failures_flags_scenario_mismatch_as_soft_failure() -> None:
    record = _sample_record()

    hard_failures, soft_failures = text_generation.collect_alignment_failures(
        [record],
        {0: "I stopped by yesterday, and I do not have any specific standout detail to mention."},
    )

    assert hard_failures == {}
    assert 0 in soft_failures
    assert "review_scenario" in str(soft_failures[0])


def test_deterministic_evaluation_flags_extreme_rating_contradiction() -> None:
    record = _sample_record()
    record["rating_score"] = "1"
    record["sentiment_polarity"] = spec.RATING_TO_SENTIMENT["1"]
    record["review_scenario"] = spec.REVIEW_SCENARIO_MAPPING[
        (record["primary_focus"], record["sentiment_polarity"])
    ][0]
    record["review_label_normal"] = spec.build_granularity_label(
        record["primary_focus"],
        record["sentiment_polarity"],
    )
    record["review_label_fine"] = spec.build_granularity_label(
        record["primary_focus"],
        record["sentiment_polarity"],
        record["review_scenario"],
    )

    result = PRESET.evaluate_row_deterministically(
        record,
        "Amazing meal, perfect service, and I loved every second of it.",
    )

    assert result.rating == "need_attention"
    assert "extreme_rating_contradiction" in result.reasons


def test_deterministic_evaluation_flags_length_mismatch() -> None:
    record = _sample_record()
    record["length_style"] = "detailed"
    result = PRESET.evaluate_row_deterministically(record, "Good food, fast service.")

    assert result.rating == "need_attention"
    assert "length_style_mismatch" in result.reasons


def test_build_parser_uses_restaurant_review_defaults() -> None:
    parser = build_parser("restaurant_review")
    args = parser.parse_args([])

    assert args.output == "data/restaurant_review_dataset.csv"
    assert args.evaluate_and_regenerate is True


def test_generic_parser_accepts_restaurant_review_preset() -> None:
    parser = build_parser()
    args = parser.parse_args(["--preset", "restaurant_review", "--rows", "10"])

    assert args.preset == "restaurant_review"
    assert args.rows == 10


def test_module_generate_restaurant_review_dataset_help_runs_from_repo_root() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "dataset_generation.presets.restaurant_review", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--mode {balanced,random}" in completed.stdout
