"""Tests for the social media dataset preset."""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

from dataset_generation.cli import build_parser
from dataset_generation.presets.social_media import (
    PRESET,
    sampling,
    spec,
    text_generation,
)
from dataset_generation.schema import prompt_safe_schema


def _sample_record(seed: int = 7) -> dict[str, str]:
    rng = random.Random(seed)
    target = PRESET.reachable_coverage_targets()[0]
    record = sampling.sample_record_for_topic_target(target, rng, difficulty="medium")
    sampling.validate_record(record)
    return record


def test_social_media_schema_exposes_hierarchical_label_fields() -> None:
    assert "content_scenario" in spec.SCHEMA
    assert "length_style" in spec.SCHEMA
    assert "content_label_coarse" in spec.SCHEMA
    assert "content_label_fine" in spec.SCHEMA
    assert "channel_label_fine" in spec.SCHEMA
    assert isinstance(next(iter(spec.SCHEMA["content_scenario"]["mapping"])), tuple)
    assert spec.GRANULARITY_LABEL_FIELDS["content"]["fine"] == "content_label_fine"
    assert spec.GRANULARITY_LABEL_FIELDS["channel"]["normal"] == "channel_label_normal"


def test_prompt_safe_schema_serializes_tuple_key_mappings_for_content_scenario() -> None:
    prompt_schema = prompt_safe_schema(spec.SCHEMA, ("content_scenario",))
    mapping = prompt_schema["content_scenario"]["mapping"]

    assert isinstance(mapping, list)
    assert set(mapping[0]["parents"]) == {"topic_domain", "post_act"}
    assert isinstance(mapping[0]["values"], list)
    assert mapping[0]["values"]


def test_sampling_applies_granularity_labels_consistently() -> None:
    record = _sample_record()
    label_suites = sampling.build_label_suites(record)

    assert record["content_label_coarse"] == label_suites["content"]["coarse"]
    assert record["content_label_normal"] == label_suites["content"]["normal"]
    assert record["content_label_fine"] == label_suites["content"]["fine"]
    assert record["channel_label_coarse"] == label_suites["channel"]["coarse"]
    assert record["channel_label_normal"] == label_suites["channel"]["normal"]
    assert record["channel_label_fine"] == label_suites["channel"]["fine"]


def test_generate_records_balanced_covers_each_topic_once_when_enough_rows() -> None:
    records = sampling.generate_records(
        len(spec.TOPIC_DOMAIN_VALUES),
        random.Random(7),
        mode="balanced",
        difficulty="medium",
        min_coverage_count=None,
    )

    observed = {record["content_label_coarse"] for record in records}
    assert observed == set(PRESET.reachable_coverage_targets())


def test_generate_records_for_topic_targets_respects_requested_targets() -> None:
    targets = PRESET.reachable_coverage_targets()[:3]
    records = sampling.generate_records_for_topic_targets(
        targets,
        random.Random(11),
        difficulty="medium",
    )

    assert [record["content_label_coarse"] for record in records] == targets


def test_build_row_payload_includes_surface_metadata_cluster_labels_and_controls() -> None:
    record = _sample_record()
    payload = PRESET.build_row_payload(
        record,
        0,
        random.Random(7),
        "medium",
        entity_mention_probability=0.5,
    )

    assert payload["row_id"] == 0
    assert payload["surface_metadata"]["content_scenario"] == record["content_scenario"]
    assert payload["surface_metadata"]["length_style"] == record["length_style"]
    assert payload["cluster_labels"]["content_fine"] == record["content_label_fine"]
    assert payload["hidden_generation_guidance"]
    assert payload["generation_controls"]["opening_style"]


def test_build_prompt_mentions_content_scenario_and_cluster_labels() -> None:
    record = _sample_record()
    payload = PRESET.build_row_payload(
        record,
        0,
        random.Random(7),
        "medium",
        entity_mention_probability=None,
    )
    prompt = PRESET.build_prompt([payload], "medium")

    assert "content_scenario" in prompt
    assert "cluster_labels" in prompt
    assert "thread_starter" in prompt or "text_plus_link" in prompt or "reply" in prompt


def test_deterministic_evaluation_flags_modality_mismatch() -> None:
    record = _sample_record()
    record["content_modality"] = "text_plus_image"
    record["channel_label_fine"] = spec.build_granularity_label(
        record["platform"],
        record["audience_targeting"],
        record["content_modality"],
    )
    scenario_phrase = text_generation.preferred_content_scenario_phrase(record["content_scenario"])
    text = f"Still thinking about {scenario_phrase}."

    result = PRESET.evaluate_row_deterministically(record, text)

    assert result.rating == "need_attention"
    assert "content_modality_mismatch" in result.reasons


def test_deterministic_evaluation_flags_scenario_mismatch_as_need_attention() -> None:
    record = _sample_record()
    result = PRESET.evaluate_row_deterministically(
        record,
        "Still thinking about grocery bills and hidden airline fees all week.",
    )

    assert result.rating == "need_attention"
    assert "content_scenario_mismatch" in result.reasons


def test_collect_alignment_failures_flags_scenario_mismatch_as_soft_failure() -> None:
    record = _sample_record()

    hard_failures, soft_failures = text_generation.collect_alignment_failures(
        [record],
        {0: "Still thinking about grocery bills and hidden airline fees all week."},
    )

    assert hard_failures == {}
    assert 0 in soft_failures
    assert "content_scenario" in str(soft_failures[0])


def test_deterministic_evaluation_flags_length_mismatch_as_need_attention() -> None:
    record = _sample_record()
    record["length_style"] = "threadish"
    result = PRESET.evaluate_row_deterministically(record, "Short update on this.")

    assert result.rating == "need_attention"
    assert "length_style_mismatch" in result.reasons


def test_build_parser_uses_social_media_defaults() -> None:
    parser = build_parser("social_media")
    args = parser.parse_args([])

    assert args.output == "data/social_media_dataset.csv"
    assert args.evaluate_and_regenerate is True


def test_generic_parser_accepts_social_media_preset() -> None:
    parser = build_parser()
    args = parser.parse_args(["--preset", "social_media", "--rows", "10"])

    assert args.preset == "social_media"
    assert args.rows == 10


def test_module_generate_social_media_dataset_help_runs_from_repo_root() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "dataset_generation.presets.social_media", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--mode {balanced,random}" in completed.stdout
