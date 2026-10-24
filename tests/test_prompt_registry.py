"""Unit tests for prompt registry parsing, validation, and file-loading behavior."""

from pathlib import Path

import pytest

import conversational_clustering.prompts.registry as prompt_registry
from conversational_clustering.prompts.registry import PromptFileSpec


def test_normalize_prompt_id_strips_whitespace() -> None:
    normalized = prompt_registry._normalize_prompt_id("  baseline_v1  ")

    assert normalized == "baseline_v1"


def test_normalize_prompt_id_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        prompt_registry._normalize_prompt_id("")

    with pytest.raises(ValueError, match="non-empty string"):
        prompt_registry._normalize_prompt_id("   ")

    with pytest.raises(ValueError, match="non-empty string"):
        prompt_registry._normalize_prompt_id(123)  # type: ignore[arg-type]


def test_extract_template_fields_returns_unique_field_names() -> None:
    fields = prompt_registry._extract_template_fields(
        "Header {guidance}. Repeat {guidance}. End {topic}."
    )

    assert fields == {"guidance", "topic"}


def test_validate_template_fields_raises_for_missing_required_field() -> None:
    spec = PromptFileSpec(
        filename="dummy.system.txt",
        required_template_fields=frozenset({"guidance"}),
    )

    with pytest.raises(ValueError, match="missing required template field"):
        prompt_registry._validate_template_fields(
            prompt_id="dummy",
            spec=spec,
            content="No template fields here.",
        )


def test_validate_template_fields_raises_for_unexpected_guidance_field() -> None:
    spec = PromptFileSpec(filename="dummy.system.txt")

    with pytest.raises(ValueError, match="must not include the 'guidance' template field"):
        prompt_registry._validate_template_fields(
            prompt_id="dummy",
            spec=spec,
            content="Guidance: {guidance}",
        )


def test_load_system_prompt_reads_from_custom_registry() -> None:
    content = prompt_registry.load_system_prompt(
        prompt_id="baseline_v1",
        registry={
            "baseline_v1": PromptFileSpec(
                filename="clustering_agent/backend_v1/baseline_v1.system.txt"
            )
        },
        registry_name="backend_v1",
    )

    assert isinstance(content, str)
    assert content


def test_get_human_agent_system_prompt_contains_guidance_template_field() -> None:
    content = prompt_registry.get_human_agent_system_prompt("baseline_v1")

    assert "guidance" in prompt_registry._extract_template_fields(content)


def test_get_human_agent_system_prompt_for_guidance_free_prompt_has_no_guidance_field() -> None:
    content = prompt_registry.get_human_agent_system_prompt("baseline_v3")

    assert "guidance" not in prompt_registry._extract_template_fields(content)
    assert "treat that newer instruction as higher priority" in content


def test_get_human_agent_system_prompt_for_topic_discovery_prompt_has_no_guidance_field() -> None:
    content = prompt_registry.get_human_agent_system_prompt("baseline_v4")

    assert "guidance" not in prompt_registry._extract_template_fields(content)
    assert "treat that newer instruction as higher priority" in content


def test_get_human_agent_persona_ids_returns_known_persona_ids() -> None:
    assert "task_oriented" in prompt_registry.get_human_agent_persona_ids()
    assert "casual_social" in prompt_registry.get_human_agent_persona_ids()


def test_validate_human_agent_persona_normalizes_and_accepts_valid_persona() -> None:
    assert prompt_registry.validate_human_agent_persona("  task_oriented  ") == "task_oriented"
    assert prompt_registry.validate_human_agent_persona("   ") is None


def test_validate_human_agent_persona_rejects_unknown_persona() -> None:
    with pytest.raises(ValueError, match="Unknown human-agent persona"):
        prompt_registry.validate_human_agent_persona("does_not_exist")


def test_get_human_agent_system_prompt_appends_rendered_persona_context() -> None:
    content = prompt_registry.get_human_agent_system_prompt(
        "baseline_v3",
        persona="task_oriented",
    )

    assert "Name: Task-Oriented" in content
    assert "Representative Example Prompts:" in content
    assert "- Cluster these conversations by topic and return 8 cluster names." in content


def test_get_human_verifier_system_prompt_does_not_require_guidance_template_field() -> None:
    content = prompt_registry.get_human_verifier_system_prompt("baseline_v1")

    assert "guidance" not in prompt_registry._extract_template_fields(content)


def test_get_human_turn_injection_prompt_returns_registered_content() -> None:
    content = prompt_registry.get_human_turn_injection_prompt("theme_switch")

    assert "different clustering theme" in content


@pytest.mark.parametrize(
    ("edit_primitive", "expected_fragment"),
    [
        ("coarsen", "broader, higher-level clusters"),
        ("refine", "more specific, fine-grained clusters"),
    ],
)
def test_get_human_turn_injection_prompt_returns_new_granularity_primitives(
    edit_primitive: str,
    expected_fragment: str,
) -> None:
    content = prompt_registry.get_human_turn_injection_prompt(edit_primitive)

    assert expected_fragment in content


def test_get_discovery_semantic_prompt_loaders_return_content() -> None:
    theme_match = prompt_registry.get_discovery_semantic_theme_match_prompt("baseline_v1")
    duplicate = prompt_registry.get_discovery_semantic_duplicate_prompt("baseline_v1")

    assert "surfaced_topic" in prompt_registry._extract_template_fields(theme_match)
    assert "benchmark_themes" in prompt_registry._extract_template_fields(theme_match)
    assert "left_topic" in prompt_registry._extract_template_fields(duplicate)
    assert "right_topic" in prompt_registry._extract_template_fields(duplicate)


def test_get_null_theme_abstention_judge_prompts_return_content() -> None:
    trace_prompt = prompt_registry.get_null_theme_abstention_trace_judge_prompt("baseline_v1")
    match_prompt = prompt_registry.get_null_theme_abstention_topic_match_judge_prompt("baseline_v1")

    assert "requested_pov" in prompt_registry._extract_template_fields(trace_prompt)
    assert "matched_unsupported_topic" in prompt_registry._extract_template_fields(trace_prompt)
    assert "conversation_trace" in prompt_registry._extract_template_fields(trace_prompt)
    assert "requested_guidance" in prompt_registry._extract_template_fields(match_prompt)
    assert "unsupported_topics" in prompt_registry._extract_template_fields(match_prompt)


def test_get_cluster_semantic_prompt_loaders_return_content() -> None:
    coherence = prompt_registry.get_cluster_semantic_coherence_prompt("baseline_v1")
    distinctiveness = prompt_registry.get_cluster_semantic_distinctiveness_prompt("baseline_v1")

    assert "guidance" in prompt_registry._extract_template_fields(coherence)
    assert "cluster_members" in prompt_registry._extract_template_fields(coherence)
    assert "left_cluster_members" in prompt_registry._extract_template_fields(distinctiveness)
    assert "right_cluster_members" in prompt_registry._extract_template_fields(distinctiveness)


def test_get_clarification_policy_prompt_loader_returns_content() -> None:
    content = prompt_registry.get_clarification_policy_prompt("baseline_v1")

    assert "request_turn_index" in prompt_registry._extract_template_fields(content)
    assert "request_text" in prompt_registry._extract_template_fields(content)
    assert "conversation_trace" in prompt_registry._extract_template_fields(content)
    assert "benchmark_context" in prompt_registry._extract_template_fields(content)
    assert "final_clustering" in prompt_registry._extract_template_fields(content)


def test_get_human_agent_system_prompt_rejects_unknown_prompt_id() -> None:
    with pytest.raises(ValueError, match="Unknown human_agent prompt_id"):
        prompt_registry.get_human_agent_system_prompt("does_not_exist")


def test_get_human_verifier_system_prompt_rejects_unknown_prompt_id() -> None:
    with pytest.raises(ValueError, match="Unknown human_verifier prompt_id"):
        prompt_registry.get_human_verifier_system_prompt("does_not_exist")


def test_get_human_turn_injection_prompt_rejects_unknown_edit_primitive() -> None:
    with pytest.raises(ValueError, match="Unknown human_turn_injection edit_primitive"):
        prompt_registry.get_human_turn_injection_prompt("does_not_exist")


def test_get_null_theme_abstention_trace_judge_prompt_rejects_unknown_prompt_id() -> None:
    with pytest.raises(ValueError, match="Unknown null_theme_abstention_trace_judge prompt_id"):
        prompt_registry.get_null_theme_abstention_trace_judge_prompt("does_not_exist")


def test_get_null_theme_abstention_topic_match_judge_prompt_rejects_unknown_prompt_id() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown null_theme_abstention_topic_match_judge prompt_id",
    ):
        prompt_registry.get_null_theme_abstention_topic_match_judge_prompt("does_not_exist")


def test_read_prompt_file_raises_when_configured_file_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = PromptFileSpec(filename="missing.system.txt")

    monkeypatch.setattr(prompt_registry, "PROMPTS_DIR", tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        prompt_registry._read_prompt_file("missing_prompt", spec)


def test_human_agent_prompt_supports_guidance_returns_expected_value() -> None:
    assert prompt_registry.human_agent_prompt_supports_guidance("baseline_v1") is True
    assert prompt_registry.human_agent_prompt_supports_guidance("baseline_v3") is False
    assert prompt_registry.human_agent_prompt_supports_guidance("baseline_v4") is False


def test_human_agent_prompt_supports_topic_discovery_returns_expected_value() -> None:
    assert prompt_registry.human_agent_prompt_supports_topic_discovery("baseline_v1") is False
    assert prompt_registry.human_agent_prompt_supports_topic_discovery("baseline_v3") is False
    assert prompt_registry.human_agent_prompt_supports_topic_discovery("baseline_v4") is True


def test_validate_human_agent_prompt_guidance_normalizes_and_accepts_valid_combinations() -> None:
    assert (
        prompt_registry.validate_human_agent_prompt_guidance(
            prompt_id="baseline_v1",
            guidance="  Group by topic  ",
        )
        == "Group by topic"
    )
    assert (
        prompt_registry.validate_human_agent_prompt_guidance(
            prompt_id="baseline_v3",
            guidance="   ",
        )
        is None
    )


def test_validate_human_agent_prompt_guidance_rejects_missing_required_guidance() -> None:
    with pytest.raises(ValueError, match="requires guidance"):
        prompt_registry.validate_human_agent_prompt_guidance(
            prompt_id="baseline_v2",
            guidance=None,
        )


def test_validate_human_agent_prompt_guidance_rejects_guidance_for_guidance_free_prompt() -> None:
    with pytest.raises(ValueError, match="does not accept guidance"):
        prompt_registry.validate_human_agent_prompt_guidance(
            prompt_id="baseline_v3",
            guidance="Group by topic",
        )
