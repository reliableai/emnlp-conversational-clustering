"""Unit tests for human persona prompt asset structure and template alignment."""

import json
from pathlib import Path

import conversational_clustering.prompts.registry as prompt_registry

PERSONA_DIR = (
    Path(__file__).resolve().parents[1]
    / "conversational_clustering"
    / "prompts"
    / "human_agent"
    / "personas"
)
PERSONAS_PATH = PERSONA_DIR / "personas.json"
PERSONA_CONTEXT_PATH = PERSONA_DIR / "persona_context.txt"
REQUIRED_PERSONA_KEYS = {
    "id",
    "clarity",
    "conversation_style",
    "example_prompts",
    "intent_stability",
    "name",
    "response_length",
    "topic_specificity",
    "user_profile",
}
PERSONA_TEMPLATE_KEYS = REQUIRED_PERSONA_KEYS - {"id"}


def test_persona_assets_exist_in_personas_subdirectory() -> None:
    assert PERSONAS_PATH.exists()
    assert PERSONA_CONTEXT_PATH.exists()


def test_personas_json_entries_match_revised_schema() -> None:
    personas = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))

    assert isinstance(personas, list)
    assert personas

    for persona in personas:
        assert set(persona) == REQUIRED_PERSONA_KEYS
        assert persona["id"].strip()
        assert persona["name"].strip()
        assert persona["user_profile"].strip()
        assert persona["conversation_style"].strip()
        assert persona["response_length"].strip()
        assert isinstance(persona["example_prompts"], list)
        assert persona["example_prompts"]
        assert all(
            isinstance(prompt, str) and prompt.strip() for prompt in persona["example_prompts"]
        )


def test_persona_context_placeholders_match_revised_schema() -> None:
    template = PERSONA_CONTEXT_PATH.read_text(encoding="utf-8")

    assert prompt_registry._extract_template_fields(template) == PERSONA_TEMPLATE_KEYS
