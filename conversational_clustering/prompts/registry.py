"""Prompt registry for loading versioned agent prompts and persona prompt assets."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import TypedDict

PROMPTS_DIR = Path(__file__).resolve().parent

_HUMAN_PERSONA_FILENAME = "human_agent/personas/personas.json"
_HUMAN_PERSONA_CONTEXT_FILENAME = "human_agent/personas/persona_context.txt"
_HUMAN_PERSONA_ID_PATTERN = re.compile(r"[a-z0-9_]+")
_HUMAN_TURN_INJECTION_PROMPTS_FILENAME = "human_turn_injection/primitives.json"
_HUMAN_PERSONA_REQUIRED_FIELDS = frozenset(
    {
        "id",
        "name",
        "user_profile",
        "conversation_style",
        "example_prompts",
        "clarity",
        "intent_stability",
        "topic_specificity",
        "response_length",
    }
)
_HUMAN_PERSONA_TEMPLATE_FIELDS = frozenset(_HUMAN_PERSONA_REQUIRED_FIELDS - {"id"})


class HumanPersona(TypedDict):
    id: str
    name: str
    user_profile: str
    conversation_style: str
    example_prompts: list[str]
    clarity: str
    intent_stability: str
    topic_specificity: str
    response_length: str


@dataclass(frozen=True)
class PromptFileSpec:
    filename: str
    required_template_fields: frozenset[str] = frozenset()
    supports_topic_discovery: bool = False

    def supports_guidance(self) -> bool:
        return "guidance" in self.required_template_fields


_HUMAN_AGENT_PROMPT_FILE_REGISTRY: dict[str, PromptFileSpec] = {
    "baseline_v1": PromptFileSpec(
        filename="human_agent/baseline_v1.system.txt",
        required_template_fields=frozenset({"guidance"}),
    ),
    "baseline_v2": PromptFileSpec(
        filename="human_agent/baseline_v2.system.txt",
        required_template_fields=frozenset({"guidance"}),
    ),
    "baseline_v3": PromptFileSpec(filename="human_agent/baseline_v3.system.txt"),
    "baseline_v4": PromptFileSpec(
        filename="human_agent/baseline_v4.system.txt",
        supports_topic_discovery=True,
    ),
}

_HUMAN_VERIFIER_PROMPT_FILE_REGISTRY: dict[str, PromptFileSpec] = {
    "baseline_v1": PromptFileSpec(
        filename="human_verifier/baseline_v1.system.txt",
    )
}

_SEMANTIC_CONSISTENCY_INFERENCE_PROMPT_FILE_REGISTRY: dict[str, PromptFileSpec] = {
    "baseline_v1": PromptFileSpec(
        filename="semantic_consistency/inference/baseline_v1.txt",
        required_template_fields=frozenset({"clustering"}),
    ),
}

_SEMANTIC_CONSISTENCY_JUDGE_PROMPT_FILE_REGISTRY: dict[str, PromptFileSpec] = {
    "baseline_v1": PromptFileSpec(
        filename="semantic_consistency/judge/baseline_v1.txt",
        required_template_fields=frozenset({"actual_guidance", "inferred_guidance"}),
    ),
}

_DISCOVERY_SEMANTIC_THEME_MATCH_PROMPT_FILE_REGISTRY: dict[str, PromptFileSpec] = {
    "baseline_v1": PromptFileSpec(
        filename="evaluation/discovery_semantic_quality_theme_match/baseline_v1.txt",
        required_template_fields=frozenset({"surfaced_topic", "benchmark_themes"}),
    ),
}

_DISCOVERY_SEMANTIC_DUPLICATE_PROMPT_FILE_REGISTRY: dict[str, PromptFileSpec] = {
    "baseline_v1": PromptFileSpec(
        filename="evaluation/discovery_semantic_quality_duplicate/baseline_v1.txt",
        required_template_fields=frozenset({"left_topic", "right_topic"}),
    ),
}

_NULL_THEME_ABSTENTION_TRACE_JUDGE_PROMPT_FILE_REGISTRY: dict[str, PromptFileSpec] = {
    "baseline_v1": PromptFileSpec(
        filename="evaluation/null_theme_abstention_trace_judge/baseline_v1.txt",
        required_template_fields=frozenset(
            {"requested_pov", "matched_unsupported_topic", "conversation_trace"}
        ),
    ),
}

_NULL_THEME_ABSTENTION_TOPIC_MATCH_JUDGE_PROMPT_FILE_REGISTRY: dict[str, PromptFileSpec] = {
    "baseline_v1": PromptFileSpec(
        filename="evaluation/null_theme_abstention_topic_match_judge/baseline_v1.txt",
        required_template_fields=frozenset({"requested_guidance", "unsupported_topics"}),
    ),
}

_CLUSTER_SEMANTIC_COHERENCE_PROMPT_FILE_REGISTRY: dict[str, PromptFileSpec] = {
    "baseline_v1": PromptFileSpec(
        filename="evaluation/cluster_semantic_quality_coherence/baseline_v1.txt",
        required_template_fields=frozenset({"guidance", "cluster_name", "cluster_members"}),
    ),
}

_CLUSTER_SEMANTIC_DISTINCTIVENESS_PROMPT_FILE_REGISTRY: dict[str, PromptFileSpec] = {
    "baseline_v1": PromptFileSpec(
        filename="evaluation/cluster_semantic_quality_distinctiveness/baseline_v1.txt",
        required_template_fields=frozenset(
            {
                "guidance",
                "left_cluster_name",
                "left_cluster_members",
                "right_cluster_name",
                "right_cluster_members",
            }
        ),
    ),
}

_CLARIFICATION_POLICY_PROMPT_FILE_REGISTRY: dict[str, PromptFileSpec] = {
    "baseline_v1": PromptFileSpec(
        filename="evaluation/clarification_policy/baseline_v1.txt",
        required_template_fields=frozenset(
            {
                "request_turn_index",
                "request_text",
                "conversation_trace",
                "benchmark_context",
                "final_clustering",
            }
        ),
    ),
}


def _normalize_prompt_id(prompt_id: str) -> str:
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise ValueError("prompt_id must be a non-empty string")
    return prompt_id.strip()


def _extract_template_fields(template: str) -> set[str]:
    fields: set[str] = set()
    formatter = Formatter()
    for _literal_text, field_name, _format_spec, _conversion in formatter.parse(template):
        if field_name:
            fields.add(field_name)
    return fields


def _validate_template_fields(
    *,
    prompt_id: str,
    spec: PromptFileSpec,
    content: str,
) -> None:
    template_fields = _extract_template_fields(content)
    required_fields = spec.required_template_fields
    missing_required = sorted(required_fields - template_fields)
    if missing_required:
        missing = ", ".join(missing_required)
        raise ValueError(
            f"Prompt file '{spec.filename}' for prompt ID '{prompt_id}' is missing required "
            f"template field(s): {missing}"
        )
    if not spec.supports_guidance() and "guidance" in template_fields:
        raise ValueError(
            f"Prompt file '{spec.filename}' for prompt ID '{prompt_id}' must not include the "
            "'guidance' template field"
        )


def _read_text_file(path: Path, *, description: str) -> str:
    if not path.exists():
        raise ValueError(f"{description} '{path.relative_to(PROMPTS_DIR)}' does not exist")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"{description} '{path.relative_to(PROMPTS_DIR)}' is empty")
    return content


def _read_prompt_file(prompt_id: str, spec: PromptFileSpec) -> str:
    path = PROMPTS_DIR / spec.filename
    content = _read_text_file(path, description="Prompt file")
    _validate_template_fields(prompt_id=prompt_id, spec=spec, content=content)
    return content


def _get_prompt_spec(
    *,
    prompt_id: str,
    registry: Mapping[str, PromptFileSpec],
    registry_name: str,
) -> tuple[str, PromptFileSpec]:
    normalized_prompt_id = _normalize_prompt_id(prompt_id)
    spec = registry.get(normalized_prompt_id)
    if spec is None:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown {registry_name} prompt_id '{normalized_prompt_id}'. "
            f"Available prompt IDs: {available}"
        )
    return normalized_prompt_id, spec


def _validate_human_persona_record(
    raw_persona: object,
    *,
    index: int,
) -> HumanPersona:
    if not isinstance(raw_persona, dict):
        raise ValueError(f"Human persona #{index} must be an object")
    persona = dict(raw_persona)
    field_names = set(persona)
    missing_fields = sorted(_HUMAN_PERSONA_REQUIRED_FIELDS - field_names)
    if missing_fields:
        raise ValueError(
            f"Human persona #{index} is missing required field(s): {', '.join(missing_fields)}"
        )
    extra_fields = sorted(field_names - _HUMAN_PERSONA_REQUIRED_FIELDS)
    if extra_fields:
        raise ValueError(
            f"Human persona #{index} has unsupported field(s): {', '.join(extra_fields)}"
        )
    example_prompts = persona["example_prompts"]
    if not isinstance(example_prompts, list) or not example_prompts:
        raise ValueError(f"Human persona #{index}.example_prompts must be a non-empty list")
    normalized_prompts: list[str] = []
    for prompt_index, example_prompt in enumerate(example_prompts, start=1):
        if not isinstance(example_prompt, str) or not example_prompt.strip():
            raise ValueError(
                f"Human persona #{index}.example_prompts[{prompt_index}] must be a non-empty string"
            )
        normalized_prompts.append(example_prompt.strip())

    normalized_fields: dict[str, str] = {}
    for field_name in sorted(_HUMAN_PERSONA_REQUIRED_FIELDS - {"example_prompts"}):
        raw_value = persona[field_name]
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(f"Human persona #{index}.{field_name} must be a non-empty string")
        normalized_fields[field_name] = raw_value.strip()

    persona_id = normalized_fields["id"]
    if not _HUMAN_PERSONA_ID_PATTERN.fullmatch(persona_id):
        raise ValueError(
            f"Human persona #{index}.id must match '{_HUMAN_PERSONA_ID_PATTERN.pattern}'"
        )

    return HumanPersona(
        id=persona_id,
        name=normalized_fields["name"],
        user_profile=normalized_fields["user_profile"],
        conversation_style=normalized_fields["conversation_style"],
        example_prompts=normalized_prompts,
        clarity=normalized_fields["clarity"],
        intent_stability=normalized_fields["intent_stability"],
        topic_specificity=normalized_fields["topic_specificity"],
        response_length=normalized_fields["response_length"],
    )


def _load_human_persona_registry() -> dict[str, HumanPersona]:
    path = PROMPTS_DIR / _HUMAN_PERSONA_FILENAME
    content = _read_text_file(path, description="Human persona catalog")
    try:
        raw_personas = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Human persona catalog '{_HUMAN_PERSONA_FILENAME}' is invalid JSON"
        ) from exc
    if not isinstance(raw_personas, list) or not raw_personas:
        raise ValueError(
            f"Human persona catalog '{_HUMAN_PERSONA_FILENAME}' must be a non-empty list"
        )

    registry: dict[str, HumanPersona] = {}
    for index, raw_persona in enumerate(raw_personas, start=1):
        persona = _validate_human_persona_record(raw_persona, index=index)
        persona_id = persona["id"]
        if persona_id in registry:
            raise ValueError(f"Duplicate human persona id '{persona_id}'")
        registry[persona_id] = persona
    return registry


def _read_human_persona_context_template() -> str:
    path = PROMPTS_DIR / _HUMAN_PERSONA_CONTEXT_FILENAME
    template = _read_text_file(path, description="Human persona context template")
    template_fields = frozenset(_extract_template_fields(template))
    if template_fields != _HUMAN_PERSONA_TEMPLATE_FIELDS:
        missing_fields = sorted(_HUMAN_PERSONA_TEMPLATE_FIELDS - template_fields)
        extra_fields = sorted(template_fields - _HUMAN_PERSONA_TEMPLATE_FIELDS)
        details: list[str] = []
        if missing_fields:
            details.append(f"missing field(s): {', '.join(missing_fields)}")
        if extra_fields:
            details.append(f"unexpected field(s): {', '.join(extra_fields)}")
        raise ValueError(
            f"Human persona context template '{_HUMAN_PERSONA_CONTEXT_FILENAME}' is invalid: "
            + "; ".join(details)
        )
    return template


def _format_persona_example_prompts(example_prompts: list[str]) -> str:
    return "\n".join(f"- {example_prompt}" for example_prompt in example_prompts)


def _render_human_persona_context(persona_id: str) -> str:
    registry = _load_human_persona_registry()
    persona = registry[persona_id]
    template = _read_human_persona_context_template()
    return template.format(
        name=persona["name"],
        user_profile=persona["user_profile"],
        conversation_style=persona["conversation_style"],
        response_length=persona["response_length"],
        clarity=persona["clarity"],
        intent_stability=persona["intent_stability"],
        topic_specificity=persona["topic_specificity"],
        example_prompts=_format_persona_example_prompts(persona["example_prompts"]),
    )


def load_system_prompt(
    *,
    prompt_id: str,
    registry: Mapping[str, PromptFileSpec],
    registry_name: str,
) -> str:
    normalized_prompt_id, spec = _get_prompt_spec(
        prompt_id=prompt_id,
        registry=registry,
        registry_name=registry_name,
    )
    return _read_prompt_file(normalized_prompt_id, spec)


def get_human_agent_persona_ids() -> tuple[str, ...]:
    return tuple(sorted(_load_human_persona_registry()))


def get_human_turn_injection_prompt(edit_primitive: str) -> str:
    if not isinstance(edit_primitive, str) or not edit_primitive.strip():
        raise ValueError("edit_primitive must be a non-empty string")
    normalized_edit_primitive = edit_primitive.strip()
    registry = _load_human_turn_injection_prompt_registry()
    prompt = registry.get(normalized_edit_primitive)
    if prompt is None:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown human_turn_injection edit_primitive '{normalized_edit_primitive}'. "
            f"Available edit primitives: {available}"
        )
    return prompt


def _load_human_turn_injection_prompt_registry() -> dict[str, str]:
    path = PROMPTS_DIR / _HUMAN_TURN_INJECTION_PROMPTS_FILENAME
    content = _read_text_file(path, description="Human turn injection prompt registry")
    try:
        raw_payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Human turn injection prompt registry "
            f"'{_HUMAN_TURN_INJECTION_PROMPTS_FILENAME}' is invalid JSON"
        ) from exc
    if not isinstance(raw_payload, dict) or not raw_payload:
        raise ValueError(
            "Human turn injection prompt registry "
            f"'{_HUMAN_TURN_INJECTION_PROMPTS_FILENAME}' must be a non-empty object"
        )

    registry: dict[str, str] = {}
    for raw_edit_primitive, raw_prompt in raw_payload.items():
        if not isinstance(raw_edit_primitive, str) or not raw_edit_primitive.strip():
            raise ValueError("Human turn injection prompt registry keys must be non-empty strings")
        edit_primitive = raw_edit_primitive.strip()
        if not _HUMAN_PERSONA_ID_PATTERN.fullmatch(edit_primitive):
            raise ValueError(
                "Human turn injection edit_primitive "
                f"'{edit_primitive}' must match '{_HUMAN_PERSONA_ID_PATTERN.pattern}'"
            )
        if not isinstance(raw_prompt, str) or not raw_prompt.strip():
            raise ValueError(
                "Human turn injection prompt registry "
                f"entry '{edit_primitive}' must be a non-empty string"
            )
        registry[edit_primitive] = raw_prompt.strip()
    return registry


def validate_human_agent_persona(persona: str | None) -> str | None:
    if persona is None:
        return None
    if not isinstance(persona, str):
        raise ValueError("persona must be a string or None")
    normalized_persona = persona.strip()
    if not normalized_persona:
        return None
    registry = _load_human_persona_registry()
    if normalized_persona not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown human-agent persona '{normalized_persona}'. Available personas: {available}"
        )
    return normalized_persona


def get_human_agent_system_prompt(prompt_id: str, *, persona: str | None = None) -> str:
    prompt = load_system_prompt(
        prompt_id=prompt_id,
        registry=_HUMAN_AGENT_PROMPT_FILE_REGISTRY,
        registry_name="human_agent",
    )
    normalized_persona = validate_human_agent_persona(persona)
    if normalized_persona is None:
        return prompt
    persona_context = _render_human_persona_context(normalized_persona)
    return f"{prompt}\n\n{persona_context}"


def human_agent_prompt_supports_guidance(prompt_id: str) -> bool:
    normalized_prompt_id, spec = _get_prompt_spec(
        prompt_id=prompt_id,
        registry=_HUMAN_AGENT_PROMPT_FILE_REGISTRY,
        registry_name="human_agent",
    )
    _read_prompt_file(normalized_prompt_id, spec)
    return spec.supports_guidance()


def human_agent_prompt_supports_topic_discovery(prompt_id: str) -> bool:
    normalized_prompt_id, spec = _get_prompt_spec(
        prompt_id=prompt_id,
        registry=_HUMAN_AGENT_PROMPT_FILE_REGISTRY,
        registry_name="human_agent",
    )
    _read_prompt_file(normalized_prompt_id, spec)
    return spec.supports_topic_discovery


def normalize_guidance(guidance: str | None) -> str | None:
    if guidance is None:
        return None
    if not isinstance(guidance, str):
        raise ValueError("guidance must be a string or None")
    normalized_guidance = guidance.strip()
    if not normalized_guidance:
        return None
    return normalized_guidance


def validate_human_agent_prompt_guidance(*, prompt_id: str, guidance: str | None) -> str | None:
    normalized_guidance = normalize_guidance(guidance)
    if human_agent_prompt_supports_guidance(prompt_id):
        if normalized_guidance is None:
            raise ValueError(
                f"Human-agent prompt ID '{_normalize_prompt_id(prompt_id)}' requires guidance."
            )
    elif normalized_guidance is not None:
        raise ValueError(
            f"Human-agent prompt ID '{_normalize_prompt_id(prompt_id)}' does not accept guidance."
        )
    return normalized_guidance


def get_human_verifier_system_prompt(prompt_id: str) -> str:
    return load_system_prompt(
        prompt_id=prompt_id,
        registry=_HUMAN_VERIFIER_PROMPT_FILE_REGISTRY,
        registry_name="human_verifier",
    )


def get_semantic_consistency_inference_prompt(prompt_id: str) -> str:
    return load_system_prompt(
        prompt_id=prompt_id,
        registry=_SEMANTIC_CONSISTENCY_INFERENCE_PROMPT_FILE_REGISTRY,
        registry_name="semantic_consistency_inference",
    )


def get_semantic_consistency_judge_prompt(prompt_id: str) -> str:
    return load_system_prompt(
        prompt_id=prompt_id,
        registry=_SEMANTIC_CONSISTENCY_JUDGE_PROMPT_FILE_REGISTRY,
        registry_name="semantic_consistency_judge",
    )


def get_discovery_semantic_theme_match_prompt(prompt_id: str) -> str:
    return load_system_prompt(
        prompt_id=prompt_id,
        registry=_DISCOVERY_SEMANTIC_THEME_MATCH_PROMPT_FILE_REGISTRY,
        registry_name="discovery_semantic_theme_match",
    )


def get_discovery_semantic_duplicate_prompt(prompt_id: str) -> str:
    return load_system_prompt(
        prompt_id=prompt_id,
        registry=_DISCOVERY_SEMANTIC_DUPLICATE_PROMPT_FILE_REGISTRY,
        registry_name="discovery_semantic_duplicate",
    )


def get_null_theme_abstention_trace_judge_prompt(prompt_id: str) -> str:
    return load_system_prompt(
        prompt_id=prompt_id,
        registry=_NULL_THEME_ABSTENTION_TRACE_JUDGE_PROMPT_FILE_REGISTRY,
        registry_name="null_theme_abstention_trace_judge",
    )


def get_null_theme_abstention_topic_match_judge_prompt(prompt_id: str) -> str:
    return load_system_prompt(
        prompt_id=prompt_id,
        registry=_NULL_THEME_ABSTENTION_TOPIC_MATCH_JUDGE_PROMPT_FILE_REGISTRY,
        registry_name="null_theme_abstention_topic_match_judge",
    )


def get_cluster_semantic_coherence_prompt(prompt_id: str) -> str:
    return load_system_prompt(
        prompt_id=prompt_id,
        registry=_CLUSTER_SEMANTIC_COHERENCE_PROMPT_FILE_REGISTRY,
        registry_name="cluster_semantic_coherence",
    )


def get_cluster_semantic_distinctiveness_prompt(prompt_id: str) -> str:
    return load_system_prompt(
        prompt_id=prompt_id,
        registry=_CLUSTER_SEMANTIC_DISTINCTIVENESS_PROMPT_FILE_REGISTRY,
        registry_name="cluster_semantic_distinctiveness",
    )


def get_clarification_policy_prompt(prompt_id: str) -> str:
    return load_system_prompt(
        prompt_id=prompt_id,
        registry=_CLARIFICATION_POLICY_PROMPT_FILE_REGISTRY,
        registry_name="clarification_policy",
    )
