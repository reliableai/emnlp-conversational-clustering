"""Tests for the structured dataset engine and enterprise-ticket preset."""

from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from dataset_generation.cli import build_parser
from dataset_generation.llm import (
    extract_structured_output_payload,
    generate_texts_with_openrouter,
    parse_generated_text_batch,
    response_content_to_text,
)
from dataset_generation.presets.enterprise_ticket import (
    PRESET,
    alignment,
    sampling,
    spec,
    text_generation,
)
from dataset_generation.runtime import generate_dataset_rows, write_dataset_csv
from dataset_generation.sampling import build_balanced_coverage_plan, weighted_choice
from dataset_generation.schema import prompt_safe_schema
from dataset_generation.types import (
    DeterministicEvaluationResult,
    GeneratedTextBatch,
    OpenRouterConfig,
    StructuredDatasetPreset,
)


def _sample_record(seed: int = 7) -> dict[str, str]:
    rng = random.Random(seed)
    record = sampling.sample_record_for_issue_target(
        "IT",
        "access_request",
        "access_request",
        rng,
        sampling.difficulty_profile("medium"),
    )
    corrected = sampling.apply_business_rules(record, rng, difficulty="medium")
    sampling.validate_record(corrected)
    return corrected


def _sample_variation_profile() -> spec.TextVariationProfile:
    return spec.TextVariationProfile(
        structural_mode="symptom_first",
        knowledge_state="guessing",
        intent_mode="ask_status",
        emotional_tone="apologetic",
        temporal_pov="after_change_or_update",
        interaction_pov="redirected_here",
        message_channel="chat_message",
        length_style="short",
        completeness_mode="minimal",
        messiness="slightly_messy",
        mention_system_name=False,
        include_greeting=False,
        include_context=True,
        mention_previous_attempts_or_follow_up=True,
        mention_time_reference=True,
        mention_work_consequence=True,
        mention_other_people_affected=True,
        include_explicit_resolution_request=False,
        prefer_indirect_issue_description=True,
        prefer_symptom_before_request=True,
        mention_uncertainty=True,
        mention_self_help_attempt=False,
        allow_incomplete_details=True,
        allow_fragment_style=True,
        sentence_pattern="compressed",
    )


class _StubPreset(StructuredDatasetPreset):
    name = "stub"
    default_rows = 1
    default_output_path = "unused.csv"
    default_mode = "balanced"
    default_difficulty = "medium"
    default_evaluate_and_regenerate = False
    schema = {"label": {"type": "string"}}
    text_column_name = "text"
    output_columns = ("text", "label")
    system_prompt = "stub system prompt"
    evaluation_system_prompt = "stub evaluation system prompt"
    difficulty_choices = ("medium",)
    mode_choices = ("balanced",)
    coverage_field = "label"

    def __init__(self) -> None:
        self.prompt_batches: list[list[dict[str, Any]]] = []
        self.repair_calls: list[tuple[dict[str, str], str]] = []

    def audit_schema_support(self) -> dict[str, list[str]]:
        return {}

    def generate_records(
        self,
        n: int,
        rng: random.Random | None = None,
        *,
        mode: str,
        difficulty: str,
        min_coverage_count: int | None,
    ) -> list[dict[str, str]]:
        _ = rng, mode, difficulty, min_coverage_count
        return [{"label": f"record-{row_id}"} for row_id in range(n)]

    def reachable_coverage_targets(self) -> list[str]:
        return ["stub"]

    def generate_records_for_coverage_targets(
        self,
        coverage_targets: list[str],
        rng: random.Random | None = None,
        *,
        difficulty: str,
    ) -> list[dict[str, str]]:
        _ = rng, difficulty
        return [{"label": coverage_target} for coverage_target in coverage_targets]

    def build_row_payload(
        self,
        record: dict[str, str],
        row_id: int,
        rng: random.Random,
        difficulty: str,
        entity_mention_probability: float | None,
    ) -> dict[str, Any]:
        return {"row_id": row_id, "label": record["label"]}

    def build_prompt(self, batch_rows: list[dict[str, Any]], difficulty: str) -> str:
        self.prompt_batches.append(batch_rows)
        return json.dumps(batch_rows)

    def collect_alignment_failures(
        self,
        records: list[dict[str, str]],
        parsed: dict[int, str],
    ) -> tuple[dict[int, ValueError], dict[int, ValueError]]:
        hard_failures: dict[int, ValueError] = {}
        soft_failures: dict[int, ValueError] = {}
        for row_id, ticket_text in parsed.items():
            if ticket_text.startswith("hard:"):
                hard_failures[row_id] = ValueError(f"hard failure for row {row_id}")
            elif ticket_text.startswith("soft:"):
                soft_failures[row_id] = ValueError(f"soft failure for row {row_id}")
        return hard_failures, soft_failures

    def build_alignment_retry_prefix(
        self,
        records: list[dict[str, str]],
        parsed: dict[int, str],
    ) -> str | None:
        return "Retry the misaligned rows."

    def repair_generated_text(self, record: dict[str, str], text: str) -> str:
        self.repair_calls.append((record, text))
        return f"repaired:{record['label']}:{text or 'empty'}"

    def build_emergency_fallback_text(self, record: dict[str, str]) -> str:
        return f"fallback:{record['label']}"

    def build_evaluation_row_payload(self, row_id: int, row: dict[str, str]) -> dict[str, Any]:
        return {"row_id": row_id, "text": row["text"], "metadata": row}

    def build_evaluation_prompt(self, batch_rows: list[dict[str, Any]]) -> str:
        return json.dumps(batch_rows)

    def evaluate_row_deterministically(
        self,
        record: dict[str, str],
        text: str,
    ) -> DeterministicEvaluationResult:
        _ = record, text
        return DeterministicEvaluationResult(rating="pass", reasons=(), details={})

    def validate_record(self, record: dict[str, str]) -> None:
        return None

    def build_summary(
        self,
        rows: list[dict[str, str]],
        *,
        mode: str,
        difficulty: str,
        schema_support_report: dict[str, list[str]],
        entity_mention_probability: float | None = None,
        openrouter_model: str | None = None,
    ) -> str:
        return "summary"


class _NonBridgePreset:
    name = "non_bridge"
    default_rows = 1
    default_output_path = "unused.csv"
    default_mode = "balanced"
    default_difficulty = "medium"
    coverage_target_name = "issue_type"
    schema = {"label": {"type": "string"}}
    output_columns = ("ticket_text", "label")
    system_prompt = "stub system prompt"
    difficulty_choices = ("medium",)
    mode_choices = ("balanced",)

    def audit_schema_support(self) -> dict[str, list[str]]:
        return {}

    def generate_records(
        self,
        n: int,
        rng: random.Random | None = None,
        *,
        mode: str,
        difficulty: str,
        min_coverage_target_count: int | None,
    ) -> list[dict[str, str]]:
        _ = rng, mode, difficulty, min_coverage_target_count
        return [{"label": f"record-{row_id}"} for row_id in range(n)]

    def build_row_payload(
        self,
        record: dict[str, str],
        row_id: int,
        rng: random.Random,
        difficulty: str,
        system_mention_probability: float | None,
    ) -> dict[str, Any]:
        _ = row_id, rng, difficulty, system_mention_probability
        return {"label": record["label"]}

    def build_prompt(self, batch_rows: list[dict[str, Any]], difficulty: str) -> str:
        _ = batch_rows, difficulty
        return "prompt"

    def collect_alignment_failures(
        self,
        records: list[dict[str, str]],
        parsed: dict[int, str],
    ) -> tuple[dict[int, ValueError], dict[int, ValueError]]:
        _ = records, parsed
        return {}, {}

    def build_alignment_retry_prefix(
        self,
        records: list[dict[str, str]],
        parsed: dict[int, str],
    ) -> str | None:
        _ = records, parsed
        return None

    def repair_generated_text(self, record: dict[str, str], ticket_text: str) -> str:
        _ = record
        return ticket_text

    def validate_record(self, record: dict[str, str]) -> None:
        _ = record

    def build_summary(
        self,
        rows: list[dict[str, str]],
        *,
        mode: str,
        difficulty: str,
        schema_support_report: dict[str, list[str]],
        system_mention_probability: float | None = None,
        openrouter_model: str | None = None,
    ) -> str:
        _ = rows, mode, difficulty, schema_support_report, system_mention_probability
        _ = openrouter_model
        return "summary"


def _install_openrouter_stubs(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[object],
    prompts: list[str],
    *,
    client_kwargs: list[dict[str, Any]] | None = None,
) -> None:
    response_queue = list(responses)

    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.__dict__["load_dotenv"] = lambda: None

    langchain_core_module = types.ModuleType("langchain_core")
    messages_module = types.ModuleType("langchain_core.messages")

    class _BaseMessage:
        def __init__(self, content: str) -> None:
            self.content = content

    messages_module.__dict__["HumanMessage"] = _BaseMessage
    messages_module.__dict__["SystemMessage"] = _BaseMessage
    langchain_core_module.__dict__["messages"] = messages_module

    openrouter_module = types.ModuleType("langchain_openrouter")

    class _FakeChatOpenRouter:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self._structured_schema: type | None = None
            if client_kwargs is not None:
                client_kwargs.append(kwargs)

        def with_structured_output(
            self,
            schema: object,
            *,
            strict: bool = False,
            include_raw: bool = False,
        ) -> _FakeChatOpenRouter:
            _ = strict, include_raw
            self._structured_schema = schema if isinstance(schema, type) else None
            return self

        def invoke(self, messages: list[_BaseMessage]) -> Any:
            prompts.append(messages[-1].content)
            if not response_queue:
                raise AssertionError("No fake responses remaining for ChatOpenRouter.invoke().")
            response = response_queue.pop(0)
            if isinstance(response, BaseException):
                raise response
            if self._structured_schema is not None and isinstance(response, dict):
                response_dict = dict(response)
                parsed = response_dict.get("parsed")
                if isinstance(parsed, dict):
                    return {
                        **response_dict,
                        "parsed": self._structured_schema(**parsed),
                    }
            return response

    openrouter_module.__dict__["ChatOpenRouter"] = _FakeChatOpenRouter

    monkeypatch.setitem(sys.modules, "dotenv", dotenv_module)
    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core_module)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", messages_module)
    monkeypatch.setitem(sys.modules, "langchain_openrouter", openrouter_module)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def _openrouter_config(*, batch_size: int = 10, max_retries: int = 0) -> OpenRouterConfig:
    return OpenRouterConfig(
        model="demo/model",
        temperature=0.2,
        batch_size=batch_size,
        max_retries=max_retries,
    )


def _capture_sample_weights(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    captured_weights: dict[str, float] = {}

    def _fake_weighted_choice(_rng: random.Random, weights: dict[str, float]) -> str:
        captured_weights.clear()
        captured_weights.update(weights)
        return next(iter(weights))

    monkeypatch.setattr(sampling, "weighted_choice", _fake_weighted_choice)
    return captured_weights


def test_sample_approval_required_applies_table_driven_overrides_and_adjustments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_weights = _capture_sample_weights(monkeypatch)

    sampling.sample_approval_required(
        "Security",
        "purchase_request",
        "admin_access",
        random.Random(7),
    )

    assert captured_weights == pytest.approx(
        {
            "none": 0.05,
            "manager_approval": 0.50,
            "finance_approval": 0.34,
            "security_approval": 0.48,
            "multi_step_approval": 0.48,
        }
    )


def test_sample_dependency_type_applies_table_driven_blocked_workflow_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_weights = _capture_sample_weights(monkeypatch)

    sampling.sample_dependency_type(
        {
            "approval_required": "multi_step_approval",
            "sub_intent": "missing_tax_form",
            "workflow_stage": "blocked",
        },
        random.Random(7),
    )

    assert captured_weights == pytest.approx(
        {
            "none": 0.0,
            "waiting_for_document": 0.40,
            "waiting_for_manager": 0.27,
            "waiting_for_vendor": 0.29,
            "waiting_for_system_sync": 0.16,
            "waiting_for_another_team": 0.22,
        }
    )


def test_sample_severity_applies_table_driven_security_and_category_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_weights = _capture_sample_weights(monkeypatch)

    sampling.sample_severity(
        {
            "urgency": "high",
            "department": "Security",
            "sub_intent": "phishing_report",
            "ticket_category": "developer_tooling",
        },
        random.Random(7),
    )

    assert captured_weights == pytest.approx(
        {
            "minor": 0.0,
            "moderate": 0.34,
            "major": 0.72,
            "business_critical": 0.32,
        }
    )


def test_sample_impact_scope_applies_table_driven_security_and_urgency_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_weights = _capture_sample_weights(monkeypatch)

    sampling.sample_impact_scope(
        {
            "ticket_category": "support_platform",
            "department": "Security",
            "sub_intent": "phishing_report",
            "urgency": "critical",
        },
        random.Random(7),
    )

    assert captured_weights == pytest.approx(
        {
            "single_user": 0.16,
            "single_team": 0.22,
            "single_department": 0.37,
            "multi_department": 0.42,
            "company_wide": 0.21,
        }
    )


def test_sample_writing_style_applies_table_driven_pressure_and_role_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_weights = _capture_sample_weights(monkeypatch)

    sampling.sample_writing_style(
        {
            "department": "Engineering",
            "urgency": "high",
            "workflow_stage": "blocked",
            "requester_role": "executive_assistant",
        },
        random.Random(7),
    )

    assert captured_weights == pytest.approx(
        {
            "concise": 0.20,
            "formal": 0.34,
            "informal": 0.18,
            "frustrated": 0.24,
            "technical": 0.32,
            "non_native_english": 0.12,
        }
    )


def test_sample_clarity_level_applies_table_driven_writing_style_and_role_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_weights = _capture_sample_weights(monkeypatch)

    sampling.sample_clarity_level(
        {
            "writing_style": "technical",
            "requester_role": "intern",
        },
        random.Random(7),
    )

    assert captured_weights == pytest.approx(
        {
            "clear": 0.52,
            "moderately_clear": 0.36,
            "vague": 0.04,
            "missing_some_details": 0.28,
        }
    )


def test_spec_module_loads_expected_resources() -> None:
    issue_types = prompt_safe_schema(spec.SCHEMA)["issue_type"]["mapping"]

    assert spec.CSV_COLUMNS[0] == "ticket_text"
    assert "issue_type" in spec.SCHEMA
    assert "intent_group" in spec.SCHEMA
    assert "sub_intent" in spec.SCHEMA
    assert "intent_label_coarse" in spec.SCHEMA
    assert "organizational_label_fine" in spec.SCHEMA
    assert "granularity_family" not in spec.SCHEMA
    assert isinstance(next(iter(spec.SCHEMA["sub_intent"]["mapping"])), tuple)
    assert ("IT", "access_request") in spec.SUB_DEPARTMENT_WEIGHTS
    assert len(issue_types) >= 8
    assert isinstance(spec.ACCESS_ISSUES, frozenset)
    assert isinstance(spec.SYSTEM_SYNC_ISSUES, frozenset)
    assert spec.GRANULARITY_LABEL_COMPONENTS["organizational"]["fine"] == (
        "department",
        "sub_department",
        "owning_team",
    )
    assert spec.GRANULARITY_LABEL_COMPONENTS["intent"]["coarse"] == ("intent_group",)
    assert spec.GRANULARITY_LABEL_FIELDS["intent"]["fine"] == "intent_label_fine"
    assert "synthetic internal enterprise workflow tickets" in spec.OPENROUTER_SYSTEM_PROMPT


def test_alignment_resource_covers_declared_issue_types_and_sub_intents() -> None:
    raw_alignment = json.loads((spec.RESOURCE_DIR / "alignment.json").read_text(encoding="utf-8"))
    declared_issue_types = {
        issue_type
        for values in spec.SCHEMA["issue_type"]["mapping"].values()
        for issue_type in values
    }
    declared_sub_intents = {
        sub_intent
        for values in spec.SCHEMA["sub_intent"]["mapping"].values()
        for sub_intent in values
    }

    assert declared_issue_types <= set(raw_alignment["ISSUE_TYPE_HINTS"])
    assert declared_sub_intents <= set(raw_alignment["SUB_INTENT_HINTS"])


def test_guidance_resource_covers_runtime_value_maps() -> None:
    raw_guidance = json.loads((spec.RESOURCE_DIR / "guidance.json").read_text(encoding="utf-8"))

    assert set(spec.SCHEMA["workflow_stage"]["values"]) == set(raw_guidance["WORKFLOW_GUIDANCE"])
    assert set(spec.SCHEMA["approval_required"]["values"]) <= set(raw_guidance["APPROVAL_GUIDANCE"])
    assert set(spec.SCHEMA["dependency_type"]["values"]) == set(raw_guidance["DEPENDENCY_GUIDANCE"])
    assert set(spec.SCHEMA["dependency_type"]["values"]) == set(
        raw_guidance["DEPENDENCY_GUIDANCE_PROBABILITY"]
    )
    assert set(spec.SCHEMA["severity"]["values"]) == set(raw_guidance["SEVERITY_GUIDANCE"])
    assert set(spec.SCHEMA["severity"]["values"]) == set(
        raw_guidance["SEVERITY_GUIDANCE_PROBABILITY"]
    )
    assert set(spec.SCHEMA["impact_scope"]["values"]) == set(raw_guidance["IMPACT_GUIDANCE"])
    assert set(spec.SCHEMA["impact_scope"]["values"]) == set(
        raw_guidance["IMPACT_GUIDANCE_PROBABILITY"]
    )

    # Variation profile values sampled in text_generation.py
    assert {
        "symptom_first",
        "request_first",
        "context_first",
        "fragmented",
        "follow_up",
        "narrative",
    } == set(raw_guidance["STRUCTURE_GUIDANCE"])
    assert {
        "correct_understanding",
        "confused",
        "misdiagnosing",
        "guessing",
        "unsure_if_correct_channel",
    } == set(raw_guidance["KNOWLEDGE_GUIDANCE"])
    assert {
        "request_fix",
        "ask_question",
        "ask_status",
        "report_issue_only",
        "confirm_action",
        "escalate",
    } == set(raw_guidance["INTENT_GUIDANCE"])
    assert {
        "neutral",
        "frustrated",
        "confused",
        "apologetic",
        "urgent",
        "passive",
        "slightly_annoyed",
    } == set(raw_guidance["EMOTION_GUIDANCE"])
    assert {"complete", "partial", "minimal"} == set(raw_guidance["COMPLETENESS_GUIDANCE"])

    assert set(raw_guidance["TEMPORAL_DIRECTIVE_MAP"]) == set(raw_guidance["TEMPORAL_GUIDANCE"])
    interaction_guidance_keys = set(raw_guidance["INTERACTION_GUIDANCE"])
    assert set(raw_guidance["INTERACTION_DIRECTIVE_MAP"]) - {"none"} <= interaction_guidance_keys
    assert raw_guidance["INTERACTION_DIRECTIVE_MAP"]["none"] in interaction_guidance_keys

    for guidance_key in (
        "DEPENDENCY_GUIDANCE_PROBABILITY",
        "SEVERITY_GUIDANCE_PROBABILITY",
        "IMPACT_GUIDANCE_PROBABILITY",
    ):
        assert all(0.0 <= value <= 1.0 for value in raw_guidance[guidance_key].values())


def test_weighted_choice_rejects_non_positive_weight_maps() -> None:
    with pytest.raises(ValueError, match="at least one positive weight"):
        weighted_choice(random.Random(7), {"a": 0.0, "b": -1.0})


def test_sample_business_event_context_uses_declarative_department_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_weights: dict[str, float] = {}

    def _capture_weighted_choice(_rng: random.Random, weights: dict[str, float]) -> str:
        captured_weights.clear()
        captured_weights.update(weights)
        return "routine_operation"

    monkeypatch.setattr(sampling, "weighted_choice", _capture_weighted_choice)

    sampled = sampling.sample_business_event_context("HR", None, None, random.Random(7))

    assert sampled == "routine_operation"
    for (
        event_context,
        allowed_departments,
    ) in spec.BUSINESS_EVENT_CONTEXT_ALLOWED_DEPARTMENTS.items():
        if "HR" not in allowed_departments:
            assert event_context not in captured_weights
    assert captured_weights["new_hire_onboarding"] == pytest.approx(
        spec.BUSINESS_EVENT_CONTEXT_BASE_WEIGHTS["new_hire_onboarding"]
        + spec.BUSINESS_EVENT_CONTEXT_DEPARTMENT_ADJUSTMENTS["HR"]["new_hire_onboarding"]
        + spec.BUSINESS_EVENT_CONTEXT_PREFERRED_DEPARTMENTS["new_hire_onboarding"]["HR"]
    )


def test_sample_system_name_keeps_declared_systems_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_weights: dict[str, float] = {}

    def _capture_weighted_choice(_rng: random.Random, weights: dict[str, float]) -> str:
        captured_weights.clear()
        captured_weights.update(weights)
        return next(iter(weights))

    monkeypatch.setattr(sampling, "weighted_choice", _capture_weighted_choice)

    for department, systems in spec.SCHEMA["system_name"]["mapping"].items():
        ticket_category = spec.SCHEMA["ticket_category"]["mapping"][department][0]
        sub_intent = spec.SCHEMA["sub_intent"]["mapping"][(department, ticket_category)][0]

        sampling.sample_system_name(department, ticket_category, sub_intent, random.Random(7))

        for system_name in systems:
            assert system_name in captured_weights
            assert captured_weights[system_name] > 0


def test_split_engineering_and_data_aliases_keep_intentional_representative_targets() -> None:
    assert sampling.canonicalize_issue_type("engineering_operations_request") == (
        "release_operations_issue"
    )
    assert sampling.canonicalize_issue_type("data_operations_request") == (
        "metrics_governance_issue"
    )


def test_sales_crm_support_sub_department_weights_favor_sales_operations() -> None:
    weights = spec.SUB_DEPARTMENT_WEIGHTS[("Sales", "crm_support")]

    assert weights["Sales Operations"] > weights["Territory Planning"]


def test_build_balanced_issue_plan_assigns_each_core_issue_type_once_with_min_count_one() -> None:
    targets = sampling.reachable_issue_targets()

    plan = sampling.build_balanced_issue_plan(
        len(targets),
        random.Random(7),
        min_coverage_target_count=1,
    )

    assert {target: plan.count(target) for target in plan} == {target: 1 for target in targets}


def test_build_balanced_coverage_plan_balances_targets_without_minimum_count() -> None:
    targets = ["a", "b", "c", "d"]

    plan = build_balanced_coverage_plan(targets, 10, random.Random(7))
    counts = {target: plan.count(target) for target in targets}

    assert len(plan) == 10
    assert set(plan) == set(targets)
    assert set(counts.values()) == {2, 3}


def test_build_balanced_coverage_plan_respects_exact_minimum_allocation() -> None:
    targets = ["a", "b", "c", "d"]

    plan = build_balanced_coverage_plan(targets, 8, random.Random(7), min_count=2)
    counts = {target: plan.count(target) for target in targets}

    assert len(plan) == 8
    assert counts == {target: 2 for target in targets}


def test_build_balanced_coverage_plan_distributes_large_remainder_across_targets() -> None:
    targets = [f"target-{index}" for index in range(25)]

    plan = build_balanced_coverage_plan(
        targets,
        210,
        random.Random(7),
        min_count=1,
    )

    counts = {target: plan.count(target) for target in targets}

    assert len(plan) == 210
    assert set(counts.values()) == {8, 9}


def test_build_balanced_coverage_plan_rejects_negative_minimum_count() -> None:
    with pytest.raises(ValueError, match="min_coverage_target_count must be non-negative"):
        build_balanced_coverage_plan(["a", "b"], 4, random.Random(7), min_count=-1)


def test_build_balanced_coverage_plan_rejects_impossible_minimum_request() -> None:
    with pytest.raises(
        ValueError,
        match="rows is too small for the requested minimum issue_type coverage: 4 < 6",
    ):
        build_balanced_coverage_plan(
            ["a", "b", "c"],
            4,
            random.Random(7),
            min_count=2,
            coverage_label="issue_type",
        )


def test_generate_records_balanced_min_count_covers_every_core_issue_type_once() -> None:
    targets = sampling.reachable_issue_targets()

    records = sampling.generate_records(
        len(targets),
        random.Random(7),
        mode="balanced",
        difficulty="medium",
        min_coverage_target_count=1,
    )

    observed = {target: 0 for target in targets}
    for record in records:
        observed[record["issue_type"]] += 1

    assert observed == {target: 1 for target in targets}


def test_generate_records_balanced_min_count_preserves_requested_row_count_with_surplus() -> None:
    targets = sampling.reachable_issue_targets()
    requested_rows = len(targets) + 15

    records = sampling.generate_records(
        requested_rows,
        random.Random(7),
        mode="balanced",
        difficulty="medium",
        min_coverage_target_count=1,
    )

    observed = {target: 0 for target in targets}
    for record in records:
        observed[record["issue_type"]] += 1

    assert len(records) == requested_rows
    assert all(count >= 1 for count in observed.values())


def test_random_generation_skews_toward_access_related_rows() -> None:
    records = sampling.generate_records(
        600,
        random.Random(7),
        mode="random",
        difficulty="medium",
    )

    access_related_count = sum(
        1
        for record in records
        if record["issue_type"] in {"access_request", "authentication_issue"}
    )
    access_related_share = access_related_count / len(records)

    assert 0.30 <= access_related_share <= 0.45


def test_generate_records_add_both_granularity_label_suites() -> None:
    records = sampling.generate_records(
        40,
        random.Random(7),
        mode="random",
        difficulty="medium",
    )

    assert all("intent_label_coarse" in record for record in records)
    assert all("organizational_label_coarse" in record for record in records)


def test_sample_record_adds_parseable_granularity_annotations() -> None:
    for seed in range(7, 17):
        record = _sample_record(seed=seed)

        label_suites = sampling.granularity_label_suites_for_record(record)
        assert label_suites["intent"]["coarse"] == record["intent_label_coarse"]
        assert label_suites["intent"]["normal"] == record["intent_label_normal"]
        assert label_suites["intent"]["fine"] == record["intent_label_fine"]
        assert label_suites["organizational"]["coarse"] == record["organizational_label_coarse"]
        assert label_suites["organizational"]["normal"] == record["organizational_label_normal"]
        assert label_suites["organizational"]["fine"] == record["organizational_label_fine"]

        assert label_suites["organizational"]["coarse"] == spec.build_granularity_label(
            record["department"]
        )
        assert label_suites["organizational"]["normal"] == spec.build_granularity_label(
            record["department"],
            record["sub_department"],
        )
        assert label_suites["organizational"]["fine"] == spec.build_granularity_label(
            record["department"],
            record["sub_department"],
            record["owning_team"],
        )
        assert label_suites["intent"]["coarse"] == spec.build_granularity_label(
            record["intent_group"]
        )
        assert label_suites["intent"]["normal"] == spec.build_granularity_label(
            record["intent_group"],
            record["issue_type"],
        )
        assert label_suites["intent"]["fine"] == spec.build_granularity_label(
            record["intent_group"],
            record["issue_type"],
            record["sub_intent"],
        )


def test_apply_business_rules_avoids_critical_single_user_rows() -> None:
    record = {
        **_sample_record(),
        "department": "Security",
        "sub_intent": "credential_compromise",
        "issue_type": "security_incident",
        "urgency": "critical",
        "impact_scope": "single_user",
        "severity": "major",
    }

    corrected = sampling.apply_business_rules(record, random.Random(7), difficulty="medium")

    assert not (corrected["urgency"] == "critical" and corrected["impact_scope"] == "single_user")


def test_build_llm_generation_directives_maps_temporal_and_interaction_aliases() -> None:
    directives = text_generation.build_llm_generation_directives(
        _sample_record(),
        random.Random(7),
        "medium",
        variation_profile=_sample_variation_profile(),
    )

    assert directives["structural_pov"] == "symptom_first"
    assert directives["knowledge_pov"] == "guessing"
    assert directives["intent_pov"] == "ask_status"
    assert directives["temporal_pov"] == "after_change"
    assert directives["interaction_pov"] == "told_to_submit_ticket"
    assert directives["allow_fragment_style"] is True


def test_build_hidden_generation_guidance_humanizes_profile_without_backend_labels() -> None:
    guidance = text_generation.build_hidden_generation_guidance(
        _sample_record(),
        random.Random(7),
        variation_profile=_sample_variation_profile(),
    )

    assert "open with the symptom before the explanation or ask" in guidance
    assert "let the requester hedge and guess about the cause" in guidance
    assert "make the requester mainly ask for a status update" in guidance
    assert "sound a little apologetic or hesitant" in guidance
    assert any(item.startswith("connect it to a recent change, setup, update") for item in guidance)
    assert any("submit this here" in item for item in guidance)
    assert (
        "do not force the product name into the ticket if the user would naturally omit it"
        in guidance
    )
    assert "mention what this is delaying, blocking, or making harder" in guidance
    assert all("manager_approval" not in item for item in guidance)
    assert all("waiting_for_manager" not in item for item in guidance)


def test_prompt_safe_schema_serializes_tuple_key_mappings_for_sub_intents() -> None:
    prompt_schema = prompt_safe_schema(spec.SCHEMA, ("sub_intent",))
    mapping = prompt_schema["sub_intent"]["mapping"]

    assert isinstance(mapping, list)
    assert set(mapping[0]["parents"]) == {"department", "ticket_category"}
    assert isinstance(mapping[0]["values"], list)
    assert mapping[0]["values"]


def test_parse_generated_text_batch_normalizes_whitespace() -> None:
    payload = GeneratedTextBatch.model_validate(
        {
            "items": [
                {"row_id": 0, "text": "Need   VPN\naccess"},
                {"row_id": 1, "text": " Still blocked\t today "},
            ]
        }
    )

    parsed = parse_generated_text_batch(payload, {0, 1})

    assert parsed == {0: "Need VPN access", 1: "Still blocked today"}


def test_parse_generated_text_batch_rejects_missing_row_ids() -> None:
    payload = GeneratedTextBatch.model_validate({"items": [{"row_id": 0, "text": "Only one item"}]})

    with pytest.raises(ValueError, match="row_ids do not match"):
        parse_generated_text_batch(payload, {0, 1})


def test_response_content_to_text_supports_string_and_text_blocks() -> None:
    assert response_content_to_text("plain text") == "plain text"

    content = [
        {"text": "hello "},
        "world",
        type("TextBlock", (), {"text": " again"})(),
    ]

    assert response_content_to_text(content) == "hello world again"


def test_build_openrouter_evaluation_prompt_uses_resource_template() -> None:
    prompt = text_generation.build_openrouter_evaluation_prompt(
        [{"row_id": 0, "text": "Need VPN access", "metadata": {"issue_type": "access_request"}}]
    )

    assert prompt.startswith(spec.OPENROUTER_EVALUATION_PROMPT)
    assert '"row_id": 0' in prompt
    assert '"text": "Need VPN access"' in prompt


def test_extract_structured_output_payload_returns_direct_payload() -> None:
    payload = GeneratedTextBatch.model_validate({"items": []})

    assert extract_structured_output_payload(payload) == payload


def test_extract_structured_output_payload_returns_parsed_payload_from_wrapper() -> None:
    payload = GeneratedTextBatch.model_validate({"items": []})

    assert extract_structured_output_payload({"parsed": payload, "parsing_error": None}) == payload


def test_extract_structured_output_payload_raises_on_parsing_error() -> None:
    with pytest.raises(ValueError, match="structured output parsing failed"):
        extract_structured_output_payload(
            {
                "parsed": None,
                "parsing_error": ValueError("bad schema"),
                "raw": types.SimpleNamespace(content="bad"),
            }
        )


def test_validate_ticket_text_alignment_accepts_matching_text_and_rejects_drift() -> None:
    record = _sample_record()

    text_generation.validate_ticket_text_alignment(
        record,
        "Need VPN access for a new starter today",
    )

    with pytest.raises(ValueError, match="label alignment checks"):
        text_generation.validate_ticket_text_alignment(
            record,
            "My invoice payment is still delayed and reimbursement is missing",
        )


def test_ticket_text_alignment_accepts_access_paraphrases_without_literal_leaf_label() -> None:
    assert alignment.ticket_text_matches_labels(
        "access_request",
        "temporary_access",
        "Can I get elevated rights for today so I can finish the audit?",
    )
    assert alignment.ticket_text_matches_labels(
        "access_request",
        "admin_access",
        "Please grant administrator privileges in Okta for this change window",
    )
    assert alignment.ticket_text_matches_labels(
        "authentication_issue",
        "account_locked",
        "I am locked out again and cannot sign in after the MFA prompts",
    )


def test_ticket_text_alignment_accepts_legal_and_workspace_paraphrases() -> None:
    assert alignment.ticket_text_matches_labels(
        "legal_compliance_request",
        "data_deletion_request",
        (
            "Can you remove this person's personal information from our systems under "
            "the privacy request?"
        ),
    )
    assert alignment.ticket_text_matches_labels(
        "workspace_support",
        "lighting_problem",
        "The lights over our desks keep flickering and half the office area is dark.",
    )


def test_ticket_text_alignment_accepts_travel_receipt_match_paraphrase() -> None:
    record = {
        **_sample_record(),
        "department": "Travel",
        "sub_department": "Expense Operations",
        "ticket_category": "travel_expense",
        "issue_type": "travel_operations_issue",
        "sub_intent": "receipt_match_issue",
        "system_name": "Concur",
    }

    assessment = alignment.assess_ticket_text_alignment(
        record,
        (
            "The receipts I uploaded are not showing as matched with my corporate card "
            "charges in Concur since the system update."
        ),
    )

    assert assessment.issue_type_match is True
    assert assessment.sub_intent_match is True


def test_ticket_text_alignment_rejects_short_alias_matches_inside_other_words() -> None:
    assert not alignment.ticket_text_matches_labels(
        "workspace_support",
        "ac_issue",
        "Badge access to the finance suite is still broken today.",
    )


def test_assess_ticket_text_alignment_rejects_context_only_matches() -> None:
    record = {
        **_sample_record(),
        "department": "IT",
        "sub_department": "IAM",
        "ticket_category": "access_request",
        "issue_type": "access_request",
        "sub_intent": "temporary_access",
        "system_name": "Okta",
    }

    assessment = alignment.assess_ticket_text_alignment(
        record,
        "Okta for the IAM team in IT is still blocked for me today.",
    )

    assert assessment.context_terms
    assert assessment.issue_type_match is False
    assert assessment.sub_intent_match is False


def test_collect_alignment_failures_treats_leaf_miss_as_soft_failure() -> None:
    record = {
        **_sample_record(),
        "issue_type": "service_issue",
        "sub_intent": "application_crash",
        "ticket_category": "software_issue",
        "system_name": "Google Workspace",
    }

    hard_failures, soft_failures = text_generation.collect_alignment_failures(
        [record],
        {0: "The software in Google Workspace is not working for me today and I need help."},
    )

    assert hard_failures == {}
    assert 0 in soft_failures


def test_collect_alignment_failures_treats_banned_phrase_as_soft_failure() -> None:
    record = {
        **_sample_record(),
        "issue_type": "access_request",
        "sub_intent": "temporary_access",
        "ticket_category": "access_request",
        "system_name": "Okta",
    }

    hard_failures, soft_failures = text_generation.collect_alignment_failures(
        [record],
        {
            0: (
                "Need temporary access in Okta today because the system sync still "
                "has not reflected the role change."
            )
        },
    )

    assert hard_failures == {}
    assert 0 in soft_failures
    assert "banned internal process wording" in str(soft_failures[0])


def test_repair_ticket_text_alignment_replaces_soft_failed_text_with_fallback() -> None:
    record = {
        **_sample_record(),
        "issue_type": "finance_transaction_issue",
        "sub_intent": "duplicate_invoice",
        "ticket_category": "invoice_issue",
        "department": "Finance",
        "sub_department": "Accounts Payable",
        "system_name": "NetSuite",
    }

    source_text = "Still seeing that double entry in NetSuite from the migration last week."
    repaired = text_generation.repair_ticket_text_alignment(record, source_text)

    assessment = alignment.assess_ticket_text_alignment(record, repaired)
    assert repaired != source_text
    assert assessment.issue_type_match is True
    assert assessment.sub_intent_match is True


def test_build_row_payload_includes_row_id() -> None:
    payload = PRESET.build_row_payload(
        _sample_record(),
        7,
        random.Random(7),
        "medium",
        None,
    )

    assert payload["row_id"] == 7


def test_build_openrouter_batch_prompt_includes_row_id() -> None:
    payload = PRESET.build_row_payload(
        _sample_record(),
        3,
        random.Random(7),
        "medium",
        None,
    )

    prompt = PRESET.build_prompt([payload], "medium")

    assert '"row_id": 3' in prompt


def test_generate_dataset_rows_passes_sequential_row_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _StubPreset()
    captured_row_payloads: list[dict[str, Any]] = []

    def _fake_generate_texts(*args: Any, **kwargs: Any) -> list[str]:
        nonlocal captured_row_payloads
        captured_row_payloads = kwargs["row_payloads"] if "row_payloads" in kwargs else args[2]
        return ["text zero", "text one"]

    monkeypatch.setattr(
        "dataset_generation.runtime.generate_texts_with_openrouter",
        _fake_generate_texts,
    )

    rows = generate_dataset_rows(
        preset,
        2,
        random.Random(7),
        mode="balanced",
        difficulty="medium",
        min_coverage_count=None,
        entity_mention_probability=None,
        openrouter_config=_openrouter_config(batch_size=2),
    )

    assert captured_row_payloads == [
        {"row_id": 0, "label": "record-0"},
        {"row_id": 1, "label": "record-1"},
    ]
    assert [row["text"] for row in rows] == ["text zero", "text one"]


def test_generate_texts_with_openrouter_repairs_terminal_soft_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _StubPreset()
    prompts: list[str] = []
    _install_openrouter_stubs(
        monkeypatch,
        [{"parsed": {"items": [{"row_id": 0, "text": "soft:first"}]}}],
        prompts,
    )

    texts = generate_texts_with_openrouter(
        preset,
        [{"label": "record-0"}],
        [{"row_id": 0, "label": "record-0"}],
        random.Random(7),
        "medium",
        config=_openrouter_config(max_retries=0),
    )

    assert texts == ["repaired:record-0:soft:first"]
    assert preset.repair_calls == [({"label": "record-0"}, "soft:first")]


def test_generate_texts_with_openrouter_repairs_repeated_soft_failures_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _StubPreset()
    prompts: list[str] = []
    _install_openrouter_stubs(
        monkeypatch,
        [
            {"parsed": {"items": [{"row_id": 0, "text": "soft:first"}]}},
            {"parsed": {"items": [{"row_id": 0, "text": "soft:second"}]}},
        ],
        prompts,
    )

    texts = generate_texts_with_openrouter(
        preset,
        [{"label": "record-0"}],
        [{"row_id": 0, "label": "record-0"}],
        random.Random(7),
        "medium",
        config=_openrouter_config(max_retries=1),
    )

    assert texts == ["repaired:record-0:soft:second"]
    assert preset.repair_calls == [({"label": "record-0"}, "soft:second")]
    assert len(prompts) == 2


def test_generate_texts_with_openrouter_preserves_passing_rows_and_repairs_soft_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _StubPreset()
    prompts: list[str] = []
    _install_openrouter_stubs(
        monkeypatch,
        [
            {
                "parsed": {
                    "items": [
                        {"row_id": 0, "text": "pass:keep"},
                        {"row_id": 1, "text": "soft:fix"},
                    ]
                }
            }
        ],
        prompts,
    )

    texts = generate_texts_with_openrouter(
        preset,
        [{"label": "record-0"}, {"label": "record-1"}],
        [{"row_id": 0, "label": "record-0"}, {"row_id": 1, "label": "record-1"}],
        random.Random(7),
        "medium",
        config=_openrouter_config(batch_size=2, max_retries=0),
    )

    assert texts == ["pass:keep", "repaired:record-1:soft:fix"]
    assert preset.repair_calls == [({"label": "record-1"}, "soft:fix")]


def test_generate_texts_with_openrouter_raises_after_malformed_json_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _StubPreset()
    prompts: list[str] = []
    _install_openrouter_stubs(
        monkeypatch,
        [
            {
                "parsed": None,
                "parsing_error": ValueError("bad json"),
                "raw": types.SimpleNamespace(content="not json"),
            },
            {
                "parsed": None,
                "parsing_error": ValueError("still bad json"),
                "raw": types.SimpleNamespace(content="still not json"),
            },
        ],
        prompts,
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to generate a valid OpenRouter response for rows 0-0",
    ):
        generate_texts_with_openrouter(
            preset,
            [{"label": "record-0"}],
            [{"row_id": 0, "label": "record-0"}],
            random.Random(7),
            "medium",
            config=_openrouter_config(max_retries=1),
        )

    assert preset.repair_calls == []


def test_generate_texts_with_openrouter_repairs_after_terminal_hard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _StubPreset()
    prompts: list[str] = []
    _install_openrouter_stubs(
        monkeypatch,
        [{"parsed": {"items": [{"row_id": 0, "text": "hard:still wrong"}]}}],
        prompts,
    )

    texts = generate_texts_with_openrouter(
        preset,
        [{"label": "record-0"}],
        [{"row_id": 0, "label": "record-0"}],
        random.Random(7),
        "medium",
        config=_openrouter_config(max_retries=0),
    )

    assert texts == ["repaired:record-0:hard:still wrong"]
    assert preset.repair_calls == [({"label": "record-0"}, "hard:still wrong")]


def test_generate_texts_with_openrouter_preserves_passing_rows_and_repairs_mixed_terminal_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _StubPreset()
    prompts: list[str] = []
    _install_openrouter_stubs(
        monkeypatch,
        [
            {
                "parsed": {
                    "items": [
                        {"row_id": 0, "text": "pass:keep"},
                        {"row_id": 1, "text": "hard:still wrong"},
                    ]
                }
            }
        ],
        prompts,
    )

    texts = generate_texts_with_openrouter(
        preset,
        [{"label": "record-0"}, {"label": "record-1"}],
        [{"row_id": 0, "label": "record-0"}, {"row_id": 1, "label": "record-1"}],
        random.Random(7),
        "medium",
        config=_openrouter_config(batch_size=2, max_retries=0),
    )

    assert texts == ["pass:keep", "repaired:record-1:hard:still wrong"]
    assert preset.repair_calls == [({"label": "record-1"}, "hard:still wrong")]


def test_generate_texts_with_openrouter_regenerates_row_when_repair_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _StubPreset()
    prompts: list[str] = []
    _install_openrouter_stubs(
        monkeypatch,
        [
            {"parsed": {"items": [{"row_id": 0, "text": "hard:still wrong"}]}},
            {"parsed": {"items": [{"row_id": 0, "text": "pass:regenerated"}]}},
        ],
        prompts,
    )

    repair_attempts: list[tuple[dict[str, str], str]] = []

    def _raise_repair_error(record: dict[str, str], ticket_text: str) -> str:
        repair_attempts.append((record, ticket_text))
        raise ValueError("repair failed")

    monkeypatch.setattr(preset, "repair_generated_text", _raise_repair_error)

    texts = generate_texts_with_openrouter(
        preset,
        [{"label": "record-0"}],
        [{"row_id": 0, "label": "record-0"}],
        random.Random(7),
        "medium",
        config=_openrouter_config(max_retries=0),
    )

    assert texts == ["pass:regenerated"]
    assert repair_attempts == [({"label": "record-0"}, "hard:still wrong")]
    assert len(preset.prompt_batches) == 2


def test_generate_texts_with_openrouter_uses_emergency_fallback_when_repair_and_regeneration_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _StubPreset()
    prompts: list[str] = []
    _install_openrouter_stubs(
        monkeypatch,
        [
            {"parsed": {"items": [{"row_id": 0, "text": "hard:still wrong"}]}},
            RuntimeError("provider down"),
        ],
        prompts,
    )

    repair_attempts: list[tuple[dict[str, str], str]] = []

    def _raise_repair_error(record: dict[str, str], ticket_text: str) -> str:
        repair_attempts.append((record, ticket_text))
        raise ValueError("repair failed")

    monkeypatch.setattr(preset, "repair_generated_text", _raise_repair_error)

    texts = generate_texts_with_openrouter(
        preset,
        [{"label": "record-0", "issue_type": "access_request", "sub_intent": "temporary_access"}],
        [{"row_id": 0, "label": "record-0"}],
        random.Random(7),
        "medium",
        config=_openrouter_config(max_retries=0),
    )

    assert repair_attempts == [
        (
            {"label": "record-0", "issue_type": "access_request", "sub_intent": "temporary_access"},
            "hard:still wrong",
        )
    ]
    assert texts == ["fallback:record-0"]


def test_generate_texts_with_openrouter_raises_after_row_id_mismatch_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _StubPreset()
    prompts: list[str] = []
    _install_openrouter_stubs(
        monkeypatch,
        [
            {"parsed": {"items": [{"row_id": 1, "text": "wrong"}]}},
            {"parsed": {"items": [{"row_id": 1, "text": "still wrong"}]}},
        ],
        prompts,
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to generate a valid OpenRouter response for rows 0-0",
    ):
        generate_texts_with_openrouter(
            preset,
            [{"label": "record-0"}],
            [{"row_id": 0, "label": "record-0"}],
            random.Random(7),
            "medium",
            config=_openrouter_config(max_retries=1),
        )

    assert preset.repair_calls == []


def test_generate_dataset_rows_uses_generated_records_and_ticket_texts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [_sample_record(seed=7), _sample_record(seed=8)]

    monkeypatch.setattr(PRESET, "generate_records", lambda *args, **kwargs: records)
    monkeypatch.setattr(
        "dataset_generation.runtime.generate_texts_with_openrouter",
        lambda *args, **kwargs: ["text one", "text two"],
    )

    rows = generate_dataset_rows(
        PRESET,
        2,
        random.Random(7),
        mode="balanced",
        difficulty="medium",
        min_coverage_count=None,
        entity_mention_probability=None,
        openrouter_config=OpenRouterConfig(
            model="demo/model",
            temperature=0.7,
            batch_size=2,
            max_retries=0,
        ),
    )

    assert [row["ticket_text"] for row in rows] == ["text one", "text two"]
    assert rows[0]["issue_type"] == records[0]["issue_type"]
    assert set(rows[0].keys()) == set(spec.CSV_COLUMNS)


def test_build_dataset_summary_includes_access_share_and_schema_support() -> None:
    record = _sample_record()
    row = {"ticket_text": "Need VPN access for my new role", **record}

    summary = PRESET.build_summary(
        [row],
        mode="balanced",
        difficulty="medium",
        schema_support_report={},
        entity_mention_probability=0.4,
        openrouter_model="demo/model",
    )

    assert "Dataset summary" in summary
    assert "Text generation backend: openrouter" in summary
    assert "OpenRouter model: demo/model" in summary
    assert "System mention probability: 0.400" in summary
    assert "Access-related issue share: 1.000" in summary
    assert "Schema support audit: no unsupported declared labels found." in summary
    assert "intent_group x issue_type:" in summary
    assert "issue_type x sub_intent:" in summary


def test_write_dataset_csv_writes_expected_columns_and_rows(tmp_path: Path) -> None:
    record = _sample_record()
    row = {"ticket_text": "Need VPN access for my new role", **record}
    output_path = tmp_path / "enterprise.csv"

    write_dataset_csv(PRESET, str(output_path), [row])

    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == spec.CSV_COLUMNS
    assert rows == [row]


def test_build_parser_accepts_enterprise_dataset_flags() -> None:
    parser = build_parser("enterprise_ticket")
    args = parser.parse_args(
        [
            "--rows",
            "100",
            "--mode",
            "random",
            "--difficulty",
            "hard",
            "--batch-size",
            "10",
            "--evaluate-and-regenerate",
            "--evaluator-model",
            "demo/evaluator",
            "--max-evaluation-rounds",
            "6",
        ]
    )

    assert args.rows == 100
    assert args.mode == "random"
    assert args.difficulty == "hard"
    assert args.batch_size == 10
    assert args.evaluate_and_regenerate is True
    assert args.evaluator_model == "demo/evaluator"
    assert args.max_evaluation_rounds == 6


def test_build_parser_uses_enterprise_preset_defaults() -> None:
    parser = build_parser("enterprise_ticket")
    args = parser.parse_args([])

    assert args.output == "data/enterprise_ticket_dataset.csv"
    assert args.mode == "balanced"
    assert args.difficulty == "medium"
    assert args.evaluate_and_regenerate is True


def test_build_parser_accepts_generic_preset_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["--preset", "enterprise_ticket", "--rows", "10"])

    assert args.preset == "enterprise_ticket"
    assert args.rows == 10


def test_main_uses_selected_preset_defaults_for_generic_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataset_generation.cli import main

    captured_kwargs: dict[str, Any] = {}

    def _fake_bridge_runner(*args: Any, **kwargs: Any) -> int:
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return 0

    monkeypatch.setattr(
        "dataset_generation.cli.run_generation_with_evaluation_bridge",
        _fake_bridge_runner,
    )

    exit_code = main(["--preset", "enterprise_ticket"])

    assert exit_code == 0
    assert captured_kwargs["output"] == "data/enterprise_ticket_dataset.csv"
    assert captured_kwargs["mode"] == "balanced"
    assert captured_kwargs["difficulty"] == "medium"
    assert captured_kwargs["bridge_config"].max_rounds == 8


def test_main_preserves_explicit_overrides_for_generic_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataset_generation.cli import main

    captured_kwargs: dict[str, Any] = {}

    def _fake_bridge_runner(*args: Any, **kwargs: Any) -> int:
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return 0

    monkeypatch.setattr(
        "dataset_generation.cli.run_generation_with_evaluation_bridge",
        _fake_bridge_runner,
    )

    exit_code = main(
        [
            "--preset",
            "enterprise_ticket",
            "--output",
            "custom.csv",
            "--mode",
            "random",
            "--difficulty",
            "hard",
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["output"] == "custom.csv"
    assert captured_kwargs["mode"] == "random"
    assert captured_kwargs["difficulty"] == "hard"


def test_main_can_disable_bridge_mode_for_enterprise_ticket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataset_generation.cli import main

    captured_kwargs: dict[str, Any] = {}

    def _fake_run_generation(*args: Any, **kwargs: Any) -> int:
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return 0

    monkeypatch.setattr("dataset_generation.cli.run_generation", _fake_run_generation)

    exit_code = main(["--preset", "enterprise_ticket", "--no-evaluate-and-regenerate"])

    assert exit_code == 0
    assert captured_kwargs["output"] == "data/enterprise_ticket_dataset.csv"


def test_main_routes_bridge_mode_to_evaluation_bridge_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataset_generation.cli import main

    captured_kwargs: dict[str, Any] = {}

    def _fake_bridge_runner(*args: Any, **kwargs: Any) -> int:
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return 0

    monkeypatch.setattr(
        "dataset_generation.cli.run_generation_with_evaluation_bridge",
        _fake_bridge_runner,
    )

    exit_code = main(
        [
            "--preset",
            "enterprise_ticket",
            "--evaluate-and-regenerate",
            "--evaluator-model",
            "demo/evaluator",
            "--max-evaluation-rounds",
            "5",
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["evaluation_openrouter_config"].model == "demo/evaluator"
    assert captured_kwargs["bridge_config"].max_rounds == 5


def test_main_allows_bridge_mode_for_non_enterprise_presets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataset_generation.cli import main

    captured_kwargs: dict[str, Any] = {}

    def _fake_bridge_runner(*args: Any, **kwargs: Any) -> int:
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return 0

    monkeypatch.setattr(
        "dataset_generation.cli.PRESETS",
        {
            "enterprise_ticket": PRESET,
            "stub": _StubPreset(),
        },
    )
    monkeypatch.setattr(
        "dataset_generation.cli.run_generation_with_evaluation_bridge",
        _fake_bridge_runner,
    )

    exit_code = main(["--preset", "stub", "--evaluate-and-regenerate"])

    assert exit_code == 0
    assert captured_kwargs["bridge_config"].max_rounds == 8


def test_openrouter_config_accepts_zero_max_retries() -> None:
    config = OpenRouterConfig(
        model="demo/model",
        temperature=0.2,
        batch_size=10,
        max_retries=0,
    )

    assert config.max_retries == 0


def test_openrouter_config_rejects_negative_max_retries() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        OpenRouterConfig(
            model="demo/model",
            temperature=0.2,
            batch_size=10,
            max_retries=-1,
        )


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), float("-inf"), -1, 3])
def test_openrouter_config_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="'temperature' must be a number between 0.0 and 2.0"):
        OpenRouterConfig(
            model="demo/model",
            temperature=temperature,
            batch_size=10,
            max_retries=0,
        )


def test_openrouter_config_rejects_boolean_batch_size() -> None:
    with pytest.raises(ValueError, match="boolean values are not allowed"):
        OpenRouterConfig(
            model="demo/model",
            temperature=0.2,
            batch_size=True,
            max_retries=0,
        )


def test_openrouter_config_rejects_boolean_max_retries() -> None:
    with pytest.raises(ValueError, match="boolean values are not allowed"):
        OpenRouterConfig(
            model="demo/model",
            temperature=0.2,
            batch_size=10,
            max_retries=False,
        )


def test_main_reports_post_parse_probability_validation_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dataset_generation.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--preset", "enterprise_ticket", "--entity-mention-probability", "1.1"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "usage:" in captured.err
    assert "entity_mention_probability must be between 0 and 1 inclusive." in captured.err


def test_main_reports_post_parse_temperature_validation_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dataset_generation.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--preset", "enterprise_ticket", "--temperature", "3"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "usage:" in captured.err
    assert "'temperature' must be a number between 0.0 and 2.0" in captured.err


def test_main_reports_post_parse_evaluator_batch_size_validation_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dataset_generation.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--preset",
                "enterprise_ticket",
                "--evaluate-and-regenerate",
                "--batch-size",
                "10",
                "--evaluator-batch-size",
                "0",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "usage:" in captured.err
    assert "must be positive" in captured.err


def test_main_preserves_runtime_failures_outside_parser_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataset_generation.cli import main

    def _raise_runtime_failure(*args: Any, **kwargs: Any) -> int:
        raise RuntimeError("generation failed")

    monkeypatch.setattr(
        "dataset_generation.cli.run_generation_with_evaluation_bridge",
        _raise_runtime_failure,
    )

    with pytest.raises(RuntimeError, match="generation failed"):
        main(["--preset", "enterprise_ticket"])


def test_generate_texts_with_openrouter_uses_one_attempt_when_retries_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _StubPreset()
    prompts: list[str] = []
    client_kwargs: list[dict[str, Any]] = []
    _install_openrouter_stubs(
        monkeypatch,
        [{"parsed": {"items": [{"row_id": 0, "text": "soft:first"}]}}],
        prompts,
        client_kwargs=client_kwargs,
    )

    generate_texts_with_openrouter(
        preset,
        [{"label": "record-0"}],
        [{"row_id": 0, "label": "record-0"}],
        random.Random(7),
        "medium",
        config=_openrouter_config(max_retries=0),
    )

    assert len(prompts) == 1
    assert len(client_kwargs) == 1
    assert client_kwargs[0]["openrouter_api_base"] == "https://openrouter.ai/api/v1"
    assert client_kwargs[0]["model"] == "demo/model"
    assert client_kwargs[0]["temperature"] == 0.2
    assert client_kwargs[0]["max_retries"] == 0
    assert client_kwargs[0]["openrouter_api_key"] is not None


def test_generate_texts_with_openrouter_uses_two_total_attempts_for_one_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _StubPreset()
    prompts: list[str] = []
    _install_openrouter_stubs(
        monkeypatch,
        [
            {"parsed": {"items": [{"row_id": 0, "text": "soft:first"}]}},
            {"parsed": {"items": [{"row_id": 0, "text": "soft:second"}]}},
        ],
        prompts,
    )

    generate_texts_with_openrouter(
        preset,
        [{"label": "record-0"}],
        [{"row_id": 0, "label": "record-0"}],
        random.Random(7),
        "medium",
        config=_openrouter_config(max_retries=1),
    )

    assert len(prompts) == 2


def test_generate_texts_with_openrouter_retries_provider_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _StubPreset()
    prompts: list[str] = []
    _install_openrouter_stubs(
        monkeypatch,
        [
            RuntimeError("provider down"),
            {"parsed": {"items": [{"row_id": 0, "text": "pass:keep"}]}},
        ],
        prompts,
    )

    texts = generate_texts_with_openrouter(
        preset,
        [{"label": "record-0"}],
        [{"row_id": 0, "label": "record-0"}],
        random.Random(7),
        "medium",
        config=_openrouter_config(max_retries=1),
    )

    assert texts == ["pass:keep"]
    assert len(prompts) == 2


def test_generate_texts_with_openrouter_raises_after_provider_failure_without_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _StubPreset()
    prompts: list[str] = []
    _install_openrouter_stubs(
        monkeypatch,
        [RuntimeError("provider down")],
        prompts,
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to generate a valid OpenRouter response for rows 0-0",
    ):
        generate_texts_with_openrouter(
            preset,
            [{"label": "record-0"}],
            [{"row_id": 0, "label": "record-0"}],
            random.Random(7),
            "medium",
            config=_openrouter_config(max_retries=0),
        )

    assert preset.repair_calls == []
    assert len(prompts) == 1


def test_generate_texts_with_openrouter_limits_provider_attempts_to_retry_budget_then_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _StubPreset()
    prompts: list[str] = []
    _install_openrouter_stubs(
        monkeypatch,
        [
            RuntimeError("provider down"),
            RuntimeError("provider still down"),
            RuntimeError("provider exhausted"),
        ],
        prompts,
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to generate a valid OpenRouter response for rows 0-0",
    ):
        generate_texts_with_openrouter(
            preset,
            [{"label": "record-0"}],
            [{"row_id": 0, "label": "record-0"}],
            random.Random(7),
            "medium",
            config=_openrouter_config(max_retries=2),
        )

    assert preset.repair_calls == []
    assert len(prompts) == 3


def test_module_generate_structured_dataset_help_runs_from_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dataset_generation", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Generate a structured synthetic dataset from a registered preset." in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_module_generate_enterprise_ticket_dataset_help_runs_from_repo_root() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dataset_generation.presets.enterprise_ticket", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Generate a structured synthetic dataset from a registered preset." in result.stdout
    assert "ModuleNotFoundError" not in result.stderr
