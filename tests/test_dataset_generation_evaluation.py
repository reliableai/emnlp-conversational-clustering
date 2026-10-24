"""Tests for preset-driven dataset evaluation helpers and CLI wiring."""

from __future__ import annotations

import csv
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from dataset_generation.evaluation import (
    RowEvaluationBatch,
    evaluate_dataset_rows,
    evaluate_rows_with_openrouter,
    parse_row_evaluation_batch,
    read_dataset_rows,
    worse_rating,
)
from dataset_generation.presets.enterprise_ticket import PRESET
from dataset_generation.types import OpenRouterConfig


def _sample_row() -> dict[str, str]:
    return {
        "text": "Please grant temporary access in Okta for today.",
        "department": "IT",
        "sub_department": "IAM",
        "requester_role": "manager",
        "owning_team": "IAM Team",
        "ticket_category": "access_request",
        "issue_type": "access_request",
        "intent_group": "access_identity",
        "sub_intent": "temporary_access",
        "intent_label_coarse": "access_identity",
        "intent_label_normal": "access_identity__access_request",
        "intent_label_fine": "access_identity__access_request__temporary_access",
        "organizational_label_coarse": "it",
        "organizational_label_normal": "it__iam",
        "organizational_label_fine": "it__iam__iam_team",
        "system_name": "Okta",
        "business_event_context": "routine_operation",
        "workflow_stage": "new_submission",
        "approval_required": "none",
        "dependency_type": "none",
        "urgency": "medium",
        "severity": "moderate",
        "impact_scope": "single_user",
        "writing_style": "formal",
        "clarity_level": "clear",
    }


def _openrouter_config(*, batch_size: int = 10, max_retries: int = 0) -> OpenRouterConfig:
    return OpenRouterConfig(
        model="demo/model",
        temperature=0.0,
        batch_size=batch_size,
        max_retries=max_retries,
    )


def _install_openrouter_stubs(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[object],
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
            _ = messages
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


def test_worse_rating_returns_most_severe_rating() -> None:
    assert worse_rating("pass", "need_attention") == "need_attention"
    assert worse_rating("pass", "need_attention", "not_pass") == "not_pass"


def test_enterprise_preset_deterministic_evaluation_flags_issue_type_mismatch_as_not_pass() -> None:
    row = _sample_row()
    deterministic = PRESET.evaluate_row_deterministically(
        {field: row[field] for field in PRESET.schema.keys()},
        "My reimbursement is still missing in Concur after last week.",
    )

    assert deterministic.rating == "not_pass"
    assert "issue_type_mismatch" in deterministic.reasons


def test_enterprise_preset_deterministic_evaluation_flags_template_text() -> None:
    row = _sample_row()
    deterministic = PRESET.evaluate_row_deterministically(
        {field: row[field] for field in PRESET.schema.keys()},
        (
            "I need help with temporary access in Okta. "
            "This is still affecting my work and needs follow-up."
        ),
    )

    assert deterministic.rating == "need_attention"
    assert "template_like_fallback_text" in deterministic.reasons


def test_parse_row_evaluation_batch_rejects_row_id_mismatch() -> None:
    payload = RowEvaluationBatch.model_validate(
        {"items": [{"row_id": 1, "llm_rating": "pass", "llm_reason": "aligned"}]}
    )

    with pytest.raises(ValueError, match="row_ids do not match"):
        parse_row_evaluation_batch(payload, {0})


def test_evaluate_rows_with_openrouter_falls_back_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_openrouter_stubs(
        monkeypatch,
        [
            RuntimeError("provider down"),
            RuntimeError("provider still down"),
        ],
    )

    payloads = [PRESET.build_evaluation_row_payload(0, _sample_row())]
    results = evaluate_rows_with_openrouter(
        PRESET,
        payloads,
        config=_openrouter_config(max_retries=1),
    )

    assert results[0].llm_rating == "need_attention"
    assert "llm_evaluation_unavailable" in results[0].llm_reason


def test_evaluate_rows_with_openrouter_parses_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_openrouter_stubs(
        monkeypatch,
        [
            {
                "parsed": {
                    "items": [{"row_id": 0, "llm_rating": "pass", "llm_reason": "clear alignment"}]
                }
            }
        ],
    )

    payloads = [PRESET.build_evaluation_row_payload(0, _sample_row())]
    results = evaluate_rows_with_openrouter(
        PRESET,
        payloads,
        config=_openrouter_config(max_retries=0),
    )

    assert results[0].llm_rating == "pass"
    assert results[0].llm_reason == "clear alignment"


def test_evaluate_dataset_rows_uses_worst_of_deterministic_and_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _sample_row()
    row["ticket_text"] = "My reimbursement in Concur is still missing."

    def _fake_llm(*args: Any, **kwargs: Any) -> dict[int, Any]:
        _ = args, kwargs
        return {
            0: types.SimpleNamespace(
                row_id=0,
                llm_rating="pass",
                llm_reason="llm said pass",
            )
        }

    monkeypatch.setattr("dataset_generation.evaluation.evaluate_rows_with_openrouter", _fake_llm)

    evaluated_rows = evaluate_dataset_rows(PRESET, [row], config=_openrouter_config())

    assert evaluated_rows[0]["evaluation_llm_rating"] == "pass"
    assert evaluated_rows[0]["evaluation_final_rating"] == "not_pass"
    assert evaluated_rows[0]["evaluation_deterministic_details"] == (
        '{"banned_phrase": "", "issue_type_match": "false", "sub_intent_match": "false"}'
    )


def test_read_dataset_rows_requires_all_schema_columns(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    with dataset_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "department"])
        writer.writeheader()
        writer.writerow({"text": "hello", "department": "IT"})

    with pytest.raises(ValueError, match="missing required columns"):
        read_dataset_rows(PRESET, str(dataset_path))
