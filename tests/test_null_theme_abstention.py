"""Tests for the null-theme abstention evaluator and its runner integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conversational_clustering.evaluation.artifacts import LoadedClustering
from conversational_clustering.evaluation.base import EvaluationRequest
from conversational_clustering.evaluation.methods.null_theme_abstention import (
    NullThemeAbstentionEvaluator,
    RequestedTopic,
    _resolve_requested_topic,
)
from conversational_clustering.evaluation.methods.semantic_quality_judges import (
    NullThemeAbstentionJudgment,
    NullThemeUnsupportedTopicMatchJudgment,
)
from conversational_clustering.evaluation.runner import run_evaluation_config


def _predictions() -> LoadedClustering:
    return LoadedClustering(
        assignment={"apple": 0, "banana": 0, "car": 1, "truck": 1},
        cluster_names=["cluster_a", "cluster_b"],
    )


def _unsupported_topics() -> list[dict[str, object]]:
    return [
        {
            "topic_id": "astrology",
            "label": "Astrology",
            "guidance": "Group items by zodiac meaning.",
        }
    ]


def _conversation_trace(*messages: tuple[str, str]) -> dict[str, object]:
    return {
        "messages": [{"role": role, "content": content} for role, content in messages],
        "total_tool_calls_by_type": {},
        "total_failed_tool_calls_by_type": {},
    }


def _request(
    *,
    artifacts: dict[str, object] | None = None,
) -> EvaluationRequest:
    return EvaluationRequest(
        predictions=_predictions(),
        artifacts=artifacts or {},
        metadata={"config_slug": "cfg", "run_id": "run-1"},
    )


def _mock_topic_match_judge(
    monkeypatch: pytest.MonkeyPatch,
    *,
    matched_topic_id: str | None = "astrology",
    confidence: float = 0.88,
) -> None:
    from conversational_clustering.evaluation.methods import null_theme_abstention as module

    def mock_match(**kwargs: object) -> NullThemeUnsupportedTopicMatchJudgment:
        assert "requested_guidance" in kwargs
        assert "conversation_trace" not in kwargs
        return NullThemeUnsupportedTopicMatchJudgment(
            matched_unsupported_topic_id=matched_topic_id,
            confidence=confidence,
            reasoning="Synthetic topic match.",
        )

    monkeypatch.setattr(module, "judge_null_theme_unsupported_topic_match", mock_match)


def test_null_theme_abstention_from_params_accepts_valid_configuration() -> None:
    evaluator = NullThemeAbstentionEvaluator.from_params(
        {
            "unsupported_topics": _unsupported_topics(),
            "match_judge_prompt_id": "baseline_v1",
            "trace_judge_prompt_id": "baseline_v1",
            "model_config": {"model": "openai/gpt-5.4-mini"},
        }
    )

    assert evaluator.match_judge_prompt_id == "baseline_v1"
    assert evaluator.trace_judge_prompt_id == "baseline_v1"
    assert evaluator.model_config is not None
    assert evaluator.unsupported_topics[0].topic_id == "astrology"


def test_null_theme_abstention_from_params_requires_unsupported_topics() -> None:
    with pytest.raises(ValueError, match="requires 'unsupported_topics' as a non-empty list"):
        NullThemeAbstentionEvaluator.from_params({})


def test_null_theme_abstention_from_params_rejects_empty_unsupported_topics() -> None:
    with pytest.raises(ValueError, match="requires 'unsupported_topics' as a non-empty list"):
        NullThemeAbstentionEvaluator.from_params({"unsupported_topics": []})


def test_null_theme_abstention_from_params_rejects_missing_required_topic_fields() -> None:
    with pytest.raises(ValueError, match="missing required field\\(s\\): guidance"):
        NullThemeAbstentionEvaluator.from_params(
            {
                "unsupported_topics": [
                    {
                        "topic_id": "astrology",
                        "label": "Astrology",
                    }
                ]
            }
        )


def test_null_theme_abstention_from_params_rejects_unknown_topic_fields() -> None:
    with pytest.raises(ValueError, match="unsupported field\\(s\\): extra"):
        NullThemeAbstentionEvaluator.from_params(
            {
                "unsupported_topics": [
                    {
                        "topic_id": "astrology",
                        "label": "Astrology",
                        "guidance": "Group items by zodiac meaning.",
                        "extra": "nope",
                    }
                ]
            }
        )


def test_null_theme_abstention_from_params_rejects_legacy_aliases() -> None:
    with pytest.raises(ValueError, match="unsupported field\\(s\\): aliases"):
        NullThemeAbstentionEvaluator.from_params(
            {
                "unsupported_topics": [
                    {
                        "topic_id": "astrology",
                        "label": "Astrology",
                        "guidance": "Group items by zodiac meaning.",
                        "aliases": ["zodiac"],
                    }
                ]
            }
        )


def test_null_theme_abstention_from_params_rejects_unknown_evaluator_params() -> None:
    with pytest.raises(ValueError, match="received unsupported parameter\\(s\\): nope"):
        NullThemeAbstentionEvaluator.from_params(
            {
                "unsupported_topics": _unsupported_topics(),
                "nope": True,
            }
        )


def test_null_theme_abstention_from_params_rejects_invalid_trace_judge_prompt_id() -> None:
    with pytest.raises(ValueError, match="Invalid trace_judge_prompt_id"):
        NullThemeAbstentionEvaluator.from_params(
            {
                "unsupported_topics": _unsupported_topics(),
                "trace_judge_prompt_id": "does_not_exist",
            }
        )


def test_null_theme_abstention_from_params_rejects_invalid_match_judge_prompt_id() -> None:
    with pytest.raises(ValueError, match="Invalid match_judge_prompt_id"):
        NullThemeAbstentionEvaluator.from_params(
            {
                "unsupported_topics": _unsupported_topics(),
                "match_judge_prompt_id": "does_not_exist",
            }
        )


def test_null_theme_abstention_from_params_rejects_invalid_model_config() -> None:
    with pytest.raises(TypeError, match="model_config must be a dict"):
        NullThemeAbstentionEvaluator.from_params(
            {
                "unsupported_topics": _unsupported_topics(),
                "model_config": "not-a-dict",
            }
        )


def test_null_theme_abstention_evaluator_scores_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from conversational_clustering.evaluation.methods import null_theme_abstention as module

    def mock_judge(**kwargs: object) -> NullThemeAbstentionJudgment:
        assert "human: cluster these by zodiac sign" in str(kwargs["conversation_trace"])
        return NullThemeAbstentionJudgment(
            rejected_requested_pov=True,
            assistant_agreed_with_requested_pov=False,
            confidence=0.91,
            reasoning="The assistant rejects the zodiac framing.",
        )

    _mock_topic_match_judge(monkeypatch)
    monkeypatch.setattr(module, "judge_null_theme_abstention_trace", mock_judge)
    evaluator = NullThemeAbstentionEvaluator.from_params(
        {"unsupported_topics": _unsupported_topics()}
    )
    request = _request(
        artifacts={
            "selected_topic": {
                "topic_id": "astrology",
                "label": "Astrology",
                "guidance": "Group items by zodiac meaning.",
            },
            "conversation_trace": _conversation_trace(
                ("human", "cluster these by zodiac sign"),
                (
                    "assistant",
                    "I can't cluster on zodiac meaning because the data doesn't support it.",
                ),
                ("system", "ignored"),
            ),
        }
    )

    result = evaluator.evaluate(request)

    assert result.metrics == {
        "abstention_success": 1.0,
        "explicit_rejection_detected": 1.0,
        "assistant_agreement_detected": 0.0,
    }
    assert result.details["resolved_requested_topic"] == {
        "topic_id": "astrology",
        "label": "Astrology",
        "guidance": "Group items by zodiac meaning.",
    }
    assert result.details["matched_unsupported_topic"] == {
        "topic_id": "astrology",
        "label": "Astrology",
        "guidance": "Group items by zodiac meaning.",
    }
    assert result.details["topic_match_judgment"] == {
        "matched_unsupported_topic_id": "astrology",
        "confidence": 0.88,
        "reasoning": "Synthetic topic match.",
    }
    assert "system:" not in result.details["trace_transcript"]


def test_null_theme_abstention_evaluator_scores_failure_without_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import null_theme_abstention as module

    def mock_judge(**_: object) -> NullThemeAbstentionJudgment:
        return NullThemeAbstentionJudgment(
            rejected_requested_pov=False,
            assistant_agreed_with_requested_pov=False,
            confidence=0.7,
            reasoning="No rejection.",
        )

    _mock_topic_match_judge(monkeypatch)
    monkeypatch.setattr(module, "judge_null_theme_abstention_trace", mock_judge)
    evaluator = NullThemeAbstentionEvaluator.from_params(
        {"unsupported_topics": _unsupported_topics()}
    )
    request = _request(
        artifacts={
            "batch_run_config": {
                "human_agent_config": {"guidance": "Group items by zodiac meaning."}
            },
            "conversation_trace": _conversation_trace(
                ("human", "Group these by zodiac meaning."),
                ("assistant", "I'll take a look."),
            ),
        }
    )

    result = evaluator.evaluate(request)

    assert result.metrics["abstention_success"] == 0.0
    assert result.metrics["explicit_rejection_detected"] == 0.0
    assert result.metrics["assistant_agreement_detected"] == 0.0


def test_null_theme_abstention_evaluator_scores_failure_when_assistant_agrees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import null_theme_abstention as module

    def mock_judge(**_: object) -> NullThemeAbstentionJudgment:
        return NullThemeAbstentionJudgment(
            rejected_requested_pov=True,
            assistant_agreed_with_requested_pov=True,
            confidence=0.82,
            reasoning="The assistant hedges but still agrees.",
        )

    _mock_topic_match_judge(monkeypatch)
    monkeypatch.setattr(module, "judge_null_theme_abstention_trace", mock_judge)
    evaluator = NullThemeAbstentionEvaluator.from_params(
        {"unsupported_topics": _unsupported_topics()}
    )
    request = _request(
        artifacts={
            "batch_run_config": {
                "human_agent_config": {"guidance": "Group items by zodiac meaning."}
            },
            "conversation_trace": _conversation_trace(
                ("human", "Cluster these by zodiac meaning."),
                ("assistant", "That may be weak, but I'll try clustering by zodiac anyway."),
            ),
        }
    )

    result = evaluator.evaluate(request)

    assert result.metrics["abstention_success"] == 0.0
    assert result.metrics["explicit_rejection_detected"] == 1.0
    assert result.metrics["assistant_agreement_detected"] == 1.0


def test_null_theme_abstention_evaluator_requires_conversation_trace() -> None:
    evaluator = NullThemeAbstentionEvaluator.from_params(
        {"unsupported_topics": _unsupported_topics()}
    )
    request = _request(
        artifacts={
            "batch_run_config": {
                "human_agent_config": {"guidance": "Group items by zodiac meaning."}
            }
        }
    )

    with pytest.raises(ValueError, match="requires 'conversation_trace'"):
        evaluator.evaluate(request)


def test_null_theme_abstention_evaluator_requires_requested_topic_resolution() -> None:
    evaluator = NullThemeAbstentionEvaluator.from_params(
        {"unsupported_topics": _unsupported_topics()}
    )
    request = _request(artifacts={"conversation_trace": _conversation_trace(("human", "hello"))})

    with pytest.raises(ValueError, match="could not resolve the requested unsupported POV"):
        evaluator.evaluate(request)


def test_null_theme_abstention_evaluator_rejects_unmatched_requested_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_topic_match_judge(monkeypatch, matched_topic_id=None)
    evaluator = NullThemeAbstentionEvaluator.from_params(
        {"unsupported_topics": _unsupported_topics()}
    )
    request = _request(
        artifacts={
            "batch_run_config": {
                "human_agent_config": {"guidance": "Group items by vehicle type."}
            },
            "conversation_trace": _conversation_trace(("human", "group by vehicle type")),
        }
    )

    with pytest.raises(ValueError, match="could not match the initial requested POV"):
        evaluator.evaluate(request)


def test_null_theme_abstention_evaluator_uses_topic_match_judge_for_nonliteral_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import null_theme_abstention as module

    def mock_match(**kwargs: object) -> NullThemeUnsupportedTopicMatchJudgment:
        assert kwargs["requested_guidance"] == "Group items by cosmic archetypes."
        assert "conversation_trace" not in kwargs
        return NullThemeUnsupportedTopicMatchJudgment(
            matched_unsupported_topic_id="astrology",
            confidence=0.93,
            reasoning="The initial human request asks for a zodiac-style clustering.",
        )

    def mock_judge(**kwargs: object) -> NullThemeAbstentionJudgment:
        assert "topic_id=astrology" in str(kwargs["matched_unsupported_topic"])
        return NullThemeAbstentionJudgment(
            rejected_requested_pov=True,
            assistant_agreed_with_requested_pov=False,
            confidence=0.9,
            reasoning="Rejected.",
        )

    monkeypatch.setattr(module, "judge_null_theme_unsupported_topic_match", mock_match)
    monkeypatch.setattr(module, "judge_null_theme_abstention_trace", mock_judge)
    evaluator = NullThemeAbstentionEvaluator.from_params(
        {"unsupported_topics": _unsupported_topics()}
    )
    request = _request(
        artifacts={
            "batch_run_config": {
                "human_agent_config": {"guidance": "Group items by cosmic archetypes."}
            },
            "conversation_trace": _conversation_trace(
                ("human", "Can you cluster these by zodiac sign?"),
                ("assistant", "I can't use zodiac meaning because the data does not support it."),
            ),
        }
    )

    result = evaluator.evaluate(request)

    assert result.metrics["abstention_success"] == 1.0
    assert result.details["topic_match_judgment"]["matched_unsupported_topic_id"] == "astrology"


def test_resolve_requested_topic_prefers_selected_topic() -> None:
    topic = _resolve_requested_topic(
        {
            "selected_topic": {
                "topic_id": "astrology",
                "label": "Astrology",
                "guidance": "Group items by zodiac meaning.",
            },
            "batch_run_config": {
                "metadata": {
                    "topic_branch_topic_id": "vehicle",
                    "topic_branch_topic_label": "Vehicle",
                    "topic_branch_topic_guidance": "Group items by vehicle type.",
                },
                "human_agent_config": {"guidance": "Group items by fruitness."},
            },
        },
        evaluator_name="null_theme_abstention",
    )

    assert topic == RequestedTopic(
        topic_id="astrology",
        label="Astrology",
        guidance="Group items by zodiac meaning.",
    )


def test_resolve_requested_topic_falls_back_to_batch_metadata() -> None:
    topic = _resolve_requested_topic(
        {
            "batch_run_config": {
                "metadata": {
                    "topic_branch_topic_id": "astrology",
                    "topic_branch_topic_label": "Astrology",
                    "topic_branch_topic_guidance": "Group items by zodiac meaning.",
                },
                "human_agent_config": {"guidance": "Group items by vehicle type."},
            }
        },
        evaluator_name="null_theme_abstention",
    )

    assert topic == RequestedTopic(
        topic_id="astrology",
        label="Astrology",
        guidance="Group items by zodiac meaning.",
    )


def test_resolve_requested_topic_falls_back_to_guidance() -> None:
    topic = _resolve_requested_topic(
        {
            "batch_run_config": {
                "human_agent_config": {"guidance": "Group items by zodiac meaning."}
            }
        },
        evaluator_name="null_theme_abstention",
    )

    assert topic == RequestedTopic(
        topic_id=None,
        label=None,
        guidance="Group items by zodiac meaning.",
    )


def test_run_evaluation_config_executes_null_theme_abstention_and_averages_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import null_theme_abstention as module

    def mock_judge(**kwargs: object) -> NullThemeAbstentionJudgment:
        trace_text = str(kwargs["conversation_trace"])
        rejected = "can't" in trace_text.lower()
        agreed = "i'll cluster by zodiac" in trace_text.lower()
        return NullThemeAbstentionJudgment(
            rejected_requested_pov=rejected,
            assistant_agreed_with_requested_pov=agreed,
            confidence=0.8,
            reasoning="Synthetic test judgment.",
        )

    _mock_topic_match_judge(monkeypatch)
    monkeypatch.setattr(module, "judge_null_theme_abstention_trace", mock_judge)

    input_dir = tmp_path / "batch_output"
    config_slug = "single-config"
    run_payload = {
        "clusters": [
            {"name": "A", "members": ["alpha", "beta"]},
            {"name": "B", "members": ["gamma"]},
        ]
    }
    for run_id, assistant_message in {
        "001-run": "I can't cluster by zodiac meaning because the data doesn't support it.",
        "002-run": "I'll cluster by zodiac meaning anyway.",
    }.items():
        run_dir = input_dir / config_slug / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "clustering.json").write_text(json.dumps(run_payload), encoding="utf-8")
        (run_dir / "conversation_trace.json").write_text(
            json.dumps(
                _conversation_trace(
                    ("human", "Cluster these by zodiac meaning."),
                    ("assistant", assistant_message),
                )
            ),
            encoding="utf-8",
        )

    (input_dir / "results.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "config": {
                            "name": "Single",
                            "human_agent_config": {"guidance": "Group items by zodiac meaning."},
                        },
                        "config_slug": config_slug,
                        "run_ids": ["001-run", "002-run"],
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report_path = tmp_path / "null_theme_report.json"
    config_path = tmp_path / "evaluation.toml"
    config_path.write_text(
        "\n".join(
            [
                f"input_dir = '{input_dir}'",
                'evaluators = ["null_theme_abstention"]',
                f"report_path = '{report_path}'",
                "",
                "[evaluator_params.null_theme_abstention]",
                'match_judge_prompt_id = "baseline_v1"',
                'trace_judge_prompt_id = "baseline_v1"',
                (
                    'unsupported_topics = [{ topic_id = "astrology", label = "Astrology", '
                    'guidance = "Group items by zodiac meaning." }]'
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    returned_report_path = run_evaluation_config(config_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    details_path = (
        report_path.parent / f"{report_path.stem}_details" / "null_theme_abstention.jsonl"
    )
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert returned_report_path == str(report_path)
    assert report["summary_by_evaluator"]["null_theme_abstention"]["metrics"] == {
        "abstention_success": 0.5,
        "assistant_agreement_detected": 0.5,
        "explicit_rejection_detected": 0.5,
    }
    assert report["summary_by_evaluator"]["null_theme_abstention"]["details"]["num_results"] == 2
    assert len(detail_records) == 2
    assert detail_records[0]["method"] == "null_theme_abstention"


def test_run_evaluation_config_executes_null_theme_abstention_without_clustering_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import null_theme_abstention as module

    def mock_judge(**_: object) -> NullThemeAbstentionJudgment:
        return NullThemeAbstentionJudgment(
            rejected_requested_pov=True,
            assistant_agreed_with_requested_pov=False,
            confidence=0.8,
            reasoning="Synthetic test judgment.",
        )

    _mock_topic_match_judge(monkeypatch)
    monkeypatch.setattr(module, "judge_null_theme_abstention_trace", mock_judge)

    input_dir = tmp_path / "batch_output"
    config_slug = "single-config"
    run_id = "001-run"
    run_dir = input_dir / config_slug / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "conversation_trace.json").write_text(
        json.dumps(
            _conversation_trace(
                ("human", "Cluster these by zodiac meaning."),
                ("assistant", "I can't cluster by zodiac meaning."),
            )
        ),
        encoding="utf-8",
    )
    (input_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "config": {
                    "name": "Single",
                    "human_agent_config": {"guidance": "Group items by zodiac meaning."},
                },
                "config_slug": config_slug,
                "run_ids": [run_id],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report_path = tmp_path / "null_theme_report.json"
    config_path = tmp_path / "evaluation.toml"
    config_path.write_text(
        "\n".join(
            [
                f"input_dir = '{input_dir}'",
                'evaluators = ["null_theme_abstention"]',
                f"report_path = '{report_path}'",
                "",
                "[evaluator_params.null_theme_abstention]",
                'match_judge_prompt_id = "baseline_v1"',
                'trace_judge_prompt_id = "baseline_v1"',
                (
                    'unsupported_topics = [{ topic_id = "astrology", label = "Astrology", '
                    'guidance = "Group items by zodiac meaning." }]'
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    returned_report_path = run_evaluation_config(config_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    details_path = (
        report_path.parent / f"{report_path.stem}_details" / "null_theme_abstention.jsonl"
    )
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert returned_report_path == str(report_path)
    assert report["summary_by_evaluator"]["null_theme_abstention"]["metrics"] == {
        "abstention_success": 1.0,
        "assistant_agreement_detected": 0.0,
        "explicit_rejection_detected": 1.0,
    }
    assert detail_records[0]["clustering_path"] is None
