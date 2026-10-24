"""Tests for the clarification-policy evaluator and its trace requirements."""

from __future__ import annotations

import pytest

from conversational_clustering.evaluation.artifacts import LoadedClustering
from conversational_clustering.evaluation.base import EvaluationRequest
from conversational_clustering.evaluation.benchmark_spec import (
    ReferenceClusteringSpec,
    SyntheticBenchmarkSpec,
    ThemeSpec,
)
from conversational_clustering.evaluation.methods.clarification_policy import (
    ClarificationPolicyEvaluator,
)
from conversational_clustering.evaluation.methods.semantic_quality_judges import (
    ClarificationOpportunityJudgment,
)


def _benchmark_spec() -> SyntheticBenchmarkSpec:
    return SyntheticBenchmarkSpec(
        dataset_name="demo_dataset",
        item_texts=("apple", "banana", "car", "truck"),
        themes=(
            ThemeSpec(
                theme_id="topic",
                label="Topic",
                guidance="Group items by topic.",
            ),
        ),
        references=(
            ReferenceClusteringSpec(
                reference_id="topic_normal",
                topic_id="topic",
                granularity="normal",
                assignment={
                    "apple": 0,
                    "banana": 0,
                    "car": 1,
                    "truck": 1,
                },
                cluster_names=["fruits", "vehicles"],
            ),
        ),
    )


def _predictions() -> LoadedClustering:
    return LoadedClustering(
        assignment={"apple": 0, "banana": 0, "car": 1, "truck": 1},
        cluster_names=["fruits", "vehicles"],
    )


def test_clarification_policy_from_params_uses_defaults() -> None:
    evaluator = ClarificationPolicyEvaluator.from_params(None)

    assert evaluator.model_config is None
    assert evaluator.prompt_id == "baseline_v1"


def test_clarification_policy_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match="unsupported parameter"):
        ClarificationPolicyEvaluator.from_params({"unknown": 1})

    with pytest.raises(TypeError, match="model_config must be a dict"):
        ClarificationPolicyEvaluator.from_params({"model_config": "bad"})


def test_clarification_policy_requires_message_list() -> None:
    evaluator = ClarificationPolicyEvaluator.from_params(None)
    request = EvaluationRequest(
        predictions=_predictions(),
        artifacts={
            "conversation_trace": {"messages": "invalid"},
            "benchmark_spec": _benchmark_spec(),
        },
    )

    with pytest.raises(ValueError, match="messages to be a list"):
        evaluator.evaluate(request)


def test_clarification_policy_rejects_malformed_message() -> None:
    evaluator = ClarificationPolicyEvaluator.from_params(None)
    request = EvaluationRequest(
        predictions=_predictions(),
        artifacts={
            "conversation_trace": {
                "messages": ["not-an-object"],
            },
            "benchmark_spec": _benchmark_spec(),
        },
    )

    with pytest.raises(ValueError, match="entries must be objects"):
        evaluator.evaluate(request)


def test_clarification_policy_aggregates_useful_and_missed_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import clarification_policy as module

    captured_traces: list[str] = []
    captured_turn_indices: list[int] = []
    captured_request_texts: list[str] = []
    judgments = iter(
        [
            ClarificationOpportunityJudgment(
                opportunity_exists=True,
                clarification_needed=True,
                asked_clarification=True,
                useful_clarification=True,
                unnecessary_clarification=False,
                missed_clarification=False,
                downstream_gain=0.8,
                reasoning="The clarification resolved a real ambiguity.",
            ),
            ClarificationOpportunityJudgment(
                opportunity_exists=True,
                clarification_needed=True,
                asked_clarification=False,
                useful_clarification=False,
                unnecessary_clarification=False,
                missed_clarification=True,
                downstream_gain=0.4,
                reasoning="The assistant should have asked before committing.",
            ),
            ClarificationOpportunityJudgment(
                opportunity_exists=False,
                clarification_needed=False,
                asked_clarification=False,
                useful_clarification=False,
                unnecessary_clarification=False,
                missed_clarification=False,
                downstream_gain=0.0,
                reasoning="Final request was already specific enough.",
            ),
        ]
    )

    def _fake_judge(**kwargs: object) -> ClarificationOpportunityJudgment:
        captured_traces.append(str(kwargs["conversation_trace"]))
        request_turn_index = kwargs["request_turn_index"]
        assert isinstance(request_turn_index, int)
        captured_turn_indices.append(request_turn_index)
        captured_request_texts.append(str(kwargs["request_text"]))
        return next(judgments)

    monkeypatch.setattr(module, "judge_clarification_opportunity", _fake_judge)

    evaluator = ClarificationPolicyEvaluator.from_params(None)
    request = EvaluationRequest(
        predictions=_predictions(),
        artifacts={
            "conversation_trace": {
                "messages": [
                    {"role": "human", "content": "Group these items."},
                    {
                        "role": "assistant",
                        "content": "Could you clarify if you want topic or sentiment clusters?",
                    },
                    {"role": "human", "content": "By topic."},
                    {"role": "assistant", "content": "Understood."},
                    {"role": "human", "content": "Split them into sensible buckets."},
                    {"role": "assistant", "content": "I grouped them broadly."},
                ]
            },
            "benchmark_spec": _benchmark_spec(),
            "batch_run_config": {
                "human_agent_config": {"guidance": "Group items by topic."},
            },
        },
    )

    result = evaluator.evaluate(request)

    assert result.method == "clarification_policy"
    assert result.metrics == {
        "useful_clarification_rate": 1.0,
        "unnecessary_clarification_rate": 0.0,
        "missed_clarification_rate": 0.5,
        "downstream_gain_from_clarification": 0.8,
    }
    assert result.details["num_opportunities"] == 2
    assert result.details["num_needed_clarifications"] == 2
    assert result.details["num_asked_clarifications"] == 1
    assert result.details["num_useful_clarifications"] == 1
    assert result.details["num_unnecessary_clarifications"] == 0
    assert result.details["num_missed_clarifications"] == 1
    assert result.details["sum_downstream_gain"] == 0.8
    assert result.details["mean_downstream_gain_for_asked"] == 0.8
    assert len(result.details["opportunities"]) == 2
    full_trace = (
        "Turn 1 human: Group these items.\n"
        "Turn 1 assistant: Could you clarify if you want topic or sentiment clusters?\n"
        "Turn 2 human: By topic.\n"
        "Turn 2 assistant: Understood.\n"
        "Turn 3 human: Split them into sensible buckets.\n"
        "Turn 3 assistant: I grouped them broadly."
    )
    assert captured_traces == [full_trace, full_trace, full_trace]
    assert captured_turn_indices == [1, 2, 3]
    assert captured_request_texts == [
        "Group these items.",
        "By topic.",
        "Split them into sensible buckets.",
    ]
    assert result.details["opportunities"][0]["conversation_trace"] == full_trace


def test_clarification_policy_zeroes_gain_for_unasked_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import clarification_policy as module

    def _fake_judge(**kwargs: object) -> ClarificationOpportunityJudgment:
        _ = kwargs
        return ClarificationOpportunityJudgment(
            opportunity_exists=True,
            clarification_needed=True,
            asked_clarification=False,
            useful_clarification=False,
            unnecessary_clarification=False,
            missed_clarification=True,
            downstream_gain=0.9,
            reasoning="Clarification was missed.",
        )

    monkeypatch.setattr(module, "judge_clarification_opportunity", _fake_judge)

    evaluator = ClarificationPolicyEvaluator.from_params(None)
    request = EvaluationRequest(
        predictions=_predictions(),
        artifacts={
            "conversation_trace": {
                "messages": [
                    {"role": "human", "content": "Group these items."},
                    {"role": "assistant", "content": "I grouped them broadly."},
                ]
            },
            "benchmark_spec": _benchmark_spec(),
            "batch_run_config": {
                "human_agent_config": {"guidance": "Group items by topic."},
            },
        },
    )

    result = evaluator.evaluate(request)

    assert result.metrics["downstream_gain_from_clarification"] == 0.0
    assert result.details["num_asked_clarifications"] == 0
    assert result.details["num_missed_clarifications"] == 1


def test_clarification_policy_keeps_follow_up_clarification_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import clarification_policy as module

    judgments = iter(
        [
            ClarificationOpportunityJudgment(
                opportunity_exists=True,
                clarification_needed=True,
                asked_clarification=True,
                useful_clarification=True,
                unnecessary_clarification=False,
                missed_clarification=False,
                downstream_gain=0.6,
                reasoning="First clarification helped.",
            ),
            ClarificationOpportunityJudgment(
                opportunity_exists=True,
                clarification_needed=True,
                asked_clarification=True,
                useful_clarification=True,
                unnecessary_clarification=False,
                missed_clarification=False,
                downstream_gain=0.7,
                reasoning="Follow-up clarification helped.",
            ),
            ClarificationOpportunityJudgment(
                opportunity_exists=False,
                clarification_needed=False,
                asked_clarification=False,
                useful_clarification=False,
                unnecessary_clarification=False,
                missed_clarification=False,
                downstream_gain=0.0,
                reasoning="Final answer turn is not an opportunity.",
            ),
        ]
    )

    def _fake_judge(**kwargs: object) -> ClarificationOpportunityJudgment:
        _ = kwargs
        return next(judgments)

    monkeypatch.setattr(module, "judge_clarification_opportunity", _fake_judge)

    evaluator = ClarificationPolicyEvaluator.from_params(None)
    request = EvaluationRequest(
        predictions=_predictions(),
        artifacts={
            "conversation_trace": {
                "messages": [
                    {"role": "human", "content": "Group these items."},
                    {"role": "assistant", "content": "By topic or sentiment?"},
                    {"role": "human", "content": "By topic, but keep urgent tickets separate."},
                    {"role": "assistant", "content": "Should urgency override topic grouping?"},
                    {"role": "human", "content": "No, only separate obviously urgent ones."},
                ]
            },
            "benchmark_spec": _benchmark_spec(),
            "batch_run_config": {
                "human_agent_config": {"guidance": "Group items by topic."},
            },
        },
    )

    result = evaluator.evaluate(request)

    assert result.details["num_opportunities"] == 2
    assert result.details["num_asked_clarifications"] == 2
    assert result.details["num_useful_clarifications"] == 2
    assert result.metrics["useful_clarification_rate"] == 1.0


def test_clarification_policy_does_not_count_unnecessary_clarifications_as_opportunities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import clarification_policy as module

    def _fake_judge(**kwargs: object) -> ClarificationOpportunityJudgment:
        _ = kwargs
        return ClarificationOpportunityJudgment(
            opportunity_exists=False,
            clarification_needed=False,
            asked_clarification=True,
            useful_clarification=False,
            unnecessary_clarification=True,
            missed_clarification=False,
            downstream_gain=0.0,
            reasoning="The assistant asked an avoidable question.",
        )

    monkeypatch.setattr(module, "judge_clarification_opportunity", _fake_judge)

    evaluator = ClarificationPolicyEvaluator.from_params(None)
    request = EvaluationRequest(
        predictions=_predictions(),
        artifacts={
            "conversation_trace": {
                "messages": [
                    {"role": "human", "content": "Group these by topic."},
                    {"role": "assistant", "content": "Do you want me to proceed?"},
                    {"role": "human", "content": "Yes."},
                ]
            },
            "benchmark_spec": _benchmark_spec(),
            "batch_run_config": {
                "human_agent_config": {"guidance": "Group items by topic."},
            },
        },
    )

    result = evaluator.evaluate(request)

    assert result.details["num_opportunities"] == 0
    assert result.details["num_asked_clarifications"] == 2
    assert result.details["num_unnecessary_clarifications"] == 2
    assert result.details["opportunities"] == []


def test_clarification_policy_uses_full_trace_for_each_evaluated_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import clarification_policy as module

    captured_traces: list[str] = []
    captured_turn_indices: list[int] = []
    captured_request_texts: list[str] = []
    judgments = iter(
        [
            ClarificationOpportunityJudgment(
                opportunity_exists=False,
                clarification_needed=False,
                asked_clarification=False,
                useful_clarification=False,
                unnecessary_clarification=False,
                missed_clarification=False,
                downstream_gain=0.0,
                reasoning="First request was fully specified.",
            ),
            ClarificationOpportunityJudgment(
                opportunity_exists=False,
                clarification_needed=False,
                asked_clarification=False,
                useful_clarification=False,
                unnecessary_clarification=False,
                missed_clarification=False,
                downstream_gain=0.0,
                reasoning="Second request was fully specified.",
            ),
        ]
    )

    def _fake_judge(**kwargs: object) -> ClarificationOpportunityJudgment:
        captured_traces.append(str(kwargs["conversation_trace"]))
        request_turn_index = kwargs["request_turn_index"]
        assert isinstance(request_turn_index, int)
        captured_turn_indices.append(request_turn_index)
        captured_request_texts.append(str(kwargs["request_text"]))
        return next(judgments)

    monkeypatch.setattr(module, "judge_clarification_opportunity", _fake_judge)

    evaluator = ClarificationPolicyEvaluator.from_params(None)
    request = EvaluationRequest(
        predictions=_predictions(),
        artifacts={
            "conversation_trace": {
                "messages": [
                    {"role": "human", "content": "Group these by topic."},
                    {"role": "assistant", "content": "I grouped them into topic buckets."},
                    {"role": "human", "content": "Now regroup them by urgency."},
                    {"role": "assistant", "content": "I regrouped them by urgency."},
                ]
            },
            "benchmark_spec": _benchmark_spec(),
            "batch_run_config": {
                "human_agent_config": {"guidance": "Group items by topic."},
            },
        },
    )

    evaluator.evaluate(request)

    full_trace = (
        "Turn 1 human: Group these by topic.\n"
        "Turn 1 assistant: I grouped them into topic buckets.\n"
        "Turn 2 human: Now regroup them by urgency.\n"
        "Turn 2 assistant: I regrouped them by urgency."
    )
    assert captured_traces == [full_trace, full_trace]
    assert captured_turn_indices == [1, 2]
    assert captured_request_texts == ["Group these by topic.", "Now regroup them by urgency."]


def test_clarification_policy_includes_full_trace_for_follow_up_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import clarification_policy as module

    captured_traces: list[str] = []
    captured_turn_indices: list[int] = []
    captured_request_texts: list[str] = []
    judgments = iter(
        [
            ClarificationOpportunityJudgment(
                opportunity_exists=True,
                clarification_needed=True,
                asked_clarification=True,
                useful_clarification=True,
                unnecessary_clarification=False,
                missed_clarification=False,
                downstream_gain=0.5,
                reasoning="Initial request needed clarification.",
            ),
            ClarificationOpportunityJudgment(
                opportunity_exists=False,
                clarification_needed=False,
                asked_clarification=False,
                useful_clarification=False,
                unnecessary_clarification=False,
                missed_clarification=False,
                downstream_gain=0.0,
                reasoning="Follow-up answer resolved the prior clarification.",
            ),
        ]
    )

    def _fake_judge(**kwargs: object) -> ClarificationOpportunityJudgment:
        captured_traces.append(str(kwargs["conversation_trace"]))
        request_turn_index = kwargs["request_turn_index"]
        assert isinstance(request_turn_index, int)
        captured_turn_indices.append(request_turn_index)
        captured_request_texts.append(str(kwargs["request_text"]))
        return next(judgments)

    monkeypatch.setattr(module, "judge_clarification_opportunity", _fake_judge)

    evaluator = ClarificationPolicyEvaluator.from_params(None)
    request = EvaluationRequest(
        predictions=_predictions(),
        artifacts={
            "conversation_trace": {
                "messages": [
                    {"role": "human", "content": "Group these items."},
                    {
                        "role": "assistant",
                        "content": "Please clarify whether topic or urgency matters more",
                    },
                    {"role": "human", "content": "Topic matters more."},
                ]
            },
            "benchmark_spec": _benchmark_spec(),
            "batch_run_config": {
                "human_agent_config": {"guidance": "Group items by topic."},
            },
        },
    )

    evaluator.evaluate(request)

    full_trace = (
        "Turn 1 human: Group these items.\n"
        "Turn 1 assistant: Please clarify whether topic or urgency matters more\n"
        "Turn 2 human: Topic matters more."
    )
    assert captured_traces == [full_trace, full_trace]
    assert captured_turn_indices == [1, 2]
    assert captured_request_texts == ["Group these items.", "Topic matters more."]


def test_clarification_policy_requires_resolved_active_theme() -> None:
    evaluator = ClarificationPolicyEvaluator.from_params(None)
    request = EvaluationRequest(
        predictions=_predictions(),
        artifacts={
            "conversation_trace": {
                "messages": [
                    {"role": "human", "content": "Group these items."},
                    {"role": "assistant", "content": "I grouped them broadly."},
                ]
            },
            "benchmark_spec": _benchmark_spec(),
        },
    )

    with pytest.raises(
        ValueError,
        match="could not resolve the active benchmark theme",
    ):
        evaluator.evaluate(request)


def test_clarification_policy_zeroes_gain_for_non_useful_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import clarification_policy as module

    def _fake_judge(**kwargs: object) -> ClarificationOpportunityJudgment:
        _ = kwargs
        return ClarificationOpportunityJudgment(
            opportunity_exists=True,
            clarification_needed=True,
            asked_clarification=True,
            useful_clarification=False,
            unnecessary_clarification=False,
            missed_clarification=False,
            downstream_gain=0.4,
            reasoning="Clarification did not improve the outcome.",
        )

    monkeypatch.setattr(module, "judge_clarification_opportunity", _fake_judge)

    evaluator = ClarificationPolicyEvaluator.from_params(None)
    request = EvaluationRequest(
        predictions=_predictions(),
        artifacts={
            "conversation_trace": {
                "messages": [
                    {"role": "human", "content": "Group these items."},
                    {"role": "assistant", "content": "By topic or sentiment?"},
                    {"role": "human", "content": "By topic."},
                ]
            },
            "benchmark_spec": _benchmark_spec(),
            "batch_run_config": {
                "human_agent_config": {"guidance": "Group items by topic."},
            },
        },
    )

    result = evaluator.evaluate(request)

    assert result.metrics["downstream_gain_from_clarification"] == 0.0
    assert result.details["sum_downstream_gain"] == 0.0
