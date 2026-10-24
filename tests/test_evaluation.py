"""Unit tests for evaluation artifact loading, registry lookup, and result summarization."""

import json
from pathlib import Path

import pytest

from conversational_clustering.evaluation.artifacts import LoadedClustering, load_saved_clustering
from conversational_clustering.evaluation.base import EvaluationRequest, EvaluationResult
from conversational_clustering.evaluation.methods.basic_metrics import BasicMetricsEvaluator
from conversational_clustering.evaluation.methods.clarification_policy import (
    ClarificationPolicyEvaluator,
)
from conversational_clustering.evaluation.methods.cluster_semantic_quality import (
    ClusterSemanticQualityEvaluator,
)
from conversational_clustering.evaluation.methods.clustering_pov_semantic_quality import (
    ClusteringPovSemanticQualityEvaluator,
)
from conversational_clustering.evaluation.methods.controllability import (
    ControllabilityEvaluator,
)
from conversational_clustering.evaluation.methods.conversation_trace_metrics import (
    ConversationTraceMetricsEvaluator,
)
from conversational_clustering.evaluation.methods.discovery_semantic_quality import (
    DiscoverySemanticQualityEvaluator,
)
from conversational_clustering.evaluation.methods.efficiency_utility import (
    EfficiencyUtilityEvaluator,
    InferredThemeEfficiencyUtilityEvaluator,
)
from conversational_clustering.evaluation.methods.human_effort_metrics import (
    HumanEffortMetricsEvaluator,
)
from conversational_clustering.evaluation.methods.null_theme_abstention import (
    NullThemeAbstentionEvaluator,
)
from conversational_clustering.evaluation.methods.topic_conditioned_quality import (
    TopicConditionedQualityEvaluator,
)
from conversational_clustering.evaluation.registry import get_evaluator, list_evaluators


def _write_json(directory: Path, payload: dict, filename: str = "clustering.json") -> Path:
    path = directory / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_saved_clustering_returns_assignment_and_cluster_names(tmp_path: Path) -> None:
    payload = {
        "clusters": [
            {"name": "Group A", "members": ["alpha", "beta"]},
            {"name": "Group B", "members": ["gamma"]},
        ]
    }
    path = _write_json(tmp_path, payload)

    loaded = load_saved_clustering(path)

    assert loaded.cluster_names == ["Group A", "Group B"]
    assert loaded.assignment == {"alpha": 0, "beta": 0, "gamma": 1}


def test_load_saved_clustering_rejects_duplicate_items_across_clusters(tmp_path: Path) -> None:
    payload = {
        "clusters": [
            {"name": "Group A", "members": ["alpha"]},
            {"name": "Group B", "members": ["alpha"]},
        ]
    }
    path = _write_json(tmp_path, payload)

    with pytest.raises(ValueError, match="appears multiple times"):
        load_saved_clustering(path)


def test_load_saved_clustering_rejects_empty_members(tmp_path: Path) -> None:
    payload = {"clusters": [{"name": "Group A", "members": []}]}
    path = _write_json(tmp_path, payload)

    with pytest.raises(ValueError, match="invalid/empty 'members'"):
        load_saved_clustering(path)


def test_basic_metrics_evaluator_counts_items_and_clusters() -> None:
    evaluator = BasicMetricsEvaluator.from_params()
    request = EvaluationRequest(
        predictions=LoadedClustering(
            assignment={"alpha": 0, "beta": 1, "gamma": 1},
            cluster_names=["Group A", "Group B"],
        )
    )

    result = evaluator.evaluate(request)

    assert result.method == "basic_metrics"
    assert result.metrics == {}
    assert result.details == {"num_items": 3, "num_clusters": 2}


def test_conversation_trace_metrics_evaluator_counts_messages_and_tool_calls() -> None:
    evaluator = ConversationTraceMetricsEvaluator.from_params()
    request = EvaluationRequest(
        predictions=LoadedClustering(assignment={"alpha": 0}, cluster_names=["Group A"]),
        artifacts={
            "conversation_trace": {
                "messages": [
                    {"role": "human", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                    {"role": "human", "content": "please cluster"},
                ],
                "total_tool_calls_by_type": {"read_dataset": 2, "save_clustering": 1},
                "total_failed_tool_calls_by_type": {"save_clustering": 1},
            }
        },
    )

    result = evaluator.evaluate(request)

    assert result.method == "conversation_trace_metrics"
    assert result.metrics == {}
    assert result.details == {
        "conversation_length_messages": 3,
        "num_tool_calls": 3,
        "num_successful_tool_calls": 2,
        "num_failed_tool_calls": 1,
        "total_tool_calls_by_type": {"read_dataset": 2, "save_clustering": 1},
        "total_failed_tool_calls_by_type": {"save_clustering": 1},
    }


def test_conversation_trace_metrics_evaluator_requires_trace_artifact() -> None:
    evaluator = ConversationTraceMetricsEvaluator.from_params()
    request = EvaluationRequest(
        predictions=LoadedClustering(assignment={"alpha": 0}, cluster_names=["Group A"])
    )

    with pytest.raises(ValueError, match="requires 'conversation_trace'"):
        evaluator.evaluate(request)


def test_conversation_trace_metrics_evaluator_rejects_failed_count_exceeding_total() -> None:
    evaluator = ConversationTraceMetricsEvaluator.from_params()
    request = EvaluationRequest(
        predictions=LoadedClustering(assignment={"alpha": 0}, cluster_names=["Group A"]),
        artifacts={
            "conversation_trace": {
                "messages": [],
                "total_tool_calls_by_type": {"save_clustering": 1},
                "total_failed_tool_calls_by_type": {"save_clustering": 2},
            }
        },
    )

    with pytest.raises(ValueError, match="cannot exceed"):
        evaluator.evaluate(request)


def test_human_effort_metrics_evaluator_counts_turns_and_average_lengths() -> None:
    evaluator = HumanEffortMetricsEvaluator.from_params()
    request = EvaluationRequest(
        predictions=LoadedClustering(assignment={"alpha": 0}, cluster_names=["Group A"]),
        artifacts={
            "conversation_trace": {
                "messages": [
                    {"role": "human", "content": "hello"},
                    {"role": "assistant", "content": "hi there"},
                    {"role": "system", "content": "internal note"},
                    {"role": "human", "content": "cluster these items"},
                    {"role": "assistant", "content": "done"},
                ]
            }
        },
    )

    result = evaluator.evaluate(request)

    assert result.method == "human_effort_metrics"
    assert result.metrics == {}
    assert result.details == {
        "turn_count": 4,
        "human_turn_count": 2,
        "agent_turn_count": 2,
        "avg_human_message_length": 2.0,
        "avg_agent_message_length": 1.5,
        "human_to_agent_word_ratio": pytest.approx(4 / 3),
    }


def test_human_effort_metrics_evaluator_requires_trace_artifact() -> None:
    evaluator = HumanEffortMetricsEvaluator.from_params()
    request = EvaluationRequest(
        predictions=LoadedClustering(assignment={"alpha": 0}, cluster_names=["Group A"])
    )

    with pytest.raises(ValueError, match="requires 'conversation_trace'"):
        evaluator.evaluate(request)


def test_human_effort_metrics_evaluator_rejects_non_object_message_entries() -> None:
    evaluator = HumanEffortMetricsEvaluator.from_params()
    request = EvaluationRequest(
        predictions=LoadedClustering(assignment={"alpha": 0}, cluster_names=["Group A"]),
        artifacts={"conversation_trace": {"messages": ["not-an-object"]}},
    )

    with pytest.raises(ValueError, match="entries must be objects"):
        evaluator.evaluate(request)


def test_human_effort_metrics_evaluator_sets_ratio_to_none_when_no_agent_words() -> None:
    evaluator = HumanEffortMetricsEvaluator.from_params()
    request = EvaluationRequest(
        predictions=LoadedClustering(assignment={"alpha": 0}, cluster_names=["Group A"]),
        artifacts={
            "conversation_trace": {
                "messages": [
                    {"role": "human", "content": "cluster these items please"},
                    {"role": "system", "content": "internal"},
                ]
            }
        },
    )

    result = evaluator.evaluate(request)

    assert result.details["human_turn_count"] == 1
    assert result.details["agent_turn_count"] == 0
    assert result.details["human_to_agent_word_ratio"] is None


def test_evaluation_result_summarize_averages_metrics_and_numeric_details() -> None:
    results = [
        EvaluationResult(
            method="m1",
            metrics={"score": 0.2, "coverage": 0.4},
            details={"turns": 2, "success": True},
        ),
        EvaluationResult(
            method="m2",
            metrics={"score": 0.6, "coverage": 0.8},
            details={"turns": 4, "success": False},
        ),
    ]

    summary = EvaluationResult.summarize(results)

    assert summary.method == "summary"
    assert summary.metrics["coverage"] == pytest.approx(0.6)
    assert summary.metrics["score"] == pytest.approx(0.4)
    assert summary.details == {
        "num_results": 2,
        "methods": ["m1", "m2"],
        "mean_numeric_details": {"turns": 3.0},
    }


def test_evaluation_result_summarize_skips_none_detail_values() -> None:
    results = [
        EvaluationResult(
            method="m1",
            metrics={},
            details={"human_to_agent_word_ratio": None},
        ),
        EvaluationResult(
            method="m2",
            metrics={},
            details={"human_to_agent_word_ratio": 2.0},
        ),
    ]

    summary = EvaluationResult.summarize(results)

    assert summary.details == {
        "num_results": 2,
        "methods": ["m1", "m2"],
        "mean_numeric_details": {"human_to_agent_word_ratio": 2.0},
    }


def test_evaluation_result_summarize_handles_empty_inputs() -> None:
    summary = EvaluationResult.summarize([])

    assert summary.method == "summary"
    assert summary.metrics == {}
    assert summary.details == {
        "num_results": 0,
        "methods": [],
        "mean_numeric_details": {},
    }


def test_registry_lists_and_loads_basic_metrics_evaluator() -> None:
    evaluator_names = list_evaluators()
    evaluator = get_evaluator("basic_metrics")
    clarification_policy_evaluator = get_evaluator("clarification_policy")
    conversation_trace_evaluator = get_evaluator("conversation_trace_metrics")
    controllability_evaluator = get_evaluator("controllability")
    human_effort_evaluator = get_evaluator("human_effort_metrics")
    clustering_pov_semantic_quality_evaluator = get_evaluator("clustering_pov_semantic_quality")
    discovery_semantic_quality_evaluator = get_evaluator("discovery_semantic_quality")
    cluster_semantic_quality_evaluator = get_evaluator("cluster_semantic_quality")
    topic_conditioned_quality_evaluator = get_evaluator("topic_conditioned_quality")
    efficiency_utility_evaluator = get_evaluator("efficiency_utility")
    inferred_theme_efficiency_utility_evaluator = get_evaluator("inferred_theme_efficiency_utility")
    null_theme_abstention_evaluator = get_evaluator(
        "null_theme_abstention",
        params={
            "unsupported_topics": [
                {
                    "topic_id": "astrology",
                    "label": "Astrology",
                    "guidance": "Group items by zodiac meaning.",
                }
            ]
        },
    )

    assert "basic_metrics" in evaluator_names
    assert "clarification_policy" in evaluator_names
    assert "clustering_pov_semantic_quality" in evaluator_names
    assert "cluster_semantic_quality" in evaluator_names
    assert "controllability" in evaluator_names
    assert "conversation_trace_metrics" in evaluator_names
    assert "discovery_semantic_quality" in evaluator_names
    assert "efficiency_utility" in evaluator_names
    assert "inferred_theme_efficiency_utility" in evaluator_names
    assert "human_effort_metrics" in evaluator_names
    assert "null_theme_abstention" in evaluator_names
    assert "topic_conditioned_quality" in evaluator_names
    assert isinstance(evaluator, BasicMetricsEvaluator)
    assert isinstance(clarification_policy_evaluator, ClarificationPolicyEvaluator)
    assert isinstance(
        clustering_pov_semantic_quality_evaluator, ClusteringPovSemanticQualityEvaluator
    )
    assert isinstance(cluster_semantic_quality_evaluator, ClusterSemanticQualityEvaluator)
    assert isinstance(conversation_trace_evaluator, ConversationTraceMetricsEvaluator)
    assert isinstance(controllability_evaluator, ControllabilityEvaluator)
    assert isinstance(discovery_semantic_quality_evaluator, DiscoverySemanticQualityEvaluator)
    assert isinstance(efficiency_utility_evaluator, EfficiencyUtilityEvaluator)
    assert isinstance(
        inferred_theme_efficiency_utility_evaluator,
        InferredThemeEfficiencyUtilityEvaluator,
    )
    assert isinstance(human_effort_evaluator, HumanEffortMetricsEvaluator)
    assert isinstance(null_theme_abstention_evaluator, NullThemeAbstentionEvaluator)
    assert isinstance(topic_conditioned_quality_evaluator, TopicConditionedQualityEvaluator)


def test_registry_rejects_unknown_evaluator() -> None:
    with pytest.raises(ValueError, match="Unknown evaluator"):
        get_evaluator("not_registered")


def test_registry_rejects_unsupported_evaluator_params() -> None:
    with pytest.raises(ValueError, match="does not support parameters yet"):
        get_evaluator("basic_metrics", params={"foo": "bar"})
