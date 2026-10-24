"""Tests for benchmark specs and benchmark-aware evaluators."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from conversational_clustering.accounting import UsageCollector
from conversational_clustering.batch_runs.config import GENERATED_BENCHMARK_SPEC_METADATA_KEY
from conversational_clustering.data_processing.datasets import DATASETS
from conversational_clustering.evaluation.artifacts import LoadedClustering
from conversational_clustering.evaluation.base import (
    EvaluationRequest,
    EvaluationResult,
    PairwiseEvaluationRequest,
)
from conversational_clustering.evaluation.benchmark_spec import (
    SyntheticBenchmarkSpec,
    load_benchmark_spec_for_dataset,
    parse_synthetic_benchmark_spec,
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
from conversational_clustering.evaluation.methods.discovery_semantic_quality import (
    DiscoverySemanticQualityEvaluator,
)
from conversational_clustering.evaluation.methods.efficiency_utility import (
    EfficiencyUtilityEvaluator,
    InferredThemeEfficiencyUtilityEvaluator,
)
from conversational_clustering.evaluation.methods.null_theme_abstention import (
    NullThemeAbstentionEvaluator,
)
from conversational_clustering.evaluation.methods.robustness_expectations import (
    RobustnessExpectationsEvaluator,
)
from conversational_clustering.evaluation.methods.semantic_consistency.evaluator import (
    SemanticConsistencyEvaluator,
)
from conversational_clustering.evaluation.methods.semantic_quality_judges import (
    ClarificationOpportunityJudgment,
    ClusterCoherenceJudgment,
    ClusterDistinctivenessJudgment,
    DuplicateTopicJudgment,
    ThemeMatchJudgment,
)
from conversational_clustering.evaluation.methods.topic_conditioned_quality import (
    TopicConditionedQualityEvaluator,
)
from conversational_clustering.evaluation.runner import (
    EvaluationRunFailed,
    iter_batch_run_requests,
    run_evaluation_config,
)
from conversational_clustering.models import default_semantic_consistency_judge_model_config


@pytest.mark.parametrize(
    ("evaluator_name", "resume_version"),
    [
        ("cluster_semantic_quality", ClusterSemanticQualityEvaluator.resume_version),
        ("clustering_pov_semantic_quality", ClusteringPovSemanticQualityEvaluator.resume_version),
        ("discovery_semantic_quality", DiscoverySemanticQualityEvaluator.resume_version),
        ("efficiency_utility", EfficiencyUtilityEvaluator.resume_version),
        (
            "inferred_theme_efficiency_utility",
            InferredThemeEfficiencyUtilityEvaluator.resume_version,
        ),
        ("robustness_expectations", RobustnessExpectationsEvaluator.resume_version),
        ("semantic_consistency", SemanticConsistencyEvaluator.resume_version),
        ("topic_conditioned_quality", TopicConditionedQualityEvaluator.resume_version),
    ],
)
def test_accounting_evaluators_bump_resume_version(
    evaluator_name: str,
    resume_version: str,
) -> None:
    assert resume_version == "v2", f"{evaluator_name} must invalidate pre-accounting rows"


def test_null_theme_abstention_resume_version_invalidates_legacy_matching_rows() -> None:
    assert NullThemeAbstentionEvaluator.resume_version == "v3"


def _benchmark_spec() -> SyntheticBenchmarkSpec:
    return parse_synthetic_benchmark_spec(
        {
            "dataset_name": "synthetic",
            "items": ["apple", "banana", "car", "truck"],
            "themes": [
                {
                    "theme_id": "fruit",
                    "label": "Fruit",
                    "guidance": "Group items by fruitness.",
                    "weight": 2.0,
                    "granularity": "normal",
                },
                {
                    "theme_id": "vehicle",
                    "label": "Vehicle",
                    "guidance": "Group items by vehicle type.",
                    "weight": 1.0,
                    "granularity": "normal",
                },
                {
                    "theme_id": "astrology",
                    "label": "Astrology",
                    "guidance": "Group items by zodiac meaning.",
                    "weight": 1.0,
                    "is_foil": True,
                    "granularity": "normal",
                },
            ],
            "references": [
                {
                    "reference_id": "fruit__normal",
                    "topic_id": "fruit",
                    "granularity": "normal",
                    "assignment": {"apple": 1, "banana": 1, "car": 0, "truck": 0},
                    "cluster_names": ["other", "fruit"],
                },
                {
                    "reference_id": "fruit__initial",
                    "topic_id": "fruit",
                    "granularity": "initial",
                    "assignment": {"apple": 0, "banana": 0, "car": 0, "truck": 0},
                    "cluster_names": ["all_items"],
                },
                {
                    "reference_id": "fruit__separate",
                    "topic_id": "fruit",
                    "granularity": "fine",
                    "assignment": {"apple": 0, "banana": 1, "car": 2, "truck": 3},
                    "cluster_names": ["apple", "banana", "car", "truck"],
                },
                {
                    "reference_id": "vehicle__normal",
                    "topic_id": "vehicle",
                    "granularity": "normal",
                    "assignment": {"apple": 0, "banana": 0, "car": 1, "truck": 1},
                    "cluster_names": ["other", "vehicle"],
                },
            ],
        }
    )


def _enterprise_ticket_benchmark_spec() -> SyntheticBenchmarkSpec:
    return parse_synthetic_benchmark_spec(
        {
            "dataset_name": "enterprise_ticket",
            "items": ["okta locked", "badge broken", "invoice overdue", "vpn down"],
            "themes": [
                {
                    "theme_id": "department",
                    "label": "Department",
                    "guidance": "Group tickets by department.",
                    "granularity": "normal",
                },
                {
                    "theme_id": "issue_type",
                    "label": "Issue Type",
                    "guidance": "Group tickets by issue type.",
                    "granularity": "normal",
                },
            ],
            "references": [
                {
                    "reference_id": "department__normal",
                    "topic_id": "department",
                    "granularity": "normal",
                    "assignment": {
                        "okta locked": 1,
                        "badge broken": 1,
                        "invoice overdue": 2,
                        "vpn down": 1,
                    },
                    "cluster_names": ["Unused", "IT", "Finance"],
                },
                {
                    "reference_id": "issue_type__normal",
                    "topic_id": "issue_type",
                    "granularity": "normal",
                    "assignment": {
                        "okta locked": 0,
                        "badge broken": 1,
                        "invoice overdue": 2,
                        "vpn down": 3,
                    },
                    "cluster_names": [
                        "access_and_authentication",
                        "badge_problem",
                        "finance_transaction",
                        "service_issue",
                    ],
                },
            ],
        }
    )


def _grouped_pov_benchmark_spec() -> SyntheticBenchmarkSpec:
    return parse_synthetic_benchmark_spec(
        {
            "dataset_name": "enterprise_ticket",
            "items": ["okta locked", "badge broken", "invoice overdue", "vpn down"],
            "themes": [
                {
                    "theme_id": "organizational_label_coarse",
                    "label": "Organizational (Coarse)",
                    "guidance": (
                        "Group tickets by the broad organizational area "
                        "responsible for handling the request."
                    ),
                    "pov_group_id": "organizational",
                    "granularity": "coarse",
                },
                {
                    "theme_id": "organizational_label_normal",
                    "label": "Organizational (Normal)",
                    "guidance": (
                        "Group tickets by the primary team area responsible "
                        "for handling the request, such as identity access, "
                        "badge operations, accounts payable, or network "
                        "operations."
                    ),
                    "pov_group_id": "organizational",
                    "granularity": "normal",
                },
                {
                    "theme_id": "intent_label_coarse",
                    "label": "Intent (Coarse)",
                    "guidance": (
                        "Group tickets by the broad user intent or workflow "
                        "family behind the request."
                    ),
                    "pov_group_id": "intent",
                    "granularity": "coarse",
                },
                {
                    "theme_id": "intent_label_normal",
                    "label": "Intent (Normal)",
                    "guidance": (
                        "Group tickets by the primary request type or problem "
                        "category expressed by the ticket."
                    ),
                    "pov_group_id": "intent",
                    "granularity": "normal",
                },
                {
                    "theme_id": "intent_label_fine",
                    "label": "Intent (Fine)",
                    "guidance": (
                        "Group tickets by the most specific request or issue "
                        "subtype, separating distinct problems such as MFA "
                        "delivery failure, badge malfunction, invoice "
                        "mismatch, or VPN outage."
                    ),
                    "pov_group_id": "intent",
                    "granularity": "fine",
                },
            ],
            "references": [
                {
                    "reference_id": "organizational_label_coarse__coarse",
                    "topic_id": "organizational_label_coarse",
                    "granularity": "coarse",
                    "pov_group_id": "organizational",
                    "granularity_level": "coarse",
                    "assignment": {
                        "okta locked": 0,
                        "badge broken": 0,
                        "invoice overdue": 1,
                        "vpn down": 0,
                    },
                    "cluster_names": ["operations", "finance"],
                },
                {
                    "reference_id": "organizational_label_normal__normal",
                    "topic_id": "organizational_label_normal",
                    "granularity": "normal",
                    "pov_group_id": "organizational",
                    "granularity_level": "normal",
                    "assignment": {
                        "okta locked": 0,
                        "badge broken": 1,
                        "invoice overdue": 2,
                        "vpn down": 3,
                    },
                    "cluster_names": [
                        "identity_access",
                        "badge_ops",
                        "accounts_payable",
                        "network_ops",
                    ],
                },
                {
                    "reference_id": "intent_label_coarse__coarse",
                    "topic_id": "intent_label_coarse",
                    "granularity": "coarse",
                    "pov_group_id": "intent",
                    "granularity_level": "coarse",
                    "assignment": {
                        "okta locked": 0,
                        "badge broken": 0,
                        "invoice overdue": 1,
                        "vpn down": 0,
                    },
                    "cluster_names": ["access_or_service", "finance"],
                },
                {
                    "reference_id": "intent_label_normal__normal",
                    "topic_id": "intent_label_normal",
                    "granularity": "normal",
                    "pov_group_id": "intent",
                    "granularity_level": "normal",
                    "assignment": {
                        "okta locked": 0,
                        "badge broken": 0,
                        "invoice overdue": 1,
                        "vpn down": 2,
                    },
                    "cluster_names": ["access_identity", "finance_procurement", "network_ops"],
                },
                {
                    "reference_id": "intent_label_fine__fine",
                    "topic_id": "intent_label_fine",
                    "granularity": "fine",
                    "pov_group_id": "intent",
                    "granularity_level": "fine",
                    "assignment": {
                        "okta locked": 0,
                        "badge broken": 1,
                        "invoice overdue": 2,
                        "vpn down": 3,
                    },
                    "cluster_names": [
                        "mfa_problem",
                        "badge_issue",
                        "invoice_issue",
                        "vpn_outage",
                    ],
                },
            ],
        }
    )


def _predictions_payload() -> dict[str, object]:
    return {
        "clusters": [
            {"name": "other", "members": ["car", "truck"]},
            {"name": "fruit", "members": ["apple", "banana"]},
        ]
    }


def _clustering_payload_from_loaded(clustering: LoadedClustering) -> dict[str, object]:
    clusters: list[dict[str, object]] = []
    for cluster_id, cluster_name in enumerate(clustering.cluster_names):
        members = [
            item_text
            for item_text, assigned_cluster_id in clustering.assignment.items()
            if assigned_cluster_id == cluster_id
        ]
        if not members:
            continue
        clusters.append({"name": cluster_name, "members": sorted(members)})
    return {"clusters": clusters}


def _request(*, benchmark_spec: SyntheticBenchmarkSpec | None = None) -> EvaluationRequest:
    benchmark = benchmark_spec or _benchmark_spec()
    return EvaluationRequest(
        predictions=benchmark.reference_by_id["fruit__normal"].to_loaded_clustering(),
        artifacts={
            "benchmark_spec": benchmark,
            "selected_topic": {
                "topic_id": "fruit",
                "label": "Fruit",
                "guidance": "Group items by fruitness.",
            },
            "topic_candidates": {
                "topics": [
                    {
                        "topic_id": "fruit",
                        "label": "Fruit",
                        "guidance": "Group items by fruitness.",
                        "rationale": "Fruit items align.",
                    },
                    {
                        "topic_id": "vehicle",
                        "label": "Vehicle",
                        "guidance": "Group items by vehicle type.",
                        "rationale": "Vehicle items align.",
                    },
                ],
                "rounds": [
                    {
                        "round_index": 1,
                        "generated_at_turn": 1,
                        "new_topics": [
                            {
                                "topic_id": "fruit",
                                "label": "Fruit",
                                "guidance": "Group items by fruitness.",
                            }
                        ],
                    },
                    {
                        "round_index": 2,
                        "generated_at_turn": 2,
                        "new_topics": [
                            {
                                "topic_id": "vehicle",
                                "label": "Vehicle",
                                "guidance": "Group items by vehicle type.",
                            }
                        ],
                    },
                ],
            },
            "clustering_pov_history": {
                "snapshots": [
                    {"turn_index": 1, "povs": ["Fruit"], "num_povs": 1},
                    {"turn_index": 2, "povs": ["Fruit", "Vehicle"], "num_povs": 2},
                ],
                "num_snapshots": 2,
            },
            "turns_run": 2,
            "conversation_trace": {
                "messages": [
                    {"role": "human", "content": "Group items by fruitness."},
                    {
                        "role": "assistant",
                        "content": "Should I separate fruit from non-fruit items?",
                    },
                    {"role": "human", "content": "Yes, do that."},
                ]
            },
            "judge_clustering_trace": {
                "clusterings": [
                    {
                        "turn_index": 1,
                        "clustering": {
                            "clusters": [
                                {"name": "all", "members": ["apple", "banana", "car", "truck"]}
                            ]
                        },
                    },
                    {"turn_index": 2, "clustering": _predictions_payload()},
                ],
                "num_clusterings": 2,
            },
            "batch_run_config": {
                "dataset_name": "synthetic",
                "human_agent_config": {
                    "guidance": "Group items by fruitness.",
                },
                "metadata": {
                    "topic_branch_topic_id": "fruit",
                    "topic_branch_topic_label": "Fruit",
                    "topic_branch_topic_guidance": "Group items by fruitness.",
                },
            },
        },
        metadata={"config_slug": "cfg", "run_id": "run-1"},
    )


def _controllability_request(
    *,
    benchmark_spec: SyntheticBenchmarkSpec | None = None,
    predictions: LoadedClustering | None = None,
    judge_clusterings: list[dict[str, object]] | None = None,
    turn_injection: dict[str, object] | None = None,
    guidance: str | None = "Group items by fruitness.",
    human_turn_messages: list[tuple[int, str]] | None = None,
    selected_topic: dict[str, object] | None = None,
    batch_run_metadata: dict[str, object] | None = None,
) -> EvaluationRequest:
    benchmark = benchmark_spec or _benchmark_spec()
    resolved_judge_clusterings = judge_clusterings or [
        {
            "turn_index": 1,
            "clustering": {
                "clusters": [
                    {"name": "all", "members": ["apple", "banana", "car", "truck"]},
                ]
            },
        },
        {
            "turn_index": 2,
            "clustering": {
                "clusters": [
                    {"name": "mixed", "members": ["apple", "banana", "car"]},
                    {"name": "truck", "members": ["truck"]},
                ]
            },
        },
        {
            "turn_index": 3,
            "clustering": _predictions_payload(),
        },
    ]
    resolved_turn_injection = turn_injection or _theme_switch_turn_injection_payload()
    resolved_human_turn_messages = human_turn_messages or [
        (1, "Group items by vehicle type."),
        (2, "Please switch to fruitness."),
        (3, "Keep fruit items together."),
    ]
    return EvaluationRequest(
        predictions=predictions
        or benchmark.reference_by_id["fruit__normal"].to_loaded_clustering(),
        artifacts={
            "benchmark_spec": benchmark,
            "conversation_trace": {
                "messages": [
                    {"role": "human", "content": message}
                    for _turn_index, message in resolved_human_turn_messages
                ]
            },
            "human_thought_trace": {
                "messages": [
                    {"turn_index": turn_index, "message": message, "thought": "test thought"}
                    for turn_index, message in resolved_human_turn_messages
                ]
            },
            "judge_clustering_trace": {
                "clusterings": resolved_judge_clusterings,
                "num_clusterings": len(resolved_judge_clusterings),
            },
            "batch_run_config": {
                "dataset_name": "synthetic",
                "human_agent_config": {
                    "prompt_id": "baseline_v4",
                    "guidance": guidance,
                    "turn_injection": resolved_turn_injection,
                },
                **({"metadata": batch_run_metadata} if batch_run_metadata is not None else {}),
            },
            **({"selected_topic": selected_topic} if selected_topic is not None else {}),
        },
        metadata={"config_slug": "cfg", "run_id": "run-1"},
    )


def _theme_switch_turn_injection_payload(
    *,
    turn_index: int = 2,
    source_theme_id: str = "vehicle",
    target_theme_id: str = "fruit",
) -> dict[str, object]:
    return {
        "turn_index": turn_index,
        "edit_primitive": "theme_switch",
        "source_theme_id": source_theme_id,
        "target_theme_id": target_theme_id,
    }


def _granularity_turn_injection_payload(
    *,
    turn_index: int = 2,
    edit_primitive: str,
) -> dict[str, object]:
    return {
        "turn_index": turn_index,
        "edit_primitive": edit_primitive,
    }


def test_controllability_evaluator_computes_final_metrics_and_lag() -> None:
    evaluator = ControllabilityEvaluator.from_params({"alpha": 0.8, "beta": 0.1})

    result = evaluator.evaluate(_controllability_request())

    assert result.method == "controllability"
    assert result.metrics == {
        "target_change_success": 1.0,
        "collateral_churn": 0.0,
        "lag_success_rate": 1.0,
    }
    assert result.details["request_turn"] == 2
    assert result.details["pre_request_turn"] == 1
    assert result.details["lag"] == 1
    assert result.details["acceptable_turn"] == 3
    assert result.details["num_change_pairs"] == 4
    assert result.details["num_keep_pairs"] == 2
    assert result.details["best_target_change_success"] == 1.0
    assert result.details["min_collateral_churn"] == 0.0
    assert result.details["final_target_change_success"] == 1.0
    assert result.details["final_collateral_churn"] == 0.0
    assert result.details["source_granularity"] == "normal"
    assert result.details["target_granularity"] == "normal"


def test_controllability_evaluator_uses_final_saved_clustering_for_terminal_metrics() -> None:
    evaluator = ControllabilityEvaluator.from_params({"alpha": 0.8, "beta": 0.1})
    benchmark = _benchmark_spec()

    result = evaluator.evaluate(
        _controllability_request(
            predictions=benchmark.reference_by_id["fruit__normal"].to_loaded_clustering(),
            judge_clusterings=[
                {
                    "turn_index": 1,
                    "clustering": {
                        "clusters": [
                            {"name": "all", "members": ["apple", "banana", "car", "truck"]},
                        ]
                    },
                },
                {
                    "turn_index": 2,
                    "clustering": {
                        "clusters": [
                            {"name": "mixed", "members": ["apple", "banana", "car"]},
                            {"name": "truck", "members": ["truck"]},
                        ]
                    },
                },
            ],
        )
    )

    assert result.metrics["target_change_success"] == 1.0
    assert result.metrics["collateral_churn"] == 0.0
    assert result.details["lag"] == 1
    assert result.details["acceptable_turn"] == 3


def test_controllability_evaluator_rejects_missing_turn_injection_metadata() -> None:
    evaluator = ControllabilityEvaluator.from_params()
    request = _controllability_request()
    request.artifacts["batch_run_config"]["human_agent_config"].pop("turn_injection")

    with pytest.raises(ValueError, match="requires an explicit"):
        evaluator.evaluate(request)


def test_controllability_evaluator_returns_invalid_episode_when_no_pairs_need_to_change() -> None:
    evaluator = ControllabilityEvaluator.from_params()
    benchmark = parse_synthetic_benchmark_spec(
        {
            "dataset_name": "synthetic",
            "items": ["apple", "banana", "car", "truck"],
            "themes": [
                {
                    "theme_id": "bucket_a",
                    "label": "Bucket A",
                    "guidance": "Group items into the shared two-bucket layout.",
                },
                {
                    "theme_id": "bucket_b",
                    "label": "Bucket B",
                    "guidance": "Group items into the shared two-bucket layout.",
                },
            ],
            "references": [
                {
                    "reference_id": "bucket_a__normal",
                    "topic_id": "bucket_a",
                    "granularity": "normal",
                    "assignment": {"apple": 0, "banana": 0, "car": 1, "truck": 1},
                },
                {
                    "reference_id": "bucket_b__normal",
                    "topic_id": "bucket_b",
                    "granularity": "normal",
                    "assignment": {"apple": 0, "banana": 0, "car": 1, "truck": 1},
                },
            ],
        }
    )

    result = evaluator.evaluate(
        _controllability_request(
            benchmark_spec=benchmark,
            predictions=benchmark.reference_by_id["bucket_b__normal"].to_loaded_clustering(),
            judge_clusterings=[
                {
                    "turn_index": 1,
                    "clustering": _clustering_payload_from_loaded(
                        benchmark.reference_by_id["bucket_a__normal"].to_loaded_clustering()
                    ),
                },
                {
                    "turn_index": 2,
                    "clustering": _clustering_payload_from_loaded(
                        benchmark.reference_by_id["bucket_b__normal"].to_loaded_clustering()
                    ),
                },
            ],
            turn_injection=_theme_switch_turn_injection_payload(
                source_theme_id="bucket_a",
                target_theme_id="bucket_b",
            ),
        )
    )

    assert result.metrics == {}
    assert result.details["invalid_episode"] is True
    assert result.details["invalid_reason"] == "no_change_pairs"
    assert result.details["num_change_pairs"] == 0
    assert result.details["num_keep_pairs"] == 6


def test_controllability_evaluator_sets_collateral_churn_to_zero_when_no_keep_pairs() -> None:
    evaluator = ControllabilityEvaluator.from_params({"alpha": 0.8, "beta": 0.1})
    benchmark = parse_synthetic_benchmark_spec(
        {
            "dataset_name": "synthetic_pair",
            "items": ["apple", "banana"],
            "themes": [
                {
                    "theme_id": "split",
                    "label": "Split",
                    "guidance": "Separate the two items.",
                },
                {
                    "theme_id": "together",
                    "label": "Together",
                    "guidance": "Keep the two items together.",
                },
            ],
            "references": [
                {
                    "reference_id": "split__normal",
                    "topic_id": "split",
                    "granularity": "normal",
                    "assignment": {"apple": 0, "banana": 1},
                },
                {
                    "reference_id": "together__normal",
                    "topic_id": "together",
                    "granularity": "normal",
                    "assignment": {"apple": 0, "banana": 0},
                },
            ],
        }
    )

    result = evaluator.evaluate(
        _controllability_request(
            benchmark_spec=benchmark,
            predictions=benchmark.reference_by_id["together__normal"].to_loaded_clustering(),
            judge_clusterings=[
                {
                    "turn_index": 1,
                    "clustering": _clustering_payload_from_loaded(
                        benchmark.reference_by_id["split__normal"].to_loaded_clustering()
                    ),
                },
                {
                    "turn_index": 2,
                    "clustering": _clustering_payload_from_loaded(
                        benchmark.reference_by_id["together__normal"].to_loaded_clustering()
                    ),
                },
            ],
            turn_injection=_theme_switch_turn_injection_payload(
                source_theme_id="split",
                target_theme_id="together",
            ),
        )
    )

    assert result.metrics["target_change_success"] == 1.0
    assert result.metrics["collateral_churn"] == 0.0
    assert result.details["num_keep_pairs"] == 0
    assert result.details["lag"] == 0


def test_controllability_evaluator_uses_later_partial_post_request_snapshot() -> None:
    evaluator = ControllabilityEvaluator.from_params({"alpha": 0.8, "beta": 0.1})
    benchmark = _benchmark_spec()

    result = evaluator.evaluate(
        _controllability_request(
            predictions=benchmark.reference_by_id["fruit__normal"].to_loaded_clustering(),
            judge_clusterings=[
                {
                    "turn_index": 1,
                    "clustering": {
                        "clusters": [
                            {"name": "all", "members": ["apple", "banana", "car", "truck"]},
                        ]
                    },
                },
                {
                    "turn_index": 2,
                    "clustering": {
                        "clusters": [
                            {"name": "partial", "members": ["apple", "banana"]},
                        ]
                    },
                },
                {
                    "turn_index": 3,
                    "clustering": _predictions_payload(),
                },
            ],
            turn_injection=_theme_switch_turn_injection_payload(),
        )
    )

    assert "invalid_episode" not in result.details
    assert result.metrics["target_change_success"] == pytest.approx(1.0)
    assert result.details["final_target_change_success"] == pytest.approx(1.0)


def test_parse_synthetic_benchmark_spec_round_trips_core_fields() -> None:
    benchmark_spec = _benchmark_spec()

    assert benchmark_spec.dataset_name == "synthetic"
    assert benchmark_spec.theme_by_id["fruit"].weight == pytest.approx(2.0)
    assert benchmark_spec.resolve_reference(topic_id="fruit").reference_id == "fruit__normal"


def test_parse_synthetic_benchmark_spec_supports_grouped_pov_metadata() -> None:
    benchmark_spec = _grouped_pov_benchmark_spec()

    intent_normal = benchmark_spec.theme_by_id["intent_label_normal"]
    assert intent_normal.pov_group_id == "intent"
    assert intent_normal.granularity == "normal"
    assert [
        reference.reference_id
        for reference in benchmark_spec.references_for_theme_or_group("intent_label_normal")
    ] == [
        "intent_label_coarse__coarse",
        "intent_label_normal__normal",
        "intent_label_fine__fine",
    ]


def test_parse_synthetic_benchmark_spec_accepts_null_optional_grouping_fields() -> None:
    benchmark = parse_synthetic_benchmark_spec(
        {
            "dataset_name": "synthetic",
            "items": ["apple", "banana"],
            "themes": [
                {
                    "theme_id": "fruit",
                    "label": "Fruit",
                    "guidance": "Group items by fruitness.",
                    "pov_group_id": None,
                    "granularity": None,
                }
            ],
            "references": [
                {
                    "reference_id": "fruit__normal",
                    "topic_id": "fruit",
                    "granularity": "fine",
                    "assignment": {"apple": 0, "banana": 0},
                    "pov_group_id": None,
                    "granularity_level": None,
                }
            ],
        }
    )

    assert benchmark.theme_by_id["fruit"].pov_group_id is None
    assert benchmark.theme_by_id["fruit"].granularity == "normal"
    assert benchmark.reference_by_id["fruit__normal"].pov_group_id is None
    assert benchmark.reference_by_id["fruit__normal"].granularity == "fine"
    assert benchmark.reference_by_id["fruit__normal"].granularity_level == "fine"


def test_controllability_evaluator_uses_benchmark_source_reference_for_turn_one_injection() -> None:
    evaluator = ControllabilityEvaluator.from_params({"alpha": 0.8, "beta": 0.1})

    result = evaluator.evaluate(
        _controllability_request(
            judge_clusterings=[
                {
                    "turn_index": 1,
                    "clustering": _predictions_payload(),
                },
            ],
            turn_injection=_theme_switch_turn_injection_payload(turn_index=1),
        )
    )

    assert result.details["pre_request_turn"] == 0
    assert result.details["source_theme_id"] == "vehicle"
    assert result.details["source_reference_id"] == "vehicle__normal"


def test_controllability_evaluator_uses_configured_theme_ids_for_theme_switch() -> None:
    evaluator = ControllabilityEvaluator.from_params()
    benchmark = _benchmark_spec()

    result = evaluator.evaluate(
        _controllability_request(
            benchmark_spec=benchmark,
            predictions=benchmark.reference_by_id["vehicle__normal"].to_loaded_clustering(),
            judge_clusterings=[
                {
                    "turn_index": 1,
                    "clustering": _clustering_payload_from_loaded(
                        benchmark.reference_by_id["fruit__separate"].to_loaded_clustering()
                    ),
                },
                {
                    "turn_index": 2,
                    "clustering": _clustering_payload_from_loaded(
                        benchmark.reference_by_id["vehicle__normal"].to_loaded_clustering()
                    ),
                },
            ],
            turn_injection=_theme_switch_turn_injection_payload(
                source_theme_id="fruit",
                target_theme_id="vehicle",
            ),
        )
    )

    assert result.details["source_theme_id"] == "fruit"
    assert result.details["source_reference_id"] == "fruit__separate"
    assert result.details["source_granularity"] == "fine"
    assert result.details["target_theme_id"] == "vehicle"
    assert result.details["target_reference_id"] == "vehicle__normal"
    assert result.details["target_granularity"] == "normal"


def test_controllability_evaluator_coarsens_within_active_theme() -> None:
    evaluator = ControllabilityEvaluator.from_params()
    benchmark = _grouped_pov_benchmark_spec()

    result = evaluator.evaluate(
        _controllability_request(
            benchmark_spec=benchmark,
            predictions=benchmark.reference_by_id[
                "intent_label_coarse__coarse"
            ].to_loaded_clustering(),
            judge_clusterings=[
                {
                    "turn_index": 1,
                    "clustering": _clustering_payload_from_loaded(
                        benchmark.reference_by_id[
                            "intent_label_normal__normal"
                        ].to_loaded_clustering()
                    ),
                },
                {
                    "turn_index": 2,
                    "clustering": _clustering_payload_from_loaded(
                        benchmark.reference_by_id[
                            "intent_label_coarse__coarse"
                        ].to_loaded_clustering()
                    ),
                },
            ],
            turn_injection=_granularity_turn_injection_payload(edit_primitive="coarsen"),
            guidance=(
                "Group tickets by the primary request type or problem category "
                "expressed by the ticket."
            ),
        )
    )

    assert result.details["source_theme_id"] == "intent_label_normal"
    assert result.details["target_theme_id"] == "intent_label_coarse"
    assert result.details["source_granularity"] == "normal"
    assert result.details["target_granularity"] == "coarse"
    assert result.details["source_reference_id"] == "intent_label_normal__normal"
    assert result.details["target_reference_id"] == "intent_label_coarse__coarse"


def test_controllability_evaluator_refines_within_active_theme() -> None:
    evaluator = ControllabilityEvaluator.from_params()
    benchmark = _grouped_pov_benchmark_spec()

    result = evaluator.evaluate(
        _controllability_request(
            benchmark_spec=benchmark,
            predictions=benchmark.reference_by_id["intent_label_fine__fine"].to_loaded_clustering(),
            judge_clusterings=[
                {
                    "turn_index": 1,
                    "clustering": _clustering_payload_from_loaded(
                        benchmark.reference_by_id[
                            "intent_label_normal__normal"
                        ].to_loaded_clustering()
                    ),
                },
                {
                    "turn_index": 2,
                    "clustering": _clustering_payload_from_loaded(
                        benchmark.reference_by_id["intent_label_fine__fine"].to_loaded_clustering()
                    ),
                },
            ],
            turn_injection=_granularity_turn_injection_payload(edit_primitive="refine"),
            guidance=(
                "Group tickets by the primary request type or problem category "
                "expressed by the ticket."
            ),
        )
    )

    assert result.details["source_theme_id"] == "intent_label_normal"
    assert result.details["target_theme_id"] == "intent_label_fine"
    assert result.details["source_granularity"] == "normal"
    assert result.details["target_granularity"] == "fine"
    assert result.details["source_reference_id"] == "intent_label_normal__normal"
    assert result.details["target_reference_id"] == "intent_label_fine__fine"


def test_controllability_evaluator_uses_best_matching_target_reference_for_granularity_shift() -> (
    None
):
    evaluator = ControllabilityEvaluator.from_params()
    benchmark = parse_synthetic_benchmark_spec(
        {
            "dataset_name": "enterprise_ticket",
            "items": ["okta locked", "badge broken", "invoice overdue", "vpn down"],
            "themes": [
                {
                    "theme_id": "intent_label_normal",
                    "label": "Intent Label Normal",
                    "guidance": "Group tickets by the normal intent POV.",
                    "pov_group_id": "intent",
                    "granularity": "normal",
                },
                {
                    "theme_id": "intent_label_coarse_a",
                    "label": "Intent Label Coarse A",
                    "guidance": "Group tickets by the coarse intent POV A.",
                    "pov_group_id": "intent",
                    "granularity": "coarse",
                },
                {
                    "theme_id": "intent_label_coarse_b",
                    "label": "Intent Label Coarse B",
                    "guidance": "Group tickets by the coarse intent POV B.",
                    "pov_group_id": "intent",
                    "granularity": "coarse",
                },
            ],
            "references": [
                {
                    "reference_id": "intent_label_normal__normal",
                    "topic_id": "intent_label_normal",
                    "granularity": "normal",
                    "granularity_level": "normal",
                    "pov_group_id": "intent",
                    "assignment": {
                        "okta locked": 0,
                        "badge broken": 1,
                        "invoice overdue": 2,
                        "vpn down": 3,
                    },
                },
                {
                    "reference_id": "intent_label_coarse_a__coarse",
                    "topic_id": "intent_label_coarse_a",
                    "granularity": "coarse",
                    "granularity_level": "coarse",
                    "pov_group_id": "intent",
                    "assignment": {
                        "okta locked": 0,
                        "badge broken": 0,
                        "invoice overdue": 1,
                        "vpn down": 1,
                    },
                },
                {
                    "reference_id": "intent_label_coarse_b__coarse",
                    "topic_id": "intent_label_coarse_b",
                    "granularity": "coarse",
                    "granularity_level": "coarse",
                    "pov_group_id": "intent",
                    "assignment": {
                        "okta locked": 0,
                        "badge broken": 1,
                        "invoice overdue": 1,
                        "vpn down": 1,
                    },
                },
            ],
        }
    )
    final_predictions = benchmark.reference_by_id[
        "intent_label_coarse_b__coarse"
    ].to_loaded_clustering()

    result = evaluator.evaluate(
        _controllability_request(
            benchmark_spec=benchmark,
            predictions=final_predictions,
            judge_clusterings=[
                {
                    "turn_index": 1,
                    "clustering": _clustering_payload_from_loaded(
                        benchmark.reference_by_id[
                            "intent_label_normal__normal"
                        ].to_loaded_clustering()
                    ),
                },
                {
                    "turn_index": 2,
                    "clustering": _clustering_payload_from_loaded(final_predictions),
                },
            ],
            turn_injection=_granularity_turn_injection_payload(edit_primitive="coarsen"),
            guidance="Group tickets by the normal intent POV.",
            human_turn_messages=[
                (1, "Group tickets by the normal intent POV."),
                (2, "Please coarsen the grouping."),
            ],
        )
    )

    assert result.details["source_reference_id"] == "intent_label_normal__normal"
    assert result.details["target_reference_id"] == "intent_label_coarse_b__coarse"
    assert result.metrics["target_change_success"] == pytest.approx(1.0)


def test_controllability_evaluator_uses_active_theme_fallback_for_turn_one_coarsen() -> None:
    evaluator = ControllabilityEvaluator.from_params()
    benchmark = _grouped_pov_benchmark_spec()

    result = evaluator.evaluate(
        _controllability_request(
            benchmark_spec=benchmark,
            predictions=benchmark.reference_by_id[
                "intent_label_coarse__coarse"
            ].to_loaded_clustering(),
            judge_clusterings=[
                {
                    "turn_index": 1,
                    "clustering": _clustering_payload_from_loaded(
                        benchmark.reference_by_id[
                            "intent_label_coarse__coarse"
                        ].to_loaded_clustering()
                    ),
                }
            ],
            turn_injection=_granularity_turn_injection_payload(
                turn_index=1,
                edit_primitive="coarsen",
            ),
            guidance=(
                "Group tickets by the primary request type or problem category "
                "expressed by the ticket."
            ),
        )
    )

    assert result.details["pre_request_turn"] == 0
    assert result.details["source_theme_id"] == "intent_label_normal"
    assert result.details["source_reference_id"] == "intent_label_normal__normal"
    assert result.details["target_reference_id"] == "intent_label_coarse__coarse"


def test_controllability_evaluator_rejects_coarsen_from_boundary_granularity() -> None:
    evaluator = ControllabilityEvaluator.from_params()
    benchmark = _grouped_pov_benchmark_spec()

    result = evaluator.evaluate(
        _controllability_request(
            benchmark_spec=benchmark,
            predictions=benchmark.reference_by_id[
                "intent_label_coarse__coarse"
            ].to_loaded_clustering(),
            judge_clusterings=[
                {
                    "turn_index": 1,
                    "clustering": _clustering_payload_from_loaded(
                        benchmark.reference_by_id[
                            "intent_label_coarse__coarse"
                        ].to_loaded_clustering()
                    ),
                },
                {
                    "turn_index": 2,
                    "clustering": _clustering_payload_from_loaded(
                        benchmark.reference_by_id[
                            "intent_label_coarse__coarse"
                        ].to_loaded_clustering()
                    ),
                },
            ],
            turn_injection=_granularity_turn_injection_payload(edit_primitive="coarsen"),
            guidance=(
                "Group tickets by the primary request type or problem "
                "category expressed by the ticket."
            ),
        )
    )

    assert result.metrics == {}
    assert result.details["invalid_episode"] is True
    assert result.details["invalid_reason"] == "granularity_shift_at_boundary"
    assert result.details["source_theme_id"] == "intent_label_normal"
    assert result.details["source_granularity"] == "coarse"


def test_controllability_evaluator_rejects_refine_when_target_granularity_missing() -> None:
    evaluator = ControllabilityEvaluator.from_params()
    benchmark = parse_synthetic_benchmark_spec(
        {
            "dataset_name": "synthetic",
            "items": ["apple", "banana", "car", "truck"],
            "themes": [
                {
                    "theme_id": "fruit_coarse",
                    "label": "Fruit Coarse",
                    "guidance": "Group items into broad fruit buckets.",
                    "pov_group_id": "fruit_family",
                    "granularity": "coarse",
                },
                {
                    "theme_id": "fruit_normal",
                    "label": "Fruit Normal",
                    "guidance": "Group items by fruit family.",
                    "pov_group_id": "fruit_family",
                    "granularity": "normal",
                },
            ],
            "references": [
                {
                    "reference_id": "fruit_coarse__coarse",
                    "topic_id": "fruit_coarse",
                    "granularity": "coarse",
                    "pov_group_id": "fruit_family",
                    "assignment": {"apple": 0, "banana": 0, "car": 1, "truck": 1},
                },
                {
                    "reference_id": "fruit_normal__normal",
                    "topic_id": "fruit_normal",
                    "granularity": "normal",
                    "pov_group_id": "fruit_family",
                    "assignment": {"apple": 0, "banana": 1, "car": 2, "truck": 2},
                },
            ],
        }
    )

    with pytest.raises(ValueError, match="target granularity 'fine'"):
        evaluator.evaluate(
            _controllability_request(
                benchmark_spec=benchmark,
                predictions=benchmark.reference_by_id[
                    "fruit_normal__normal"
                ].to_loaded_clustering(),
                judge_clusterings=[
                    {
                        "turn_index": 1,
                        "clustering": _clustering_payload_from_loaded(
                            benchmark.reference_by_id["fruit_normal__normal"].to_loaded_clustering()
                        ),
                    },
                    {
                        "turn_index": 2,
                        "clustering": _clustering_payload_from_loaded(
                            benchmark.reference_by_id["fruit_normal__normal"].to_loaded_clustering()
                        ),
                    },
                ],
                turn_injection=_granularity_turn_injection_payload(edit_primitive="refine"),
                guidance="Group items by fruit family.",
            )
        )


def test_controllability_evaluator_rejects_missing_configured_source_theme_id() -> None:
    evaluator = ControllabilityEvaluator.from_params()

    with pytest.raises(
        ValueError,
        match="received invalid turn-injection metadata: source_theme_id is required",
    ):
        evaluator.evaluate(
            _controllability_request(
                turn_injection={
                    "turn_index": 2,
                    "edit_primitive": "theme_switch",
                    "target_theme_id": "fruit",
                }
            )
        )


def test_controllability_evaluator_rejects_unknown_configured_source_theme_id() -> None:
    evaluator = ControllabilityEvaluator.from_params()

    with pytest.raises(ValueError, match="source_theme_id='does_not_exist'"):
        evaluator.evaluate(
            _controllability_request(
                turn_injection=_theme_switch_turn_injection_payload(
                    source_theme_id="does_not_exist",
                    target_theme_id="fruit",
                )
            )
        )


def test_controllability_evaluator_rejects_unknown_configured_target_theme_id() -> None:
    evaluator = ControllabilityEvaluator.from_params()

    with pytest.raises(ValueError, match="target_theme_id='does_not_exist'"):
        evaluator.evaluate(
            _controllability_request(
                turn_injection=_theme_switch_turn_injection_payload(
                    target_theme_id="does_not_exist"
                )
            )
        )


def test_controllability_evaluator_uses_best_matching_target_reference_for_theme_switch() -> None:
    evaluator = ControllabilityEvaluator.from_params()
    benchmark = parse_synthetic_benchmark_spec(
        {
            "dataset_name": "enterprise_ticket",
            "items": ["okta locked", "badge broken", "invoice overdue", "vpn down"],
            "themes": [
                {
                    "theme_id": "department",
                    "label": "Department",
                    "guidance": "Group tickets by department.",
                },
                {
                    "theme_id": "issue_type",
                    "label": "Issue Type",
                    "guidance": "Group tickets by issue type.",
                },
            ],
            "references": [
                {
                    "reference_id": "department__normal",
                    "topic_id": "department",
                    "granularity": "normal",
                    "assignment": {
                        "okta locked": 0,
                        "badge broken": 1,
                        "invoice overdue": 2,
                        "vpn down": 1,
                    },
                    "cluster_names": ["Access", "IT Support", "Finance"],
                },
                {
                    "reference_id": "issue_type__normal",
                    "topic_id": "issue_type",
                    "granularity": "normal",
                    "assignment": {
                        "okta locked": 0,
                        "badge broken": 1,
                        "invoice overdue": 2,
                        "vpn down": 3,
                    },
                    "cluster_names": [
                        "authentication_issue",
                        "badge_issue",
                        "finance_issue",
                        "network_issue",
                    ],
                },
                {
                    "reference_id": "issue_type__coarse",
                    "topic_id": "issue_type",
                    "granularity": "coarse",
                    "assignment": {
                        "okta locked": 0,
                        "badge broken": 0,
                        "invoice overdue": 1,
                        "vpn down": 1,
                    },
                    "cluster_names": ["access_or_identity", "service_or_finance"],
                },
            ],
        }
    )
    final_predictions = benchmark.reference_by_id["issue_type__coarse"].to_loaded_clustering()

    result = evaluator.evaluate(
        _controllability_request(
            benchmark_spec=benchmark,
            predictions=final_predictions,
            judge_clusterings=[
                {
                    "turn_index": 1,
                    "clustering": _clustering_payload_from_loaded(
                        benchmark.reference_by_id["department__normal"].to_loaded_clustering()
                    ),
                },
                {
                    "turn_index": 2,
                    "clustering": _clustering_payload_from_loaded(
                        benchmark.reference_by_id["department__normal"].to_loaded_clustering()
                    ),
                },
                {
                    "turn_index": 3,
                    "clustering": _clustering_payload_from_loaded(final_predictions),
                },
            ],
            turn_injection=_theme_switch_turn_injection_payload(
                source_theme_id="department",
                target_theme_id="issue_type",
            ),
            guidance="Group tickets by department.",
            human_turn_messages=[
                (1, "Group tickets by department."),
                (2, "Please switch to issue type."),
                (3, "Keep the new issue-type grouping."),
            ],
        )
    )

    assert result.details["target_theme_id"] == "issue_type"
    assert result.details["target_reference_id"] == "issue_type__coarse"
    assert result.metrics["target_change_success"] == pytest.approx(1.0)


def test_controllability_evaluator_stays_within_grouped_pov_for_reference_selection() -> None:
    evaluator = ControllabilityEvaluator.from_params()
    benchmark = _grouped_pov_benchmark_spec()
    final_predictions = benchmark.reference_by_id["intent_label_fine__fine"].to_loaded_clustering()

    result = evaluator.evaluate(
        _controllability_request(
            benchmark_spec=benchmark,
            predictions=final_predictions,
            judge_clusterings=[
                {
                    "turn_index": 1,
                    "clustering": _clustering_payload_from_loaded(
                        benchmark.reference_by_id[
                            "organizational_label_coarse__coarse"
                        ].to_loaded_clustering()
                    ),
                },
                {
                    "turn_index": 2,
                    "clustering": _clustering_payload_from_loaded(
                        benchmark.reference_by_id[
                            "organizational_label_coarse__coarse"
                        ].to_loaded_clustering()
                    ),
                },
                {
                    "turn_index": 3,
                    "clustering": _clustering_payload_from_loaded(final_predictions),
                },
            ],
            turn_injection=_theme_switch_turn_injection_payload(
                source_theme_id="organizational_label_normal",
                target_theme_id="intent_label_normal",
            ),
            guidance="Group tickets by the normal organizational POV.",
            human_turn_messages=[
                (1, "Group tickets by the normal organizational POV."),
                (2, "Please switch to the normal intent POV."),
                (3, "Keep the new intent grouping."),
            ],
        )
    )

    assert result.details["source_theme_id"] == "organizational_label_normal"
    assert result.details["source_reference_id"] == "organizational_label_coarse__coarse"
    assert result.details["target_theme_id"] == "intent_label_normal"
    assert result.details["target_reference_id"] == "intent_label_fine__fine"
    assert result.metrics["target_change_success"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("theme_overrides", "expected_error"),
    [
        ({"theme_id": 123}, "themes\\[1\\]\\.theme_id must be a string"),
        ({"label": 123}, "themes\\[1\\]\\.label must be a string"),
        ({"guidance": 123}, "themes\\[1\\]\\.guidance must be a string"),
        ({"is_foil": "false"}, "themes\\[1\\]\\.is_foil must be a boolean"),
    ],
)
def test_parse_synthetic_benchmark_spec_rejects_invalid_theme_field_types(
    theme_overrides: dict[str, object], expected_error: str
) -> None:
    payload = {
        "dataset_name": "synthetic",
        "items": ["apple", "banana"],
        "themes": [
            {
                "theme_id": "fruit",
                "label": "Fruit",
                "guidance": "Group items by fruitness.",
                **theme_overrides,
            }
        ],
        "references": [
            {
                "reference_id": "fruit__normal",
                "topic_id": "fruit",
                "granularity": "normal",
                "assignment": {"apple": 0, "banana": 0},
            }
        ],
    }

    with pytest.raises(ValueError, match=expected_error):
        parse_synthetic_benchmark_spec(payload)


def test_parse_synthetic_benchmark_spec_requires_non_empty_references_list() -> None:
    with pytest.raises(
        ValueError, match="Synthetic benchmark spec must define a non-empty 'references' list"
    ):
        parse_synthetic_benchmark_spec(
            {
                "dataset_name": "synthetic",
                "items": ["apple", "banana"],
                "themes": [
                    {
                        "theme_id": "fruit",
                        "label": "Fruit",
                        "guidance": "Group items by fruitness.",
                    },
                    {
                        "theme_id": "vehicle",
                        "label": "Vehicle",
                        "guidance": "Group items by vehicle type.",
                    },
                ],
            }
        )


def test_iter_batch_run_requests_loads_optional_artifacts_and_benchmark_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark_path = tmp_path / "synthetic.benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "dataset_name": "synthetic",
                "items": ["apple", "banana"],
                "themes": [
                    {
                        "theme_id": "fruit",
                        "label": "Fruit",
                        "guidance": "Group items by fruitness.",
                    },
                    {
                        "theme_id": "vehicle",
                        "label": "Vehicle",
                        "guidance": "Group items by vehicle type.",
                    },
                ],
                "references": [
                    {
                        "reference_id": "fruit__normal",
                        "topic_id": "fruit",
                        "granularity": "normal",
                        "assignment": {"apple": 1, "banana": 1},
                        "cluster_names": ["other", "fruit"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        cast(dict[str, str], DATASETS["synthetic"]),
        "benchmark_spec_path",
        str(benchmark_path),
    )

    input_dir = tmp_path / "batch_output"
    run_dir = input_dir / "cfg" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "clustering.json").write_text(json.dumps(_predictions_payload()), encoding="utf-8")
    (run_dir / "conversation_trace.json").write_text(
        json.dumps({"messages": [{"role": "human", "content": "hello"}]}), encoding="utf-8"
    )
    (run_dir / "clustering_pov_history.json").write_text(
        json.dumps(
            {
                "snapshots": [
                    {"turn_index": 1, "povs": ["fruit"], "num_povs": 1},
                    {"turn_index": 2, "povs": ["fruit", "color"], "num_povs": 2},
                ],
                "num_snapshots": 2,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "topic_candidates.json").write_text(
        json.dumps({"topics": [{"topic_id": "fruit", "label": "Fruit", "guidance": "fruit"}]}),
        encoding="utf-8",
    )
    (run_dir / "selected_topic.json").write_text(
        json.dumps(
            {"topic_id": "fruit", "label": "Fruit", "guidance": "Group items by fruitness."}
        ),
        encoding="utf-8",
    )
    (input_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "config": {"dataset_name": "synthetic", "human_agent_config": {"guidance": "x"}},
                "config_slug": "cfg",
                "run_ids": ["run-1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    requests = list(iter_batch_run_requests(input_dir))
    assert len(requests) == 1
    request = requests[0]
    assert request.artifacts["clustering_pov_history"]["num_snapshots"] == 2
    assert request.artifacts["topic_candidates"]["topics"][0]["topic_id"] == "fruit"
    assert request.artifacts["selected_topic"]["topic_id"] == "fruit"
    assert request.artifacts["benchmark_spec"] is not None


def test_iter_batch_run_requests_prefers_generated_benchmark_spec_from_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default_benchmark_path = tmp_path / "synthetic.default.benchmark.json"
    default_benchmark_path.write_text(
        json.dumps(
            {
                "dataset_name": "synthetic",
                "items": ["apple", "banana", "car"],
                "themes": [
                    {
                        "theme_id": "fruit",
                        "label": "Fruit",
                        "guidance": "Group items by fruitness.",
                    },
                    {
                        "theme_id": "vehicle",
                        "label": "Vehicle",
                        "guidance": "Group items by vehicle type.",
                    },
                ],
                "references": [
                    {
                        "reference_id": "fruit__normal",
                        "topic_id": "fruit",
                        "granularity": "normal",
                        "assignment": {"apple": 1, "banana": 1, "car": 0},
                        "cluster_names": ["other", "fruit"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        cast(dict[str, str], DATASETS["synthetic"]),
        "benchmark_spec_path",
        str(default_benchmark_path),
    )

    input_dir = tmp_path / "batch_output"
    generated_benchmark_path = input_dir / "_benchmark_specs" / "synthetic.sample.benchmark.json"
    generated_benchmark_path.parent.mkdir(parents=True)
    generated_benchmark_path.write_text(
        json.dumps(
            {
                "dataset_name": "synthetic",
                "items": ["apple", "banana"],
                "themes": [
                    {
                        "theme_id": "fruit",
                        "label": "Fruit",
                        "guidance": "Group items by fruitness.",
                    },
                    {
                        "theme_id": "vehicle",
                        "label": "Vehicle",
                        "guidance": "Group items by vehicle type.",
                    },
                ],
                "references": [
                    {
                        "reference_id": "fruit__normal",
                        "topic_id": "fruit",
                        "granularity": "normal",
                        "assignment": {"apple": 1, "banana": 1},
                        "cluster_names": ["other", "fruit"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    run_dir = input_dir / "cfg" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "clustering.json").write_text(json.dumps(_predictions_payload()), encoding="utf-8")
    (run_dir / "conversation_trace.json").write_text(
        json.dumps({"messages": [{"role": "human", "content": "hello"}]}),
        encoding="utf-8",
    )
    (input_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "config": {
                    "dataset_name": "synthetic",
                    "human_agent_config": {"guidance": "x"},
                    "metadata": {
                        GENERATED_BENCHMARK_SPEC_METADATA_KEY: (
                            "_benchmark_specs/synthetic.sample.benchmark.json"
                        )
                    },
                },
                "config_slug": "cfg",
                "run_ids": ["run-1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    requests = list(iter_batch_run_requests(input_dir))

    assert len(requests) == 1
    benchmark_spec = request.artifacts["benchmark_spec"] if (request := requests[0]) else None
    assert benchmark_spec is not None
    assert benchmark_spec.item_texts == ("apple", "banana")


def test_iter_batch_run_requests_loads_shared_topic_candidates_for_topic_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark_path = tmp_path / "enterprise_ticket.benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "dataset_name": "enterprise_ticket",
                "items": ["ticket a", "ticket b"],
                "themes": [
                    {
                        "theme_id": "department",
                        "label": "Department",
                        "guidance": "Group tickets by department.",
                    }
                ],
                "references": [
                    {
                        "reference_id": "department__normal",
                        "topic_id": "department",
                        "granularity": "normal",
                        "assignment": {"ticket a": 0, "ticket b": 0},
                        "cluster_names": ["department"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        cast(dict[str, str], DATASETS["enterprise_ticket"]),
        "benchmark_spec_path",
        str(benchmark_path),
    )

    input_dir = tmp_path / "batch_output"
    run_dir = input_dir / "cfg--topic-abc123" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "clustering.json").write_text(
        json.dumps({"clusters": [{"name": "department", "members": ["ticket a", "ticket b"]}]}),
        encoding="utf-8",
    )
    (run_dir / "conversation_trace.json").write_text(
        json.dumps({"messages": [{"role": "human", "content": "hello"}]}),
        encoding="utf-8",
    )
    (run_dir / "selected_topic.json").write_text(
        json.dumps(
            {
                "topic_id": "finance-and-procurement",
                "label": "Finance and Procurement",
                "guidance": "Group finance-related tickets.",
                "generated_at_turn": 3,
            }
        ),
        encoding="utf-8",
    )

    shared_dir = input_dir / "_shared" / "cfg" / "shared-run-1"
    shared_dir.mkdir(parents=True)
    (shared_dir / "topic_candidates.json").write_text(
        json.dumps(
            {
                "topics": [
                    {
                        "topic_id": "finance-and-procurement",
                        "label": "Finance and Procurement",
                        "guidance": "Group finance-related tickets.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (shared_dir / "topic_branch_mapping.json").write_text(
        json.dumps(
            {
                "shared_config_slug": "cfg",
                "shared_run_id": "shared-run-1",
                "branches": [
                    {
                        "config_slug": "cfg--topic-abc123",
                        "run_id": "run-1",
                        "topic_id": "finance-and-procurement",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    (input_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "config": {
                    "dataset_name": "enterprise_ticket",
                    "human_agent_config": {"guidance": None},
                    "metadata": {
                        "topic_branch_mode": "auto_topic_discovery",
                        "topic_branch_source_config_slug": "cfg",
                    },
                },
                "config_slug": "cfg--topic-abc123",
                "run_ids": ["run-1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    requests = list(iter_batch_run_requests(input_dir))

    assert len(requests) == 1
    request = requests[0]
    assert (
        request.artifacts["topic_candidates"]["topics"][0]["topic_id"] == "finance-and-procurement"
    )


def test_iter_batch_run_requests_coalesces_shared_topic_candidates_for_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark_path = tmp_path / "enterprise_ticket.benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "dataset_name": "enterprise_ticket",
                "items": ["ticket a", "ticket b"],
                "themes": [
                    {
                        "theme_id": "department",
                        "label": "Department",
                        "guidance": "Group tickets by department.",
                    }
                ],
                "references": [
                    {
                        "reference_id": "department__normal",
                        "topic_id": "department",
                        "granularity": "normal",
                        "assignment": {"ticket a": 0, "ticket b": 0},
                        "cluster_names": ["department"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        cast(dict[str, str], DATASETS["enterprise_ticket"]),
        "benchmark_spec_path",
        str(benchmark_path),
    )

    input_dir = tmp_path / "batch_output"
    for config_slug, run_id, topic_id in [
        ("cfg--topic-a", "run-1", "finance-and-procurement"),
        ("cfg--topic-b", "run-2", "identity-and-access"),
    ]:
        run_dir = input_dir / config_slug / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "clustering.json").write_text(
            json.dumps({"clusters": [{"name": "department", "members": ["ticket a", "ticket b"]}]}),
            encoding="utf-8",
        )
        (run_dir / "conversation_trace.json").write_text(
            json.dumps({"messages": [{"role": "human", "content": "hello"}]}),
            encoding="utf-8",
        )
        (run_dir / "selected_topic.json").write_text(
            json.dumps(
                {
                    "topic_id": topic_id,
                    "label": topic_id.replace("-", " ").title(),
                    "guidance": f"Group tickets by {topic_id}.",
                    "generated_at_turn": 3,
                }
            ),
            encoding="utf-8",
        )

    shared_dir = input_dir / "_shared" / "cfg" / "shared-run-1"
    shared_dir.mkdir(parents=True)
    (shared_dir / "topic_candidates.json").write_text(
        json.dumps(
            {
                "topics": [
                    {
                        "topic_id": "finance-and-procurement",
                        "label": "Finance And Procurement",
                        "guidance": "Group tickets by finance-and-procurement.",
                    },
                    {
                        "topic_id": "identity-and-access",
                        "label": "Identity And Access",
                        "guidance": "Group tickets by identity-and-access.",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (shared_dir / "topic_branch_mapping.json").write_text(
        json.dumps(
            {
                "shared_config_slug": "cfg",
                "shared_run_id": "shared-run-1",
                "branches": [
                    {
                        "config_slug": "cfg--topic-a",
                        "run_id": "run-1",
                        "topic_id": "finance-and-procurement",
                    },
                    {
                        "config_slug": "cfg--topic-b",
                        "run_id": "run-2",
                        "topic_id": "identity-and-access",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (input_dir / "results.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "config": {
                            "dataset_name": "enterprise_ticket",
                            "human_agent_config": {"guidance": None},
                            "metadata": {
                                "topic_branch_mode": "auto_topic_discovery",
                                "topic_branch_source_config_slug": "cfg",
                            },
                        },
                        "config_slug": "cfg--topic-a",
                        "run_ids": ["run-1"],
                    }
                ),
                json.dumps(
                    {
                        "config": {
                            "dataset_name": "enterprise_ticket",
                            "human_agent_config": {"guidance": None},
                            "metadata": {
                                "topic_branch_mode": "auto_topic_discovery",
                                "topic_branch_source_config_slug": "cfg",
                            },
                        },
                        "config_slug": "cfg--topic-b",
                        "run_ids": ["run-2"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    requests = list(iter_batch_run_requests(input_dir, coalesce_auto_topic_shared_runs=True))

    assert len(requests) == 1
    request = requests[0]
    assert request.metadata["config_slug"] == "cfg--topic-a"
    assert request.metadata["run_id"] == "run-1"
    assert len(request.artifacts["topic_candidates"]["topics"]) == 2


def test_run_evaluation_config_skips_benchmark_loading_for_non_benchmark_evaluators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_benchmark_path = tmp_path / "missing.benchmark.json"
    monkeypatch.setitem(
        cast(dict[str, str], DATASETS["synthetic"]),
        "benchmark_spec_path",
        str(missing_benchmark_path),
    )

    input_dir = tmp_path / "batch_output"
    run_dir = input_dir / "cfg" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "clustering.json").write_text(json.dumps(_predictions_payload()), encoding="utf-8")
    (run_dir / "conversation_trace.json").write_text(
        json.dumps({"messages": [{"role": "human", "content": "hello"}]}), encoding="utf-8"
    )
    (input_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "config": {"dataset_name": "synthetic", "human_agent_config": {"guidance": "x"}},
                "config_slug": "cfg",
                "run_ids": ["run-1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    evaluation_config_path = tmp_path / "evaluation.toml"
    evaluation_config_path.write_text(
        "\n".join(
            [
                f'input_dir = "{input_dir}"',
                'evaluators = ["basic_metrics"]',
                f'report_path = "{tmp_path / "report.json"}"',
                "",
                "[evaluator_params.basic_metrics]",
            ]
        ),
        encoding="utf-8",
    )

    report_path = run_evaluation_config(evaluation_config_path)

    assert Path(report_path).is_file()


def test_run_evaluation_config_loads_benchmark_spec_for_discovery_semantic_quality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conversational_clustering.evaluation.methods import discovery_semantic_quality as module

    benchmark_path = tmp_path / "synthetic.benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "dataset_name": "synthetic",
                "items": ["apple", "banana"],
                "themes": [
                    {
                        "theme_id": "fruit",
                        "label": "Fruit",
                        "guidance": "Group items by fruitness.",
                    }
                ],
                "references": [
                    {
                        "reference_id": "fruit__normal",
                        "topic_id": "fruit",
                        "granularity": "normal",
                        "assignment": {"apple": 0, "banana": 0},
                        "cluster_names": ["fruit"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        cast(dict[str, str], DATASETS["synthetic"]),
        "benchmark_spec_path",
        str(benchmark_path),
    )

    def mock_match(**_: object) -> ThemeMatchJudgment:
        return ThemeMatchJudgment(
            supported=True,
            benchmark_theme_id="fruit",
            confidence=0.9,
            reasoning="Matches fruit theme.",
        )

    def mock_duplicate(**_: object) -> DuplicateTopicJudgment:
        return DuplicateTopicJudgment(duplicate=False, confidence=0.9, reasoning="Distinct.")

    monkeypatch.setattr(module, "judge_theme_match", mock_match)
    monkeypatch.setattr(module, "judge_duplicate_topics", mock_duplicate)

    input_dir = tmp_path / "batch_output"
    run_dir = input_dir / "cfg" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "clustering.json").write_text(
        json.dumps({"clusters": [{"name": "fruit", "members": ["apple", "banana"]}]}),
        encoding="utf-8",
    )
    (run_dir / "conversation_trace.json").write_text(
        json.dumps({"messages": [{"role": "human", "content": "hello"}]}),
        encoding="utf-8",
    )
    (run_dir / "topic_candidates.json").write_text(
        json.dumps(
            {
                "topics": [
                    {
                        "topic_id": "fruit",
                        "label": "Fruit",
                        "guidance": "Group items by fruitness.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (input_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "config": {"dataset_name": "synthetic", "human_agent_config": {"guidance": "x"}},
                "config_slug": "cfg",
                "run_ids": ["run-1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    evaluation_config_path = tmp_path / "evaluation.toml"
    evaluation_config_path.write_text(
        "\n".join(
            [
                f'input_dir = "{input_dir}"',
                'evaluators = ["discovery_semantic_quality"]',
                f'report_path = "{tmp_path / "report.json"}"',
            ]
        ),
        encoding="utf-8",
    )

    report_path = run_evaluation_config(evaluation_config_path)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))

    assert report["summary_by_evaluator"]["discovery_semantic_quality"]["metrics"][
        "supported_theme_precision"
    ] == pytest.approx(1.0)


def test_run_evaluation_config_continues_after_single_evaluator_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conversational_clustering.evaluation import runner as module

    class FlakySingleEvaluator:
        name = "basic_metrics"
        resume_version = "v1"

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
            self.calls += 1
            if self.calls == 1:
                raise ValueError("synthetic single-evaluator failure")
            return EvaluationResult(
                method=self.name,
                metrics={"score": 1.0},
                details={"num_clusters": float(len(request.predictions.cluster_names))},
            )

    flaky_evaluator = FlakySingleEvaluator()
    monkeypatch.setattr(module, "get_evaluator", lambda name, params=None: flaky_evaluator)

    input_dir = tmp_path / "batch_output"
    for run_id in ("run-1", "run-2"):
        run_dir = input_dir / "cfg" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "clustering.json").write_text(
            json.dumps(_predictions_payload()),
            encoding="utf-8",
        )
        (run_dir / "conversation_trace.json").write_text(
            json.dumps({"messages": [{"role": "human", "content": "hello"}]}),
            encoding="utf-8",
        )
    (input_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "config": {"dataset_name": "synthetic", "human_agent_config": {"guidance": "x"}},
                "config_slug": "cfg",
                "run_ids": ["run-1", "run-2"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    evaluation_config_path = tmp_path / "evaluation.toml"
    evaluation_config_path.write_text(
        "\n".join(
            [
                f'input_dir = "{input_dir}"',
                'evaluators = ["basic_metrics"]',
                f'report_path = "{tmp_path / "report.json"}"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationRunFailed) as exc_info:
        run_evaluation_config(evaluation_config_path)
    report = json.loads(Path(exc_info.value.report_path).read_text(encoding="utf-8"))
    details_path = Path(report["results_files_by_evaluator"]["basic_metrics"])
    detail_rows = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert report["num_artifacts"] == 2
    assert report["num_failures_by_evaluator"] == {"basic_metrics": 1}
    assert report["summary_by_evaluator"]["basic_metrics"]["details"]["num_results"] == 1
    assert report["summary_by_config"]["cfg"]["num_failures_by_evaluator"] == {"basic_metrics": 1}
    assert len(detail_rows) == 2
    assert detail_rows[0]["status"] == "error"
    assert detail_rows[0]["details"]["error_message"] == "synthetic single-evaluator failure"
    assert detail_rows[1]["metrics"]["score"] == pytest.approx(1.0)


def test_run_evaluation_config_loads_benchmark_spec_for_controllability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark_path = tmp_path / "synthetic.benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "dataset_name": "synthetic",
                "items": ["apple", "banana", "car", "truck"],
                "themes": [
                    {
                        "theme_id": "fruit",
                        "label": "Fruit",
                        "guidance": "Group items by fruitness.",
                    },
                    {
                        "theme_id": "vehicle",
                        "label": "Vehicle",
                        "guidance": "Group items by vehicle type.",
                    },
                ],
                "references": [
                    {
                        "reference_id": "fruit__normal",
                        "topic_id": "fruit",
                        "granularity": "normal",
                        "assignment": {"apple": 1, "banana": 1, "car": 0, "truck": 0},
                        "cluster_names": ["other", "fruit"],
                    },
                    {
                        "reference_id": "fruit__initial",
                        "topic_id": "fruit",
                        "granularity": "initial",
                        "assignment": {"apple": 0, "banana": 0, "car": 0, "truck": 0},
                        "cluster_names": ["all_items"],
                    },
                    {
                        "reference_id": "vehicle__normal",
                        "topic_id": "vehicle",
                        "granularity": "normal",
                        "assignment": {"apple": 0, "banana": 0, "car": 1, "truck": 1},
                        "cluster_names": ["other", "vehicle"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        cast(dict[str, str], DATASETS["synthetic"]),
        "benchmark_spec_path",
        str(benchmark_path),
    )

    input_dir = tmp_path / "batch_output"
    run_dir = input_dir / "cfg" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "clustering.json").write_text(json.dumps(_predictions_payload()), encoding="utf-8")
    (run_dir / "conversation_trace.json").write_text(
        json.dumps({"messages": [{"role": "human", "content": "hello"}]}),
        encoding="utf-8",
    )
    (run_dir / "judge_clustering_trace.json").write_text(
        json.dumps(
            {
                "clusterings": [
                    {
                        "turn_index": 1,
                        "clustering": {
                            "clusters": [
                                {
                                    "name": "all",
                                    "members": ["apple", "banana", "car", "truck"],
                                }
                            ]
                        },
                    },
                    {
                        "turn_index": 2,
                        "clustering": _predictions_payload(),
                    },
                ],
                "num_clusterings": 2,
            }
        ),
        encoding="utf-8",
    )
    (input_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "config": {
                    "dataset_name": "synthetic",
                    "human_agent_config": {
                        "guidance": "Group items by fruitness.",
                        "turn_injection": {
                            "turn_index": 2,
                            "edit_primitive": "theme_switch",
                            "source_theme_id": "vehicle",
                            "target_theme_id": "fruit",
                        },
                    },
                },
                "config_slug": "cfg",
                "run_ids": ["run-1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    evaluation_config_path = tmp_path / "evaluation.toml"
    evaluation_config_path.write_text(
        "\n".join(
            [
                f'input_dir = "{input_dir}"',
                'evaluators = ["controllability"]',
                f'report_path = "{tmp_path / "report.json"}"',
                "",
                "[evaluator_params.controllability]",
                "alpha = 0.8",
                "beta = 0.1",
            ]
        ),
        encoding="utf-8",
    )

    report_path = run_evaluation_config(evaluation_config_path)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))

    assert Path(report_path).is_file()
    assert report["num_artifacts"] == 1
    assert report["num_failures_by_evaluator"] == {"controllability": 0}
    assert report["summary_by_evaluator"]["controllability"]["metrics"] == {
        "target_change_success": pytest.approx(1.0),
        "collateral_churn": pytest.approx(0.0),
        "lag_success_rate": pytest.approx(1.0),
    }


def test_run_evaluation_config_coalesces_discovery_evaluation_for_topic_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conversational_clustering.evaluation.methods import discovery_semantic_quality as module

    benchmark_path = tmp_path / "enterprise_ticket.benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "dataset_name": "enterprise_ticket",
                "items": ["ticket a", "ticket b"],
                "themes": [
                    {
                        "theme_id": "department",
                        "label": "Department",
                        "guidance": "Group tickets by department.",
                    }
                ],
                "references": [
                    {
                        "reference_id": "department__normal",
                        "topic_id": "department",
                        "granularity": "normal",
                        "assignment": {"ticket a": 0, "ticket b": 0},
                        "cluster_names": ["department"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        cast(dict[str, str], DATASETS["enterprise_ticket"]),
        "benchmark_spec_path",
        str(benchmark_path),
    )

    def mock_match(**_: object) -> ThemeMatchJudgment:
        return ThemeMatchJudgment(
            supported=True,
            benchmark_theme_id="department",
            confidence=0.9,
            reasoning="Matches department theme.",
        )

    def mock_duplicate(**_: object) -> DuplicateTopicJudgment:
        return DuplicateTopicJudgment(duplicate=False, confidence=0.9, reasoning="Distinct.")

    monkeypatch.setattr(module, "judge_theme_match", mock_match)
    monkeypatch.setattr(module, "judge_duplicate_topics", mock_duplicate)

    input_dir = tmp_path / "batch_output"
    for config_slug, run_id, topic_id in [
        ("cfg--topic-a", "run-1", "finance-and-procurement"),
        ("cfg--topic-b", "run-2", "identity-and-access"),
    ]:
        run_dir = input_dir / config_slug / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "clustering.json").write_text(
            json.dumps({"clusters": [{"name": "department", "members": ["ticket a", "ticket b"]}]}),
            encoding="utf-8",
        )
        (run_dir / "conversation_trace.json").write_text(
            json.dumps({"messages": [{"role": "human", "content": "hello"}]}),
            encoding="utf-8",
        )
        (run_dir / "selected_topic.json").write_text(
            json.dumps(
                {
                    "topic_id": topic_id,
                    "label": topic_id.replace("-", " ").title(),
                    "guidance": f"Group tickets by {topic_id}.",
                    "generated_at_turn": 3,
                }
            ),
            encoding="utf-8",
        )

    shared_dir = input_dir / "_shared" / "cfg" / "shared-run-1"
    shared_dir.mkdir(parents=True)
    (shared_dir / "topic_candidates.json").write_text(
        json.dumps(
            {
                "topics": [
                    {
                        "topic_id": "finance-and-procurement",
                        "label": "Finance And Procurement",
                        "guidance": "Group tickets by finance-and-procurement.",
                    },
                    {
                        "topic_id": "identity-and-access",
                        "label": "Identity And Access",
                        "guidance": "Group tickets by identity-and-access.",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (shared_dir / "topic_branch_mapping.json").write_text(
        json.dumps(
            {
                "shared_config_slug": "cfg",
                "shared_run_id": "shared-run-1",
                "branches": [
                    {
                        "config_slug": "cfg--topic-a",
                        "run_id": "run-1",
                        "topic_id": "finance-and-procurement",
                    },
                    {
                        "config_slug": "cfg--topic-b",
                        "run_id": "run-2",
                        "topic_id": "identity-and-access",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (input_dir / "results.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "config": {
                            "dataset_name": "enterprise_ticket",
                            "human_agent_config": {"guidance": None},
                            "metadata": {
                                "topic_branch_mode": "auto_topic_discovery",
                                "topic_branch_source_config_slug": "cfg",
                            },
                        },
                        "config_slug": "cfg--topic-a",
                        "run_ids": ["run-1"],
                    }
                ),
                json.dumps(
                    {
                        "config": {
                            "dataset_name": "enterprise_ticket",
                            "human_agent_config": {"guidance": None},
                            "metadata": {
                                "topic_branch_mode": "auto_topic_discovery",
                                "topic_branch_source_config_slug": "cfg",
                            },
                        },
                        "config_slug": "cfg--topic-b",
                        "run_ids": ["run-2"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    evaluation_config_path = tmp_path / "evaluation.toml"
    evaluation_config_path.write_text(
        "\n".join(
            [
                f'input_dir = "{input_dir}"',
                'evaluators = ["discovery_semantic_quality"]',
                f'report_path = "{tmp_path / "report.json"}"',
            ]
        ),
        encoding="utf-8",
    )

    report_path = run_evaluation_config(evaluation_config_path)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))

    assert report["num_artifacts"] == 1
    assert "vary_by" not in report
    assert report["num_artifacts_by_config"] == {"cfg": 1}
    assert set(report["summary_by_config"]) == {"cfg"}
    assert report["summary_by_config"]["cfg"]["num_artifacts"] == 1
    assert report["summary_by_evaluator"]["discovery_semantic_quality"]["metrics"][
        "supported_theme_precision"
    ] == pytest.approx(1.0)


def test_run_evaluation_config_loads_benchmark_spec_for_efficiency_utility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark_path = tmp_path / "synthetic.benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "dataset_name": "synthetic",
                "items": ["apple", "banana", "car", "truck"],
                "themes": [
                    {
                        "theme_id": "fruit",
                        "label": "Fruit",
                        "guidance": "Group items by fruitness.",
                    }
                ],
                "references": [
                    {
                        "reference_id": "fruit__normal",
                        "topic_id": "fruit",
                        "granularity": "normal",
                        "assignment": {"apple": 1, "banana": 1, "car": 0, "truck": 0},
                        "cluster_names": ["other", "fruit"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        cast(dict[str, str], DATASETS["synthetic"]),
        "benchmark_spec_path",
        str(benchmark_path),
    )

    input_dir = tmp_path / "batch_output"
    run_dir = input_dir / "cfg" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "clustering.json").write_text(json.dumps(_predictions_payload()), encoding="utf-8")
    (run_dir / "conversation_trace.json").write_text(
        json.dumps({"messages": [{"role": "human", "content": "hello"}]}), encoding="utf-8"
    )
    (run_dir / "judge_clustering_trace.json").write_text(
        json.dumps(
            {
                "clusterings": [
                    {
                        "turn_index": 1,
                        "clustering": _predictions_payload(),
                    }
                ],
                "num_clusterings": 1,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "selected_topic.json").write_text(
        json.dumps(
            {"topic_id": "fruit", "label": "Fruit", "guidance": "Group items by fruitness."}
        ),
        encoding="utf-8",
    )
    (input_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "config": {"dataset_name": "synthetic", "human_agent_config": {"guidance": "x"}},
                "config_slug": "cfg",
                "run_ids": ["run-1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    evaluation_config_path = tmp_path / "evaluation.toml"
    evaluation_config_path.write_text(
        "\n".join(
            [
                f'input_dir = "{input_dir}"',
                'evaluators = ["efficiency_utility"]',
                f'report_path = "{tmp_path / "report.json"}"',
            ]
        ),
        encoding="utf-8",
    )

    report_path = run_evaluation_config(evaluation_config_path)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))

    assert report["summary_by_evaluator"]["efficiency_utility"]["metrics"][
        "best_score_seen_structural"
    ] == pytest.approx(1.0)


def test_run_evaluation_config_reports_discovery_utility_auc_for_clustering_pov_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conversational_clustering.evaluation.methods import (
        clustering_pov_semantic_quality as module,
    )

    benchmark_path = tmp_path / "synthetic.benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "dataset_name": "synthetic",
                "items": ["apple", "banana"],
                "themes": [
                    {
                        "theme_id": "fruit",
                        "label": "Fruit",
                        "guidance": "Group items by fruitness.",
                    },
                    {
                        "theme_id": "vehicle",
                        "label": "Vehicle",
                        "guidance": "Group items by vehicle type.",
                    },
                ],
                "references": [
                    {
                        "reference_id": "fruit__normal",
                        "topic_id": "fruit",
                        "granularity": "normal",
                        "assignment": {"apple": 0, "banana": 0},
                        "cluster_names": ["fruit"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        cast(dict[str, str], DATASETS["synthetic"]),
        "benchmark_spec_path",
        str(benchmark_path),
    )

    def mock_match(**kwargs: object) -> ThemeMatchJudgment:
        surfaced_topic = cast(str, kwargs["surfaced_topic"])
        if surfaced_topic.lower() == "fruit":
            return ThemeMatchJudgment(
                supported=True,
                benchmark_theme_id="fruit",
                confidence=0.9,
                reasoning="Matches fruit theme.",
            )
        return ThemeMatchJudgment(
            supported=True,
            benchmark_theme_id="vehicle",
            confidence=0.9,
            reasoning="Matches vehicle theme.",
        )

    def mock_duplicate(**_: object) -> DuplicateTopicJudgment:
        return DuplicateTopicJudgment(duplicate=False, confidence=0.9, reasoning="Distinct.")

    monkeypatch.setattr(module, "judge_theme_match", mock_match)
    monkeypatch.setattr(module, "judge_duplicate_topics", mock_duplicate)

    input_dir = tmp_path / "batch_output"
    run_dir = input_dir / "cfg" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "clustering.json").write_text(
        json.dumps({"clusters": [{"name": "fruit", "members": ["apple", "banana"]}]}),
        encoding="utf-8",
    )
    (run_dir / "conversation_trace.json").write_text(
        json.dumps({"messages": [{"role": "human", "content": "hello"}]}),
        encoding="utf-8",
    )
    (run_dir / "clustering_pov_history.json").write_text(
        json.dumps(
            {
                "snapshots": [
                    {"turn_index": 1, "povs": ["Fruit"], "num_povs": 1},
                    {"turn_index": 3, "povs": ["Fruit", "Vehicle"], "num_povs": 2},
                ],
                "num_snapshots": 2,
            }
        ),
        encoding="utf-8",
    )
    (input_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "config": {"dataset_name": "synthetic", "human_agent_config": {"guidance": "x"}},
                "config_slug": "cfg",
                "run_ids": ["run-1"],
                "results": [{"turns_run": 4, "stopped_reason": "max_turns"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    evaluation_config_path = tmp_path / "evaluation.toml"
    evaluation_config_path.write_text(
        "\n".join(
            [
                f'input_dir = "{input_dir}"',
                'evaluators = ["clustering_pov_semantic_quality"]',
                f'report_path = "{tmp_path / "report.json"}"',
            ]
        ),
        encoding="utf-8",
    )

    report_path = run_evaluation_config(evaluation_config_path)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))

    assert report["summary_by_evaluator"]["clustering_pov_semantic_quality"]["metrics"][
        "discovery_utility_auc"
    ] == pytest.approx(0.75)


def test_run_evaluation_config_loads_benchmark_spec_for_inferred_theme_efficiency_utility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conversational_clustering.evaluation.methods import topic_conditioned_quality as module

    benchmark_path = tmp_path / "enterprise_ticket.benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "dataset_name": "enterprise_ticket",
                "items": ["okta locked", "badge broken", "invoice overdue", "vpn down"],
                "themes": [
                    {
                        "theme_id": "department",
                        "label": "Department",
                        "guidance": "Group tickets by department.",
                    }
                ],
                "references": [
                    {
                        "reference_id": "department__normal",
                        "topic_id": "department",
                        "granularity": "normal",
                        "assignment": {
                            "okta locked": 0,
                            "badge broken": 1,
                            "invoice overdue": 2,
                            "vpn down": 0,
                        },
                        "cluster_names": ["IT", "Facilities", "Finance"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        cast(dict[str, str], DATASETS["enterprise_ticket"]),
        "benchmark_spec_path",
        str(benchmark_path),
    )

    def mock_match(**_: object) -> ThemeMatchJudgment:
        return ThemeMatchJudgment(
            supported=False,
            benchmark_theme_id=None,
            confidence=0.2,
            reasoning="No semantic benchmark match.",
        )

    monkeypatch.setattr(module, "judge_theme_match", mock_match)

    input_dir = tmp_path / "batch_output"
    run_dir = input_dir / "cfg--topic-support-domain" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "clustering.json").write_text(
        json.dumps(
            {
                "clusters": [
                    {"name": "IT", "members": ["okta locked", "vpn down"]},
                    {"name": "Facilities", "members": ["badge broken"]},
                    {"name": "Finance", "members": ["invoice overdue"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "conversation_trace.json").write_text(
        json.dumps({"messages": [{"role": "human", "content": "hello"}]}), encoding="utf-8"
    )
    (run_dir / "judge_clustering_trace.json").write_text(
        json.dumps(
            {
                "clusterings": [
                    {
                        "turn_index": 1,
                        "clustering": {
                            "clusters": [
                                {"name": "IT", "members": ["okta locked", "vpn down"]},
                                {"name": "Facilities", "members": ["badge broken"]},
                                {"name": "Finance", "members": ["invoice overdue"]},
                            ]
                        },
                    }
                ],
                "num_clusterings": 1,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "selected_topic.json").write_text(
        json.dumps(
            {
                "topic_id": "support-domain",
                "label": "Support Domain",
                "guidance": "Group tickets by support domain.",
                "generated_at_turn": 3,
            }
        ),
        encoding="utf-8",
    )
    (input_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "config": {
                    "dataset_name": "enterprise_ticket",
                    "human_agent_config": {
                        "guidance": None,
                        "topic_discovery": {
                            "enabled": True,
                            "trigger_mode": "turn_budget",
                            "turn_budget": 3,
                            "max_rounds": 2,
                            "min_new_topics_per_round": 3,
                        },
                    },
                    "metadata": {
                        "topic_branch_mode": "auto_topic_discovery",
                        "topic_branch_topic_id": "support-domain",
                        "topic_branch_topic_label": "Support Domain",
                        "topic_branch_topic_guidance": "Group tickets by support domain.",
                    },
                },
                "config_slug": "cfg--topic-support-domain",
                "run_ids": ["run-1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    evaluation_config_path = tmp_path / "evaluation.toml"
    evaluation_config_path.write_text(
        "\n".join(
            [
                f'input_dir = "{input_dir}"',
                'evaluators = ["inferred_theme_efficiency_utility"]',
                f'report_path = "{tmp_path / "report.json"}"',
            ]
        ),
        encoding="utf-8",
    )

    report_path = run_evaluation_config(evaluation_config_path)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))

    assert report["summary_by_evaluator"]["inferred_theme_efficiency_utility"]["metrics"][
        "best_score_seen_structural"
    ] == pytest.approx(1.0)


def test_run_evaluation_config_loads_benchmark_spec_for_clarification_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conversational_clustering.evaluation.methods import clarification_policy as module

    benchmark_path = tmp_path / "synthetic.benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "dataset_name": "synthetic",
                "items": ["apple", "banana"],
                "themes": [
                    {
                        "theme_id": "fruit",
                        "label": "Fruit",
                        "guidance": "Group items by fruitness.",
                    }
                ],
                "references": [
                    {
                        "reference_id": "fruit__normal",
                        "topic_id": "fruit",
                        "granularity": "normal",
                        "assignment": {"apple": 0, "banana": 0},
                        "cluster_names": ["fruit"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        cast(dict[str, str], DATASETS["synthetic"]),
        "benchmark_spec_path",
        str(benchmark_path),
    )

    def mock_judge(**_: object) -> ClarificationOpportunityJudgment:
        return ClarificationOpportunityJudgment(
            opportunity_exists=True,
            clarification_needed=True,
            asked_clarification=True,
            useful_clarification=True,
            unnecessary_clarification=False,
            missed_clarification=False,
            downstream_gain=0.75,
            reasoning="Clarification improved the outcome.",
        )

    monkeypatch.setattr(module, "judge_clarification_opportunity", mock_judge)

    input_dir = tmp_path / "batch_output"
    run_dir = input_dir / "cfg" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "clustering.json").write_text(
        json.dumps({"clusters": [{"name": "fruit", "members": ["apple", "banana"]}]}),
        encoding="utf-8",
    )
    (run_dir / "conversation_trace.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "human", "content": "Group these items."},
                    {"role": "assistant", "content": "Which topic matters most?"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (input_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "config": {
                    "dataset_name": "synthetic",
                    "human_agent_config": {"guidance": "Group items by fruitness."},
                },
                "config_slug": "cfg",
                "run_ids": ["run-1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    evaluation_config_path = tmp_path / "evaluation.toml"
    evaluation_config_path.write_text(
        "\n".join(
            [
                f'input_dir = "{input_dir}"',
                'evaluators = ["clarification_policy"]',
                f'report_path = "{tmp_path / "report.json"}"',
            ]
        ),
        encoding="utf-8",
    )

    report_path = run_evaluation_config(evaluation_config_path)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))

    assert report["summary_by_evaluator"]["clarification_policy"]["metrics"][
        "useful_clarification_rate"
    ] == pytest.approx(1.0)


def test_load_benchmark_spec_for_dataset_rejects_mismatched_dataset_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark_path = tmp_path / "synthetic.benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "dataset_name": "different_dataset",
                "items": ["apple", "banana"],
                "themes": [
                    {
                        "theme_id": "fruit",
                        "label": "Fruit",
                        "guidance": "Group items by fruitness.",
                    }
                ],
                "references": [
                    {
                        "reference_id": "fruit__normal",
                        "topic_id": "fruit",
                        "granularity": "normal",
                        "assignment": {"apple": 1, "banana": 1},
                        "cluster_names": ["other", "fruit"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        cast(dict[str, str], DATASETS["synthetic"]),
        "benchmark_spec_path",
        str(benchmark_path),
    )

    with pytest.raises(ValueError, match="declares dataset_name='different_dataset'"):
        load_benchmark_spec_for_dataset("synthetic")


def test_discovery_semantic_quality_evaluator_scores_supported_and_duplicate_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import discovery_semantic_quality as module

    def mock_match(**kwargs: object) -> ThemeMatchJudgment:
        surfaced_topic = cast(str, kwargs["surfaced_topic"])
        if "fruit" in surfaced_topic.lower():
            return ThemeMatchJudgment(
                supported=True,
                benchmark_theme_id="fruit",
                confidence=0.9,
                reasoning="Matches fruit theme.",
            )
        return ThemeMatchJudgment(
            supported=True,
            benchmark_theme_id="vehicle",
            confidence=0.9,
            reasoning="Matches vehicle theme.",
        )

    def mock_duplicate(**kwargs: object) -> DuplicateTopicJudgment:
        return DuplicateTopicJudgment(duplicate=False, confidence=0.9, reasoning="Distinct topics.")

    monkeypatch.setattr(module, "judge_theme_match", mock_match)
    monkeypatch.setattr(module, "judge_duplicate_topics", mock_duplicate)

    evaluator = DiscoverySemanticQualityEvaluator.from_params()
    result = evaluator.evaluate(_request())

    assert result.metrics["supported_theme_precision"] == pytest.approx(1.0)
    assert result.metrics["distinct_valid_discovery_precision"] == pytest.approx(1.0)
    assert result.metrics["non_redundancy"] == pytest.approx(1.0)
    assert result.metrics["valid_theme_coverage"] == pytest.approx(1.0)
    assert result.metrics["num_supported_topics"] == pytest.approx(2.0)
    assert result.metrics["num_distinct_valid_topics"] == pytest.approx(2.0)


def test_discovery_semantic_quality_evaluator_returns_zero_count_metrics_for_empty_topics() -> None:
    request = _request()
    request.artifacts["topic_candidates"] = {"topics": []}

    evaluator = DiscoverySemanticQualityEvaluator.from_params()
    result = evaluator.evaluate(request)

    assert result.metrics["supported_theme_precision"] == pytest.approx(0.0)
    assert result.metrics["distinct_valid_discovery_precision"] == pytest.approx(0.0)
    assert result.metrics["non_redundancy"] == pytest.approx(0.0)
    assert result.metrics["valid_theme_coverage"] == pytest.approx(0.0)
    assert result.metrics["num_supported_topics"] == pytest.approx(0.0)
    assert result.metrics["num_distinct_valid_topics"] == pytest.approx(0.0)


def test_clustering_pov_semantic_quality_evaluator_requires_pov_history_artifact() -> None:
    request = _request()
    del request.artifacts["clustering_pov_history"]

    evaluator = ClusteringPovSemanticQualityEvaluator.from_params()

    with pytest.raises(ValueError, match="requires a clustering_pov_history artifact"):
        evaluator.evaluate(request)


def test_clustering_pov_semantic_quality_evaluator_returns_zero_metrics_for_empty_history() -> None:
    request = _request()
    request.artifacts["clustering_pov_history"] = {"snapshots": []}

    evaluator = ClusteringPovSemanticQualityEvaluator.from_params()
    result = evaluator.evaluate(request)

    assert result.metrics["supported_theme_precision"] == pytest.approx(0.0)
    assert result.metrics["distinct_valid_discovery_precision"] == pytest.approx(0.0)
    assert result.metrics["non_redundancy"] == pytest.approx(0.0)
    assert result.metrics["valid_theme_coverage"] == pytest.approx(0.0)
    assert result.metrics["discovery_utility_auc"] == pytest.approx(0.0)
    assert result.details["num_history_snapshots"] == 0
    assert result.details["history_judged_povs"] == []


def test_clustering_pov_semantic_quality_evaluator_scores_final_bank_and_auc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import (
        clustering_pov_semantic_quality as module,
    )

    def mock_match(**kwargs: object) -> ThemeMatchJudgment:
        surfaced_topic = cast(str, kwargs["surfaced_topic"])
        theme_id = "fruit" if surfaced_topic.lower() == "fruit" else "vehicle"
        return ThemeMatchJudgment(
            supported=True,
            benchmark_theme_id=theme_id,
            confidence=0.9,
            reasoning="Supported theme.",
        )

    def mock_duplicate(**_: object) -> DuplicateTopicJudgment:
        return DuplicateTopicJudgment(duplicate=False, confidence=0.9, reasoning="Distinct.")

    monkeypatch.setattr(module, "judge_theme_match", mock_match)
    monkeypatch.setattr(module, "judge_duplicate_topics", mock_duplicate)

    request = _request()
    request.artifacts["clustering_pov_history"] = {
        "snapshots": [
            {"turn_index": 1, "povs": ["Fruit"], "num_povs": 1},
            {"turn_index": 3, "povs": ["Fruit", "Vehicle"], "num_povs": 2},
        ]
    }
    request.artifacts["turns_run"] = 4

    evaluator = ClusteringPovSemanticQualityEvaluator.from_params()
    result = evaluator.evaluate(request)

    assert result.metrics["supported_theme_precision"] == pytest.approx(1.0)
    assert result.metrics["distinct_valid_discovery_precision"] == pytest.approx(1.0)
    assert result.metrics["non_redundancy"] == pytest.approx(1.0)
    assert result.metrics["valid_theme_coverage"] == pytest.approx(1.0)
    assert result.metrics["discovery_utility_auc"] == pytest.approx(0.75)
    assert result.details["discovery_utility_curve"] == [
        {"turn_index": 1, "utility": 0.5},
        {"turn_index": 2, "utility": 0.5},
        {"turn_index": 3, "utility": 1.0},
        {"turn_index": 4, "utility": 1.0},
    ]


def test_clustering_pov_semantic_quality_evaluator_collapses_duplicate_final_povs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import (
        clustering_pov_semantic_quality as module,
    )

    def mock_match(**kwargs: object) -> ThemeMatchJudgment:
        surfaced_topic = cast(str, kwargs["surfaced_topic"]).lower()
        theme_id = "vehicle" if "vehicle" in surfaced_topic else "fruit"
        return ThemeMatchJudgment(
            supported=True,
            benchmark_theme_id=theme_id,
            confidence=0.9,
            reasoning="Supported theme.",
        )

    def mock_duplicate(**kwargs: object) -> DuplicateTopicJudgment:
        left_topic = cast(str, kwargs["left_topic"]).lower()
        right_topic = cast(str, kwargs["right_topic"]).lower()
        duplicate = "fruit" in left_topic and "fruit" in right_topic
        return DuplicateTopicJudgment(
            duplicate=duplicate,
            confidence=0.9,
            reasoning="Duplicate fruit phrasing." if duplicate else "Distinct.",
        )

    monkeypatch.setattr(module, "judge_theme_match", mock_match)
    monkeypatch.setattr(module, "judge_duplicate_topics", mock_duplicate)

    request = _request()
    request.artifacts["clustering_pov_history"] = {
        "snapshots": [
            {
                "turn_index": 2,
                "povs": ["Fruit", "Fruit grouping", "Vehicle"],
                "num_povs": 3,
            }
        ]
    }

    evaluator = ClusteringPovSemanticQualityEvaluator.from_params()
    result = evaluator.evaluate(request)

    assert result.metrics["supported_theme_precision"] == pytest.approx(1.0)
    assert result.metrics["distinct_valid_discovery_precision"] == pytest.approx(2 / 3)


def test_clustering_pov_semantic_quality_auc_falls_back_to_last_snapshot_turn_and_dedupes_themes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import (
        clustering_pov_semantic_quality as module,
    )

    def mock_match(**kwargs: object) -> ThemeMatchJudgment:
        surfaced_topic = cast(str, kwargs["surfaced_topic"]).lower()
        if "unsupported" in surfaced_topic:
            return ThemeMatchJudgment(
                supported=False,
                benchmark_theme_id=None,
                confidence=0.9,
                reasoning="Unsupported topic.",
            )
        return ThemeMatchJudgment(
            supported=True,
            benchmark_theme_id="fruit",
            confidence=0.9,
            reasoning="Fruit theme.",
        )

    def mock_duplicate(**_: object) -> DuplicateTopicJudgment:
        return DuplicateTopicJudgment(duplicate=False, confidence=0.9, reasoning="Distinct.")

    monkeypatch.setattr(module, "judge_theme_match", mock_match)
    monkeypatch.setattr(module, "judge_duplicate_topics", mock_duplicate)

    request = _request()
    request.artifacts["clustering_pov_history"] = {
        "snapshots": [
            {"turn_index": 1, "povs": ["Fruit"], "num_povs": 1},
            {"turn_index": 2, "povs": ["Fruit grouping"], "num_povs": 1},
            {"turn_index": 3, "povs": ["Unsupported angle"], "num_povs": 1},
        ]
    }
    request.artifacts["turns_run"] = None

    evaluator = ClusteringPovSemanticQualityEvaluator.from_params()
    result = evaluator.evaluate(request)

    assert result.details["total_turns_evaluated"] == 3
    assert result.metrics["discovery_utility_auc"] == pytest.approx(0.5)
    assert result.details["first_valid_discovery_turn_by_theme"] == {"fruit": 1}


def test_clustering_pov_semantic_quality_foil_theme_does_not_count_toward_auc_or_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import (
        clustering_pov_semantic_quality as module,
    )

    def mock_match(**kwargs: object) -> ThemeMatchJudgment:
        surfaced_topic = cast(str, kwargs["surfaced_topic"]).lower()
        theme_id = "astrology" if "astrology" in surfaced_topic else "fruit"
        return ThemeMatchJudgment(
            supported=True,
            benchmark_theme_id=theme_id,
            confidence=0.9,
            reasoning="Matched theme.",
        )

    def mock_duplicate(**_: object) -> DuplicateTopicJudgment:
        return DuplicateTopicJudgment(duplicate=False, confidence=0.9, reasoning="Distinct.")

    monkeypatch.setattr(module, "judge_theme_match", mock_match)
    monkeypatch.setattr(module, "judge_duplicate_topics", mock_duplicate)

    request = _request()
    request.artifacts["clustering_pov_history"] = {
        "snapshots": [
            {"turn_index": 1, "povs": ["Astrology"], "num_povs": 1},
            {"turn_index": 2, "povs": ["Astrology", "Fruit"], "num_povs": 2},
        ]
    }
    request.artifacts["turns_run"] = 2

    evaluator = ClusteringPovSemanticQualityEvaluator.from_params()
    result = evaluator.evaluate(request)

    assert result.metrics["valid_theme_coverage"] == pytest.approx(0.5)
    assert result.metrics["discovery_utility_auc"] == pytest.approx(0.25)
    assert result.details["first_valid_discovery_turn_by_theme"] == {"fruit": 2}


def test_discovery_semantic_quality_evaluator_counts_duplicate_groups_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import discovery_semantic_quality as module

    def mock_match(**_: object) -> ThemeMatchJudgment:
        return ThemeMatchJudgment(
            supported=True,
            benchmark_theme_id="fruit",
            confidence=0.9,
            reasoning="Matches fruit theme.",
        )

    def mock_duplicate(**_: object) -> DuplicateTopicJudgment:
        return DuplicateTopicJudgment(duplicate=True, confidence=0.9, reasoning="Duplicate.")

    monkeypatch.setattr(module, "judge_theme_match", mock_match)
    monkeypatch.setattr(module, "judge_duplicate_topics", mock_duplicate)

    request = _request()
    request.artifacts["topic_candidates"] = {
        "topics": [
            {"topic_id": "fruit-1", "label": "Fruit 1", "guidance": "Fruit topic one."},
            {"topic_id": "fruit-2", "label": "Fruit 2", "guidance": "Fruit topic two."},
            {"topic_id": "fruit-3", "label": "Fruit 3", "guidance": "Fruit topic three."},
            {"topic_id": "fruit-4", "label": "Fruit 4", "guidance": "Fruit topic four."},
        ]
    }

    evaluator = DiscoverySemanticQualityEvaluator.from_params()
    result = evaluator.evaluate(request)

    assert result.metrics["supported_theme_precision"] == pytest.approx(1.0)
    assert result.metrics["distinct_valid_discovery_precision"] == pytest.approx(0.25)
    assert result.metrics["non_redundancy"] == pytest.approx(0.25)
    assert result.metrics["valid_theme_coverage"] == pytest.approx(0.5)


def test_discovery_semantic_quality_evaluator_collapses_supported_duplicate_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import discovery_semantic_quality as module

    def mock_match(**_: object) -> ThemeMatchJudgment:
        return ThemeMatchJudgment(
            supported=True,
            benchmark_theme_id="fruit",
            confidence=0.9,
            reasoning="Matches fruit theme.",
        )

    duplicate_pairs = {
        (0, 2),
        (1, 2),
    }

    def mock_duplicate(**kwargs: object) -> DuplicateTopicJudgment:
        left_topic = cast(str, kwargs["left_topic"])
        right_topic = cast(str, kwargs["right_topic"])
        left_index = int(left_topic.split("fruit-")[1].split(" |", 1)[0]) - 1
        right_index = int(right_topic.split("fruit-")[1].split(" |", 1)[0]) - 1
        return DuplicateTopicJudgment(
            duplicate=(left_index, right_index) in duplicate_pairs,
            confidence=0.9,
            reasoning="Duplicate component edge.",
        )

    monkeypatch.setattr(module, "judge_theme_match", mock_match)
    monkeypatch.setattr(module, "judge_duplicate_topics", mock_duplicate)

    request = _request()
    request.artifacts["topic_candidates"] = {
        "topics": [
            {"topic_id": "fruit-1", "label": "Fruit 1", "guidance": "Fruit topic one."},
            {"topic_id": "fruit-2", "label": "Fruit 2", "guidance": "Fruit topic two."},
            {"topic_id": "fruit-3", "label": "Fruit 3", "guidance": "Fruit topic three."},
        ]
    }

    evaluator = DiscoverySemanticQualityEvaluator.from_params()
    result = evaluator.evaluate(request)

    assert result.metrics["supported_theme_precision"] == pytest.approx(1.0)
    assert result.metrics["distinct_valid_discovery_precision"] == pytest.approx(1 / 3)
    assert result.metrics["non_redundancy"] == pytest.approx(1 / 3)
    assert result.metrics["valid_theme_coverage"] == pytest.approx(0.5)


def test_discovery_semantic_quality_penalizes_supported_topics_duplicated_by_unsupported_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import discovery_semantic_quality as module

    def mock_match(**kwargs: object) -> ThemeMatchJudgment:
        surfaced_topic = cast(str, kwargs["surfaced_topic"])
        if "fruit" in surfaced_topic.lower():
            return ThemeMatchJudgment(
                supported=True,
                benchmark_theme_id="fruit",
                confidence=0.9,
                reasoning="Matches fruit theme.",
            )
        return ThemeMatchJudgment(
            supported=False,
            benchmark_theme_id=None,
            confidence=0.9,
            reasoning="Unsupported topic.",
        )

    def mock_duplicate(**_: object) -> DuplicateTopicJudgment:
        return DuplicateTopicJudgment(duplicate=True, confidence=0.9, reasoning="Duplicate.")

    monkeypatch.setattr(module, "judge_theme_match", mock_match)
    monkeypatch.setattr(module, "judge_duplicate_topics", mock_duplicate)

    request = _request()
    request.artifacts["topic_candidates"] = {
        "topics": [
            {"topic_id": "fruit", "label": "Fruit", "guidance": "Fruit topic."},
            {"topic_id": "other", "label": "Other", "guidance": "Off-topic unsupported phrasing."},
        ]
    }

    evaluator = DiscoverySemanticQualityEvaluator.from_params()
    result = evaluator.evaluate(request)

    assert result.metrics["supported_theme_precision"] == pytest.approx(0.5)
    assert result.metrics["distinct_valid_discovery_precision"] == pytest.approx(0.0)
    assert result.metrics["non_redundancy"] == pytest.approx(0.5)
    assert result.metrics["valid_theme_coverage"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("params", "expected_error"),
    [
        ({"theme_match_prompt_id": "does_not_exist"}, "Invalid theme_match_prompt_id"),
        ({"duplicate_prompt_id": "does_not_exist"}, "Invalid duplicate_prompt_id"),
    ],
)
def test_discovery_semantic_quality_evaluator_rejects_invalid_prompt_ids(
    params: dict[str, str], expected_error: str
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        DiscoverySemanticQualityEvaluator.from_params(params)


@pytest.mark.parametrize(
    ("params", "expected_error"),
    [
        ({"theme_match_prompt_id": "does_not_exist"}, "Invalid theme_match_prompt_id"),
        ({"duplicate_prompt_id": "does_not_exist"}, "Invalid duplicate_prompt_id"),
    ],
)
def test_clustering_pov_semantic_quality_evaluator_rejects_invalid_prompt_ids(
    params: dict[str, str], expected_error: str
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        ClusteringPovSemanticQualityEvaluator.from_params(params)


def test_topic_conditioned_quality_evaluator_rejects_invalid_prompt_ids() -> None:
    with pytest.raises(ValueError, match="Invalid theme_match_prompt_id"):
        TopicConditionedQualityEvaluator.from_params({"theme_match_prompt_id": "does_not_exist"})


def test_cluster_semantic_quality_evaluator_reports_coherence_and_distinctiveness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import cluster_semantic_quality as module

    def mock_coherence(**_: object) -> ClusterCoherenceJudgment:
        return ClusterCoherenceJudgment(
            score=8.0,
            supported_by_guidance=True,
            reasoning="Coherent cluster.",
        )

    def mock_distinctiveness(**_: object) -> ClusterDistinctivenessJudgment:
        return ClusterDistinctivenessJudgment(
            score=7.0,
            overlapping_or_redundant=False,
            reasoning="Distinct clusters.",
        )

    monkeypatch.setattr(module, "judge_cluster_coherence", mock_coherence)
    monkeypatch.setattr(module, "judge_cluster_distinctiveness", mock_distinctiveness)

    evaluator = ClusterSemanticQualityEvaluator.from_params()
    result = evaluator.evaluate(_request())

    assert result.metrics["average_cluster_coherence"] == pytest.approx(0.8)
    assert result.metrics["cluster_distinctiveness"] == pytest.approx(0.7)
    assert result.metrics["supported_cluster_rate"] == pytest.approx(1.0)


def test_cluster_semantic_quality_evaluator_returns_zero_distinctiveness_for_single_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import cluster_semantic_quality as module

    def mock_coherence(**_: object) -> ClusterCoherenceJudgment:
        return ClusterCoherenceJudgment(
            score=8.0,
            supported_by_guidance=True,
            reasoning="Coherent cluster.",
        )

    def fail_distinctiveness(**_: object) -> ClusterDistinctivenessJudgment:
        raise AssertionError("distinctiveness should not be judged for a single non-empty cluster")

    monkeypatch.setattr(module, "judge_cluster_coherence", mock_coherence)
    monkeypatch.setattr(module, "judge_cluster_distinctiveness", fail_distinctiveness)

    base_request = _request()
    request = EvaluationRequest(
        predictions=LoadedClustering(
            assignment={"apple": 0, "banana": 0, "car": 0, "truck": 0},
            cluster_names=["all_items"],
        ),
        artifacts=base_request.artifacts,
        metadata=base_request.metadata,
    )

    evaluator = ClusterSemanticQualityEvaluator.from_params()
    result = evaluator.evaluate(request)

    assert result.metrics["average_cluster_coherence"] == pytest.approx(0.8)
    assert result.metrics["cluster_distinctiveness"] == pytest.approx(0.0)
    assert result.metrics["supported_cluster_rate"] == pytest.approx(1.0)


def test_cluster_semantic_quality_prices_default_judge_model_when_model_config_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import cluster_semantic_quality as module

    captured_configs: list[object] = []

    def mock_coherence(**_: object) -> ClusterCoherenceJudgment:
        return ClusterCoherenceJudgment(
            score=8.0,
            supported_by_guidance=True,
            reasoning="Coherent cluster.",
        )

    def mock_distinctiveness(**_: object) -> ClusterDistinctivenessJudgment:
        return ClusterDistinctivenessJudgment(
            score=7.0,
            overlapping_or_redundant=False,
            reasoning="Distinct clusters.",
        )

    class FakePricingClient:
        def get_text_prices(self, model_id: str) -> None:
            _ = model_id
            return None

    def fake_build_pricing_client_from_configs(configs: list[object]) -> FakePricingClient:
        captured_configs.extend(configs)
        return FakePricingClient()

    monkeypatch.setattr(module, "judge_cluster_coherence", mock_coherence)
    monkeypatch.setattr(module, "judge_cluster_distinctiveness", mock_distinctiveness)
    monkeypatch.setattr(
        module,
        "build_pricing_client_from_configs",
        fake_build_pricing_client_from_configs,
    )

    evaluator = ClusterSemanticQualityEvaluator.from_params()
    _ = evaluator.evaluate(_request())

    assert captured_configs == [default_semantic_consistency_judge_model_config()]


def test_judge_cluster_distinctiveness_retries_on_null_structured_output() -> None:
    from conversational_clustering.evaluation.methods import semantic_quality_judges as module

    class FakeStructuredLLM:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, prompt: str) -> object:
            self.calls += 1
            if self.calls == 1:
                return None
            assert "Objective guidance" in prompt
            return ClusterDistinctivenessJudgment(
                score=7.0,
                overlapping_or_redundant=False,
                reasoning="Distinct clusters.",
            )

    fake_llm = FakeStructuredLLM()
    result = module.judge_cluster_distinctiveness(
        guidance="Group items by fruitness.",
        left_cluster_name="fruit",
        left_cluster_members="- apple\n- banana",
        right_cluster_name="vehicle",
        right_cluster_members="- car\n- truck",
        structured_llm=lambda: fake_llm,
    )

    assert result.score == pytest.approx(7.0)
    assert result.overlapping_or_redundant is False
    assert fake_llm.calls == 2


def test_judge_cluster_distinctiveness_retries_on_schema_parse_failure() -> None:
    from conversational_clustering.evaluation.methods import semantic_quality_judges as module

    class FakeStructuredLLM:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, prompt: str) -> object:
            self.calls += 1
            if self.calls == 1:
                ClusterDistinctivenessJudgment.model_validate(
                    {
                        "score": "bad",
                        "overlapping_or_redundant": False,
                        "reasoning": "Invalid score type.",
                    }
                )
            assert "Objective guidance" in prompt
            return ClusterDistinctivenessJudgment(
                score=7.0,
                overlapping_or_redundant=False,
                reasoning="Distinct clusters.",
            )

    fake_llm = FakeStructuredLLM()
    result = module.judge_cluster_distinctiveness(
        guidance="Group items by fruitness.",
        left_cluster_name="fruit",
        left_cluster_members="- apple\n- banana",
        right_cluster_name="vehicle",
        right_cluster_members="- car\n- truck",
        structured_llm=lambda: fake_llm,
    )

    assert result.score == pytest.approx(7.0)
    assert result.overlapping_or_redundant is False
    assert fake_llm.calls == 2


def test_judge_theme_match_uses_default_judge_model_for_usage_fallback() -> None:
    from conversational_clustering.evaluation.methods import semantic_quality_judges as module

    collector = UsageCollector()
    result = module.judge_theme_match(
        surfaced_topic="Department routing",
        benchmark_themes="- department",
        usage_collector=collector,
        structured_llm=lambda: SimpleNamespace(
            invoke=lambda prompt: {
                "raw": SimpleNamespace(usage={"prompt_tokens": 5, "total_tokens": 5}),
                "parsed": ThemeMatchJudgment(
                    supported=True,
                    benchmark_theme_id="department",
                    confidence=0.9,
                    reasoning="Matches the benchmark theme.",
                ),
            }
        ),
    )

    usage = collector.snapshot(pricer=None)

    assert "Department routing" in result.reasoning or result.supported is True
    assert default_semantic_consistency_judge_model_config().model in usage.by_model
    assert usage.by_model[default_semantic_consistency_judge_model_config().model].total_tokens == 5


def test_topic_conditioned_quality_evaluator_compares_against_reference() -> None:
    evaluator = TopicConditionedQualityEvaluator.from_params({"granularity": "normal"})
    result = evaluator.evaluate(_request())

    assert result.metrics["cond_ari"] == pytest.approx(1.0)
    assert result.metrics["cond_nmi"] == pytest.approx(1.0)
    assert result.metrics["cond_vi"] == pytest.approx(0.0)
    assert result.details["topic_resolution_mode"] == "explicit_or_lexical"


def test_topic_conditioned_quality_evaluator_best_matches_within_same_pov_when_unspecified() -> (
    None
):
    evaluator = TopicConditionedQualityEvaluator.from_params()
    benchmark_spec = _grouped_pov_benchmark_spec()
    request = EvaluationRequest(
        predictions=benchmark_spec.reference_by_id[
            "intent_label_fine__fine"
        ].to_loaded_clustering(),
        artifacts={
            "benchmark_spec": benchmark_spec,
            "selected_topic": {
                "topic_id": "intent_label_normal",
                "label": "Intent (Normal)",
                "guidance": "Group tickets by the normal intent POV.",
            },
        },
        metadata={"config_slug": "cfg", "run_id": "run-1"},
    )

    result = evaluator.evaluate(request)

    assert result.details["topic_id"] == "intent_label_normal"
    assert result.details["reference_id"] == "intent_label_fine__fine"
    assert result.details["granularity"] == "fine"
    assert result.details["requested_granularity"] is None
    assert result.metrics["cond_ari"] == pytest.approx(1.0)


def test_topic_conditioned_quality_evaluator_keeps_requested_granularity_within_same_pov() -> None:
    evaluator = TopicConditionedQualityEvaluator.from_params({"granularity": "normal"})
    benchmark_spec = _grouped_pov_benchmark_spec()
    request = EvaluationRequest(
        predictions=benchmark_spec.reference_by_id[
            "intent_label_fine__fine"
        ].to_loaded_clustering(),
        artifacts={
            "benchmark_spec": benchmark_spec,
            "selected_topic": {
                "topic_id": "intent_label_normal",
                "label": "Intent (Normal)",
                "guidance": "Group tickets by the normal intent POV.",
            },
        },
        metadata={"config_slug": "cfg", "run_id": "run-1"},
    )

    result = evaluator.evaluate(request)

    assert result.details["topic_id"] == "intent_label_normal"
    assert result.details["reference_id"] == "intent_label_normal__normal"
    assert result.details["requested_granularity"] == "normal"
    assert result.metrics["cond_ari"] < 1.0


def test_topic_conditioned_quality_evaluator_prefers_normal_on_same_pov_tie_when_unspecified() -> (
    None
):
    benchmark_spec = parse_synthetic_benchmark_spec(
        {
            "dataset_name": "enterprise_ticket",
            "items": ["okta locked", "badge broken", "invoice overdue", "vpn down"],
            "themes": [
                {
                    "theme_id": "intent_label_normal",
                    "label": "Intent (Normal)",
                    "guidance": "Group tickets by the primary request type or problem category.",
                    "pov_group_id": "intent",
                    "granularity": "normal",
                },
                {
                    "theme_id": "intent_label_fine",
                    "label": "Intent (Fine)",
                    "guidance": "Group tickets by the most specific request or issue subtype.",
                    "pov_group_id": "intent",
                    "granularity": "fine",
                },
            ],
            "references": [
                {
                    "reference_id": "intent_label_normal__normal",
                    "topic_id": "intent_label_normal",
                    "granularity": "normal",
                    "pov_group_id": "intent",
                    "granularity_level": "normal",
                    "assignment": {
                        "okta locked": 0,
                        "badge broken": 0,
                        "invoice overdue": 1,
                        "vpn down": 2,
                    },
                    "cluster_names": ["access_identity", "finance_procurement", "network_ops"],
                },
                {
                    "reference_id": "intent_label_fine__fine",
                    "topic_id": "intent_label_fine",
                    "granularity": "fine",
                    "pov_group_id": "intent",
                    "granularity_level": "fine",
                    "assignment": {
                        "okta locked": 0,
                        "badge broken": 0,
                        "invoice overdue": 1,
                        "vpn down": 2,
                    },
                    "cluster_names": ["access_identity", "finance_procurement", "network_ops"],
                },
            ],
        }
    )
    evaluator = TopicConditionedQualityEvaluator.from_params()
    request = EvaluationRequest(
        predictions=benchmark_spec.reference_by_id[
            "intent_label_fine__fine"
        ].to_loaded_clustering(),
        artifacts={
            "benchmark_spec": benchmark_spec,
            "selected_topic": {
                "topic_id": "intent_label_fine",
                "label": "Intent (Fine)",
                "guidance": "Group tickets by the most specific request or issue subtype.",
            },
        },
        metadata={"config_slug": "cfg", "run_id": "run-1"},
    )

    result = evaluator.evaluate(request)

    assert result.details["reference_id"] == "intent_label_normal__normal"
    assert result.details["granularity"] == "normal"
    assert result.details["requested_granularity"] is None


def test_topic_conditioned_quality_evaluator_prefers_requested_granularity_on_same_pov_tie() -> (
    None
):
    benchmark_spec = parse_synthetic_benchmark_spec(
        {
            "dataset_name": "enterprise_ticket",
            "items": ["okta locked", "badge broken", "invoice overdue", "vpn down"],
            "themes": [
                {
                    "theme_id": "intent_label_normal",
                    "label": "Intent (Normal)",
                    "guidance": "Group tickets by the primary request type or problem category.",
                    "pov_group_id": "intent",
                    "granularity": "normal",
                },
                {
                    "theme_id": "intent_label_fine",
                    "label": "Intent (Fine)",
                    "guidance": "Group tickets by the most specific request or issue subtype.",
                    "pov_group_id": "intent",
                    "granularity": "fine",
                },
            ],
            "references": [
                {
                    "reference_id": "intent_label_normal__normal",
                    "topic_id": "intent_label_normal",
                    "granularity": "normal",
                    "pov_group_id": "intent",
                    "granularity_level": "normal",
                    "assignment": {
                        "okta locked": 0,
                        "badge broken": 0,
                        "invoice overdue": 1,
                        "vpn down": 2,
                    },
                    "cluster_names": ["access_identity", "finance_procurement", "network_ops"],
                },
                {
                    "reference_id": "intent_label_fine__fine",
                    "topic_id": "intent_label_fine",
                    "granularity": "fine",
                    "pov_group_id": "intent",
                    "granularity_level": "fine",
                    "assignment": {
                        "okta locked": 0,
                        "badge broken": 0,
                        "invoice overdue": 1,
                        "vpn down": 2,
                    },
                    "cluster_names": ["access_identity", "finance_procurement", "network_ops"],
                },
            ],
        }
    )
    evaluator = TopicConditionedQualityEvaluator.from_params({"granularity": "fine"})
    request = EvaluationRequest(
        predictions=benchmark_spec.reference_by_id[
            "intent_label_normal__normal"
        ].to_loaded_clustering(),
        artifacts={
            "benchmark_spec": benchmark_spec,
            "selected_topic": {
                "topic_id": "intent_label_normal",
                "label": "Intent (Normal)",
                "guidance": "Group tickets by the primary request type or problem category.",
            },
        },
        metadata={"config_slug": "cfg", "run_id": "run-1"},
    )

    result = evaluator.evaluate(request)

    assert result.details["reference_id"] == "intent_label_fine__fine"
    assert result.details["granularity"] == "fine"
    assert result.details["requested_granularity"] == "fine"


def test_topic_conditioned_quality_evaluator_uses_semantic_theme_match_for_guided_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import topic_conditioned_quality as module

    def mock_match(**_: object) -> ThemeMatchJudgment:
        return ThemeMatchJudgment(
            supported=True,
            benchmark_theme_id="department",
            confidence=0.95,
            reasoning="Department is the intended theme.",
        )

    monkeypatch.setattr(module, "judge_theme_match", mock_match)

    evaluator = TopicConditionedQualityEvaluator.from_params({"granularity": "normal"})
    benchmark_spec = _enterprise_ticket_benchmark_spec()
    request = EvaluationRequest(
        predictions=benchmark_spec.reference_by_id["department__normal"].to_loaded_clustering(),
        artifacts={
            "benchmark_spec": benchmark_spec,
            "batch_run_config": {
                "dataset_name": "enterprise_ticket",
                "human_agent_config": {"guidance": "Sort tickets into departments."},
            },
        },
        metadata={"config_slug": "cfg", "run_id": "run-1"},
    )

    result = evaluator.evaluate(request)

    assert result.details["topic_id"] == "department"
    assert result.details["reference_id"] == "department__normal"
    assert result.details["topic_resolution_mode"] == "semantic_judge"
    assert result.metrics["cond_ari"] == pytest.approx(1.0)


def test_topic_conditioned_quality_evaluator_requires_theme_resolution_for_guided_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import topic_conditioned_quality as module

    def mock_match(**_: object) -> ThemeMatchJudgment:
        return ThemeMatchJudgment(
            supported=False,
            benchmark_theme_id=None,
            confidence=0.2,
            reasoning="No supported benchmark theme.",
        )

    monkeypatch.setattr(module, "judge_theme_match", mock_match)

    evaluator = TopicConditionedQualityEvaluator.from_params({"granularity": "normal"})
    benchmark_spec = _enterprise_ticket_benchmark_spec()
    request = EvaluationRequest(
        predictions=benchmark_spec.reference_by_id["department__normal"].to_loaded_clustering(),
        artifacts={
            "benchmark_spec": benchmark_spec,
            "batch_run_config": {
                "dataset_name": "enterprise_ticket",
                "human_agent_config": {"guidance": "Sort tickets into departments."},
            },
        },
        metadata={"config_slug": "cfg", "run_id": "run-1"},
    )

    with pytest.raises(ValueError, match="could not resolve the requested benchmark theme"):
        evaluator.evaluate(request)


def test_topic_conditioned_quality_evaluator_uses_explicit_theme_id_when_provided() -> None:
    evaluator = TopicConditionedQualityEvaluator.from_params({"granularity": "normal"})
    benchmark_spec = _benchmark_spec()
    request = EvaluationRequest(
        predictions=benchmark_spec.reference_by_id["vehicle__normal"].to_loaded_clustering(),
        artifacts={
            "benchmark_spec": benchmark_spec,
            "batch_run_config": {
                "dataset_name": "synthetic",
                "metadata": {"benchmark_theme_id": "vehicle"},
                "human_agent_config": {"guidance": "Sort items however makes sense."},
            },
        },
        metadata={"config_slug": "cfg", "run_id": "run-1"},
    )

    result = evaluator.evaluate(request)

    assert result.details["topic_id"] == "vehicle"
    assert result.details["reference_id"] == "vehicle__normal"
    assert result.details["topic_resolution_mode"] == "explicit_or_lexical"
    assert result.metrics["cond_ari"] == pytest.approx(1.0)


def test_topic_conditioned_quality_evaluator_uses_semantic_theme_match_for_auto_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import topic_conditioned_quality as module

    def mock_match(**_: object) -> ThemeMatchJudgment:
        return ThemeMatchJudgment(
            supported=True,
            benchmark_theme_id="issue_type",
            confidence=0.9,
            reasoning="This discovered topic is really issue type.",
        )

    monkeypatch.setattr(module, "judge_theme_match", mock_match)

    benchmark_spec = _enterprise_ticket_benchmark_spec()
    evaluator = TopicConditionedQualityEvaluator.from_params({"granularity": "normal"})
    request = EvaluationRequest(
        predictions=benchmark_spec.reference_by_id["issue_type__normal"].to_loaded_clustering(),
        artifacts={
            "benchmark_spec": benchmark_spec,
            "selected_topic": {
                "topic_id": "access-and-authentication-issues",
                "label": "Access and Authentication Issues",
                "guidance": "Group tickets by access and authentication issues.",
                "generated_at_turn": 2,
            },
            "batch_run_config": {
                "dataset_name": "enterprise_ticket",
                "human_agent_config": {"guidance": None},
                "metadata": {
                    "topic_branch_mode": "auto_topic_discovery",
                    "topic_branch_topic_id": "access-and-authentication-issues",
                    "topic_branch_topic_label": "Access and Authentication Issues",
                    "topic_branch_topic_guidance": (
                        "Group tickets by access and authentication issues."
                    ),
                },
            },
        },
        metadata={"config_slug": "cfg", "run_id": "run-1"},
    )

    result = evaluator.evaluate(request)

    assert result.details["topic_id"] == "issue_type"
    assert result.details["original_topic_id"] == "access-and-authentication-issues"
    assert result.details["original_topic_label"] == "Access and Authentication Issues"
    assert (
        result.details["original_topic_guidance"]
        == "Group tickets by access and authentication issues."
    )
    assert result.details["resolved_theme_id"] == "issue_type"
    assert result.details["resolved_theme_label"] == "Issue Type"
    assert result.details["reference_id"] == "issue_type__normal"
    assert result.details["topic_resolution_mode"] == "semantic_judge"
    assert result.metrics["cond_ari"] == pytest.approx(1.0)


def test_topic_conditioned_quality_evaluator_uses_best_reference_fallback_for_auto_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import topic_conditioned_quality as module

    def mock_match(**_: object) -> ThemeMatchJudgment:
        return ThemeMatchJudgment(
            supported=False,
            benchmark_theme_id=None,
            confidence=0.2,
            reasoning="No semantic benchmark match.",
        )

    monkeypatch.setattr(module, "judge_theme_match", mock_match)

    benchmark_spec = _enterprise_ticket_benchmark_spec()
    evaluator = TopicConditionedQualityEvaluator.from_params({"granularity": "normal"})
    request = EvaluationRequest(
        predictions=benchmark_spec.reference_by_id["department__normal"].to_loaded_clustering(),
        artifacts={
            "benchmark_spec": benchmark_spec,
            "selected_topic": {
                "topic_id": "access-and-authentication-issues",
                "label": "Access and Authentication Issues",
                "guidance": "Group tickets by access and authentication issues.",
                "generated_at_turn": 2,
            },
            "batch_run_config": {
                "dataset_name": "enterprise_ticket",
                "human_agent_config": {"guidance": None},
                "metadata": {
                    "topic_branch_mode": "auto_topic_discovery",
                    "topic_branch_topic_id": "access-and-authentication-issues",
                    "topic_branch_topic_label": "Access and Authentication Issues",
                    "topic_branch_topic_guidance": (
                        "Group tickets by access and authentication issues."
                    ),
                },
            },
        },
        metadata={"config_slug": "cfg", "run_id": "run-1"},
    )

    result = evaluator.evaluate(request)

    assert result.details["topic_id"] == "department"
    assert result.details["reference_id"] == "department__normal"
    assert result.details["topic_resolution_mode"] == "best_reference_fallback"
    assert result.metrics["cond_ari"] == pytest.approx(1.0)
    assert result.metrics["cond_nmi"] == pytest.approx(1.0)
    assert result.metrics["cond_vi"] == pytest.approx(0.0)


def test_topic_conditioned_quality_preserves_resolved_theme_mismatch_for_auto_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import topic_conditioned_quality as module

    def mock_match(**_: object) -> ThemeMatchJudgment:
        return ThemeMatchJudgment(
            supported=True,
            benchmark_theme_id="issue_type",
            confidence=0.9,
            reasoning="This discovered topic is really issue type.",
        )

    monkeypatch.setattr(module, "judge_theme_match", mock_match)

    benchmark_spec = _enterprise_ticket_benchmark_spec()
    evaluator = TopicConditionedQualityEvaluator.from_params({"granularity": "normal"})
    request = EvaluationRequest(
        predictions=LoadedClustering(
            assignment={"orange": 0, "pear": 0},
            cluster_names=["mismatch"],
        ),
        artifacts={
            "benchmark_spec": benchmark_spec,
            "selected_topic": {
                "topic_id": "access-and-authentication-issues",
                "label": "Access and Authentication Issues",
                "guidance": "Group tickets by access and authentication issues.",
                "generated_at_turn": 2,
            },
            "batch_run_config": {
                "dataset_name": "enterprise_ticket",
                "human_agent_config": {"guidance": None},
                "metadata": {
                    "topic_branch_mode": "auto_topic_discovery",
                    "topic_branch_topic_id": "access-and-authentication-issues",
                    "topic_branch_topic_label": "Access and Authentication Issues",
                    "topic_branch_topic_guidance": (
                        "Group tickets by access and authentication issues."
                    ),
                },
            },
        },
        metadata={"config_slug": "cfg", "run_id": "run-1"},
    )

    with pytest.raises(ValueError, match="no overlapping items"):
        evaluator.evaluate(request)


def test_topic_conditioned_quality_best_reference_fallback_excludes_foil_themes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import topic_conditioned_quality as module

    def mock_match(**_: object) -> ThemeMatchJudgment:
        return ThemeMatchJudgment(
            supported=False,
            benchmark_theme_id=None,
            confidence=0.2,
            reasoning="No supported benchmark theme.",
        )

    monkeypatch.setattr(module, "judge_theme_match", mock_match)

    benchmark_spec = _benchmark_spec()
    evaluator = TopicConditionedQualityEvaluator.from_params({"granularity": "normal"})
    request = EvaluationRequest(
        predictions=LoadedClustering(
            assignment={"apple": 1, "banana": 1, "car": 0, "truck": 0},
            cluster_names=["other", "foil_like"],
        ),
        artifacts={
            "benchmark_spec": benchmark_spec,
            "selected_topic": {
                "topic_id": "mystic-vibes",
                "label": "Mystic Vibes",
                "guidance": "Group items by cosmic intuition.",
                "generated_at_turn": 2,
            },
            "batch_run_config": {
                "dataset_name": "synthetic",
                "human_agent_config": {"guidance": None},
                "metadata": {
                    "topic_branch_mode": "auto_topic_discovery",
                    "topic_branch_topic_id": "mystic-vibes",
                    "topic_branch_topic_label": "Mystic Vibes",
                    "topic_branch_topic_guidance": "Group items by cosmic intuition.",
                },
            },
        },
        metadata={"config_slug": "cfg", "run_id": "run-1"},
    )

    result = evaluator.evaluate(request)

    assert result.details["topic_id"] != "astrology"
    assert result.details["topic_resolution_mode"] == "best_reference_fallback"


def test_topic_conditioned_quality_evaluator_rejects_disjoint_item_sets() -> None:
    evaluator = TopicConditionedQualityEvaluator.from_params({"granularity": "normal"})
    request = _request()
    request = EvaluationRequest(
        predictions=LoadedClustering(
            assignment={"orange": 0, "pear": 0},
            cluster_names=["fruit"],
        ),
        artifacts=request.artifacts,
        metadata=request.metadata,
    )

    with pytest.raises(ValueError, match="no overlapping items"):
        evaluator.evaluate(request)


def test_efficiency_utility_evaluator_reports_structural_and_semantic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import cluster_semantic_quality as module

    coherence_scores = iter([6.0, 9.0, 6.0, 9.0])

    def mock_coherence(**_: object) -> ClusterCoherenceJudgment:
        return ClusterCoherenceJudgment(
            score=next(coherence_scores),
            supported_by_guidance=True,
            reasoning="Coherent cluster.",
        )

    def mock_distinctiveness(**_: object) -> ClusterDistinctivenessJudgment:
        return ClusterDistinctivenessJudgment(
            score=8.0,
            overlapping_or_redundant=False,
            reasoning="Distinct clusters.",
        )

    monkeypatch.setattr(module, "judge_cluster_coherence", mock_coherence)
    monkeypatch.setattr(module, "judge_cluster_distinctiveness", mock_distinctiveness)

    evaluator = EfficiencyUtilityEvaluator.from_params({"include_semantic_utility": True})
    result = evaluator.evaluate(_request())

    assert result.metrics["best_score_seen_structural"] == pytest.approx(1.0)
    assert "structural_threshold_reached_rate" not in result.metrics
    assert result.metrics["best_score_seen_semantic"] >= 0.6
    assert "area_under_utility_curve_semantic" in result.metrics
    assert "semantic_threshold_reached_rate" not in result.metrics
    assert "threshold" not in result.details


def test_efficiency_utility_evaluator_emits_threshold_metrics_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import cluster_semantic_quality as module

    def mock_coherence(**_: object) -> ClusterCoherenceJudgment:
        return ClusterCoherenceJudgment(
            score=6.0,
            supported_by_guidance=True,
            reasoning="Coherent cluster.",
        )

    def mock_distinctiveness(**_: object) -> ClusterDistinctivenessJudgment:
        return ClusterDistinctivenessJudgment(
            score=8.0,
            overlapping_or_redundant=False,
            reasoning="Distinct clusters.",
        )

    monkeypatch.setattr(module, "judge_cluster_coherence", mock_coherence)
    monkeypatch.setattr(module, "judge_cluster_distinctiveness", mock_distinctiveness)

    evaluator = EfficiencyUtilityEvaluator.from_params(
        {"include_semantic_utility": True, "threshold": 0.8}
    )
    result = evaluator.evaluate(_request())

    assert result.metrics["structural_threshold_reached_rate"] == pytest.approx(1.0)
    assert result.metrics["semantic_threshold_reached_rate"] == pytest.approx(0.0)
    assert result.details["threshold"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("params", "expected_error"),
    [
        (
            {
                "include_semantic_utility": True,
                "coherence_prompt_id": "does_not_exist",
            },
            "Invalid coherence_prompt_id",
        ),
        (
            {
                "include_semantic_utility": True,
                "distinctiveness_prompt_id": "does_not_exist",
            },
            "Invalid distinctiveness_prompt_id",
        ),
        (
            {
                "include_semantic_utility": True,
                "semantic_snapshot_mode": "does_not_exist",
            },
            "'semantic_snapshot_mode' must be one of",
        ),
    ],
)
def test_efficiency_utility_evaluator_rejects_invalid_semantic_prompt_ids(
    params: dict[str, object], expected_error: str
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        EfficiencyUtilityEvaluator.from_params(params)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf")])
def test_efficiency_utility_evaluator_rejects_non_finite_thresholds(threshold: float) -> None:
    with pytest.raises(ValueError, match="finite number between -1.0 and 1.0"):
        EfficiencyUtilityEvaluator.from_params({"threshold": threshold})


def test_inferred_theme_efficiency_utility_evaluator_rejects_invalid_theme_match_prompt_id() -> (
    None
):
    with pytest.raises(ValueError, match="Invalid theme_match_prompt_id"):
        InferredThemeEfficiencyUtilityEvaluator.from_params(
            {"theme_match_prompt_id": "does_not_exist"}
        )


def test_inferred_theme_efficiency_utility_evaluator_rejects_invalid_semantic_snapshot_mode() -> (
    None
):
    with pytest.raises(ValueError, match="'semantic_snapshot_mode' must be one of"):
        InferredThemeEfficiencyUtilityEvaluator.from_params(
            {"semantic_snapshot_mode": "does_not_exist"}
        )


def test_efficiency_utility_semantic_utility_uses_resolved_theme_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import cluster_semantic_quality as module

    coherence_guidances: list[str] = []
    distinctiveness_guidances: list[str] = []

    def mock_coherence(**kwargs: object) -> ClusterCoherenceJudgment:
        coherence_guidances.append(str(kwargs["guidance"]))
        return ClusterCoherenceJudgment(
            score=7.0,
            supported_by_guidance=True,
            reasoning="Coherent cluster.",
        )

    def mock_distinctiveness(**kwargs: object) -> ClusterDistinctivenessJudgment:
        distinctiveness_guidances.append(str(kwargs["guidance"]))
        return ClusterDistinctivenessJudgment(
            score=8.0,
            overlapping_or_redundant=False,
            reasoning="Distinct clusters.",
        )

    monkeypatch.setattr(module, "judge_cluster_coherence", mock_coherence)
    monkeypatch.setattr(module, "judge_cluster_distinctiveness", mock_distinctiveness)

    benchmark_spec = _benchmark_spec()
    evaluator = EfficiencyUtilityEvaluator.from_params({"include_semantic_utility": True})
    result = evaluator.evaluate(
        EvaluationRequest(
            predictions=benchmark_spec.reference_by_id["vehicle__normal"].to_loaded_clustering(),
            artifacts={
                "benchmark_spec": benchmark_spec,
                "selected_topic": {
                    "topic_id": "vehicle",
                    "label": "Vehicle",
                    "guidance": "Stale topic guidance that should be ignored.",
                },
                "batch_run_config": {
                    "dataset_name": "synthetic",
                    "metadata": {"benchmark_theme_id": "vehicle"},
                    "human_agent_config": {"guidance": "Also stale batch guidance."},
                },
                "conversation_trace": {
                    "messages": [
                        {"role": "human", "content": "First try."},
                        {"role": "assistant", "content": "Here is a clustering."},
                        {"role": "human", "content": "Finalize it."},
                    ]
                },
                "judge_clustering_trace": {
                    "clusterings": [
                        {
                            "turn_index": 1,
                            "clustering": {
                                "clusters": [
                                    {"name": "other", "members": ["apple", "banana"]},
                                    {"name": "vehicle", "members": ["car", "truck"]},
                                ]
                            },
                        }
                    ]
                },
            },
            metadata={"config_slug": "cfg", "run_id": "run-1"},
        )
    )

    assert result.metrics["best_score_seen_semantic"] == pytest.approx(0.7)
    assert coherence_guidances == [
        "Group items by vehicle type.",
        "Group items by vehicle type.",
        "Group items by vehicle type.",
        "Group items by vehicle type.",
    ]
    assert distinctiveness_guidances == [
        "Group items by vehicle type.",
        "Group items by vehicle type.",
    ]


def test_efficiency_utility_evaluator_resolves_active_theme_from_guidance_only() -> None:
    benchmark_spec = _benchmark_spec()
    evaluator = EfficiencyUtilityEvaluator.from_params()

    result = evaluator.evaluate(
        EvaluationRequest(
            predictions=benchmark_spec.reference_by_id["vehicle__normal"].to_loaded_clustering(),
            artifacts={
                "benchmark_spec": benchmark_spec,
                "batch_run_config": {
                    "dataset_name": "synthetic",
                    "human_agent_config": {"guidance": "Group items by vehicle type."},
                },
                "conversation_trace": {"messages": []},
                "judge_clustering_trace": {
                    "clusterings": [
                        {
                            "turn_index": 1,
                            "clustering": _clustering_payload_from_loaded(
                                benchmark_spec.reference_by_id[
                                    "vehicle__normal"
                                ].to_loaded_clustering()
                            ),
                        }
                    ]
                },
            },
            metadata={"config_slug": "cfg", "run_id": "run-1"},
        )
    )

    assert result.details["topic_id"] == "vehicle"
    assert result.details["reference_id"] == "vehicle__normal"


def test_efficiency_utility_evaluator_final_only_limits_semantic_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import cluster_semantic_quality as module

    coherence_calls: list[str] = []
    distinctiveness_calls: list[tuple[str, str]] = []

    def mock_coherence(**kwargs: object) -> ClusterCoherenceJudgment:
        coherence_calls.append(str(kwargs["cluster_name"]))
        return ClusterCoherenceJudgment(
            score=7.0,
            supported_by_guidance=True,
            reasoning="Coherent cluster.",
        )

    def mock_distinctiveness(**kwargs: object) -> ClusterDistinctivenessJudgment:
        distinctiveness_calls.append(
            (str(kwargs["left_cluster_name"]), str(kwargs["right_cluster_name"]))
        )
        return ClusterDistinctivenessJudgment(
            score=8.0,
            overlapping_or_redundant=False,
            reasoning="Distinct clusters.",
        )

    monkeypatch.setattr(module, "judge_cluster_coherence", mock_coherence)
    monkeypatch.setattr(module, "judge_cluster_distinctiveness", mock_distinctiveness)

    benchmark_spec = _benchmark_spec()
    evaluator = EfficiencyUtilityEvaluator.from_params(
        {
            "include_semantic_utility": True,
            "semantic_snapshot_mode": "final_only",
        }
    )
    evaluator.evaluate(
        EvaluationRequest(
            predictions=LoadedClustering(
                assignment={"apple": 0, "banana": 0, "car": 1, "truck": 1},
                cluster_names=["fruit", "vehicle"],
            ),
            artifacts={
                "benchmark_spec": benchmark_spec,
                "selected_topic": {
                    "topic_id": "vehicle",
                    "label": "Vehicle",
                    "guidance": "Stale topic guidance that should be ignored.",
                },
                "batch_run_config": {
                    "dataset_name": "synthetic",
                    "metadata": {"benchmark_theme_id": "vehicle"},
                    "human_agent_config": {"guidance": "Also stale batch guidance."},
                },
                "conversation_trace": {
                    "messages": [
                        {"role": "human", "content": "First try."},
                        {"role": "assistant", "content": "Here is a clustering."},
                        {"role": "human", "content": "Finalize it."},
                    ]
                },
                "judge_clustering_trace": {
                    "clusterings": [
                        {
                            "turn_index": 1,
                            "clustering": {
                                "clusters": [
                                    {"name": "other", "members": ["apple", "banana"]},
                                    {"name": "vehicle", "members": ["car", "truck"]},
                                ]
                            },
                        }
                    ]
                },
            },
            metadata={"config_slug": "cfg", "run_id": "run-1"},
        )
    )

    assert coherence_calls == ["fruit", "vehicle"]
    assert distinctiveness_calls == [("fruit", "vehicle")]


def test_efficiency_utility_evaluator_allows_grouped_theme_resolve_normal_sibling_reference() -> (
    None
):
    benchmark_spec = _grouped_pov_benchmark_spec()
    evaluator = EfficiencyUtilityEvaluator.from_params({"granularity": "normal"})

    result = evaluator.evaluate(
        EvaluationRequest(
            predictions=benchmark_spec.reference_by_id[
                "intent_label_normal__normal"
            ].to_loaded_clustering(),
            artifacts={
                "benchmark_spec": benchmark_spec,
                "selected_topic": {
                    "topic_id": "intent_label_fine",
                    "label": "Intent (Fine)",
                    "guidance": (
                        "Group tickets by the most specific request or issue subtype, "
                        "separating distinct problems such as MFA delivery failure, "
                        "badge malfunction, invoice mismatch, or VPN outage."
                    ),
                },
                "conversation_trace": {"messages": []},
                "judge_clustering_trace": {
                    "clusterings": [
                        {
                            "turn_index": 1,
                            "clustering": _clustering_payload_from_loaded(
                                benchmark_spec.reference_by_id[
                                    "intent_label_normal__normal"
                                ].to_loaded_clustering()
                            ),
                        }
                    ]
                },
            },
            metadata={"config_slug": "cfg", "run_id": "run-1"},
        )
    )

    assert result.details["topic_id"] == "intent_label_fine"
    assert result.details["reference_id"] == "intent_label_normal__normal"
    assert result.details["granularity"] == "normal"
    assert result.details["requested_granularity"] == "normal"


def test_efficiency_utility_evaluator_best_matches_within_same_pov_when_unspecified() -> None:
    benchmark_spec = _grouped_pov_benchmark_spec()
    evaluator = EfficiencyUtilityEvaluator.from_params()

    result = evaluator.evaluate(
        EvaluationRequest(
            predictions=benchmark_spec.reference_by_id[
                "intent_label_fine__fine"
            ].to_loaded_clustering(),
            artifacts={
                "benchmark_spec": benchmark_spec,
                "selected_topic": {
                    "topic_id": "intent_label_fine",
                    "label": "Intent (Fine)",
                    "guidance": (
                        "Group tickets by the most specific request or issue subtype, "
                        "separating distinct problems such as MFA delivery failure, "
                        "badge malfunction, invoice mismatch, or VPN outage."
                    ),
                },
                "conversation_trace": {"messages": []},
                "judge_clustering_trace": {
                    "clusterings": [
                        {
                            "turn_index": 1,
                            "clustering": _clustering_payload_from_loaded(
                                benchmark_spec.reference_by_id[
                                    "intent_label_normal__normal"
                                ].to_loaded_clustering()
                            ),
                        }
                    ]
                },
            },
            metadata={"config_slug": "cfg", "run_id": "run-1"},
        )
    )

    assert result.details["topic_id"] == "intent_label_fine"
    assert result.details["reference_id"] == "intent_label_fine__fine"
    assert result.details["granularity"] == "fine"
    assert result.details["requested_granularity"] is None


def test_efficiency_utility_evaluator_final_only_uses_final_clustering_without_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import cluster_semantic_quality as module

    coherence_calls: list[str] = []

    def mock_coherence(**kwargs: object) -> ClusterCoherenceJudgment:
        cluster_name = str(kwargs["cluster_name"])
        coherence_calls.append(cluster_name)
        score = 9.0 if cluster_name == "fruit" else 5.0
        return ClusterCoherenceJudgment(
            score=score,
            supported_by_guidance=True,
            reasoning="Coherent cluster.",
        )

    def mock_distinctiveness(**_: object) -> ClusterDistinctivenessJudgment:
        return ClusterDistinctivenessJudgment(
            score=8.0,
            overlapping_or_redundant=False,
            reasoning="Distinct clusters.",
        )

    monkeypatch.setattr(module, "judge_cluster_coherence", mock_coherence)
    monkeypatch.setattr(module, "judge_cluster_distinctiveness", mock_distinctiveness)

    benchmark_spec = _benchmark_spec()
    evaluator = EfficiencyUtilityEvaluator.from_params(
        {
            "include_semantic_utility": True,
            "semantic_snapshot_mode": "final_only",
        }
    )
    result = evaluator.evaluate(
        EvaluationRequest(
            predictions=LoadedClustering(
                assignment={"apple": 0, "banana": 0, "car": 1, "truck": 1},
                cluster_names=["fruit", "vehicle"],
            ),
            artifacts={
                "benchmark_spec": benchmark_spec,
                "selected_topic": {
                    "topic_id": "vehicle",
                    "label": "Vehicle",
                    "guidance": "Group items by vehicle type.",
                },
                "judge_clustering_trace": {
                    "clusterings": [
                        {
                            "turn_index": 1,
                            "clustering": {
                                "clusters": [
                                    {"name": "other", "members": ["apple", "banana"]},
                                    {"name": "vehicle", "members": ["car", "truck"]},
                                ]
                            },
                        }
                    ]
                },
            },
            metadata={"config_slug": "cfg", "run_id": "run-1"},
        )
    )

    assert coherence_calls == ["fruit", "vehicle"]
    assert result.metrics["best_score_seen_semantic"] == pytest.approx(0.7)
    assert result.details["best_so_far_semantic_by_turn"] == [
        {
            "turn_index": 2,
            "current_score": pytest.approx(0.7),
            "best_score": pytest.approx(0.7),
        }
    ]
    assert -1.0 <= result.metrics["area_under_utility_curve_structural"] <= 1.0


def test_efficiency_utility_evaluator_preserves_negative_structural_scores() -> None:
    evaluator = EfficiencyUtilityEvaluator.from_params({"threshold": -0.25})
    result = evaluator.evaluate(
        EvaluationRequest(
            predictions=LoadedClustering(
                assignment={"apple": 0, "banana": 1, "car": 0, "truck": 1},
                cluster_names=["mixed_a", "mixed_b"],
            ),
            artifacts={
                "benchmark_spec": _benchmark_spec(),
                "selected_topic": {
                    "topic_id": "fruit",
                    "label": "Fruit",
                    "guidance": "Group items by fruitness.",
                },
                "conversation_trace": {
                    "messages": [
                        {"role": "human", "content": "Propose a split."},
                        {"role": "assistant", "content": "Here is one."},
                        {"role": "human", "content": "Try again."},
                    ]
                },
                "judge_clustering_trace": {
                    "clusterings": [
                        {
                            "turn_index": 1,
                            "clustering": {
                                "clusters": [
                                    {"name": "mixed_a", "members": ["apple", "car"]},
                                    {"name": "mixed_b", "members": ["banana", "truck"]},
                                ]
                            },
                        }
                    ]
                },
            },
            metadata={"config_slug": "cfg", "run_id": "run-1"},
        )
    )

    assert result.metrics["best_score_seen_structural"] == pytest.approx(-0.5)
    assert result.metrics["area_under_utility_curve_structural"] == pytest.approx(-0.5)
    assert result.metrics["structural_threshold_reached_rate"] == pytest.approx(0.0)
    assert result.details["structural_threshold_reached"] is False
    assert "turn_to_threshold_structural" not in result.metrics
    structural_by_turn = result.details["best_so_far_structural_by_turn"]
    assert [point["turn_index"] for point in structural_by_turn] == [1, 2]
    assert [point["current_score"] for point in structural_by_turn] == pytest.approx([-0.5, -0.5])
    assert [point["best_score"] for point in structural_by_turn] == pytest.approx([-0.5, -0.5])


def test_inferred_theme_efficiency_utility_uses_best_reference_fallback_for_discovery_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import topic_conditioned_quality as module

    benchmark_spec = _enterprise_ticket_benchmark_spec()
    evaluator = InferredThemeEfficiencyUtilityEvaluator.from_params()

    def mock_match(**_: object) -> ThemeMatchJudgment:
        return ThemeMatchJudgment(
            supported=False,
            benchmark_theme_id=None,
            confidence=0.2,
            reasoning="No semantic benchmark match.",
        )

    monkeypatch.setattr(module, "judge_theme_match", mock_match)

    result = evaluator.evaluate(
        EvaluationRequest(
            predictions=benchmark_spec.reference_by_id["department__normal"].to_loaded_clustering(),
            artifacts={
                "benchmark_spec": benchmark_spec,
                "selected_topic": {
                    "topic_id": "support-domain",
                    "label": "Support Domain",
                    "guidance": "Group tickets by support domain.",
                    "generated_at_turn": 3,
                },
                "batch_run_config": {
                    "dataset_name": "enterprise_ticket",
                    "human_agent_config": {
                        "guidance": None,
                        "topic_discovery": {
                            "enabled": True,
                            "trigger_mode": "turn_budget",
                            "turn_budget": 3,
                            "max_rounds": 2,
                            "min_new_topics_per_round": 3,
                        },
                    },
                    "metadata": {
                        "topic_branch_mode": "auto_topic_discovery",
                        "topic_branch_topic_id": "support-domain",
                        "topic_branch_topic_label": "Support Domain",
                        "topic_branch_topic_guidance": "Group tickets by support domain.",
                    },
                },
                "conversation_trace": {
                    "messages": [
                        {"role": "human", "content": "Summarize the dataset."},
                        {"role": "assistant", "content": "Here is the summary."},
                        {"role": "human", "content": "Save a clustering."},
                    ]
                },
                "judge_clustering_trace": {
                    "clusterings": [
                        {
                            "turn_index": 1,
                            "clustering": {
                                "clusters": [
                                    {"name": "IT", "members": ["okta locked", "vpn down"]},
                                    {"name": "Facilities", "members": ["badge broken"]},
                                    {"name": "Finance", "members": ["invoice overdue"]},
                                ]
                            },
                        }
                    ]
                },
            },
            metadata={"config_slug": "cfg", "run_id": "run-1"},
        )
    )

    assert result.details["topic_id"] == "department"
    assert result.details["original_topic_id"] == "support-domain"
    assert result.details["original_topic_label"] == "Support Domain"
    assert result.details["original_topic_guidance"] == "Group tickets by support domain."
    assert result.details["resolved_theme_id"] == "department"
    assert result.details["resolved_theme_label"] == "Department"
    assert result.details["reference_id"] == "department__normal"
    assert result.details["topic_resolution_mode"] == "best_reference_fallback"
    assert result.metrics["best_score_seen_structural"] == pytest.approx(1.0)


def test_inferred_theme_efficiency_utility_preserves_selected_fallback_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import topic_conditioned_quality as module

    def mock_match(**_: object) -> ThemeMatchJudgment:
        return ThemeMatchJudgment(
            supported=False,
            benchmark_theme_id=None,
            confidence=0.2,
            reasoning="No semantic benchmark match.",
        )

    monkeypatch.setattr(module, "judge_theme_match", mock_match)

    benchmark_spec = parse_synthetic_benchmark_spec(
        {
            "dataset_name": "enterprise_ticket",
            "items": ["okta locked", "badge broken", "invoice overdue", "vpn down"],
            "themes": [
                {
                    "theme_id": "department",
                    "label": "Department",
                    "guidance": "Group tickets by department.",
                }
            ],
            "references": [
                {
                    "reference_id": "department__normal_a",
                    "topic_id": "department",
                    "granularity": "normal",
                    "assignment": {
                        "okta locked": 0,
                        "badge broken": 0,
                        "invoice overdue": 0,
                        "vpn down": 0,
                    },
                    "cluster_names": ["all"],
                },
                {
                    "reference_id": "department__normal_b",
                    "topic_id": "department",
                    "granularity": "normal",
                    "assignment": {
                        "okta locked": 1,
                        "badge broken": 1,
                        "invoice overdue": 2,
                        "vpn down": 1,
                    },
                    "cluster_names": ["Unused", "IT", "Finance"],
                },
            ],
        }
    )

    evaluator = InferredThemeEfficiencyUtilityEvaluator.from_params({"granularity": "normal"})
    result = evaluator.evaluate(
        EvaluationRequest(
            predictions=benchmark_spec.reference_by_id[
                "department__normal_b"
            ].to_loaded_clustering(),
            artifacts={
                "benchmark_spec": benchmark_spec,
                "selected_topic": {
                    "topic_id": "support-domain",
                    "label": "Support Domain",
                    "guidance": "Group tickets by support domain.",
                    "generated_at_turn": 3,
                },
                "batch_run_config": {
                    "dataset_name": "enterprise_ticket",
                    "human_agent_config": {"guidance": None},
                    "metadata": {
                        "topic_branch_mode": "auto_topic_discovery",
                        "topic_branch_topic_id": "support-domain",
                        "topic_branch_topic_label": "Support Domain",
                        "topic_branch_topic_guidance": "Group tickets by support domain.",
                    },
                },
                "conversation_trace": {
                    "messages": [{"role": "human", "content": "Save a clustering."}]
                },
                "judge_clustering_trace": {
                    "clusterings": [
                        {
                            "turn_index": 1,
                            "clustering": {
                                "clusters": [
                                    {
                                        "name": "IT and Finance",
                                        "members": [
                                            "okta locked",
                                            "badge broken",
                                            "invoice overdue",
                                            "vpn down",
                                        ],
                                    }
                                ]
                            },
                        }
                    ]
                },
            },
            metadata={"config_slug": "cfg", "run_id": "run-1"},
        )
    )

    assert result.details["reference_id"] == "department__normal_b"
    assert result.details["granularity"] == "normal"
    assert result.details["topic_resolution_mode"] == "best_reference_fallback"
    assert result.metrics["best_score_seen_structural"] == pytest.approx(1.0)


def test_robustness_expectations_evaluator_classifies_invariant_pairs() -> None:
    benchmark = _benchmark_spec()
    request = PairwiseEvaluationRequest(
        left_predictions=benchmark.reference_by_id["fruit__normal"].to_loaded_clustering(),
        right_predictions=benchmark.reference_by_id["fruit__normal"].to_loaded_clustering(),
        artifacts={
            "left_benchmark_spec": benchmark,
            "right_benchmark_spec": benchmark,
            "left_selected_topic": {"topic_id": "fruit"},
            "right_selected_topic": {"topic_id": "fruit"},
        },
        metadata={"pairwise_scope": "inter"},
    )

    evaluator = RobustnessExpectationsEvaluator.from_params()
    result = evaluator.evaluate(request)

    assert result.metrics["adjusted_rand_index"] == pytest.approx(1.0)
    assert result.metrics["normalized_mutual_info"] == pytest.approx(1.0)
    assert result.metrics["rand_index"] == pytest.approx(1.0)
    assert result.metrics["coassignment_jaccard"] == pytest.approx(1.0)
    assert result.metrics["fowlkes_mallows"] == pytest.approx(1.0)
    assert result.metrics["invariance_pair_count"] == pytest.approx(1.0)
    assert result.metrics["responsiveness_pair_count"] == pytest.approx(0.0)
    assert "invariance_success_rate" not in result.metrics
    assert "responsiveness_success_rate" not in result.metrics


def test_robustness_expectations_evaluator_handles_zero_overlap() -> None:
    benchmark = _benchmark_spec()
    request = PairwiseEvaluationRequest(
        left_predictions=LoadedClustering(
            assignment={"apple": 0, "banana": 0},
            cluster_names=["fruit"],
        ),
        right_predictions=LoadedClustering(
            assignment={"car": 0, "truck": 0},
            cluster_names=["vehicle"],
        ),
        artifacts={
            "left_benchmark_spec": benchmark,
            "right_benchmark_spec": benchmark,
            "left_selected_topic": {"topic_id": "fruit"},
            "right_selected_topic": {"topic_id": "vehicle"},
        },
        metadata={"pairwise_scope": "inter"},
    )

    evaluator = RobustnessExpectationsEvaluator.from_params()
    result = evaluator.evaluate(request)

    assert result.metrics["adjusted_rand_index"] == pytest.approx(0.0)
    assert result.metrics["normalized_mutual_info"] == pytest.approx(0.0)
    assert result.metrics["rand_index"] == pytest.approx(0.0)
    assert result.metrics["coassignment_jaccard"] == pytest.approx(0.0)
    assert result.metrics["fowlkes_mallows"] == pytest.approx(0.0)
    assert result.metrics["invariance_pair_count"] == pytest.approx(0.0)
    assert result.metrics["responsiveness_pair_count"] == pytest.approx(1.0)
    assert "invariance_success_rate" not in result.metrics
    assert "responsiveness_success_rate" not in result.metrics
    assert result.details["expected_pair_type"] == "responsive"
    assert result.details["num_common_items"] == 0
    assert result.details["num_union_items"] == 4
    assert result.details["common_item_coverage"] == pytest.approx(0.0)
    assert result.details["same_item_set"] is False
    assert result.details["num_item_pairs_compared"] == 0
    assert result.details["false_positive_pairs"] == 0
    assert result.details["false_negative_pairs"] == 0


def test_robustness_expectations_evaluator_uses_explicit_themes_without_benchmark_overlap() -> None:
    benchmark = _benchmark_spec()
    request = PairwiseEvaluationRequest(
        left_predictions=LoadedClustering(
            assignment={"orange": 0, "pear": 0},
            cluster_names=["fruit"],
        ),
        right_predictions=LoadedClustering(
            assignment={"sedan": 0, "suv": 0},
            cluster_names=["vehicle"],
        ),
        artifacts={
            "left_benchmark_spec": benchmark,
            "right_benchmark_spec": benchmark,
            "left_selected_topic": {"topic_id": "fruit"},
            "right_selected_topic": {"topic_id": "vehicle"},
        },
        metadata={"pairwise_scope": "inter"},
    )

    evaluator = RobustnessExpectationsEvaluator.from_params()
    result = evaluator.evaluate(request)

    assert result.details["left_theme_id"] == "fruit"
    assert result.details["right_theme_id"] == "vehicle"
    assert result.details["expected_pair_type"] == "responsive"
    assert result.details["num_common_items"] == 0
    assert result.metrics["adjusted_rand_index"] == pytest.approx(0.0)


def test_robustness_expectations_evaluator_uses_side_specific_benchmark_specs() -> None:
    left_benchmark = _benchmark_spec()
    right_benchmark = parse_synthetic_benchmark_spec(
        {
            "dataset_name": "synthetic_alt",
            "items": ["ticket a", "ticket b"],
            "themes": [
                {
                    "theme_id": "support",
                    "label": "Support",
                    "guidance": "Group tickets by support workflow.",
                }
            ],
            "references": [
                {
                    "reference_id": "support__normal",
                    "topic_id": "support",
                    "granularity": "normal",
                    "assignment": {"ticket a": 0, "ticket b": 0},
                    "cluster_names": ["support"],
                }
            ],
        }
    )
    request = PairwiseEvaluationRequest(
        left_predictions=left_benchmark.reference_by_id["fruit__normal"].to_loaded_clustering(),
        right_predictions=right_benchmark.reference_by_id["support__normal"].to_loaded_clustering(),
        artifacts={
            "left_benchmark_spec": left_benchmark,
            "right_benchmark_spec": right_benchmark,
            "left_selected_topic": {"topic_id": "fruit"},
            "right_selected_topic": {"topic_id": "support"},
        },
        metadata={"pairwise_scope": "inter"},
    )

    evaluator = RobustnessExpectationsEvaluator.from_params()
    result = evaluator.evaluate(request)

    assert result.details["left_theme_id"] == "fruit"
    assert result.details["right_theme_id"] == "support"
    assert result.details["expected_pair_type"] == "responsive"
    assert "responsiveness_success_rate" not in result.metrics


def test_robustness_expectations_evaluator_uses_semantic_theme_match_for_auto_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import robustness_expectations as module

    def mock_match(**_: object) -> ThemeMatchJudgment:
        return ThemeMatchJudgment(
            supported=True,
            benchmark_theme_id="issue_type",
            confidence=0.9,
            reasoning="Both discovered topics align to issue type.",
        )

    monkeypatch.setattr(module, "judge_theme_match", mock_match)

    benchmark = _enterprise_ticket_benchmark_spec()
    request = PairwiseEvaluationRequest(
        left_predictions=benchmark.reference_by_id["issue_type__normal"].to_loaded_clustering(),
        right_predictions=benchmark.reference_by_id["issue_type__normal"].to_loaded_clustering(),
        artifacts={
            "left_benchmark_spec": benchmark,
            "right_benchmark_spec": benchmark,
            "left_selected_topic": {
                "topic_id": "access-and-authentication-issues",
                "label": "Access and Authentication Issues",
                "guidance": "Group tickets by access and authentication issues.",
                "generated_at_turn": 2,
            },
            "right_selected_topic": {
                "topic_id": "authorization-escalations",
                "label": "Authorization Escalations",
                "guidance": "Group tickets by authorization escalations.",
                "generated_at_turn": 3,
            },
            "left_batch_run_config": {
                "dataset_name": "enterprise_ticket",
                "human_agent_config": {"guidance": None},
                "metadata": {
                    "topic_branch_mode": "auto_topic_discovery",
                    "topic_branch_topic_id": "access-and-authentication-issues",
                    "topic_branch_topic_label": "Access and Authentication Issues",
                    "topic_branch_topic_guidance": (
                        "Group tickets by access and authentication issues."
                    ),
                },
            },
            "right_batch_run_config": {
                "dataset_name": "enterprise_ticket",
                "human_agent_config": {"guidance": None},
                "metadata": {
                    "topic_branch_mode": "auto_topic_discovery",
                    "topic_branch_topic_id": "authorization-escalations",
                    "topic_branch_topic_label": "Authorization Escalations",
                    "topic_branch_topic_guidance": ("Group tickets by authorization escalations."),
                },
            },
        },
        metadata={"pairwise_scope": "inter"},
    )

    evaluator = RobustnessExpectationsEvaluator.from_params()
    result = evaluator.evaluate(request)

    assert result.details["left_theme_id"] == "issue_type"
    assert result.details["right_theme_id"] == "issue_type"
    assert result.details["left_original_topic_id"] == "access-and-authentication-issues"
    assert result.details["left_original_topic_label"] == "Access and Authentication Issues"
    assert result.details["left_resolved_theme_id"] == "issue_type"
    assert result.details["left_resolved_theme_label"] == "Issue Type"
    assert result.details["right_original_topic_id"] == "authorization-escalations"
    assert result.details["right_original_topic_label"] == "Authorization Escalations"
    assert result.details["right_resolved_theme_id"] == "issue_type"
    assert result.details["right_resolved_theme_label"] == "Issue Type"
    assert result.details["left_topic_resolution_mode"] == "semantic_judge"
    assert result.details["right_topic_resolution_mode"] == "semantic_judge"
    assert result.details["expected_pair_type"] == "invariant"


def test_robustness_expectations_evaluator_uses_best_reference_fallback_for_auto_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import robustness_expectations as module

    def mock_match(**_: object) -> ThemeMatchJudgment:
        return ThemeMatchJudgment(
            supported=False,
            benchmark_theme_id=None,
            confidence=0.2,
            reasoning="No semantic benchmark match.",
        )

    monkeypatch.setattr(module, "judge_theme_match", mock_match)

    benchmark = _enterprise_ticket_benchmark_spec()
    request = PairwiseEvaluationRequest(
        left_predictions=benchmark.reference_by_id["department__normal"].to_loaded_clustering(),
        right_predictions=benchmark.reference_by_id["issue_type__normal"].to_loaded_clustering(),
        artifacts={
            "left_benchmark_spec": benchmark,
            "right_benchmark_spec": benchmark,
            "left_selected_topic": {
                "topic_id": "support-domain",
                "label": "Support Domain",
                "guidance": "Group tickets by support domain.",
                "generated_at_turn": 2,
            },
            "right_selected_topic": {
                "topic_id": "password-policies",
                "label": "Password Policies",
                "guidance": "Group tickets by password policies.",
                "generated_at_turn": 3,
            },
            "left_batch_run_config": {
                "dataset_name": "enterprise_ticket",
                "human_agent_config": {"guidance": None},
                "metadata": {
                    "topic_branch_mode": "auto_topic_discovery",
                    "topic_branch_topic_id": "support-domain",
                    "topic_branch_topic_label": "Support Domain",
                    "topic_branch_topic_guidance": "Group tickets by support domain.",
                },
            },
            "right_batch_run_config": {
                "dataset_name": "enterprise_ticket",
                "human_agent_config": {"guidance": None},
                "metadata": {
                    "topic_branch_mode": "auto_topic_discovery",
                    "topic_branch_topic_id": "password-policies",
                    "topic_branch_topic_label": "Password Policies",
                    "topic_branch_topic_guidance": "Group tickets by password policies.",
                },
            },
        },
        metadata={"pairwise_scope": "inter"},
    )

    evaluator = RobustnessExpectationsEvaluator.from_params()
    result = evaluator.evaluate(request)

    assert result.details["left_theme_id"] == "department"
    assert result.details["right_theme_id"] == "issue_type"
    assert result.details["left_topic_resolution_mode"] == "best_reference_fallback"
    assert result.details["right_topic_resolution_mode"] == "best_reference_fallback"
    assert result.details["expected_pair_type"] == "responsive"


def test_robustness_expectations_evaluator_uses_fallback_without_default_granularity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods import robustness_expectations as module

    def mock_match(**_: object) -> ThemeMatchJudgment:
        return ThemeMatchJudgment(
            supported=False,
            benchmark_theme_id=None,
            confidence=0.2,
            reasoning="No semantic benchmark match.",
        )

    monkeypatch.setattr(module, "judge_theme_match", mock_match)

    benchmark = parse_synthetic_benchmark_spec(
        {
            "dataset_name": "enterprise_ticket",
            "items": ["okta locked", "badge broken", "invoice overdue", "vpn down"],
            "themes": [
                {
                    "theme_id": "department",
                    "label": "Department",
                    "guidance": "Group tickets by department.",
                },
                {
                    "theme_id": "issue_type",
                    "label": "Issue Type",
                    "guidance": "Group tickets by issue type.",
                },
            ],
            "references": [
                {
                    "reference_id": "department__coarse",
                    "topic_id": "department",
                    "granularity": "coarse",
                    "assignment": {
                        "okta locked": 1,
                        "badge broken": 1,
                        "invoice overdue": 2,
                        "vpn down": 1,
                    },
                    "cluster_names": ["Unused", "IT", "Finance"],
                },
                {
                    "reference_id": "issue_type__coarse",
                    "topic_id": "issue_type",
                    "granularity": "coarse",
                    "assignment": {
                        "okta locked": 0,
                        "badge broken": 1,
                        "invoice overdue": 2,
                        "vpn down": 3,
                    },
                    "cluster_names": [
                        "access_and_authentication",
                        "badge_problem",
                        "finance_transaction",
                        "service_issue",
                    ],
                },
            ],
        }
    )

    request = PairwiseEvaluationRequest(
        left_predictions=benchmark.reference_by_id["department__coarse"].to_loaded_clustering(),
        right_predictions=benchmark.reference_by_id["issue_type__coarse"].to_loaded_clustering(),
        artifacts={
            "left_benchmark_spec": benchmark,
            "right_benchmark_spec": benchmark,
            "left_selected_topic": {
                "topic_id": "support-domain",
                "label": "Support Domain",
                "guidance": "Group tickets by support domain.",
                "generated_at_turn": 2,
            },
            "right_selected_topic": {
                "topic_id": "password-policies",
                "label": "Password Policies",
                "guidance": "Group tickets by password policies.",
                "generated_at_turn": 3,
            },
            "left_batch_run_config": {
                "dataset_name": "enterprise_ticket",
                "human_agent_config": {"guidance": None},
                "metadata": {
                    "topic_branch_mode": "auto_topic_discovery",
                    "topic_branch_topic_id": "support-domain",
                    "topic_branch_topic_label": "Support Domain",
                    "topic_branch_topic_guidance": "Group tickets by support domain.",
                },
            },
            "right_batch_run_config": {
                "dataset_name": "enterprise_ticket",
                "human_agent_config": {"guidance": None},
                "metadata": {
                    "topic_branch_mode": "auto_topic_discovery",
                    "topic_branch_topic_id": "password-policies",
                    "topic_branch_topic_label": "Password Policies",
                    "topic_branch_topic_guidance": "Group tickets by password policies.",
                },
            },
        },
        metadata={"pairwise_scope": "inter"},
    )

    evaluator = RobustnessExpectationsEvaluator.from_params()
    result = evaluator.evaluate(request)

    assert result.details["left_theme_id"] == "department"
    assert result.details["right_theme_id"] == "issue_type"
    assert result.details["left_topic_resolution_mode"] == "best_reference_fallback"
    assert result.details["right_topic_resolution_mode"] == "best_reference_fallback"
    assert result.details["expected_pair_type"] == "responsive"


def test_robustness_expectations_evaluator_emits_success_metrics_when_thresholds_configured() -> (
    None
):
    benchmark = _benchmark_spec()
    request = PairwiseEvaluationRequest(
        left_predictions=benchmark.reference_by_id["fruit__normal"].to_loaded_clustering(),
        right_predictions=benchmark.reference_by_id["fruit__normal"].to_loaded_clustering(),
        artifacts={
            "left_benchmark_spec": benchmark,
            "right_benchmark_spec": benchmark,
            "left_selected_topic": {"topic_id": "fruit"},
            "right_selected_topic": {"topic_id": "fruit"},
        },
        metadata={"pairwise_scope": "inter"},
    )

    evaluator = RobustnessExpectationsEvaluator.from_params({"invariance_threshold": 0.8})
    result = evaluator.evaluate(request)

    assert result.metrics["invariance_success_rate"] == pytest.approx(1.0)
    assert result.details["invariance_threshold"] == pytest.approx(0.8)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf")])
def test_robustness_expectations_evaluator_rejects_non_finite_thresholds(
    threshold: float,
) -> None:
    with pytest.raises(ValueError, match="finite number between -1.0 and 1.0"):
        RobustnessExpectationsEvaluator.from_params({"invariance_threshold": threshold})
    with pytest.raises(ValueError, match="finite number between -1.0 and 1.0"):
        RobustnessExpectationsEvaluator.from_params({"responsiveness_threshold": threshold})


def test_robustness_expectations_summary_stays_stable_for_mixed_pair_types() -> None:
    benchmark = _benchmark_spec()
    evaluator = RobustnessExpectationsEvaluator.from_params(
        {"invariance_threshold": 0.8, "responsiveness_threshold": 0.4}
    )

    invariant_result = evaluator.evaluate(
        PairwiseEvaluationRequest(
            left_predictions=benchmark.reference_by_id["fruit__normal"].to_loaded_clustering(),
            right_predictions=benchmark.reference_by_id["fruit__normal"].to_loaded_clustering(),
            artifacts={
                "left_benchmark_spec": benchmark,
                "right_benchmark_spec": benchmark,
                "left_selected_topic": {"topic_id": "fruit"},
                "right_selected_topic": {"topic_id": "fruit"},
            },
            metadata={"pairwise_scope": "inter"},
        )
    )
    responsive_result = evaluator.evaluate(
        PairwiseEvaluationRequest(
            left_predictions=benchmark.reference_by_id["fruit__normal"].to_loaded_clustering(),
            right_predictions=benchmark.reference_by_id["vehicle__normal"].to_loaded_clustering(),
            artifacts={
                "left_benchmark_spec": benchmark,
                "right_benchmark_spec": benchmark,
                "left_selected_topic": {"topic_id": "fruit"},
                "right_selected_topic": {"topic_id": "vehicle"},
            },
            metadata={"pairwise_scope": "inter"},
        )
    )

    summary = EvaluationResult.summarize([invariant_result, responsive_result])

    assert summary.metrics["invariance_pair_count"] == pytest.approx(0.5)
    assert summary.metrics["invariance_success_rate"] == pytest.approx(1.0)
    assert summary.metrics["responsiveness_pair_count"] == pytest.approx(0.5)
    assert summary.metrics["responsiveness_success_rate"] == pytest.approx(0.0)


def test_robustness_expectations_summary_omits_inapplicable_pair_type_rate() -> None:
    benchmark = _benchmark_spec()
    evaluator = RobustnessExpectationsEvaluator.from_params({"responsiveness_threshold": 0.4})
    responsive_result = evaluator.evaluate(
        PairwiseEvaluationRequest(
            left_predictions=LoadedClustering(
                assignment={"apple": 0, "banana": 0},
                cluster_names=["fruit"],
            ),
            right_predictions=LoadedClustering(
                assignment={"car": 0, "truck": 0},
                cluster_names=["vehicle"],
            ),
            artifacts={
                "left_benchmark_spec": benchmark,
                "right_benchmark_spec": benchmark,
                "left_selected_topic": {"topic_id": "fruit"},
                "right_selected_topic": {"topic_id": "vehicle"},
            },
            metadata={"pairwise_scope": "inter"},
        )
    )

    summary = EvaluationResult.summarize([responsive_result])

    assert summary.metrics["invariance_pair_count"] == pytest.approx(0.0)
    assert "invariance_success_rate" not in summary.metrics
    assert summary.metrics["responsiveness_pair_count"] == pytest.approx(1.0)
    assert summary.metrics["responsiveness_success_rate"] == pytest.approx(1.0)


def test_efficiency_utility_summary_includes_threshold_misses() -> None:
    summary = EvaluationResult.summarize(
        [
            EvaluationResult(
                method="efficiency_utility",
                metrics={
                    "best_score_seen_structural": 1.0,
                    "structural_threshold_reached_rate": 1.0,
                    "turn_to_threshold_structural": 1.0,
                },
                details={},
            ),
            EvaluationResult(
                method="efficiency_utility",
                metrics={
                    "best_score_seen_structural": 0.2,
                    "structural_threshold_reached_rate": 0.0,
                },
                details={},
            ),
        ]
    )

    assert summary.metrics["best_score_seen_structural"] == pytest.approx(0.6)
    assert summary.metrics["structural_threshold_reached_rate"] == pytest.approx(0.5)
    assert summary.metrics["turn_to_threshold_structural"] == pytest.approx(1.0)


def test_run_evaluation_config_loads_benchmark_spec_for_robustness_expectations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark_path = tmp_path / "synthetic.benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "dataset_name": "synthetic",
                "items": ["apple", "banana", "car", "truck"],
                "themes": [
                    {
                        "theme_id": "fruit",
                        "label": "Fruit",
                        "guidance": "Group items by fruitness.",
                    }
                ],
                "references": [
                    {
                        "reference_id": "fruit__normal",
                        "topic_id": "fruit",
                        "granularity": "normal",
                        "assignment": {"apple": 1, "banana": 1, "car": 0, "truck": 0},
                        "cluster_names": ["other", "fruit"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        cast(dict[str, str], DATASETS["synthetic"]),
        "benchmark_spec_path",
        str(benchmark_path),
    )

    input_dir = tmp_path / "batch_output"
    for config_slug, run_id in (("cfg-a", "run-1"), ("cfg-b", "run-2")):
        run_dir = input_dir / config_slug / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "clustering.json").write_text(
            json.dumps(_predictions_payload()),
            encoding="utf-8",
        )
        (run_dir / "conversation_trace.json").write_text(
            json.dumps({"messages": [{"role": "human", "content": "hello"}]}), encoding="utf-8"
        )
        (run_dir / "selected_topic.json").write_text(
            json.dumps(
                {"topic_id": "fruit", "label": "Fruit", "guidance": "Group items by fruitness."}
            ),
            encoding="utf-8",
        )
    (input_dir / "results.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "config": {
                            "dataset_name": "synthetic",
                            "human_agent_config": {"prompt_id": "a", "guidance": "fruit"},
                        },
                        "config_slug": "cfg-a",
                        "run_ids": ["run-1"],
                    }
                ),
                json.dumps(
                    {
                        "config": {
                            "dataset_name": "synthetic",
                            "human_agent_config": {"prompt_id": "b", "guidance": "fruit"},
                        },
                        "config_slug": "cfg-b",
                        "run_ids": ["run-2"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    evaluation_config_path = tmp_path / "pairwise_evaluation.toml"
    evaluation_config_path.write_text(
        "\n".join(
            [
                f'input_dir = "{input_dir}"',
                'evaluators = ["robustness_expectations"]',
                'pairwise_scope = "inter"',
                f'report_path = "{tmp_path / "pairwise_report.json"}"',
            ]
        ),
        encoding="utf-8",
    )

    report_path = run_evaluation_config(evaluation_config_path)
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))

    assert report["summary_by_evaluator"]["robustness_expectations"]["metrics"][
        "adjusted_rand_index"
    ] == pytest.approx(1.0)


def test_run_evaluation_config_continues_after_pairwise_evaluator_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conversational_clustering.evaluation import runner as module

    benchmark_path = tmp_path / "synthetic.benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "dataset_name": "synthetic",
                "items": ["apple", "banana", "car", "truck"],
                "themes": [
                    {
                        "theme_id": "fruit",
                        "label": "Fruit",
                        "guidance": "Group items by fruitness.",
                    }
                ],
                "references": [
                    {
                        "reference_id": "fruit__normal",
                        "topic_id": "fruit",
                        "granularity": "normal",
                        "assignment": {"apple": 1, "banana": 1, "car": 0, "truck": 0},
                        "cluster_names": ["other", "fruit"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        cast(dict[str, str], DATASETS["synthetic"]),
        "benchmark_spec_path",
        str(benchmark_path),
    )

    class FlakyPairwiseEvaluator:
        name = "robustness_expectations"
        resume_version = "v1"

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, request: PairwiseEvaluationRequest) -> EvaluationResult:
            self.calls += 1
            if self.calls == 1:
                raise ValueError("synthetic pairwise-evaluator failure")
            return EvaluationResult(
                method=self.name,
                metrics={"adjusted_rand_index": 1.0},
                details={"num_common_items": float(len(request.left_predictions.assignment))},
            )

    flaky_evaluator = FlakyPairwiseEvaluator()
    monkeypatch.setattr(
        module,
        "get_pairwise_evaluator",
        lambda name, params=None: flaky_evaluator,
    )

    input_dir = tmp_path / "batch_output"
    for config_slug, _prompt_id, run_id in (
        ("cfg-a", "a", "run-a"),
        ("cfg-b", "b", "run-b"),
        ("cfg-c", "c", "run-c"),
    ):
        run_dir = input_dir / config_slug / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "clustering.json").write_text(
            json.dumps(_predictions_payload()),
            encoding="utf-8",
        )
        (run_dir / "conversation_trace.json").write_text(
            json.dumps({"messages": [{"role": "human", "content": "hello"}]}),
            encoding="utf-8",
        )
        (run_dir / "selected_topic.json").write_text(
            json.dumps(
                {"topic_id": "fruit", "label": "Fruit", "guidance": "Group items by fruitness."}
            ),
            encoding="utf-8",
        )
    (input_dir / "results.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "config": {
                            "dataset_name": "synthetic",
                            "human_agent_config": {"prompt_id": "a", "guidance": "fruit"},
                        },
                        "config_slug": "cfg-a",
                        "run_ids": ["run-a"],
                    }
                ),
                json.dumps(
                    {
                        "config": {
                            "dataset_name": "synthetic",
                            "human_agent_config": {"prompt_id": "b", "guidance": "fruit"},
                        },
                        "config_slug": "cfg-b",
                        "run_ids": ["run-b"],
                    }
                ),
                json.dumps(
                    {
                        "config": {
                            "dataset_name": "synthetic",
                            "human_agent_config": {"prompt_id": "c", "guidance": "fruit"},
                        },
                        "config_slug": "cfg-c",
                        "run_ids": ["run-c"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    evaluation_config_path = tmp_path / "pairwise_evaluation.toml"
    evaluation_config_path.write_text(
        "\n".join(
            [
                f'input_dir = "{input_dir}"',
                'evaluators = ["robustness_expectations"]',
                'pairwise_scope = "inter"',
                f'report_path = "{tmp_path / "pairwise_report.json"}"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationRunFailed) as exc_info:
        run_evaluation_config(evaluation_config_path)
    report = json.loads(Path(exc_info.value.report_path).read_text(encoding="utf-8"))
    details_path = Path(report["results_files_by_evaluator"]["robustness_expectations"])
    detail_rows = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert report["num_pairwise_requests"] == 3
    assert report["num_failures_by_evaluator"] == {"robustness_expectations": 1}
    assert report["summary_by_evaluator"]["robustness_expectations"]["details"]["num_results"] == 2
    assert len(detail_rows) == 3
    assert detail_rows[0]["status"] == "error"
    assert detail_rows[0]["details"]["error_message"] == "synthetic pairwise-evaluator failure"
    assert detail_rows[1]["metrics"]["adjusted_rand_index"] == pytest.approx(1.0)
