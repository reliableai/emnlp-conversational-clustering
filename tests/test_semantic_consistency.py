"""Tests for semantic consistency evaluator."""

from __future__ import annotations

import pytest

from conversational_clustering.evaluation.artifacts import LoadedClustering
from conversational_clustering.evaluation.base import EvaluationRequest
from conversational_clustering.evaluation.methods.semantic_consistency import (
    SemanticConsistencyEvaluator,
)
from conversational_clustering.evaluation.methods.semantic_consistency.inference import (
    calculate_variance,
)
from conversational_clustering.models import (
    ModelConfig,
    default_semantic_consistency_inference_model_config,
    default_semantic_consistency_judge_model_config,
)


def _predictions() -> LoadedClustering:
    return LoadedClustering(
        assignment={"apple": 0, "banana": 0, "car": 1, "truck": 1},
        cluster_names=["fruits", "vehicles"],
    )


def test_semantic_consistency_requires_positive_num_samples() -> None:
    with pytest.raises(ValueError, match="num_samples must be positive"):
        SemanticConsistencyEvaluator(
            model_config=None,
            num_samples=0,
            inference_prompt_id="baseline_v1",
            judge_prompt_id="baseline_v1",
        )

    with pytest.raises(ValueError, match="num_samples must be positive"):
        SemanticConsistencyEvaluator(
            model_config=None,
            num_samples=-1,
            inference_prompt_id="baseline_v1",
            judge_prompt_id="baseline_v1",
        )


def test_semantic_consistency_rejects_invalid_num_samples_type() -> None:
    with pytest.raises(ValueError, match="'num_samples' must be an integer"):
        SemanticConsistencyEvaluator.from_params({"num_samples": 3.5})

    with pytest.raises(ValueError, match="'num_samples' must be an integer"):
        SemanticConsistencyEvaluator.from_params({"num_samples": True})

    with pytest.raises(ValueError, match="'num_samples' must be positive"):
        SemanticConsistencyEvaluator.from_params({"num_samples": 0})

    with pytest.raises(ValueError, match="'num_samples' must be positive"):
        SemanticConsistencyEvaluator.from_params({"num_samples": -5})


def test_semantic_consistency_from_params_uses_defaults() -> None:
    evaluator = SemanticConsistencyEvaluator.from_params(None)
    assert evaluator.num_samples == 5
    assert evaluator.model_config is None
    assert evaluator.inference_prompt_id == "baseline_v1"
    assert evaluator.judge_prompt_id == "baseline_v1"


def test_semantic_consistency_from_params_accepts_custom_values() -> None:
    evaluator = SemanticConsistencyEvaluator.from_params(
        {
            "num_samples": 3,
            "model_config": {"model": "openai/gpt-5.4", "temperature": 0.5},
            "inference_prompt_id": "baseline_v1",
            "judge_prompt_id": "baseline_v1",
        }
    )
    assert evaluator.num_samples == 3
    assert evaluator.model_config is not None
    assert evaluator.model_config.temperature == 0.5
    assert evaluator.inference_prompt_id == "baseline_v1"
    assert evaluator.judge_prompt_id == "baseline_v1"


def test_semantic_consistency_from_params_accepts_model_config_dict() -> None:
    evaluator = SemanticConsistencyEvaluator.from_params(
        {
            "model_config": {
                "model": "openai/gpt-4",
                "base_url": "https://openrouter.ai/api/v1",
            }
        }
    )
    assert evaluator.model_config is not None
    assert isinstance(evaluator.model_config, ModelConfig)
    assert evaluator.model_config.model == "openai/gpt-4"


def test_semantic_consistency_returns_error_when_no_guidance() -> None:
    evaluator = SemanticConsistencyEvaluator.from_params({"num_samples": 2})
    request = EvaluationRequest(
        predictions=_predictions(),
        artifacts={"batch_run_config": {}},
        metadata={"run_id": "test"},
    )
    with pytest.raises(ValueError, match="Cannot evaluate semantic consistency: no guidance found"):
        evaluator.evaluate(request)


def test_semantic_consistency_uses_topic_branch_guidance_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods.semantic_consistency import evaluator

    def mock_infer(
        clustering_text: str,  # noqa: ARG001
        model_config: object,  # noqa: ARG001
        inference_prompt_id: str,  # noqa: ARG001
        usage_collector: object | None = None,  # noqa: ARG001
    ) -> str:
        return "Group by inferred pattern"

    def mock_judge(
        actual_guidance: str,  # noqa: ARG001
        inferred_guidance: str,  # noqa: ARG001
        model_config: object,  # noqa: ARG001
        judge_prompt_id: str,  # noqa: ARG001
        usage_collector: object | None = None,  # noqa: ARG001
    ) -> float:
        return 9.0

    monkeypatch.setattr(evaluator, "infer_guidance_from_clustering", mock_infer)
    monkeypatch.setattr(evaluator, "judge_similarity", mock_judge)

    evaluator_instance = SemanticConsistencyEvaluator.from_params({"num_samples": 2})
    request = EvaluationRequest(
        predictions=_predictions(),
        artifacts={
            "batch_run_config": {
                "human_agent_config": {"guidance": None},
                "metadata": {"topic_branch_topic_guidance": " Group items by tone. "},
            }
        },
        metadata={"run_id": "test"},
    )

    result = evaluator_instance.evaluate(request)

    assert result.metrics["score"] == 0.9
    assert result.details["actual_guidance"] == "Group items by tone."


def test_remove_labels_formats_clustering_without_names() -> None:
    evaluator = SemanticConsistencyEvaluator.from_params(None)
    predictions = LoadedClustering(
        assignment={"item1": 0, "item2": 0, "item3": 1},
        cluster_names=["cluster_a", "cluster_b"],
    )
    unlabeled = evaluator._remove_labels(predictions)

    assert "Cluster 1:" in unlabeled
    assert "Cluster 2:" in unlabeled
    assert "cluster_a" not in unlabeled
    assert "cluster_b" not in unlabeled
    assert "item1" in unlabeled
    assert "item2" in unlabeled
    assert "item3" in unlabeled


def test_calculate_variance_returns_zero_for_empty_list() -> None:
    assert calculate_variance([]) == 0.0


def test_calculate_variance_returns_zero_for_single_value() -> None:
    assert calculate_variance([5.0]) == 0.0


def test_calculate_variance_computes_correctly() -> None:
    # For [1, 2, 3], mean=2, variance=((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = 2/3
    variance = calculate_variance([1.0, 2.0, 3.0])
    assert abs(variance - 2 / 3) < 1e-10


def test_calculate_variance_handles_identical_values() -> None:
    assert calculate_variance([5.0, 5.0, 5.0]) == 0.0


def test_semantic_consistency_rejects_non_dict_model_config() -> None:
    with pytest.raises(TypeError, match="model_config must be a dict"):
        SemanticConsistencyEvaluator.from_params({"model_config": "not-a-dict"})


def test_semantic_consistency_rejects_invalid_inference_prompt_id() -> None:
    with pytest.raises(ValueError, match="Invalid inference_prompt_id"):
        SemanticConsistencyEvaluator.from_params({"inference_prompt_id": "nonexistent_prompt"})


def test_semantic_consistency_rejects_invalid_judge_prompt_id() -> None:
    with pytest.raises(ValueError, match="Invalid judge_prompt_id"):
        SemanticConsistencyEvaluator.from_params({"judge_prompt_id": "nonexistent_prompt"})


def test_semantic_consistency_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="Invalid parameter\\(s\\) for 'semantic_consistency'"):
        SemanticConsistencyEvaluator.from_params({"invalid_param": "value"})

    with pytest.raises(ValueError, match="Invalid parameter\\(s\\) for 'semantic_consistency'"):
        SemanticConsistencyEvaluator.from_params({"num_samples": 3, "typo_param": 123})


def test_semantic_consistency_evaluate_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from conversational_clustering.evaluation.methods.semantic_consistency import evaluator

    # Mock the inference and judge functions to avoid actual LLM calls
    def mock_infer(
        clustering_text: str,  # noqa: ARG001
        model_config: object,  # noqa: ARG001
        inference_prompt_id: str,  # noqa: ARG001
        usage_collector: object | None = None,  # noqa: ARG001
    ) -> str:
        return "Group items by their category"

    def mock_judge(
        actual_guidance: str,  # noqa: ARG001
        inferred_guidance: str,  # noqa: ARG001
        model_config: object,  # noqa: ARG001
        judge_prompt_id: str,  # noqa: ARG001
        usage_collector: object | None = None,  # noqa: ARG001
    ) -> float:
        return 8.5

    monkeypatch.setattr(evaluator, "infer_guidance_from_clustering", mock_infer)
    monkeypatch.setattr(evaluator, "judge_similarity", mock_judge)

    evaluator_instance = SemanticConsistencyEvaluator.from_params({"num_samples": 3})
    request = EvaluationRequest(
        predictions=_predictions(),
        artifacts={"batch_run_config": {"human_agent_config": {"guidance": "Group by type"}}},
        metadata={"run_id": "test"},
    )

    result = evaluator_instance.evaluate(request)

    assert result.method == "semantic_consistency"
    assert "score" in result.metrics
    assert result.metrics["score"] == 0.85  # 8.5 / 10.0
    assert "variance" in result.metrics
    assert result.metrics["variance"] == 0.0  # All mock scores are identical
    assert "average_similarity" in result.metrics
    assert result.metrics["average_similarity"] == 8.5
    assert "inferred_guidances" in result.details
    assert len(result.details["inferred_guidances"]) == 3
    assert all(g == "Group items by their category" for g in result.details["inferred_guidances"])


def test_semantic_consistency_prices_default_models_when_model_config_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conversational_clustering.evaluation.methods.semantic_consistency import evaluator

    captured_configs: list[object] = []

    def mock_infer(
        clustering_text: str,  # noqa: ARG001
        model_config: object,  # noqa: ARG001
        inference_prompt_id: str,  # noqa: ARG001
        usage_collector: object | None = None,  # noqa: ARG001
    ) -> str:
        return "Group items by their category"

    def mock_judge(
        actual_guidance: str,  # noqa: ARG001
        inferred_guidance: str,  # noqa: ARG001
        model_config: object,  # noqa: ARG001
        judge_prompt_id: str,  # noqa: ARG001
        usage_collector: object | None = None,  # noqa: ARG001
    ) -> float:
        return 8.5

    class FakePricingClient:
        def get_text_prices(self, model_id: str) -> None:
            _ = model_id
            return None

    def fake_build_pricing_client_from_configs(configs: list[object]) -> FakePricingClient:
        captured_configs.extend(configs)
        return FakePricingClient()

    monkeypatch.setattr(evaluator, "infer_guidance_from_clustering", mock_infer)
    monkeypatch.setattr(evaluator, "judge_similarity", mock_judge)
    monkeypatch.setattr(
        evaluator,
        "build_pricing_client_from_configs",
        fake_build_pricing_client_from_configs,
    )

    evaluator_instance = SemanticConsistencyEvaluator.from_params({"num_samples": 1})
    request = EvaluationRequest(
        predictions=_predictions(),
        artifacts={"batch_run_config": {"human_agent_config": {"guidance": "Group by type"}}},
        metadata={"run_id": "test"},
    )

    _ = evaluator_instance.evaluate(request)

    assert default_semantic_consistency_inference_model_config() in captured_configs
    assert default_semantic_consistency_judge_model_config() in captured_configs
