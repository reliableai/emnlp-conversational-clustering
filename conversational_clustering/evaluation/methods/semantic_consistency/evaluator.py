"""Semantic consistency evaluator using LLM inference and similarity judging."""

from collections.abc import Mapping
from typing import Any

from ....accounting import UsageCollector, build_pricing_client_from_configs
from ....models import (
    ModelConfig,
    default_semantic_consistency_inference_model_config,
    default_semantic_consistency_judge_model_config,
    parse_model_config,
)
from ....prompts.registry import (
    get_semantic_consistency_inference_prompt,
    get_semantic_consistency_judge_prompt,
)
from ...base import EvaluationRequest, EvaluationResult, Evaluator
from .inference import (
    calculate_variance,
    infer_guidance_from_clustering,
    judge_similarity,
)


class SemanticConsistencyEvaluator(Evaluator):
    name = "semantic_consistency"
    resume_version = "v2"

    def __init__(
        self,
        model_config: ModelConfig | None,
        num_samples: int,
        inference_prompt_id: str,
        judge_prompt_id: str,
    ):
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        self.model_config = model_config
        self.num_samples = num_samples
        self.inference_prompt_id = inference_prompt_id
        self.judge_prompt_id = judge_prompt_id

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None = None) -> "SemanticConsistencyEvaluator":
        if params is None:
            params = {}

        valid_params = {
            "model_config",
            "num_samples",
            "inference_prompt_id",
            "judge_prompt_id",
        }
        invalid_params = set(params.keys()) - valid_params
        if invalid_params:
            invalid = ", ".join(sorted(invalid_params))
            raise ValueError(f"Invalid parameter(s) for '{cls.name}': {invalid}")

        model_config_dict = params.get("model_config")
        model_config = None
        if model_config_dict is not None:
            if not isinstance(model_config_dict, dict):
                type_name = type(model_config_dict).__name__
                raise TypeError(f"model_config must be a dict, got {type_name}")
            try:
                model_config = parse_model_config(model_config_dict)
            except ValueError as exc:
                raise ValueError(f"'model_config' is invalid: {exc}") from exc

        num_samples_value = params.get("num_samples", 5)
        if isinstance(num_samples_value, bool) or not isinstance(num_samples_value, int):
            raise ValueError(
                f"'num_samples' must be an integer, got {type(num_samples_value).__name__}"
            )
        if num_samples_value <= 0:
            raise ValueError(f"'num_samples' must be positive, got {num_samples_value}")

        inference_prompt_id = params.get("inference_prompt_id", "baseline_v1")
        try:
            get_semantic_consistency_inference_prompt(inference_prompt_id)
        except ValueError as exc:
            raise ValueError(f"Invalid inference_prompt_id: {exc}") from exc

        judge_prompt_id = params.get("judge_prompt_id", "baseline_v1")
        try:
            get_semantic_consistency_judge_prompt(judge_prompt_id)
        except ValueError as exc:
            raise ValueError(f"Invalid judge_prompt_id: {exc}") from exc

        return cls(
            model_config=model_config,
            num_samples=num_samples_value,
            inference_prompt_id=inference_prompt_id,
            judge_prompt_id=judge_prompt_id,
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        usage_collector = UsageCollector()
        raw_batch_run_config = request.artifacts.get("batch_run_config", {})
        batch_run_config = raw_batch_run_config if isinstance(raw_batch_run_config, Mapping) else {}
        actual_guidance = self._resolve_actual_guidance(batch_run_config=batch_run_config)

        if not actual_guidance:
            raise ValueError(
                "Cannot evaluate semantic consistency: no guidance found. "
                "Guidance is required in "
                "batch_run_config['human_agent_config']['guidance'] or "
                "batch_run_config['metadata']['topic_branch_topic_guidance']."
            )

        unlabeled_clustering = self._remove_labels(request.predictions)

        inferred_guidances = []
        for _ in range(self.num_samples):
            inferred = infer_guidance_from_clustering(
                unlabeled_clustering,
                model_config=self.model_config,
                inference_prompt_id=self.inference_prompt_id,
                usage_collector=usage_collector,
            )
            inferred_guidances.append(inferred)

        similarity_scores = []
        for inferred in inferred_guidances:
            score = judge_similarity(
                actual_guidance=actual_guidance,
                inferred_guidance=inferred,
                model_config=self.model_config,
                judge_prompt_id=self.judge_prompt_id,
                usage_collector=usage_collector,
            )
            similarity_scores.append(score)

        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        variance = calculate_variance(similarity_scores)

        return EvaluationResult(
            method=self.name,
            metrics={
                "score": avg_similarity / 10.0,
                "average_similarity": avg_similarity,
                "variance": variance,
            },
            details={
                "similarity_scores": similarity_scores,
                "inferred_guidances": inferred_guidances,
                "actual_guidance": actual_guidance,
                "unlabeled_clustering": unlabeled_clustering,
                "num_samples": self.num_samples,
            },
            accounting=usage_collector.snapshot(
                pricer=build_pricing_client_from_configs(
                    [self.model_config]
                    if self.model_config
                    else [
                        default_semantic_consistency_inference_model_config(),
                        default_semantic_consistency_judge_model_config(),
                    ]
                )
            ).to_dict(),
        )

    def _resolve_actual_guidance(self, *, batch_run_config: Mapping[str, Any]) -> str:
        human_agent_config = batch_run_config.get("human_agent_config")
        if isinstance(human_agent_config, Mapping):
            raw_guidance = human_agent_config.get("guidance")
            if isinstance(raw_guidance, str):
                normalized_guidance = raw_guidance.strip()
                if normalized_guidance:
                    return normalized_guidance

        metadata = batch_run_config.get("metadata")
        if isinstance(metadata, Mapping):
            raw_topic_branch_guidance = metadata.get("topic_branch_topic_guidance")
            if isinstance(raw_topic_branch_guidance, str):
                normalized_topic_branch_guidance = raw_topic_branch_guidance.strip()
                if normalized_topic_branch_guidance:
                    return normalized_topic_branch_guidance

        return ""

    def _remove_labels(self, predictions) -> str:
        lines = []
        for i, _cluster_name in enumerate(predictions.cluster_names, start=1):
            members = [
                item
                for item, assignment_idx in zip(
                    predictions.assignment.keys(),
                    predictions.assignment.values(),
                    strict=True,
                )
                if assignment_idx == i - 1
            ]
            lines.append(f"Cluster {i}:")
            for member in members:
                lines.append(f"  - {member}")
            lines.append("")
        return "\n".join(lines)
