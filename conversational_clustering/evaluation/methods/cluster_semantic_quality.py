"""LLM-judged semantic coherence and distinctiveness for final clusterings."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from conversational_clustering.accounting import UsageCollector, build_pricing_client_from_configs
from conversational_clustering.evaluation.artifacts import LoadedClustering
from conversational_clustering.evaluation.base import EvaluationRequest, EvaluationResult, Evaluator
from conversational_clustering.evaluation.methods.semantic_quality_judges import (
    ClusterCoherenceJudgment,
    ClusterDistinctivenessJudgment,
    build_structured_semantic_judge,
    judge_cluster_coherence,
    judge_cluster_distinctiveness,
)
from conversational_clustering.models import (
    ModelConfig,
    default_semantic_consistency_judge_model_config,
    parse_model_config,
)
from conversational_clustering.prompts.registry import (
    get_cluster_semantic_coherence_prompt,
    get_cluster_semantic_distinctiveness_prompt,
)


@dataclass(frozen=True)
class ClusterSemanticQualityEvaluator(Evaluator):
    name = "cluster_semantic_quality"
    resume_version = "v2"
    model_config: ModelConfig | None = None
    coherence_prompt_id: str = "baseline_v1"
    distinctiveness_prompt_id: str = "baseline_v1"

    @classmethod
    def from_params(
        cls, params: Mapping[str, Any] | None = None
    ) -> ClusterSemanticQualityEvaluator:
        normalized_params = dict(params or {})
        supported_keys = {"model_config", "coherence_prompt_id", "distinctiveness_prompt_id"}
        unsupported_keys = sorted(set(normalized_params) - supported_keys)
        if unsupported_keys:
            raise ValueError(
                f"Evaluator '{cls.name}' received unsupported parameter(s): "
                + ", ".join(unsupported_keys)
            )
        model_config = _parse_optional_model_config(normalized_params.get("model_config"))
        coherence_prompt_id = str(
            normalized_params.get("coherence_prompt_id", "baseline_v1")
        ).strip()
        distinctiveness_prompt_id = str(
            normalized_params.get("distinctiveness_prompt_id", "baseline_v1")
        ).strip()
        try:
            get_cluster_semantic_coherence_prompt(coherence_prompt_id)
        except ValueError as exc:
            raise ValueError(f"Invalid coherence_prompt_id: {exc}") from exc
        try:
            get_cluster_semantic_distinctiveness_prompt(distinctiveness_prompt_id)
        except ValueError as exc:
            raise ValueError(f"Invalid distinctiveness_prompt_id: {exc}") from exc
        return cls(
            model_config=model_config,
            coherence_prompt_id=coherence_prompt_id,
            distinctiveness_prompt_id=distinctiveness_prompt_id,
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        return self.evaluate_with_progress(request, progress_callback=None)

    def evaluate_with_progress(
        self,
        request: EvaluationRequest,
        *,
        progress_callback: Callable[[str], None] | None = None,
        usage_collector: UsageCollector | None = None,
    ) -> EvaluationResult:
        resolved_usage_collector = usage_collector or UsageCollector()
        guidance = _resolve_actual_guidance(request.artifacts)
        if not guidance:
            raise ValueError(
                f"Evaluator '{self.name}' requires active guidance from selected_topic "
                "or batch_run_config"
            )

        clusters = _clusters_from_predictions(request.predictions)
        if not clusters:
            return EvaluationResult(
                method=self.name,
                metrics={
                    "average_cluster_coherence": 0.0,
                    "cluster_distinctiveness": 0.0,
                    "supported_cluster_rate": 0.0,
                },
                details={"cluster_judgments": [], "distinctiveness_pairs": []},
            )

        coherence_llm: Any | None = None
        distinctiveness_llm: Any | None = None

        def get_coherence_llm() -> Any:
            nonlocal coherence_llm
            if coherence_llm is None:
                coherence_llm = build_structured_semantic_judge(
                    schema=ClusterCoherenceJudgment,
                    model_config=self.model_config,
                )
            return coherence_llm

        def get_distinctiveness_llm() -> Any:
            nonlocal distinctiveness_llm
            if distinctiveness_llm is None:
                distinctiveness_llm = build_structured_semantic_judge(
                    schema=ClusterDistinctivenessJudgment,
                    model_config=self.model_config,
                )
            return distinctiveness_llm

        cluster_judgments: list[dict[str, Any]] = []
        supported_count = 0
        coherence_scores: list[float] = []
        for coherence_index, (cluster_name, members) in enumerate(clusters, start=1):
            if progress_callback is not None:
                progress_callback(
                    "coherence "
                    f"{coherence_index}/{len(clusters)} for cluster '{cluster_name}' started"
                )
            judgment = judge_cluster_coherence(
                guidance=guidance,
                cluster_name=cluster_name,
                cluster_members="\n".join(f"- {member}" for member in members),
                model_config=self.model_config,
                prompt_id=self.coherence_prompt_id,
                structured_llm=get_coherence_llm,
                usage_collector=resolved_usage_collector,
            )
            score = _clamp_score(judgment.score)
            coherence_scores.append(score)
            if judgment.supported_by_guidance:
                supported_count += 1
            cluster_judgments.append(
                {
                    "cluster_name": cluster_name,
                    "size": len(members),
                    "score": score,
                    "supported_by_guidance": judgment.supported_by_guidance,
                    "reasoning": judgment.reasoning,
                }
            )
            if progress_callback is not None:
                progress_callback(
                    "coherence "
                    f"{coherence_index}/{len(clusters)} for cluster '{cluster_name}' finished"
                )

        distinctiveness_pairs: list[dict[str, Any]] = []
        distinctiveness_scores: list[float] = []
        total_distinctiveness_pairs = (len(clusters) * (len(clusters) - 1)) // 2
        distinctiveness_index = 0
        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                left_cluster_name, left_members = clusters[left_index]
                right_cluster_name, right_members = clusters[right_index]
                distinctiveness_index += 1
                if progress_callback is not None:
                    progress_callback(
                        "distinctiveness "
                        f"{distinctiveness_index}/{total_distinctiveness_pairs} for "
                        f"'{left_cluster_name}' vs '{right_cluster_name}' started"
                    )
                judgment = judge_cluster_distinctiveness(
                    guidance=guidance,
                    left_cluster_name=left_cluster_name,
                    left_cluster_members="\n".join(f"- {member}" for member in left_members),
                    right_cluster_name=right_cluster_name,
                    right_cluster_members="\n".join(f"- {member}" for member in right_members),
                    model_config=self.model_config,
                    prompt_id=self.distinctiveness_prompt_id,
                    structured_llm=get_distinctiveness_llm,
                    usage_collector=resolved_usage_collector,
                )
                score = _clamp_score(judgment.score)
                distinctiveness_scores.append(score)
                distinctiveness_pairs.append(
                    {
                        "left_cluster_name": left_cluster_name,
                        "right_cluster_name": right_cluster_name,
                        "score": score,
                        "overlapping_or_redundant": judgment.overlapping_or_redundant,
                        "reasoning": judgment.reasoning,
                    }
                )
                if progress_callback is not None:
                    progress_callback(
                        "distinctiveness "
                        f"{distinctiveness_index}/{total_distinctiveness_pairs} for "
                        f"'{left_cluster_name}' vs '{right_cluster_name}' finished"
                    )

        return EvaluationResult(
            method=self.name,
            metrics={
                "average_cluster_coherence": _mean(coherence_scores) / 10.0,
                "cluster_distinctiveness": (
                    _mean(distinctiveness_scores) / 10.0 if distinctiveness_scores else 0.0
                ),
                "supported_cluster_rate": _safe_ratio(
                    supported_count,
                    len(cluster_judgments),
                    default=0.0,
                ),
            },
            details={
                "guidance": guidance,
                "cluster_judgments": cluster_judgments,
                "distinctiveness_pairs": distinctiveness_pairs,
            },
            accounting=resolved_usage_collector.snapshot(
                pricer=build_pricing_client_from_configs(
                    [self.model_config]
                    if self.model_config
                    else [default_semantic_consistency_judge_model_config()]
                )
            ).to_dict(),
        )


def _parse_optional_model_config(raw_model_config: object) -> ModelConfig | None:
    if raw_model_config is None:
        return None
    if not isinstance(raw_model_config, Mapping):
        raise TypeError("model_config must be a dict")
    try:
        return parse_model_config(cast(Mapping[str, object], raw_model_config))
    except ValueError as exc:
        raise ValueError(f"'model_config' is invalid: {exc}") from exc


def _resolve_actual_guidance(artifacts: Mapping[str, Any]) -> str:
    selected_topic = artifacts.get("selected_topic")
    if isinstance(selected_topic, Mapping):
        for key in ("guidance", "label", "topic_id"):
            raw_value = selected_topic.get(key)
            if isinstance(raw_value, str) and raw_value.strip():
                return raw_value.strip()
    batch_run_config = artifacts.get("batch_run_config")
    if isinstance(batch_run_config, Mapping):
        human_agent_config = batch_run_config.get("human_agent_config")
        if isinstance(human_agent_config, Mapping):
            raw_guidance = human_agent_config.get("guidance")
            if isinstance(raw_guidance, str) and raw_guidance.strip():
                return raw_guidance.strip()
        metadata = batch_run_config.get("metadata")
        if isinstance(metadata, Mapping):
            raw_guidance = metadata.get("topic_branch_topic_guidance")
            if isinstance(raw_guidance, str) and raw_guidance.strip():
                return raw_guidance.strip()
    return ""


def _clusters_from_predictions(predictions: LoadedClustering) -> list[tuple[str, list[str]]]:
    clusters: list[list[str]] = [[] for _ in predictions.cluster_names]
    for item_text, cluster_id in predictions.assignment.items():
        if 0 <= cluster_id < len(clusters):
            clusters[cluster_id].append(item_text)
    return [
        (cluster_name, members)
        for cluster_name, members in zip(predictions.cluster_names, clusters, strict=True)
        if members
    ]


def _clamp_score(value: float) -> float:
    return max(0.0, min(10.0, float(value)))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_ratio(numerator: float, denominator: float, *, default: float) -> float:
    if denominator == 0:
        return default
    return numerator / denominator
