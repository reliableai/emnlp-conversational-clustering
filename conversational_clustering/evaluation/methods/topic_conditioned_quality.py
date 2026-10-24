"""Topic-conditioned clustering quality evaluator against benchmark references."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from conversational_clustering.accounting import UsageCollector, build_pricing_client_from_configs
from conversational_clustering.evaluation.base import EvaluationRequest, EvaluationResult, Evaluator
from conversational_clustering.evaluation.benchmark_spec import (
    SyntheticBenchmarkSpec,
)
from conversational_clustering.evaluation.methods.benchmark_theme_resolution import (
    ResolvedBenchmarkReference,
    build_theme_resolution_details,
    resolve_reference_for_request,
)
from conversational_clustering.evaluation.methods.benchmark_utils import require_benchmark_spec
from conversational_clustering.evaluation.methods.semantic_quality_judges import judge_theme_match
from conversational_clustering.models import (
    ModelConfig,
    default_semantic_consistency_judge_model_config,
    parse_model_config,
)
from conversational_clustering.prompts.registry import get_discovery_semantic_theme_match_prompt


@dataclass(frozen=True)
class TopicConditionedQualityEvaluator(Evaluator):
    name = "topic_conditioned_quality"
    resume_version = "v2"
    granularity: str | None = None
    model_config: ModelConfig | None = None
    theme_match_prompt_id: str = "baseline_v1"

    @classmethod
    def from_params(
        cls, params: Mapping[str, Any] | None = None
    ) -> TopicConditionedQualityEvaluator:
        normalized_params = dict(params or {})
        supported_keys = {"granularity", "model_config", "theme_match_prompt_id"}
        unsupported_keys = sorted(set(normalized_params) - supported_keys)
        if unsupported_keys:
            raise ValueError(
                f"Evaluator '{cls.name}' received unsupported parameter(s): "
                + ", ".join(unsupported_keys)
            )
        raw_granularity = normalized_params.get("granularity")
        if raw_granularity is not None and (
            not isinstance(raw_granularity, str) or not raw_granularity.strip()
        ):
            raise ValueError("'granularity' must be a non-empty string when provided")
        theme_match_prompt_id = str(
            normalized_params.get("theme_match_prompt_id", "baseline_v1")
        ).strip()
        try:
            get_discovery_semantic_theme_match_prompt(theme_match_prompt_id)
        except ValueError as exc:
            raise ValueError(f"Invalid theme_match_prompt_id: {exc}") from exc
        return cls(
            granularity=raw_granularity.strip() if isinstance(raw_granularity, str) else None,
            model_config=_parse_optional_model_config(normalized_params.get("model_config")),
            theme_match_prompt_id=theme_match_prompt_id,
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        usage_collector = UsageCollector()
        benchmark_spec = require_benchmark_spec(request.artifacts, evaluator_name=self.name)
        resolution = _resolve_reference_for_request(
            predictions=request.predictions,
            artifacts=request.artifacts,
            benchmark_spec=benchmark_spec,
            granularity=self.granularity,
            evaluator_name=self.name,
            model_config=self.model_config,
            theme_match_prompt_id=self.theme_match_prompt_id,
            usage_collector=usage_collector,
        )
        return EvaluationResult(
            method=self.name,
            metrics={
                "cond_ari": resolution.comparison_metrics["adjusted_rand_index"],
                "cond_nmi": resolution.comparison_metrics["normalized_mutual_info"],
                "cond_vi": resolution.comparison_metrics["variation_of_information"],
            },
            details={
                "topic_id": resolution.theme.theme_id,
                "topic_label": resolution.theme.label,
                "reference_id": resolution.reference.reference_id,
                "granularity": resolution.reference.granularity,
                "requested_granularity": self.granularity,
                **build_theme_resolution_details(
                    artifacts=request.artifacts,
                    resolved_theme=resolution.theme,
                    resolution_mode=resolution.resolution_mode,
                ),
                **resolution.comparison_details,
            },
            accounting=usage_collector.snapshot(
                pricer=build_pricing_client_from_configs(
                    [self.model_config]
                    if self.model_config
                    else [default_semantic_consistency_judge_model_config()]
                )
            ).to_dict(),
        )


_ResolvedTopicConditionedReference = ResolvedBenchmarkReference


def _resolve_reference_for_request(
    *,
    predictions: Any,
    artifacts: Mapping[str, Any],
    benchmark_spec: SyntheticBenchmarkSpec,
    granularity: str | None,
    evaluator_name: str,
    model_config: ModelConfig | None,
    theme_match_prompt_id: str,
    usage_collector: UsageCollector | None,
) -> _ResolvedTopicConditionedReference:
    return resolve_reference_for_request(
        predictions=predictions,
        artifacts=artifacts,
        benchmark_spec=benchmark_spec,
        granularity=granularity,
        evaluator_name=evaluator_name,
        model_config=model_config,
        theme_match_prompt_id=theme_match_prompt_id,
        theme_matcher=judge_theme_match,
        usage_collector=usage_collector,
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
