"""LLM-judged semantic quality evaluator for surfaced discovery topics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from conversational_clustering.accounting import UsageCollector, build_pricing_client_from_configs
from conversational_clustering.evaluation.base import EvaluationRequest, EvaluationResult, Evaluator
from conversational_clustering.evaluation.methods.benchmark_utils import require_benchmark_spec
from conversational_clustering.evaluation.methods.discovery_semantic_quality_shared import (
    DiscoverySemanticQualityConfig,
    evaluate_surfaced_topics_semantic_quality,
    format_discovery_topic,
    parse_discovery_semantic_quality_config,
)
from conversational_clustering.evaluation.methods.semantic_quality_judges import (
    judge_duplicate_topics,
    judge_theme_match,
)
from conversational_clustering.models import (
    ModelConfig,
    default_semantic_consistency_judge_model_config,
)


@dataclass(frozen=True)
class DiscoverySemanticQualityEvaluator(Evaluator):
    name = "discovery_semantic_quality"
    resume_version = "v2"
    model_config: ModelConfig | None = None
    theme_match_prompt_id: str = "baseline_v1"
    duplicate_prompt_id: str = "baseline_v1"

    @classmethod
    def from_params(
        cls, params: Mapping[str, Any] | None = None
    ) -> DiscoverySemanticQualityEvaluator:
        config = parse_discovery_semantic_quality_config(evaluator_name=cls.name, params=params)
        return cls(
            model_config=config.model_config,
            theme_match_prompt_id=config.theme_match_prompt_id,
            duplicate_prompt_id=config.duplicate_prompt_id,
        )

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        usage_collector = UsageCollector()
        benchmark_spec = require_benchmark_spec(request.artifacts, evaluator_name=self.name)
        topic_candidates = request.artifacts.get("topic_candidates")
        if not isinstance(topic_candidates, Mapping):
            raise ValueError(
                f"Evaluator '{self.name}' requires a topic_candidates artifact in the request"
            )
        raw_topics = topic_candidates.get("topics")
        if not isinstance(raw_topics, list):
            raise ValueError("topic_candidates.topics must be a list")

        surfaced_topics = [format_discovery_topic(raw_topic) for raw_topic in raw_topics]
        normalized_topics = [topic for topic in surfaced_topics if topic is not None]
        outcome = evaluate_surfaced_topics_semantic_quality(
            surfaced_topics=normalized_topics,
            benchmark_spec=benchmark_spec,
            config=self._config,
            theme_match_judge=lambda **kwargs: judge_theme_match(
                **kwargs, usage_collector=usage_collector
            ),
            duplicate_judge=lambda **kwargs: judge_duplicate_topics(
                **kwargs, usage_collector=usage_collector
            ),
        )
        return EvaluationResult(
            method=self.name,
            metrics=outcome.metrics,
            details=outcome.details,
            accounting=usage_collector.snapshot(
                pricer=build_pricing_client_from_configs(
                    [self.model_config]
                    if self.model_config
                    else [default_semantic_consistency_judge_model_config()]
                )
            ).to_dict(),
        )

    @property
    def _config(self) -> DiscoverySemanticQualityConfig:
        return DiscoverySemanticQualityConfig(
            model_config=self.model_config,
            theme_match_prompt_id=self.theme_match_prompt_id,
            duplicate_prompt_id=self.duplicate_prompt_id,
        )
