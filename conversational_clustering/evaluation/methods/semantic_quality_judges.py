"""Structured LLM judges for evaluators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, Field, ValidationError

from ...accounting import UsageCollector, extract_response_model, extract_token_usage
from ...models import (
    ModelConfig,
    build_semantic_consistency_judge_llm,
    default_semantic_consistency_judge_model_config,
)
from ...prompts import (
    get_clarification_policy_prompt,
    get_cluster_semantic_coherence_prompt,
    get_cluster_semantic_distinctiveness_prompt,
    get_discovery_semantic_duplicate_prompt,
    get_discovery_semantic_theme_match_prompt,
    get_null_theme_abstention_topic_match_judge_prompt,
    get_null_theme_abstention_trace_judge_prompt,
)


class ThemeMatchJudgment(BaseModel):
    supported: bool = Field(description="Whether the surfaced topic maps to a supported theme")
    benchmark_theme_id: str | None = Field(
        default=None,
        description="Matching benchmark theme id when supported, else null",
    )
    confidence: float = Field(description="Confidence from 0 to 1")
    reasoning: str = Field(description="Brief explanation")


class DuplicateTopicJudgment(BaseModel):
    duplicate: bool = Field(description="Whether the topics are semantically redundant")
    confidence: float = Field(description="Confidence from 0 to 1")
    reasoning: str = Field(description="Brief explanation")


class NullThemeAbstentionJudgment(BaseModel):
    rejected_requested_pov: bool = Field(
        description="Whether the assistant rejected the user's requested unsupported POV"
    )
    assistant_agreed_with_requested_pov: bool = Field(
        description="Whether the assistant agreed with or reinforced the unsupported POV"
    )
    confidence: float = Field(description="Confidence from 0 to 1")
    reasoning: str = Field(description="Brief explanation")


class NullThemeUnsupportedTopicMatchJudgment(BaseModel):
    matched_unsupported_topic_id: str | None = Field(
        default=None,
        description="Configured unsupported topic id matching the initial requested theme",
    )
    confidence: float = Field(description="Confidence from 0 to 1")
    reasoning: str = Field(description="Brief explanation")


class ClusterCoherenceJudgment(BaseModel):
    score: float = Field(description="Semantic coherence score from 0 to 10")
    supported_by_guidance: bool = Field(
        description="Whether the cluster is meaningfully aligned with the guidance"
    )
    reasoning: str = Field(description="Brief explanation")


class ClusterDistinctivenessJudgment(BaseModel):
    score: float = Field(description="Distinctiveness score from 0 to 10")
    overlapping_or_redundant: bool = Field(
        description="Whether the two clusters semantically overlap or are redundant"
    )
    reasoning: str = Field(description="Brief explanation")


class ClarificationOpportunityJudgment(BaseModel):
    opportunity_exists: bool = Field(
        description="Whether this request turn presented a real clarification opportunity"
    )
    clarification_needed: bool = Field(
        description="Whether a clarification question should have been asked"
    )
    asked_clarification: bool = Field(
        description="Whether the assistant asked a clarification question for this request"
    )
    useful_clarification: bool = Field(
        description="Whether the clarification materially improved the downstream outcome"
    )
    unnecessary_clarification: bool = Field(
        description="Whether the clarification question could have been avoided"
    )
    missed_clarification: bool = Field(description="Whether clarification was needed but not asked")
    downstream_gain: float = Field(
        ge=0.0, le=1.0, description="Improvement attributable to clarification from 0 to 1"
    )
    reasoning: str = Field(description="Brief explanation")


def build_structured_semantic_judge(
    *, schema: type[BaseModel], model_config: ModelConfig | None = None
) -> Any:
    llm = build_semantic_consistency_judge_llm(model_config)
    llm.temperature = 0.0
    return llm.with_structured_output(schema, include_raw=True)


def _resolve_structured_llm(structured_llm: Any | Callable[[], Any] | None) -> Any:
    if structured_llm is None:
        return None
    if callable(structured_llm):
        return structured_llm()
    return structured_llm


_TStructuredJudgment = TypeVar("_TStructuredJudgment", bound=BaseModel)
_MAX_INVALID_STRUCTURED_OUTPUT_ATTEMPTS = 2


def _is_retryable_structured_output_error(exc: Exception) -> bool:
    return isinstance(exc, ValidationError) or exc.__class__.__name__ == "OutputParserException"


def _coerce_structured_judgment(
    *, schema: type[_TStructuredJudgment], result: object
) -> _TStructuredJudgment | None:
    if isinstance(result, schema):
        return result
    if isinstance(result, BaseModel):
        payload: object = result.model_dump()
    else:
        payload = result
    try:
        return schema.model_validate(payload)
    except ValidationError:
        return None


def _invoke_structured_judge(
    *,
    schema: type[_TStructuredJudgment],
    prompt: str,
    model_config: ModelConfig | None,
    structured_llm: Any | Callable[[], Any] | None,
    judge_name: str,
    usage_collector: UsageCollector | None = None,
    usage_source: str,
) -> _TStructuredJudgment:
    last_error: Exception | None = None
    last_result: object | None = None
    for _ in range(_MAX_INVALID_STRUCTURED_OUTPUT_ATTEMPTS):
        resolved_llm = _resolve_structured_llm(structured_llm)
        if resolved_llm is None:
            resolved_llm = build_structured_semantic_judge(
                schema=schema,
                model_config=model_config,
            )
        try:
            result = resolved_llm.invoke(prompt)
        except Exception as exc:
            if not _is_retryable_structured_output_error(exc):
                raise
            last_error = exc
            last_result = None
            continue
        parse_target = result
        if isinstance(result, dict):
            raw_result = result.get("raw")
            if raw_result is not None and usage_collector is not None:
                fallback_model = (
                    model_config.model
                    if model_config is not None
                    else default_semantic_consistency_judge_model_config().model
                )
                usage_collector.record(
                    extract_response_model(raw_result, fallback_model=fallback_model),
                    extract_token_usage(raw_result),
                    source=usage_source,
                )
            parse_target = result.get("parsed")
        parsed = _coerce_structured_judgment(schema=schema, result=parse_target)
        if parsed is not None:
            return parsed
        last_error = None
        last_result = result

    if last_error is not None:
        raise ValueError(
            f"{judge_name} failed to parse structured output after "
            f"{_MAX_INVALID_STRUCTURED_OUTPUT_ATTEMPTS} attempt(s). "
            f"Expected {schema.__name__}."
        ) from last_error
    result_type = type(last_result).__name__ if last_result is not None else "NoneType"
    raise ValueError(
        f"{judge_name} returned invalid structured output of type '{result_type}' "
        f"after {_MAX_INVALID_STRUCTURED_OUTPUT_ATTEMPTS} attempt(s). "
        f"Expected {schema.__name__}."
    )


def judge_theme_match(
    *,
    surfaced_topic: str,
    benchmark_themes: str,
    model_config: ModelConfig | None = None,
    prompt_id: str = "baseline_v1",
    structured_llm: Any | Callable[[], Any] | None = None,
    usage_collector: UsageCollector | None = None,
) -> ThemeMatchJudgment:
    prompt_template = get_discovery_semantic_theme_match_prompt(prompt_id)
    return _invoke_structured_judge(
        schema=ThemeMatchJudgment,
        prompt=prompt_template.format(
            surfaced_topic=surfaced_topic,
            benchmark_themes=benchmark_themes,
        ),
        model_config=model_config,
        structured_llm=structured_llm,
        judge_name="Theme match judge",
        usage_collector=usage_collector,
        usage_source="judge_theme_match",
    )


def judge_duplicate_topics(
    *,
    left_topic: str,
    right_topic: str,
    model_config: ModelConfig | None = None,
    prompt_id: str = "baseline_v1",
    structured_llm: Any | Callable[[], Any] | None = None,
    usage_collector: UsageCollector | None = None,
) -> DuplicateTopicJudgment:
    prompt_template = get_discovery_semantic_duplicate_prompt(prompt_id)
    return _invoke_structured_judge(
        schema=DuplicateTopicJudgment,
        prompt=prompt_template.format(left_topic=left_topic, right_topic=right_topic),
        model_config=model_config,
        structured_llm=structured_llm,
        judge_name="Duplicate-topic judge",
        usage_collector=usage_collector,
        usage_source="judge_duplicate_topics",
    )


def judge_null_theme_abstention_trace(
    *,
    requested_pov: str,
    matched_unsupported_topic: str,
    conversation_trace: str,
    model_config: ModelConfig | None = None,
    prompt_id: str = "baseline_v1",
    structured_llm: Any | Callable[[], Any] | None = None,
    usage_collector: UsageCollector | None = None,
) -> NullThemeAbstentionJudgment:
    prompt_template = get_null_theme_abstention_trace_judge_prompt(prompt_id)
    return _invoke_structured_judge(
        schema=NullThemeAbstentionJudgment,
        prompt=prompt_template.format(
            requested_pov=requested_pov,
            matched_unsupported_topic=matched_unsupported_topic,
            conversation_trace=conversation_trace,
        ),
        model_config=model_config,
        structured_llm=structured_llm,
        judge_name="Null-theme abstention judge",
        usage_collector=usage_collector,
        usage_source="judge_null_theme_abstention_trace",
    )


def judge_null_theme_unsupported_topic_match(
    *,
    requested_guidance: str,
    unsupported_topics: str,
    model_config: ModelConfig | None = None,
    prompt_id: str = "baseline_v1",
    structured_llm: Any | Callable[[], Any] | None = None,
    usage_collector: UsageCollector | None = None,
) -> NullThemeUnsupportedTopicMatchJudgment:
    prompt_template = get_null_theme_abstention_topic_match_judge_prompt(prompt_id)
    return _invoke_structured_judge(
        schema=NullThemeUnsupportedTopicMatchJudgment,
        prompt=prompt_template.format(
            requested_guidance=requested_guidance,
            unsupported_topics=unsupported_topics,
        ),
        model_config=model_config,
        structured_llm=structured_llm,
        judge_name="Null-theme unsupported-topic match judge",
        usage_collector=usage_collector,
        usage_source="judge_null_theme_unsupported_topic_match",
    )


def judge_cluster_coherence(
    *,
    guidance: str,
    cluster_name: str,
    cluster_members: str,
    model_config: ModelConfig | None = None,
    prompt_id: str = "baseline_v1",
    structured_llm: Any | Callable[[], Any] | None = None,
    usage_collector: UsageCollector | None = None,
) -> ClusterCoherenceJudgment:
    prompt_template = get_cluster_semantic_coherence_prompt(prompt_id)
    return _invoke_structured_judge(
        schema=ClusterCoherenceJudgment,
        prompt=prompt_template.format(
            guidance=guidance,
            cluster_name=cluster_name,
            cluster_members=cluster_members,
        ),
        model_config=model_config,
        structured_llm=structured_llm,
        judge_name="Cluster-coherence judge",
        usage_collector=usage_collector,
        usage_source="judge_cluster_coherence",
    )


def judge_cluster_distinctiveness(
    *,
    guidance: str,
    left_cluster_name: str,
    left_cluster_members: str,
    right_cluster_name: str,
    right_cluster_members: str,
    model_config: ModelConfig | None = None,
    prompt_id: str = "baseline_v1",
    structured_llm: Any | Callable[[], Any] | None = None,
    usage_collector: UsageCollector | None = None,
) -> ClusterDistinctivenessJudgment:
    prompt_template = get_cluster_semantic_distinctiveness_prompt(prompt_id)
    return _invoke_structured_judge(
        schema=ClusterDistinctivenessJudgment,
        prompt=prompt_template.format(
            guidance=guidance,
            left_cluster_name=left_cluster_name,
            left_cluster_members=left_cluster_members,
            right_cluster_name=right_cluster_name,
            right_cluster_members=right_cluster_members,
        ),
        model_config=model_config,
        structured_llm=structured_llm,
        judge_name="Cluster-distinctiveness judge",
        usage_collector=usage_collector,
        usage_source="judge_cluster_distinctiveness",
    )


def judge_clarification_opportunity(
    *,
    request_turn_index: int,
    request_text: str,
    conversation_trace: str,
    benchmark_context: str,
    final_clustering: str,
    model_config: ModelConfig | None = None,
    prompt_id: str = "baseline_v1",
    structured_llm: Any | Callable[[], Any] | None = None,
    usage_collector: UsageCollector | None = None,
) -> ClarificationOpportunityJudgment:
    prompt_template = get_clarification_policy_prompt(prompt_id)
    return _invoke_structured_judge(
        schema=ClarificationOpportunityJudgment,
        prompt=prompt_template.format(
            request_turn_index=request_turn_index,
            request_text=request_text,
            conversation_trace=conversation_trace,
            benchmark_context=benchmark_context,
            final_clustering=final_clustering,
        ),
        model_config=model_config,
        structured_llm=structured_llm,
        judge_name="Clarification-policy judge",
        usage_collector=usage_collector,
        usage_source="judge_clarification_opportunity",
    )
