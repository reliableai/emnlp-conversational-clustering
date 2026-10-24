"""LLM-based guidance inference and similarity judging for semantic consistency evaluation."""

from pydantic import BaseModel, Field

from ....accounting import UsageCollector, extract_response_model, extract_token_usage
from ....models import (
    ModelConfig,
    build_semantic_consistency_inference_llm,
    build_semantic_consistency_judge_llm,
    default_semantic_consistency_inference_model_config,
    default_semantic_consistency_judge_model_config,
)
from ....prompts import (
    get_semantic_consistency_inference_prompt,
    get_semantic_consistency_judge_prompt,
)


class SimilarityScore(BaseModel):
    score: float = Field(description="Similarity score between 0 and 10")
    reasoning: str = Field(description="Brief explanation of the score")


def infer_guidance_from_clustering(
    clustering_text: str,
    model_config: ModelConfig | None = None,
    inference_prompt_id: str = "baseline_v1",
    usage_collector: UsageCollector | None = None,
) -> str:
    llm = build_semantic_consistency_inference_llm(model_config)

    prompt_template = get_semantic_consistency_inference_prompt(inference_prompt_id)
    prompt = prompt_template.format(clustering=clustering_text)

    response = llm.invoke(prompt)
    if usage_collector is not None:
        fallback_model = (
            model_config.model
            if model_config is not None
            else default_semantic_consistency_inference_model_config().model
        )
        usage_collector.record(
            extract_response_model(response, fallback_model=fallback_model),
            extract_token_usage(response),
            source="semantic_consistency_inference",
        )
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(str(item) for item in content).strip()
    raise TypeError(
        f"Unexpected LLM response content type: {type(content).__name__}. "
        f"Expected str or list, got {type(content)}."
    )


def judge_similarity(
    actual_guidance: str,
    inferred_guidance: str,
    model_config: ModelConfig | None = None,
    judge_prompt_id: str = "baseline_v1",
    usage_collector: UsageCollector | None = None,
) -> float:
    llm = build_semantic_consistency_judge_llm(model_config)
    llm.temperature = 0.0

    structured_llm = llm.with_structured_output(SimilarityScore, include_raw=True)

    prompt_template = get_semantic_consistency_judge_prompt(judge_prompt_id)
    prompt = prompt_template.format(
        actual_guidance=actual_guidance,
        inferred_guidance=inferred_guidance,
    )

    response = structured_llm.invoke(prompt)
    if isinstance(response, dict):
        raw_result = response.get("raw")
        if raw_result is not None and usage_collector is not None:
            fallback_model = (
                model_config.model
                if model_config is not None
                else default_semantic_consistency_judge_model_config().model
            )
            usage_collector.record(
                extract_response_model(raw_result, fallback_model=fallback_model),
                extract_token_usage(raw_result),
                source="semantic_consistency_judge",
            )
        result = response.get("parsed")
    else:
        result = response
    if not isinstance(result, SimilarityScore):
        raise TypeError("Semantic consistency judge returned an unexpected payload type.")
    score = max(0.0, min(10.0, result.score))
    return score


def calculate_variance(scores: list[float]) -> float:
    if not scores:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
    return variance
