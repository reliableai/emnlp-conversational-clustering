"""Configuration schema and parsing helpers for the instruction_kmeans_v1 backend."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from ....models import (
    EMBEDDING_BACKEND_HUGGINGFACE_LOCAL,
    EMBEDDING_BACKEND_OPENROUTER,
    EmbeddingModelConfig,
    ModelConfig,
    parse_embedding_model_config,
    parse_model_config,
)
from ..base import ClusteringAgentBackendConfig
from .prompts import get_instruction_kmeans_v1_system_prompt

INSTRUCTION_KMEANS_V1_IDENTIFIER = "instruction_kmeans_v1"
DEFAULT_INSTRUCTION_KMEANS_V1_PROMPT_ID = "baseline_v1"
_INSTRUCTION_KMEANS_V1_LOCAL_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
_INSTRUCTION_KMEANS_V1_OPENROUTER_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
INSTRUCTION_KMEANS_V1_SUPPORTED_EMBEDDING_MODELS = [
    _INSTRUCTION_KMEANS_V1_LOCAL_EMBEDDING_MODEL,
    _INSTRUCTION_KMEANS_V1_OPENROUTER_EMBEDDING_MODEL,
]

_INSTRUCTION_KMEANS_V1_BACKEND_CONFIG_FIELDS = {
    "prompt_id",
    "model_config",
    "embedding_model_config",
    "summary_model_config",
    "intermittent_clustering_judge_model",
    "sample_window_size",
    "shuffle_seed",
    "cluster_summary_example_limit",
    "embedding_batch_size",
}


def default_instruction_kmeans_v1_model_config() -> ModelConfig:
    return ModelConfig(
        model="openai/gpt-5.4",
        allow_fallbacks=True,
    )


def default_instruction_kmeans_v1_embedding_model_config() -> EmbeddingModelConfig:
    return EmbeddingModelConfig(
        model=_INSTRUCTION_KMEANS_V1_LOCAL_EMBEDDING_MODEL,
        backend=EMBEDDING_BACKEND_HUGGINGFACE_LOCAL,
    )


def default_instruction_kmeans_v1_openrouter_embedding_model_config() -> EmbeddingModelConfig:
    return EmbeddingModelConfig(
        model=_INSTRUCTION_KMEANS_V1_OPENROUTER_EMBEDDING_MODEL,
        backend=EMBEDDING_BACKEND_OPENROUTER,
    )


def default_instruction_kmeans_v1_summary_model_config() -> ModelConfig:
    return ModelConfig(
        model="openai/gpt-5.4-nano",
        allow_fallbacks=True,
    )


def default_instruction_kmeans_v1_intermittent_clustering_judge_model() -> ModelConfig:
    return ModelConfig(
        model="openai/gpt-5.4",
        allow_fallbacks=True,
    )


def _validate_instruction_kmeans_v1_embedding_model_config(
    embedding_model_config: EmbeddingModelConfig,
) -> None:
    supported_models_by_backend = {
        EMBEDDING_BACKEND_HUGGINGFACE_LOCAL: [_INSTRUCTION_KMEANS_V1_LOCAL_EMBEDDING_MODEL],
        EMBEDDING_BACKEND_OPENROUTER: [_INSTRUCTION_KMEANS_V1_OPENROUTER_EMBEDDING_MODEL],
    }
    supported_models = supported_models_by_backend.get(embedding_model_config.backend)
    if supported_models is None:
        raise ValueError(
            "instruction_kmeans_v1 only supports embedding backends [huggingface_local, openrouter]"
        )
    if embedding_model_config.model in supported_models:
        return
    supported_models_text = ", ".join(supported_models)
    raise ValueError(
        "instruction_kmeans_v1 only supports embedding_model_config.model values from "
        f"[{supported_models_text}] when embedding_model_config.backend="
        f"'{embedding_model_config.backend}'"
    )


@dataclass(frozen=True)
class InstructionKMeansV1ClusteringBackendConfig(ClusteringAgentBackendConfig):
    prompt_id: str = DEFAULT_INSTRUCTION_KMEANS_V1_PROMPT_ID
    model_config: ModelConfig = field(default_factory=default_instruction_kmeans_v1_model_config)
    embedding_model_config: EmbeddingModelConfig = field(
        default_factory=default_instruction_kmeans_v1_embedding_model_config
    )
    summary_model_config: ModelConfig = field(
        default_factory=default_instruction_kmeans_v1_summary_model_config
    )
    intermittent_clustering_judge_model: ModelConfig = field(
        default_factory=default_instruction_kmeans_v1_intermittent_clustering_judge_model
    )
    sample_window_size: int = 200
    shuffle_seed: int = 42
    cluster_summary_example_limit: int = 15
    embedding_batch_size: int = 64

    @property
    def backend(self) -> str:
        return INSTRUCTION_KMEANS_V1_IDENTIFIER

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_id, str) or not self.prompt_id.strip():
            raise ValueError("prompt_id must be a non-empty string")
        if self.prompt_id != self.prompt_id.strip():
            raise ValueError("prompt_id must not include leading/trailing whitespace")
        get_instruction_kmeans_v1_system_prompt(self.prompt_id)
        if not isinstance(self.model_config, ModelConfig):
            raise ValueError("model_config must be a ModelConfig")
        if not isinstance(self.embedding_model_config, EmbeddingModelConfig):
            raise ValueError("embedding_model_config must be an EmbeddingModelConfig")
        _validate_instruction_kmeans_v1_embedding_model_config(self.embedding_model_config)
        if not isinstance(self.summary_model_config, ModelConfig):
            raise ValueError("summary_model_config must be a ModelConfig")
        if not isinstance(self.intermittent_clustering_judge_model, ModelConfig):
            raise ValueError("intermittent_clustering_judge_model must be a ModelConfig")
        if (
            isinstance(self.sample_window_size, bool)
            or not isinstance(self.sample_window_size, int)
            or self.sample_window_size < 1
        ):
            raise ValueError("sample_window_size must be an integer >= 1")
        if (
            isinstance(self.shuffle_seed, bool)
            or not isinstance(self.shuffle_seed, int)
            or self.shuffle_seed < 0
        ):
            raise ValueError("shuffle_seed must be an integer >= 0")
        if (
            isinstance(self.cluster_summary_example_limit, bool)
            or not isinstance(self.cluster_summary_example_limit, int)
            or self.cluster_summary_example_limit < 1
        ):
            raise ValueError("cluster_summary_example_limit must be an integer >= 1")
        if (
            isinstance(self.embedding_batch_size, bool)
            or not isinstance(self.embedding_batch_size, int)
            or self.embedding_batch_size < 1
        ):
            raise ValueError("embedding_batch_size must be an integer >= 1")


def _parse_model_config_field(
    raw_model_config: Mapping[str, object],
    *,
    field_name: str,
) -> ModelConfig:
    try:
        return parse_model_config(raw_model_config)
    except ValueError as exc:
        raise ValueError(f"'{field_name}' is invalid: {exc}") from exc


def _parse_embedding_model_config_field(
    raw_model_config: Mapping[str, object],
) -> EmbeddingModelConfig:
    try:
        normalized_raw_model_config = dict(raw_model_config)
        backend = normalized_raw_model_config.get("backend", EMBEDDING_BACKEND_HUGGINGFACE_LOCAL)
        if backend == EMBEDDING_BACKEND_OPENROUTER:
            normalized_raw_model_config.setdefault(
                "model",
                _INSTRUCTION_KMEANS_V1_OPENROUTER_EMBEDDING_MODEL,
            )
        else:
            normalized_raw_model_config.setdefault(
                "model",
                _INSTRUCTION_KMEANS_V1_LOCAL_EMBEDDING_MODEL,
            )
        embedding_model_config = parse_embedding_model_config(normalized_raw_model_config)
        _validate_instruction_kmeans_v1_embedding_model_config(embedding_model_config)
        return embedding_model_config
    except ValueError as exc:
        raise ValueError(f"'embedding_model_config' is invalid: {exc}") from exc


def _parse_positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"'{field_name}' must be an integer >= 1")
    return value


def _parse_non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"'{field_name}' must be an integer >= 0")
    return value


def coerce_instruction_kmeans_v1_backend_config(
    backend_config: InstructionKMeansV1ClusteringBackendConfig | Mapping[str, object] | None,
) -> InstructionKMeansV1ClusteringBackendConfig:
    if backend_config is None:
        return InstructionKMeansV1ClusteringBackendConfig()
    if isinstance(backend_config, InstructionKMeansV1ClusteringBackendConfig):
        return backend_config
    if not isinstance(backend_config, Mapping):
        raise TypeError("backend_config must be either a mapping, typed backend config, or null")

    invalid_backend_config_keys = [key for key in backend_config if not isinstance(key, str)]
    if invalid_backend_config_keys:
        raise ValueError("backend_config must only use string keys")

    raw_backend_config = cast(dict[str, object], dict(backend_config))
    unknown_fields = sorted(set(raw_backend_config) - _INSTRUCTION_KMEANS_V1_BACKEND_CONFIG_FIELDS)
    if unknown_fields:
        raise ValueError(
            "Unknown field(s) in clustering agent backend config: " + ", ".join(unknown_fields)
        )

    prompt_id = raw_backend_config.get("prompt_id", DEFAULT_INSTRUCTION_KMEANS_V1_PROMPT_ID)
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise ValueError("'prompt_id' must be a non-empty string")
    if prompt_id != prompt_id.strip():
        raise ValueError("'prompt_id' must not include leading/trailing whitespace")
    get_instruction_kmeans_v1_system_prompt(prompt_id)

    raw_model_config = raw_backend_config.get("model_config")
    if raw_model_config is None:
        model_config = default_instruction_kmeans_v1_model_config()
    elif isinstance(raw_model_config, ModelConfig):
        model_config = raw_model_config
    elif isinstance(raw_model_config, Mapping):
        model_config = _parse_model_config_field(
            cast(Mapping[str, object], raw_model_config),
            field_name="model_config",
        )
    else:
        raise ValueError("'model_config' must be a model config table")

    raw_embedding_model_config = raw_backend_config.get("embedding_model_config")
    if raw_embedding_model_config is None:
        embedding_model_config = default_instruction_kmeans_v1_embedding_model_config()
    elif isinstance(raw_embedding_model_config, EmbeddingModelConfig):
        embedding_model_config = raw_embedding_model_config
    elif isinstance(raw_embedding_model_config, Mapping):
        embedding_model_config = _parse_embedding_model_config_field(
            cast(Mapping[str, object], raw_embedding_model_config)
        )
    else:
        raise ValueError("'embedding_model_config' must be an embedding model config table")

    raw_summary_model_config = raw_backend_config.get("summary_model_config")
    if raw_summary_model_config is None:
        summary_model_config = default_instruction_kmeans_v1_summary_model_config()
    elif isinstance(raw_summary_model_config, ModelConfig):
        summary_model_config = raw_summary_model_config
    elif isinstance(raw_summary_model_config, Mapping):
        summary_model_config = _parse_model_config_field(
            cast(Mapping[str, object], raw_summary_model_config),
            field_name="summary_model_config",
        )
    else:
        raise ValueError("'summary_model_config' must be a model config table")

    raw_judge_model_config = raw_backend_config.get("intermittent_clustering_judge_model")
    if raw_judge_model_config is None:
        intermittent_clustering_judge_model = (
            default_instruction_kmeans_v1_intermittent_clustering_judge_model()
        )
    elif isinstance(raw_judge_model_config, ModelConfig):
        intermittent_clustering_judge_model = raw_judge_model_config
    elif isinstance(raw_judge_model_config, Mapping):
        intermittent_clustering_judge_model = _parse_model_config_field(
            cast(Mapping[str, object], raw_judge_model_config),
            field_name="intermittent_clustering_judge_model",
        )
    else:
        raise ValueError("'intermittent_clustering_judge_model' must be a model config table")

    sample_window_size = _parse_positive_int(
        raw_backend_config.get("sample_window_size", 200),
        field_name="sample_window_size",
    )
    shuffle_seed = _parse_non_negative_int(
        raw_backend_config.get("shuffle_seed", 42),
        field_name="shuffle_seed",
    )
    cluster_summary_example_limit = _parse_positive_int(
        raw_backend_config.get("cluster_summary_example_limit", 15),
        field_name="cluster_summary_example_limit",
    )
    embedding_batch_size = _parse_positive_int(
        raw_backend_config.get("embedding_batch_size", 64),
        field_name="embedding_batch_size",
    )

    return InstructionKMeansV1ClusteringBackendConfig(
        prompt_id=prompt_id,
        model_config=model_config,
        embedding_model_config=embedding_model_config,
        summary_model_config=summary_model_config,
        intermittent_clustering_judge_model=intermittent_clustering_judge_model,
        sample_window_size=sample_window_size,
        shuffle_seed=shuffle_seed,
        cluster_summary_example_limit=cluster_summary_example_limit,
        embedding_batch_size=embedding_batch_size,
    )
