"""Configuration schema and parsing helpers for the backend_v1 clustering backend."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from ....models import ModelConfig, parse_model_config
from ..base import ClusteringAgentBackendConfig
from .prompts import get_backend_v1_system_prompt

BACKEND_V1_IDENTIFIER = "backend_v1"

DEFAULT_CLUSTERING_AGENT_PROMPT_ID = "baseline_v2"

_BACKEND_V1_BACKEND_CONFIG_FIELDS = {
    "intermittent_clustering_judge_model",
    "prompt_id",
    "model_config",
}


def default_backend_v1_intermittent_clustering_judge_model() -> ModelConfig:
    return ModelConfig(
        model="openai/gpt-5.4",
        allow_fallbacks=True,
    )


@dataclass(frozen=True)
class BackendV1ClusteringBackendConfig(ClusteringAgentBackendConfig):
    prompt_id: str = DEFAULT_CLUSTERING_AGENT_PROMPT_ID
    model_config: ModelConfig | None = None
    intermittent_clustering_judge_model: ModelConfig = field(
        default_factory=default_backend_v1_intermittent_clustering_judge_model
    )

    @property
    def backend(self) -> str:
        return BACKEND_V1_IDENTIFIER

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_id, str) or not self.prompt_id.strip():
            raise ValueError("prompt_id must be a non-empty string")
        if self.prompt_id != self.prompt_id.strip():
            raise ValueError("prompt_id must not include leading/trailing whitespace")
        get_backend_v1_system_prompt(self.prompt_id)
        if self.model_config is not None and not isinstance(self.model_config, ModelConfig):
            raise ValueError("model_config must be a ModelConfig or null")
        if not isinstance(self.intermittent_clustering_judge_model, ModelConfig):
            raise ValueError("intermittent_clustering_judge_model must be a ModelConfig")


def _parse_backend_v1_model_config(raw_model_config: Mapping[str, object]) -> ModelConfig:
    try:
        return parse_model_config(raw_model_config)
    except ValueError as exc:
        raise ValueError(f"'model_config' is invalid: {exc}") from exc


def coerce_backend_v1_backend_config(
    backend_config: BackendV1ClusteringBackendConfig | Mapping[str, object] | None,
) -> BackendV1ClusteringBackendConfig:
    if backend_config is None:
        return BackendV1ClusteringBackendConfig()
    if isinstance(backend_config, BackendV1ClusteringBackendConfig):
        return backend_config
    if not isinstance(backend_config, Mapping):
        raise TypeError("backend_config must be either a mapping, typed backend config, or null")

    invalid_backend_config_keys = [key for key in backend_config if not isinstance(key, str)]
    if invalid_backend_config_keys:
        raise ValueError("backend_config must only use string keys")

    raw_backend_config = cast(dict[str, object], dict(backend_config))
    unknown_fields = sorted(set(raw_backend_config) - _BACKEND_V1_BACKEND_CONFIG_FIELDS)
    if unknown_fields:
        raise ValueError(
            "Unknown field(s) in clustering agent backend config: " + ", ".join(unknown_fields)
        )

    prompt_id = raw_backend_config.get("prompt_id", DEFAULT_CLUSTERING_AGENT_PROMPT_ID)
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise ValueError("'prompt_id' must be a non-empty string")
    if prompt_id != prompt_id.strip():
        raise ValueError("'prompt_id' must not include leading/trailing whitespace")
    get_backend_v1_system_prompt(prompt_id)

    model_config_payload = raw_backend_config.get("model_config")
    model_config: ModelConfig | None
    if model_config_payload is None:
        model_config = None
    elif isinstance(model_config_payload, ModelConfig):
        model_config = model_config_payload
    elif isinstance(model_config_payload, Mapping):
        model_config = _parse_backend_v1_model_config(
            cast(Mapping[str, object], model_config_payload)
        )
    else:
        raise ValueError("'model_config' must be either a model config table or null")

    intermittent_judge_model_payload = raw_backend_config.get("intermittent_clustering_judge_model")
    intermittent_clustering_judge_model: ModelConfig
    if intermittent_judge_model_payload is None:
        intermittent_clustering_judge_model = (
            default_backend_v1_intermittent_clustering_judge_model()
        )
    elif isinstance(intermittent_judge_model_payload, ModelConfig):
        intermittent_clustering_judge_model = intermittent_judge_model_payload
    elif isinstance(intermittent_judge_model_payload, Mapping):
        intermittent_clustering_judge_model = _parse_backend_v1_model_config(
            cast(Mapping[str, object], intermittent_judge_model_payload)
        )
    else:
        raise ValueError("'intermittent_clustering_judge_model' must be a model config table")

    return BackendV1ClusteringBackendConfig(
        prompt_id=prompt_id,
        model_config=model_config,
        intermittent_clustering_judge_model=intermittent_clustering_judge_model,
    )
