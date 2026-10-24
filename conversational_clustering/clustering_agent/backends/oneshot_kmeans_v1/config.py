"""Configuration schema and parsing helpers for the oneshot_kmeans_v1 clustering backend."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from ....models import ModelConfig, parse_model_config
from ..base import ClusteringAgentBackendConfig

ONESHOT_KMEANS_V1_IDENTIFIER = "oneshot_kmeans_v1"

_ONESHOT_KMEANS_V1_BACKEND_CONFIG_FIELDS = {
    "model_config",
}


def default_oneshot_kmeans_v1_model_config() -> ModelConfig:
    return ModelConfig(
        model="openai/gpt-oss-20b",
        allow_fallbacks=True,
        temperature=0.0,
    )


@dataclass(frozen=True)
class OneShotKMeansV1ClusteringBackendConfig(ClusteringAgentBackendConfig):
    model_config: ModelConfig = field(default_factory=default_oneshot_kmeans_v1_model_config)

    @property
    def backend(self) -> str:
        return ONESHOT_KMEANS_V1_IDENTIFIER

    def __post_init__(self) -> None:
        if not isinstance(self.model_config, ModelConfig):
            raise ValueError("model_config must be a ModelConfig")


def _parse_oneshot_kmeans_v1_model_config(raw_model_config: Mapping[str, object]) -> ModelConfig:
    try:
        return parse_model_config(raw_model_config)
    except ValueError as exc:
        raise ValueError(f"'model_config' is invalid: {exc}") from exc


def coerce_oneshot_kmeans_v1_backend_config(
    backend_config: OneShotKMeansV1ClusteringBackendConfig | Mapping[str, object] | None,
) -> OneShotKMeansV1ClusteringBackendConfig:
    if backend_config is None:
        return OneShotKMeansV1ClusteringBackendConfig()
    if isinstance(backend_config, OneShotKMeansV1ClusteringBackendConfig):
        return backend_config
    if not isinstance(backend_config, Mapping):
        raise TypeError("backend_config must be either a mapping, typed backend config, or null")

    invalid_backend_config_keys = [key for key in backend_config if not isinstance(key, str)]
    if invalid_backend_config_keys:
        raise ValueError("backend_config must only use string keys")

    raw_backend_config = cast(dict[str, object], dict(backend_config))
    unknown_fields = sorted(set(raw_backend_config) - _ONESHOT_KMEANS_V1_BACKEND_CONFIG_FIELDS)
    if unknown_fields:
        raise ValueError(
            "Unknown field(s) in clustering agent backend config: " + ", ".join(unknown_fields)
        )

    model_config_payload = raw_backend_config.get("model_config")
    if model_config_payload is None:
        model_config = default_oneshot_kmeans_v1_model_config()
    elif isinstance(model_config_payload, ModelConfig):
        model_config = model_config_payload
    elif isinstance(model_config_payload, Mapping):
        model_config = _parse_oneshot_kmeans_v1_model_config(
            cast(Mapping[str, object], model_config_payload)
        )
    else:
        raise ValueError("'model_config' must be either a model config table or null")

    return OneShotKMeansV1ClusteringBackendConfig(model_config=model_config)
