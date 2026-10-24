from collections.abc import Mapping
from typing import Any, cast

from ..base import (
    ClusteringAgentBackendConfig,
    ClusteringBackendRegistryEntry,
    ConversationalClusteringMethod,
)
from .config import (
    BACKEND_V1_IDENTIFIER,
    BackendV1ClusteringBackendConfig,
    coerce_backend_v1_backend_config,
    default_backend_v1_intermittent_clustering_judge_model,
)
from .method import BackendV1ConversationalClusteringMethod
from .prompts import get_backend_v1_system_prompt


def _coerce_backend_v1_registry_config(
    backend_config: ClusteringAgentBackendConfig | Mapping[str, object] | None,
) -> ClusteringAgentBackendConfig:
    typed_or_mapping_backend_config: BackendV1ClusteringBackendConfig | Mapping[str, object] | None
    if backend_config is None or isinstance(backend_config, Mapping):
        typed_or_mapping_backend_config = cast(Mapping[str, object] | None, backend_config)
    elif isinstance(backend_config, BackendV1ClusteringBackendConfig):
        typed_or_mapping_backend_config = backend_config
    else:
        raise ValueError(
            f"Typed backend_config is not compatible with backend '{BACKEND_V1_IDENTIFIER}'"
        )
    return coerce_backend_v1_backend_config(backend_config=typed_or_mapping_backend_config)


def _build_backend_v1_registry_method(
    backend_config: ClusteringAgentBackendConfig | None,
) -> ConversationalClusteringMethod:
    resolved_config = _coerce_backend_v1_registry_config(backend_config=backend_config)
    if not isinstance(resolved_config, BackendV1ClusteringBackendConfig):
        raise ValueError(
            f"Typed backend_config is not compatible with backend '{BACKEND_V1_IDENTIFIER}'"
        )
    return BackendV1ConversationalClusteringMethod(config=resolved_config)


def _backend_v1_registry_config_to_dict(
    backend_config: ClusteringAgentBackendConfig | None,
) -> dict[str, Any]:
    resolved_config = _coerce_backend_v1_registry_config(backend_config=backend_config)
    if not isinstance(resolved_config, BackendV1ClusteringBackendConfig):
        raise ValueError(
            f"Typed backend_config is not compatible with backend '{BACKEND_V1_IDENTIFIER}'"
        )
    return {
        "prompt_id": resolved_config.prompt_id,
        "intermittent_clustering_judge_model": (
            resolved_config.intermittent_clustering_judge_model.__dict__
        ),
        "model_config": (
            resolved_config.model_config.__dict__ if resolved_config.model_config else None
        ),
    }


BACKEND_V1_REGISTRY_ENTRY = ClusteringBackendRegistryEntry(
    coerce_config=_coerce_backend_v1_registry_config,
    build_method=_build_backend_v1_registry_method,
    config_to_dict=_backend_v1_registry_config_to_dict,
)


__all__ = [
    "BACKEND_V1_IDENTIFIER",
    "BACKEND_V1_REGISTRY_ENTRY",
    "BackendV1ClusteringBackendConfig",
    "BackendV1ConversationalClusteringMethod",
    "coerce_backend_v1_backend_config",
    "default_backend_v1_intermittent_clustering_judge_model",
    "get_backend_v1_system_prompt",
]
