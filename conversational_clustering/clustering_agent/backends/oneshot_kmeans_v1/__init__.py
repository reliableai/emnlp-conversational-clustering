from collections.abc import Mapping
from typing import Any, cast

from ..base import (
    ClusteringAgentBackendConfig,
    ClusteringBackendRegistryEntry,
    ConversationalClusteringMethod,
)
from .config import (
    ONESHOT_KMEANS_V1_IDENTIFIER,
    OneShotKMeansV1ClusteringBackendConfig,
    coerce_oneshot_kmeans_v1_backend_config,
)
from .method import OneShotKMeansV1ConversationalClusteringMethod
from .prompts import get_oneshot_kmeans_v1_system_prompt


def _coerce_oneshot_kmeans_v1_registry_config(
    backend_config: ClusteringAgentBackendConfig | Mapping[str, object] | None,
) -> ClusteringAgentBackendConfig:
    typed_or_mapping_backend_config: (
        OneShotKMeansV1ClusteringBackendConfig | Mapping[str, object] | None
    )
    if backend_config is None or isinstance(backend_config, Mapping):
        typed_or_mapping_backend_config = cast(Mapping[str, object] | None, backend_config)
    elif isinstance(backend_config, OneShotKMeansV1ClusteringBackendConfig):
        typed_or_mapping_backend_config = backend_config
    else:
        raise ValueError(
            f"Typed backend_config is not compatible with backend '{ONESHOT_KMEANS_V1_IDENTIFIER}'"
        )
    return coerce_oneshot_kmeans_v1_backend_config(backend_config=typed_or_mapping_backend_config)


def _build_oneshot_kmeans_v1_registry_method(
    backend_config: ClusteringAgentBackendConfig | None,
) -> ConversationalClusteringMethod:
    resolved_config = _coerce_oneshot_kmeans_v1_registry_config(backend_config=backend_config)
    if not isinstance(resolved_config, OneShotKMeansV1ClusteringBackendConfig):
        raise ValueError(
            f"Typed backend_config is not compatible with backend '{ONESHOT_KMEANS_V1_IDENTIFIER}'"
        )
    return OneShotKMeansV1ConversationalClusteringMethod(config=resolved_config)


def _oneshot_kmeans_v1_registry_config_to_dict(
    backend_config: ClusteringAgentBackendConfig | None,
) -> dict[str, Any]:
    resolved_config = _coerce_oneshot_kmeans_v1_registry_config(backend_config=backend_config)
    if not isinstance(resolved_config, OneShotKMeansV1ClusteringBackendConfig):
        raise ValueError(
            f"Typed backend_config is not compatible with backend '{ONESHOT_KMEANS_V1_IDENTIFIER}'"
        )
    return {
        "model_config": resolved_config.model_config.__dict__,
    }


ONESHOT_KMEANS_V1_REGISTRY_ENTRY = ClusteringBackendRegistryEntry(
    coerce_config=_coerce_oneshot_kmeans_v1_registry_config,
    build_method=_build_oneshot_kmeans_v1_registry_method,
    config_to_dict=_oneshot_kmeans_v1_registry_config_to_dict,
)


__all__ = [
    "ONESHOT_KMEANS_V1_IDENTIFIER",
    "ONESHOT_KMEANS_V1_REGISTRY_ENTRY",
    "OneShotKMeansV1ClusteringBackendConfig",
    "OneShotKMeansV1ConversationalClusteringMethod",
    "coerce_oneshot_kmeans_v1_backend_config",
    "get_oneshot_kmeans_v1_system_prompt",
]
