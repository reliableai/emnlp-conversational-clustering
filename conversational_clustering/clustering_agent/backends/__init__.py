from collections.abc import Mapping
from typing import Any

from .backend_v1 import (
    BACKEND_V1_IDENTIFIER,
    BACKEND_V1_REGISTRY_ENTRY,
    BackendV1ClusteringBackendConfig,
)
from .base import (
    ClusteringAgentBackendConfig,
    ClusteringBackendRegistryEntry,
    ConversationalClusteringMethod,
)
from .instruction_kmeans_v1 import (
    INSTRUCTION_KMEANS_V1_IDENTIFIER,
    INSTRUCTION_KMEANS_V1_REGISTRY_ENTRY,
    InstructionKMeansV1ClusteringBackendConfig,
)
from .oneshot_kmeans_v1 import (
    ONESHOT_KMEANS_V1_IDENTIFIER,
    ONESHOT_KMEANS_V1_REGISTRY_ENTRY,
    OneShotKMeansV1ClusteringBackendConfig,
)

DEFAULT_CLUSTERING_AGENT_BACKEND = INSTRUCTION_KMEANS_V1_IDENTIFIER


def _validate_supported_backend(backend: str) -> None:
    if backend in _BACKEND_REGISTRY:
        return
    supported = ", ".join(sorted(_BACKEND_REGISTRY))
    raise ValueError(
        f"Unsupported clustering agent backend '{backend}'. Supported values: {supported}"
    )


def default_clustering_agent_backend_config(backend: str) -> ClusteringAgentBackendConfig:
    return coerce_clustering_agent_backend_config(backend=backend, backend_config=None)


_BACKEND_REGISTRY: dict[str, ClusteringBackendRegistryEntry] = {
    BACKEND_V1_IDENTIFIER: BACKEND_V1_REGISTRY_ENTRY,
    INSTRUCTION_KMEANS_V1_IDENTIFIER: INSTRUCTION_KMEANS_V1_REGISTRY_ENTRY,
    ONESHOT_KMEANS_V1_IDENTIFIER: ONESHOT_KMEANS_V1_REGISTRY_ENTRY,
}


def coerce_clustering_agent_backend_config(
    *,
    backend: str,
    backend_config: ClusteringAgentBackendConfig | Mapping[str, object] | None,
) -> ClusteringAgentBackendConfig:
    _validate_supported_backend(backend)
    if (
        isinstance(backend_config, ClusteringAgentBackendConfig)
        and backend_config.backend != backend
    ):
        raise ValueError(
            f"Typed backend_config targets backend '{backend_config.backend}' "
            f"but '{backend}' was requested"
        )
    backend_registry_entry = _BACKEND_REGISTRY[backend]
    return backend_registry_entry.coerce_config(backend_config)


def build_conversational_clustering_method(
    *,
    backend: str = DEFAULT_CLUSTERING_AGENT_BACKEND,
    backend_config: ClusteringAgentBackendConfig | None = None,
) -> ConversationalClusteringMethod:
    _validate_supported_backend(backend)
    backend_registry_entry = _BACKEND_REGISTRY[backend]
    return backend_registry_entry.build_method(backend_config)


def clustering_agent_backend_config_to_dict(
    *,
    backend: str,
    backend_config: ClusteringAgentBackendConfig | None = None,
) -> dict[str, Any]:
    _validate_supported_backend(backend)
    backend_registry_entry = _BACKEND_REGISTRY[backend]
    return backend_registry_entry.config_to_dict(backend_config)


__all__ = [
    "BackendV1ClusteringBackendConfig",
    "ClusteringAgentBackendConfig",
    "ConversationalClusteringMethod",
    "DEFAULT_CLUSTERING_AGENT_BACKEND",
    "InstructionKMeansV1ClusteringBackendConfig",
    "OneShotKMeansV1ClusteringBackendConfig",
    "build_conversational_clustering_method",
    "clustering_agent_backend_config_to_dict",
    "coerce_clustering_agent_backend_config",
    "default_clustering_agent_backend_config",
]
