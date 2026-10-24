from collections.abc import Mapping
from typing import Any, cast

from ..base import (
    ClusteringAgentBackendConfig,
    ClusteringBackendRegistryEntry,
    ConversationalClusteringMethod,
)
from .config import (
    INSTRUCTION_KMEANS_V1_IDENTIFIER,
    INSTRUCTION_KMEANS_V1_SUPPORTED_EMBEDDING_MODELS,
    InstructionKMeansV1ClusteringBackendConfig,
    coerce_instruction_kmeans_v1_backend_config,
    default_instruction_kmeans_v1_embedding_model_config,
    default_instruction_kmeans_v1_intermittent_clustering_judge_model,
    default_instruction_kmeans_v1_model_config,
    default_instruction_kmeans_v1_openrouter_embedding_model_config,
    default_instruction_kmeans_v1_summary_model_config,
)
from .method import InstructionKMeansV1ConversationalClusteringMethod
from .prompts import get_instruction_kmeans_v1_system_prompt


def _coerce_instruction_kmeans_v1_registry_config(
    backend_config: ClusteringAgentBackendConfig | Mapping[str, object] | None,
) -> ClusteringAgentBackendConfig:
    typed_or_mapping_backend_config: (
        InstructionKMeansV1ClusteringBackendConfig | Mapping[str, object] | None
    )
    if backend_config is None or isinstance(backend_config, Mapping):
        typed_or_mapping_backend_config = cast(Mapping[str, object] | None, backend_config)
    elif isinstance(backend_config, InstructionKMeansV1ClusteringBackendConfig):
        typed_or_mapping_backend_config = backend_config
    else:
        raise ValueError(
            "Typed backend_config is not compatible with backend "
            f"'{INSTRUCTION_KMEANS_V1_IDENTIFIER}'"
        )
    return coerce_instruction_kmeans_v1_backend_config(
        backend_config=typed_or_mapping_backend_config
    )


def _build_instruction_kmeans_v1_registry_method(
    backend_config: ClusteringAgentBackendConfig | None,
) -> ConversationalClusteringMethod:
    resolved_config = _coerce_instruction_kmeans_v1_registry_config(backend_config=backend_config)
    if not isinstance(resolved_config, InstructionKMeansV1ClusteringBackendConfig):
        raise ValueError(
            "Typed backend_config is not compatible with backend "
            f"'{INSTRUCTION_KMEANS_V1_IDENTIFIER}'"
        )
    return InstructionKMeansV1ConversationalClusteringMethod(config=resolved_config)


def _instruction_kmeans_v1_registry_config_to_dict(
    backend_config: ClusteringAgentBackendConfig | None,
) -> dict[str, Any]:
    resolved_config = _coerce_instruction_kmeans_v1_registry_config(backend_config=backend_config)
    if not isinstance(resolved_config, InstructionKMeansV1ClusteringBackendConfig):
        raise ValueError(
            "Typed backend_config is not compatible with backend "
            f"'{INSTRUCTION_KMEANS_V1_IDENTIFIER}'"
        )
    return {
        "prompt_id": resolved_config.prompt_id,
        "model_config": resolved_config.model_config.__dict__,
        "embedding_model_config": resolved_config.embedding_model_config.__dict__,
        "summary_model_config": resolved_config.summary_model_config.__dict__,
        "intermittent_clustering_judge_model": (
            resolved_config.intermittent_clustering_judge_model.__dict__
        ),
        "sample_window_size": resolved_config.sample_window_size,
        "shuffle_seed": resolved_config.shuffle_seed,
        "cluster_summary_example_limit": resolved_config.cluster_summary_example_limit,
        "embedding_batch_size": resolved_config.embedding_batch_size,
    }


INSTRUCTION_KMEANS_V1_REGISTRY_ENTRY = ClusteringBackendRegistryEntry(
    coerce_config=_coerce_instruction_kmeans_v1_registry_config,
    build_method=_build_instruction_kmeans_v1_registry_method,
    config_to_dict=_instruction_kmeans_v1_registry_config_to_dict,
)


__all__ = [
    "INSTRUCTION_KMEANS_V1_IDENTIFIER",
    "INSTRUCTION_KMEANS_V1_REGISTRY_ENTRY",
    "INSTRUCTION_KMEANS_V1_SUPPORTED_EMBEDDING_MODELS",
    "InstructionKMeansV1ClusteringBackendConfig",
    "InstructionKMeansV1ConversationalClusteringMethod",
    "coerce_instruction_kmeans_v1_backend_config",
    "default_instruction_kmeans_v1_embedding_model_config",
    "default_instruction_kmeans_v1_intermittent_clustering_judge_model",
    "default_instruction_kmeans_v1_model_config",
    "default_instruction_kmeans_v1_openrouter_embedding_model_config",
    "default_instruction_kmeans_v1_summary_model_config",
    "get_instruction_kmeans_v1_system_prompt",
]
