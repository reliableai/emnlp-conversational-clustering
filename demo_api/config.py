"""Configuration helpers for the standalone demo API and fixed interactive session defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from conversational_clustering.clustering_agent.backends import (
    INSTRUCTION_KMEANS_V1_IDENTIFIER,
    ClusteringAgentBackendConfig,
    InstructionKMeansV1ClusteringBackendConfig,
)
from conversational_clustering.clustering_agent.backends.instruction_kmeans_v1 import (
    default_instruction_kmeans_v1_openrouter_embedding_model_config,
)
from conversational_clustering.models import EMBEDDING_BACKEND_HUGGINGFACE_LOCAL

_DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
)


def _parse_csv_env(name: str, *, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    parsed = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    return parsed or default


def _parse_bool_env(name: str, *, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class DemoSettings:
    dataset_name: str = "synthetic"
    clustering_agent_backend: str = INSTRUCTION_KMEANS_V1_IDENTIFIER
    cors_origins: tuple[str, ...] = _DEFAULT_CORS_ORIGINS
    disable_local_embeddings: bool = False


def resolve_demo_backend_config(
    backend_config: ClusteringAgentBackendConfig,
    *,
    settings: DemoSettings,
) -> ClusteringAgentBackendConfig:
    if not settings.disable_local_embeddings:
        return backend_config
    if not isinstance(backend_config, InstructionKMeansV1ClusteringBackendConfig):
        return backend_config
    if backend_config.embedding_model_config.backend != EMBEDDING_BACKEND_HUGGINGFACE_LOCAL:
        return backend_config
    return replace(
        backend_config,
        embedding_model_config=default_instruction_kmeans_v1_openrouter_embedding_model_config(),
    )


def load_demo_settings() -> DemoSettings:
    return DemoSettings(
        dataset_name=os.getenv("DEMO_DATASET", "synthetic").strip() or "synthetic",
        clustering_agent_backend=(
            os.getenv("DEMO_CLUSTERING_BACKEND", INSTRUCTION_KMEANS_V1_IDENTIFIER).strip()
            or INSTRUCTION_KMEANS_V1_IDENTIFIER
        ),
        cors_origins=_parse_csv_env(
            "DEMO_CORS_ORIGINS",
            default=_DEFAULT_CORS_ORIGINS,
        ),
        disable_local_embeddings=_parse_bool_env(
            "DEMO_DISABLE_LOCAL_EMBEDDINGS",
            default=False,
        ),
    )
