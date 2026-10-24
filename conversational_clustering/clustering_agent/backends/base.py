"""Abstract interface for pluggable conversational clustering backends."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ...accounting import UsageCollector
from ...data_processing.data_loader import DataLoader
from ...logging_utils import log_json
from ..state import AgentState


class ClusteringAgentBackendConfig(ABC):
    @property
    @abstractmethod
    def backend(self) -> str:
        """Backend identifier used to match typed configs to backend implementations."""


class ConversationalClusteringMethod(ABC):
    @property
    @abstractmethod
    def backend(self) -> str:
        """Backend identifier used for config and result serialization."""

    @abstractmethod
    def resolve_app(
        self,
        *,
        data_loader: DataLoader | None,
        save_path: str | None = None,
        usage_collector: UsageCollector | None = None,
    ) -> object:
        """Build and return the runnable LangGraph/LangChain app for this method."""

    def build_intermittent_clustering_save_path(
        self,
        *,
        clustering_save_path: str | None,
    ) -> str | None:
        """Return the backend-specific intermittent clustering snapshot path, if any."""
        return None

    def capture_intermittent_clustering(
        self,
        *,
        state: AgentState,
        data_loader: DataLoader | None,
        logger: logging.Logger,
        output_path: str | None,
    ) -> AgentState:
        """Optionally capture backend-specific intermittent clustering state."""
        _ = data_loader
        if not state.get("intermittent_clustering_unsupported_logged"):
            log_json(
                logger,
                "intermittent_clustering_unsupported",
                {
                    "backend": self.backend,
                    "output_path": output_path,
                },
            )
            state["intermittent_clustering_unsupported_logged"] = True
        return state


@dataclass(frozen=True)
class ClusteringBackendRegistryEntry:
    coerce_config: Callable[
        [ClusteringAgentBackendConfig | Mapping[str, object] | None],
        ClusteringAgentBackendConfig,
    ]
    build_method: Callable[[ClusteringAgentBackendConfig | None], ConversationalClusteringMethod]
    config_to_dict: Callable[[ClusteringAgentBackendConfig | None], dict[str, Any]]
