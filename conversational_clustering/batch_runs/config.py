"""Batch-run configuration and serializable result data structures."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..clustering_agent.backends import (
    DEFAULT_CLUSTERING_AGENT_BACKEND,
    ClusteringAgentBackendConfig,
    clustering_agent_backend_config_to_dict,
    default_clustering_agent_backend_config,
)
from ..human_turn_injection import HumanTurnInjectionConfig
from ..models import EmbeddingModelConfig, ModelConfig, default_embedding_model_config
from ..prompts import (
    get_human_agent_system_prompt,
    human_agent_prompt_supports_topic_discovery,
    validate_human_agent_persona,
    validate_human_agent_prompt_guidance,
)
from ..simulation.runner import SimulationResult
from ..topic_discovery import TopicDiscoveryConfig

_INTERNAL_BRANCH_METADATA_KEYS = {
    "branch_pre_vary_value",
    "branch_pre_clustering_agent_backend_config",
}
GENERATED_BENCHMARK_SPEC_METADATA_KEY = "generated_benchmark_spec_path"
AUTO_TOPIC_BRANCH_METADATA_KEYS = frozenset(
    {
        "topic_branch_mode",
        "topic_branch_outcome",
        "topic_branch_source_config_slug",
        "topic_branch_source_run_id",
        "topic_branch_source_run_index",
        "topic_branch_topic_id",
        "topic_branch_topic_label",
        "topic_branch_topic_guidance",
        "topic_branch_generated_at_turn",
    }
)
AUTO_TOPIC_BRANCH_MODE = "auto_topic_discovery"
_DEFAULT_HUMAN_AGENT_PROMPT_ID = "baseline_v3"
_EMBEDDINGS_REQUIRED_CLUSTERING_AGENT_BACKENDS = frozenset({"oneshot_kmeans_v1"})


@dataclass(frozen=True)
class HumanAgentConfig:
    prompt_id: str = _DEFAULT_HUMAN_AGENT_PROMPT_ID
    guidance: str | None = None
    persona: str | None = None
    topic_discovery: TopicDiscoveryConfig = field(default_factory=TopicDiscoveryConfig)
    turn_injection: HumanTurnInjectionConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_id, str) or not self.prompt_id.strip():
            raise ValueError("prompt_id must be a non-empty string")
        normalized_prompt_id = self.prompt_id.strip()
        try:
            get_human_agent_system_prompt(normalized_prompt_id)
        except ValueError as exc:
            raise ValueError(f"invalid prompt_id: {exc}") from exc
        try:
            normalized_guidance = validate_human_agent_prompt_guidance(
                prompt_id=normalized_prompt_id,
                guidance=self.guidance,
            )
        except ValueError as exc:
            raise ValueError(f"invalid guidance for prompt_id: {exc}") from exc
        try:
            normalized_persona = validate_human_agent_persona(self.persona)
        except ValueError as exc:
            raise ValueError(f"invalid persona: {exc}") from exc
        object.__setattr__(self, "prompt_id", normalized_prompt_id)
        object.__setattr__(self, "guidance", normalized_guidance)
        object.__setattr__(self, "persona", normalized_persona)
        if not isinstance(self.topic_discovery, TopicDiscoveryConfig):
            raise ValueError("topic_discovery must be a TopicDiscoveryConfig")
        if self.topic_discovery.enabled and not human_agent_prompt_supports_topic_discovery(
            normalized_prompt_id
        ):
            raise ValueError(
                "topic_discovery is only supported for prompt_ids that support topic discovery"
            )
        if self.turn_injection is not None and not isinstance(
            self.turn_injection, HumanTurnInjectionConfig
        ):
            raise ValueError("turn_injection must be a HumanTurnInjectionConfig or None")
        if self.topic_discovery.enabled and self.turn_injection is not None:
            raise ValueError(
                "turn_injection cannot be combined with enabled topic_discovery because "
                "topic-discovery branching may consume the injection turn before branch runs "
                "resume"
                # This could be supported in the future with additional logic to ensure the
                # injection turn is reserved and not consumed by topic-discovery branching,
                # but for now we disallow the combination to avoid complexity
            )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "prompt_id": self.prompt_id,
            "guidance": self.guidance,
        }
        if self.persona is not None:
            payload["persona"] = self.persona
        if self.topic_discovery.enabled:
            payload["topic_discovery"] = self.topic_discovery.to_dict()
        if self.turn_injection is not None:
            payload["turn_injection"] = self.turn_injection.to_dict()
        return payload


@dataclass(frozen=True)
class BatchRunConfig:
    name: str
    dataset_name: str
    clustering_agent_backend: str = DEFAULT_CLUSTERING_AGENT_BACKEND
    clustering_agent_backend_config: ClusteringAgentBackendConfig | None = None
    human_agent_config: HumanAgentConfig = field(default_factory=HumanAgentConfig)
    human_verifier_prompt_id: str = "baseline_v1"
    max_verification_retries: int = 2
    limit: int | None = None
    human_agent_model_config: ModelConfig | None = None
    human_verifier_model_config: ModelConfig | None = None
    enable_embeddings: bool = False
    enable_intermittent_clustering: bool = False
    embedding_model_config: EmbeddingModelConfig | None = None
    embedding_force_recompute: bool = False
    embedding_batch_size: int = 64
    max_turns: int = 8
    num_runs: int = 1
    random_seed: int = 42
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_runs < 1:
            raise ValueError("num_runs must be >= 1")
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if self.limit and self.limit < 1:
            raise ValueError("limit must be >= 1")
        if self.random_seed < 0:
            raise ValueError("random_seed must be an integer >= 0")
        if self.max_verification_retries < 0:
            raise ValueError("max_verification_retries must be >= 0")
        if not isinstance(self.enable_embeddings, bool):
            raise ValueError("enable_embeddings must be a boolean")
        if not isinstance(self.enable_intermittent_clustering, bool):
            raise ValueError("enable_intermittent_clustering must be a boolean")
        if not isinstance(self.embedding_force_recompute, bool):
            raise ValueError("embedding_force_recompute must be a boolean")
        if isinstance(self.embedding_batch_size, bool) or self.embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be >= 1")
        if (
            not isinstance(self.clustering_agent_backend, str)
            or not self.clustering_agent_backend.strip()
        ):
            raise ValueError("clustering_agent_backend must be a non-empty string")
        if self.clustering_agent_backend != self.clustering_agent_backend.strip():
            raise ValueError(
                "clustering_agent_backend must not include leading/trailing whitespace"
            )
        clustering_agent_backend = self.clustering_agent_backend
        clustering_agent_backend_config = self.clustering_agent_backend_config
        if clustering_agent_backend_config is None:
            clustering_agent_backend_config = default_clustering_agent_backend_config(
                clustering_agent_backend
            )
            object.__setattr__(
                self, "clustering_agent_backend_config", clustering_agent_backend_config
            )
        try:
            clustering_agent_backend_config_to_dict(
                backend=clustering_agent_backend,
                backend_config=clustering_agent_backend_config,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid clustering_agent_backend_config: {exc}") from exc
        if (
            clustering_agent_backend in _EMBEDDINGS_REQUIRED_CLUSTERING_AGENT_BACKENDS
            and not self.enable_embeddings
        ):
            raise ValueError(f"{clustering_agent_backend} requires enable_embeddings=true")
        if not isinstance(self.human_agent_config, HumanAgentConfig):
            raise ValueError("human_agent_config must be a HumanAgentConfig")
        topic_discovery = self.human_agent_config.topic_discovery
        if (
            topic_discovery.enabled
            and topic_discovery.trigger_mode == "turn_budget"
            and topic_discovery.turn_budget is not None
            and topic_discovery.turn_budget >= self.max_turns
        ):
            raise ValueError(
                "turn_budget must be less than max_turns when topic discovery is enabled"
            )
        if (
            self.human_agent_config.turn_injection is not None
            and self.human_agent_config.turn_injection.turn_index > self.max_turns
        ):
            raise ValueError("turn_injection.turn_index must be less than or equal to max_turns")
        if (
            not isinstance(self.human_verifier_prompt_id, str)
            or not self.human_verifier_prompt_id.strip()
        ):
            raise ValueError("human_verifier_prompt_id must be a non-empty string")
        object.__setattr__(self, "human_verifier_prompt_id", self.human_verifier_prompt_id.strip())
        embedding_model_config = self.embedding_model_config
        if embedding_model_config is None:
            embedding_model_config = default_embedding_model_config()
            object.__setattr__(self, "embedding_model_config", embedding_model_config)

    @property
    def human_agent_prompt_id(self) -> str:
        return self.human_agent_config.prompt_id

    @property
    def guidance(self) -> str | None:
        return self.human_agent_config.guidance

    def to_dict(self) -> dict[str, Any]:
        serialized_metadata = {
            key: value
            for key, value in self.metadata.items()
            if key not in _INTERNAL_BRANCH_METADATA_KEYS
        }
        return {
            "name": self.name,
            "dataset_name": self.dataset_name,
            "clustering_agent_backend": self.clustering_agent_backend,
            "clustering_agent_backend_config": clustering_agent_backend_config_to_dict(
                backend=self.clustering_agent_backend,
                backend_config=self.clustering_agent_backend_config,
            ),
            "human_agent_config": self.human_agent_config.to_dict(),
            "human_verifier_prompt_id": self.human_verifier_prompt_id,
            "max_verification_retries": self.max_verification_retries,
            "limit": self.limit,
            "human_agent_model_config": (
                self.human_agent_model_config.__dict__ if self.human_agent_model_config else None
            ),
            "human_verifier_model_config": (
                self.human_verifier_model_config.__dict__
                if self.human_verifier_model_config
                else None
            ),
            "enable_embeddings": self.enable_embeddings,
            "enable_intermittent_clustering": self.enable_intermittent_clustering,
            "embedding_model_config": (
                self.embedding_model_config.__dict__ if self.embedding_model_config else None
            ),
            "embedding_force_recompute": self.embedding_force_recompute,
            "embedding_batch_size": self.embedding_batch_size,
            "max_turns": self.max_turns,
            "num_runs": self.num_runs,
            "random_seed": self.random_seed,
            "metadata": serialized_metadata,
        }


@dataclass(frozen=True)
class BatchRunResult:
    config: BatchRunConfig
    simulation_results: list[SimulationResult]
    config_slug: str
    run_ids: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))

    def __post_init__(self) -> None:
        if not isinstance(self.config_slug, str) or not self.config_slug.strip():
            raise ValueError("config_slug must be a non-empty string")
        object.__setattr__(self, "config_slug", self.config_slug.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_ids": self.run_ids,
            "config_slug": self.config_slug,
            "timestamp": self.timestamp,
            "config": self.config.to_dict(),
            "results": [
                {
                    "turns_run": result.turns_run,
                    "stopped_reason": result.stopped_reason,
                    "clustering_agent_accounting": result.clustering_agent_accounting,
                    "human_agent_accounting": result.human_agent_accounting,
                }
                for result in self.simulation_results
            ],
        }
