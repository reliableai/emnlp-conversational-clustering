"""Unit tests for clustering-agent backend config coercion and validation."""

from collections.abc import Callable
from typing import Any, cast
from unittest import mock

import pytest
from langchain_core.messages import HumanMessage

from conversational_clustering.clustering_agent import backends as backend_module
from conversational_clustering.clustering_agent.backends import (
    BackendV1ClusteringBackendConfig,
    ClusteringAgentBackendConfig,
    InstructionKMeansV1ClusteringBackendConfig,
    OneShotKMeansV1ClusteringBackendConfig,
    coerce_clustering_agent_backend_config,
)
from conversational_clustering.clustering_agent.backends.backend_v1 import (
    BACKEND_V1_REGISTRY_ENTRY,
    get_backend_v1_system_prompt,
)
from conversational_clustering.clustering_agent.backends.backend_v1.prompts import (
    get_backend_v1_intermittent_clustering_judge_prompt,
)
from conversational_clustering.clustering_agent.backends.instruction_kmeans_v1 import (
    INSTRUCTION_KMEANS_V1_REGISTRY_ENTRY,
    INSTRUCTION_KMEANS_V1_SUPPORTED_EMBEDDING_MODELS,
    get_instruction_kmeans_v1_system_prompt,
)
from conversational_clustering.clustering_agent.backends.oneshot_kmeans_v1 import (
    ONESHOT_KMEANS_V1_REGISTRY_ENTRY,
    get_oneshot_kmeans_v1_system_prompt,
)
from conversational_clustering.clustering_agent.state import AgentState
from conversational_clustering.models import EmbeddingModelConfig, ModelConfig


def test_backend_v1_config_implements_abstract_backend_config() -> None:
    typed_config = BackendV1ClusteringBackendConfig()

    assert isinstance(typed_config, ClusteringAgentBackendConfig)
    assert typed_config.backend == "backend_v1"
    assert typed_config.prompt_id == "baseline_v2"


def test_coerce_backend_config_returns_typed_config_unchanged() -> None:
    typed_config = BackendV1ClusteringBackendConfig(prompt_id="baseline_v1")

    resolved = coerce_clustering_agent_backend_config(
        backend="backend_v1",
        backend_config=typed_config,
    )

    assert resolved is typed_config


def test_coerce_backend_config_parses_mapping_with_model_config() -> None:
    resolved = coerce_clustering_agent_backend_config(
        backend="backend_v1",
        backend_config={
            "prompt_id": "baseline_v1",
            "model_config": {
                "model": "openai/gpt-5-mini",
            },
        },
    )

    assert resolved == BackendV1ClusteringBackendConfig(
        prompt_id="baseline_v1",
        model_config=ModelConfig(model="openai/gpt-5-mini"),
    )


def test_coerce_backend_config_parses_intermittent_clustering_judge_model() -> None:
    resolved = coerce_clustering_agent_backend_config(
        backend="backend_v1",
        backend_config={
            "intermittent_clustering_judge_model": {
                "model": "openai/gpt-5-mini",
                "temperature": 0.0,
            }
        },
    )

    assert resolved == BackendV1ClusteringBackendConfig(
        intermittent_clustering_judge_model=ModelConfig(
            model="openai/gpt-5-mini",
            temperature=0.0,
        )
    )


def test_backend_v1_prompt_loader_returns_non_empty_prompt_text() -> None:
    content = get_backend_v1_system_prompt("baseline_v2")

    assert isinstance(content, str)
    assert content


def test_backend_v1_baseline_v1_prompt_loader_still_returns_non_empty_prompt_text() -> None:
    content = get_backend_v1_system_prompt("baseline_v1")

    assert isinstance(content, str)
    assert content


def test_backend_v1_intermittent_judge_prompt_loader_returns_non_empty_prompt_text() -> None:
    content = get_backend_v1_intermittent_clustering_judge_prompt("baseline_v2")

    assert isinstance(content, str)
    assert content


def test_typed_backend_v1_config_rejects_unknown_prompt_id() -> None:
    with pytest.raises(ValueError, match="Unknown backend_v1 prompt_id"):
        BackendV1ClusteringBackendConfig(prompt_id="does_not_exist")


def test_typed_backend_v1_config_rejects_whitespace_padded_prompt_id() -> None:
    with pytest.raises(ValueError, match="must not include leading/trailing whitespace"):
        BackendV1ClusteringBackendConfig(prompt_id=" baseline_v1 ")


def test_coerce_backend_config_rejects_unknown_backend_v1_prompt_id() -> None:
    with pytest.raises(ValueError, match="Unknown backend_v1 prompt_id"):
        coerce_clustering_agent_backend_config(
            backend="backend_v1",
            backend_config={"prompt_id": "does_not_exist"},
        )


def test_coerce_backend_config_rejects_whitespace_padded_backend_v1_prompt_id() -> None:
    with pytest.raises(ValueError, match="must not include leading/trailing whitespace"):
        coerce_clustering_agent_backend_config(
            backend="backend_v1",
            backend_config={"prompt_id": " baseline_v1 "},
        )


def test_coerce_backend_config_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported clustering agent backend"):
        coerce_clustering_agent_backend_config(
            backend="unknown_backend",
            backend_config=None,
        )


def test_coerce_backend_config_rejects_unknown_config_field() -> None:
    with pytest.raises(ValueError, match="Unknown field\\(s\\)"):
        coerce_clustering_agent_backend_config(
            backend="backend_v1",
            backend_config={
                "prompt_id": "baseline_v1",
                "unexpected": "value",
            },
        )


def test_coerce_backend_config_for_backend_v1_is_not_coupled_to_default_backend() -> None:
    with mock.patch.object(
        backend_module,
        "DEFAULT_CLUSTERING_AGENT_BACKEND",
        "future_default_backend",
    ):
        resolved = coerce_clustering_agent_backend_config(
            backend="backend_v1",
            backend_config=None,
        )

    assert resolved == BackendV1ClusteringBackendConfig()


def test_backend_v1_registry_entry_builds_method_and_serializes_config() -> None:
    typed_config = BACKEND_V1_REGISTRY_ENTRY.coerce_config(None)

    assert typed_config == BackendV1ClusteringBackendConfig()
    method = BACKEND_V1_REGISTRY_ENTRY.build_method(typed_config)
    assert method.backend == "backend_v1"
    assert BACKEND_V1_REGISTRY_ENTRY.config_to_dict(typed_config) == {
        "prompt_id": "baseline_v2",
        "intermittent_clustering_judge_model": {
            "model": "openai/gpt-5.4",
            "base_url": "https://openrouter.ai/api/v1",
            "provider_order": None,
            "allow_fallbacks": True,
            "api_key_env": "OPENROUTER_API_KEY",
            "temperature": None,
        },
        "model_config": None,
    }


def test_oneshot_kmeans_v1_config_implements_abstract_backend_config() -> None:
    typed_config = OneShotKMeansV1ClusteringBackendConfig()

    assert isinstance(typed_config, ClusteringAgentBackendConfig)
    assert typed_config.backend == "oneshot_kmeans_v1"


def test_coerce_backend_config_parses_oneshot_kmeans_v1_mapping() -> None:
    resolved = coerce_clustering_agent_backend_config(
        backend="oneshot_kmeans_v1",
        backend_config={
            "model_config": {
                "model": "openai/gpt-5-mini",
                "temperature": 0.0,
            },
        },
    )

    assert resolved == OneShotKMeansV1ClusteringBackendConfig(
        model_config=ModelConfig(model="openai/gpt-5-mini", temperature=0.0),
    )


def test_oneshot_kmeans_v1_prompt_loader_returns_non_empty_prompt_text() -> None:
    content = get_oneshot_kmeans_v1_system_prompt()

    assert isinstance(content, str)
    assert content


def test_coerce_backend_config_uses_oneshot_kmeans_v1_default_model() -> None:
    resolved = coerce_clustering_agent_backend_config(
        backend="oneshot_kmeans_v1",
        backend_config=None,
    )

    assert resolved == OneShotKMeansV1ClusteringBackendConfig(
        model_config=ModelConfig(
            model="openai/gpt-oss-20b",
            temperature=0.0,
            allow_fallbacks=True,
        )
    )


def test_oneshot_kmeans_v1_registry_entry_builds_method_and_serializes_config() -> None:
    typed_config = ONESHOT_KMEANS_V1_REGISTRY_ENTRY.coerce_config(None)

    assert typed_config == OneShotKMeansV1ClusteringBackendConfig()
    method = ONESHOT_KMEANS_V1_REGISTRY_ENTRY.build_method(typed_config)
    assert method.backend == "oneshot_kmeans_v1"
    assert ONESHOT_KMEANS_V1_REGISTRY_ENTRY.config_to_dict(typed_config) == {
        "model_config": {
            "model": "openai/gpt-oss-20b",
            "base_url": "https://openrouter.ai/api/v1",
            "provider_order": None,
            "allow_fallbacks": True,
            "api_key_env": "OPENROUTER_API_KEY",
            "temperature": 0.0,
        }
    }


def test_instruction_kmeans_v1_config_implements_abstract_backend_config() -> None:
    typed_config = InstructionKMeansV1ClusteringBackendConfig()

    assert isinstance(typed_config, ClusteringAgentBackendConfig)
    assert typed_config.backend == "instruction_kmeans_v1"
    assert typed_config.embedding_model_config == EmbeddingModelConfig(
        model="Qwen/Qwen3-Embedding-8B",
        backend="huggingface_local",
    )


def test_coerce_backend_config_parses_instruction_kmeans_v1_mapping() -> None:
    resolved = coerce_clustering_agent_backend_config(
        backend="instruction_kmeans_v1",
        backend_config={
            "model_config": {
                "model": "openai/gpt-5-mini",
            },
            "embedding_model_config": {
                "model": "Qwen/Qwen3-Embedding-8B",
                "backend": "huggingface_local",
                "input_type": "query",
            },
            "summary_model_config": {
                "model": "openai/gpt-5.4-nano",
            },
            "sample_window_size": 123,
            "shuffle_seed": 9,
            "cluster_summary_example_limit": 7,
            "embedding_batch_size": 16,
        },
    )

    assert resolved == InstructionKMeansV1ClusteringBackendConfig(
        model_config=ModelConfig(model="openai/gpt-5-mini"),
        embedding_model_config=EmbeddingModelConfig(
            model="Qwen/Qwen3-Embedding-8B",
            backend="huggingface_local",
            input_type="query",
        ),
        summary_model_config=ModelConfig(model="openai/gpt-5.4-nano"),
        sample_window_size=123,
        shuffle_seed=9,
        cluster_summary_example_limit=7,
        embedding_batch_size=16,
    )


def test_instruction_kmeans_v1_exposes_supported_embedding_models() -> None:
    assert INSTRUCTION_KMEANS_V1_SUPPORTED_EMBEDDING_MODELS == [
        "Qwen/Qwen3-Embedding-8B",
        "qwen/qwen3-embedding-8b",
    ]


def test_coerce_backend_config_rejects_unsupported_instruction_kmeans_v1_embedding_model() -> None:
    with pytest.raises(
        ValueError,
        match="instruction_kmeans_v1 only supports embedding_model_config.model values",
    ):
        coerce_clustering_agent_backend_config(
            backend="instruction_kmeans_v1",
            backend_config={
                "embedding_model_config": {
                    "model": "openai/text-embedding-3-small",
                }
            },
        )


def test_coerce_backend_config_accepts_explicit_instruction_kmeans_openrouter_override() -> None:
    resolved = coerce_clustering_agent_backend_config(
        backend="instruction_kmeans_v1",
        backend_config={
            "embedding_model_config": {
                "backend": "openrouter",
                "model": "qwen/qwen3-embedding-8b",
            }
        },
    )

    assert resolved == InstructionKMeansV1ClusteringBackendConfig(
        embedding_model_config=EmbeddingModelConfig(
            backend="openrouter",
            model="qwen/qwen3-embedding-8b",
        )
    )


def test_instruction_kmeans_v1_prompt_loader_returns_non_empty_prompt_text() -> None:
    content = get_instruction_kmeans_v1_system_prompt("baseline_v1")

    assert isinstance(content, str)
    assert content
    assert "`embedding_instruction` argument to `propose_clustering`" in content
    assert "not clustering rules" in content
    assert "Put the requested cluster count only in `num_clusters`" in content


@pytest.mark.parametrize(
    ("factory", "error_message"),
    [
        (
            lambda: InstructionKMeansV1ClusteringBackendConfig(sample_window_size=cast(Any, 1.5)),
            "sample_window_size must be an integer >= 1",
        ),
        (
            lambda: InstructionKMeansV1ClusteringBackendConfig(shuffle_seed=cast(Any, 1.5)),
            "shuffle_seed must be an integer >= 0",
        ),
        (
            lambda: InstructionKMeansV1ClusteringBackendConfig(
                cluster_summary_example_limit=cast(Any, 1.5)
            ),
            "cluster_summary_example_limit must be an integer >= 1",
        ),
        (
            lambda: InstructionKMeansV1ClusteringBackendConfig(embedding_batch_size=cast(Any, 1.5)),
            "embedding_batch_size must be an integer >= 1",
        ),
    ],
)
def test_typed_instruction_kmeans_v1_config_rejects_non_integer_numeric_fields(
    factory: Callable[[], InstructionKMeansV1ClusteringBackendConfig],
    error_message: str,
) -> None:
    with pytest.raises(ValueError, match=error_message):
        factory()


def test_instruction_kmeans_v1_registry_entry_builds_method_and_serializes_config() -> None:
    typed_config = INSTRUCTION_KMEANS_V1_REGISTRY_ENTRY.coerce_config(None)

    assert typed_config == InstructionKMeansV1ClusteringBackendConfig()
    method = INSTRUCTION_KMEANS_V1_REGISTRY_ENTRY.build_method(typed_config)
    assert method.backend == "instruction_kmeans_v1"
    assert INSTRUCTION_KMEANS_V1_REGISTRY_ENTRY.config_to_dict(typed_config) == {
        "prompt_id": "baseline_v1",
        "model_config": {
            "model": "openai/gpt-5.4",
            "base_url": "https://openrouter.ai/api/v1",
            "provider_order": None,
            "allow_fallbacks": True,
            "api_key_env": "OPENROUTER_API_KEY",
            "temperature": None,
        },
        "embedding_model_config": {
            "model": "Qwen/Qwen3-Embedding-8B",
            "backend": "huggingface_local",
            "device": "auto",
            "base_url": "https://openrouter.ai/api/v1",
            "provider_order": None,
            "allow_fallbacks": False,
            "api_key_env": "OPENROUTER_API_KEY",
            "dimensions": None,
            "input_type": None,
        },
        "summary_model_config": {
            "model": "openai/gpt-5.4-nano",
            "base_url": "https://openrouter.ai/api/v1",
            "provider_order": None,
            "allow_fallbacks": True,
            "api_key_env": "OPENROUTER_API_KEY",
            "temperature": None,
        },
        "intermittent_clustering_judge_model": {
            "model": "openai/gpt-5.4",
            "base_url": "https://openrouter.ai/api/v1",
            "provider_order": None,
            "allow_fallbacks": True,
            "api_key_env": "OPENROUTER_API_KEY",
            "temperature": None,
        },
        "sample_window_size": 200,
        "shuffle_seed": 42,
        "cluster_summary_example_limit": 15,
        "embedding_batch_size": 64,
    }


def test_unsupported_backend_logs_intermittent_clustering_once() -> None:
    method = ONESHOT_KMEANS_V1_REGISTRY_ENTRY.build_method(OneShotKMeansV1ClusteringBackendConfig())
    state: AgentState = {
        "messages": [HumanMessage(content="Group by topic.")],
        "dataset_metadata": {"size": 1},
        "iteration_count": 0,
    }
    logger = mock.Mock()

    with mock.patch(
        "conversational_clustering.clustering_agent.backends.base.log_json"
    ) as log_json_mock:
        resolved_state = method.capture_intermittent_clustering(
            state=state,
            data_loader=mock.Mock(),
            logger=logger,
            output_path=None,
        )
        method.capture_intermittent_clustering(
            state=state,
            data_loader=mock.Mock(),
            logger=logger,
            output_path=None,
        )

    assert resolved_state is state
    assert state["intermittent_clustering_unsupported_logged"] is True
    log_json_mock.assert_called_once_with(
        logger,
        "intermittent_clustering_unsupported",
        {
            "backend": "oneshot_kmeans_v1",
            "output_path": None,
        },
    )
