"""Unit tests for model configuration helpers and LLM factory wiring."""

from unittest import mock

import pytest
from pydantic import SecretStr

import conversational_clustering.models as models
from conversational_clustering.models import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_BACKEND_HUGGINGFACE_LOCAL,
    EMBEDDING_BACKEND_OPENROUTER,
    EMBEDDING_DEVICE_AUTO,
    EMBEDDING_DEVICE_MPS,
    EmbeddingModelConfig,
    ModelConfig,
    default_embedding_model_config,
    parse_embedding_model_config,
    parse_model_config,
)


def test_build_openrouter_provider_returns_none_without_provider_order() -> None:
    config = ModelConfig(model="openai/gpt-5-mini", provider_order=None)

    payload = models._build_openrouter_provider(config)

    assert payload is None


def test_build_openrouter_provider_includes_order_and_fallbacks() -> None:
    config = ModelConfig(
        model="openai/gpt-5-mini",
        provider_order=["openai", "anthropic"],
        allow_fallbacks=True,
    )

    payload = models._build_openrouter_provider(config)

    assert payload == {
        "order": ["openai", "anthropic"],
        "allow_fallbacks": True,
    }


def test_build_llm_passes_expected_kwargs_and_secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ModelConfig(
        model="openai/gpt-5-mini",
        base_url="https://example.com/v1",
        provider_order=["openai"],
        allow_fallbacks=False,
        api_key_env="CUSTOM_API_KEY",
    )
    monkeypatch.setenv("CUSTOM_API_KEY", "secret-value")

    with mock.patch.object(models, "ChatOpenRouter", autospec=True) as chat_openrouter_mock:
        models.build_llm(config)

    chat_openrouter_mock.assert_called_once()
    kwargs = chat_openrouter_mock.call_args.kwargs
    assert kwargs["openrouter_api_base"] == "https://example.com/v1"
    assert kwargs["model"] == "openai/gpt-5-mini"
    assert kwargs["openrouter_provider"] == {"order": ["openai"], "allow_fallbacks": False}
    assert isinstance(kwargs["openrouter_api_key"], SecretStr)
    assert kwargs["openrouter_api_key"].get_secret_value() == "secret-value"


def test_build_llm_includes_temperature_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ModelConfig(
        model="openai/gpt-5-mini",
        temperature=0.4,
        api_key_env="CUSTOM_API_KEY",
    )
    monkeypatch.setenv("CUSTOM_API_KEY", "secret-value")

    with mock.patch.object(models, "ChatOpenRouter", autospec=True) as chat_openrouter_mock:
        models.build_llm(config)

    kwargs = chat_openrouter_mock.call_args.kwargs
    assert kwargs["temperature"] == pytest.approx(0.4)


def test_parse_model_config_parses_temperature() -> None:
    config = parse_model_config({"model": "openai/gpt-5-mini", "temperature": 1})

    assert config.temperature == pytest.approx(1.0)


def test_parse_model_config_rejects_out_of_range_temperature() -> None:
    with pytest.raises(ValueError, match="'temperature' must be a number between 0.0 and 2.0"):
        parse_model_config({"model": "openai/gpt-5-mini", "temperature": 2.1})


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), float("-inf")])
def test_parse_model_config_rejects_non_finite_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="'temperature' must be a number between 0.0 and 2.0"):
        parse_model_config({"model": "openai/gpt-5-mini", "temperature": temperature})


def test_build_clustering_agent_llm_uses_expected_default_config() -> None:
    with mock.patch.object(models, "build_llm", autospec=True, return_value=object()) as build_llm:
        models.build_clustering_agent_llm()

    called_config = build_llm.call_args.args[0]
    assert called_config.model == "openai/gpt-5.4"
    assert called_config.base_url == models.DEFAULT_BASE_URL
    assert called_config.provider_order is None
    assert called_config.allow_fallbacks is True


def test_build_human_llm_uses_expected_default_config() -> None:
    with mock.patch.object(models, "build_llm", autospec=True, return_value=object()) as build_llm:
        models.build_human_llm()

    called_config = build_llm.call_args.args[0]
    assert called_config.model == "x-ai/grok-4"
    assert called_config.base_url == models.DEFAULT_BASE_URL
    assert called_config.provider_order is None
    assert called_config.allow_fallbacks is True


def test_parse_embedding_model_config_parses_defaults() -> None:
    config = parse_embedding_model_config({})

    assert config == EmbeddingModelConfig(model=DEFAULT_EMBEDDING_MODEL)


def test_parse_embedding_model_config_parses_valid_fields() -> None:
    config = parse_embedding_model_config(
        {
            "model": "openai/text-embedding-3-large",
            "backend": "huggingface_local",
            "device": "cuda",
            "base_url": "https://example.com/v1",
            "provider_order": ["openai"],
            "allow_fallbacks": True,
            "api_key_env": "CUSTOM_EMBEDDING_KEY",
            "dimensions": 256,
            "input_type": "document",
        }
    )

    assert config == EmbeddingModelConfig(
        model="openai/text-embedding-3-large",
        backend=EMBEDDING_BACKEND_HUGGINGFACE_LOCAL,
        device="cuda",
        base_url="https://example.com/v1",
        provider_order=["openai"],
        allow_fallbacks=True,
        api_key_env="CUSTOM_EMBEDDING_KEY",
        dimensions=256,
        input_type="document",
    )


def test_parse_embedding_model_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        parse_embedding_model_config({"model": "x", "unexpected": "y"})


def test_parse_embedding_model_config_normalizes_metal_device_to_mps() -> None:
    config = parse_embedding_model_config({"backend": "huggingface_local", "device": "metal"})

    assert config.backend == EMBEDDING_BACKEND_HUGGINGFACE_LOCAL
    assert config.device == EMBEDDING_DEVICE_MPS


def test_parse_embedding_model_config_rejects_invalid_backend() -> None:
    with pytest.raises(ValueError, match="'backend' must be one of: openrouter, huggingface_local"):
        parse_embedding_model_config({"backend": "local"})


def test_parse_embedding_model_config_rejects_invalid_device() -> None:
    with pytest.raises(ValueError, match="'device' must be one of: auto, cpu, cuda, mps, metal"):
        parse_embedding_model_config({"device": "gpu"})


def test_parse_embedding_model_config_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="'dimensions' must be an integer >= 1"):
        parse_embedding_model_config({"dimensions": 0})


def test_default_embedding_model_config_uses_expected_model() -> None:
    config = default_embedding_model_config()

    assert config.model == DEFAULT_EMBEDDING_MODEL
    assert config.backend == EMBEDDING_BACKEND_OPENROUTER
    assert config.device == EMBEDDING_DEVICE_AUTO
