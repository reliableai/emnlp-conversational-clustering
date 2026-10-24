"""Model and embedding configuration helpers for OpenRouter and local embedding clients."""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr

load_dotenv()

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_REQUEST_TIMEOUT_MS = 60_000
DEFAULT_OPENROUTER_MAX_RETRIES = 2
EMBEDDING_BACKEND_OPENROUTER = "openrouter"
EMBEDDING_BACKEND_HUGGINGFACE_LOCAL = "huggingface_local"
EMBEDDING_DEVICE_AUTO = "auto"
EMBEDDING_DEVICE_CPU = "cpu"
EMBEDDING_DEVICE_CUDA = "cuda"
EMBEDDING_DEVICE_MPS = "mps"
_MODEL_CONFIG_FIELDS = {
    "model",
    "base_url",
    "provider_order",
    "allow_fallbacks",
    "api_key_env",
    "temperature",
}
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
_EMBEDDING_MODEL_CONFIG_FIELDS = {
    "model",
    "backend",
    "device",
    "base_url",
    "provider_order",
    "allow_fallbacks",
    "api_key_env",
    "dimensions",
    "input_type",
}


@dataclass(frozen=True)
class ModelConfig:
    model: str
    base_url: str = DEFAULT_BASE_URL
    provider_order: list[str] | None = None
    allow_fallbacks: bool = False
    api_key_env: str = "OPENROUTER_API_KEY"
    temperature: float | None = None


@dataclass(frozen=True)
class EmbeddingModelConfig:
    model: str = DEFAULT_EMBEDDING_MODEL
    backend: Literal["openrouter", "huggingface_local"] = EMBEDDING_BACKEND_OPENROUTER
    device: Literal["auto", "cpu", "cuda", "mps"] = EMBEDDING_DEVICE_AUTO
    base_url: str = DEFAULT_BASE_URL
    provider_order: list[str] | None = None
    allow_fallbacks: bool = False
    api_key_env: str = "OPENROUTER_API_KEY"
    dimensions: int | None = None
    input_type: str | None = None


def default_clustering_agent_model_config() -> ModelConfig:
    return ModelConfig(
        model="openai/gpt-5.4",
        base_url=DEFAULT_BASE_URL,
        provider_order=None,
        allow_fallbacks=True,
    )


def default_human_model_config() -> ModelConfig:
    return ModelConfig(
        model="x-ai/grok-4",
        base_url=DEFAULT_BASE_URL,
        provider_order=None,
        allow_fallbacks=True,
    )


def default_human_verifier_model_config() -> ModelConfig:
    return ModelConfig(
        model="google/gemini-3.1-pro-preview",
        base_url=DEFAULT_BASE_URL,
        provider_order=None,
        allow_fallbacks=True,
    )


def default_semantic_consistency_inference_model_config() -> ModelConfig:
    return ModelConfig(
        model="openai/gpt-5.4",
        base_url=DEFAULT_BASE_URL,
        provider_order=None,
        allow_fallbacks=True,
    )


def default_semantic_consistency_judge_model_config() -> ModelConfig:
    return ModelConfig(
        model="openai/gpt-5.4",
        base_url=DEFAULT_BASE_URL,
        provider_order=None,
        allow_fallbacks=True,
    )


def parse_model_config(raw_model_config: Mapping[str, object]) -> ModelConfig:
    invalid_keys = [key for key in raw_model_config if not isinstance(key, str)]
    if invalid_keys:
        raise ValueError("all keys must be strings")

    unknown_fields = sorted(set(raw_model_config) - _MODEL_CONFIG_FIELDS)
    if unknown_fields:
        raise ValueError("unknown field(s): " + ", ".join(unknown_fields))

    model_value = raw_model_config.get("model")
    if not isinstance(model_value, str) or not model_value.strip():
        raise ValueError("'model' must be a non-empty string")

    base_url_value = raw_model_config.get("base_url", DEFAULT_BASE_URL)
    if not isinstance(base_url_value, str) or not base_url_value.strip():
        raise ValueError("'base_url' must be a non-empty string")

    provider_order_value = raw_model_config.get("provider_order")
    provider_order: list[str] | None
    if provider_order_value is None:
        provider_order = None
    elif isinstance(provider_order_value, list) and all(
        isinstance(provider, str) and provider for provider in provider_order_value
    ):
        provider_order = cast(list[str], provider_order_value)
    else:
        raise ValueError("'provider_order' must be a list of non-empty strings")

    allow_fallbacks_value = raw_model_config.get("allow_fallbacks", False)
    if not isinstance(allow_fallbacks_value, bool):
        raise ValueError("'allow_fallbacks' must be a boolean")

    api_key_env_value = raw_model_config.get("api_key_env", "OPENROUTER_API_KEY")
    if not isinstance(api_key_env_value, str) or not api_key_env_value.strip():
        raise ValueError("'api_key_env' must be a non-empty string")

    temperature_value = raw_model_config.get("temperature")
    temperature: float | None
    if temperature_value is None:
        temperature = None
    elif isinstance(temperature_value, bool) or not isinstance(temperature_value, int | float):
        raise ValueError("'temperature' must be a number between 0.0 and 2.0")
    else:
        parsed_temperature = float(temperature_value)
        if (
            not math.isfinite(parsed_temperature)
            or parsed_temperature < 0.0
            or parsed_temperature > 2.0
        ):
            raise ValueError("'temperature' must be a number between 0.0 and 2.0")
        temperature = parsed_temperature

    return ModelConfig(
        model=model_value,
        base_url=base_url_value,
        provider_order=provider_order,
        allow_fallbacks=allow_fallbacks_value,
        api_key_env=api_key_env_value,
        temperature=temperature,
    )


def parse_embedding_model_config(
    raw_embedding_model_config: Mapping[str, object],
) -> EmbeddingModelConfig:
    invalid_keys = [key for key in raw_embedding_model_config if not isinstance(key, str)]
    if invalid_keys:
        raise ValueError("all keys must be strings")

    unknown_fields = sorted(set(raw_embedding_model_config) - _EMBEDDING_MODEL_CONFIG_FIELDS)
    if unknown_fields:
        raise ValueError("unknown field(s): " + ", ".join(unknown_fields))

    model_value = raw_embedding_model_config.get("model", DEFAULT_EMBEDDING_MODEL)
    if not isinstance(model_value, str) or not model_value.strip():
        raise ValueError("'model' must be a non-empty string")

    backend_value = raw_embedding_model_config.get("backend", EMBEDDING_BACKEND_OPENROUTER)
    if not isinstance(backend_value, str) or not backend_value.strip():
        raise ValueError("'backend' must be one of: openrouter, huggingface_local")
    if backend_value not in {EMBEDDING_BACKEND_OPENROUTER, EMBEDDING_BACKEND_HUGGINGFACE_LOCAL}:
        raise ValueError("'backend' must be one of: openrouter, huggingface_local")

    device_value = raw_embedding_model_config.get("device", EMBEDDING_DEVICE_AUTO)
    if not isinstance(device_value, str) or not device_value.strip():
        raise ValueError("'device' must be one of: auto, cpu, cuda, mps, metal")
    normalized_device = device_value.strip().lower()
    if normalized_device == "metal":
        normalized_device = EMBEDDING_DEVICE_MPS
    if normalized_device not in {
        EMBEDDING_DEVICE_AUTO,
        EMBEDDING_DEVICE_CPU,
        EMBEDDING_DEVICE_CUDA,
        EMBEDDING_DEVICE_MPS,
    }:
        raise ValueError("'device' must be one of: auto, cpu, cuda, mps, metal")

    base_url_value = raw_embedding_model_config.get("base_url", DEFAULT_BASE_URL)
    if not isinstance(base_url_value, str) or not base_url_value.strip():
        raise ValueError("'base_url' must be a non-empty string")

    provider_order_value = raw_embedding_model_config.get("provider_order")
    provider_order: list[str] | None
    if provider_order_value is None:
        provider_order = None
    elif isinstance(provider_order_value, list) and all(
        isinstance(provider, str) and provider for provider in provider_order_value
    ):
        provider_order = cast(list[str], provider_order_value)
    else:
        raise ValueError("'provider_order' must be a list of non-empty strings")

    allow_fallbacks_value = raw_embedding_model_config.get("allow_fallbacks", False)
    if not isinstance(allow_fallbacks_value, bool):
        raise ValueError("'allow_fallbacks' must be a boolean")

    api_key_env_value = raw_embedding_model_config.get("api_key_env", "OPENROUTER_API_KEY")
    if not isinstance(api_key_env_value, str) or not api_key_env_value.strip():
        raise ValueError("'api_key_env' must be a non-empty string")

    dimensions_value = raw_embedding_model_config.get("dimensions")
    dimensions: int | None
    if dimensions_value is None:
        dimensions = None
    elif isinstance(dimensions_value, bool) or not isinstance(dimensions_value, int):
        raise ValueError("'dimensions' must be an integer >= 1")
    elif dimensions_value < 1:
        raise ValueError("'dimensions' must be an integer >= 1")
    else:
        dimensions = dimensions_value

    input_type_value = raw_embedding_model_config.get("input_type")
    input_type: str | None
    if input_type_value is None:
        input_type = None
    elif isinstance(input_type_value, str) and input_type_value.strip():
        input_type = input_type_value.strip()
    else:
        raise ValueError("'input_type' must be a non-empty string when provided")

    return EmbeddingModelConfig(
        model=model_value,
        backend=cast(
            Literal["openrouter", "huggingface_local"],
            backend_value,
        ),
        device=cast(
            Literal["auto", "cpu", "cuda", "mps"],
            normalized_device,
        ),
        base_url=base_url_value,
        provider_order=provider_order,
        allow_fallbacks=allow_fallbacks_value,
        api_key_env=api_key_env_value,
        dimensions=dimensions,
        input_type=input_type,
    )


def default_embedding_model_config() -> EmbeddingModelConfig:
    return EmbeddingModelConfig(model=DEFAULT_EMBEDDING_MODEL)


def _build_openrouter_provider(config: ModelConfig) -> dict[str, object] | None:
    if not config.provider_order:
        return None
    return {
        "order": config.provider_order,
        "allow_fallbacks": config.allow_fallbacks,
    }


def _resolve_api_key(config: ModelConfig) -> SecretStr | None:
    api_key = os.getenv(config.api_key_env)
    if api_key:
        return SecretStr(api_key)
    return None


def build_llm(config: ModelConfig) -> ChatOpenRouter:
    provider_config = _build_openrouter_provider(config)
    if config.temperature is None:
        return ChatOpenRouter(
            openrouter_api_base=config.base_url,
            openrouter_api_key=_resolve_api_key(config),
            model=config.model,
            openrouter_provider=provider_config,
            request_timeout=DEFAULT_OPENROUTER_REQUEST_TIMEOUT_MS,
            max_retries=DEFAULT_OPENROUTER_MAX_RETRIES,
        )
    return ChatOpenRouter(
        openrouter_api_base=config.base_url,
        openrouter_api_key=_resolve_api_key(config),
        model=config.model,
        openrouter_provider=provider_config,
        temperature=config.temperature,
        request_timeout=DEFAULT_OPENROUTER_REQUEST_TIMEOUT_MS,
        max_retries=DEFAULT_OPENROUTER_MAX_RETRIES,
    )


def build_clustering_agent_llm(
    clustering_agent_model_config: ModelConfig | None = None,
) -> ChatOpenRouter:
    if clustering_agent_model_config is None:
        clustering_agent_model_config = default_clustering_agent_model_config()
    return build_llm(clustering_agent_model_config)


def build_human_llm(model_config: ModelConfig | None = None) -> ChatOpenRouter:
    if model_config is None:
        model_config = default_human_model_config()
    return build_llm(model_config)


def build_human_verifier_llm(model_config: ModelConfig | None = None) -> ChatOpenRouter:
    if model_config is None:
        model_config = default_human_verifier_model_config()
    return build_llm(model_config)


def build_semantic_consistency_inference_llm(
    model_config: ModelConfig | None = None,
) -> ChatOpenRouter:
    if model_config is None:
        model_config = default_semantic_consistency_inference_model_config()
    return build_llm(model_config)


def build_semantic_consistency_judge_llm(
    model_config: ModelConfig | None = None,
) -> ChatOpenRouter:
    if model_config is None:
        model_config = default_semantic_consistency_judge_model_config()
    return build_llm(model_config)
