"""Embedding helpers for deterministic item IDs, cache IO, and remote or local embedding runs."""

import hashlib
import json
import os
import threading
from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import Any

import pandas as pd
from openrouter import OpenRouter
from openrouter.utils import BackoffStrategy, RetryConfig

from ..accounting import UsageCollector, extract_token_usage
from ..models import (
    DEFAULT_OPENROUTER_MAX_RETRIES,
    DEFAULT_OPENROUTER_REQUEST_TIMEOUT_MS,
    EMBEDDING_BACKEND_HUGGINGFACE_LOCAL,
    EMBEDDING_BACKEND_OPENROUTER,
    EMBEDDING_DEVICE_AUTO,
    EMBEDDING_DEVICE_CUDA,
    EMBEDDING_DEVICE_MPS,
    EmbeddingModelConfig,
)
from .item_records import normalize_item_text

_EMBEDDING_CACHE_ROOT = "data/embeddings"
_EMBEDDING_CACHE_COLUMNS = ("item_id", "embedding", "model")
_OPENROUTER_RETRY_INITIAL_INTERVAL_MS = 500
_OPENROUTER_RETRY_MAX_INTERVAL_MS = 60_000
_OPENROUTER_RETRY_EXPONENT = 1.5
_OPENROUTER_RETRY_MAX_ELAPSED_TIME_MS_PER_RETRY = 150_000
_LOCAL_HUGGINGFACE_EMBEDDING_LOCK = threading.Lock()


def item_id_from_text(item_text: str) -> str:
    normalized = normalize_item_text(item_text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def resolve_item_ids(items: list[str]) -> list[str]:
    return [item_id_from_text(item) for item in items]


def build_embeddings_cache_path(
    *,
    dataset_name: str,
    model_config: EmbeddingModelConfig,
    cache_root: str = _EMBEDDING_CACHE_ROOT,
) -> str:
    model_slug = _slugify(model_config.model)
    config_hash = embedding_model_config_cache_hash(model_config)
    return os.path.join(
        cache_root,
        f"{_slugify(dataset_name)}__{model_slug}__{config_hash}.parquet",
    )


def embedding_model_config_cache_hash(model_config: EmbeddingModelConfig) -> str:
    payload_dict: dict[str, object] = {
        "model": model_config.model,
        "backend": model_config.backend,
        "dimensions": model_config.dimensions,
        "input_type": model_config.input_type,
    }
    if model_config.backend == EMBEDDING_BACKEND_OPENROUTER:
        payload_dict.update(
            {
                "base_url": model_config.base_url,
                "provider_order": model_config.provider_order,
                "allow_fallbacks": model_config.allow_fallbacks,
                "api_key_env": model_config.api_key_env,
            }
        )
    payload = json.dumps(payload_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _slugify(value: str) -> str:
    chars: list[str] = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        else:
            chars.append("-")
    collapsed = "".join(chars).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return collapsed or "default"


def _build_provider_preferences(config: EmbeddingModelConfig) -> dict[str, object] | None:
    if not config.provider_order:
        return None
    return {
        "order": config.provider_order,
        "allow_fallbacks": config.allow_fallbacks,
    }


def _build_openrouter_client(config: EmbeddingModelConfig) -> OpenRouter:
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        raise ValueError(
            "Embedding API key env var "
            f"'{config.api_key_env}' is not set; "
            "cannot compute embeddings."
        )
    retry_config: RetryConfig | None = None
    if DEFAULT_OPENROUTER_MAX_RETRIES > 0:
        retry_config = RetryConfig(
            strategy="backoff",
            backoff=BackoffStrategy(
                initial_interval=_OPENROUTER_RETRY_INITIAL_INTERVAL_MS,
                max_interval=_OPENROUTER_RETRY_MAX_INTERVAL_MS,
                exponent=_OPENROUTER_RETRY_EXPONENT,
                max_elapsed_time=(
                    DEFAULT_OPENROUTER_MAX_RETRIES * _OPENROUTER_RETRY_MAX_ELAPSED_TIME_MS_PER_RETRY
                ),
            ),
            retry_connection_errors=True,
        )
    return OpenRouter(
        api_key=api_key,
        server_url=config.base_url,
        retry_config=retry_config,
        timeout_ms=DEFAULT_OPENROUTER_REQUEST_TIMEOUT_MS,
    )


def _embed_text_batch(
    *,
    client: OpenRouter,
    texts: list[str],
    config: EmbeddingModelConfig,
    usage_collector: UsageCollector | None = None,
) -> list[list[float]]:
    response = client.embeddings.generate(
        input=texts,
        model=config.model,
        encoding_format="float",
        dimensions=config.dimensions,
        provider=_build_provider_preferences(config),
        input_type=config.input_type,
    )
    if isinstance(response, str):
        raise ValueError("Expected JSON embeddings payload but received str.")
    if usage_collector is not None:
        usage_collector.record(
            config.model,
            extract_token_usage(response),
            source="remote_embeddings",
        )

    embedded_rows: list[tuple[int, list[float]]] = []
    for idx, row in enumerate(response.data):
        vector_payload = row.embedding
        if isinstance(vector_payload, str):
            raise ValueError("Expected float embeddings but received base64-encoded payload.")
        vector = [float(value) for value in vector_payload]
        index = idx if row.index is None else int(row.index)
        embedded_rows.append((index, vector))

    embedded_rows.sort(key=lambda pair: pair[0])
    vectors = [vector for _, vector in embedded_rows]
    if len(vectors) != len(texts):
        raise ValueError(
            "OpenRouter returned an unexpected number of embeddings "
            f"(expected {len(texts)}, received {len(vectors)})."
        )
    return vectors


def _generate_embeddings_with_openrouter(
    *,
    texts: list[str],
    model_config: EmbeddingModelConfig,
    batch_size: int,
    usage_collector: UsageCollector | None = None,
) -> list[list[float]]:
    client = _build_openrouter_client(model_config)
    resolved_vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        chunk_texts = texts[start : start + batch_size]
        chunk_vectors = _embed_text_batch(
            client=client,
            texts=chunk_texts,
            config=model_config,
            usage_collector=usage_collector,
        )
        resolved_vectors.extend(list(vector) for vector in chunk_vectors)

    if len(resolved_vectors) != len(texts):
        raise ValueError(
            "OpenRouter returned an unexpected number of embeddings "
            f"(expected {len(texts)}, received {len(resolved_vectors)})."
        )
    return resolved_vectors


def _torch_mps_is_available(torch_module: Any) -> bool:
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps_backend is None:
        return False
    is_available = getattr(mps_backend, "is_available", None)
    return bool(callable(is_available) and is_available())


def _resolve_local_embedding_device(
    config: EmbeddingModelConfig,
    *,
    torch_module: Any,
) -> str:
    requested_device = config.device
    cuda_is_available = bool(torch_module.cuda.is_available())
    mps_is_available = _torch_mps_is_available(torch_module)

    if requested_device == EMBEDDING_DEVICE_AUTO:
        if cuda_is_available:
            return EMBEDDING_DEVICE_CUDA
        if mps_is_available:
            return EMBEDDING_DEVICE_MPS
        return "cpu"
    if requested_device == EMBEDDING_DEVICE_CUDA:
        if not cuda_is_available:
            raise ValueError(
                "Embedding device 'cuda' was requested but CUDA is not available in this "
                "environment."
            )
        return EMBEDDING_DEVICE_CUDA
    if requested_device == EMBEDDING_DEVICE_MPS:
        if not mps_is_available:
            raise ValueError(
                "Embedding device 'mps' was requested but Apple Metal is not available in this "
                "environment."
            )
        return EMBEDDING_DEVICE_MPS
    return requested_device


def _resolve_local_sentence_transformer_path(model_name: str) -> str:
    if os.path.isdir(model_name):
        return model_name
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return model_name
    try:
        return snapshot_download(repo_id=model_name, local_files_only=True)
    except Exception:
        return model_name


@lru_cache(maxsize=8)
def _load_local_sentence_transformer(model_name: str, resolved_device: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Local Hugging Face embeddings require sentence-transformers to be installed."
        ) from exc

    model_kwargs: dict[str, object] | None = None
    if resolved_device == EMBEDDING_DEVICE_CUDA:
        model_kwargs = {"torch_dtype": "auto"}
    resolved_model_name_or_path = _resolve_local_sentence_transformer_path(model_name)
    try:
        return SentenceTransformer(
            model_name_or_path=resolved_model_name_or_path,
            device=resolved_device,
            model_kwargs=model_kwargs,
        )
    except Exception as exc:
        if os.path.isdir(model_name):
            raise
        resolved_from_cache_suffix = ""
        if resolved_model_name_or_path != model_name:
            resolved_from_cache_suffix = (
                f" Resolved cached snapshot path: '{resolved_model_name_or_path}'."
            )
        raise RuntimeError(
            "Local embedding model "
            f"'{model_name}' could not be loaded from the local environment. "
            "If this model is not already cached locally, pre-download it before running "
            "with backend='huggingface_local', or switch to backend='openrouter'."
            f"{resolved_from_cache_suffix} "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def _generate_embeddings_with_huggingface_local(
    *,
    texts: list[str],
    model_config: EmbeddingModelConfig,
    batch_size: int,
) -> list[list[float]]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Local Hugging Face embeddings require torch to be installed.") from exc

    with _LOCAL_HUGGINGFACE_EMBEDDING_LOCK:
        resolved_device = _resolve_local_embedding_device(model_config, torch_module=torch)
        model = _load_local_sentence_transformer(model_config.model, resolved_device)
        encode_kwargs = {
            "batch_size": batch_size,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "truncate_dim": model_config.dimensions,
        }
        if model_config.input_type == "query":
            vectors = model.encode_query(texts, **encode_kwargs)
        elif model_config.input_type == "document":
            vectors = model.encode_document(texts, **encode_kwargs)
        else:
            vectors = model.encode(texts, **encode_kwargs)
    return [[float(value) for value in row] for row in vectors.tolist()]


def ensure_embeddings_for_items(
    *,
    item_texts_by_id: Mapping[str, str],
    dataset_name: str,
    model_config: EmbeddingModelConfig,
    force_recompute: bool = False,
    batch_size: int = 64,
    usage_collector: UsageCollector | None = None,
) -> dict[str, list[float]]:
    if batch_size < 1:
        raise ValueError("embedding batch_size must be >= 1")

    cache_path = build_embeddings_cache_path(
        dataset_name=dataset_name,
        model_config=model_config,
    )
    cache = {} if force_recompute else _read_embeddings_cache(cache_path)
    missing_ids = [item_id for item_id in item_texts_by_id if item_id not in cache]

    if missing_ids:
        missing_texts = [item_texts_by_id[item_id] for item_id in missing_ids]
        missing_vectors = _generate_embeddings_for_texts(
            texts=missing_texts,
            model_config=model_config,
            batch_size=batch_size,
            usage_collector=usage_collector,
        )
        for item_id, vector in zip(missing_ids, missing_vectors, strict=True):
            cache[item_id] = vector
        _write_embeddings_cache(cache_path, cache=cache, model=model_config.model)

    resolved: dict[str, list[float]] = {}
    for item_id in item_texts_by_id:
        vector = cache.get(item_id)
        if vector is None:
            raise ValueError(
                f"Could not resolve embedding for item_id '{item_id}' in dataset '{dataset_name}'."
            )
        resolved[item_id] = list(vector)
    return resolved


def generate_embeddings_for_texts(
    *,
    texts: list[str],
    model_config: EmbeddingModelConfig,
    batch_size: int = 64,
    usage_collector: UsageCollector | None = None,
) -> list[list[float]]:
    return _generate_embeddings_for_texts(
        texts=texts,
        model_config=model_config,
        batch_size=batch_size,
        usage_collector=usage_collector,
    )


def _generate_embeddings_for_texts(
    *,
    texts: list[str],
    model_config: EmbeddingModelConfig,
    batch_size: int,
    usage_collector: UsageCollector | None = None,
) -> list[list[float]]:
    if batch_size < 1:
        raise ValueError("embedding batch_size must be >= 1")
    if not texts:
        return []
    if model_config.backend == EMBEDDING_BACKEND_HUGGINGFACE_LOCAL:
        return _generate_embeddings_with_huggingface_local(
            texts=texts,
            model_config=model_config,
            batch_size=batch_size,
        )
    return _generate_embeddings_with_openrouter(
        texts=texts,
        model_config=model_config,
        batch_size=batch_size,
        usage_collector=usage_collector,
    )


def _read_embeddings_cache(cache_path: str) -> dict[str, list[float]]:
    if not os.path.exists(cache_path):
        return {}

    frame = pd.read_parquet(cache_path)
    missing_columns = [column for column in _EMBEDDING_CACHE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Embeddings cache is missing expected column(s): " + ", ".join(missing_columns)
        )

    cache: dict[str, list[float]] = {}
    for _, row in frame.iterrows():
        item_id = str(row["item_id"])
        raw_embedding = row["embedding"]
        if isinstance(raw_embedding, str):
            raise ValueError(
                "Embeddings cache row contained string payload where float vector was expected."
            )
        if isinstance(raw_embedding, bytes) or not isinstance(raw_embedding, Iterable):
            raise ValueError("Embeddings cache row contained invalid embedding payload.")
        cache[item_id] = [float(value) for value in raw_embedding]
    return cache


def _write_embeddings_cache(
    cache_path: str,
    *,
    cache: Mapping[str, list[float]],
    model: str,
) -> None:
    output_dir = os.path.dirname(cache_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    frame = pd.DataFrame(
        {
            "item_id": list(cache.keys()),
            "embedding": [list(vector) for vector in cache.values()],
            "model": [model for _ in range(len(cache))],
        }
    )
    frame.to_parquet(cache_path, index=False)
