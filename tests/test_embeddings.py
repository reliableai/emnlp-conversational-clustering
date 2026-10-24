"""Unit tests for embedding cache path generation and config-based invalidation."""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from conversational_clustering.accounting import UsageCollector
from conversational_clustering.data_processing import embeddings as embeddings_module
from conversational_clustering.data_processing.embeddings import (
    build_embeddings_cache_path,
    embedding_model_config_cache_hash,
    ensure_embeddings_for_items,
    generate_embeddings_for_texts,
)
from conversational_clustering.models import (
    EMBEDDING_BACKEND_HUGGINGFACE_LOCAL,
    EMBEDDING_DEVICE_MPS,
    EmbeddingModelConfig,
)


def test_build_embeddings_cache_path_changes_when_embedding_config_changes() -> None:
    base_config = EmbeddingModelConfig(model="openai/text-embedding-3-small")
    resized_config = EmbeddingModelConfig(
        model="openai/text-embedding-3-small",
        dimensions=512,
    )

    base_path = build_embeddings_cache_path(
        dataset_name="synthetic",
        model_config=base_config,
    )
    resized_path = build_embeddings_cache_path(
        dataset_name="synthetic",
        model_config=resized_config,
    )

    assert base_path != resized_path
    assert "synthetic__openai-text-embedding-3-small__" in base_path
    assert base_path.endswith(".parquet")


def test_build_embeddings_cache_path_is_stable_for_identical_configs() -> None:
    config = EmbeddingModelConfig(
        model="openai/text-embedding-3-small",
        provider_order=["openai"],
        allow_fallbacks=True,
    )

    first_path = build_embeddings_cache_path(
        dataset_name="synthetic",
        model_config=config,
    )
    second_path = build_embeddings_cache_path(
        dataset_name="synthetic",
        model_config=config,
    )

    assert first_path == second_path


def test_embedding_model_config_cache_hash_changes_when_backend_changes() -> None:
    openrouter_config = EmbeddingModelConfig(model="demo", backend="openrouter")
    local_config = EmbeddingModelConfig(model="demo", backend="huggingface_local")

    assert embedding_model_config_cache_hash(
        openrouter_config
    ) != embedding_model_config_cache_hash(local_config)


def test_embedding_model_config_cache_hash_ignores_device_for_local_models() -> None:
    cuda_config = EmbeddingModelConfig(
        model="Qwen/Qwen3-Embedding-8B",
        backend="huggingface_local",
        device="cuda",
    )
    mps_config = EmbeddingModelConfig(
        model="Qwen/Qwen3-Embedding-8B",
        backend="huggingface_local",
        device="mps",
    )

    assert embedding_model_config_cache_hash(cuda_config) == embedding_model_config_cache_hash(
        mps_config
    )


def test_generate_embeddings_for_texts_does_not_cache_across_calls(
    monkeypatch,
) -> None:
    model_config = EmbeddingModelConfig(model="openai/text-embedding-3-small")
    call_counts = {"client": 0, "embed": 0}

    def fake_build_client(config: EmbeddingModelConfig) -> object:
        assert config == model_config
        call_counts["client"] += 1
        return object()

    def fake_embed_text_batch(
        *,
        client: object,
        texts: list[str],
        config: EmbeddingModelConfig,
        usage_collector: UsageCollector | None = None,
    ) -> list[list[float]]:
        assert config == model_config
        _ = client, usage_collector
        call_counts["embed"] += 1
        return [[float(len(text))] for text in texts]

    monkeypatch.setattr(embeddings_module, "_build_openrouter_client", fake_build_client)
    monkeypatch.setattr(embeddings_module, "_embed_text_batch", fake_embed_text_batch)

    first_vectors = generate_embeddings_for_texts(
        texts=["alpha", "beta"],
        model_config=model_config,
        batch_size=64,
    )
    second_vectors = generate_embeddings_for_texts(
        texts=["alpha", "beta"],
        model_config=model_config,
        batch_size=64,
    )

    assert first_vectors == [[5.0], [4.0]]
    assert second_vectors == first_vectors
    assert call_counts == {"client": 2, "embed": 2}


def test_generate_embeddings_for_texts_records_remote_embedding_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = EmbeddingModelConfig(model="openai/text-embedding-3-small")
    collector = UsageCollector()

    class FakeEmbeddingsAPI:
        def generate(self, **kwargs: object) -> SimpleNamespace:
            assert kwargs["model"] == model_config.model
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=0, embedding=[1.0, 0.0]),
                    SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                ],
                model="OpenAI/text-embedding-3-small",
                usage={"prompt_tokens": 7, "total_tokens": 7},
            )

    class FakeClient:
        embeddings = FakeEmbeddingsAPI()

    monkeypatch.setattr(
        embeddings_module,
        "_build_openrouter_client",
        lambda config: FakeClient(),
    )

    vectors = generate_embeddings_for_texts(
        texts=["alpha", "beta"],
        model_config=model_config,
        batch_size=8,
        usage_collector=collector,
    )

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    usage = collector.snapshot(pricer=None)
    assert usage.total_tokens == 7
    assert usage.input_tokens == 7
    assert usage.output_tokens == 0
    assert usage.by_model[model_config.model].total_tokens == 7


def test_generate_embeddings_for_texts_uses_local_backend_dispatch_without_openrouter(
    monkeypatch,
) -> None:
    model_config = EmbeddingModelConfig(
        model="Qwen/Qwen3-Embedding-8B",
        backend="huggingface_local",
        device="auto",
    )

    def fail_build_client(config: EmbeddingModelConfig) -> object:
        raise AssertionError(f"unexpected OpenRouter client request for {config}")

    class FakeSentenceTransformer:
        def encode(self, texts: list[str], **kwargs: object) -> SimpleNamespace:
            assert texts == ["alpha", "beta"]
            assert kwargs["batch_size"] == 8
            assert kwargs["convert_to_numpy"] is True
            assert kwargs["normalize_embeddings"] is True
            assert kwargs["truncate_dim"] is None
            return SimpleNamespace(tolist=lambda: [[1.0, 0.0], [0.0, 1.0]])

    monkeypatch.setattr(embeddings_module, "_build_openrouter_client", fail_build_client)
    monkeypatch.setattr(
        embeddings_module,
        "_resolve_local_embedding_device",
        lambda config, *, torch_module: "cuda",
    )
    monkeypatch.setattr(
        embeddings_module,
        "_load_local_sentence_transformer",
        lambda model_name, resolved_device: FakeSentenceTransformer(),
    )

    vectors = generate_embeddings_for_texts(
        texts=["alpha", "beta"],
        model_config=model_config,
        batch_size=8,
    )

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]


def test_generate_embeddings_for_texts_uses_query_encoder_for_query_input_type(
    monkeypatch,
) -> None:
    model_config = EmbeddingModelConfig(
        model="Qwen/Qwen3-Embedding-8B",
        backend="huggingface_local",
        input_type="query",
    )
    call_log: list[tuple[str, list[str], dict[str, object]]] = []

    class FakeSentenceTransformer:
        def encode_query(self, texts: list[str], **kwargs: object) -> SimpleNamespace:
            call_log.append(("encode_query", texts, dict(kwargs)))
            return SimpleNamespace(tolist=lambda: [[3.0]])

        def encode_document(self, texts: list[str], **kwargs: object) -> SimpleNamespace:
            raise AssertionError("encode_document should not be called")

        def encode(self, texts: list[str], **kwargs: object) -> SimpleNamespace:
            raise AssertionError("encode should not be called")

    monkeypatch.setattr(
        embeddings_module,
        "_resolve_local_embedding_device",
        lambda config, *, torch_module: "cpu",
    )
    monkeypatch.setattr(
        embeddings_module,
        "_load_local_sentence_transformer",
        lambda model_name, resolved_device: FakeSentenceTransformer(),
    )

    vectors = generate_embeddings_for_texts(
        texts=["alpha"],
        model_config=model_config,
        batch_size=4,
    )

    assert vectors == [[3.0]]
    assert call_log == [
        (
            "encode_query",
            ["alpha"],
            {
                "batch_size": 4,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
                "truncate_dim": None,
            },
        )
    ]


def test_generate_embeddings_for_texts_serializes_local_backend_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = EmbeddingModelConfig(
        model="Qwen/Qwen3-Embedding-8B",
        backend="huggingface_local",
    )
    active_encodes = 0
    max_active_encodes = 0
    encode_calls = 0
    counter_lock = threading.Lock()
    first_encode_entered = threading.Event()
    release_first_encode = threading.Event()

    class FakeSentenceTransformer:
        def encode(self, texts: list[str], **kwargs: object) -> SimpleNamespace:
            nonlocal active_encodes, max_active_encodes, encode_calls
            _ = kwargs
            with counter_lock:
                active_encodes += 1
                encode_calls += 1
                current_call = encode_calls
                max_active_encodes = max(max_active_encodes, active_encodes)
                if current_call == 1:
                    first_encode_entered.set()
            if current_call == 1:
                assert release_first_encode.wait(timeout=2.0)
            else:
                time.sleep(0.01)
            with counter_lock:
                active_encodes -= 1
            return SimpleNamespace(tolist=lambda: [[float(len(texts[0]))]])

    fake_model = FakeSentenceTransformer()
    monkeypatch.setattr(
        embeddings_module,
        "_resolve_local_embedding_device",
        lambda config, *, torch_module: "cpu",
    )
    monkeypatch.setattr(
        embeddings_module,
        "_load_local_sentence_transformer",
        lambda model_name, resolved_device: fake_model,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            generate_embeddings_for_texts,
            texts=["alpha"],
            model_config=model_config,
            batch_size=4,
        )
        assert first_encode_entered.wait(timeout=2.0)
        second = executor.submit(
            generate_embeddings_for_texts,
            texts=["beta"],
            model_config=model_config,
            batch_size=4,
        )
        time.sleep(0.05)
        assert max_active_encodes == 1
        release_first_encode.set()

        assert first.result(timeout=2.0) == [[5.0]]
        assert second.result(timeout=2.0) == [[4.0]]

    assert max_active_encodes == 1
    assert encode_calls == 2


def test_load_local_sentence_transformer_uses_cached_snapshot_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings_module._load_local_sentence_transformer.cache_clear()
    init_calls: list[tuple[str, str, dict[str, object] | None]] = []
    snapshot_calls: list[tuple[str, bool]] = []

    class FakeSentenceTransformer:
        def __init__(
            self,
            *,
            model_name_or_path: str,
            device: str,
            model_kwargs: dict[str, object] | None,
        ) -> None:
            init_calls.append((model_name_or_path, device, model_kwargs))

    def fake_snapshot_download(*, repo_id: str, local_files_only: bool) -> str:
        snapshot_calls.append((repo_id, local_files_only))
        return "/tmp/qwen3-embedding-snapshot"

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    embeddings_module._load_local_sentence_transformer(
        "Qwen/Qwen3-Embedding-8B",
        "cpu",
    )

    assert snapshot_calls == [("Qwen/Qwen3-Embedding-8B", True)]
    assert init_calls == [("/tmp/qwen3-embedding-snapshot", "cpu", None)]
    embeddings_module._load_local_sentence_transformer.cache_clear()


def test_resolve_local_embedding_device_rejects_explicit_cuda_when_unavailable() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(
            mps=SimpleNamespace(
                is_available=lambda: False,
            )
        ),
    )

    with pytest.raises(ValueError, match="Embedding device 'cuda' was requested"):
        embeddings_module._resolve_local_embedding_device(
            EmbeddingModelConfig(backend=EMBEDDING_BACKEND_HUGGINGFACE_LOCAL, device="cuda"),
            torch_module=fake_torch,
        )


def test_resolve_local_embedding_device_rejects_explicit_mps_when_unavailable() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(
            mps=SimpleNamespace(
                is_available=lambda: False,
            )
        ),
    )

    with pytest.raises(ValueError, match="Embedding device 'mps' was requested"):
        embeddings_module._resolve_local_embedding_device(
            EmbeddingModelConfig(
                backend=EMBEDDING_BACKEND_HUGGINGFACE_LOCAL,
                device=EMBEDDING_DEVICE_MPS,
            ),
            torch_module=fake_torch,
        )


def test_ensure_embeddings_for_items_uses_disk_cache_per_dataset(
    monkeypatch,
) -> None:
    model_config = EmbeddingModelConfig(model="openai/text-embedding-3-small")
    cache_by_path: dict[str, dict[str, list[float]]] = {}

    def fake_read_embeddings_cache(cache_path: str) -> dict[str, list[float]]:
        return {
            item_id: list(vector) for item_id, vector in cache_by_path.get(cache_path, {}).items()
        }

    def fake_write_embeddings_cache(
        cache_path: str,
        *,
        cache: dict[str, list[float]],
        model: str,
    ) -> None:
        _ = model
        cache_by_path[cache_path] = {item_id: list(vector) for item_id, vector in cache.items()}

    monkeypatch.setattr(embeddings_module, "_read_embeddings_cache", fake_read_embeddings_cache)
    monkeypatch.setattr(
        embeddings_module,
        "_write_embeddings_cache",
        fake_write_embeddings_cache,
    )
    call_counts = {"client": 0, "embed": 0}

    def fake_build_client(config: EmbeddingModelConfig) -> object:
        assert config == model_config
        call_counts["client"] += 1
        return object()

    def fake_embed_text_batch(
        *,
        client: object,
        texts: list[str],
        config: EmbeddingModelConfig,
        usage_collector: UsageCollector | None = None,
    ) -> list[list[float]]:
        assert config == model_config
        _ = client, usage_collector
        call_counts["embed"] += 1
        return [[float(len(text))] for text in texts]

    monkeypatch.setattr(embeddings_module, "_build_openrouter_client", fake_build_client)
    monkeypatch.setattr(embeddings_module, "_embed_text_batch", fake_embed_text_batch)

    first_result = ensure_embeddings_for_items(
        item_texts_by_id={"a": "alpha", "b": "beta"},
        dataset_name="dataset_one",
        model_config=model_config,
    )
    second_result = ensure_embeddings_for_items(
        item_texts_by_id={"a": "alpha", "b": "beta"},
        dataset_name="dataset_one",
        model_config=model_config,
    )

    assert first_result == {"a": [5.0], "b": [4.0]}
    assert second_result == first_result
    assert call_counts == {"client": 1, "embed": 1}
