"""Unit tests for token accounting summary merging and pricing metadata preservation."""

from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest
from openrouter.operations.createembeddings import Usage as EmbeddingUsage

import conversational_clustering.accounting as accounting_module
from conversational_clustering.accounting import (
    build_pricing_client_from_configs,
    extract_response_model,
    extract_token_usage,
    merge_usage_summaries,
)
from conversational_clustering.models import DEFAULT_BASE_URL, EmbeddingModelConfig, ModelConfig


@pytest.fixture(autouse=True)
def clear_pricing_caches() -> None:
    accounting_module._SHARED_PRICING_CACHE.clear()
    accounting_module._SHARED_PRICING_FAILURE_CACHE.clear()


def test_merge_usage_summaries_preserves_top_level_totals_without_by_model() -> None:
    merged = merge_usage_summaries(
        [
            {
                "input_tokens": 3,
                "output_tokens": 4,
                "total_tokens": 7,
                "estimated_cost_usd": 0.25,
                "cost_estimate_status": "estimated",
            },
            {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
                "estimated_cost_usd": 0.1,
                "cost_estimate_status": "estimated",
            },
        ]
    )

    assert merged.input_tokens == 4
    assert merged.output_tokens == 6
    assert merged.total_tokens == 10
    assert merged.estimated_cost_usd == pytest.approx(0.35)
    assert merged.cost_estimate_status == "estimated"
    assert merged.by_model == {}
    assert merged.unpriced_models == []


def test_merge_usage_summaries_preserves_per_model_pricing_metadata() -> None:
    merged = merge_usage_summaries(
        [
            {
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
                "estimated_cost_usd": 0.5,
                "cost_estimate_status": "estimated",
                "unpriced_models": [],
                "by_model": {
                    "model-a": {
                        "model_id": "model-a",
                        "input_tokens": 3,
                        "output_tokens": 2,
                        "total_tokens": 5,
                        "estimated_cost_usd": 0.5,
                        "cost_estimate_status": "estimated",
                    }
                },
            },
            {
                "input_tokens": 4,
                "output_tokens": 1,
                "total_tokens": 5,
                "estimated_cost_usd": 0.4,
                "cost_estimate_status": "estimated",
                "unpriced_models": [],
                "by_model": {
                    "model-a": {
                        "model_id": "model-a",
                        "input_tokens": 4,
                        "output_tokens": 1,
                        "total_tokens": 5,
                        "estimated_cost_usd": 0.4,
                        "cost_estimate_status": "estimated",
                    }
                },
            },
        ]
    )

    assert merged.input_tokens == 7
    assert merged.output_tokens == 3
    assert merged.total_tokens == 10
    assert merged.estimated_cost_usd == pytest.approx(0.9)
    assert merged.cost_estimate_status == "estimated"
    assert merged.unpriced_models == []
    assert merged.by_model["model-a"].input_tokens == 7
    assert merged.by_model["model-a"].output_tokens == 3
    assert merged.by_model["model-a"].total_tokens == 10
    assert merged.by_model["model-a"].estimated_cost_usd == pytest.approx(0.9)
    assert merged.by_model["model-a"].cost_estimate_status == "estimated"


def test_merge_usage_summaries_ignores_empty_estimated_components() -> None:
    merged = merge_usage_summaries(
        [
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "cost_estimate_status": "estimated",
            },
            {
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
                "estimated_cost_usd": None,
                "cost_estimate_status": "pricing_unavailable",
                "unpriced_models": ["model-a"],
                "by_model": {
                    "model-a": {
                        "model_id": "model-a",
                        "input_tokens": 3,
                        "output_tokens": 2,
                        "total_tokens": 5,
                        "estimated_cost_usd": None,
                        "cost_estimate_status": "pricing_unavailable",
                    }
                },
            },
        ]
    )

    assert merged.total_tokens == 5
    assert merged.estimated_cost_usd is None
    assert merged.cost_estimate_status == "pricing_unavailable"
    assert merged.unpriced_models == ["model-a"]
    assert merged.by_model["model-a"].estimated_cost_usd is None
    assert merged.by_model["model-a"].cost_estimate_status == "pricing_unavailable"


def test_build_pricing_client_reuses_shared_pricing_tables_across_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_calls = 0

    def fake_fetch_openrouter_pricing(
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
    ) -> tuple[dict[str, tuple[Decimal, Decimal]], dict[str, tuple[Decimal, Decimal]]]:
        nonlocal fetch_calls
        fetch_calls += 1
        assert api_key == "test-key"
        assert base_url == DEFAULT_BASE_URL
        assert timeout_seconds == pytest.approx(10.0)
        pricing = {"openrouter/test-model": (Decimal("0.001"), Decimal("0.002"))}
        return pricing, dict(pricing)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        accounting_module, "_fetch_openrouter_pricing", fake_fetch_openrouter_pricing
    )

    config = ModelConfig(model="openai/gpt-5.4", base_url=DEFAULT_BASE_URL)
    first_client = build_pricing_client_from_configs([config])
    second_client = build_pricing_client_from_configs([config])

    assert first_client.get_text_prices("openrouter/test-model") == (
        Decimal("0.001"),
        Decimal("0.002"),
    )
    assert second_client.get_text_prices("openrouter/test-model") == (
        Decimal("0.001"),
        Decimal("0.002"),
    )
    assert fetch_calls == 1


def test_build_pricing_client_disables_pricing_for_non_openrouter_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    fetch_calls = 0

    def fake_fetch_openrouter_pricing(
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
    ) -> tuple[dict[str, tuple[Decimal, Decimal]], dict[str, tuple[Decimal, Decimal]]]:
        nonlocal fetch_calls
        fetch_calls += 1
        raise AssertionError("pricing fetch should not run for non-OpenRouter base URLs")

    monkeypatch.setattr(
        accounting_module, "_fetch_openrouter_pricing", fake_fetch_openrouter_pricing
    )

    client = build_pricing_client_from_configs(
        [
            ModelConfig(
                model="openai/gpt-5.4",
                base_url="https://proxy.example.test/openrouter/v1",
            )
        ]
    )

    assert client.get_text_prices("openai/gpt-5.4") is None
    assert fetch_calls == 0


def test_build_pricing_client_ignores_non_openrouter_configs_when_pricing_openrouter_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_calls = 0

    def fake_fetch_openrouter_pricing(
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
    ) -> tuple[dict[str, tuple[Decimal, Decimal]], dict[str, tuple[Decimal, Decimal]]]:
        nonlocal fetch_calls
        fetch_calls += 1
        assert api_key == "test-key"
        assert base_url == DEFAULT_BASE_URL
        assert timeout_seconds == pytest.approx(10.0)
        pricing = {"openai/gpt-5.4": (Decimal("0.001"), Decimal("0.002"))}
        return pricing, dict(pricing)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        accounting_module, "_fetch_openrouter_pricing", fake_fetch_openrouter_pricing
    )

    client = build_pricing_client_from_configs(
        [
            ModelConfig(model="openai/gpt-5.4", base_url=DEFAULT_BASE_URL),
            ModelConfig(model="local/demo", base_url="http://localhost:8000/v1"),
        ]
    )

    assert client.get_text_prices("openai/gpt-5.4") == (
        Decimal("0.001"),
        Decimal("0.002"),
    )
    assert client.get_text_prices("local/demo") is None
    assert fetch_calls == 1


def test_extract_response_model_and_token_usage_support_embedding_responses() -> None:
    response = SimpleNamespace(
        model="openai/text-embedding-3-small",
        usage=EmbeddingUsage(prompt_tokens=9, total_tokens=9),
    )

    model_id = extract_response_model(response)
    usage = extract_token_usage(response)

    assert model_id == "openai/text-embedding-3-small"
    assert usage is not None
    assert usage.input_tokens == 9
    assert usage.output_tokens == 0
    assert usage.total_tokens == 9


def test_extract_response_model_prefers_metadata_over_model_attribute() -> None:
    response = SimpleNamespace(
        model="openai/gpt-5.4-20260305openai/gpt-5.4-20260305",
        response_metadata={"model_name": "openai/gpt-5.4-20260305"},
    )

    assert extract_response_model(response) == "openai/gpt-5.4-20260305"


def test_extract_response_model_deduplicates_repeated_model_attribute() -> None:
    response = SimpleNamespace(
        model="openai/gpt-5.4-nano-20260317openai/gpt-5.4-nano-20260317",
    )

    assert extract_response_model(response) == "openai/gpt-5.4-nano-20260317"


def test_build_pricing_client_supports_embedding_only_configs_with_prompt_only_pricing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_calls = 0

    def fake_fetch_openrouter_pricing(
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
    ) -> tuple[dict[str, tuple[Decimal, Decimal]], dict[str, tuple[Decimal, Decimal]]]:
        nonlocal fetch_calls
        fetch_calls += 1
        assert api_key == "test-key"
        assert base_url == DEFAULT_BASE_URL
        assert timeout_seconds == pytest.approx(10.0)
        pricing = {"openai/text-embedding-3-small": (Decimal("0.00002"), Decimal("0"))}
        return pricing, dict(pricing)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        accounting_module, "_fetch_openrouter_pricing", fake_fetch_openrouter_pricing
    )

    client = build_pricing_client_from_configs(
        [EmbeddingModelConfig(model="openai/text-embedding-3-small", backend="openrouter")]
    )

    assert client.get_text_prices("openai/text-embedding-3-small") == (
        Decimal("0.00002"),
        Decimal("0"),
    )
    assert fetch_calls == 1


def test_build_pricing_client_supplements_qwen_embedding_price_when_catalog_omits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_calls = 0

    def fake_fetch_openrouter_pricing(
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
    ) -> tuple[dict[str, tuple[Decimal, Decimal]], dict[str, tuple[Decimal, Decimal]]]:
        nonlocal fetch_calls
        fetch_calls += 1
        assert api_key == "test-key"
        assert base_url == DEFAULT_BASE_URL
        assert timeout_seconds == pytest.approx(10.0)
        pricing = {"openai/gpt-5.4": (Decimal("0.001"), Decimal("0.002"))}
        return pricing, dict(pricing)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        accounting_module, "_fetch_openrouter_pricing", fake_fetch_openrouter_pricing
    )

    client = build_pricing_client_from_configs(
        [EmbeddingModelConfig(model="qwen/qwen3-embedding-8b", backend="openrouter")]
    )

    assert client.get_text_prices("qwen/qwen3-embedding-8b") == (
        Decimal("0.00000001"),
        Decimal("0"),
    )
    assert fetch_calls == 1


def test_build_pricing_client_retries_after_transient_shared_fetch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_calls = 0

    def fake_fetch_openrouter_pricing(
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
    ) -> tuple[dict[str, tuple[Decimal, Decimal]], dict[str, tuple[Decimal, Decimal]]]:
        nonlocal fetch_calls
        fetch_calls += 1
        assert api_key == "test-key"
        assert base_url == DEFAULT_BASE_URL
        assert timeout_seconds == pytest.approx(10.0)
        if fetch_calls == 1:
            raise httpx.ConnectError("pricing unavailable")
        pricing = {"openrouter/test-model": (Decimal("0.001"), Decimal("0.002"))}
        return pricing, dict(pricing)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        accounting_module, "_fetch_openrouter_pricing", fake_fetch_openrouter_pricing
    )

    config = ModelConfig(model="openai/gpt-5.4", base_url=DEFAULT_BASE_URL)
    first_client = build_pricing_client_from_configs([config])
    second_client = build_pricing_client_from_configs([config])

    assert first_client.get_text_prices("openrouter/test-model") is None
    assert second_client.get_text_prices("openrouter/test-model") == (
        Decimal("0.001"),
        Decimal("0.002"),
    )
    assert fetch_calls == 2


def test_build_pricing_client_reuses_non_retryable_failure_cache_across_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetch_calls = 0

    def fake_fetch_openrouter_pricing(
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
    ) -> tuple[dict[str, tuple[Decimal, Decimal]], dict[str, tuple[Decimal, Decimal]]]:
        nonlocal fetch_calls
        fetch_calls += 1
        request = httpx.Request("GET", f"{base_url}/models")
        response = httpx.Response(status_code=401, request=request)
        raise httpx.HTTPStatusError("unauthorized", request=request, response=response)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        accounting_module, "_fetch_openrouter_pricing", fake_fetch_openrouter_pricing
    )

    config = ModelConfig(model="openai/gpt-5.4", base_url=DEFAULT_BASE_URL)
    first_client = build_pricing_client_from_configs([config])
    second_client = build_pricing_client_from_configs([config])

    assert first_client.get_text_prices("openrouter/test-model") is None
    assert second_client.get_text_prices("openrouter/test-model") is None
    assert fetch_calls == 1


def test_parse_openrouter_pricing_table_accepts_prompt_only_entries() -> None:
    pricing_by_model_id, pricing_by_slug = accounting_module.parse_openrouter_pricing_table(
        [
            {
                "id": "openai/text-embedding-3-small",
                "canonical_slug": "openai/text-embedding-3-small",
                "pricing": {"prompt": "0.00002"},
            }
        ]
    )

    assert pricing_by_model_id["openai/text-embedding-3-small"] == (
        Decimal("0.00002"),
        Decimal("0"),
    )
    assert pricing_by_slug["openai/text-embedding-3-small"] == (
        Decimal("0.00002"),
        Decimal("0"),
    )
