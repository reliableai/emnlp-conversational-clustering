"""Shared token accounting and OpenRouter cost estimation utilities."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from threading import Lock
from typing import Any, cast

import httpx

from .models import DEFAULT_BASE_URL

_PRICING_STATUS_ESTIMATED = "estimated"
_PRICING_STATUS_PRICING_UNAVAILABLE = "pricing_unavailable"
_PRICING_STATUS_PARTIALLY_ESTIMATED = "partially_estimated"
_ZERO_DECIMAL = Decimal("0")
_PricingCacheKey = tuple[str, tuple[str, ...]]
_SHARED_PRICING_CACHE_LOCK = Lock()
_SHARED_PRICING_CACHE: dict[
    _PricingCacheKey,
    tuple[dict[str, tuple[Decimal, Decimal]], dict[str, tuple[Decimal, Decimal]]],
] = {}
_SHARED_PRICING_FAILURE_CACHE: set[_PricingCacheKey] = set()
_OPENROUTER_EMBEDDING_PRICES: dict[str, tuple[Decimal, Decimal]] = {
    # OpenRouter's /models catalog currently omits embedding models, but their public
    # model pages list input-only prices per million tokens; values here are per token.
    "openai/text-embedding-3-small": (Decimal("0.00000002"), _ZERO_DECIMAL),
    "openai/text-embedding-3-large": (Decimal("0.00000013"), _ZERO_DECIMAL),
    "qwen/qwen3-embedding-8b": (Decimal("0.00000001"), _ZERO_DECIMAL),
}


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class ModelUsage:
    model_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None
    cost_estimate_status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "cost_estimate_status": self.cost_estimate_status,
        }


@dataclass(frozen=True)
class UsageSummary:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None
    cost_estimate_status: str
    by_model: dict[str, ModelUsage]
    unpriced_models: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "cost_estimate_status": self.cost_estimate_status,
            "unpriced_models": list(self.unpriced_models),
            "by_model": {
                model_id: model_usage.to_dict()
                for model_id, model_usage in sorted(self.by_model.items())
            },
        }


@dataclass(frozen=True)
class RunComponentAccounting:
    component: str
    usage: UsageSummary

    def to_dict(self) -> dict[str, object]:
        payload = self.usage.to_dict()
        payload["component"] = self.component
        return payload


@dataclass
class _MutableUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class _MergedModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: Decimal = _ZERO_DECIMAL
    saw_estimated_cost: bool = False
    saw_unpriced_usage: bool = False


@dataclass
class UsageCollector:
    """Collect token usage across one logical runtime component."""

    _usage_by_model: dict[str, _MutableUsage] = field(default_factory=dict)

    def record(
        self,
        model_id: str | None,
        usage: TokenUsage | Mapping[str, object] | None,
        *,
        source: str | None = None,
    ) -> None:
        _ = source
        normalized_usage = coerce_token_usage(usage)
        if normalized_usage is None or model_id is None or not model_id.strip():
            return
        normalized_model_id = model_id.strip()
        current = self._usage_by_model.get(normalized_model_id)
        if current is None:
            self._usage_by_model[normalized_model_id] = _MutableUsage(
                input_tokens=normalized_usage.input_tokens,
                output_tokens=normalized_usage.output_tokens,
                total_tokens=normalized_usage.total_tokens,
            )
            return
        current.input_tokens += normalized_usage.input_tokens
        current.output_tokens += normalized_usage.output_tokens
        current.total_tokens += normalized_usage.total_tokens

    def merge(self, other: UsageCollector | UsageSummary | Mapping[str, Any] | None) -> None:
        if other is None:
            return
        if isinstance(other, UsageCollector):
            for model_id, usage in other._usage_by_model.items():
                self.record(model_id, usage.__dict__)
            return
        usage_summary = coerce_usage_summary(other)
        if usage_summary is None:
            return
        for model_id, model_usage in usage_summary.by_model.items():
            self.record(
                model_id,
                {
                    "input_tokens": model_usage.input_tokens,
                    "output_tokens": model_usage.output_tokens,
                    "total_tokens": model_usage.total_tokens,
                },
            )

    def snapshot(self, pricer: OpenRouterPricingClient | None) -> UsageSummary:
        by_model: dict[str, ModelUsage] = {}
        total_input_tokens = 0
        total_output_tokens = 0
        total_tokens = 0
        total_cost = _ZERO_DECIMAL
        unpriced_models: list[str] = []
        estimated_models = 0

        for model_id in sorted(self._usage_by_model):
            usage = self._usage_by_model[model_id]
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens
            total_tokens += usage.total_tokens
            estimated_cost_usd: float | None = None
            status = _PRICING_STATUS_PRICING_UNAVAILABLE
            if pricer is not None:
                text_prices = pricer.get_text_prices(model_id)
            else:
                text_prices = None
            if text_prices is not None:
                prompt_price, completion_price = text_prices
                estimated_cost = (Decimal(usage.input_tokens) * prompt_price) + (
                    Decimal(usage.output_tokens) * completion_price
                )
                total_cost += estimated_cost
                estimated_cost_usd = float(estimated_cost)
                status = _PRICING_STATUS_ESTIMATED
                estimated_models += 1
            else:
                unpriced_models.append(model_id)
            by_model[model_id] = ModelUsage(
                model_id=model_id,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                estimated_cost_usd=estimated_cost_usd,
                cost_estimate_status=status,
            )

        overall_status = _PRICING_STATUS_ESTIMATED
        estimated_cost_usd: float | None = float(total_cost)
        if by_model and estimated_models == 0:
            overall_status = _PRICING_STATUS_PRICING_UNAVAILABLE
            estimated_cost_usd = None
        elif unpriced_models:
            overall_status = _PRICING_STATUS_PARTIALLY_ESTIMATED
        elif not by_model:
            estimated_cost_usd = 0.0

        return UsageSummary(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
            cost_estimate_status=overall_status,
            by_model=by_model,
            unpriced_models=unpriced_models,
        )


class OpenRouterPricingClient:
    """Fetch and cache OpenRouter pricing metadata for cost estimation."""

    def __init__(
        self,
        *,
        api_key_envs: Iterable[str] | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 10.0,
        supplemental_prices: Mapping[str, tuple[Decimal, Decimal]] | None = None,
    ) -> None:
        resolved_api_key_envs = ("OPENROUTER_API_KEY",) if api_key_envs is None else api_key_envs
        self._api_key_envs = tuple(dict.fromkeys(resolved_api_key_envs))
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._supplemental_prices = dict(supplemental_prices or {})
        self._lock = Lock()
        self._pricing_by_model_id: dict[str, tuple[Decimal, Decimal]] | None = None
        self._pricing_by_canonical_slug: dict[str, tuple[Decimal, Decimal]] | None = None
        self._load_failed = False

    def get_text_prices(self, model_id: str) -> tuple[Decimal, Decimal] | None:
        normalized_model_id = model_id.strip()
        if not normalized_model_id:
            return None
        self._ensure_loaded()
        if self._pricing_by_model_id is None or self._pricing_by_canonical_slug is None:
            return self._supplemental_prices.get(normalized_model_id)
        if normalized_model_id in self._pricing_by_model_id:
            return self._pricing_by_model_id[normalized_model_id]
        return self._pricing_by_canonical_slug.get(
            normalized_model_id
        ) or self._supplemental_prices.get(normalized_model_id)

    def _ensure_loaded(self) -> None:
        if self._pricing_by_model_id is not None or self._load_failed:
            return
        with self._lock:
            if self._pricing_by_model_id is not None or self._load_failed:
                return
            cached_pricing = _get_shared_pricing_tables(
                base_url=self._base_url,
                api_key_envs=self._api_key_envs,
            )
            if cached_pricing is not None:
                self._pricing_by_model_id, self._pricing_by_canonical_slug = cached_pricing
                return
            if _has_shared_pricing_failure(
                base_url=self._base_url,
                api_key_envs=self._api_key_envs,
            ):
                self._load_failed = True
                return
            api_key = self._resolve_api_key()
            if api_key is None:
                self._load_failed = True
                return
            try:
                pricing_by_model_id, pricing_by_canonical_slug = _fetch_openrouter_pricing(
                    api_key=api_key,
                    base_url=self._base_url,
                    timeout_seconds=self._timeout_seconds,
                )
            except Exception as error:
                self._load_failed = True
                if _is_non_retryable_pricing_failure(error):
                    _store_shared_pricing_failure(
                        base_url=self._base_url,
                        api_key_envs=self._api_key_envs,
                    )
                return
            self._pricing_by_model_id = pricing_by_model_id
            self._pricing_by_canonical_slug = pricing_by_canonical_slug
            _store_shared_pricing_tables(
                base_url=self._base_url,
                api_key_envs=self._api_key_envs,
                pricing_by_model_id=pricing_by_model_id,
                pricing_by_canonical_slug=pricing_by_canonical_slug,
            )

    def _resolve_api_key(self) -> str | None:
        for env_name in self._api_key_envs:
            api_key = os.getenv(env_name)
            if api_key:
                return api_key
        return None


def build_pricing_client_from_configs(
    configs: Iterable[object],
    *,
    base_url: str | None = None,
) -> OpenRouterPricingClient:
    resolved_base_url = base_url.rstrip("/") if base_url is not None else None
    if resolved_base_url is not None and resolved_base_url != DEFAULT_BASE_URL.rstrip("/"):
        return OpenRouterPricingClient(api_key_envs=())
    env_names: list[str] = []
    supplemental_prices: dict[str, tuple[Decimal, Decimal]] = {}
    saw_priceable_config = False
    for config in configs:
        if not _supports_openrouter_catalog_pricing(config):
            continue
        config_base_url = getattr(config, "base_url", None)
        if not isinstance(config_base_url, str):
            continue
        saw_priceable_config = True
        normalized_config_base_url = config_base_url.rstrip("/")
        if resolved_base_url is None:
            resolved_base_url = normalized_config_base_url
        elif normalized_config_base_url != resolved_base_url:
            return OpenRouterPricingClient(api_key_envs=())
        api_key_env = getattr(config, "api_key_env", None)
        if isinstance(api_key_env, str) and api_key_env.strip():
            env_names.append(api_key_env.strip())
        model_id = getattr(config, "model", None)
        if isinstance(model_id, str):
            static_prices = _OPENROUTER_EMBEDDING_PRICES.get(model_id.strip())
            if static_prices is not None:
                supplemental_prices[model_id.strip()] = static_prices
    if not saw_priceable_config:
        return OpenRouterPricingClient(api_key_envs=())
    if resolved_base_url is None:
        resolved_base_url = DEFAULT_BASE_URL.rstrip("/")
    env_names.append("OPENROUTER_API_KEY")
    return OpenRouterPricingClient(
        api_key_envs=env_names,
        base_url=resolved_base_url,
        supplemental_prices=supplemental_prices,
    )


def _supports_openrouter_catalog_pricing(config: object) -> bool:
    backend = getattr(config, "backend", None)
    if isinstance(backend, str) and backend.strip() and backend.strip() != "openrouter":
        return False
    config_base_url = getattr(config, "base_url", None)
    return isinstance(config_base_url, str) and (
        config_base_url.rstrip("/") == DEFAULT_BASE_URL.rstrip("/")
    )


def _get_shared_pricing_tables(
    *,
    base_url: str,
    api_key_envs: tuple[str, ...],
) -> tuple[dict[str, tuple[Decimal, Decimal]], dict[str, tuple[Decimal, Decimal]]] | None:
    cache_key = (base_url, api_key_envs)
    with _SHARED_PRICING_CACHE_LOCK:
        return _SHARED_PRICING_CACHE.get(cache_key)


def _has_shared_pricing_failure(
    *,
    base_url: str,
    api_key_envs: tuple[str, ...],
) -> bool:
    cache_key = (base_url, api_key_envs)
    with _SHARED_PRICING_CACHE_LOCK:
        return cache_key in _SHARED_PRICING_FAILURE_CACHE


def _store_shared_pricing_tables(
    *,
    base_url: str,
    api_key_envs: tuple[str, ...],
    pricing_by_model_id: dict[str, tuple[Decimal, Decimal]],
    pricing_by_canonical_slug: dict[str, tuple[Decimal, Decimal]],
) -> None:
    cache_key = (base_url, api_key_envs)
    with _SHARED_PRICING_CACHE_LOCK:
        _SHARED_PRICING_FAILURE_CACHE.discard(cache_key)
        _SHARED_PRICING_CACHE[cache_key] = (
            pricing_by_model_id,
            pricing_by_canonical_slug,
        )


def _store_shared_pricing_failure(
    *,
    base_url: str,
    api_key_envs: tuple[str, ...],
) -> None:
    cache_key = (base_url, api_key_envs)
    with _SHARED_PRICING_CACHE_LOCK:
        _SHARED_PRICING_FAILURE_CACHE.add(cache_key)


def _is_non_retryable_pricing_failure(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        return 400 <= status_code < 500 and status_code not in {408, 429}
    return False


def empty_usage_summary() -> UsageSummary:
    return UsageSummary(
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        estimated_cost_usd=0.0,
        cost_estimate_status=_PRICING_STATUS_ESTIMATED,
        by_model={},
        unpriced_models=[],
    )


def merge_usage_summaries(
    summaries: Iterable[UsageSummary | Mapping[str, Any] | None],
) -> UsageSummary:
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    estimated_cost = _ZERO_DECIMAL
    saw_any = False
    saw_estimated_cost = False
    has_unpriced_usage = False
    merged_by_model: dict[str, _MergedModelUsage] = {}
    unpriced_models: set[str] = set()
    for summary_like in summaries:
        summary = coerce_usage_summary(summary_like)
        if summary is None:
            continue
        saw_any = True
        summary_has_usage = _usage_has_tokens(
            input_tokens=summary.input_tokens,
            output_tokens=summary.output_tokens,
            total_tokens=summary.total_tokens,
        )
        total_input_tokens += summary.input_tokens
        total_output_tokens += summary.output_tokens
        total_tokens += summary.total_tokens
        if summary.estimated_cost_usd is not None and summary_has_usage:
            estimated_cost += Decimal(str(summary.estimated_cost_usd))
            saw_estimated_cost = True
        if summary.cost_estimate_status != _PRICING_STATUS_ESTIMATED and summary_has_usage:
            has_unpriced_usage = True
        if summary_has_usage:
            unpriced_models.update(summary.unpriced_models)
        for model_id, model_usage in summary.by_model.items():
            merged_usage = merged_by_model.setdefault(model_id, _MergedModelUsage())
            merged_usage.input_tokens += model_usage.input_tokens
            merged_usage.output_tokens += model_usage.output_tokens
            merged_usage.total_tokens += model_usage.total_tokens
            model_has_usage = _usage_has_tokens(
                input_tokens=model_usage.input_tokens,
                output_tokens=model_usage.output_tokens,
                total_tokens=model_usage.total_tokens,
            )
            if model_usage.estimated_cost_usd is not None and model_has_usage:
                merged_usage.estimated_cost_usd += Decimal(str(model_usage.estimated_cost_usd))
                merged_usage.saw_estimated_cost = True
            if model_has_usage and (
                model_usage.cost_estimate_status != _PRICING_STATUS_ESTIMATED
                or model_usage.estimated_cost_usd is None
            ):
                merged_usage.saw_unpriced_usage = True
    if not saw_any:
        return empty_usage_summary()
    by_model: dict[str, ModelUsage] = {}
    for model_id in sorted(merged_by_model):
        merged_usage = merged_by_model[model_id]
        model_estimated_cost_usd: float | None = (
            float(merged_usage.estimated_cost_usd) if merged_usage.saw_estimated_cost else None
        )
        model_status = _PRICING_STATUS_ESTIMATED
        if merged_usage.total_tokens == 0:
            model_estimated_cost_usd = (
                0.0 if model_estimated_cost_usd is None else model_estimated_cost_usd
            )
        elif merged_usage.saw_unpriced_usage and not merged_usage.saw_estimated_cost:
            model_status = _PRICING_STATUS_PRICING_UNAVAILABLE
            model_estimated_cost_usd = None
        elif merged_usage.saw_unpriced_usage:
            model_status = _PRICING_STATUS_PARTIALLY_ESTIMATED
        if model_status != _PRICING_STATUS_ESTIMATED:
            unpriced_models.add(model_id)
        by_model[model_id] = ModelUsage(
            model_id=model_id,
            input_tokens=merged_usage.input_tokens,
            output_tokens=merged_usage.output_tokens,
            total_tokens=merged_usage.total_tokens,
            estimated_cost_usd=model_estimated_cost_usd,
            cost_estimate_status=model_status,
        )
    estimated_cost_usd: float | None = float(estimated_cost) if saw_estimated_cost else None
    status = _PRICING_STATUS_ESTIMATED
    if total_tokens == 0:
        estimated_cost_usd = 0.0 if estimated_cost_usd is None else estimated_cost_usd
    elif not saw_estimated_cost:
        status = _PRICING_STATUS_PRICING_UNAVAILABLE
        estimated_cost_usd = None
    elif has_unpriced_usage or unpriced_models:
        status = _PRICING_STATUS_PARTIALLY_ESTIMATED
    return UsageSummary(
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd,
        cost_estimate_status=status,
        by_model=by_model,
        unpriced_models=sorted(unpriced_models),
    )


def coerce_token_usage(usage: TokenUsage | Mapping[str, object] | None) -> TokenUsage | None:
    if usage is None:
        return None
    if isinstance(usage, TokenUsage):
        return usage
    input_tokens = _coerce_int(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
    output_tokens = _coerce_int(usage.get("output_tokens", usage.get("completion_tokens", 0)))
    total_tokens_raw = usage.get("total_tokens")
    total_tokens = _coerce_int(total_tokens_raw) if total_tokens_raw is not None else None
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def extract_token_usage(message: Any) -> TokenUsage | None:
    usage = getattr(message, "usage", None)
    if _usage_mapping(usage) is not None:
        return coerce_token_usage(_usage_mapping(usage))

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        token_usage = additional_kwargs.get("token_usage")
        if isinstance(token_usage, dict) and token_usage:
            return coerce_token_usage(token_usage)

    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        return coerce_token_usage(usage)

    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
        if isinstance(token_usage, dict) and token_usage:
            return coerce_token_usage(token_usage)

    return None


def extract_response_model(message: Any, *, fallback_model: str | None = None) -> str | None:
    response_metadata = getattr(message, "response_metadata", None)
    if isinstance(response_metadata, dict):
        for key in ("model_name", "model", "response_model"):
            raw_model = response_metadata.get(key)
            if isinstance(raw_model, str) and raw_model.strip():
                return _normalize_response_model_id(raw_model)
    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        raw_model = additional_kwargs.get("model_name")
        if isinstance(raw_model, str) and raw_model.strip():
            return _normalize_response_model_id(raw_model)
    raw_model = getattr(message, "model", None)
    if isinstance(raw_model, str) and raw_model.strip():
        return _normalize_response_model_id(raw_model)
    if fallback_model is not None and fallback_model.strip():
        return fallback_model.strip()
    return None


def coerce_usage_summary(
    summary_like: UsageSummary | Mapping[str, Any] | None,
) -> UsageSummary | None:
    if summary_like is None:
        return None
    if isinstance(summary_like, UsageSummary):
        return summary_like
    raw_by_model = summary_like.get("by_model", {})
    by_model: dict[str, ModelUsage] = {}
    if isinstance(raw_by_model, Mapping):
        for model_id, raw_usage in raw_by_model.items():
            if not isinstance(model_id, str) or not isinstance(raw_usage, Mapping):
                continue
            estimated_cost_usd = raw_usage.get("estimated_cost_usd")
            by_model[model_id] = ModelUsage(
                model_id=model_id,
                input_tokens=int(raw_usage.get("input_tokens", 0)),
                output_tokens=int(raw_usage.get("output_tokens", 0)),
                total_tokens=int(raw_usage.get("total_tokens", 0)),
                estimated_cost_usd=(
                    float(estimated_cost_usd)
                    if isinstance(estimated_cost_usd, int | float)
                    else None
                ),
                cost_estimate_status=str(
                    raw_usage.get("cost_estimate_status", _PRICING_STATUS_PRICING_UNAVAILABLE)
                ),
            )
    estimated_cost_usd = summary_like.get("estimated_cost_usd")
    raw_unpriced_models = summary_like.get("unpriced_models", [])
    unpriced_models = (
        [model_id for model_id in raw_unpriced_models if isinstance(model_id, str)]
        if isinstance(raw_unpriced_models, list)
        else []
    )
    return UsageSummary(
        input_tokens=int(summary_like.get("input_tokens", 0)),
        output_tokens=int(summary_like.get("output_tokens", 0)),
        total_tokens=int(summary_like.get("total_tokens", 0)),
        estimated_cost_usd=(
            float(estimated_cost_usd) if isinstance(estimated_cost_usd, int | float) else None
        ),
        cost_estimate_status=str(
            summary_like.get("cost_estimate_status", _PRICING_STATUS_PRICING_UNAVAILABLE)
        ),
        by_model=by_model,
        unpriced_models=sorted(set(unpriced_models)),
    )


def _fetch_openrouter_pricing(
    *,
    api_key: str,
    base_url: str,
    timeout_seconds: float,
) -> tuple[dict[str, tuple[Decimal, Decimal]], dict[str, tuple[Decimal, Decimal]]]:
    response = httpx.get(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    if not isinstance(data, list):
        raise ValueError("OpenRouter models payload is missing 'data' list")
    return parse_openrouter_pricing_table(data)


def parse_openrouter_pricing_table(
    data: list[object],
) -> tuple[dict[str, tuple[Decimal, Decimal]], dict[str, tuple[Decimal, Decimal]]]:
    pricing_by_model_id: dict[str, tuple[Decimal, Decimal]] = {}
    pricing_by_canonical_slug: dict[str, tuple[Decimal, Decimal]] = {}
    for raw_entry in data:
        if not isinstance(raw_entry, Mapping):
            continue
        entry = cast(Mapping[str, object], raw_entry)
        model_id = entry.get("id")
        pricing = entry.get("pricing")
        if (
            not isinstance(model_id, str)
            or not model_id.strip()
            or not isinstance(pricing, Mapping)
        ):
            continue
        pricing_mapping = cast(Mapping[str, object], pricing)
        prompt_price = _parse_decimal(pricing_mapping.get("prompt"))
        completion_price = _parse_decimal(pricing_mapping.get("completion"))
        if prompt_price is None:
            continue
        if completion_price is None:
            completion_price = _ZERO_DECIMAL
        if completion_price < 0:
            continue
        normalized_model_id = model_id.strip()
        pricing_by_model_id[normalized_model_id] = (prompt_price, completion_price)
        canonical_slug = entry.get("canonical_slug")
        if isinstance(canonical_slug, str) and canonical_slug.strip():
            pricing_by_canonical_slug[canonical_slug.strip()] = (prompt_price, completion_price)
    return pricing_by_model_id, pricing_by_canonical_slug


def _parse_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _coerce_int(value: object) -> int:
    if value is None:
        return 0
    return int(cast(Any, value))


def _usage_mapping(usage: object) -> Mapping[str, object] | None:
    if isinstance(usage, Mapping):
        return cast(Mapping[str, object], usage)
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        dumped_usage = model_dump()
        if isinstance(dumped_usage, Mapping):
            return cast(Mapping[str, object], dumped_usage)
    return None


def _normalize_response_model_id(model_id: str) -> str:
    normalized_model_id = model_id.strip()
    midpoint, remainder = divmod(len(normalized_model_id), 2)
    if remainder == 0 and normalized_model_id[:midpoint] == normalized_model_id[midpoint:]:
        return normalized_model_id[:midpoint]
    return normalized_model_id


def _usage_has_tokens(*, input_tokens: int, output_tokens: int, total_tokens: int) -> bool:
    return total_tokens > 0 or input_tokens > 0 or output_tokens > 0
