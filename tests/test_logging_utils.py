"""Unit tests for token usage normalization and extraction from model message payloads."""

from types import SimpleNamespace

from conversational_clustering.logging_utils import _normalize_token_usage, extract_token_usage


def test_normalize_prefers_standard_input_output_keys() -> None:
    normalized = _normalize_token_usage({"input_tokens": 10, "output_tokens": 4})

    assert normalized == {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}


def test_normalize_falls_back_to_prompt_completion_keys() -> None:
    normalized = _normalize_token_usage({"prompt_tokens": 7, "completion_tokens": 2})

    assert normalized == {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9}


def test_normalize_respects_explicit_total_tokens() -> None:
    normalized = _normalize_token_usage({"input_tokens": 7, "output_tokens": 2, "total_tokens": 99})

    assert normalized == {"input_tokens": 7, "output_tokens": 2, "total_tokens": 99}


def test_extract_uses_additional_kwargs_token_usage_first() -> None:
    message = SimpleNamespace(
        additional_kwargs={"token_usage": {"input_tokens": 3, "output_tokens": 2}},
        usage_metadata={"input_tokens": 100, "output_tokens": 100},
        response_metadata={"token_usage": {"input_tokens": 100, "output_tokens": 100}},
    )

    extracted = extract_token_usage(message)

    assert extracted == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}


def test_extract_falls_back_to_usage_metadata() -> None:
    message = SimpleNamespace(
        additional_kwargs={},
        usage_metadata={"prompt_tokens": 11, "completion_tokens": 9},
        response_metadata=None,
    )

    extracted = extract_token_usage(message)

    assert extracted == {"input_tokens": 11, "output_tokens": 9, "total_tokens": 20}


def test_extract_reads_response_metadata_token_usage() -> None:
    message = SimpleNamespace(
        additional_kwargs=None,
        usage_metadata=None,
        response_metadata={"token_usage": {"input_tokens": 5, "output_tokens": 1}},
    )

    extracted = extract_token_usage(message)

    assert extracted == {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6}


def test_extract_reads_response_metadata_usage_alias() -> None:
    message = SimpleNamespace(
        additional_kwargs=None,
        usage_metadata=None,
        response_metadata={"usage": {"prompt_tokens": 8, "completion_tokens": 3}},
    )

    extracted = extract_token_usage(message)

    assert extracted == {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11}


def test_extract_returns_none_when_usage_is_absent() -> None:
    message = SimpleNamespace(
        additional_kwargs={},
        usage_metadata={},
        response_metadata={},
    )

    assert extract_token_usage(message) is None
