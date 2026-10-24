"""Logging utilities for structured JSON events and token accounting metadata."""

import json
import logging
import os
import sys
from typing import Any

from .accounting import coerce_token_usage
from .accounting import extract_token_usage as _extract_accounting_token_usage


def _normalize_token_usage(token_usage: dict[str, object]) -> dict[str, int]:
    normalized_usage = coerce_token_usage(token_usage)
    assert normalized_usage is not None
    return normalized_usage.to_dict()


def extract_token_usage(message: Any) -> dict[str, int] | None:
    normalized_usage = _extract_accounting_token_usage(message)
    if normalized_usage is None:
        return None
    return normalized_usage.to_dict()


def configure_logger(log_path: str = "data/logs/agent.log") -> logging.Logger:
    logger = logging.getLogger("conversational_clustering")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def configure_console_logger() -> logging.Logger:
    logger = logging.getLogger("conversational_clustering")
    logger.setLevel(logging.INFO)

    has_stream_handler = any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )
    if has_stream_handler:
        return logger

    handler = logging.StreamHandler(stream=sys.stderr)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def log_json(logger: logging.Logger, event: str, payload: dict[str, Any]) -> None:
    logger.info(json.dumps({"event": event, **payload}, ensure_ascii=True))
