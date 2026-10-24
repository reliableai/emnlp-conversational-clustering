"""Dataset item record helpers for compact, deterministic LLM-facing item identity."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetItemRecord:
    uuid: str
    value: str


def normalize_item_text(item: object) -> str:
    return str(item).strip()


def build_dataset_item_uuid(index: int) -> str:
    if index < 0:
        raise ValueError("index must be >= 0")
    return f"u{_encode_base36(index + 1)}"


def build_dataset_item_records(items: Sequence[object]) -> list[DatasetItemRecord]:
    return [
        DatasetItemRecord(uuid=build_dataset_item_uuid(index), value=normalize_item_text(item))
        for index, item in enumerate(items)
    ]


def _encode_base36(value: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    if value < 0:
        raise ValueError("value must be >= 0")
    if value == 0:
        return "0"

    encoded: list[str] = []
    remaining = value
    while remaining:
        remaining, digit = divmod(remaining, 36)
        encoded.append(digits[digit])
    return "".join(reversed(encoded))
