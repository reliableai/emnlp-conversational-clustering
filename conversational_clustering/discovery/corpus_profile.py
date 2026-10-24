"""Deterministic descriptive statistics for a corpus, used as the first-message facts strip.

Produces a typed CorpusProfile: per-column kind, fill rate, cardinality, and a shape-appropriate
summary (value counts, numeric spread, or text-length spread). No LLM, no embeddings.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

# A column whose distinct-value count is at or below this is treated as categorical rather than
# high-cardinality; the item/text column is detected separately by mean length.
_MAX_CATEGORICAL_UNIQUE = 50
_TEXT_MEAN_CHAR_THRESHOLD = 60
_TOP_VALUES_LIMIT = 10

ColumnKind = Literal[
    "constant",
    "boolean",
    "numeric",
    "categorical",
    "high_cardinality",
    "text",
    "empty",
]


class ValueCount(BaseModel):
    value: str
    count: int


class NumericSummary(BaseModel):
    minimum: float
    maximum: float
    mean: float
    median: float
    std: float


class LengthSummary(BaseModel):
    unit: Literal["characters", "words"]
    minimum: int
    median: float
    mean: float
    maximum: int


class ColumnProfile(BaseModel):
    name: str
    kind: ColumnKind
    non_null: int
    fill_rate: float
    n_unique: int
    top_values: list[ValueCount] = Field(default_factory=list)
    numeric_summary: NumericSummary | None = None
    text_length: list[LengthSummary] = Field(default_factory=list)
    is_item_column: bool = False


class CorpusProfile(BaseModel):
    dataset_name: str
    n_rows: int
    n_columns: int
    item_column: str
    columns: list[ColumnProfile]


def _length_summaries(series: pd.Series) -> list[LengthSummary]:
    text = series.dropna().astype(str)
    if text.empty:
        return []
    char_len = text.str.len()
    word_len = text.str.split().map(len)
    return [
        LengthSummary(
            unit="characters",
            minimum=int(char_len.min()),
            median=float(char_len.median()),
            mean=float(char_len.mean()),
            maximum=int(char_len.max()),
        ),
        LengthSummary(
            unit="words",
            minimum=int(word_len.min()),
            median=float(word_len.median()),
            mean=float(word_len.mean()),
            maximum=int(word_len.max()),
        ),
    ]


def _top_values(series: pd.Series) -> list[ValueCount]:
    counts = series.dropna().astype(str).value_counts().head(_TOP_VALUES_LIMIT)
    return [ValueCount(value=str(value), count=int(count)) for value, count in counts.items()]


def _numeric_summary(series: pd.Series) -> NumericSummary:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return NumericSummary(
        minimum=float(numeric.min()),
        maximum=float(numeric.max()),
        mean=float(numeric.mean()),
        median=float(numeric.median()),
        std=float(numeric.std(ddof=0)),
    )


def _infer_kind(
    series: pd.Series, *, n_rows: int, mean_char_len: float, is_item: bool
) -> ColumnKind:
    n_unique = int(series.nunique(dropna=True))
    if series.dropna().empty:
        return "empty"
    if n_unique <= 1:
        return "constant"
    if pd.api.types.is_bool_dtype(series) or n_unique == 2:
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if is_item or mean_char_len >= _TEXT_MEAN_CHAR_THRESHOLD:
        return "text"
    if n_unique <= max(_MAX_CATEGORICAL_UNIQUE, int(0.2 * n_rows)):
        return "categorical"
    return "high_cardinality"


def profile_column(series: pd.Series, *, name: str, n_rows: int, is_item: bool) -> ColumnProfile:
    non_null = int(series.notna().sum())
    n_unique = int(series.nunique(dropna=True))
    text = series.dropna().astype(str)
    mean_char_len = float(text.str.len().mean()) if not text.empty else 0.0
    kind = _infer_kind(series, n_rows=n_rows, mean_char_len=mean_char_len, is_item=is_item)

    profile = ColumnProfile(
        name=name,
        kind=kind,
        non_null=non_null,
        fill_rate=round(non_null / n_rows, 4) if n_rows else 0.0,
        n_unique=n_unique,
        is_item_column=is_item,
    )
    if kind == "text":
        profile.text_length = _length_summaries(series)
    elif kind == "numeric":
        profile.numeric_summary = _numeric_summary(series)
    elif kind in ("boolean", "categorical", "high_cardinality", "constant"):
        profile.top_values = _top_values(series)
    return profile


def profile_corpus(df: pd.DataFrame, *, dataset_name: str, item_column: str) -> CorpusProfile:
    if item_column not in df.columns:
        raise ValueError(
            f"item_column '{item_column}' not found in dataframe columns: {list(df.columns)}"
        )
    n_rows = int(len(df))
    if n_rows == 0:
        raise ValueError(
            f"Dataset '{dataset_name}' is empty; cannot profile a corpus with no rows."
        )
    columns = [
        profile_column(df[name], name=str(name), n_rows=n_rows, is_item=(name == item_column))
        for name in df.columns
    ]
    return CorpusProfile(
        dataset_name=dataset_name,
        n_rows=n_rows,
        n_columns=int(df.shape[1]),
        item_column=item_column,
        columns=columns,
    )
