"""Unit tests for CSV-backed data loading and context generation behavior."""

from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from conversational_clustering.accounting import UsageCollector
from conversational_clustering.data_processing.csv_loader import CsvDataLoader
from conversational_clustering.models import EmbeddingModelConfig


def _write_csv(directory: Path, rows: list[dict], filename: str = "items.csv") -> Path:
    path = directory / filename
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_without_limit_returns_all_rows(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        [
            {"item": "one", "value": 1},
            {"item": "two", "value": 2},
            {"item": "three", "value": 3},
        ],
    )
    loader = CsvDataLoader(path=str(path))

    loaded = loader.load()

    assert len(loaded) == 3
    assert loaded["item"].tolist() == ["one", "two", "three"]


def test_load_returns_copy_not_shared_dataframe(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, [{"item": "one"}, {"item": "two"}])
    loader = CsvDataLoader(path=str(path))

    loaded = loader.load()
    loaded.loc[0, "item"] = "changed"

    reloaded = loader.load()
    assert reloaded.loc[0, "item"] == "one"


def test_load_with_limit_is_deterministic_for_fixed_random_seed(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        [{"item": f"item-{i}"} for i in range(10)],
    )
    loader = CsvDataLoader(path=str(path), limit=4, random_seed=7)

    first_sample = loader.load()["item"].tolist()
    second_sample = loader.load()["item"].tolist()

    assert len(first_sample) == 4
    assert first_sample == second_sample


def test_to_context_coerces_items_to_strings(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        [{"item": 1}, {"item": 2}, {"item": None}],
    )
    loader = CsvDataLoader(path=str(path), item_column="item")

    context = loader.to_context()

    assert context[:2] == ["1.0", "2.0"]
    assert pd.isna(context[2]) or context[2] == "nan"


def test_to_context_raises_when_item_column_is_missing(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        [{"title": "a"}, {"title": "b"}],
    )
    loader = CsvDataLoader(path=str(path), item_column="item")

    with pytest.raises(ValueError, match="Column 'item' not found"):
        loader.to_context()


def test_get_item_ids_is_deterministic_for_same_dataset(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, [{"item": "Alpha"}, {"item": "Beta"}])
    loader = CsvDataLoader(path=str(path), item_column="item")

    first = loader.get_item_ids()
    second = loader.get_item_ids()

    assert len(first) == 2
    assert first == second


def test_get_embeddings_raises_when_embeddings_are_disabled(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, [{"item": "Alpha"}])
    loader = CsvDataLoader(path=str(path), item_column="item")

    with pytest.raises(ValueError, match="Embeddings are not available"):
        loader.get_embeddings()


def test_loader_with_embeddings_uses_embedding_helper(tmp_path: Path) -> None:
    path = _write_csv(tmp_path, [{"item": "Alpha"}, {"item": "Beta"}])
    collector = UsageCollector()
    with mock.patch(
        "conversational_clustering.data_processing.csv_loader.ensure_embeddings_for_items",
        return_value={"alpha_id": [0.1, 0.2], "beta_id": [0.3, 0.4]},
    ) as ensure_embeddings:
        loader = CsvDataLoader(
            path=str(path),
            item_column="item",
            dataset_name="demo",
            enable_embeddings=True,
            embedding_model_config=EmbeddingModelConfig(model="openai/text-embedding-3-small"),
            embedding_batch_size=10,
            usage_collector=collector,
        )

    assert loader.has_embeddings() is True
    assert loader.get_embeddings() == {"alpha_id": [0.1, 0.2], "beta_id": [0.3, 0.4]}
    ensure_embeddings.assert_called_once()
    kwargs = ensure_embeddings.call_args.kwargs
    assert kwargs["dataset_name"] == "demo"
    assert kwargs["batch_size"] == 10
    assert kwargs["usage_collector"] is collector
