"""CSV-backed data loader implementation with optional cached embedding preparation."""

import os

import pandas as pd

from ..accounting import UsageCollector
from ..models import EmbeddingModelConfig, default_embedding_model_config
from .data_loader import DataLoader
from .embeddings import ensure_embeddings_for_items, normalize_item_text, resolve_item_ids


class CsvDataLoader(DataLoader):
    """
    Loads a CSV file with an 'item' column.
    """

    def __init__(
        self,
        path: str,
        item_column: str = "item",
        limit: int | None = None,
        random_seed: int | None = None,
        dataset_name: str | None = None,
        enable_embeddings: bool = False,
        embedding_model_config: EmbeddingModelConfig | None = None,
        embedding_force_recompute: bool = False,
        embedding_batch_size: int = 64,
        usage_collector: UsageCollector | None = None,
    ):
        self.path = path
        self.item_column = item_column
        self.limit = limit
        self.random_seed = random_seed if random_seed is not None else 42
        self.dataset_name = dataset_name or os.path.splitext(os.path.basename(path))[0]
        self.enable_embeddings = enable_embeddings
        self.embedding_model_config = embedding_model_config or default_embedding_model_config()
        self.embedding_force_recompute = embedding_force_recompute
        self.embedding_batch_size = embedding_batch_size
        self.usage_collector = usage_collector
        self._df = self._load_dataframe()
        self._item_context: list[str] | None = None
        self._item_ids: list[str] | None = None
        self._embeddings_by_item_id: dict[str, list[float]] | None = None
        if self.enable_embeddings:
            self._prepare_embeddings()

    def _load_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.path, low_memory=False).drop_duplicates().reset_index(drop=True)
        if self.limit:
            return df.sample(self.limit, random_state=self.random_seed).reset_index(drop=True)
        return df

    def load(self) -> pd.DataFrame:
        return self._df.copy(deep=True)

    def to_context(self) -> list[str]:
        return list(self._resolve_item_context())

    def get_item_ids(self) -> list[str]:
        if self._item_ids is None:
            self._item_ids = resolve_item_ids(self._resolve_item_context())
        return list(self._item_ids)

    def get_unique_item_texts_by_id(self) -> dict[str, str]:
        item_context = self._resolve_item_context()
        item_ids = self.get_item_ids()
        unique_items_by_id: dict[str, str] = {}
        for item_id, item_text in zip(item_ids, item_context, strict=True):
            unique_items_by_id.setdefault(item_id, item_text)
        return unique_items_by_id

    def has_embeddings(self) -> bool:
        return self._embeddings_by_item_id is not None

    def get_embeddings(self) -> dict[str, list[float]]:
        if self._embeddings_by_item_id is None:
            raise ValueError(
                "Embeddings are not available. Enable embeddings when creating the data loader."
            )
        return {item_id: list(vector) for item_id, vector in self._embeddings_by_item_id.items()}

    def _resolve_item_context(self) -> list[str]:
        if self._item_context is not None:
            return self._item_context
        if self.item_column not in self._df.columns:
            raise ValueError(f"Column '{self.item_column}' not found in {self.path}")
        self._item_context = [normalize_item_text(value) for value in self._df[self.item_column]]
        return self._item_context

    def _prepare_embeddings(self) -> None:
        self._embeddings_by_item_id = ensure_embeddings_for_items(
            item_texts_by_id=self.get_unique_item_texts_by_id(),
            dataset_name=self.dataset_name,
            model_config=self.embedding_model_config,
            force_recompute=self.embedding_force_recompute,
            batch_size=self.embedding_batch_size,
            usage_collector=self.usage_collector,
        )
