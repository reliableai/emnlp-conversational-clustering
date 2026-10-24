"""Abstract and in-memory data loader interfaces for dataset ingestion."""

from abc import ABC, abstractmethod

import pandas as pd


class DataLoader(ABC):
    """
    Abstract base class for data loaders
    """

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Loads data from the source and returns it as a standardized pandas DataFrame
        """
        pass

    @abstractmethod
    def to_context(self) -> list[str]:
        """
        Outputs list of all datapoints in string format for injestion into LLM context.
        """
        pass

    @abstractmethod
    def get_item_ids(self) -> list[str]:
        """
        Returns deterministic item IDs aligned to the context ordering.
        """
        pass

    @abstractmethod
    def has_embeddings(self) -> bool:
        """
        Indicates whether embeddings are loaded and available for retrieval.
        """
        pass

    @abstractmethod
    def get_embeddings(self) -> dict[str, list[float]]:
        """
        Returns embeddings keyed by deterministic item ID.
        """
        pass
