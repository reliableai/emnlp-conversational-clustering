"""Unit tests for the oneshot_kmeans_v1 backend tool and embedding requirements."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from conversational_clustering.clustering_agent.backends.oneshot_kmeans_v1.config import (
    OneShotKMeansV1ClusteringBackendConfig,
)
from conversational_clustering.clustering_agent.backends.oneshot_kmeans_v1.method import (
    OneShotKMeansV1ConversationalClusteringMethod,
)
from conversational_clustering.clustering_agent.backends.oneshot_kmeans_v1.tools import (
    build_tools,
    make_save_clustering_from_kmeans,
)
from conversational_clustering.data_processing.data_loader import DataLoader


class FakeEmbeddingLoader(DataLoader):
    def __init__(
        self,
        *,
        items: list[str],
        item_ids: list[str] | None = None,
        embeddings: dict[str, list[float]] | None = None,
    ) -> None:
        self._items = items
        self._item_ids = item_ids or items
        self._embeddings = embeddings

    def load(self) -> pd.DataFrame:
        return pd.DataFrame({"item": self._items})

    def to_context(self) -> list[str]:
        return list(self._items)

    def get_item_ids(self) -> list[str]:
        return list(self._item_ids)

    def has_embeddings(self) -> bool:
        return self._embeddings is not None

    def get_embeddings(self) -> dict[str, list[float]]:
        if self._embeddings is None:
            raise ValueError("embeddings unavailable")
        return {item_id: list(vector) for item_id, vector in self._embeddings.items()}


def test_make_save_clustering_from_kmeans_writes_saved_clustering(tmp_path: Path) -> None:
    loader = FakeEmbeddingLoader(
        items=["alpha", "beta", "gamma", "delta"],
        embeddings={
            "alpha": [0.0, 0.0],
            "beta": [0.1, 0.0],
            "gamma": [10.0, 10.0],
            "delta": [10.2, 9.9],
        },
    )
    output_path = tmp_path / "clustering.json"
    save_tool = make_save_clustering_from_kmeans(loader, default_output_path=str(output_path))

    invoke_payload: Any = {"num_clusters": 2}
    response = save_tool.invoke(invoke_payload)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert "Clustering saved" in response
    assert [cluster["name"] for cluster in payload["clusters"]] == ["Cluster 1", "Cluster 2"]
    assert payload["clusters"][0]["members"] == ["alpha", "beta"]
    assert payload["clusters"][1]["members"] == ["gamma", "delta"]


def test_make_save_clustering_from_kmeans_expands_duplicate_rows(tmp_path: Path) -> None:
    loader = FakeEmbeddingLoader(
        items=["alpha", "alpha", "gamma"],
        item_ids=["alpha", "alpha", "gamma"],
        embeddings={
            "alpha": [0.0, 0.0],
            "gamma": [10.0, 10.0],
        },
    )
    output_path = tmp_path / "clustering.json"
    save_tool = make_save_clustering_from_kmeans(loader, default_output_path=str(output_path))

    invoke_payload: Any = {"num_clusters": 2}
    response = save_tool.invoke(invoke_payload)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert "Clustering saved" in response
    assert payload["clusters"][0]["members"] == ["alpha", "alpha"]
    assert payload["clusters"][1]["members"] == ["gamma"]


def test_make_save_clustering_from_kmeans_rejects_too_many_clusters() -> None:
    loader = FakeEmbeddingLoader(
        items=["alpha", "beta"],
        embeddings={
            "alpha": [0.0, 0.0],
            "beta": [1.0, 1.0],
        },
    )
    save_tool = make_save_clustering_from_kmeans(loader)

    invoke_payload: Any = {"num_clusters": 3}
    response = save_tool.invoke(invoke_payload)

    assert "Could not save clustering" in response
    assert "num_clusters must be <=" in response


def test_build_tools_returns_only_save_clustering_tool() -> None:
    tools = build_tools(
        FakeEmbeddingLoader(
            items=["alpha"],
            embeddings={"alpha": [0.0, 0.0]},
        )
    )

    assert [tool.name for tool in tools] == ["save_clustering"]


def test_oneshot_kmeans_method_requires_embeddings() -> None:
    method = OneShotKMeansV1ConversationalClusteringMethod(OneShotKMeansV1ClusteringBackendConfig())

    with pytest.raises(ValueError, match="requires embeddings"):
        method.resolve_app(
            data_loader=FakeEmbeddingLoader(items=["alpha"], embeddings=None),
            save_path=None,
        )
