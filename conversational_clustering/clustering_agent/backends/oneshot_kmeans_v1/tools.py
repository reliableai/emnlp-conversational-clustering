"""Tool definitions for the oneshot_kmeans_v1 backend's embedding-driven K-means clustering."""

from collections import defaultdict
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from ....data_processing.data_loader import DataLoader
from ....data_processing.item_records import build_dataset_item_records
from ..backend_v1.tools import ClusterSpec, make_save_clustering
from ..kmeans_utils import ordered_unique_items, run_kmeans


class SaveClusteringFromKMeansInput(BaseModel):
    num_clusters: int = Field(
        ...,
        ge=1,
        description="The exact number of clusters to produce with K-means.",
    )


def _build_cluster_specs(data_loader: DataLoader, *, num_clusters: int) -> list[ClusterSpec]:
    if not data_loader.has_embeddings():
        raise ValueError(
            "Embeddings are not available. Enable embeddings when creating the data loader."
        )

    item_records = build_dataset_item_records(data_loader.to_context())
    item_ids = data_loader.get_item_ids()
    if len(item_records) != len(item_ids):
        raise ValueError("Dataset item IDs must align to the dataset context.")

    ordered_items = ordered_unique_items(data_loader)
    if not ordered_items:
        raise ValueError("Dataset is empty.")

    embeddings_by_item_id = data_loader.get_embeddings()
    vectors: list[list[float]] = []
    for item_id, _item_text in ordered_items:
        vector = embeddings_by_item_id.get(item_id)
        if vector is None:
            raise ValueError(f"Missing embedding for item_id '{item_id}'.")
        vectors.append(list(vector))

    assignments = run_kmeans(vectors=vectors, num_clusters=num_clusters)
    assignment_by_item_id = {
        item_id: cluster_index
        for (item_id, _item_text), cluster_index in zip(ordered_items, assignments, strict=True)
    }
    members_by_cluster: dict[int, list[str]] = defaultdict(list)
    first_position_by_cluster: dict[int, int] = {}
    for item_index, (item_record, item_id) in enumerate(zip(item_records, item_ids, strict=True)):
        cluster_index = assignment_by_item_id.get(item_id)
        if cluster_index is None:
            raise ValueError(f"Missing cluster assignment for item_id '{item_id}'.")
        members_by_cluster[cluster_index].append(item_record.uuid)
        first_position_by_cluster.setdefault(cluster_index, item_index)

    sorted_cluster_indices = sorted(
        members_by_cluster,
        key=lambda cluster_index: first_position_by_cluster[cluster_index],
    )
    return [
        ClusterSpec(
            name=f"Cluster {cluster_number}",
            member_uuids=members_by_cluster[cluster_index],
        )
        for cluster_number, cluster_index in enumerate(sorted_cluster_indices, start=1)
    ]


def make_save_clustering_from_kmeans(
    data_loader: DataLoader | None,
    default_output_path: str = "data/clusterings/latest.json",
) -> BaseTool:
    delegate_save_tool = make_save_clustering(
        data_loader,
        default_output_path=default_output_path,
    )

    @tool("save_clustering", args_schema=SaveClusteringFromKMeansInput)
    def save_clustering(num_clusters: int) -> str:
        """Runs K-means over the preloaded embeddings, then saves the resulting clustering."""
        try:
            if data_loader is None:
                raise ValueError("Dataset is unavailable.")
            clusters = _build_cluster_specs(data_loader, num_clusters=num_clusters)
            invoke_payload: Any = {"clusters": [cluster.model_dump() for cluster in clusters]}
            return str(delegate_save_tool.invoke(invoke_payload))
        except ValueError as exc:
            return f"Could not save clustering. Reason: {exc}"
        except Exception:
            return "Could not save clustering due to an unexpected error."

    return save_clustering


def build_tools(
    data_loader: DataLoader | None,
    save_path: str | None = None,
) -> list[BaseTool]:
    save_tool = (
        make_save_clustering_from_kmeans(data_loader)
        if save_path is None
        else make_save_clustering_from_kmeans(data_loader, default_output_path=save_path)
    )
    return [save_tool]
