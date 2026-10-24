"""Shared helper functions for embedding-based K-means clustering backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...data_processing.data_loader import DataLoader


def ordered_unique_items(data_loader: DataLoader) -> list[tuple[str, str]]:
    ordered_items: list[tuple[str, str]] = []
    seen_item_ids: set[str] = set()
    for item_id, item_text in zip(
        data_loader.get_item_ids(),
        data_loader.to_context(),
        strict=True,
    ):
        if item_id in seen_item_ids:
            continue
        seen_item_ids.add(item_id)
        ordered_items.append((item_id, item_text))
    return ordered_items


def coerce_vector_matrix(vectors: list[list[float]]) -> np.ndarray:
    try:
        vector_matrix = np.asarray(vectors, dtype=np.float64)
    except ValueError as exc:
        raise ValueError("Embedding dimensions must match for K-means.") from exc
    if vector_matrix.ndim != 2:
        raise ValueError("Embedding dimensions must match for K-means.")
    return vector_matrix


def pairwise_squared_distances(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    vector_squared_norms = np.sum(vectors * vectors, axis=1, keepdims=True)
    centroid_squared_norms = np.sum(centroids * centroids, axis=1)
    squared_distances = (
        vector_squared_norms - (2.0 * vectors @ centroids.T) + centroid_squared_norms
    )
    np.maximum(squared_distances, 0.0, out=squared_distances)
    return squared_distances


def _initialize_centroids(vector_matrix: np.ndarray, num_clusters: int) -> np.ndarray:
    selected_indices = np.empty(num_clusters, dtype=np.intp)
    selected_mask = np.zeros(vector_matrix.shape[0], dtype=bool)
    selected_indices[0] = 0
    selected_mask[0] = True

    min_squared_distances = pairwise_squared_distances(vector_matrix, vector_matrix[[0]]).reshape(
        -1
    )
    min_squared_distances[0] = -np.inf

    for selected_count in range(1, num_clusters):
        candidate_distances = np.where(selected_mask, -np.inf, min_squared_distances)
        best_index = int(np.argmax(candidate_distances))
        if selected_mask[best_index]:
            raise ValueError("Could not initialize K-means centroids.")
        selected_indices[selected_count] = best_index
        selected_mask[best_index] = True
        next_distances = pairwise_squared_distances(
            vector_matrix, vector_matrix[[best_index]]
        ).reshape(-1)
        min_squared_distances = np.minimum(min_squared_distances, next_distances)
        min_squared_distances[selected_mask] = -np.inf

    return vector_matrix[selected_indices].copy()


def _rebalance_empty_clusters(
    *,
    vector_matrix: np.ndarray,
    assignments: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    updated_assignments = assignments.copy()
    counts = np.bincount(updated_assignments, minlength=centroids.shape[0])
    if np.all(counts > 0):
        return updated_assignments

    distances_to_assigned_centroid = np.sum(
        (vector_matrix - centroids[updated_assignments]) ** 2,
        axis=1,
    )
    for empty_cluster_index in np.flatnonzero(counts == 0):
        donor_mask = counts[updated_assignments] >= 2
        if not np.any(donor_mask):
            raise ValueError("Could not rebalance empty K-means cluster.")

        donor_distances = np.where(donor_mask, distances_to_assigned_centroid, -np.inf)
        donor_item_index = int(np.argmax(donor_distances))
        donor_cluster_index = int(updated_assignments[donor_item_index])
        if counts[donor_cluster_index] < 2:
            raise ValueError("Could not rebalance empty K-means cluster.")

        updated_assignments[donor_item_index] = empty_cluster_index
        counts[donor_cluster_index] -= 1
        counts[empty_cluster_index] += 1

    return updated_assignments


def _recompute_centroids(
    *,
    vector_matrix: np.ndarray,
    assignments: np.ndarray,
    num_clusters: int,
) -> np.ndarray:
    counts = np.bincount(assignments, minlength=num_clusters)
    if np.any(counts == 0):
        raise ValueError("K-means produced an empty cluster after rebalancing.")

    centroid_sums = np.zeros((num_clusters, vector_matrix.shape[1]), dtype=np.float64)
    np.add.at(centroid_sums, assignments, vector_matrix)
    return centroid_sums / counts[:, np.newaxis]


def run_kmeans(
    *,
    vectors: list[list[float]],
    num_clusters: int,
    max_iterations: int = 50,
) -> list[int]:
    if not vectors:
        raise ValueError("K-means requires at least one embedding vector.")
    if num_clusters < 1:
        raise ValueError("num_clusters must be >= 1.")
    if num_clusters > len(vectors):
        raise ValueError("num_clusters must be <= the number of unique dataset items.")

    vector_matrix = coerce_vector_matrix(vectors)
    centroids = _initialize_centroids(vector_matrix, num_clusters)
    assignments = np.full(vector_matrix.shape[0], fill_value=-1, dtype=np.intp)
    for _ in range(max_iterations):
        updated_assignments = np.argmin(
            pairwise_squared_distances(vector_matrix, centroids),
            axis=1,
        ).astype(np.intp, copy=False)
        updated_assignments = _rebalance_empty_clusters(
            vector_matrix=vector_matrix,
            assignments=updated_assignments,
            centroids=centroids,
        )
        if np.array_equal(updated_assignments, assignments):
            return updated_assignments.tolist()
        assignments = updated_assignments
        centroids = _recompute_centroids(
            vector_matrix=vector_matrix,
            assignments=assignments,
            num_clusters=num_clusters,
        )
    return assignments.tolist()
