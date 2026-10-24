"""Unit tests for shared K-means helpers used across clustering backends."""

import pytest

from conversational_clustering.clustering_agent.backends.kmeans_utils import run_kmeans


def test_run_kmeans_returns_assignment_per_vector() -> None:
    assignments = run_kmeans(
        vectors=[
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.2, 9.9],
        ],
        num_clusters=2,
    )

    assert len(assignments) == 4
    assert assignments[0] == assignments[1]
    assert assignments[2] == assignments[3]
    assert assignments[0] != assignments[2]


def test_run_kmeans_rebalances_empty_clusters() -> None:
    assignments = run_kmeans(
        vectors=[
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        num_clusters=2,
    )

    assert len(assignments) == 3
    assert set(assignments) == {0, 1}


def test_run_kmeans_rejects_mismatched_embedding_dimensions() -> None:
    with pytest.raises(ValueError, match="Embedding dimensions must match"):
        run_kmeans(vectors=[[0.0, 0.0], [1.0]], num_clusters=1)
