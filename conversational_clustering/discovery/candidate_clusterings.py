"""Candidate clusterings that seed dimension discovery: a lexical view and a semantic view.

Each backend returns a ClusteringResult with per-cluster top terms and representative items. These
are deliberately different lenses (bag-of-words vs. embedding geometry) so the downstream LLM sees
more than one axis of variation, not a single dominant grouping.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

_DEFAULT_K = 20
_DEFAULT_SVD_DIM = 100
_DEFAULT_SEED = 42
_TOP_TERMS = 10
_REPRESENTATIVES = 4


class ClusterSummary(BaseModel):
    cluster_id: int
    size: int
    top_terms: list[str]
    representative_texts: list[str]


class ClusteringResult(BaseModel):
    method: str
    n_clusters: int
    n_items: int
    labels: list[int] = Field(repr=False)
    clusters: list[ClusterSummary]


def _fit_tfidf(texts: list[str]) -> tuple[np.ndarray, list[str], TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
    )
    matrix = vectorizer.fit_transform(texts)
    feature_names = list(vectorizer.get_feature_names_out())
    return matrix, feature_names, vectorizer


def _top_terms_per_cluster(
    tfidf_matrix, feature_names: list[str], labels: np.ndarray, top_n: int
) -> dict[int, list[str]]:
    terms_by_cluster: dict[int, list[str]] = {}
    for cluster_id in sorted(set(int(label) for label in labels)):
        mask = labels == cluster_id
        mean_row = np.asarray(tfidf_matrix[mask].mean(axis=0)).reshape(-1)
        if mean_row.size == 0:
            terms_by_cluster[cluster_id] = []
            continue
        top_indices = np.argsort(mean_row)[::-1][:top_n]
        terms_by_cluster[cluster_id] = [feature_names[i] for i in top_indices if mean_row[i] > 0.0]
    return terms_by_cluster


def _representatives_per_cluster(
    space: np.ndarray, labels: np.ndarray, texts: list[str], n: int
) -> dict[int, list[str]]:
    reps_by_cluster: dict[int, list[str]] = {}
    for cluster_id in sorted(set(int(label) for label in labels)):
        member_indices = np.where(labels == cluster_id)[0]
        if member_indices.size == 0:
            reps_by_cluster[cluster_id] = []
            continue
        centroid = space[member_indices].mean(axis=0, keepdims=True)
        distances = np.linalg.norm(space[member_indices] - centroid, axis=1)
        nearest = member_indices[np.argsort(distances)[:n]]
        reps_by_cluster[cluster_id] = [texts[i] for i in nearest]
    return reps_by_cluster


def _assemble(
    *,
    method: str,
    labels: np.ndarray,
    tfidf_matrix,
    feature_names: list[str],
    space: np.ndarray,
    texts: list[str],
) -> ClusteringResult:
    top_terms = _top_terms_per_cluster(tfidf_matrix, feature_names, labels, _TOP_TERMS)
    reps = _representatives_per_cluster(space, labels, texts, _REPRESENTATIVES)
    cluster_ids = sorted(set(int(label) for label in labels))
    clusters = [
        ClusterSummary(
            cluster_id=cluster_id,
            size=int(np.sum(labels == cluster_id)),
            top_terms=top_terms.get(cluster_id, []),
            representative_texts=reps.get(cluster_id, []),
        )
        for cluster_id in cluster_ids
    ]
    return ClusteringResult(
        method=method,
        n_clusters=len(cluster_ids),
        n_items=len(texts),
        labels=[int(label) for label in labels],
        clusters=clusters,
    )


def tfidf_svd_kmeans(
    texts: list[str],
    *,
    k: int = _DEFAULT_K,
    svd_dim: int = _DEFAULT_SVD_DIM,
    seed: int = _DEFAULT_SEED,
) -> ClusteringResult:
    """Lexical view: TF-IDF (1-2 grams) -> truncated SVD -> K-means."""
    if len(texts) < k:
        raise ValueError(f"Need at least k={k} items for K-means, got {len(texts)}.")
    tfidf_matrix, feature_names, _ = _fit_tfidf(texts)
    n_components = min(svd_dim, tfidf_matrix.shape[1] - 1)
    if n_components < 2:
        raise ValueError("TF-IDF vocabulary too small for SVD; check the input texts.")
    reduced = TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(tfidf_matrix)
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(reduced)
    return _assemble(
        method=f"tfidf_svd_kmeans(k={k})",
        labels=labels,
        tfidf_matrix=tfidf_matrix,
        feature_names=feature_names,
        space=reduced,
        texts=texts,
    )


def embedding_kmeans(
    texts: list[str],
    embeddings: list[list[float]],
    *,
    k: int = _DEFAULT_K,
    seed: int = _DEFAULT_SEED,
) -> ClusteringResult:
    """Semantic view: K-means over item embeddings; terms named via TF-IDF for readability."""
    if len(texts) != len(embeddings):
        raise ValueError(f"texts ({len(texts)}) and embeddings ({len(embeddings)}) must align.")
    if len(texts) < k:
        raise ValueError(f"Need at least k={k} items for K-means, got {len(texts)}.")
    space = np.asarray(embeddings, dtype=np.float64)
    if space.ndim != 2:
        raise ValueError("Embeddings must form a 2-D matrix with consistent dimensionality.")
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(space)
    tfidf_matrix, feature_names, _ = _fit_tfidf(texts)
    return _assemble(
        method=f"embedding_kmeans(k={k})",
        labels=labels,
        tfidf_matrix=tfidf_matrix,
        feature_names=feature_names,
        space=space,
        texts=texts,
    )
