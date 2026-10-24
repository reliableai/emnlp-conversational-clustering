"""Tests for step-one discovery: corpus profiling, candidate clustering, and diverse sampling."""

from __future__ import annotations

import pandas as pd
import pytest

from conversational_clustering.discovery.candidate_clusterings import tfidf_svd_kmeans
from conversational_clustering.discovery.corpus_profile import profile_corpus
from conversational_clustering.discovery.pipeline import diverse_sample


def _frame() -> pd.DataFrame:
    rows = []
    for i in range(60):
        bucket = i % 3
        if bucket == 0:
            text = f"password reset login access failure for user {i}"
            urgency = "high"
        elif bucket == 1:
            text = f"invoice billing refund mismatch on order {i}"
            urgency = "low"
        else:
            text = f"onboarding new hire laptop setup request {i}"
            urgency = "medium"
        rows.append({"ticket_text": text, "urgency": urgency, "team": f"team_{bucket}"})
    return pd.DataFrame(rows)


def test_profile_corpus_infers_kinds_and_fill() -> None:
    profile = profile_corpus(_frame(), dataset_name="toy", item_column="ticket_text")
    assert profile.n_rows == 60
    assert profile.item_column == "ticket_text"
    by_name = {c.name: c for c in profile.columns}
    assert by_name["ticket_text"].kind == "text"
    assert by_name["ticket_text"].is_item_column
    assert by_name["ticket_text"].text_length
    assert by_name["urgency"].kind == "categorical"
    assert by_name["urgency"].fill_rate == 1.0
    assert {v.value for v in by_name["urgency"].top_values} == {"high", "low", "medium"}


def test_profile_corpus_rejects_missing_item_column() -> None:
    with pytest.raises(ValueError, match="item_column"):
        profile_corpus(_frame(), dataset_name="toy", item_column="does_not_exist")


def test_profile_corpus_rejects_empty_frame() -> None:
    empty = pd.DataFrame({"ticket_text": pd.Series([], dtype="object")})
    with pytest.raises(ValueError, match="empty"):
        profile_corpus(empty, dataset_name="toy", item_column="ticket_text")


def test_tfidf_svd_kmeans_shapes_and_terms() -> None:
    texts = _frame()["ticket_text"].tolist()
    result = tfidf_svd_kmeans(texts, k=3)
    assert result.n_clusters == 3
    assert result.n_items == len(texts)
    assert len(result.labels) == len(texts)
    assert set(result.labels) == {0, 1, 2}
    for cluster in result.clusters:
        assert cluster.size > 0
        assert cluster.top_terms
        assert cluster.representative_texts


def test_tfidf_svd_kmeans_rejects_too_few_items() -> None:
    with pytest.raises(ValueError, match="at least"):
        tfidf_svd_kmeans(["only one document here"], k=20)


def test_diverse_sample_one_per_cluster() -> None:
    texts = _frame()["ticket_text"].tolist()
    result = tfidf_svd_kmeans(texts, k=3)
    sample = diverse_sample(result, per_cluster=1)
    assert len(sample) == 3
