"""Step-one discovery orchestration: profile a corpus, cluster it two ways, propose dimensions.

Ties the deterministic parts (corpus profile, candidate clusterings) to the LLM dimension proposal.
The semantic clustering arm is optional; when enabled it uses local embeddings so discovery stays
cheap and on-machine.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

from ..models import (
    EMBEDDING_BACKEND_HUGGINGFACE_LOCAL,
    EmbeddingModelConfig,
    ModelConfig,
    default_clustering_agent_model_config,
)
from .candidate_clusterings import ClusteringResult, embedding_kmeans, tfidf_svd_kmeans
from .corpus_profile import CorpusProfile, profile_corpus
from .dimension_proposal import DimensionProposals, propose_dimensions

_DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DiscoveryResult(BaseModel):
    profile: CorpusProfile
    clusterings: list[ClusteringResult]
    proposals: DimensionProposals


def default_discovery_embedding_config() -> EmbeddingModelConfig:
    return EmbeddingModelConfig(
        model=_DEFAULT_LOCAL_EMBEDDING_MODEL,
        backend=EMBEDDING_BACKEND_HUGGINGFACE_LOCAL,
    )


def diverse_sample(clustering: ClusteringResult, per_cluster: int = 1) -> list[str]:
    """One (or few) representative items per cluster -- a spread across the corpus, not the head."""
    sample: list[str] = []
    for cluster in clustering.clusters:
        sample.extend(cluster.representative_texts[:per_cluster])
    return sample


def run_discovery(
    *,
    df: pd.DataFrame,
    dataset_name: str,
    item_column: str,
    k: int = 20,
    use_embeddings: bool = True,
    embedding_model_config: EmbeddingModelConfig | None = None,
    model_config: ModelConfig | None = None,
) -> DiscoveryResult:
    if item_column not in df.columns:
        raise ValueError(f"item_column '{item_column}' not in columns: {list(df.columns)}")
    texts = [str(value) for value in df[item_column].tolist()]

    profile = profile_corpus(df, dataset_name=dataset_name, item_column=item_column)

    clusterings: list[ClusteringResult] = [tfidf_svd_kmeans(texts, k=k)]
    if use_embeddings:
        from ..data_processing.embeddings import generate_embeddings_for_texts

        embeddings = generate_embeddings_for_texts(
            texts=texts,
            model_config=embedding_model_config or default_discovery_embedding_config(),
        )
        clusterings.append(embedding_kmeans(texts, embeddings, k=k))

    sample_items = diverse_sample(clusterings[0], per_cluster=1)
    proposals = propose_dimensions(
        profile=profile,
        clusterings=clusterings,
        sample_items=sample_items,
        model_config=model_config or default_clustering_agent_model_config(),
    )
    return DiscoveryResult(profile=profile, clusterings=clusterings, proposals=proposals)
