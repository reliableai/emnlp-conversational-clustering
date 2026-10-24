"""LLM step of discovery: abstract candidate dimensions (axes of variation) from the evidence.

Given the corpus profile and one or more candidate clusterings, an LLM proposes the orthogonal
ways the corpus could be grouped -- axes, not clusters -- with a shape and grounding for each. Uses
validated structured output; every LLM-facing field is required (no defaults) to satisfy strict
schema validation.
"""

from __future__ import annotations

from typing import Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ..models import ModelConfig, build_llm
from .candidate_clusterings import ClusteringResult
from .corpus_profile import CorpusProfile

_MAX_CLUSTERS_IN_PROMPT = 20
_MAX_REPRESENTATIVE_CHARS = 160

DimensionShape = Literal["categorical", "ordinal", "graded", "multi_valued"]
DimensionGrounding = Literal["metadata", "computable", "inferable", "subjective"]


class ProposedDimension(BaseModel):
    name: str
    description: str
    example_labels: list[str]
    shape: DimensionShape
    grounding: DimensionGrounding
    already_in_metadata: bool
    rationale: str


class DimensionProposals(BaseModel):
    dimensions: list[ProposedDimension]
    notes: str


_SYSTEM_PROMPT = (
    "You analyze a document corpus to discover the AXES OF VARIATION along which its items "
    "could be grouped -- the different points of view a user might take -- not to restate any "
    "single clustering. You are given descriptive statistics, one or more candidate clusterings "
    "produced by different algorithms, and a sample of items. Clusterings are noisy evidence: a "
    "keyword clustering often locks onto surface phrasing rather than meaning. Your job is to "
    "abstract the underlying dimensions.\n\n"
    "Return several (aim for 5-8) DISTINCT, roughly orthogonal dimensions. Deliberately include "
    "subtle or low-signal axes that a keyword clustering would miss, not just the dominant one. "
    "For each dimension give: a short name; a one-line description; 3-6 example labels (the values "
    "items would take on that axis); its shape (categorical, ordinal, graded, or multi_valued); "
    "its grounding (metadata if a column already captures it, computable if a rule over the text "
    "would, inferable if it needs reading and judgement, subjective if it depends on the user's "
    "definition); whether a metadata column already captures it; and a one-line rationale citing "
    "the evidence. Do not invent axes the data does not support."
)


def _format_profile(profile: CorpusProfile) -> str:
    lines = [
        f"Corpus '{profile.dataset_name}': {profile.n_rows} items, "
        f"{profile.n_columns} columns. Item column: '{profile.item_column}'.",
        "Columns (name | kind | fill | distinct | examples):",
    ]
    for column in profile.columns:
        if column.is_item_column:
            examples = "free text"
        elif column.top_values:
            examples = ", ".join(v.value for v in column.top_values[:5])
        elif column.numeric_summary:
            s = column.numeric_summary
            examples = f"range {s.minimum:g}..{s.maximum:g}, median {s.median:g}"
        else:
            examples = ""
        lines.append(
            f"  {column.name} | {column.kind} | {column.fill_rate:.2f} | "
            f"{column.n_unique} | {examples}"
        )
    return "\n".join(lines)


def _format_clustering(result: ClusteringResult) -> str:
    lines = [f"Clustering '{result.method}' ({result.n_clusters} clusters):"]
    for cluster in result.clusters[:_MAX_CLUSTERS_IN_PROMPT]:
        terms = ", ".join(cluster.top_terms[:8])
        example = ""
        if cluster.representative_texts:
            example = cluster.representative_texts[0][:_MAX_REPRESENTATIVE_CHARS].replace("\n", " ")
        lines.append(f"  c{cluster.cluster_id} (n={cluster.size}) terms: {terms} | e.g. {example}")
    return "\n".join(lines)


def _format_sample(sample_items: list[str]) -> str:
    lines = ["Item sample:"]
    for index, text in enumerate(sample_items):
        lines.append(f"  [{index}] {text[:_MAX_REPRESENTATIVE_CHARS].strip()}")
    return "\n".join(lines)


def build_discovery_prompt(
    *,
    profile: CorpusProfile,
    clusterings: list[ClusteringResult],
    sample_items: list[str],
) -> str:
    blocks = [_format_profile(profile)]
    for result in clusterings:
        blocks.append(_format_clustering(result))
    blocks.append(_format_sample(sample_items))
    return "\n\n".join(blocks)


def propose_dimensions(
    *,
    profile: CorpusProfile,
    clusterings: list[ClusteringResult],
    sample_items: list[str],
    model_config: ModelConfig,
) -> DimensionProposals:
    prompt = build_discovery_prompt(
        profile=profile, clusterings=clusterings, sample_items=sample_items
    )
    structured_llm = build_llm(model_config).with_structured_output(
        DimensionProposals, strict=True, include_raw=True
    )
    result = structured_llm.invoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    if not isinstance(result, dict):
        raise ValueError(
            "Structured discovery LLM did not return the expected include_raw payload."
        )
    if result.get("parsing_error"):
        raise ValueError(
            f"Discovery LLM structured-output parsing failed: {result['parsing_error']}"
        )
    parsed = result.get("parsed")
    if parsed is None:
        raise ValueError("Discovery LLM returned no parsed DimensionProposals.")
    return cast(DimensionProposals, parsed)
