"""Tool definitions for the instruction_kmeans_v1 backend's guided embedding K-means flow."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, cast

import httpcore
import httpx
import numpy as np
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from ....accounting import UsageCollector, extract_response_model, extract_token_usage
from ....data_processing.data_loader import DataLoader
from ....data_processing.embeddings import generate_embeddings_for_texts
from ....data_processing.item_records import build_dataset_item_records
from ....models import EmbeddingModelConfig, ModelConfig, build_llm
from ..backend_v1.tools import ClusterSpec, make_save_clustering
from ..kmeans_utils import (
    coerce_vector_matrix,
    ordered_unique_items,
    pairwise_squared_distances,
    run_kmeans,
)
from ..pov_bank import make_append_clustering_povs_tool


@dataclass
class InstructionKMeansRuntimeState:
    latest_proposal: dict[str, object] | None = None
    latest_proposal_artifacts: dict[str, object] | None = None
    proposals_by_id: dict[str, dict[str, object]] | None = None
    clustering_pov_bank: list[str] | None = None
    next_proposal_index: int = 1


class SampleDatasetInput(BaseModel):
    start_index: int = Field(
        ...,
        ge=0,
        description="Start index in the deterministic shuffled dataset order.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of items to return from that starting position.",
    )


class InstructionKMeansInput(BaseModel):
    embedding_instruction: str = Field(
        ...,
        min_length=1,
        description=(
            "Semantic embedding lens prepended to each item before vectorization. "
            "This is not a clustering rule and is not read by K-means directly. "
            "Do not include the cluster count, algorithm instructions, or phrases like "
            "'cluster based on'. Use a concise semantic focus, such as "
            "'support issue type' or 'root cause or failure mode'."
        ),
    )
    num_clusters: int = Field(
        ...,
        ge=1,
        description="The exact number of clusters to produce with K-means.",
    )


class SaveClusteringInput(BaseModel):
    proposal_id: str | None = Field(
        default=None,
        description="The proposal ID returned by propose_clustering for the clustering to save.",
    )


class ClusterSummaryDraft(BaseModel):
    name: str = Field(description="A short label for the cluster theme.")
    summary: str = Field(
        description="A brief description of the cluster theme grounded in the examples."
    )


def format_instruction_embedding_input(*, embedding_instruction: str, item_text: str) -> str:
    return f"Instruct: {embedding_instruction}\nQuery:{item_text}"


def _next_proposal_id(runtime_state: InstructionKMeansRuntimeState) -> str:
    proposal_id = f"proposal_{runtime_state.next_proposal_index}"
    runtime_state.next_proposal_index += 1
    return proposal_id


def _dataset_window_records(
    data_loader: DataLoader | None,
    *,
    shuffle_seed: int,
) -> list[dict[str, object]]:
    if data_loader is None:
        return []
    items = list(data_loader.to_context())
    indexed_items = [{"index": index, "value": item} for index, item in enumerate(items)]
    random.Random(shuffle_seed).shuffle(indexed_items)
    for shuffled_index, item in enumerate(indexed_items):
        item["index"] = shuffled_index
    return indexed_items


def _select_representative_points(
    *,
    texts: list[str],
    vectors: np.ndarray,
    limit: int,
) -> list[str]:
    if not texts:
        return []
    if len(texts) <= limit:
        return list(texts)

    centroid = np.mean(vectors, axis=0, keepdims=True)
    distances = pairwise_squared_distances(vectors, centroid).reshape(-1)
    sorted_indices = np.argsort(distances)
    selected_indices: list[int] = []
    preferred_count = min(len(sorted_indices), min(10, limit))
    selected_indices.extend(int(index) for index in sorted_indices[:preferred_count])

    remaining_slots = limit - len(selected_indices)
    if remaining_slots > 0:
        spread_candidates = sorted_indices[preferred_count:]
        if spread_candidates.size > 0:
            spread_positions = np.linspace(
                0,
                spread_candidates.size - 1,
                num=min(remaining_slots, spread_candidates.size),
                dtype=int,
            )
            selected_indices.extend(
                int(spread_candidates[position]) for position in spread_positions
            )

    representative_points: list[str] = []
    seen_texts: set[str] = set()
    for index in selected_indices:
        text = texts[index]
        if text in seen_texts:
            continue
        representative_points.append(text)
        seen_texts.add(text)
        if len(representative_points) >= limit:
            break
    return representative_points


def _normalize_cluster_name(name: str, *, fallback_cluster_number: int) -> str:
    normalized = " ".join(str(name).split())
    if normalized:
        return normalized
    return f"Cluster {fallback_cluster_number}"


def _fallback_cluster_summary(
    *,
    cluster_number: int,
    representative_points: list[str],
) -> ClusterSummaryDraft:
    if representative_points:
        preview = "; ".join(representative_points[:3])
        summary = f"Representative items include: {preview}"
    else:
        summary = "No representative items were available."
    return ClusterSummaryDraft(name=f"Cluster {cluster_number}", summary=summary)


def _is_timeout_error(exc: Exception) -> bool:
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        if isinstance(current, TimeoutError | httpx.TimeoutException | httpcore.TimeoutException):
            return True
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return False


def _summarize_cluster(
    *,
    summary_model_config: ModelConfig,
    cluster_number: int,
    cluster_size: int,
    representative_points: list[str],
    usage_collector: UsageCollector | None = None,
) -> ClusterSummaryDraft:
    prompt = (
        "Summarize this cluster for a conversational clustering workflow.\n"
        f"Cluster number: {cluster_number}\n"
        f"Cluster size: {cluster_size}\n"
        "Representative points:\n"
        + "\n".join(f"- {point}" for point in representative_points)
        + "\nReturn a short thematic name and a brief summary grounded in the examples."
    )
    try:
        structured_llm = build_llm(summary_model_config).with_structured_output(
            ClusterSummaryDraft,
            include_raw=True,
        )
        response = structured_llm.invoke(prompt)
        if not isinstance(response, dict):
            raise ValueError("Cluster summary model returned an unexpected payload.")
        raw_result = response.get("raw")
        if raw_result is not None and usage_collector is not None:
            usage_collector.record(
                extract_response_model(raw_result, fallback_model=summary_model_config.model),
                extract_token_usage(raw_result),
                source="instruction_kmeans_v1_cluster_summary",
            )
        parsing_error = response.get("parsing_error")
        if parsing_error is not None:
            raise ValueError("Cluster summary model parsing failed.") from parsing_error
        result = response.get("parsed")
        if not isinstance(result, ClusterSummaryDraft):
            raise ValueError("Cluster summary model returned an unexpected payload.")
        return ClusterSummaryDraft(
            name=_normalize_cluster_name(result.name, fallback_cluster_number=cluster_number),
            summary=" ".join(result.summary.split()),
        )
    except Exception:
        return _fallback_cluster_summary(
            cluster_number=cluster_number,
            representative_points=representative_points,
        )


def _compute_guided_kmeans_result(
    data_loader: DataLoader | None,
    *,
    embedding_instruction: str,
    num_clusters: int,
    embedding_model_config: EmbeddingModelConfig,
    summary_model_config: ModelConfig,
    enable_cluster_summary_model: bool,
    cluster_summary_example_limit: int,
    embedding_batch_size: int,
    usage_collector: UsageCollector | None = None,
) -> dict[str, object]:
    normalized_embedding_instruction = " ".join(embedding_instruction.split())
    if not normalized_embedding_instruction:
        raise ValueError("embedding_instruction must be a non-empty string.")
    if data_loader is None:
        raise ValueError("Dataset is unavailable.")

    item_records = build_dataset_item_records(data_loader.to_context())
    item_ids = data_loader.get_item_ids()
    if len(item_records) != len(item_ids):
        raise ValueError("Dataset item IDs must align to the dataset context.")

    ordered_items = ordered_unique_items(data_loader)
    if not ordered_items:
        raise ValueError("Dataset is empty.")
    if num_clusters > len(ordered_items):
        raise ValueError("num_clusters must be <= the number of unique dataset items.")

    guided_texts = [
        format_instruction_embedding_input(
            embedding_instruction=normalized_embedding_instruction,
            item_text=item_text,
        )
        for _, item_text in ordered_items
    ]
    vectors = generate_embeddings_for_texts(
        texts=guided_texts,
        model_config=embedding_model_config,
        batch_size=embedding_batch_size,
        usage_collector=usage_collector,
    )
    assignments = run_kmeans(vectors=vectors, num_clusters=num_clusters)
    vector_matrix = coerce_vector_matrix(vectors)

    assignment_by_item_id = {
        item_id: cluster_index
        for (item_id, _item_text), cluster_index in zip(ordered_items, assignments, strict=True)
    }
    members_by_cluster_uuid: dict[int, list[str]] = defaultdict(list)
    members_by_cluster_text: dict[int, list[str]] = defaultdict(list)
    first_position_by_cluster: dict[int, int] = {}
    for item_index, (item_record, item_id) in enumerate(zip(item_records, item_ids, strict=True)):
        cluster_index = assignment_by_item_id.get(item_id)
        if cluster_index is None:
            raise ValueError(f"Missing cluster assignment for item_id '{item_id}'.")
        members_by_cluster_uuid[cluster_index].append(item_record.uuid)
        members_by_cluster_text[cluster_index].append(item_record.value)
        first_position_by_cluster.setdefault(cluster_index, item_index)

    unique_cluster_indices = defaultdict(list)
    for ordered_position, ((item_id, item_text), cluster_index) in enumerate(
        zip(ordered_items, assignments, strict=True)
    ):
        _ = item_id
        unique_cluster_indices[cluster_index].append((ordered_position, item_text))

    sorted_cluster_indices = sorted(
        members_by_cluster_uuid,
        key=lambda cluster_index: first_position_by_cluster[cluster_index],
    )

    clusters_for_save: list[ClusterSpec] = []
    summarized_clusters: list[dict[str, object]] = []
    for cluster_number, cluster_index in enumerate(sorted_cluster_indices, start=1):
        unique_texts = [text for _, text in unique_cluster_indices[cluster_index]]
        unique_positions = [position for position, _ in unique_cluster_indices[cluster_index]]
        cluster_vectors = vector_matrix[np.asarray(unique_positions, dtype=np.intp)]
        representative_points = _select_representative_points(
            texts=unique_texts,
            vectors=cluster_vectors,
            limit=cluster_summary_example_limit,
        )
        if enable_cluster_summary_model:
            summary = _summarize_cluster(
                summary_model_config=summary_model_config,
                cluster_number=cluster_number,
                cluster_size=len(members_by_cluster_uuid[cluster_index]),
                representative_points=representative_points,
                usage_collector=usage_collector,
            )
        else:
            summary = _fallback_cluster_summary(
                cluster_number=cluster_number,
                representative_points=representative_points,
            )
        cluster_name = _normalize_cluster_name(
            summary.name,
            fallback_cluster_number=cluster_number,
        )
        clusters_for_save.append(
            ClusterSpec(
                name=cluster_name,
                member_uuids=members_by_cluster_uuid[cluster_index],
            )
        )
        summarized_clusters.append(
            {
                "name": cluster_name,
                "size": len(members_by_cluster_uuid[cluster_index]),
                "summary": summary.summary,
                "representative_points": representative_points[
                    : min(3, len(representative_points))
                ],
            }
        )

    clustering_payload = {
        "clusters": [
            {
                "name": cluster.name,
                "members": list(members_by_cluster_text[cluster_index]),
            }
            for cluster_index, cluster in zip(
                sorted_cluster_indices, clusters_for_save, strict=True
            )
        ]
    }
    intermittent_payload = {
        "embedding_instruction": normalized_embedding_instruction,
        "num_clusters": num_clusters,
        "cluster_summaries": summarized_clusters,
        "clusters": list(clustering_payload["clusters"]),
    }
    return {
        "embedding_instruction": normalized_embedding_instruction,
        "num_clusters": num_clusters,
        "clusters": summarized_clusters,
        "clustering": clustering_payload,
        "intermittent_clustering": intermittent_payload,
        "save_clusters": [cluster.model_dump() for cluster in clusters_for_save],
    }


def make_sample_dataset_tool(
    data_loader: DataLoader | None,
    *,
    shuffle_seed: int,
    sample_window_size: int,
) -> BaseTool:
    shuffled_records = _dataset_window_records(data_loader, shuffle_seed=shuffle_seed)

    @tool("sample_dataset", args_schema=SampleDatasetInput)
    def sample_dataset(start_index: int, limit: int | None = None) -> str:
        """Returns a consecutive window from a deterministically shuffled dataset order."""
        resolved_limit = sample_window_size if limit is None else min(limit, sample_window_size)
        window = shuffled_records[start_index : start_index + resolved_limit]
        return json.dumps(
            {
                "total_items": len(shuffled_records),
                "start_index": start_index,
                "end_index_exclusive": start_index + len(window),
                "items": window,
            },
            ensure_ascii=True,
        )

    return sample_dataset


def make_propose_clustering_tool(
    data_loader: DataLoader | None,
    *,
    runtime_state: InstructionKMeansRuntimeState,
    embedding_model_config: EmbeddingModelConfig,
    summary_model_config: ModelConfig,
    enable_cluster_summary_model: bool = True,
    cluster_summary_example_limit: int,
    embedding_batch_size: int,
    usage_collector: UsageCollector | None = None,
) -> BaseTool:
    @tool("propose_clustering", args_schema=InstructionKMeansInput)
    def propose_clustering(embedding_instruction: str, num_clusters: int) -> str:
        """Generates guided embeddings, runs K-means, and returns cluster summaries."""
        try:
            proposal = _compute_guided_kmeans_result(
                data_loader,
                embedding_instruction=embedding_instruction,
                num_clusters=num_clusters,
                embedding_model_config=embedding_model_config,
                summary_model_config=summary_model_config,
                enable_cluster_summary_model=enable_cluster_summary_model,
                cluster_summary_example_limit=cluster_summary_example_limit,
                embedding_batch_size=embedding_batch_size,
                usage_collector=usage_collector,
            )
            proposal_id = _next_proposal_id(runtime_state)
            intermittent_payload = cast(dict[str, object], proposal["intermittent_clustering"])
            runtime_state.latest_proposal = dict(intermittent_payload)
            runtime_state.latest_proposal_artifacts = dict(proposal)
            if runtime_state.proposals_by_id is None:
                runtime_state.proposals_by_id = {}
            runtime_state.proposals_by_id[proposal_id] = dict(proposal)
            response_payload = {
                key: value
                for key, value in proposal.items()
                if key in {"embedding_instruction", "num_clusters", "clusters"}
            }
            response_payload["proposal_id"] = proposal_id
            return json.dumps(response_payload, ensure_ascii=True)
        except ValueError as exc:
            return f"Could not propose clustering. Reason: {exc}"
        except Exception as exc:
            if _is_timeout_error(exc):
                return "Could not propose clustering. Reason: request timed out."
            detail = str(exc).strip() or "no additional details"
            return (
                f"Could not propose clustering. Reason: unexpected {type(exc).__name__}: {detail}"
            )

    return propose_clustering


def make_save_clustering_tool(
    data_loader: DataLoader | None,
    *,
    runtime_state: InstructionKMeansRuntimeState,
    default_output_path: str,
) -> BaseTool:
    delegate_save_tool = make_save_clustering(data_loader, default_output_path=default_output_path)

    @tool("save_clustering", args_schema=SaveClusteringInput)
    def save_clustering(proposal_id: str | None = None) -> str:
        """Saves a previously proposed clustering by proposal ID and returns its summaries."""
        try:
            if proposal_id is None or not proposal_id.strip():
                raise ValueError(
                    "proposal_id is required. Call propose_clustering and pass "
                    "one of its returned IDs."
                )
            resolved_proposal_id = proposal_id.strip()
            proposals_by_id = runtime_state.proposals_by_id or {}
            proposal = proposals_by_id.get(resolved_proposal_id)
            if proposal is None:
                raise ValueError(
                    f"Unknown proposal_id '{resolved_proposal_id}'. "
                    "Call propose_clustering to create a proposal first."
                )
            save_message = cast(Any, delegate_save_tool).invoke(
                {"clusters": proposal["save_clusters"]}
            )
            if isinstance(save_message, str) and save_message.strip().lower().startswith(
                "could not"
            ):
                return save_message
            response_payload = {
                "proposal_id": resolved_proposal_id,
                "embedding_instruction": proposal["embedding_instruction"],
                "num_clusters": proposal["num_clusters"],
                "clusters": proposal["clusters"],
                "save_message": save_message,
            }
            return json.dumps(response_payload, ensure_ascii=True)
        except ValueError as exc:
            return f"Could not save clustering. Reason: {exc}"
        except Exception as exc:
            detail = str(exc).strip() or "no additional details"
            return f"Could not save clustering. Reason: unexpected {type(exc).__name__}: {detail}"

    return save_clustering


def build_tools(
    data_loader: DataLoader | None,
    *,
    runtime_state: InstructionKMeansRuntimeState,
    embedding_model_config: EmbeddingModelConfig,
    summary_model_config: ModelConfig,
    enable_cluster_summary_model: bool = True,
    sample_window_size: int,
    shuffle_seed: int,
    cluster_summary_example_limit: int,
    embedding_batch_size: int,
    save_path: str | None = None,
    usage_collector: UsageCollector | None = None,
    include_save_tool: bool = True,
    include_pov_bank_tool: bool = True,
) -> list[BaseTool]:
    resolved_save_path = save_path or "data/clusterings/latest.json"
    tools: list[BaseTool] = [
        make_sample_dataset_tool(
            data_loader,
            shuffle_seed=shuffle_seed,
            sample_window_size=sample_window_size,
        )
    ]
    if include_pov_bank_tool:
        tools.append(make_append_clustering_povs_tool(runtime_state=runtime_state))
    tools.append(
        make_propose_clustering_tool(
            data_loader,
            runtime_state=runtime_state,
            embedding_model_config=embedding_model_config,
            summary_model_config=summary_model_config,
            enable_cluster_summary_model=enable_cluster_summary_model,
            cluster_summary_example_limit=cluster_summary_example_limit,
            embedding_batch_size=embedding_batch_size,
            usage_collector=usage_collector,
        )
    )
    if include_save_tool:
        tools.append(
            make_save_clustering_tool(
                data_loader,
                runtime_state=runtime_state,
                default_output_path=resolved_save_path,
            )
        )
    return tools
