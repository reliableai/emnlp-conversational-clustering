"""Tool definitions for backend_v1 dataset access and clustering persistence."""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from ....data_processing.data_loader import DataLoader
from ....data_processing.item_records import DatasetItemRecord, build_dataset_item_records
from ..pov_bank import make_append_clustering_povs_tool

DEFAULT_CLUSTERING_SAVE_PATH = "data/clusterings/latest.json"


@dataclass
class BackendV1RuntimeState:
    clustering_pov_bank: list[str] | None = None


def _dataset_to_text(data_loader: DataLoader | None) -> str:
    if data_loader is None:
        return ""
    return "\n".join(
        json.dumps(
            {"uuid": record.uuid, "value": record.value},
            ensure_ascii=True,
            separators=(",", ":"),
        )
        for record in _build_dataset_records(data_loader)
    )


def _build_dataset_records(data_loader: DataLoader | None) -> list[DatasetItemRecord]:
    if data_loader is None:
        return []
    return build_dataset_item_records(data_loader.to_context())


def make_read_dataset(data_loader: DataLoader | None) -> BaseTool:
    @tool
    def read_dataset() -> str:
        """Returns the user's current dataset as newline-delimited JSON records with uuid/value."""
        return _dataset_to_text(data_loader)

    return read_dataset


class ClusterSpec(BaseModel):
    name: str = Field(..., description="Short label for the cluster.")
    member_uuids: list[str] = Field(
        ...,
        description="Dataset item UUIDs from read_dataset that belong to this cluster.",
        min_length=1,
    )


def _normalize_member_uuids(member_uuids: list[str]) -> list[str]:
    return [str(member_uuid).strip() for member_uuid in member_uuids]


def _validate_clustering(data_loader: DataLoader | None, clusters: list[ClusterSpec]) -> None:
    all_member_uuids: list[str] = []
    for cluster in clusters:
        normalized_member_uuids = _normalize_member_uuids(cluster.member_uuids)
        all_member_uuids.extend(normalized_member_uuids)

    duplicate_member_uuids = [
        member_uuid for member_uuid, count in Counter(all_member_uuids).items() if count > 1
    ]
    if duplicate_member_uuids:
        raise ValueError(
            "Each dataset UUID must appear exactly once. "
            "These UUIDs are duplicated: "
            + ", ".join(sorted(duplicate_member_uuids)[:10])
            + ("..." if len(duplicate_member_uuids) > 10 else "")
        )

    if data_loader is None:
        return

    dataset_uuids = [record.uuid for record in _build_dataset_records(data_loader)]
    dataset_unique = sorted(set(dataset_uuids))
    clustering_unique = sorted(set(all_member_uuids))
    missing_items = sorted(set(dataset_unique) - set(clustering_unique))
    extra_items = sorted(set(clustering_unique) - set(dataset_uuids))

    if missing_items or extra_items:
        parts = []
        if missing_items:
            parts.append(
                "Missing UUIDs: "
                + ", ".join(missing_items[:10])
                + ("..." if len(missing_items) > 10 else "")
            )
        if extra_items:
            parts.append(
                "Extra UUIDs not in dataset: "
                + ", ".join(extra_items[:10])
                + ("..." if len(extra_items) > 10 else "")
            )
        raise ValueError("Clustering must enumerate all dataset UUIDs. " + " ".join(parts))


def _duplicate_dataset_note(data_loader: DataLoader | None) -> str:
    if data_loader is None:
        return ""
    counts = Counter(record.value for record in _build_dataset_records(data_loader))
    duplicates = [(item, count) for item, count in counts.items() if count > 1]
    if not duplicates:
        return ""
    duplicates.sort(key=lambda pair: (-pair[1], pair[0]))
    formatted = [f"{item} (x{count})" for item, count in duplicates[:10]]
    suffix = "..." if len(duplicates) > 10 else ""
    return " Note: dataset contains duplicates: " + ", ".join(formatted) + suffix


def make_save_clustering(
    data_loader: DataLoader | None,
    default_output_path: str = DEFAULT_CLUSTERING_SAVE_PATH,
) -> BaseTool:
    @tool
    def save_clustering(clusters: list[ClusterSpec]) -> str:
        """
        Saves a complete clustering to disk. Requires every dataset item to appear
        exactly once across all clusters using the item UUIDs returned by read_dataset.
        """
        try:
            if not clusters:
                raise ValueError("Clusters must be a non-empty list.")

            _validate_clustering(data_loader, clusters)
            records_by_uuid = {
                record.uuid: record for record in _build_dataset_records(data_loader)
            }

            output_dir = os.path.dirname(default_output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            payload = {
                "clusters": [
                    {
                        "name": cluster.name,
                        "members": [
                            records_by_uuid[member_uuid].value
                            for member_uuid in _normalize_member_uuids(cluster.member_uuids)
                        ],
                    }
                    for cluster in clusters
                ],
            }
            with open(default_output_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2)

            total_items = sum(len(cluster.member_uuids) for cluster in clusters)
            return (
                "Clustering saved to "
                f"{default_output_path} with {len(clusters)} clusters and {total_items} items."
                + _duplicate_dataset_note(data_loader)
            )
        except ValueError as exc:
            return f"Could not save clustering. Reason: {exc}"
        except Exception:
            return "Could not save clustering due to an unexpected error."

    return save_clustering


def build_tools(
    data_loader: DataLoader | None,
    save_path: str | None = None,
    *,
    runtime_state: BackendV1RuntimeState | None = None,
) -> list[BaseTool]:
    resolved_runtime_state = runtime_state or BackendV1RuntimeState()
    save_tool = (
        make_save_clustering(data_loader)
        if save_path is None
        else make_save_clustering(data_loader, default_output_path=save_path)
    )
    return [
        make_read_dataset(data_loader),
        make_append_clustering_povs_tool(runtime_state=resolved_runtime_state),
        save_tool,
    ]
