"""Load and represent saved clustering artifacts for evaluator inputs."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LoadedClustering:
    assignment: dict[str, int]  # item_text -> cluster_id
    cluster_names: list[str]  # cluster_id -> cluster name


def parse_clustering_payload(payload: Mapping[str, Any]) -> LoadedClustering:
    clusters = payload.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        raise ValueError("Invalid clustering JSON: expected non-empty 'clusters' list")

    assignment: dict[str, int] = {}
    cluster_names: list[str] = []

    for idx, c in enumerate(clusters):
        if not isinstance(c, dict):
            raise ValueError(f"Invalid clustering JSON: cluster {idx} must be an object")

        name = str(c.get("name", f"cluster_{idx}")).strip() or f"cluster_{idx}"
        members = c.get("members", [])

        if not isinstance(members, list) or not members:
            raise ValueError(f"Invalid clustering JSON: cluster {idx} has invalid/empty 'members'")

        cluster_names.append(name)

        for item in members:
            item_str = str(item).strip()
            if not item_str:
                raise ValueError(f"Invalid clustering JSON: cluster {idx} has an empty member item")

            if item_str in assignment:
                if assignment[item_str] != idx:
                    raise ValueError(f"Item appears multiple times in clustering: {item_str!r}")
                continue

            assignment[item_str] = idx

    return LoadedClustering(assignment=assignment, cluster_names=cluster_names)


def load_saved_clustering(path: str | os.PathLike[str]) -> LoadedClustering:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Invalid clustering JSON: expected top-level object")
    return parse_clustering_payload(payload)
