"""Batch-run entrypoints and TOML config parsing for batch-run execution."""

import os
import tomllib
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Literal, cast

from conversational_clustering.clustering_agent.backends import (
    DEFAULT_CLUSTERING_AGENT_BACKEND,
    coerce_clustering_agent_backend_config,
)
from conversational_clustering.data_processing import DATASETS
from conversational_clustering.models import (
    EmbeddingModelConfig,
    ModelConfig,
    parse_embedding_model_config,
    parse_model_config,
)
from conversational_clustering.prompts import (
    get_human_agent_system_prompt,
    get_human_verifier_system_prompt,
    normalize_guidance,
)
from conversational_clustering.simulation.reporters import (
    ConsoleReporter,
    NullReporter,
    SimulationReporter,
)

from ..human_turn_injection import HumanTurnInjectionConfig
from ..topic_discovery import TopicDiscoveryConfig
from .benchmark_specs import attach_generated_benchmark_specs
from .config import (
    AUTO_TOPIC_BRANCH_METADATA_KEYS,
    GENERATED_BENCHMARK_SPEC_METADATA_KEY,
    BatchRunConfig,
    HumanAgentConfig,
)
from .io import write_jsonl, write_summary
from .runner import BatchRunner

_SUPPORTED_TOP_LEVEL_CONFIG_KEYS = {
    "output_dir",
    "num_runners",
    "reporter",
    "base",
    "sweep",
    "branch",
}
_SUPPORTED_BATCH_RUN_FIELDS = {
    "name",
    "dataset_name",
    "clustering_agent_backend",
    "clustering_agent_backend_config",
    "human_agent_config",
    "human_verifier_prompt_id",
    "max_verification_retries",
    "limit",
    "human_agent_model_config",
    "human_verifier_model_config",
    "enable_embeddings",
    "enable_intermittent_clustering",
    "embedding_model_config",
    "embedding_force_recompute",
    "embedding_batch_size",
    "max_turns",
    "num_runs",
    "random_seed",
    "metadata",
}
_SWEEPABLE_BATCH_RUN_FIELDS = {
    "clustering_agent_backend",
    "clustering_agent_backend_config",
    "human_agent_config",
    "human_verifier_prompt_id",
    "max_verification_retries",
    "human_agent_model_config",
    "human_verifier_model_config",
    "max_turns",
    "random_seed",
}
_BRANCHABLE_BATCH_RUN_FIELDS = {
    "clustering_agent_backend",
    "clustering_agent_backend_config",
    "human_agent_config",
    "human_verifier_prompt_id",
    "max_verification_retries",
    "human_agent_model_config",
    "human_verifier_model_config",
}
_BRANCH_RESERVED_METADATA_KEYS = {
    "branch_group",
    "branch_shared_turns",
    "branch_vary_by",
    "branch_pre_vary_value",
    "branch_pre_clustering_agent_backend_config",
}
_RESERVED_METADATA_KEYS = _BRANCH_RESERVED_METADATA_KEYS | set(AUTO_TOPIC_BRANCH_METADATA_KEYS)
_RESERVED_METADATA_KEYS.add(GENERATED_BENCHMARK_SPEC_METADATA_KEY)
_ENTRY_RESERVED_METADATA_KEYS = set(AUTO_TOPIC_BRANCH_METADATA_KEYS)
_ENTRY_RESERVED_METADATA_KEYS.add(GENERATED_BENCHMARK_SPEC_METADATA_KEY)


class _ReporterFactory:
    def __init__(
        self,
        reporter_cls: Callable[[], SimulationReporter],
        *,
        batch_progress_enabled: bool,
    ) -> None:
        self._reporter_cls = reporter_cls
        self.batch_progress_enabled = batch_progress_enabled

    def __call__(self, _config: BatchRunConfig) -> SimulationReporter:
        return self._reporter_cls()


def _batch_runs_entry_require_positive_int(
    value: object,
    *,
    field_name: str,
    entry_index: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"'{field_name}' in batch-run entry #{entry_index} must be an integer >= 1"
        )
    if value < 1:
        raise ValueError(
            f"'{field_name}' in batch-run entry #{entry_index} must be an integer >= 1"
        )
    return value


def _batch_runs_entry_require_non_negative_int(
    value: object,
    *,
    field_name: str,
    entry_index: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"'{field_name}' in batch-run entry #{entry_index} must be an integer >= 0"
        )
    if value < 0:
        raise ValueError(
            f"'{field_name}' in batch-run entry #{entry_index} must be an integer >= 0"
        )
    return value


def _parse_batch_run_config_toml(
    config_file: str,
) -> tuple[str, list[BatchRunConfig], _ReporterFactory, int]:
    with open(config_file, "rb") as handle:
        config_data = tomllib.load(handle)

    if not isinstance(config_data, dict):
        raise ValueError("batch-runs config must parse to a table")
    unknown_top_level_keys = sorted(set(config_data) - _SUPPORTED_TOP_LEVEL_CONFIG_KEYS)
    if unknown_top_level_keys:
        raise ValueError(
            "batch-runs config has unsupported top-level key(s): "
            + ", ".join(unknown_top_level_keys)
        )

    output_dir = config_data.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("batch-runs config must define a non-empty 'output_dir' string")

    num_runners = config_data.get("num_runners")
    if isinstance(num_runners, bool) or not isinstance(num_runners, int) or num_runners < 1:
        raise ValueError("batch-runs config must define 'num_runners' as an integer >= 1")

    reporter_spec = config_data.get("reporter", "null")
    if isinstance(reporter_spec, dict):
        reporter_type = reporter_spec.get("type", "null")
    else:
        reporter_type = reporter_spec
    if not isinstance(reporter_type, str):
        raise ValueError("'reporter' must be a string or a table with a string 'type'")

    reporter_type = reporter_type.lower()
    if reporter_type == "batch_progress":
        reporter_cls: Callable[[], SimulationReporter] = NullReporter
    else:
        reporter_map: dict[str, Callable[[], SimulationReporter]] = {
            "console": ConsoleReporter,
            "null": NullReporter,
        }
        reporter_cls = reporter_map.get(reporter_type)
        if reporter_cls is None:
            supported = ", ".join(sorted((*reporter_map.keys(), "batch_progress")))
            raise ValueError(
                f"unsupported reporter '{reporter_type}'. Supported values: {supported}"
            )

    reporter_factory = _ReporterFactory(
        reporter_cls,
        batch_progress_enabled=reporter_type == "batch_progress",
    )

    raw_base = config_data.get("base")
    if not isinstance(raw_base, dict):
        raise ValueError("batch-runs config must define a [base] table")

    expanded_entries: list[dict[str, object]]
    raw_sweep = config_data.get("sweep")
    sweep_vary_by: str | None = None
    if raw_sweep is None:
        expanded_entries = [dict(raw_base)]
    else:
        if not isinstance(raw_sweep, dict):
            raise ValueError("'sweep' must be a table when provided")

        raw_vary_by = raw_sweep.get("vary_by")
        if not isinstance(raw_vary_by, str) or not raw_vary_by.strip():
            raise ValueError("'sweep.vary_by' must be a non-empty string")
        sweep_vary_by = raw_vary_by.strip()
        if sweep_vary_by not in _SWEEPABLE_BATCH_RUN_FIELDS:
            supported = ", ".join(sorted(_SWEEPABLE_BATCH_RUN_FIELDS))
            raise ValueError(f"'sweep.vary_by' must be one of: {supported}")
        raw_values = raw_sweep.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError("'sweep.values' must be a non-empty list")
        normalized_sweep_values: list[object] = []
        seen_value_keys: set[object] = set()
        for value_index, value in enumerate(raw_values):
            normalized_value = _normalize_batch_run_vary_value(
                vary_by=sweep_vary_by,
                vary_value=value,
                location=f"'sweep.values[{value_index}]'",
            )
            comparison_key = _build_sweep_value_comparison_key(normalized_value)
            if comparison_key in seen_value_keys:
                raise ValueError("'sweep.values' must not contain duplicate values")
            seen_value_keys.add(comparison_key)
            normalized_sweep_values.append(normalized_value)
        _validate_single_factor_human_agent_config_variation(
            vary_by=sweep_vary_by,
            values=normalized_sweep_values,
            location="'sweep.values'",
        )

        if sweep_vary_by in raw_base:
            raise ValueError(
                f"'base' must not define the field selected by 'sweep.vary_by' ('{sweep_vary_by}')"
            )
        if sweep_vary_by == "random_seed" and raw_base.get("limit") is None:
            raise ValueError("'base.limit' must be defined when 'sweep.vary_by' is 'random_seed'")

        expanded_entries = []
        for vary_value in normalized_sweep_values:
            expanded_entry = dict(raw_base)
            expanded_entry[sweep_vary_by] = vary_value
            expanded_entry["name"] = _build_sweep_entry_name(
                vary_by=sweep_vary_by,
                vary_value=vary_value,
            )
            expanded_entries.append(expanded_entry)

    raw_branch = config_data.get("branch")
    if raw_branch is not None:
        if not isinstance(raw_branch, dict):
            raise ValueError("'branch' must be a table when provided")

        supported_branch_fields = {"shared_turns", "vary_by", "values"}
        unknown_branch_fields = sorted(set(raw_branch) - supported_branch_fields)
        if unknown_branch_fields:
            raise ValueError(
                "'branch' has unsupported field(s): " + ", ".join(unknown_branch_fields)
            )

        raw_shared_turns = raw_branch.get("shared_turns")
        if isinstance(raw_shared_turns, bool) or not isinstance(raw_shared_turns, int):
            raise ValueError("'branch.shared_turns' must be an integer >= 1")
        if raw_shared_turns < 1:
            raise ValueError("'branch.shared_turns' must be an integer >= 1")
        shared_turns = raw_shared_turns

        raw_branch_vary_by = raw_branch.get("vary_by")
        if not isinstance(raw_branch_vary_by, str) or not raw_branch_vary_by.strip():
            raise ValueError("'branch.vary_by' must be a non-empty string")
        branch_vary_by = raw_branch_vary_by.strip()
        if branch_vary_by not in _BRANCHABLE_BATCH_RUN_FIELDS:
            supported = ", ".join(sorted(_BRANCHABLE_BATCH_RUN_FIELDS))
            raise ValueError(f"'branch.vary_by' must be one of: {supported}")
        if sweep_vary_by is not None and branch_vary_by == sweep_vary_by:
            raise ValueError("'branch.vary_by' must differ from 'sweep.vary_by'")

        raw_branch_values = raw_branch.get("values")
        if not isinstance(raw_branch_values, list) or not raw_branch_values:
            raise ValueError("'branch.values' must be a non-empty list")
        normalized_branch_values: list[object] = []
        seen_value_keys: set[object] = set()
        for value_index, value in enumerate(raw_branch_values):
            normalized_value = _normalize_batch_run_vary_value(
                vary_by=branch_vary_by,
                vary_value=value,
                location=f"'branch.values[{value_index}]'",
            )
            comparison_key = _build_sweep_value_comparison_key(normalized_value)
            if comparison_key in seen_value_keys:
                raise ValueError("'branch.values' must not contain duplicate values")
            seen_value_keys.add(comparison_key)
            normalized_branch_values.append(normalized_value)
        _validate_single_factor_human_agent_config_variation(
            vary_by=branch_vary_by,
            values=normalized_branch_values,
            location="'branch.values'",
        )

        branched_entries: list[dict[str, object]] = []
        for entry_index, expanded_entry in enumerate(expanded_entries, start=1):
            max_turns_raw = expanded_entry.get("max_turns", 8)
            max_turns = _batch_runs_entry_require_positive_int(
                max_turns_raw,
                field_name="max_turns",
                entry_index=entry_index,
            )
            if shared_turns >= max_turns:
                raise ValueError(
                    "'branch.shared_turns' must be less than resolved "
                    f"'max_turns' for expanded batch-run entry #{entry_index}"
                )

            branch_group_name = expanded_entry.get("name")
            if isinstance(branch_group_name, str) and branch_group_name.strip():
                branch_group = branch_group_name.strip()
            else:
                branch_group = f"entry_{entry_index}"

            raw_metadata = expanded_entry.get("metadata", {})
            if raw_metadata is None:
                base_metadata: dict[str, object] = {}
            elif isinstance(raw_metadata, dict):
                base_metadata = dict(raw_metadata)
            else:
                raise ValueError(
                    f"'metadata' in expanded batch-run entry #{entry_index} must be a table"
                )
            reserved_metadata_keys = sorted(
                key
                for key in base_metadata
                if isinstance(key, str) and key in _RESERVED_METADATA_KEYS
            )
            if reserved_metadata_keys:
                raise ValueError(
                    "'metadata' in expanded batch-run entry "
                    f"#{entry_index} must not define reserved key(s) when 'branch' "
                    "is enabled: " + ", ".join(reserved_metadata_keys)
                )
            pre_branch_entry = _parse_batch_run_entry(
                dict(expanded_entry),
                entry_index=entry_index,
            )
            if pre_branch_entry.human_agent_config.topic_discovery.enabled:
                raise ValueError(
                    "'branch' must not be used when topic discovery is enabled; "
                    "topic-discovery branching is automatic"
                )
            pre_branch_vary_value = getattr(pre_branch_entry, branch_vary_by)
            if branch_vary_by == "human_agent_config":
                pre_branch_vary_value = _resolve_branch_shared_prefix_human_agent_config(
                    pre_branch_value=pre_branch_entry.human_agent_config,
                    branch_values=normalized_branch_values,
                    shared_turns=shared_turns,
                    location=f"'branch.values' for expanded batch-run entry #{entry_index}",
                )
                _validate_branch_human_agent_config_shared_prefix_safety(
                    pre_branch_value=pre_branch_vary_value,
                    branch_values=normalized_branch_values,
                    shared_turns=shared_turns,
                    location=f"'branch.values' for expanded batch-run entry #{entry_index}",
                )
            pre_branch_backend_config = None
            if branch_vary_by == "clustering_agent_backend":
                pre_branch_backend_config = pre_branch_entry.clustering_agent_backend_config

            for branch_value in normalized_branch_values:
                branched_entry = dict(expanded_entry)
                branched_entry[branch_vary_by] = branch_value
                branched_entry["name"] = _build_branch_entry_name(
                    base_name=branch_group,
                    vary_by=branch_vary_by,
                    vary_value=branch_value,
                )
                metadata = dict(base_metadata)
                metadata["branch_group"] = branch_group
                metadata["branch_shared_turns"] = shared_turns
                metadata["branch_vary_by"] = branch_vary_by
                metadata["branch_pre_vary_value"] = pre_branch_vary_value
                if pre_branch_backend_config is not None:
                    metadata["branch_pre_clustering_agent_backend_config"] = (
                        pre_branch_backend_config
                    )
                branched_entry["metadata"] = metadata
                branched_entries.append(branched_entry)
        expanded_entries = branched_entries

    batch_runs = [
        _parse_batch_run_entry(expanded_entry, entry_index=index)
        for index, expanded_entry in enumerate(expanded_entries, start=1)
    ]

    return output_dir, batch_runs, reporter_factory, num_runners


def _normalize_sweep_value_for_comparison(value: object) -> object:
    if isinstance(value, str):
        return value.strip()
    return value


def _build_sweep_value_comparison_key(value: object) -> object:
    normalized = _normalize_sweep_value_for_comparison(value)
    try:
        hash(normalized)
    except TypeError:
        return _freeze_unhashable_sweep_value(normalized)
    return normalized


def _freeze_unhashable_sweep_value(value: object) -> object:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        frozen_items = [
            (
                _freeze_unhashable_sweep_value(key),
                _freeze_unhashable_sweep_value(nested_value),
            )
            for key, nested_value in value.items()
        ]
        return tuple(sorted(frozen_items, key=repr))
    if isinstance(value, list | tuple):
        sequence = cast(list[object] | tuple[object, ...], value)
        return tuple(_freeze_unhashable_sweep_value(nested_value) for nested_value in sequence)
    if isinstance(value, set):
        frozen_items = (_freeze_unhashable_sweep_value(nested_value) for nested_value in value)
        return tuple(sorted(frozen_items, key=repr))
    return value


def _normalize_vary_value(vary_value: object) -> str:
    if isinstance(vary_value, str):
        normalized = vary_value.strip()
        if normalized:
            return normalized
        return "empty"
    if isinstance(vary_value, dict):
        if not vary_value:
            return "empty"
        vary_value_dict = cast(dict[object, object], vary_value)
        parts = []
        for key in sorted(vary_value_dict.keys(), key=repr):
            parts.append(
                f"{_normalize_vary_value(key)}_{_normalize_vary_value(vary_value_dict[key])}"
            )
        return "__".join(parts)
    if isinstance(vary_value, list | tuple):
        sequence = cast(list[object] | tuple[object, ...], vary_value)
        if not sequence:
            return "empty"
        return "__".join(_normalize_vary_value(item) for item in sequence)
    if isinstance(vary_value, set):
        values = sorted(_normalize_vary_value(item) for item in vary_value)
        if not values:
            return "empty"
        return "__".join(values)
    return str(vary_value)


def _build_sweep_entry_name(*, vary_by: str, vary_value: object) -> str:
    normalized_value = _normalize_vary_value(vary_value)
    return f"{vary_by}_{normalized_value}"


def _build_branch_entry_name(*, base_name: str, vary_by: str, vary_value: object) -> str:
    normalized_value = _normalize_vary_value(vary_value)
    return f"{base_name}__{vary_by}_{normalized_value}"


def _normalize_batch_run_vary_value(
    *,
    vary_by: str,
    vary_value: object,
    location: str,
) -> object:
    if vary_by == "human_agent_config":
        return _parse_human_agent_config_value(
            vary_value,
            location=location,
        ).to_dict()
    return vary_value


def _validate_single_factor_human_agent_config_variation(
    *,
    vary_by: str,
    values: list[object],
    location: str,
) -> None:
    if vary_by != "human_agent_config" or len(values) < 2:
        return

    typed_values: list[dict[str, object]] = []
    for value in values:
        if not isinstance(value, dict):
            return
        typed_values.append(cast(dict[str, object], value))

    varying_fields: list[str] = []
    for field_name in ("prompt_id", "guidance", "persona", "topic_discovery", "turn_injection"):
        comparison_keys = {
            _build_sweep_value_comparison_key(value.get(field_name)) for value in typed_values
        }
        if len(comparison_keys) > 1:
            varying_fields.append(field_name)

    if len(varying_fields) <= 1:
        return

    raise ValueError(
        f"{location} may vary at most one top-level field within 'human_agent_config'. "
        "Found differences in: " + ", ".join(varying_fields)
    )


def _validate_branch_human_agent_config_shared_prefix_safety(
    *,
    pre_branch_value: object,
    branch_values: list[object],
    shared_turns: int,
    location: str,
) -> None:
    if isinstance(pre_branch_value, HumanAgentConfig):
        pre_branch_dict = pre_branch_value.to_dict()
    elif isinstance(pre_branch_value, dict):
        pre_branch_dict = cast(dict[str, object], pre_branch_value)
    else:
        return

    pre_branch_topic_discovery = pre_branch_dict.get("topic_discovery")
    pre_branch_key = _build_sweep_value_comparison_key(pre_branch_topic_discovery)
    pre_branch_persona = pre_branch_dict.get("persona")
    pre_branch_persona_key = _build_sweep_value_comparison_key(pre_branch_persona)
    pre_branch_turn_injection = pre_branch_dict.get("turn_injection")
    pre_branch_turn_injection_key = _build_sweep_value_comparison_key(pre_branch_turn_injection)
    pre_branch_has_shared_prefix_turn_injection = _turn_injection_activates_during_shared_prefix(
        turn_injection=pre_branch_turn_injection,
        shared_turns=shared_turns,
    )
    saw_omitted_turn_injection = False
    saw_explicit_shared_prefix_turn_injection = False
    for branch_value in branch_values:
        if not isinstance(branch_value, dict):
            return
        branch_value_dict = cast(dict[str, object], branch_value)
        branch_topic_discovery = branch_value_dict.get("topic_discovery")
        branch_key = _build_sweep_value_comparison_key(branch_topic_discovery)
        if branch_key != pre_branch_key:
            raise ValueError(
                f"{location} must not vary 'human_agent_config.topic_discovery' across branch "
                "runs; use a sweep instead"
            )
        if "persona" in branch_value_dict:
            branch_persona = branch_value_dict.get("persona")
        else:
            branch_persona = pre_branch_persona
        branch_persona_key = _build_sweep_value_comparison_key(branch_persona)
        if branch_persona_key != pre_branch_persona_key:
            raise ValueError(
                f"{location} must not vary 'human_agent_config.persona' across branch runs "
                "when shared turns are enabled; use a sweep instead"
            )
        if "turn_injection" not in branch_value_dict:
            saw_omitted_turn_injection = True
            continue
        branch_turn_injection = branch_value_dict.get("turn_injection")
        if _turn_injection_activates_during_shared_prefix(
            turn_injection=branch_turn_injection,
            shared_turns=shared_turns,
        ):
            saw_explicit_shared_prefix_turn_injection = True
        branch_turn_injection_key = _build_sweep_value_comparison_key(branch_turn_injection)
        if branch_turn_injection_key == pre_branch_turn_injection_key:
            continue
        if _turn_injection_activates_during_shared_prefix(
            turn_injection=branch_turn_injection,
            shared_turns=shared_turns,
        ):
            raise ValueError(
                f"{location} must not vary 'human_agent_config.turn_injection' across branch "
                "runs when it activates during shared turns; require turn_index > "
                "'branch.shared_turns' or use a sweep instead"
            )
    if (
        saw_omitted_turn_injection
        and saw_explicit_shared_prefix_turn_injection
        and not pre_branch_has_shared_prefix_turn_injection
    ):
        raise ValueError(
            f"{location} must not mix omitted and explicit "
            "'human_agent_config.turn_injection' values when the explicit injection activates "
            "during shared turns; define the shared-prefix injection in the base config or use "
            "a sweep instead"
        )


def _resolve_branch_shared_prefix_human_agent_config(
    *,
    pre_branch_value: HumanAgentConfig,
    branch_values: list[object],
    shared_turns: int,
    location: str,
) -> HumanAgentConfig:
    resolved_pre_branch_value = pre_branch_value
    branch_persona = _resolve_explicit_shared_branch_persona(branch_values)
    if branch_persona is not None and resolved_pre_branch_value.persona != branch_persona:
        if resolved_pre_branch_value.persona is not None:
            raise ValueError(
                f"{location} must not define both base and branch shared-prefix "
                "'human_agent_config.persona' values; multiple personas are unsupported"
            )
        resolved_pre_branch_value = replace(resolved_pre_branch_value, persona=branch_persona)

    branch_turn_injection = _resolve_explicit_shared_branch_turn_injection(branch_values)
    if branch_turn_injection is None:
        return resolved_pre_branch_value
    if branch_turn_injection.turn_index > shared_turns:
        return resolved_pre_branch_value
    pre_branch_turn_injection = resolved_pre_branch_value.turn_injection
    if pre_branch_turn_injection is not None:
        if pre_branch_turn_injection == branch_turn_injection:
            return resolved_pre_branch_value
        if pre_branch_turn_injection.turn_index <= shared_turns:
            raise ValueError(
                f"{location} must not define both base and branch shared-prefix "
                "'human_agent_config.turn_injection' values; multiple scheduled injections are "
                "unsupported"
            )
        raise ValueError(
            f"{location} must not combine a base late 'human_agent_config.turn_injection' "
            "value with a branch shared-prefix value; multiple scheduled injections are "
            "unsupported"
        )
    return replace(resolved_pre_branch_value, turn_injection=branch_turn_injection)


def _resolve_explicit_shared_branch_persona(
    branch_values: list[object],
) -> str | None:
    shared_persona: str | None = None
    for branch_value in branch_values:
        if not isinstance(branch_value, dict):
            return None
        branch_value_dict = cast(dict[str, object], branch_value)
        raw_persona = branch_value_dict.get("persona")
        if raw_persona is None:
            return None
        if not isinstance(raw_persona, str):
            return None
        if shared_persona is None:
            shared_persona = raw_persona
            continue
        if raw_persona != shared_persona:
            return None
    return shared_persona


def _resolve_explicit_shared_branch_turn_injection(
    branch_values: list[object],
) -> HumanTurnInjectionConfig | None:
    shared_raw_turn_injection: object | None = None
    shared_turn_injection_key: object | None = None
    for branch_value in branch_values:
        if not isinstance(branch_value, dict):
            return None
        branch_value_dict = cast(dict[str, object], branch_value)
        if "turn_injection" not in branch_value_dict:
            return None
        raw_turn_injection = branch_value_dict.get("turn_injection")
        turn_injection_key = _build_sweep_value_comparison_key(raw_turn_injection)
        if shared_turn_injection_key is None:
            shared_raw_turn_injection = raw_turn_injection
            shared_turn_injection_key = turn_injection_key
            continue
        if turn_injection_key != shared_turn_injection_key:
            return None
    if shared_turn_injection_key is None:
        return None
    return _parse_turn_injection_config(
        shared_raw_turn_injection,
        location="'branch.values'.turn_injection",
    )


def _turn_injection_activates_during_shared_prefix(
    *,
    turn_injection: object,
    shared_turns: int,
) -> bool:
    if not isinstance(turn_injection, dict):
        return False
    turn_injection_dict = cast(dict[str, object], turn_injection)
    raw_turn_index = turn_injection_dict.get("turn_index")
    if isinstance(raw_turn_index, bool) or not isinstance(raw_turn_index, int):
        return False
    return raw_turn_index <= shared_turns


def _parse_batch_run_entry(raw_entry: dict[str, object], *, entry_index: int) -> BatchRunConfig:
    unknown_fields = sorted(set(raw_entry) - _SUPPORTED_BATCH_RUN_FIELDS)
    if unknown_fields:
        raise ValueError(
            "expanded batch-run entry "
            f"#{entry_index} has unsupported field(s): {', '.join(unknown_fields)}"
        )

    clustering_agent_backend = raw_entry.get(
        "clustering_agent_backend",
        DEFAULT_CLUSTERING_AGENT_BACKEND,
    )
    if not isinstance(clustering_agent_backend, str) or not clustering_agent_backend.strip():
        raise ValueError(
            "'clustering_agent_backend' in "
            f"expanded batch-run entry #{entry_index} must be a non-empty string"
        )
    clustering_agent_backend = clustering_agent_backend.strip()

    clustering_agent_backend_config_data = raw_entry.get("clustering_agent_backend_config")
    if clustering_agent_backend_config_data is None:
        clustering_agent_backend_config_data = {}
    if not isinstance(clustering_agent_backend_config_data, dict):
        raise ValueError(
            "'clustering_agent_backend_config' in "
            f"expanded batch-run entry #{entry_index} must be a table"
        )
    clustering_agent_backend_config_data = cast(
        dict[str, object], clustering_agent_backend_config_data
    )
    try:
        clustering_agent_backend_config = coerce_clustering_agent_backend_config(
            backend=clustering_agent_backend,
            backend_config=clustering_agent_backend_config_data,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "invalid clustering agent backend configuration in "
            f"expanded batch-run entry #{entry_index}: {exc}"
        ) from exc

    human_agent_model_config_data = raw_entry.get("human_agent_model_config")
    if human_agent_model_config_data is None:
        human_agent_model_config = None
    elif isinstance(human_agent_model_config_data, dict):
        human_agent_model_config = _parse_model_config(
            cast(dict[str, object], human_agent_model_config_data),
            entry_index=entry_index,
        )
    else:
        raise ValueError(
            f"'human_agent_model_config' in expanded batch-run entry #{entry_index} must be a table"
        )

    human_verifier_model_config_data = raw_entry.get("human_verifier_model_config")
    if human_verifier_model_config_data is None:
        human_verifier_model_config = None
    elif isinstance(human_verifier_model_config_data, dict):
        human_verifier_model_config = _parse_model_config(
            cast(dict[str, object], human_verifier_model_config_data),
            entry_index=entry_index,
        )
    else:
        raise ValueError(
            "'human_verifier_model_config' in "
            f"expanded batch-run entry #{entry_index} must be a table"
        )

    enable_embeddings_raw = raw_entry.get("enable_embeddings", False)
    if not isinstance(enable_embeddings_raw, bool):
        raise ValueError(
            f"'enable_embeddings' in expanded batch-run entry #{entry_index} must be a boolean"
        )
    enable_embeddings = enable_embeddings_raw

    enable_intermittent_clustering_raw = raw_entry.get("enable_intermittent_clustering", False)
    if not isinstance(enable_intermittent_clustering_raw, bool):
        raise ValueError(
            "'enable_intermittent_clustering' in "
            f"expanded batch-run entry #{entry_index} must be a boolean"
        )
    enable_intermittent_clustering = enable_intermittent_clustering_raw

    embedding_model_config_data = raw_entry.get("embedding_model_config")
    if embedding_model_config_data is None:
        embedding_model_config = None
    elif isinstance(embedding_model_config_data, dict):
        embedding_model_config = _parse_embedding_model_config(
            cast(dict[str, object], embedding_model_config_data),
            entry_index=entry_index,
        )
    else:
        raise ValueError(
            f"'embedding_model_config' in expanded batch-run entry #{entry_index} must be a table"
        )

    embedding_force_recompute_raw = raw_entry.get("embedding_force_recompute", False)
    if not isinstance(embedding_force_recompute_raw, bool):
        raise ValueError(
            "'embedding_force_recompute' in "
            f"expanded batch-run entry #{entry_index} must be a boolean"
        )
    embedding_force_recompute = embedding_force_recompute_raw

    embedding_batch_size = _batch_runs_entry_require_positive_int(
        raw_entry.get("embedding_batch_size", 64),
        field_name="embedding_batch_size",
        entry_index=entry_index,
    )

    metadata = raw_entry.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"'metadata' in expanded batch-run entry #{entry_index} must be a table")
    invalid_metadata_keys = [key for key in metadata if not isinstance(key, str)]
    if invalid_metadata_keys:
        raise ValueError(
            f"'metadata' in expanded batch-run entry #{entry_index} must only use string keys"
        )
    metadata = cast(dict[str, Any], metadata)
    reserved_metadata_keys = sorted(key for key in metadata if key in _ENTRY_RESERVED_METADATA_KEYS)
    if reserved_metadata_keys:
        raise ValueError(
            f"'metadata' in expanded batch-run entry #{entry_index} must not define reserved "
            "key(s): " + ", ".join(reserved_metadata_keys)
        )

    name = raw_entry.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"'name' in expanded batch-run entry #{entry_index} must be a non-empty string"
        )
    name = name.strip()

    dataset_name = raw_entry.get("dataset_name")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError(
            f"'dataset_name' in expanded batch-run entry #{entry_index} must be a non-empty string"
        )
    dataset_name = dataset_name.strip()

    human_agent_config = _parse_human_agent_config(raw_entry, entry_index=entry_index)

    human_verifier_prompt_id = raw_entry.get(
        "human_verifier_prompt_id",
        "baseline_v1",
    )
    if not isinstance(human_verifier_prompt_id, str) or not human_verifier_prompt_id.strip():
        raise ValueError(
            "'human_verifier_prompt_id' in "
            f"expanded batch-run entry #{entry_index} must be a non-empty string"
        )
    human_verifier_prompt_id = human_verifier_prompt_id.strip()
    try:
        get_human_verifier_system_prompt(human_verifier_prompt_id)
    except ValueError as exc:
        raise ValueError(
            "'human_verifier_prompt_id' in "
            f"expanded batch-run entry #{entry_index} is invalid: {exc}"
        ) from exc

    max_turns = _batch_runs_entry_require_positive_int(
        raw_entry.get("max_turns", 8),
        field_name="max_turns",
        entry_index=entry_index,
    )
    num_runs = _batch_runs_entry_require_positive_int(
        raw_entry.get("num_runs", 1),
        field_name="num_runs",
        entry_index=entry_index,
    )
    max_verification_retries = _batch_runs_entry_require_non_negative_int(
        raw_entry.get("max_verification_retries", 2),
        field_name="max_verification_retries",
        entry_index=entry_index,
    )

    raw_limit = raw_entry.get("limit")
    if raw_limit is None:
        limit = None
    else:
        limit = _batch_runs_entry_require_positive_int(
            raw_limit,
            field_name="limit",
            entry_index=entry_index,
        )
    raw_random_seed = raw_entry.get("random_seed")
    if raw_random_seed is None:
        random_seed = 42  # we always force a random seed on all simulation runs
    else:
        random_seed = _batch_runs_entry_require_non_negative_int(
            raw_random_seed,
            field_name="random_seed",
            entry_index=entry_index,
        )

    return BatchRunConfig(
        name=name,
        dataset_name=dataset_name,
        clustering_agent_backend=clustering_agent_backend,
        clustering_agent_backend_config=clustering_agent_backend_config,
        human_agent_config=human_agent_config,
        human_verifier_prompt_id=human_verifier_prompt_id,
        max_verification_retries=max_verification_retries,
        limit=limit,
        human_agent_model_config=human_agent_model_config,
        human_verifier_model_config=human_verifier_model_config,
        enable_embeddings=enable_embeddings,
        enable_intermittent_clustering=enable_intermittent_clustering,
        embedding_model_config=embedding_model_config,
        embedding_force_recompute=embedding_force_recompute,
        embedding_batch_size=embedding_batch_size,
        max_turns=max_turns,
        num_runs=num_runs,
        random_seed=random_seed,
        metadata=metadata,
    )


def _parse_human_agent_config(
    raw_entry: dict[str, object],
    *,
    entry_index: int,
) -> HumanAgentConfig:
    raw_human_agent_config = raw_entry.get("human_agent_config")
    if raw_human_agent_config is None:
        try:
            return HumanAgentConfig()
        except ValueError as exc:
            raise ValueError(
                f"'human_agent_config' in expanded batch-run entry #{entry_index} is invalid: {exc}"
            ) from exc
    return _parse_human_agent_config_value(
        raw_human_agent_config,
        location=f"'human_agent_config' in expanded batch-run entry #{entry_index}",
    )


def _parse_human_agent_config_value(
    raw_human_agent_config: object,
    *,
    location: str,
) -> HumanAgentConfig:
    if not isinstance(raw_human_agent_config, dict):
        raise ValueError(f"{location} must be a table")
    raw_human_agent_config = cast(dict[str, object], raw_human_agent_config)
    unknown_fields = sorted(
        set(raw_human_agent_config)
        - {"prompt_id", "guidance", "persona", "topic_discovery", "turn_injection"}
    )
    if unknown_fields:
        raise ValueError(f"{location} has unsupported field(s): {', '.join(unknown_fields)}")
    raw_prompt_id = raw_human_agent_config.get("prompt_id", "baseline_v3")
    if not isinstance(raw_prompt_id, str) or not raw_prompt_id.strip():
        raise ValueError(f"{location}.prompt_id must be a non-empty string")
    prompt_id = raw_prompt_id.strip()
    try:
        get_human_agent_system_prompt(prompt_id)
    except ValueError as exc:
        raise ValueError(f"{location}.prompt_id is invalid: {exc}") from exc
    raw_guidance = raw_human_agent_config.get("guidance")
    if raw_guidance is None:
        guidance = None
    elif isinstance(raw_guidance, str):
        guidance = normalize_guidance(raw_guidance)
    else:
        raise ValueError(f"{location}.guidance must be a string or null")
    raw_persona = raw_human_agent_config.get("persona")
    if raw_persona is None:
        persona = None
    elif isinstance(raw_persona, str):
        persona = raw_persona
    else:
        raise ValueError(f"{location}.persona must be a string or null")
    topic_discovery = _parse_topic_discovery_config(
        raw_human_agent_config.get("topic_discovery"),
        location=f"{location}.topic_discovery",
    )
    turn_injection = _parse_turn_injection_config(
        raw_human_agent_config.get("turn_injection"),
        location=f"{location}.turn_injection",
    )
    try:
        return HumanAgentConfig(
            prompt_id=prompt_id,
            guidance=guidance,
            persona=persona,
            topic_discovery=topic_discovery,
            turn_injection=turn_injection,
        )
    except ValueError as exc:
        raise ValueError(f"{location} is invalid: {exc}") from exc


def _parse_turn_injection_config(
    raw_turn_injection: object,
    *,
    location: str,
) -> HumanTurnInjectionConfig | None:
    if raw_turn_injection is None:
        return None
    if not isinstance(raw_turn_injection, dict):
        raise ValueError(f"{location} must be a table")
    turn_injection = cast(dict[str, object], raw_turn_injection)
    try:
        return HumanTurnInjectionConfig.from_dict(turn_injection)
    except ValueError as exc:
        raise ValueError(f"{location} is invalid: {exc}") from exc


def _parse_topic_discovery_config(
    raw_topic_discovery: object,
    *,
    location: str,
) -> TopicDiscoveryConfig:
    if raw_topic_discovery is None:
        return TopicDiscoveryConfig()
    if not isinstance(raw_topic_discovery, dict):
        raise ValueError(f"{location} must be a table")
    topic_discovery = cast(dict[str, object], raw_topic_discovery)
    unknown_fields = sorted(
        set(topic_discovery)
        - {
            "enabled",
            "trigger_mode",
            "turn_budget",
            "max_rounds",
            "min_new_topics_per_round",
        }
    )
    if unknown_fields:
        raise ValueError(f"{location} has unsupported field(s): {', '.join(unknown_fields)}")

    raw_enabled = topic_discovery.get("enabled", False)
    if not isinstance(raw_enabled, bool):
        raise ValueError(f"{location}.enabled must be a boolean")
    enabled = raw_enabled

    raw_trigger_mode = topic_discovery.get("trigger_mode", "turn_budget")
    if not isinstance(raw_trigger_mode, str) or not raw_trigger_mode.strip():
        raise ValueError(f"{location}.trigger_mode must be a non-empty string")
    trigger_mode = cast(Literal["turn_budget", "human_tool"], raw_trigger_mode.strip())

    raw_turn_budget = topic_discovery.get("turn_budget")
    if raw_turn_budget is None:
        turn_budget = None
    else:
        if isinstance(raw_turn_budget, bool) or not isinstance(raw_turn_budget, int):
            raise ValueError(f"{location}.turn_budget must be an integer or null")
        turn_budget = raw_turn_budget

    raw_max_rounds = topic_discovery.get("max_rounds", 3)
    if isinstance(raw_max_rounds, bool) or not isinstance(raw_max_rounds, int):
        raise ValueError(f"{location}.max_rounds must be an integer >= 1")
    max_rounds = raw_max_rounds

    raw_min_new_topics_per_round = topic_discovery.get("min_new_topics_per_round", 2)
    if isinstance(raw_min_new_topics_per_round, bool) or not isinstance(
        raw_min_new_topics_per_round, int
    ):
        raise ValueError(f"{location}.min_new_topics_per_round must be an integer >= 0")
    min_new_topics_per_round = raw_min_new_topics_per_round

    try:
        return TopicDiscoveryConfig(
            enabled=enabled,
            trigger_mode=trigger_mode,
            turn_budget=turn_budget,
            max_rounds=max_rounds,
            min_new_topics_per_round=min_new_topics_per_round,
        )
    except ValueError as exc:
        raise ValueError(f"{location} is invalid: {exc}") from exc


def _parse_model_config(raw_model_config: dict[str, object], *, entry_index: int) -> ModelConfig:
    try:
        return parse_model_config(raw_model_config)
    except ValueError as exc:
        raise ValueError(
            f"invalid model configuration in expanded batch-run entry #{entry_index}: {exc}"
        ) from exc


def _parse_embedding_model_config(
    raw_embedding_model_config: dict[str, object],
    *,
    entry_index: int,
) -> EmbeddingModelConfig:
    try:
        return parse_embedding_model_config(raw_embedding_model_config)
    except ValueError as exc:
        raise ValueError(
            "invalid embedding model configuration in "
            f"expanded batch-run entry #{entry_index}: {exc}"
        ) from exc


def validate_batch_run_datasets(batch_runs: list[BatchRunConfig]) -> None:
    valid_datasets = ", ".join(sorted(DATASETS))
    issues: list[str] = []
    for batch_run_config in batch_runs:
        dataset_name = batch_run_config.dataset_name
        dataset_config = DATASETS.get(dataset_name)
        if dataset_config is None:
            issues.append(
                f"{batch_run_config.name} -> unknown dataset '{dataset_name}' "
                f"(valid: {valid_datasets})"
            )
            continue

        dataset_path = dataset_config["path"]
        if not os.path.exists(dataset_path):
            issues.append(
                f"{batch_run_config.name} -> missing file for dataset "
                f"'{dataset_name}' at '{dataset_path}'"
            )

    if issues:
        raise ValueError("Invalid batch-run datasets: " + "; ".join(issues))


def run_batch_runs(config_file: str) -> str:
    (output_dir, batch_runs, reporter_factory, num_runners) = _parse_batch_run_config_toml(
        config_file
    )

    try:
        validate_batch_run_datasets(batch_runs)
    except ValueError as exc:
        raise ValueError(exc) from exc

    batch_runs = attach_generated_benchmark_specs(output_dir=output_dir, configs=batch_runs)

    runner = BatchRunner(
        reporter_factory=reporter_factory,
        output_dir=output_dir,
        batch_progress_enabled=reporter_factory.batch_progress_enabled,
    )

    total_runs = sum(config.num_runs for config in batch_runs)
    print(
        f"Starting batch run execution with {len(batch_runs)} configs / {total_runs} total runs..."
    )

    results = runner.run_batch_parallel(batch_runs, max_workers=num_runners)

    write_jsonl(results, os.path.join(output_dir, "results.jsonl"))
    write_summary(results, os.path.join(output_dir, "summary.json"))

    print("Batch runs complete!")
    return output_dir
