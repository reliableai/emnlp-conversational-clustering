"""Branching helpers for explicit branch groups and auto topic-discovery runs."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol, cast

from ..simulation.reporters import NullReporter, SimulationReporter
from ..simulation.runner import SessionContext, SimulationResult, build_topic_candidates_artifact
from .config import AUTO_TOPIC_BRANCH_METADATA_KEYS, AUTO_TOPIC_BRANCH_MODE, BatchRunConfig


@dataclass(frozen=True)
class RealizedBatchRun:
    source_config_index: int
    branch_index: int
    config: BatchRunConfig
    config_slug: str
    run_index: int
    run_id: str
    simulation_result: SimulationResult


@dataclass(frozen=True)
class AutoTopicBranchSpec:
    branch_index: int
    topic: dict[str, str]
    realized_config: BatchRunConfig
    realized_slug: str
    execution_branch_config: BatchRunConfig


@dataclass(frozen=True)
class AutoTopicSharedPayload:
    source_config_index: int
    source_config: BatchRunConfig
    source_config_slug: str
    run_index: int
    shared_run_id: str
    shared_result: SimulationResult
    shared_clustering_path: str
    generated_at_turn: int | None
    branch_specs: list[AutoTopicBranchSpec]


@dataclass(frozen=True)
class AutoTopicCompletedPayload:
    realized_runs: list[RealizedBatchRun]
    shared_clustering_path: str


@dataclass(frozen=True)
class BranchGroupChildSpec:
    config_index: int
    config: BatchRunConfig
    config_slug: str


@dataclass(frozen=True)
class BranchGroupSharedPayload:
    config_indexes: list[int]
    run_index: int
    shared_result: SimulationResult
    shared_clustering_path: str
    child_specs: list[BranchGroupChildSpec]


def _build_failed_topic_branch_run_id(*, run_index: int, branch_index: int) -> str:
    return f"{run_index + 1:03d}-failed-branch-{branch_index + 1:02d}"


def _build_failed_topic_postprocessing_run_id(*, run_index: int) -> str:
    return f"{run_index + 1:03d}-failed-postprocess"


class _BranchingRunner(Protocol):
    output_dir: str
    reporter_factory: Callable[[BatchRunConfig], SimulationReporter]

    def _build_run_id(self, run_index: int) -> str: ...

    def _build_run_paths(self, config_slug: str, run_id: str) -> tuple[str, str]: ...

    def _create_run_logger(
        self,
        config_slug: str,
        run_id: str,
        log_path: str,
    ) -> logging.Logger: ...

    def _close_logger(self, logger: logging.Logger) -> None: ...

    def _run_session_with_retries(
        self,
        *,
        config: BatchRunConfig,
        clustering_path: str,
        run_logger: logging.Logger,
        max_turns: int,
        reporter_factory: Callable[[], SimulationReporter],
        run_id: str,
        run_label: str,
        seed_result: SimulationResult | None = None,
        stop_on_topic_candidates: bool = False,
        log_retry_warnings: bool = True,
        log_terminal_errors: bool = True,
    ) -> tuple[SimulationResult, bool]: ...

    def _write_conversation_trace_artifact(
        self,
        *,
        clustering_path: str,
        simulation_result: SimulationResult,
    ) -> None: ...

    def _write_human_thought_trace_artifact(
        self,
        *,
        clustering_path: str,
        simulation_result: SimulationResult,
    ) -> None: ...

    def _write_judge_clustering_trace_artifact(
        self,
        *,
        clustering_path: str,
        simulation_result: SimulationResult,
    ) -> None: ...

    def _write_topic_candidates_artifact(
        self,
        *,
        clustering_path: str,
        simulation_result: SimulationResult,
    ) -> None: ...

    def _write_clustering_pov_history_artifact(
        self,
        *,
        clustering_path: str,
        simulation_result: SimulationResult,
    ) -> None: ...

    def _build_failed_simulation_result(
        self,
        *,
        reason: str,
        session: SessionContext | None = None,
    ) -> SimulationResult: ...


def config_uses_auto_topic_branching(config: BatchRunConfig) -> bool:
    return config.human_agent_config.topic_discovery.enabled


def build_topic_branch_config_slug(
    *,
    source_config_slug: str,
    run_index: int,
    branch_suffix: str | None,
    outcome: str,
) -> str:
    if outcome == "topic" and branch_suffix is not None:
        return f"{source_config_slug}--topic-{branch_suffix}"
    parts = [
        source_config_slug,
        f"run-{run_index + 1:03d}",
        outcome,
    ]
    if branch_suffix is not None:
        parts.append(branch_suffix)
    return "--".join(parts)


def build_topic_branch_slug_suffix(*, topic: dict[str, str]) -> str:
    payload = json.dumps(
        {
            "topic_id": topic["topic_id"],
            "label": topic["label"],
            "guidance": topic["guidance"],
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def build_selected_topic_artifact(config: BatchRunConfig) -> dict[str, object] | None:
    return build_selected_topic_artifact_with_generated_at_turn(
        config=config, generated_at_turn=None
    )


def build_selected_topic_artifact_with_generated_at_turn(
    *,
    config: BatchRunConfig,
    generated_at_turn: int | None,
) -> dict[str, object] | None:
    metadata = config.metadata
    raw_topic_id = metadata.get("topic_branch_topic_id")
    raw_label = metadata.get("topic_branch_topic_label")
    raw_guidance = metadata.get("topic_branch_topic_guidance")
    if not isinstance(raw_topic_id, str) or not raw_topic_id.strip():
        return None
    if not isinstance(raw_label, str) or not raw_label.strip():
        return None
    if not isinstance(raw_guidance, str) or not raw_guidance.strip():
        return None
    topic_id = raw_topic_id.strip()
    label = raw_label.strip()
    guidance = raw_guidance.strip()
    if generated_at_turn is None:
        raw_generated_at_turn = metadata.get("topic_branch_generated_at_turn")
        generated_at_turn = (
            raw_generated_at_turn
            if isinstance(raw_generated_at_turn, int)
            and not isinstance(raw_generated_at_turn, bool)
            else None
        )
    return {
        "topic_id": topic_id,
        "label": label,
        "guidance": guidance,
        "generated_at_turn": generated_at_turn,
    }


def write_selected_topic_artifact(
    *,
    clustering_path: str,
    config: BatchRunConfig,
    generated_at_turn: int | None = None,
) -> None:
    payload = build_selected_topic_artifact_with_generated_at_turn(
        config=config,
        generated_at_turn=generated_at_turn,
    )
    if payload is None:
        return
    output_path = os.path.join(os.path.dirname(clustering_path), "selected_topic.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def write_branch_topic_artifacts(
    *,
    clustering_path: str,
    config: BatchRunConfig,
    simulation_result: SimulationResult,
    generated_at_turn: int | None = None,
) -> None:
    if (
        build_selected_topic_artifact_with_generated_at_turn(
            config=config,
            generated_at_turn=generated_at_turn,
        )
        is not None
    ):
        write_selected_topic_artifact(
            clustering_path=clustering_path,
            config=config,
            generated_at_turn=generated_at_turn,
        )
        return
    payload = build_topic_candidates_artifact(simulation_result.state)
    if payload is None:
        return
    output_path = os.path.join(os.path.dirname(clustering_path), "topic_candidates.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")


def extract_branch_group_key(metadata: dict[str, object]) -> tuple[str, str, int] | None:
    branch_group = metadata.get("branch_group")
    branch_shared_turns = metadata.get("branch_shared_turns")
    branch_vary_by = metadata.get("branch_vary_by")
    if not isinstance(branch_group, str) or not branch_group.strip():
        return None
    if isinstance(branch_shared_turns, bool) or not isinstance(branch_shared_turns, int):
        return None
    if branch_shared_turns < 1:
        return None
    if not isinstance(branch_vary_by, str) or not branch_vary_by.strip():
        return None
    return (
        branch_group.strip(),
        branch_vary_by.strip(),
        branch_shared_turns,
    )


def freeze_branch_metadata_value(value: object) -> object:
    if isinstance(value, dict):
        frozen_items = [
            (
                freeze_branch_metadata_value(key),
                freeze_branch_metadata_value(nested_value),
            )
            for key, nested_value in value.items()
        ]
        return tuple(sorted(frozen_items, key=repr))
    if isinstance(value, list | tuple):
        sequence = cast(list[object] | tuple[object, ...], value)
        return tuple(freeze_branch_metadata_value(nested_value) for nested_value in sequence)
    if isinstance(value, set):
        frozen_items = (freeze_branch_metadata_value(nested_value) for nested_value in value)
        return tuple(sorted(frozen_items, key=repr))
    return value


def get_generated_at_turn(simulation_result: SimulationResult) -> int | None:
    raw_metadata = simulation_result.state.get("topic_candidates_metadata", {})
    if not isinstance(raw_metadata, dict):
        return None
    raw_generated_at_turn = raw_metadata.get("generated_at_turn")
    if isinstance(raw_generated_at_turn, bool) or not isinstance(raw_generated_at_turn, int):
        return None
    return raw_generated_at_turn


def get_topic_candidates(simulation_result: SimulationResult) -> list[dict[str, str]]:
    raw_candidates = simulation_result.state.get("topic_candidates", [])
    if not isinstance(raw_candidates, list):
        return []
    candidates: list[dict[str, str]] = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            continue
        topic_id = raw_candidate.get("topic_id")
        label = raw_candidate.get("label")
        guidance = raw_candidate.get("guidance")
        rationale = raw_candidate.get("rationale")
        if not isinstance(topic_id, str) or not topic_id.strip():
            continue
        if not isinstance(label, str) or not label.strip():
            continue
        if not isinstance(guidance, str) or not guidance.strip():
            continue
        candidate = {
            "topic_id": topic_id.strip(),
            "label": label.strip(),
            "guidance": guidance.strip(),
            "rationale": rationale.strip() if isinstance(rationale, str) else "",
        }
        candidates.append(candidate)
    return candidates


class BranchingCoordinator:
    def __init__(self, runner: _BranchingRunner) -> None:
        self._runner = runner

    def write_branch_mapping_artifact(
        self,
        *,
        shared_config_slug: str,
        branch_config_slugs: list[str],
    ) -> None:
        shared_dir = os.path.abspath(
            os.path.join(self._runner.output_dir, "_shared", shared_config_slug)
        )
        os.makedirs(shared_dir, exist_ok=True)
        output_path = os.path.join(shared_dir, "branch_mapping.json")
        payload = {
            "shared_config_slug": shared_config_slug,
            "branch_config_slugs": branch_config_slugs,
        }
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
            handle.write("\n")

    def write_topic_branch_mapping_artifact(
        self,
        *,
        shared_config_slug: str,
        shared_run_id: str,
        branch_records: list[dict[str, object]],
    ) -> None:
        shared_dir = os.path.abspath(
            os.path.join(self._runner.output_dir, "_shared", shared_config_slug, shared_run_id)
        )
        os.makedirs(shared_dir, exist_ok=True)
        output_path = os.path.join(shared_dir, "topic_branch_mapping.json")
        payload = {
            "shared_config_slug": shared_config_slug,
            "shared_run_id": shared_run_id,
            "branch_config_slugs": [
                record["config_slug"]
                for record in branch_records
                if isinstance(record.get("config_slug"), str)
            ],
            "branches": branch_records,
        }
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
            handle.write("\n")

    def run_shared_prefix(
        self,
        *,
        config: BatchRunConfig,
        shared_turns: int,
        shared_config_slug: str,
        run_index: int,
        stop_on_topic_candidates: bool = False,
    ) -> tuple[str, SimulationResult, str]:
        shared_slug = f"_shared/{shared_config_slug}"
        run_id = self._runner._build_run_id(run_index)
        clustering_path, log_path = self._runner._build_run_paths(shared_slug, run_id)
        run_logger = self._runner._create_run_logger(shared_slug, run_id, log_path)

        try:
            simulation_result, _ = self._runner._run_session_with_retries(
                config=config,
                clustering_path=clustering_path,
                run_logger=run_logger,
                max_turns=shared_turns,
                reporter_factory=NullReporter,
                run_id=run_id,
                run_label="shared prefix",
                stop_on_topic_candidates=stop_on_topic_candidates,
                log_retry_warnings=False,
                log_terminal_errors=True,
            )
            self._runner._write_conversation_trace_artifact(
                clustering_path=clustering_path,
                simulation_result=simulation_result,
            )
            self._runner._write_human_thought_trace_artifact(
                clustering_path=clustering_path,
                simulation_result=simulation_result,
            )
            self._runner._write_judge_clustering_trace_artifact(
                clustering_path=clustering_path,
                simulation_result=simulation_result,
            )
            self._runner._write_clustering_pov_history_artifact(
                clustering_path=clustering_path,
                simulation_result=simulation_result,
            )
            self._runner._write_topic_candidates_artifact(
                clustering_path=clustering_path,
                simulation_result=simulation_result,
            )
            return run_id, simulation_result, clustering_path
        finally:
            self._runner._close_logger(run_logger)

    def run_single_from_shared_prefix(
        self,
        *,
        config: BatchRunConfig,
        config_slug: str,
        run_index: int,
        shared_result: SimulationResult,
        shared_clustering_path: str,
    ) -> tuple[str, SimulationResult]:
        run_id = self._runner._build_run_id(run_index)
        clustering_path, log_path = self._runner._build_run_paths(config_slug, run_id)
        run_logger = self._runner._create_run_logger(config_slug, run_id, log_path)

        def return_shared_result() -> tuple[str, SimulationResult]:
            simulation_result = copy.deepcopy(shared_result)
            self._runner._write_conversation_trace_artifact(
                clustering_path=clustering_path,
                simulation_result=simulation_result,
            )
            self._runner._write_human_thought_trace_artifact(
                clustering_path=clustering_path,
                simulation_result=simulation_result,
            )
            self._runner._write_judge_clustering_trace_artifact(
                clustering_path=clustering_path,
                simulation_result=simulation_result,
            )
            self._runner._write_clustering_pov_history_artifact(
                clustering_path=clustering_path,
                simulation_result=simulation_result,
            )
            write_branch_topic_artifacts(
                clustering_path=clustering_path,
                config=config,
                simulation_result=simulation_result,
                generated_at_turn=get_generated_at_turn(shared_result),
            )
            return run_id, simulation_result

        try:
            if shared_result.stopped_reason not in {"max_turns", "topic_discovery_handoff"}:
                if shared_result.stopped_reason == "clustering_saved" and os.path.exists(
                    shared_clustering_path
                ):
                    os.makedirs(os.path.dirname(clustering_path), exist_ok=True)
                    shutil.copyfile(shared_clustering_path, clustering_path)
                return return_shared_result()

            remaining_turns = config.max_turns - shared_result.turns_run
            if remaining_turns <= 0:
                return return_shared_result()

            simulation_result, was_successful = self._runner._run_session_with_retries(
                config=config,
                clustering_path=clustering_path,
                run_logger=run_logger,
                max_turns=remaining_turns,
                reporter_factory=lambda: self._runner.reporter_factory(config),
                run_id=run_id,
                run_label="branched run",
                seed_result=shared_result,
            )
            if was_successful:
                self._runner._write_conversation_trace_artifact(
                    clustering_path=clustering_path,
                    simulation_result=simulation_result,
                )
                self._runner._write_human_thought_trace_artifact(
                    clustering_path=clustering_path,
                    simulation_result=simulation_result,
                )
                self._runner._write_judge_clustering_trace_artifact(
                    clustering_path=clustering_path,
                    simulation_result=simulation_result,
                )
                self._runner._write_clustering_pov_history_artifact(
                    clustering_path=clustering_path,
                    simulation_result=simulation_result,
                )
                write_branch_topic_artifacts(
                    clustering_path=clustering_path,
                    config=config,
                    simulation_result=simulation_result,
                    generated_at_turn=get_generated_at_turn(shared_result),
                )
            return run_id, simulation_result
        finally:
            self._runner._close_logger(run_logger)

    def build_topic_branch_config(
        self,
        *,
        source_config: BatchRunConfig,
        source_config_slug: str,
        run_index: int,
        topic: dict[str, str],
    ) -> tuple[BatchRunConfig, str]:
        topic_id = topic["topic_id"]
        branch_suffix = build_topic_branch_slug_suffix(topic=topic)
        config_slug = build_topic_branch_config_slug(
            source_config_slug=source_config_slug,
            run_index=run_index,
            branch_suffix=branch_suffix,
            outcome="topic",
        )
        metadata = {
            **{
                key: value
                for key, value in source_config.metadata.items()
                if key not in AUTO_TOPIC_BRANCH_METADATA_KEYS
            },
            "topic_branch_mode": AUTO_TOPIC_BRANCH_MODE,
            "topic_branch_outcome": "branched",
            "topic_branch_source_config_slug": source_config_slug,
            "topic_branch_topic_id": topic_id,
            "topic_branch_topic_label": topic["label"],
            "topic_branch_topic_guidance": topic["guidance"],
        }
        human_agent_config = replace(
            source_config.human_agent_config,
            topic_discovery=replace(
                source_config.human_agent_config.topic_discovery,
                enabled=False,
                turn_budget=None,
            ),
        )
        config = replace(
            source_config,
            name=f"{source_config.name}__topic_{topic_id}",
            human_agent_config=human_agent_config,
            num_runs=1,
            metadata=metadata,
        )
        return config, config_slug

    def build_unbranched_topic_result_config(
        self,
        *,
        source_config: BatchRunConfig,
        source_config_slug: str,
        shared_run_id: str,
        run_index: int,
        outcome: str,
    ) -> tuple[BatchRunConfig, str]:
        config_slug = build_topic_branch_config_slug(
            source_config_slug=source_config_slug,
            run_index=run_index,
            branch_suffix=None,
            outcome=outcome,
        )
        metadata = {
            **{
                key: value
                for key, value in source_config.metadata.items()
                if key not in AUTO_TOPIC_BRANCH_METADATA_KEYS
            },
            "topic_branch_mode": AUTO_TOPIC_BRANCH_MODE,
            "topic_branch_outcome": outcome,
            "topic_branch_source_config_slug": source_config_slug,
            "topic_branch_source_run_id": shared_run_id,
            "topic_branch_source_run_index": run_index,
        }
        config = replace(
            source_config,
            name=f"{source_config.name}__run_{run_index + 1:03d}__{outcome}",
            num_runs=1,
            metadata=metadata,
        )
        return config, config_slug

    def build_failed_topic_branch_result(
        self,
        *,
        source_config_index: int,
        branch_index: int,
        realized_config: BatchRunConfig,
        realized_slug: str,
        run_index: int,
    ) -> RealizedBatchRun:
        failed_run_id = _build_failed_topic_branch_run_id(
            run_index=run_index,
            branch_index=branch_index,
        )
        return RealizedBatchRun(
            source_config_index=source_config_index,
            branch_index=branch_index,
            config=realized_config,
            config_slug=realized_slug,
            run_index=run_index,
            run_id=failed_run_id,
            simulation_result=self._runner._build_failed_simulation_result(
                reason="run_error_exception",
                session=None,
            ),
        )

    @staticmethod
    def build_topic_branch_record(
        *,
        config_slug: str,
        run_id: str,
        topic: dict[str, str],
        generated_at_turn: int | None,
    ) -> dict[str, object]:
        return {
            "config_slug": config_slug,
            "run_id": run_id,
            "topic_id": topic["topic_id"],
            "label": topic["label"],
            "guidance": topic["guidance"],
            "generated_at_turn": generated_at_turn,
        }

    def finalize_auto_topic_results(
        self,
        *,
        payload: AutoTopicSharedPayload,
        branch_results_by_index: dict[int, RealizedBatchRun],
        branch_records_by_index: dict[int, dict[str, object]],
    ) -> list[RealizedBatchRun]:
        realized_runs = [
            branch_results_by_index[branch_index]
            for branch_index in sorted(branch_results_by_index)
        ]
        branch_records = [
            branch_records_by_index[branch_index]
            for branch_index in sorted(branch_records_by_index)
        ]
        try:
            self.write_topic_branch_mapping_artifact(
                shared_config_slug=payload.source_config_slug,
                shared_run_id=payload.shared_run_id,
                branch_records=branch_records,
            )
        except Exception:
            failed_run_id = _build_failed_topic_postprocessing_run_id(run_index=payload.run_index)
            fallback_config, fallback_slug = self.build_unbranched_topic_result_config(
                source_config=payload.source_config,
                source_config_slug=payload.source_config_slug,
                shared_run_id=failed_run_id,
                run_index=payload.run_index,
                outcome="shared_run_terminal",
            )
            realized_runs.append(
                RealizedBatchRun(
                    source_config_index=payload.source_config_index,
                    branch_index=len(payload.branch_specs),
                    config=fallback_config,
                    config_slug=fallback_slug,
                    run_index=payload.run_index,
                    run_id=failed_run_id,
                    simulation_result=self._runner._build_failed_simulation_result(
                        reason="run_error_exception",
                        session=None,
                    ),
                )
            )
        return realized_runs

    def finalize_branch_group_results(
        self,
        *,
        payload: BranchGroupSharedPayload,
        branch_results_by_config_index: dict[int, tuple[str, SimulationResult]],
    ) -> list[tuple[int, str, SimulationResult]]:
        return [
            (
                child_spec.config_index,
                branch_results_by_config_index[child_spec.config_index][0],
                branch_results_by_config_index[child_spec.config_index][1],
            )
            for child_spec in payload.child_specs
        ]

    @staticmethod
    def cleanup_shared_clustering_artifact(shared_clustering_path: str) -> None:
        if os.path.exists(shared_clustering_path):
            os.remove(shared_clustering_path)

    def plan_topic_branching_single_run(
        self,
        *,
        source_config: BatchRunConfig,
        execution_config: BatchRunConfig,
        config_index: int,
        config_slug: str,
        run_index: int,
    ) -> AutoTopicCompletedPayload | AutoTopicSharedPayload:
        shared_clustering_path: str | None = None
        shared_run_id, shared_result, shared_clustering_path = self.run_shared_prefix(
            config=execution_config,
            shared_turns=execution_config.max_turns,
            shared_config_slug=config_slug,
            run_index=run_index,
            stop_on_topic_candidates=True,
        )

        try:
            if shared_result.stopped_reason != "topic_discovery_handoff":
                outcome = (
                    "no_topics_generated"
                    if shared_result.stopped_reason == "max_turns"
                    else "shared_run_terminal"
                )
                realized_config, realized_slug = self.build_unbranched_topic_result_config(
                    source_config=source_config,
                    source_config_slug=config_slug,
                    shared_run_id=shared_run_id,
                    run_index=run_index,
                    outcome=outcome,
                )
                run_id = shared_run_id
                clustering_path, _log_path = self._runner._build_run_paths(realized_slug, run_id)
                os.makedirs(os.path.dirname(clustering_path), exist_ok=True)
                if os.path.exists(shared_clustering_path):
                    shutil.copyfile(shared_clustering_path, clustering_path)
                self._runner._write_conversation_trace_artifact(
                    clustering_path=clustering_path,
                    simulation_result=shared_result,
                )
                self._runner._write_human_thought_trace_artifact(
                    clustering_path=clustering_path,
                    simulation_result=shared_result,
                )
                self._runner._write_judge_clustering_trace_artifact(
                    clustering_path=clustering_path,
                    simulation_result=shared_result,
                )
                self._runner._write_clustering_pov_history_artifact(
                    clustering_path=clustering_path,
                    simulation_result=shared_result,
                )
                self._runner._write_topic_candidates_artifact(
                    clustering_path=clustering_path,
                    simulation_result=shared_result,
                )
                return AutoTopicCompletedPayload(
                    realized_runs=[
                        RealizedBatchRun(
                            source_config_index=config_index,
                            branch_index=0,
                            config=realized_config,
                            config_slug=realized_slug,
                            run_index=run_index,
                            run_id=run_id,
                            simulation_result=shared_result,
                        )
                    ],
                    shared_clustering_path=shared_clustering_path,
                )

            generated_at_turn = get_generated_at_turn(shared_result)
            topics = get_topic_candidates(shared_result)
            branch_specs: list[AutoTopicBranchSpec] = []
            for branch_index, topic in enumerate(topics):
                realized_config, realized_slug = self.build_topic_branch_config(
                    source_config=source_config,
                    source_config_slug=config_slug,
                    run_index=run_index,
                    topic=topic,
                )
                execution_branch_config = replace(
                    realized_config,
                    embedding_force_recompute=execution_config.embedding_force_recompute,
                )
                branch_specs.append(
                    AutoTopicBranchSpec(
                        branch_index=branch_index,
                        topic=topic,
                        realized_config=realized_config,
                        realized_slug=realized_slug,
                        execution_branch_config=execution_branch_config,
                    )
                )
            return AutoTopicSharedPayload(
                source_config_index=config_index,
                source_config=source_config,
                source_config_slug=config_slug,
                run_index=run_index,
                shared_run_id=shared_run_id,
                shared_result=shared_result,
                shared_clustering_path=shared_clustering_path,
                generated_at_turn=generated_at_turn,
                branch_specs=branch_specs,
            )
        except Exception:
            if shared_clustering_path is not None:
                self.cleanup_shared_clustering_artifact(shared_clustering_path)
            raise

    def plan_branch_group_single_run(
        self,
        *,
        configs: list[BatchRunConfig],
        config_slugs: list[str],
        config_indexes: list[int],
        run_index: int,
    ) -> BranchGroupSharedPayload:
        if not config_indexes:
            raise ValueError("branch_group planning requires at least one config index")

        first_config = configs[config_indexes[0]]
        raw_shared_turns = first_config.metadata.get("branch_shared_turns")
        if isinstance(raw_shared_turns, bool) or not isinstance(raw_shared_turns, int):
            raise ValueError("branch metadata 'branch_shared_turns' must be an integer")
        shared_turns = raw_shared_turns
        raw_branch_vary_by = first_config.metadata.get("branch_vary_by")
        if not isinstance(raw_branch_vary_by, str) or not raw_branch_vary_by.strip():
            raise ValueError("branch metadata 'branch_vary_by' must be a non-empty string")
        branch_vary_by = raw_branch_vary_by.strip()
        branch_pre_vary_value_sentinel = object()
        raw_pre_vary_value = first_config.metadata.get(
            "branch_pre_vary_value",
            branch_pre_vary_value_sentinel,
        )
        if raw_pre_vary_value is branch_pre_vary_value_sentinel:
            raise ValueError("branch metadata 'branch_pre_vary_value' must be provided")
        pre_vary_value_key = freeze_branch_metadata_value(raw_pre_vary_value)
        backend_config_sentinel = object()
        raw_pre_backend_config = first_config.metadata.get(
            "branch_pre_clustering_agent_backend_config",
            backend_config_sentinel,
        )
        if (
            branch_vary_by == "clustering_agent_backend"
            and raw_pre_backend_config is backend_config_sentinel
        ):
            raise ValueError(
                "branch metadata 'branch_pre_clustering_agent_backend_config' must be provided"
            )
        pre_backend_config_key = freeze_branch_metadata_value(raw_pre_backend_config)

        for config_index in config_indexes[1:]:
            config = configs[config_index]
            if config.num_runs != first_config.num_runs:
                raise ValueError("all configs in a branch_group must share the same num_runs")
            if config.max_turns != first_config.max_turns:
                raise ValueError("all configs in a branch_group must share the same max_turns")
            if config.metadata.get("branch_shared_turns") != shared_turns:
                raise ValueError("all configs in a branch_group must share branch_shared_turns")
            if config.metadata.get("branch_vary_by") != branch_vary_by:
                raise ValueError("all configs in a branch_group must share branch_vary_by")
            raw_other_pre_vary_value = config.metadata.get(
                "branch_pre_vary_value",
                branch_pre_vary_value_sentinel,
            )
            if raw_other_pre_vary_value is branch_pre_vary_value_sentinel:
                raise ValueError("all configs in a branch_group must define branch_pre_vary_value")
            if freeze_branch_metadata_value(raw_other_pre_vary_value) != pre_vary_value_key:
                raise ValueError("all configs in a branch_group must share branch_pre_vary_value")
            if branch_vary_by == "clustering_agent_backend":
                raw_other_backend_config = config.metadata.get(
                    "branch_pre_clustering_agent_backend_config",
                    backend_config_sentinel,
                )
                if raw_other_backend_config is backend_config_sentinel:
                    raise ValueError(
                        "all configs in a branch_group must define "
                        "branch_pre_clustering_agent_backend_config"
                    )
                if freeze_branch_metadata_value(raw_other_backend_config) != pre_backend_config_key:
                    raise ValueError(
                        "all configs in a branch_group must share "
                        "branch_pre_clustering_agent_backend_config"
                    )

        if branch_vary_by == "clustering_agent_backend":
            shared_config = replace(
                first_config,
                clustering_agent_backend=raw_pre_vary_value,
                clustering_agent_backend_config=raw_pre_backend_config,
            )
        else:
            shared_config = replace(
                first_config,
                **{branch_vary_by: raw_pre_vary_value},
            )

        _shared_run_id, shared_result, shared_clustering_path = self.run_shared_prefix(
            config=shared_config,
            shared_turns=shared_turns,
            shared_config_slug=config_slugs[config_indexes[0]],
            run_index=run_index,
        )

        child_specs = [
            BranchGroupChildSpec(
                config_index=config_index,
                config=configs[config_index],
                config_slug=config_slugs[config_index],
            )
            for config_index in config_indexes
        ]
        return BranchGroupSharedPayload(
            config_indexes=list(config_indexes),
            run_index=run_index,
            shared_result=shared_result,
            shared_clustering_path=shared_clustering_path,
            child_specs=child_specs,
        )
