"""Batch execution orchestration for running multiple simulation batch runs."""

import copy
import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, cast

import httpcore
import httpx

from ..accounting import empty_usage_summary, merge_usage_summaries
from ..clustering_agent.state import AgentState
from ..data_processing.embeddings import build_embeddings_cache_path, ensure_embeddings_for_items
from ..models import EmbeddingModelConfig
from ..simulation.reporters import NullReporter, SimulationReporter
from ..simulation.runner import (
    SessionContext,
    SimulationResult,
    build_clustering_pov_history_artifact,
    build_conversation_trace_artifact,
    build_data_loader,
    build_human_thought_trace_artifact,
    build_judge_clustering_trace_artifact,
    build_topic_candidates_artifact,
    create_session,
    run_human_simulation,
)
from .branching import (
    AutoTopicBranchSpec,
    AutoTopicCompletedPayload,
    AutoTopicSharedPayload,
    BranchGroupChildSpec,
    BranchGroupSharedPayload,
    BranchingCoordinator,
    RealizedBatchRun,
    config_uses_auto_topic_branching,
    extract_branch_group_key,
)
from .config import BatchRunConfig, BatchRunResult, HumanAgentConfig
from .reporting import BatchProgressReporter

_REMOTE_PROTOCOL_MAX_ATTEMPTS = 3
_REMOTE_PROTOCOL_INITIAL_BACKOFF_SECONDS = 1.0
_REPORTER_TIMEOUT_SECONDS = 0.2


@dataclass
class _EmbeddingPreparationSpec:
    dataset_name: str
    model_config: EmbeddingModelConfig
    batch_size: int
    force_recompute: bool = False
    item_texts_by_id: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _AutoTopicBranchTask:
    parent_key: tuple[int, int]
    branch_spec: AutoTopicBranchSpec


@dataclass(frozen=True)
class _BranchGroupBranchTask:
    parent_key: tuple[tuple[int, ...], int]
    child_spec: BranchGroupChildSpec


@dataclass
class _AutoTopicAggregateState:
    payload: AutoTopicSharedPayload
    branch_results_by_index: dict[int, RealizedBatchRun] = field(default_factory=dict)
    branch_records_by_index: dict[int, dict[str, object]] = field(default_factory=dict)
    pending_children: int = 0


@dataclass
class _BranchGroupAggregateState:
    payload: BranchGroupSharedPayload
    branch_results_by_config_index: dict[int, tuple[str, SimulationResult]] = field(
        default_factory=dict
    )
    pending_children: int = 0


class BatchRunner:
    def __init__(
        self,
        reporter_factory: Callable[[BatchRunConfig], SimulationReporter] | None = None,
        output_dir: str = "data/batch_runs",
        batch_progress_enabled: bool = False,
    ):
        self.reporter_factory = reporter_factory or (lambda _: NullReporter())
        self.output_dir = output_dir
        self.batch_progress_enabled = batch_progress_enabled
        self._run_slot_semaphore: threading.BoundedSemaphore | None = None
        self._branching = BranchingCoordinator(self)

    @staticmethod
    def _slugify(value: str, *, fallback: str = "run", max_len: int = 32) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
        if not normalized:
            return fallback
        return normalized[:max_len]

    def _build_config_slug(self, *, config_index: int) -> str:
        prefix = f"{config_index + 1:02d}"
        suffix = uuid.uuid4().hex[:8]
        return f"{prefix}-{suffix}"

    def _build_run_id(self, run_index: int) -> str:
        suffix = uuid.uuid4().hex[:8]
        parts = []
        parts.append(f"{run_index + 1:03d}")
        parts.append(suffix)
        return "-".join(parts)

    @staticmethod
    def _resolve_max_workers(max_workers: int | None) -> int:
        if max_workers is not None:
            return max(1, max_workers)
        cpu_count = os.cpu_count() or 1
        return min(32, cpu_count + 4)

    @contextmanager
    def _acquire_run_slot(self):
        semaphore = self._run_slot_semaphore
        if semaphore is None:
            yield
            return
        semaphore.acquire()
        try:
            yield
        finally:
            semaphore.release()

    def _build_run_paths(self, config_slug: str, run_id: str) -> tuple[str, str]:
        run_dir = os.path.abspath(os.path.join(self.output_dir, config_slug, run_id))
        clustering_path = os.path.join(run_dir, "clustering.json")
        log_path = os.path.join(run_dir, "session.log")
        return clustering_path, log_path

    @staticmethod
    def _write_conversation_trace_artifact(
        *,
        clustering_path: str,
        simulation_result: SimulationResult,
    ) -> None:
        output_path = os.path.join(os.path.dirname(clustering_path), "conversation_trace.json")
        payload = build_conversation_trace_artifact(simulation_result.state)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
            handle.write("\n")

    @staticmethod
    def _write_human_thought_trace_artifact(
        *,
        clustering_path: str,
        simulation_result: SimulationResult,
    ) -> None:
        output_path = os.path.join(os.path.dirname(clustering_path), "human_thought_trace.json")
        payload = build_human_thought_trace_artifact(simulation_result.state)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
            handle.write("\n")

    @staticmethod
    def _write_judge_clustering_trace_artifact(
        *,
        clustering_path: str,
        simulation_result: SimulationResult,
    ) -> None:
        payload = build_judge_clustering_trace_artifact(simulation_result.state)
        if payload is None:
            return
        output_path = os.path.join(
            os.path.dirname(clustering_path),
            "judge_clustering_trace.json",
        )
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
            handle.write("\n")

    @staticmethod
    def _write_clustering_pov_history_artifact(
        *,
        clustering_path: str,
        simulation_result: SimulationResult,
    ) -> None:
        payload = build_clustering_pov_history_artifact(simulation_result.state)
        if payload is None:
            return
        output_path = os.path.join(os.path.dirname(clustering_path), "clustering_pov_history.json")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
            handle.write("\n")

    @staticmethod
    def _write_topic_candidates_artifact(
        *,
        clustering_path: str,
        simulation_result: SimulationResult,
    ) -> None:
        payload = build_topic_candidates_artifact(simulation_result.state)
        if payload is None:
            return
        output_path = os.path.join(os.path.dirname(clustering_path), "topic_candidates.json")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
            handle.write("\n")

    def _create_run_logger(self, config_slug: str, run_id: str, log_path: str) -> logging.Logger:
        logger = logging.getLogger(f"conversational_clustering.run.{config_slug}.{run_id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        self._close_logger(logger)

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    @staticmethod
    def _close_logger(logger: logging.Logger) -> None:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    @staticmethod
    def _is_retryable_remote_protocol_error(exc: Exception) -> bool:
        pending: list[BaseException] = [exc]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            current_id = id(current)
            if current_id in seen:
                continue
            seen.add(current_id)
            if isinstance(current, httpx.RemoteProtocolError | httpcore.RemoteProtocolError):
                return True
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            if current.__context__ is not None:
                pending.append(current.__context__)
        return False

    @staticmethod
    def _is_timeout_error(exc: Exception) -> bool:
        pending: list[BaseException] = [exc]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            current_id = id(current)
            if current_id in seen:
                continue
            seen.add(current_id)
            if isinstance(
                current, TimeoutError | httpx.TimeoutException | httpcore.TimeoutException
            ):
                return True
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            if current.__context__ is not None:
                pending.append(current.__context__)
        return False

    @staticmethod
    def _build_failed_simulation_result(
        *,
        reason: str,
        session: SessionContext | None = None,
    ) -> SimulationResult:
        state: AgentState
        if session is None:
            state = cast(
                AgentState,
                {"messages": [], "dataset_metadata": {"size": 0}, "iteration_count": 0},
            )
            clustering_accounting = empty_usage_summary().to_dict()
            human_accounting = empty_usage_summary().to_dict()
        else:
            state = session.state
            clustering_accounting = merge_usage_summaries(
                (
                    getattr(session, "seed_clustering_agent_accounting", None),
                    session.clustering_agent_usage_collector.snapshot(
                        pricer=session.pricing_client
                    ).to_dict(),
                )
            ).to_dict()
            human_accounting = merge_usage_summaries(
                (
                    getattr(session, "seed_human_agent_accounting", None),
                    session.human_agent_usage_collector.snapshot(
                        pricer=session.pricing_client
                    ).to_dict(),
                )
            ).to_dict()
        return SimulationResult(
            state=state,
            clustering_agent_accounting=clustering_accounting,
            human_agent_accounting=human_accounting,
            turns_run=max(0, state.get("iteration_count", 0)),
            stopped_reason=reason,
        )

    @staticmethod
    def _build_failed_simulation_result_from_seed(
        *,
        reason: str,
        seed_result: SimulationResult,
    ) -> SimulationResult:
        return SimulationResult(
            state=copy.deepcopy(seed_result.state),
            clustering_agent_accounting=dict(seed_result.clustering_agent_accounting),
            human_agent_accounting=dict(seed_result.human_agent_accounting),
            turns_run=seed_result.turns_run,
            stopped_reason=reason,
        )

    @classmethod
    def _build_failed_simulation_result_for_attempt(
        cls,
        *,
        reason: str,
        session: SessionContext | None,
        seed_result: SimulationResult | None,
    ) -> SimulationResult:
        if session is None and seed_result is not None:
            return cls._build_failed_simulation_result_from_seed(
                reason=reason,
                seed_result=seed_result,
            )
        return cls._build_failed_simulation_result(
            reason=reason,
            session=session,
        )

    @staticmethod
    def _metadata_string(metadata: dict[str, object], key: str) -> str | None:
        value = metadata.get(key)
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        if not normalized:
            return None
        return normalized

    @staticmethod
    def _metadata_int(metadata: dict[str, object], key: str) -> int | None:
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value

    @staticmethod
    def _branch_pre_human_agent_config(config: BatchRunConfig) -> HumanAgentConfig | None:
        pre_branch_value = config.metadata.get("branch_pre_vary_value")
        if not isinstance(pre_branch_value, HumanAgentConfig):
            return None
        return pre_branch_value

    @staticmethod
    def _branch_shared_turns(config: BatchRunConfig) -> int | None:
        shared_turns = config.metadata.get("branch_shared_turns")
        if isinstance(shared_turns, bool) or not isinstance(shared_turns, int):
            return None
        return shared_turns

    def _effective_human_agent_config(
        self,
        config: BatchRunConfig,
    ) -> HumanAgentConfig:
        effective = config.human_agent_config
        pre_branch_value = self._branch_pre_human_agent_config(config)
        if pre_branch_value is None:
            return effective

        shared_turns = self._branch_shared_turns(config)
        guidance = effective.guidance
        if guidance is None and effective.prompt_id == pre_branch_value.prompt_id:
            guidance = pre_branch_value.guidance
        persona = effective.persona if effective.persona is not None else pre_branch_value.persona
        turn_injection = effective.turn_injection
        if turn_injection is None and pre_branch_value.turn_injection is not None:
            inherited_turn_injection = pre_branch_value.turn_injection
            if shared_turns is None or inherited_turn_injection.turn_index > shared_turns:
                turn_injection = inherited_turn_injection

        return replace(
            effective,
            guidance=guidance,
            persona=persona,
            turn_injection=turn_injection,
        )

    def _materialize_result_config(self, config: BatchRunConfig) -> BatchRunConfig:
        metadata = dict(config.metadata)
        pre_branch_value = self._branch_pre_human_agent_config(config)
        if pre_branch_value is not None:
            metadata["branch_pre_human_agent_config"] = pre_branch_value.to_dict()
        return replace(
            config,
            human_agent_config=self._effective_human_agent_config(config),
            metadata=metadata,
        )

    def _create_session_for_config(
        self,
        *,
        config: BatchRunConfig,
        clustering_path: str,
        run_logger: logging.Logger,
    ) -> SessionContext:
        effective_human_agent_config = self._effective_human_agent_config(config)
        selected_topic_label = self._metadata_string(
            config.metadata,
            "topic_branch_topic_label",
        )
        selected_topic_guidance = self._metadata_string(
            config.metadata,
            "topic_branch_topic_guidance",
        )
        selected_topic_generated_at_turn = self._metadata_int(
            config.metadata,
            "topic_branch_generated_at_turn",
        )
        return create_session(
            dataset_name=config.dataset_name,
            limit=config.limit,
            random_seed=config.random_seed,
            clustering_agent_backend=config.clustering_agent_backend,
            clustering_agent_backend_config=config.clustering_agent_backend_config,
            human_agent_model_config=config.human_agent_model_config,
            human_verifier_model_config=config.human_verifier_model_config,
            human_agent_prompt_id=effective_human_agent_config.prompt_id,
            human_agent_persona=effective_human_agent_config.persona,
            human_verifier_prompt_id=config.human_verifier_prompt_id,
            max_verification_retries=config.max_verification_retries,
            topic_discovery_config=effective_human_agent_config.topic_discovery,
            turn_injection=effective_human_agent_config.turn_injection,
            selected_topic_label=selected_topic_label,
            selected_topic_guidance=selected_topic_guidance,
            selected_topic_generated_at_turn=selected_topic_generated_at_turn,
            enable_embeddings=config.enable_embeddings,
            enable_intermittent_clustering=config.enable_intermittent_clustering,
            embedding_model_config=config.embedding_model_config,
            embedding_force_recompute=config.embedding_force_recompute,
            embedding_batch_size=config.embedding_batch_size,
            clustering_save_path=clustering_path,
            logger=run_logger,
        )

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
    ) -> tuple[SimulationResult, bool]:
        for attempt in range(1, _REMOTE_PROTOCOL_MAX_ATTEMPTS + 1):
            session: SessionContext | None = None
            try:
                with self._acquire_run_slot():
                    effective_human_agent_config = self._effective_human_agent_config(config)
                    session = self._create_session_for_config(
                        config=config,
                        clustering_path=clustering_path,
                        run_logger=run_logger,
                    )
                    if seed_result is not None:
                        session.state = copy.deepcopy(seed_result.state)
                        session.seed_clustering_agent_accounting = copy.deepcopy(
                            seed_result.clustering_agent_accounting
                        )
                        session.seed_human_agent_accounting = copy.deepcopy(
                            seed_result.human_agent_accounting
                        )

                    reporter = reporter_factory()
                    simulation_result = run_human_simulation(
                        session=session,
                        guidance=effective_human_agent_config.guidance,
                        max_turns=max_turns,
                        reporter=reporter,
                        stop_on_topic_candidates=stop_on_topic_candidates,
                    )
                if seed_result is not None:
                    simulation_result.turns_run += seed_result.turns_run
                return simulation_result, True
            except Exception as exc:
                if self._is_timeout_error(exc):
                    if log_terminal_errors:
                        run_logger.exception(
                            "%s %s failed due to request timeout.",
                            run_label.capitalize(),
                            run_id,
                        )
                    return (
                        self._build_failed_simulation_result_for_attempt(
                            reason="run_error_timeout",
                            session=session,
                            seed_result=seed_result,
                        ),
                        False,
                    )
                is_retryable = self._is_retryable_remote_protocol_error(exc)
                should_retry = is_retryable and attempt < _REMOTE_PROTOCOL_MAX_ATTEMPTS
                if should_retry:
                    backoff_seconds = _REMOTE_PROTOCOL_INITIAL_BACKOFF_SECONDS * (
                        2 ** (attempt - 1)
                    )
                    if log_retry_warnings:
                        run_logger.warning(
                            "Retrying %s %s after RemoteProtocolError "
                            "(attempt %s/%s, backoff %.1fs)",
                            run_label,
                            run_id,
                            attempt + 1,
                            _REMOTE_PROTOCOL_MAX_ATTEMPTS,
                            backoff_seconds,
                            exc_info=exc,
                        )
                    time.sleep(backoff_seconds)
                    continue

                if is_retryable:
                    if log_terminal_errors:
                        run_logger.exception(
                            "%s %s failed after %s attempts due to RemoteProtocolError.",
                            run_label.capitalize(),
                            run_id,
                            _REMOTE_PROTOCOL_MAX_ATTEMPTS,
                        )
                    return (
                        self._build_failed_simulation_result_for_attempt(
                            reason="run_error_remote_protocol",
                            session=session,
                            seed_result=seed_result,
                        ),
                        False,
                    )

                if log_terminal_errors:
                    run_logger.exception(
                        "%s %s failed due to non-retryable exception.",
                        run_label.capitalize(),
                        run_id,
                    )
                return (
                    self._build_failed_simulation_result_for_attempt(
                        reason="run_error_exception",
                        session=session,
                        seed_result=seed_result,
                    ),
                    False,
                )

        return (
            self._build_failed_simulation_result_for_attempt(
                reason="run_error_exception",
                session=None,
                seed_result=seed_result,
            ),
            False,
        )

    def _run_single(
        self,
        config: BatchRunConfig,
        run_index: int,
        config_slug: str,
    ) -> tuple[str, SimulationResult]:
        run_id = self._build_run_id(run_index)
        clustering_path, log_path = self._build_run_paths(config_slug, run_id)
        run_logger = self._create_run_logger(config_slug, run_id, log_path)

        try:
            simulation_result, was_successful = self._run_session_with_retries(
                config=config,
                clustering_path=clustering_path,
                run_logger=run_logger,
                max_turns=config.max_turns,
                reporter_factory=lambda: self.reporter_factory(config),
                run_id=run_id,
                run_label="run",
            )
            if was_successful:
                self._write_conversation_trace_artifact(
                    clustering_path=clustering_path,
                    simulation_result=simulation_result,
                )
                self._write_human_thought_trace_artifact(
                    clustering_path=clustering_path,
                    simulation_result=simulation_result,
                )
                self._write_judge_clustering_trace_artifact(
                    clustering_path=clustering_path,
                    simulation_result=simulation_result,
                )
                self._write_clustering_pov_history_artifact(
                    clustering_path=clustering_path,
                    simulation_result=simulation_result,
                )
                self._write_topic_candidates_artifact(
                    clustering_path=clustering_path,
                    simulation_result=simulation_result,
                )
            return run_id, simulation_result
        finally:
            self._close_logger(run_logger)

    @staticmethod
    def _build_embedding_preparation_specs(
        configs: list[BatchRunConfig],
    ) -> list[_EmbeddingPreparationSpec]:
        preparations_by_cache_path: dict[str, _EmbeddingPreparationSpec] = {}
        for config in configs:
            if not config.enable_embeddings:
                continue

            model_config = config.embedding_model_config
            if model_config is None:
                raise ValueError("embedding_model_config must be set when embeddings are enabled")

            cache_path = build_embeddings_cache_path(
                dataset_name=config.dataset_name,
                model_config=model_config,
            )
            preparation = preparations_by_cache_path.get(cache_path)
            if preparation is None:
                preparation = _EmbeddingPreparationSpec(
                    dataset_name=config.dataset_name,
                    model_config=model_config,
                    batch_size=config.embedding_batch_size,
                )
                preparations_by_cache_path[cache_path] = preparation
            else:
                preparation.batch_size = max(preparation.batch_size, config.embedding_batch_size)

            preparation.force_recompute = (
                preparation.force_recompute or config.embedding_force_recompute
            )
            loader = build_data_loader(
                config.dataset_name,
                limit=config.limit,
                random_seed=config.random_seed,
                enable_embeddings=False,
                embedding_model_config=model_config,
                embedding_force_recompute=False,
                embedding_batch_size=config.embedding_batch_size,
            )
            for item_id, item_text in loader.get_unique_item_texts_by_id().items():
                preparation.item_texts_by_id.setdefault(item_id, item_text)

        return list(preparations_by_cache_path.values())

    def _prepare_embeddings_for_configs(self, configs: list[BatchRunConfig]) -> None:
        for preparation in self._build_embedding_preparation_specs(configs):
            ensure_embeddings_for_items(
                item_texts_by_id=preparation.item_texts_by_id,
                dataset_name=preparation.dataset_name,
                model_config=preparation.model_config,
                force_recompute=preparation.force_recompute,
                batch_size=preparation.batch_size,
            )

    def run_batch_parallel(
        self,
        configs: list[BatchRunConfig],
        max_workers: int | None = None,
    ) -> list[BatchRunResult]:
        self._prepare_embeddings_for_configs(configs)
        worker_configs = [
            replace(config, embedding_force_recompute=False) if config.enable_embeddings else config
            for config in configs
        ]
        simulation_results_by_config: list[list[tuple[int, str, SimulationResult]]] = [
            [] for _ in range(len(configs))
        ]
        realized_topic_runs: list[RealizedBatchRun] = []
        config_slugs = [
            self._build_config_slug(config_index=config_index)
            for config_index, _ in enumerate(configs)
        ]
        total_runs = sum(config.num_runs for config in configs)
        progress_reporter: BatchProgressReporter | None = None
        if self.batch_progress_enabled and total_runs > 0:
            progress_reporter = BatchProgressReporter(
                total_runs=total_runs,
                output_dir=self.output_dir,
            )

        branch_config_indexes_by_group: dict[tuple[str, str, int], list[int]] = {}
        auto_topic_config_indexes: list[int] = []
        non_branch_config_indexes: list[int] = []
        for config_index, config in enumerate(worker_configs):
            branch_group_key = extract_branch_group_key(config.metadata)
            if branch_group_key is not None:
                branch_config_indexes_by_group.setdefault(branch_group_key, []).append(config_index)
            elif config_uses_auto_topic_branching(config):
                auto_topic_config_indexes.append(config_index)
            else:
                non_branch_config_indexes.append(config_index)

        resolved_max_workers = self._resolve_max_workers(max_workers)
        previous_run_slot_semaphore = self._run_slot_semaphore
        self._run_slot_semaphore = threading.BoundedSemaphore(resolved_max_workers)
        try:
            with ThreadPoolExecutor(max_workers=resolved_max_workers) as executor:
                futures: dict[Future[Any], tuple[str, Any, int]] = {}
                auto_topic_aggregate_states: dict[tuple[int, int], _AutoTopicAggregateState] = {}
                branch_group_aggregate_states: dict[
                    tuple[tuple[int, ...], int], _BranchGroupAggregateState
                ] = {}

                def record_config_result(
                    *,
                    config_index: int,
                    run_index: int,
                    run_id: str,
                    simulation_result: SimulationResult,
                ) -> None:
                    simulation_results_by_config[config_index].append(
                        (run_index, run_id, simulation_result)
                    )
                    if progress_reporter is not None:
                        progress_reporter.record_run(
                            config_slug=config_slugs[config_index],
                            run_id=run_id,
                            simulation_result=simulation_result,
                        )

                def record_auto_topic_results(realized_runs: list[RealizedBatchRun]) -> None:
                    if progress_reporter is not None and len(realized_runs) > 1:
                        progress_reporter.add_expected_runs(len(realized_runs) - 1)
                    realized_topic_runs.extend(realized_runs)
                    if progress_reporter is not None:
                        for realized_run in realized_runs:
                            progress_reporter.record_run(
                                config_slug=realized_run.config_slug,
                                run_id=realized_run.run_id,
                                simulation_result=realized_run.simulation_result,
                            )

                def submit_future(
                    *,
                    future: Future[Any],
                    task_type: str,
                    task_value: Any,
                    run_index: int,
                    pending_futures: set[Future[Any]] | None = None,
                ) -> None:
                    futures[future] = (task_type, task_value, run_index)
                    if pending_futures is not None:
                        pending_futures.add(future)

                for config_index in non_branch_config_indexes:
                    config = worker_configs[config_index]
                    for run_index in range(config.num_runs):
                        submit_future(
                            future=executor.submit(
                                self._run_single,
                                config,
                                run_index,
                                config_slugs[config_index],
                            ),
                            task_type="single",
                            task_value=config_index,
                            run_index=run_index,
                        )

                for config_index in auto_topic_config_indexes:
                    source_config = configs[config_index]
                    execution_config = worker_configs[config_index]
                    for run_index in range(source_config.num_runs):
                        submit_future(
                            future=executor.submit(
                                self._branching.plan_topic_branching_single_run,
                                source_config=source_config,
                                execution_config=execution_config,
                                config_index=config_index,
                                config_slug=config_slugs[config_index],
                                run_index=run_index,
                            ),
                            task_type="auto_topic_shared",
                            task_value=config_index,
                            run_index=run_index,
                        )

                for config_indexes in branch_config_indexes_by_group.values():
                    first_config = worker_configs[config_indexes[0]]
                    shared_config_slug = config_slugs[config_indexes[0]]
                    branch_config_slugs = [
                        config_slugs[config_index] for config_index in config_indexes
                    ]
                    self._branching.write_branch_mapping_artifact(
                        shared_config_slug=shared_config_slug,
                        branch_config_slugs=branch_config_slugs,
                    )
                    for run_index in range(first_config.num_runs):
                        submit_future(
                            future=executor.submit(
                                self._branching.plan_branch_group_single_run,
                                configs=worker_configs,
                                config_slugs=config_slugs,
                                config_indexes=config_indexes,
                                run_index=run_index,
                            ),
                            task_type="branch_group_shared",
                            task_value=tuple(config_indexes),
                            run_index=run_index,
                        )

                if progress_reporter is not None:
                    progress_reporter.start()

                pending_futures = set(futures)
                while pending_futures:
                    if progress_reporter is None:
                        done_futures, pending_futures = wait(
                            pending_futures,
                            return_when=FIRST_COMPLETED,
                        )
                    else:
                        done_futures, pending_futures = wait(
                            pending_futures,
                            timeout=_REPORTER_TIMEOUT_SECONDS,
                            return_when=FIRST_COMPLETED,
                        )
                    if progress_reporter is not None:
                        progress_reporter.tick()
                    for future in done_futures:
                        task_type, task_value, run_index = futures.pop(future)
                        if task_type == "single":
                            config_index = cast(int, task_value)
                            try:
                                run_id, simulation_result = future.result()
                            except Exception:
                                run_id = f"{run_index + 1:03d}-failed"
                                simulation_result = self._build_failed_simulation_result(
                                    reason="run_error_exception",
                                    session=None,
                                )
                            record_config_result(
                                config_index=config_index,
                                run_index=run_index,
                                run_id=run_id,
                                simulation_result=simulation_result,
                            )
                            continue

                        if task_type == "auto_topic_shared":
                            config_index = cast(int, task_value)
                            try:
                                auto_topic_plan = cast(
                                    AutoTopicCompletedPayload | AutoTopicSharedPayload,
                                    future.result(),
                                )
                            except Exception:
                                fallback_config, fallback_slug = (
                                    self._branching.build_unbranched_topic_result_config(
                                        source_config=configs[config_index],
                                        source_config_slug=config_slugs[config_index],
                                        shared_run_id=f"{run_index + 1:03d}-failed",
                                        run_index=run_index,
                                        outcome="shared_run_terminal",
                                    )
                                )
                                record_auto_topic_results(
                                    [
                                        RealizedBatchRun(
                                            source_config_index=config_index,
                                            branch_index=0,
                                            config=fallback_config,
                                            config_slug=fallback_slug,
                                            run_index=run_index,
                                            run_id=f"{run_index + 1:03d}-failed",
                                            simulation_result=self._build_failed_simulation_result(
                                                reason="run_error_exception",
                                                session=None,
                                            ),
                                        )
                                    ]
                                )
                                continue

                            if isinstance(auto_topic_plan, AutoTopicCompletedPayload):
                                try:
                                    record_auto_topic_results(auto_topic_plan.realized_runs)
                                finally:
                                    self._branching.cleanup_shared_clustering_artifact(
                                        auto_topic_plan.shared_clustering_path
                                    )
                                continue

                            parent_key = (
                                auto_topic_plan.source_config_index,
                                auto_topic_plan.run_index,
                            )
                            state = _AutoTopicAggregateState(
                                payload=auto_topic_plan,
                                pending_children=len(auto_topic_plan.branch_specs),
                            )
                            auto_topic_aggregate_states[parent_key] = state
                            if state.pending_children == 0:
                                try:
                                    record_auto_topic_results(
                                        self._branching.finalize_auto_topic_results(
                                            payload=state.payload,
                                            branch_results_by_index=state.branch_results_by_index,
                                            branch_records_by_index=state.branch_records_by_index,
                                        )
                                    )
                                finally:
                                    self._branching.cleanup_shared_clustering_artifact(
                                        state.payload.shared_clustering_path
                                    )
                                    del auto_topic_aggregate_states[parent_key]
                                continue

                            for branch_spec in auto_topic_plan.branch_specs:
                                submit_future(
                                    future=executor.submit(
                                        self._branching.run_single_from_shared_prefix,
                                        config=branch_spec.execution_branch_config,
                                        config_slug=branch_spec.realized_slug,
                                        run_index=run_index,
                                        shared_result=auto_topic_plan.shared_result,
                                        shared_clustering_path=auto_topic_plan.shared_clustering_path,
                                    ),
                                    task_type="auto_topic_branch",
                                    task_value=_AutoTopicBranchTask(
                                        parent_key=parent_key,
                                        branch_spec=branch_spec,
                                    ),
                                    run_index=run_index,
                                    pending_futures=pending_futures,
                                )
                            continue

                        if task_type == "auto_topic_branch":
                            branch_task = cast(_AutoTopicBranchTask, task_value)
                            state = auto_topic_aggregate_states[branch_task.parent_key]
                            branch_spec = branch_task.branch_spec
                            try:
                                run_id, simulation_result = cast(
                                    tuple[str, SimulationResult],
                                    future.result(),
                                )
                                realized_run = RealizedBatchRun(
                                    source_config_index=state.payload.source_config_index,
                                    branch_index=branch_spec.branch_index,
                                    config=branch_spec.realized_config,
                                    config_slug=branch_spec.realized_slug,
                                    run_index=state.payload.run_index,
                                    run_id=run_id,
                                    simulation_result=simulation_result,
                                )
                            except Exception:
                                realized_run = self._branching.build_failed_topic_branch_result(
                                    source_config_index=state.payload.source_config_index,
                                    branch_index=branch_spec.branch_index,
                                    realized_config=branch_spec.realized_config,
                                    realized_slug=branch_spec.realized_slug,
                                    run_index=state.payload.run_index,
                                )
                            state.branch_results_by_index[branch_spec.branch_index] = realized_run
                            state.branch_records_by_index[branch_spec.branch_index] = (
                                self._branching.build_topic_branch_record(
                                    config_slug=branch_spec.realized_slug,
                                    run_id=realized_run.run_id,
                                    topic=branch_spec.topic,
                                    generated_at_turn=state.payload.generated_at_turn,
                                )
                            )
                            state.pending_children -= 1
                            if state.pending_children == 0:
                                try:
                                    record_auto_topic_results(
                                        self._branching.finalize_auto_topic_results(
                                            payload=state.payload,
                                            branch_results_by_index=state.branch_results_by_index,
                                            branch_records_by_index=state.branch_records_by_index,
                                        )
                                    )
                                finally:
                                    self._branching.cleanup_shared_clustering_artifact(
                                        state.payload.shared_clustering_path
                                    )
                                    del auto_topic_aggregate_states[branch_task.parent_key]
                            continue

                        if task_type == "branch_group_shared":
                            config_indexes = list(cast(tuple[int, ...], task_value))
                            parent_key = (tuple(config_indexes), run_index)
                            try:
                                shared_payload = cast(
                                    BranchGroupSharedPayload,
                                    future.result(),
                                )
                            except Exception:
                                for config_index in config_indexes:
                                    record_config_result(
                                        config_index=config_index,
                                        run_index=run_index,
                                        run_id=f"{run_index + 1:03d}-failed",
                                        simulation_result=self._build_failed_simulation_result(
                                            reason="run_error_exception",
                                            session=None,
                                        ),
                                    )
                                continue

                            state = _BranchGroupAggregateState(
                                payload=shared_payload,
                                pending_children=len(shared_payload.child_specs),
                            )
                            branch_group_aggregate_states[parent_key] = state
                            for child_spec in shared_payload.child_specs:
                                submit_future(
                                    future=executor.submit(
                                        self._branching.run_single_from_shared_prefix,
                                        config=child_spec.config,
                                        config_slug=child_spec.config_slug,
                                        run_index=shared_payload.run_index,
                                        shared_result=shared_payload.shared_result,
                                        shared_clustering_path=shared_payload.shared_clustering_path,
                                    ),
                                    task_type="branch_group_branch",
                                    task_value=_BranchGroupBranchTask(
                                        parent_key=parent_key,
                                        child_spec=child_spec,
                                    ),
                                    run_index=run_index,
                                    pending_futures=pending_futures,
                                )
                            continue

                        branch_task = cast(_BranchGroupBranchTask, task_value)
                        state = branch_group_aggregate_states[branch_task.parent_key]
                        child_spec = branch_task.child_spec
                        try:
                            run_id, simulation_result = cast(
                                tuple[str, SimulationResult],
                                future.result(),
                            )
                        except Exception:
                            run_id = f"{run_index + 1:03d}-failed"
                            simulation_result = self._build_failed_simulation_result_from_seed(
                                reason="run_error_exception",
                                seed_result=state.payload.shared_result,
                            )
                        state.branch_results_by_config_index[child_spec.config_index] = (
                            run_id,
                            simulation_result,
                        )
                        state.pending_children -= 1
                        if state.pending_children == 0:
                            try:
                                branch_results = self._branching.finalize_branch_group_results(
                                    payload=state.payload,
                                    branch_results_by_config_index=state.branch_results_by_config_index,
                                )
                                for (
                                    config_index,
                                    child_run_id,
                                    child_simulation_result,
                                ) in branch_results:
                                    record_config_result(
                                        config_index=config_index,
                                        run_index=run_index,
                                        run_id=child_run_id,
                                        simulation_result=child_simulation_result,
                                    )
                            finally:
                                self._branching.cleanup_shared_clustering_artifact(
                                    state.payload.shared_clustering_path
                                )
                                del branch_group_aggregate_states[branch_task.parent_key]
        finally:
            self._run_slot_semaphore = previous_run_slot_semaphore

        auto_topic_results_by_source_config: dict[int, list[BatchRunResult]] = {}
        grouped_realized_runs_by_source_config: dict[int, dict[str, list[RealizedBatchRun]]] = {}
        realized_configs_by_source_and_slug: dict[tuple[int, str], BatchRunConfig] = {}
        for realized_run in sorted(
            realized_topic_runs,
            key=lambda item: (
                item.source_config_index,
                item.run_index,
                item.branch_index,
            ),
        ):
            source_group = grouped_realized_runs_by_source_config.setdefault(
                realized_run.source_config_index,
                {},
            )
            config_key = (realized_run.source_config_index, realized_run.config_slug)
            prior_config = realized_configs_by_source_and_slug.get(config_key)
            if prior_config is None:
                realized_configs_by_source_and_slug[config_key] = realized_run.config
            elif prior_config != realized_run.config:
                raise ValueError(
                    "Auto-topic realized config mismatch for "
                    f"source_config_index={realized_run.source_config_index}, "
                    f"config_slug='{realized_run.config_slug}'."
                )
            source_group.setdefault(realized_run.config_slug, []).append(realized_run)

        for source_config_index, runs_by_slug in grouped_realized_runs_by_source_config.items():
            auto_topic_results_by_source_config[source_config_index] = []
            for config_slug, grouped_runs in runs_by_slug.items():
                ordered_runs = sorted(
                    grouped_runs, key=lambda item: (item.run_index, item.branch_index)
                )
                auto_topic_results_by_source_config[source_config_index].append(
                    BatchRunResult(
                        config=self._materialize_result_config(
                            replace(ordered_runs[0].config, num_runs=len(ordered_runs))
                        ),
                        simulation_results=[
                            realized_run.simulation_result for realized_run in ordered_runs
                        ],
                        run_ids=[realized_run.run_id for realized_run in ordered_runs],
                        config_slug=config_slug,
                    )
                )

        auto_topic_config_index_set = set(auto_topic_config_indexes)
        results: list[BatchRunResult] = []
        for config_index, config in enumerate(configs):
            if config_index in auto_topic_config_index_set:
                results.extend(auto_topic_results_by_source_config.get(config_index, []))
                continue
            ordered_runs = sorted(
                simulation_results_by_config[config_index], key=lambda item: item[0]
            )
            run_ids = [run_id for _, run_id, _ in ordered_runs]
            simulation_results = [simulation_result for _, _, simulation_result in ordered_runs]
            results.append(
                BatchRunResult(
                    config=self._materialize_result_config(config),
                    simulation_results=simulation_results,
                    run_ids=run_ids,
                    config_slug=config_slugs[config_index],
                )
            )

        if progress_reporter is not None:
            progress_reporter.finish(results)

        return results
