"""Session orchestration and SSE event generation for the standalone demo API."""

from __future__ import annotations

import json
import logging
import queue
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from fastapi.encoders import jsonable_encoder
from langchain_core.messages import HumanMessage

from conversational_clustering.accounting import merge_usage_summaries
from conversational_clustering.clustering_agent.backends import (
    default_clustering_agent_backend_config,
)
from conversational_clustering.clustering_agent.state import AgentState
from conversational_clustering.simulation import create_session as build_interactive_session
from conversational_clustering.simulation.runner import SessionContext, run_agent_turn

from .config import DemoSettings, resolve_demo_backend_config
from .schemas import (
    ProposalClusterModel,
    ProposalSnapshotModel,
    SessionSnapshotModel,
    TokenTotalsModel,
    ToolTraceMetadataModel,
    TranscriptEntryModel,
)


def build_demo_session_context(settings: DemoSettings) -> SessionContext:
    backend_config = resolve_demo_backend_config(
        default_clustering_agent_backend_config(settings.clustering_agent_backend),
        settings=settings,
    )
    return build_interactive_session(
        dataset_name=settings.dataset_name,
        clustering_agent_backend=settings.clustering_agent_backend,
        clustering_agent_backend_config=backend_config,
        enable_intermittent_clustering=False,
    )


def _normalize_token_totals(raw_totals: dict[str, int]) -> TokenTotalsModel:
    input_tokens = max(0, int(raw_totals.get("input_tokens", 0)))
    output_tokens = max(0, int(raw_totals.get("output_tokens", 0)))
    return TokenTotalsModel(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _session_token_totals(context: SessionContext) -> TokenTotalsModel:
    accounting = merge_usage_summaries(
        (
            context.seed_clustering_agent_accounting,
            context.clustering_agent_usage_collector.snapshot(pricer=None).to_dict(),
        )
    ).to_dict()
    return _normalize_token_totals(
        {
            "input_tokens": _coerce_nonnegative_int(accounting.get("input_tokens", 0)),
            "output_tokens": _coerce_nonnegative_int(accounting.get("output_tokens", 0)),
        }
    )


def _coerce_nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return max(0, int(value))


def _proposal_index(proposal_id: str) -> int | None:
    prefix, _, suffix = proposal_id.partition("_")
    if prefix != "proposal":
        return None
    try:
        return int(suffix)
    except ValueError:
        return None


def _parse_proposal_snapshot(
    proposal_id: str, raw_proposal: object
) -> ProposalSnapshotModel | None:
    if not isinstance(raw_proposal, dict):
        return None
    proposal = cast(dict[str, object], raw_proposal)

    embedding_instruction = proposal.get("embedding_instruction")
    num_clusters = proposal.get("num_clusters")
    raw_clusters = proposal.get("clusters")
    if not isinstance(embedding_instruction, str) or not embedding_instruction.strip():
        return None
    if isinstance(num_clusters, bool) or not isinstance(num_clusters, int) or num_clusters < 1:
        return None
    if not isinstance(raw_clusters, list):
        return None

    clusters: list[ProposalClusterModel] = []
    for raw_cluster in raw_clusters:
        if not isinstance(raw_cluster, dict):
            continue
        cluster = cast(dict[str, object], raw_cluster)
        name = cluster.get("name")
        size = cluster.get("size")
        summary = cluster.get("summary")
        raw_representative_points = cluster.get("representative_points", [])
        if not isinstance(name, str) or not name.strip():
            continue
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            continue
        if not isinstance(summary, str):
            continue
        if not isinstance(raw_representative_points, list):
            raw_representative_points = []
        representative_points = [
            point for point in raw_representative_points if isinstance(point, str) and point.strip()
        ]
        clusters.append(
            ProposalClusterModel(
                name=name.strip(),
                size=size,
                summary=" ".join(summary.split()),
                representative_points=representative_points,
            )
        )

    return ProposalSnapshotModel(
        proposal_id=proposal_id,
        embedding_instruction=" ".join(embedding_instruction.split()),
        num_clusters=num_clusters,
        clusters=clusters,
    )


def _extract_proposal_history(state: Mapping[str, object]) -> list[ProposalSnapshotModel]:
    raw_runtime_state = state.get("instruction_kmeans_runtime_state")
    if not isinstance(raw_runtime_state, dict):
        return []
    runtime_state = cast(dict[str, object], raw_runtime_state)

    raw_proposals_by_id = runtime_state.get("proposals_by_id")
    if not isinstance(raw_proposals_by_id, dict) or not raw_proposals_by_id:
        return []
    proposals_by_id = cast(dict[str, object], raw_proposals_by_id)

    parsed_proposals: list[tuple[int, ProposalSnapshotModel]] = []
    for raw_proposal_id, raw_proposal in proposals_by_id.items():
        if not isinstance(raw_proposal_id, str):
            continue
        proposal_index = _proposal_index(raw_proposal_id)
        if proposal_index is None:
            continue
        proposal = _parse_proposal_snapshot(raw_proposal_id, raw_proposal)
        if proposal is None:
            continue
        parsed_proposals.append((proposal_index, proposal))

    parsed_proposals.sort(key=lambda item: item[0])
    return [proposal for _, proposal in parsed_proposals]


def _extract_latest_proposal_snapshot(
    state: Mapping[str, object],
) -> ProposalSnapshotModel | None:
    proposal_history = _extract_proposal_history(state)
    if not proposal_history:
        return None
    return proposal_history[-1]


def _humanize_tool_name(tool_name: str) -> str:
    if tool_name == "append_clustering_povs":
        return "Append Clustering POVs"
    normalized = " ".join(tool_name.strip().split("_"))
    if not normalized:
        return "Tool"
    return normalized.title()


def _extract_tool_trace_details(tool_name: str, content: object) -> list[str]:
    if not isinstance(content, str):
        return []
    stripped = content.strip()
    if not stripped:
        return []
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []

    details: list[str] = []
    if tool_name == "sample_dataset":
        raw_items = payload.get("items")
        raw_total_items = payload.get("total_items")
        raw_start_index = payload.get("start_index")
        raw_end_index = payload.get("end_index_exclusive")
        if isinstance(raw_items, list):
            details.append(f"Saw {len(raw_items)} datapoints")
        if (
            isinstance(raw_start_index, int)
            and not isinstance(raw_start_index, bool)
            and isinstance(raw_end_index, int)
            and not isinstance(raw_end_index, bool)
            and isinstance(raw_total_items, int)
            and not isinstance(raw_total_items, bool)
        ):
            details.append(f"Window {raw_start_index + 1}-{raw_end_index} of {raw_total_items}")
        return details

    if tool_name == "propose_clustering":
        raw_num_clusters = payload.get("num_clusters")
        raw_proposal_id = payload.get("proposal_id")
        if isinstance(raw_num_clusters, int) and not isinstance(raw_num_clusters, bool):
            details.append(f"{raw_num_clusters} clusters proposed")
        if isinstance(raw_proposal_id, str) and raw_proposal_id.strip():
            details.append(raw_proposal_id.strip())
        return details

    if tool_name == "save_clustering":
        raw_num_clusters = payload.get("num_clusters")
        if isinstance(raw_num_clusters, int) and not isinstance(raw_num_clusters, bool):
            details.append(f"Saved {raw_num_clusters} clusters")
        raw_save_message = payload.get("save_message")
        if isinstance(raw_save_message, str) and raw_save_message.strip():
            details.append("Saved clustering artifact")
        return details

    if tool_name == "append_clustering_povs":
        raw_num_added = payload.get("num_added")
        raw_added_povs = payload.get("added_povs")
        if isinstance(raw_num_added, int) and not isinstance(raw_num_added, bool):
            suffix = "POV" if raw_num_added == 1 else "POVs"
            details.append(f"{raw_num_added} {suffix} added")
            return details
        if isinstance(raw_added_povs, list):
            num_added = len([pov for pov in raw_added_povs if isinstance(pov, str) and pov.strip()])
            suffix = "POV" if num_added == 1 else "POVs"
            details.append(f"{num_added} {suffix} added")
        return details

    return []


@dataclass(frozen=True)
class SessionEvent:
    id: int
    type: str
    data: dict[str, object]


@dataclass
class DemoSession:
    session_id: str
    context: SessionContext
    transcript: list[TranscriptEntryModel] = field(default_factory=list)
    latest_proposal: ProposalSnapshotModel | None = None
    proposal_history: list[ProposalSnapshotModel] = field(default_factory=list)
    token_totals: TokenTotalsModel = field(default_factory=TokenTotalsModel)
    is_turn_in_progress: bool = False
    next_transcript_index: int = 1
    next_event_id: int = 1
    last_emitted_proposal_id: str | None = None
    current_turn_tool_call_count: int = 0
    current_turn_message_start_index: int = 0
    current_turn_transcript_start_index: int = 0
    _subscribers: dict[int, queue.Queue[SessionEvent | None]] = field(default_factory=dict)
    _next_subscriber_id: int = 1
    _lock: threading.RLock = field(default_factory=threading.RLock)
    turn_lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> SessionSnapshotModel:
        with self._lock:
            transcript = [entry.model_copy(deep=True) for entry in self.transcript]
            latest_proposal = (
                self.latest_proposal.model_copy(deep=True)
                if self.latest_proposal is not None
                else None
            )
            proposal_history = [
                proposal.model_copy(deep=True) for proposal in self.proposal_history
            ]
            token_totals = self.token_totals.model_copy(deep=True)
            return SessionSnapshotModel(
                session_id=self.session_id,
                transcript=transcript,
                latest_proposal=latest_proposal,
                proposal_history=proposal_history,
                is_turn_in_progress=self.is_turn_in_progress,
                token_totals=token_totals,
            )

    def add_subscriber(self) -> tuple[int, queue.Queue[SessionEvent | None]]:
        with self._lock:
            subscriber_id = self._next_subscriber_id
            self._next_subscriber_id += 1
            subscriber_queue: queue.Queue[SessionEvent | None] = queue.Queue()
            self._subscribers[subscriber_id] = subscriber_queue
            return subscriber_id, subscriber_queue

    def remove_subscriber(self, subscriber_id: int) -> None:
        with self._lock:
            subscriber_queue = self._subscribers.pop(subscriber_id, None)
        if subscriber_queue is not None:
            subscriber_queue.put(None)

    def close(self) -> None:
        with self._lock:
            subscribers = list(self._subscribers.values())
            self._subscribers.clear()
        for subscriber in subscribers:
            subscriber.put(None)

    def emit_event(self, event_type: str, data: dict[str, object]) -> SessionEvent:
        with self._lock:
            event = SessionEvent(id=self.next_event_id, type=event_type, data=data)
            self.next_event_id += 1
            subscribers = list(self._subscribers.values())
        for subscriber in subscribers:
            subscriber.put(event)
        return event

    def _append_entry_locked(
        self,
        *,
        role: str,
        content: str,
        status: str,
        cumulative_token_total: int | None = None,
        metadata: ToolTraceMetadataModel | None = None,
    ) -> TranscriptEntryModel:
        entry = TranscriptEntryModel(
            id=f"entry_{self.next_transcript_index}",
            role=cast(Any, role),
            content=content,
            status=cast(Any, status),
            cumulative_token_total=cumulative_token_total,
            metadata=metadata.model_copy(deep=True) if metadata is not None else None,
        )
        self.next_transcript_index += 1
        self.transcript.append(entry)
        return entry

    def add_user_message(self, content: str) -> TranscriptEntryModel:
        with self._lock:
            entry = self._append_entry_locked(role="user", content=content, status="complete")
        self.emit_event("user_message", {"entry": jsonable_encoder(entry)})
        return entry

    def add_system_message(
        self,
        content: str,
        *,
        status: str = "complete",
        metadata: ToolTraceMetadataModel | None = None,
    ) -> TranscriptEntryModel:
        with self._lock:
            entry = self._append_entry_locked(
                role="system",
                content=content,
                status=status,
                metadata=metadata,
            )
        self.emit_event("system_message", {"entry": jsonable_encoder(entry)})
        return entry

    def begin_turn(self) -> None:
        with self._lock:
            self.is_turn_in_progress = True
            self.current_turn_tool_call_count = 0
            self.current_turn_message_start_index = len(self.context.state["messages"])
            self.current_turn_transcript_start_index = len(self.transcript)
        self.emit_event("turn_started", {"session_id": self.session_id})

    def finish_turn(self) -> None:
        snapshot = None
        with self._lock:
            self.is_turn_in_progress = False
            snapshot = self.snapshot()
        self.emit_event("turn_finished", {"session": jsonable_encoder(snapshot)})

    def stream_agent_chunk(self, content: str, *, is_start: bool) -> None:
        with self._lock:
            if is_start or not self.transcript or self.transcript[-1].role != "assistant":
                entry = self._append_entry_locked(
                    role="assistant",
                    content=content,
                    status="streaming",
                )
            else:
                entry = self.transcript[-1]
                entry.content += content
                entry.status = "streaming"
        self.emit_event(
            "agent_chunk",
            {
                "entry_id": entry.id,
                "content": content,
                "is_start": is_start,
            },
        )

    def finalize_agent_message(self, content: str, *, streamed: bool) -> TranscriptEntryModel:
        with self._lock:
            if streamed and self.transcript and self.transcript[-1].role == "assistant":
                entry = self.transcript[-1]
                entry.content = content
                entry.status = "complete"
            else:
                entry = self._append_entry_locked(
                    role="assistant",
                    content=content,
                    status="complete",
                )
        self.emit_event(
            "agent_message",
            {
                "entry": jsonable_encoder(entry),
                "streamed": streamed,
            },
        )
        return entry

    def record_tool_call(self, tool_name: str) -> None:
        with self._lock:
            self.current_turn_tool_call_count += 1
            metadata = ToolTraceMetadataModel(
                tool_name=tool_name,
                display_name=_humanize_tool_name(tool_name),
                step_index=self.current_turn_tool_call_count,
                stage="running",
            )
        self.add_system_message(metadata.display_name, metadata=metadata)
        self.emit_event(
            "tool_call",
            {
                "tool_name": tool_name,
                "display_name": metadata.display_name,
                "step_index": metadata.step_index,
            },
        )

    def record_tool_error(self, tool_name: str) -> None:
        with self._lock:
            for entry in reversed(self.transcript):
                metadata = entry.metadata
                if (
                    entry.role == "system"
                    and metadata is not None
                    and metadata.kind == "tool_trace"
                    and metadata.tool_name == tool_name
                    and metadata.stage == "running"
                ):
                    metadata.stage = "failed"
                    entry.status = "error"
                    updated_entry = entry.model_copy(deep=True)
                    break
            else:
                step_index = max(1, self.current_turn_tool_call_count)
                metadata = ToolTraceMetadataModel(
                    tool_name=tool_name,
                    display_name=_humanize_tool_name(tool_name),
                    step_index=step_index,
                    stage="failed",
                )
                updated_entry = self._append_entry_locked(
                    role="system",
                    content=metadata.display_name,
                    status="error",
                    metadata=metadata,
                ).model_copy(deep=True)
        self.emit_event("system_message", {"entry": jsonable_encoder(updated_entry)})
        self.emit_event(
            "tool_error",
            {
                "tool_name": tool_name,
                "display_name": updated_entry.metadata.display_name
                if updated_entry.metadata is not None
                else _humanize_tool_name(tool_name),
            },
        )

    def record_tool_result(self, tool_name: str, content: str, *, is_error: bool) -> None:
        updated_entry: TranscriptEntryModel | None = None
        with self._lock:
            for entry in reversed(self.transcript):
                metadata = entry.metadata
                if (
                    entry.role == "system"
                    and metadata is not None
                    and metadata.kind == "tool_trace"
                    and metadata.tool_name == tool_name
                    and metadata.stage == "running"
                ):
                    metadata.details = _extract_tool_trace_details(tool_name, content)
                    if is_error:
                        metadata.stage = "failed"
                        entry.status = "error"
                    updated_entry = entry.model_copy(deep=True)
                    break
        if updated_entry is not None:
            self.emit_event("system_message", {"entry": jsonable_encoder(updated_entry)})

    def update_tokens(
        self, token_usage: dict[str, object], session_tokens: dict[str, object]
    ) -> None:
        session_totals = _normalize_token_totals(
            {
                "input_tokens": _coerce_nonnegative_int(session_tokens.get("input_tokens", 0)),
                "output_tokens": _coerce_nonnegative_int(session_tokens.get("output_tokens", 0)),
            }
        )
        updated_entry: TranscriptEntryModel | None = None
        with self._lock:
            self.token_totals = session_totals
            for entry in reversed(self.transcript):
                if entry.role != "assistant":
                    continue
                entry.cumulative_token_total = session_totals.total_tokens
                updated_entry = entry.model_copy(deep=True)
                break
        payload: dict[str, object] = {
            "token_usage": jsonable_encoder(token_usage),
            "token_totals": jsonable_encoder(session_totals),
        }
        if updated_entry is not None:
            payload["entry"] = jsonable_encoder(updated_entry)
        self.emit_event("token_usage", payload)

    def update_latest_proposal(self, state: Mapping[str, object]) -> ProposalSnapshotModel | None:
        proposal_history = _extract_proposal_history(state)
        proposal = _extract_latest_proposal_snapshot(state)
        if proposal is None or not proposal_history:
            return None
        with self._lock:
            self.proposal_history = proposal_history
            if proposal.proposal_id == self.last_emitted_proposal_id:
                self.latest_proposal = proposal
                return proposal
            self.latest_proposal = proposal
            self.last_emitted_proposal_id = proposal.proposal_id
        self.emit_event("proposal_updated", {"proposal": jsonable_encoder(proposal)})
        return proposal


class DemoStreamingReporter:
    def __init__(self, session: DemoSession):
        self._session = session

    def on_agent_chunk(self, content: str, *, is_start: bool) -> None:
        self._session.stream_agent_chunk(content, is_start=is_start)

    def on_agent_message(self, content: str, *, streamed: bool) -> None:
        self._session.finalize_agent_message(content, streamed=streamed)

    def on_tool_call(self, tool_name: str) -> None:
        self._session.record_tool_call(tool_name)

    def on_tool_result(
        self,
        tool_name: str,
        content: str,
        *,
        is_error: bool,
        state: Mapping[str, object] | None = None,
    ) -> None:
        self._session.record_tool_result(tool_name, content, is_error=is_error)
        if tool_name == "propose_clustering" and not is_error and state is not None:
            self._session.update_latest_proposal(state)

    def on_tool_error(self, tool_name: str) -> None:
        self._session.record_tool_error(tool_name)

    def on_tokens(self, token_usage: dict, clustering_agent_session_tokens: dict) -> None:
        self._session.update_tokens(token_usage, clustering_agent_session_tokens)

    def on_human_tokens(self, token_usage: dict, human_agent_session_tokens: dict) -> None:
        _ = token_usage, human_agent_session_tokens

    def on_system_message(self, message: str) -> None:
        self._session.add_system_message(message)

    def on_human_message(self, message: str, *, simulated: bool) -> None:
        _ = message, simulated

    def on_simulation_start(self, max_turns: int, guidance: str | None) -> None:
        _ = max_turns, guidance

    def on_simulation_end(
        self,
        reason: str,
        *,
        clustering_agent_session_tokens: dict | None = None,
        human_agent_session_tokens: dict | None = None,
    ) -> None:
        _ = reason, clustering_agent_session_tokens, human_agent_session_tokens


class SessionNotFoundError(KeyError):
    pass


class TurnInProgressError(RuntimeError):
    pass


class DemoSessionStore:
    def __init__(self, settings: DemoSettings):
        self._settings = settings
        self._sessions: dict[str, DemoSession] = {}
        self._lock = threading.Lock()

    def create_session(self) -> DemoSession:
        context = build_demo_session_context(self._settings)
        session = DemoSession(
            session_id=str(uuid.uuid4()),
            context=context,
            token_totals=_session_token_totals(context),
        )
        with self._lock:
            self._sessions[session.session_id] = session
        session.emit_event(
            "session_started",
            {
                "session": jsonable_encoder(session.snapshot()),
                "dataset_name": self._settings.dataset_name,
                "backend": self._settings.clustering_agent_backend,
            },
        )
        return session

    def get_session(self, session_id: str) -> DemoSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()

    def submit_user_message(self, session_id: str, message: str) -> DemoSession:
        session = self.get_session(session_id)
        if not session.turn_lock.acquire(blocking=False):
            raise TurnInProgressError("A turn is already running for this session.")
        try:
            with session._lock:
                if session.is_turn_in_progress:
                    raise TurnInProgressError("A turn is already running for this session.")
                session.context.state["messages"].append(HumanMessage(content=message))
            session.add_user_message(message)
            session.begin_turn()
            worker = threading.Thread(
                target=self._run_agent_turn,
                args=(session,),
                daemon=True,
                name=f"demo-session-{session.session_id}",
            )
            worker.start()
            return session
        except Exception:
            session.turn_lock.release()
            raise

    def _run_agent_turn(self, session: DemoSession) -> None:
        reporter = DemoStreamingReporter(session)
        try:
            updated_state: AgentState = run_agent_turn(
                session.context.app,
                session.context.state,
                session.context.logger,
                session.context.clustering_agent_usage_collector,
                seed_accounting=session.context.seed_clustering_agent_accounting,
                reporter=reporter,
            )
            session.context.state = updated_state
            session.update_latest_proposal(session.context.state)
        except Exception as exc:
            logging.getLogger(__name__).exception("Demo session turn failed")
            session.add_system_message(f"Turn failed: {exc}", status="error")
            session.emit_event("session_error", {"message": str(exc)})
        finally:
            session.finish_turn()
            session.turn_lock.release()


def format_sse_event(event: SessionEvent) -> str:
    payload = json.dumps(jsonable_encoder(event.data), ensure_ascii=True)
    return f"id: {event.id}\nevent: {event.type}\ndata: {payload}\n\n"
