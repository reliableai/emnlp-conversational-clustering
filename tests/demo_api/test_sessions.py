"""Tests for the standalone demo API session bridge and proposal extraction logic."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import cast

from fastapi.testclient import TestClient

from conversational_clustering.accounting import UsageCollector
from conversational_clustering.clustering_agent.backends import (
    InstructionKMeansV1ClusteringBackendConfig,
    default_clustering_agent_backend_config,
)
from conversational_clustering.data_processing.data_loader import DataLoader
from conversational_clustering.simulation.runner import SessionContext, build_initial_state
from conversational_clustering.topic_discovery import TopicDiscoveryConfig
from demo_api.app import create_app
from demo_api.config import DemoSettings, resolve_demo_backend_config
from demo_api.sessions import (
    DemoSessionStore,
    _extract_latest_proposal_snapshot,
    _extract_tool_trace_details,
    _humanize_tool_name,
)


class _FakeDataLoader(DataLoader):
    def load(self):
        raise AssertionError("load should not be called in demo_api tests")

    def to_context(self) -> list[str]:
        return []

    def get_item_ids(self) -> list[str]:
        return []

    def has_embeddings(self) -> bool:
        return False

    def get_embeddings(self) -> dict[str, list[float]]:
        return {}


class _FakeStreamingApp:
    def stream(self, state, config=None, *, stream_mode):
        _ = state, config, stream_mode
        return []


def _build_session_context() -> SessionContext:
    return SessionContext(
        app=_FakeStreamingApp(),
        data_loader=cast(DataLoader, _FakeDataLoader()),
        logger=logging.getLogger("demo-api-tests"),
        state=build_initial_state(dataset_size=18),
        clustering_agent_backend="instruction_kmeans_v1",
        clustering_agent_backend_config=default_clustering_agent_backend_config(
            "instruction_kmeans_v1"
        ),
        human_agent_prompt_id="baseline_v3",
        human_agent_system_prompt="",
        topic_discovery_config=TopicDiscoveryConfig(),
        clustering_agent_usage_collector=UsageCollector(),
        human_agent_usage_collector=UsageCollector(),
        human_verifier_system_prompt="",
    )


def _build_test_app(*, store: DemoSessionStore) -> TestClient:
    return TestClient(create_app(session_store=store))


def test_extract_latest_proposal_snapshot_uses_latest_proposal_id() -> None:
    snapshot = _extract_latest_proposal_snapshot(
        {
            "instruction_kmeans_runtime_state": {
                "proposals_by_id": {
                    "proposal_2": {
                        "embedding_instruction": "urgency",
                        "num_clusters": 2,
                        "clusters": [
                            {
                                "name": "Urgent incidents",
                                "size": 8,
                                "summary": "High urgency work.",
                                "representative_points": ["Outage", "Security"],
                            }
                        ],
                    },
                    "proposal_1": {
                        "embedding_instruction": "older proposal",
                        "num_clusters": 1,
                        "clusters": [],
                    },
                }
            }
        }
    )

    assert snapshot is not None
    assert snapshot.proposal_id == "proposal_2"
    assert snapshot.embedding_instruction == "urgency"
    assert snapshot.num_clusters == 2
    assert snapshot.clusters[0].name == "Urgent incidents"


def test_humanize_tool_name_keeps_povs_uppercase() -> None:
    assert _humanize_tool_name("append_clustering_povs") == "Append Clustering POVs"


def test_extract_tool_trace_details_summarizes_added_clustering_povs() -> None:
    details = _extract_tool_trace_details(
        "append_clustering_povs",
        json.dumps(
            {
                "added_povs": [
                    "By customer intent",
                    "By issue lifecycle",
                ],
                "num_added": 2,
                "total_povs": 5,
            }
        ),
    )

    assert details == ["2 POVs added"]


def test_resolve_demo_backend_config_disables_local_instruction_embeddings() -> None:
    backend_config = default_clustering_agent_backend_config("instruction_kmeans_v1")

    resolved_config = resolve_demo_backend_config(
        backend_config,
        settings=DemoSettings(disable_local_embeddings=True),
    )

    assert isinstance(resolved_config, InstructionKMeansV1ClusteringBackendConfig)
    assert resolved_config.embedding_model_config.backend == "openrouter"
    assert resolved_config.embedding_model_config.model == "qwen/qwen3-embedding-8b"


def test_demo_api_message_flow_emits_transcript_and_proposal(monkeypatch) -> None:
    test_store = DemoSessionStore(DemoSettings())

    def fake_build_demo_session_context(settings: DemoSettings) -> SessionContext:
        _ = settings
        return _build_session_context()

    def fake_run_agent_turn(
        app, state, logger, clustering_agent_usage_collector, reporter=None, **kwargs
    ):
        _ = app, logger, kwargs
        assert reporter is not None
        reporter.on_agent_chunk("Working", is_start=True)
        reporter.on_tool_call("propose_clustering")
        reporter.on_agent_chunk(" through the dataset.", is_start=False)
        reporter.on_agent_message("Working through the dataset.", streamed=True)
        clustering_agent_usage_collector.record(
            "test/demo-agent",
            {"input_tokens": 7, "output_tokens": 11, "total_tokens": 18},
        )
        reporter.on_tokens(
            {"input_tokens": 7, "output_tokens": 11, "total_tokens": 18},
            {"input_tokens": 7, "output_tokens": 11, "total_tokens": 18},
        )
        state["instruction_kmeans_runtime_state"] = {
            "next_proposal_index": 2,
            "proposals_by_id": {
                "proposal_1": {
                    "embedding_instruction": "support theme",
                    "num_clusters": 3,
                    "clusters": [
                        {
                            "name": "Access issues",
                            "size": 6,
                            "summary": "Authentication and login work.",
                            "representative_points": ["VPN issue", "SSO failure"],
                        }
                    ],
                }
            },
        }
        return state

    monkeypatch.setattr(
        "demo_api.sessions.build_demo_session_context", fake_build_demo_session_context
    )
    monkeypatch.setattr("demo_api.sessions.run_agent_turn", fake_run_agent_turn)

    with _build_test_app(store=test_store) as client:
        create_response = client.post("/api/session")
        assert create_response.status_code == 201
        session_id = create_response.json()["session"]["session_id"]

        submit_response = client.post(
            f"/api/session/{session_id}/messages",
            json={"message": "Please cluster these items"},
        )
        assert submit_response.status_code == 202

        for _ in range(40):
            snapshot_response = client.get(f"/api/session/{session_id}/state")
            snapshot = snapshot_response.json()
            if snapshot["latest_proposal"] is not None and not snapshot["is_turn_in_progress"]:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("Turn did not complete in time.")

    transcript_roles = [entry["role"] for entry in snapshot["transcript"]]
    assert transcript_roles[:3] == ["user", "assistant", "system"]
    assert snapshot["latest_proposal"]["proposal_id"] == "proposal_1"
    assert snapshot["token_totals"]["total_tokens"] == 18


def test_demo_api_rejects_concurrent_turns(monkeypatch) -> None:
    test_store = DemoSessionStore(DemoSettings())
    started = threading.Event()
    release = threading.Event()

    def fake_build_demo_session_context(settings: DemoSettings) -> SessionContext:
        _ = settings
        return _build_session_context()

    def slow_run_agent_turn(
        app, state, logger, clustering_agent_usage_collector, reporter=None, **kwargs
    ):
        _ = app, state, logger, clustering_agent_usage_collector, reporter, kwargs
        started.set()
        release.wait(timeout=1.0)
        return state

    monkeypatch.setattr(
        "demo_api.sessions.build_demo_session_context", fake_build_demo_session_context
    )
    monkeypatch.setattr("demo_api.sessions.run_agent_turn", slow_run_agent_turn)

    with _build_test_app(store=test_store) as client:
        session_id = client.post("/api/session").json()["session"]["session_id"]
        first = client.post(f"/api/session/{session_id}/messages", json={"message": "first turn"})
        assert first.status_code == 202
        assert started.wait(timeout=1.0)

        second = client.post(f"/api/session/{session_id}/messages", json={"message": "second turn"})
        assert second.status_code == 409

    release.set()


def test_demo_api_emits_proposal_update_before_turn_finished(monkeypatch) -> None:
    test_store = DemoSessionStore(DemoSettings())

    def fake_build_demo_session_context(settings: DemoSettings) -> SessionContext:
        _ = settings
        return _build_session_context()

    def fake_run_agent_turn(
        app, state, logger, clustering_agent_usage_collector, reporter=None, **kwargs
    ):
        _ = app, logger, clustering_agent_usage_collector, kwargs
        assert reporter is not None
        reporter.on_tool_call("propose_clustering")
        state["instruction_kmeans_runtime_state"] = {
            "next_proposal_index": 2,
            "proposals_by_id": {
                "proposal_1": {
                    "embedding_instruction": "support theme",
                    "num_clusters": 3,
                    "clusters": [
                        {
                            "name": "Access issues",
                            "size": 6,
                            "summary": "Authentication and login work.",
                            "representative_points": ["VPN issue", "SSO failure"],
                        }
                    ],
                }
            },
        }
        reporter.on_tool_result(
            "propose_clustering",
            json.dumps(
                {
                    "proposal_id": "proposal_1",
                    "embedding_instruction": "support theme",
                    "num_clusters": 3,
                    "clusters": [
                        {
                            "name": "Access issues",
                            "size": 6,
                            "summary": "Authentication and login work.",
                            "representative_points": ["VPN issue", "SSO failure"],
                        }
                    ],
                }
            ),
            is_error=False,
            state=state,
        )
        return state

    monkeypatch.setattr(
        "demo_api.sessions.build_demo_session_context", fake_build_demo_session_context
    )
    monkeypatch.setattr("demo_api.sessions.run_agent_turn", fake_run_agent_turn)

    session = test_store.create_session()
    subscriber_id, subscriber_queue = session.add_subscriber()
    try:
        test_store.submit_user_message(session.session_id, "Please cluster these items")
        event_types: list[str] = []
        for _ in range(20):
            event = subscriber_queue.get(timeout=1.0)
            if event is None:
                break
            event_types.append(event.type)
            if event.type == "turn_finished":
                break
    finally:
        session.remove_subscriber(subscriber_id)

    assert "proposal_updated" in event_types
    assert "turn_finished" in event_types
    assert event_types.index("proposal_updated") < event_types.index("turn_finished")
