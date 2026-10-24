"""Unit tests for simulation runner streaming and human-simulation flows."""

import logging
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from unittest import mock

import httpx
import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.errors import GraphRecursionError

import conversational_clustering.simulation.runner as simulation_runner
from conversational_clustering.accounting import UsageCollector
from conversational_clustering.clustering_agent.backends import (
    BackendV1ClusteringBackendConfig,
)
from conversational_clustering.clustering_agent.backends.backend_v1 import (
    intermittent_clustering_judge as backend_v1_judge_module,
)
from conversational_clustering.clustering_agent.state import AgentState
from conversational_clustering.human_turn_injection import HumanTurnInjectionConfig
from conversational_clustering.models import (
    EmbeddingModelConfig,
    default_human_model_config,
    default_human_verifier_model_config,
)
from conversational_clustering.simulation.reporters import NullReporter, SimulationReporter
from conversational_clustering.topic_discovery import TopicDiscoveryConfig


class RecordingReporter(NullReporter):
    def __init__(self) -> None:
        self.agent_chunks: list[tuple[str, bool]] = []
        self.agent_messages: list[tuple[str, bool]] = []
        self.tool_calls: list[str] = []
        self.tool_errors: list[str] = []
        self.tokens: list[tuple[dict, dict]] = []
        self.human_tokens: list[tuple[dict, dict]] = []
        self.system_messages: list[str] = []
        self.human_messages: list[tuple[str, bool]] = []
        self.simulation_start: list[tuple[int, str | None]] = []
        self.simulation_end: list[tuple[str, dict | None, dict | None]] = []

    def on_agent_chunk(self, content: str, *, is_start: bool) -> None:
        self.agent_chunks.append((content, is_start))

    def on_agent_message(self, content: str, *, streamed: bool) -> None:
        self.agent_messages.append((content, streamed))

    def on_tool_call(self, tool_name: str) -> None:
        self.tool_calls.append(tool_name)

    def on_tool_error(self, tool_name: str) -> None:
        self.tool_errors.append(tool_name)

    def on_tokens(self, token_usage: dict, clustering_agent_session_tokens: dict) -> None:
        self.tokens.append((token_usage, clustering_agent_session_tokens))

    def on_human_tokens(self, token_usage: dict, human_agent_session_tokens: dict) -> None:
        self.human_tokens.append((token_usage, human_agent_session_tokens))

    def on_system_message(self, message: str) -> None:
        self.system_messages.append(message)

    def on_human_message(self, message: str, *, simulated: bool) -> None:
        self.human_messages.append((message, simulated))

    def on_simulation_start(self, max_turns: int, guidance: str | None) -> None:
        self.simulation_start.append((max_turns, guidance))

    def on_simulation_end(
        self,
        reason: str,
        *,
        clustering_agent_session_tokens: dict | None = None,
        human_agent_session_tokens: dict | None = None,
    ) -> None:
        self.simulation_end.append(
            (reason, clustering_agent_session_tokens, human_agent_session_tokens)
        )


class FakeStreamingApp:
    def __init__(self, events: list[tuple[str, object]]) -> None:
        self._events = events

    def stream(
        self,
        state: AgentState,
        config: dict[str, int] | None = None,
        *,
        stream_mode: list[str],
    ) -> Iterable[tuple[str, object]]:
        _ = state
        _ = config
        _ = stream_mode
        yield from self._events


class FakeHumanApp:
    def __init__(self, messages: Sequence[BaseMessage]) -> None:
        self._messages = list(messages)

    def invoke(self, state: AgentState) -> AgentState:
        state["messages"] = list(self._messages)
        return state


class ReplacingHumanApp:
    def __init__(self, messages: Sequence[BaseMessage]) -> None:
        self._messages = list(messages)

    def invoke(self, state: AgentState) -> AgentState:
        return {
            "messages": list(self._messages),
            "dataset_metadata": dict(state.get("dataset_metadata", {})),
            "iteration_count": state.get("iteration_count", 0),
            "human_thought_trace": list(state.get("human_thought_trace", [])),
            "judge_clustering_trace": list(state.get("judge_clustering_trace", [])),
        }


class ExplodingStreamingApp:
    def stream(
        self,
        state: AgentState,
        config: dict[str, int] | None = None,
        *,
        stream_mode: list[str],
    ) -> Iterable[tuple[str, object]]:
        _ = state
        _ = config
        _ = stream_mode
        raise RuntimeError("stream failed")


class StreamModeCapturingApp:
    def __init__(self) -> None:
        self.stream_modes: list[list[str]] = []
        self.configs: list[dict[str, int] | None] = []

    def stream(
        self,
        state: AgentState,
        config: dict[str, int] | None = None,
        *,
        stream_mode: list[str],
    ) -> Iterable[tuple[str, object]]:
        _ = state
        self.configs.append(config)
        self.stream_modes.append(list(stream_mode))
        yield (
            "values",
            {
                "messages": [AIMessage(content="Final response")],
                "dataset_metadata": {"size": 1},
                "iteration_count": 0,
            },
        )


class FakeDataLoader:
    def load(self) -> list[dict[str, str]]:
        return [{"id": "1", "text": "example"}]


class FakeClusteringMethod:
    def resolve_app(
        self,
        *,
        data_loader: object,
        save_path: str | None,
        usage_collector: UsageCollector | None = None,
    ) -> FakeStreamingApp:
        _ = data_loader, save_path, usage_collector
        return FakeStreamingApp([])


@pytest.fixture
def reporter() -> RecordingReporter:
    return RecordingReporter()


def _usage_collector(input_tokens: int = 0, output_tokens: int = 0) -> UsageCollector:
    collector = UsageCollector()
    if input_tokens or output_tokens:
        collector.record(
            "test-model",
            {"input_tokens": input_tokens, "output_tokens": output_tokens},
        )
    return collector


def test_create_session_appends_persona_context_to_human_agent_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        simulation_runner, "build_data_loader", lambda *args, **kwargs: FakeDataLoader()
    )
    monkeypatch.setattr(
        simulation_runner,
        "default_clustering_agent_backend_config",
        lambda backend: BackendV1ClusteringBackendConfig(),
    )
    monkeypatch.setattr(
        simulation_runner,
        "build_conversational_clustering_method",
        lambda **kwargs: FakeClusteringMethod(),
    )

    session = simulation_runner.create_session(
        dataset_name="synthetic",
        human_agent_prompt_id="baseline_v3",
        human_agent_persona="task_oriented",
        logger=logging.getLogger("test.create_session.persona"),
    )

    assert session.human_agent_persona == "task_oriented"
    assert "Name: Task-Oriented" in session.human_agent_system_prompt
    assert "Representative Example Prompts:" in session.human_agent_system_prompt


def test_create_session_builds_pricing_client_with_default_human_model_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_configs: list[object] = []

    monkeypatch.setattr(
        simulation_runner, "build_data_loader", lambda *args, **kwargs: FakeDataLoader()
    )
    monkeypatch.setattr(
        simulation_runner,
        "default_clustering_agent_backend_config",
        lambda backend: BackendV1ClusteringBackendConfig(),
    )
    monkeypatch.setattr(
        simulation_runner,
        "build_conversational_clustering_method",
        lambda **kwargs: FakeClusteringMethod(),
    )

    def fake_build_pricing_client_from_configs(configs: list[object]) -> object:
        captured_configs.extend(configs)
        return object()

    monkeypatch.setattr(
        simulation_runner,
        "build_pricing_client_from_configs",
        fake_build_pricing_client_from_configs,
    )

    session = simulation_runner.create_session(
        dataset_name="synthetic",
        human_agent_prompt_id="baseline_v3",
        logger=logging.getLogger("test.create_session.pricing_defaults"),
    )

    assert session.human_agent_model_config == default_human_model_config()
    assert session.human_verifier_model_config == default_human_verifier_model_config()
    assert default_human_model_config() in captured_configs
    assert default_human_verifier_model_config() in captured_configs


def test_create_session_passes_embedding_usage_collector_and_pricing_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_loader_kwargs: dict[str, object] = {}
    captured_configs: list[object] = []
    embedding_model_config = EmbeddingModelConfig(
        model="openai/text-embedding-3-small",
        backend="openrouter",
    )

    def fake_build_data_loader(*args: object, **kwargs: object) -> FakeDataLoader:
        _ = args
        captured_loader_kwargs.update(kwargs)
        return FakeDataLoader()

    def fake_build_pricing_client_from_configs(configs: list[object]) -> object:
        captured_configs.extend(configs)
        return object()

    monkeypatch.setattr(simulation_runner, "build_data_loader", fake_build_data_loader)
    monkeypatch.setattr(
        simulation_runner,
        "default_clustering_agent_backend_config",
        lambda backend: BackendV1ClusteringBackendConfig(),
    )
    monkeypatch.setattr(
        simulation_runner,
        "build_conversational_clustering_method",
        lambda **kwargs: FakeClusteringMethod(),
    )
    monkeypatch.setattr(
        simulation_runner,
        "build_pricing_client_from_configs",
        fake_build_pricing_client_from_configs,
    )

    session = simulation_runner.create_session(
        dataset_name="synthetic",
        enable_embeddings=True,
        embedding_model_config=embedding_model_config,
        logger=logging.getLogger("test.create_session.embedding_accounting"),
    )

    assert captured_loader_kwargs["usage_collector"] is session.clustering_agent_usage_collector
    assert embedding_model_config in captured_configs


def test_run_agent_turn_streaming_updates_tokens_and_iteration(
    reporter: RecordingReporter,
) -> None:
    state = simulation_runner.build_initial_state(dataset_size=1)
    events = [
        ("messages", (AIMessageChunk(content="Hello"), {})),
        ("messages", (AIMessageChunk(content=" there"), {})),
        (
            "values",
            {
                "messages": [
                    AIMessage(
                        content="Final response",
                        additional_kwargs={"token_usage": {"input_tokens": 2, "output_tokens": 3}},
                    )
                ],
                "dataset_metadata": {"size": 1},
                "iteration_count": 0,
            },
        ),
    ]
    app = FakeStreamingApp(events)
    clustering_agent_usage_collector = _usage_collector()

    result_state = simulation_runner.run_agent_turn(
        app=app,
        state=state,
        logger=logging.getLogger("test.run_agent_turn"),
        clustering_agent_usage_collector=clustering_agent_usage_collector,
        reporter=reporter,
    )

    assert result_state["iteration_count"] == 1
    assert clustering_agent_usage_collector.snapshot(pricer=None).total_tokens == 0
    assert reporter.agent_chunks == [("Hello", True), (" there", False)]
    assert reporter.agent_messages == [("Final response", True)]
    assert reporter.tokens == [
        (
            {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
    ]


def test_run_agent_turn_includes_seed_accounting_in_session_token_snapshots(
    reporter: RecordingReporter,
) -> None:
    state = simulation_runner.build_initial_state(dataset_size=1)
    app = FakeStreamingApp(
        [
            (
                "values",
                {
                    "messages": [
                        AIMessage(
                            content="Final response",
                            additional_kwargs={
                                "token_usage": {"input_tokens": 2, "output_tokens": 3}
                            },
                        )
                    ],
                    "dataset_metadata": {"size": 1},
                    "iteration_count": 0,
                },
            ),
        ]
    )
    clustering_agent_usage_collector = _usage_collector(11, 13)

    _ = simulation_runner.run_agent_turn(
        app=app,
        state=state,
        logger=logging.getLogger("test.run_agent_turn.seeded_tokens"),
        clustering_agent_usage_collector=clustering_agent_usage_collector,
        seed_accounting={"input_tokens": 3, "output_tokens": 5, "total_tokens": 8},
        reporter=reporter,
    )

    assert reporter.tokens == [
        (
            {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
            {"input_tokens": 14, "output_tokens": 18, "total_tokens": 32},
        )
    ]


def test_run_human_simulation_preserves_latest_agent_state_on_timeout() -> None:
    class PartialThenTimeoutApp:
        def stream(
            self,
            state: AgentState,
            config: dict[str, int] | None = None,
            *,
            stream_mode: list[str],
        ) -> Iterable[tuple[str, object]]:
            _ = config
            _ = stream_mode
            yield (
                "values",
                {
                    "messages": [*state["messages"], AIMessage(content="Partial response")],
                    "dataset_metadata": dict(state["dataset_metadata"]),
                    "iteration_count": state["iteration_count"],
                    "human_thought_trace": list(state.get("human_thought_trace", [])),
                    "judge_clustering_trace": list(state.get("judge_clustering_trace", [])),
                },
            )
            raise httpx.ReadTimeout("timed out")

    session = simulation_runner.SessionContext(
        app=PartialThenTimeoutApp(),
        data_loader=mock.Mock(),
        logger=logging.getLogger("test.run_human_simulation.timeout_state"),
        state=simulation_runner.build_initial_state(dataset_size=1),
        clustering_agent_backend="backend_v1",
        clustering_agent_backend_config=BackendV1ClusteringBackendConfig(),
        human_agent_prompt_id="baseline_v3",
        human_agent_system_prompt="unused",
        topic_discovery_config=TopicDiscoveryConfig(),
        clustering_agent_usage_collector=_usage_collector(),
        human_agent_usage_collector=_usage_collector(),
        turn_injection=None,
        selected_topic_label=None,
        selected_topic_guidance=None,
        human_verifier_system_prompt="",
        human_agent_model_config=None,
        human_verifier_model_config=None,
        max_verification_retries=0,
        enable_intermittent_clustering=False,
    )

    with pytest.raises(httpx.ReadTimeout):
        simulation_runner.run_human_simulation(
            session=session,
            guidance=None,
            max_turns=1,
            reporter=RecordingReporter(),
            human_app=ReplacingHumanApp([HumanMessage(content="Need help")]),
        )

    assert [message.content for message in session.state["messages"]] == [
        "Need help",
        "Partial response",
    ]
    assert session.state["iteration_count"] == 0


def test_run_human_simulation_merges_seed_accounting_into_final_result(
    reporter: RecordingReporter,
) -> None:
    session = simulation_runner.SessionContext(
        app=FakeStreamingApp(
            [
                (
                    "values",
                    {
                        "messages": [AIMessage(content="Final response")],
                        "dataset_metadata": {"size": 1},
                        "iteration_count": 0,
                    },
                )
            ]
        ),
        data_loader=mock.Mock(),
        logger=logging.getLogger("test.run_human_simulation.seeded_accounting"),
        state=simulation_runner.build_initial_state(dataset_size=1),
        clustering_agent_backend="backend_v1",
        clustering_agent_backend_config=BackendV1ClusteringBackendConfig(),
        human_agent_prompt_id="baseline_v3",
        human_agent_system_prompt="unused",
        topic_discovery_config=TopicDiscoveryConfig(),
        clustering_agent_usage_collector=_usage_collector(11, 13),
        human_agent_usage_collector=_usage_collector(5, 7),
        seed_clustering_agent_accounting={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        seed_human_agent_accounting={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        turn_injection=None,
        selected_topic_label=None,
        selected_topic_guidance=None,
        human_verifier_system_prompt="",
        human_agent_model_config=None,
        human_verifier_model_config=None,
        max_verification_retries=0,
        enable_intermittent_clustering=False,
    )

    result = simulation_runner.run_human_simulation(
        session=session,
        guidance=None,
        max_turns=1,
        reporter=reporter,
        human_app=FakeHumanApp([HumanMessage(content="Need help")]),
    )

    assert result.clustering_agent_accounting["input_tokens"] == 14
    assert result.clustering_agent_accounting["output_tokens"] == 17
    assert result.clustering_agent_accounting["total_tokens"] == 31
    assert result.human_agent_accounting["input_tokens"] == 7
    assert result.human_agent_accounting["output_tokens"] == 8
    assert result.human_agent_accounting["total_tokens"] == 15
    assert reporter.simulation_end == [
        (
            "max_turns",
            {"input_tokens": 14, "output_tokens": 17, "total_tokens": 31},
            {"input_tokens": 7, "output_tokens": 8, "total_tokens": 15},
        )
    ]


def test_run_agent_turn_marks_tool_error_when_tool_response_is_error_like(
    reporter: RecordingReporter,
) -> None:
    state = simulation_runner.build_initial_state(dataset_size=1)
    app = FakeStreamingApp(
        [
            (
                "values",
                {
                    "messages": [
                        ToolMessage(
                            name="save_clustering",
                            content="Error: unable to save",
                            tool_call_id="1",
                        )
                    ],
                    "dataset_metadata": {"size": 1},
                    "iteration_count": 0,
                },
            )
        ]
    )

    simulation_runner.run_agent_turn(
        app=app,
        state=state,
        logger=logging.getLogger("test.run_agent_turn.tool"),
        clustering_agent_usage_collector=_usage_collector(),
        reporter=reporter,
    )

    assert reporter.tool_calls == ["save_clustering"]
    assert reporter.tool_errors == ["save_clustering"]


def test_run_agent_turn_reports_tool_call_before_tool_result_without_duplicates(
    reporter: RecordingReporter,
) -> None:
    state = simulation_runner.build_initial_state(dataset_size=1)
    app = FakeStreamingApp(
        [
            (
                "values",
                {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "propose_clustering",
                                    "id": "call_1",
                                    "args": {
                                        "embedding_instruction": "ticket topic",
                                        "num_clusters": 6,
                                    },
                                }
                            ],
                        )
                    ],
                    "dataset_metadata": {"size": 1},
                    "iteration_count": 0,
                },
            ),
            (
                "values",
                {
                    "messages": [
                        ToolMessage(
                            name="propose_clustering",
                            content='{"proposal_id":"proposal_1","num_clusters":6,"clusters":[]}',
                            tool_call_id="call_1",
                        )
                    ],
                    "dataset_metadata": {"size": 1},
                    "iteration_count": 0,
                },
            ),
        ]
    )

    simulation_runner.run_agent_turn(
        app=app,
        state=state,
        logger=logging.getLogger("test.run_agent_turn.tool_call_timing"),
        clustering_agent_usage_collector=UsageCollector(),
        reporter=reporter,
    )

    assert reporter.tool_calls == ["propose_clustering"]
    assert reporter.tool_errors == []


def test_run_agent_turn_ignores_malformed_message_payloads(
    reporter: RecordingReporter,
) -> None:
    state = simulation_runner.build_initial_state(dataset_size=1)
    app = FakeStreamingApp(
        [
            ("messages", "invalid"),
            ("messages", (AIMessageChunk(content="ignored"),)),
            (
                "values",
                {
                    "messages": [AIMessage(content="Final response")],
                    "dataset_metadata": {"size": 1},
                    "iteration_count": 0,
                },
            ),
        ]
    )

    result_state = simulation_runner.run_agent_turn(
        app=app,
        state=state,
        logger=logging.getLogger("test.run_agent_turn.malformed_messages"),
        clustering_agent_usage_collector=_usage_collector(),
        reporter=reporter,
    )

    assert result_state["iteration_count"] == 1
    assert reporter.agent_chunks == []
    assert reporter.agent_messages == [("Final response", False)]


def test_run_agent_turn_ignores_non_dict_values_payload(
    reporter: RecordingReporter,
) -> None:
    state = simulation_runner.build_initial_state(dataset_size=1)
    app = FakeStreamingApp(
        [
            ("values", "invalid"),
            (
                "values",
                {
                    "messages": [AIMessage(content="Final response")],
                    "dataset_metadata": {"size": 1},
                    "iteration_count": 0,
                },
            ),
        ]
    )

    result_state = simulation_runner.run_agent_turn(
        app=app,
        state=state,
        logger=logging.getLogger("test.run_agent_turn.invalid_values"),
        clustering_agent_usage_collector=_usage_collector(),
        reporter=reporter,
    )

    assert result_state["iteration_count"] == 1
    assert reporter.agent_messages == [("Final response", False)]


def test_run_agent_turn_propagates_stream_exception(
    reporter: RecordingReporter,
) -> None:
    state = simulation_runner.build_initial_state(dataset_size=1)

    with pytest.raises(RuntimeError, match="stream failed"):
        simulation_runner.run_agent_turn(
            app=ExplodingStreamingApp(),
            state=state,
            logger=logging.getLogger("test.run_agent_turn.exception"),
            clustering_agent_usage_collector=_usage_collector(),
            reporter=reporter,
        )


def test_run_agent_turn_uses_values_only_stream_for_plain_null_reporter() -> None:
    state = simulation_runner.build_initial_state(dataset_size=1)
    app = StreamModeCapturingApp()

    result_state = simulation_runner.run_agent_turn(
        app=app,
        state=state,
        logger=logging.getLogger("test.run_agent_turn.null_reporter"),
        clustering_agent_usage_collector=_usage_collector(),
        reporter=NullReporter(),
    )

    assert result_state["iteration_count"] == 1
    assert app.stream_modes == [["values"]]
    assert app.configs == [None]


def test_run_agent_turn_returns_latest_state_when_final_turn_hits_recursion_limit(
    reporter: RecordingReporter,
) -> None:
    state = simulation_runner.build_initial_state(dataset_size=1)

    class RecursingStreamingApp:
        def stream(
            self,
            state: AgentState,
            config: dict[str, int] | None = None,
            *,
            stream_mode: list[str],
        ) -> Iterable[tuple[str, object]]:
            _ = state
            _ = stream_mode
            assert config == {"recursion_limit": 9}
            yield (
                "values",
                {
                    "messages": [AIMessage(content="Still working.")],
                    "dataset_metadata": {"size": 1},
                    "iteration_count": 0,
                },
            )
            raise GraphRecursionError("step limit reached")

    result_state = simulation_runner.run_agent_turn(
        app=RecursingStreamingApp(),
        state=state,
        logger=logging.getLogger("test.run_agent_turn.recursion_limit"),
        clustering_agent_usage_collector=_usage_collector(),
        reporter=reporter,
        recursion_limit=9,
        allow_graph_recursion_exhaustion=True,
    )

    assert result_state["iteration_count"] == 1
    assert result_state["final_turn_recursion_limit_reached"] is True
    assert reporter.agent_messages == [("Still working.", False)]
    assert reporter.system_messages == [
        "System: Final clustering-agent turn reached the execution step limit. "
        "Ending the run with the latest available state."
    ]


def test_run_human_simulation_reports_final_turn_recursion_limit_stop_reason(
    reporter: RecordingReporter,
) -> None:
    session = simulation_runner.SessionContext(
        app=FakeStreamingApp([]),
        data_loader=mock.Mock(),
        logger=logging.getLogger("test.run_human_simulation.final_turn_recursion_limit"),
        state=simulation_runner.build_initial_state(dataset_size=1),
        clustering_agent_backend="backend_v1",
        clustering_agent_backend_config=BackendV1ClusteringBackendConfig(),
        human_agent_prompt_id="baseline_v1",
        human_agent_system_prompt="Human prompt",
        topic_discovery_config=TopicDiscoveryConfig(),
        clustering_agent_usage_collector=_usage_collector(),
        human_agent_usage_collector=_usage_collector(4, 6),
    )
    human_app = FakeHumanApp(
        messages=[HumanMessage(content="Please try one last clustering pass.")]
    )

    def _fake_run_agent_turn(
        app: simulation_runner.StreamingApp,
        state: AgentState,
        logger: logging.Logger,
        clustering_agent_session_tokens: UsageCollector,
        seed_accounting: dict[str, object] | None = None,
        reporter: SimulationReporter | None = None,
        recursion_limit: int | None = None,
        allow_graph_recursion_exhaustion: bool = False,
        state_update_callback: Callable[[AgentState], None] | None = None,
    ) -> AgentState:
        _ = app
        _ = logger
        _ = clustering_agent_session_tokens
        _ = seed_accounting
        _ = reporter
        _ = state_update_callback
        assert recursion_limit == simulation_runner._FINAL_TURN_AGENT_RECURSION_LIMIT
        assert allow_graph_recursion_exhaustion is True
        state["messages"].append(AIMessage(content="Still iterating."))
        state["final_turn_recursion_limit_reached"] = True
        return state

    with mock.patch.object(simulation_runner, "run_agent_turn", side_effect=_fake_run_agent_turn):
        result = simulation_runner.run_human_simulation(
            session=session,
            guidance="Keep clusters coherent",
            max_turns=1,
            reporter=reporter,
            human_app=human_app,
        )

    assert result.turns_run == 1
    assert result.stopped_reason == "final_turn_recursion_limit"
    assert reporter.simulation_end[0][0] == "final_turn_recursion_limit"


def test_run_human_simulation_prefers_clustering_saved_over_final_turn_recursion_limit(
    reporter: RecordingReporter,
) -> None:
    session = simulation_runner.SessionContext(
        app=FakeStreamingApp([]),
        data_loader=mock.Mock(),
        logger=logging.getLogger("test.run_human_simulation.final_turn_recursion_limit_after_save"),
        state=simulation_runner.build_initial_state(dataset_size=1),
        clustering_agent_backend="backend_v1",
        clustering_agent_backend_config=BackendV1ClusteringBackendConfig(),
        human_agent_prompt_id="baseline_v1",
        human_agent_system_prompt="Human prompt",
        topic_discovery_config=TopicDiscoveryConfig(),
        clustering_agent_usage_collector=_usage_collector(),
        human_agent_usage_collector=_usage_collector(4, 6),
    )
    human_app = FakeHumanApp(messages=[HumanMessage(content="Please save the clustering now.")])

    def _fake_run_agent_turn(
        app: simulation_runner.StreamingApp,
        state: AgentState,
        logger: logging.Logger,
        clustering_agent_session_tokens: UsageCollector,
        seed_accounting: dict[str, object] | None = None,
        reporter: SimulationReporter | None = None,
        recursion_limit: int | None = None,
        allow_graph_recursion_exhaustion: bool = False,
        state_update_callback: Callable[[AgentState], None] | None = None,
    ) -> AgentState:
        _ = app
        _ = logger
        _ = clustering_agent_session_tokens
        _ = seed_accounting
        _ = reporter
        _ = state_update_callback
        assert recursion_limit == simulation_runner._FINAL_TURN_AGENT_RECURSION_LIMIT
        assert allow_graph_recursion_exhaustion is True
        state["messages"].append(
            ToolMessage(
                name="save_clustering",
                content="Clustering saved to /tmp/path.json",
                tool_call_id="1",
            )
        )
        state["final_turn_recursion_limit_reached"] = True
        return state

    with mock.patch.object(simulation_runner, "run_agent_turn", side_effect=_fake_run_agent_turn):
        result = simulation_runner.run_human_simulation(
            session=session,
            guidance="Keep clusters coherent",
            max_turns=1,
            reporter=reporter,
            human_app=human_app,
        )

    assert result.turns_run == 1
    assert result.stopped_reason == "clustering_saved"
    assert reporter.simulation_end[0][0] == "clustering_saved"


def test_run_human_simulation_stops_when_human_message_missing(
    reporter: RecordingReporter,
) -> None:
    session = simulation_runner.SessionContext(
        app=FakeStreamingApp([]),
        data_loader=mock.Mock(),
        logger=logging.getLogger("test.run_human_simulation.empty"),
        state=simulation_runner.build_initial_state(dataset_size=1),
        clustering_agent_backend="backend_v1",
        clustering_agent_backend_config=BackendV1ClusteringBackendConfig(),
        human_agent_prompt_id="baseline_v1",
        human_agent_system_prompt="Human prompt",
        topic_discovery_config=TopicDiscoveryConfig(),
        clustering_agent_usage_collector=_usage_collector(),
        human_agent_usage_collector=_usage_collector(4, 6),
    )
    human_app = FakeHumanApp(messages=[])

    result = simulation_runner.run_human_simulation(
        session=session,
        guidance="Keep clusters coherent",
        max_turns=3,
        reporter=reporter,
        human_app=human_app,
    )

    assert result.stopped_reason == "no_human_message"
    assert result.turns_run == 0
    assert reporter.simulation_end[0][0] == "no_human_message"


def test_run_human_simulation_rejects_guidance_for_guidance_free_prompt(
    reporter: RecordingReporter,
) -> None:
    session = simulation_runner.SessionContext(
        app=FakeStreamingApp([]),
        data_loader=mock.Mock(),
        logger=logging.getLogger("test.run_human_simulation.guidance_mismatch"),
        state=simulation_runner.build_initial_state(dataset_size=1),
        clustering_agent_backend="backend_v1",
        clustering_agent_backend_config=BackendV1ClusteringBackendConfig(),
        human_agent_prompt_id="baseline_v3",
        human_agent_system_prompt="Human prompt",
        topic_discovery_config=TopicDiscoveryConfig(),
        clustering_agent_usage_collector=_usage_collector(),
        human_agent_usage_collector=_usage_collector(),
    )

    with pytest.raises(ValueError, match="does not accept guidance"):
        simulation_runner.run_human_simulation(
            session=session,
            guidance="Keep clusters coherent",
            max_turns=3,
            reporter=reporter,
            human_app=FakeHumanApp(messages=[]),
        )


def test_run_human_simulation_stops_after_clustering_saved(
    reporter: RecordingReporter,
) -> None:
    session = simulation_runner.SessionContext(
        app=FakeStreamingApp([]),
        data_loader=mock.Mock(),
        logger=logging.getLogger("test.run_human_simulation.saved"),
        state=simulation_runner.build_initial_state(dataset_size=1),
        clustering_agent_backend="backend_v1",
        clustering_agent_backend_config=BackendV1ClusteringBackendConfig(),
        human_agent_prompt_id="baseline_v1",
        human_agent_system_prompt="Human prompt",
        topic_discovery_config=TopicDiscoveryConfig(),
        clustering_agent_usage_collector=_usage_collector(),
        human_agent_usage_collector=_usage_collector(4, 6),
    )
    human_app = FakeHumanApp(
        messages=[
            HumanMessage(
                content="Please save this clustering.",
                additional_kwargs={"token_usage": {"input_tokens": 4, "output_tokens": 6}},
            )
        ]
    )

    def _fake_run_agent_turn(
        app: simulation_runner.StreamingApp,
        state: AgentState,
        logger: logging.Logger,
        clustering_agent_session_tokens: UsageCollector,
        seed_accounting: dict[str, object] | None = None,
        reporter: SimulationReporter | None = None,
        recursion_limit: int | None = None,
        allow_graph_recursion_exhaustion: bool = False,
        state_update_callback: Callable[[AgentState], None] | None = None,
    ) -> AgentState:
        _ = app
        _ = logger
        _ = clustering_agent_session_tokens
        _ = seed_accounting
        _ = reporter
        _ = recursion_limit
        _ = allow_graph_recursion_exhaustion
        _ = state_update_callback
        state["messages"].append(
            ToolMessage(
                name="save_clustering",
                content="Clustering saved to /tmp/path.json",
                tool_call_id="1",
            )
        )
        return state

    with mock.patch.object(simulation_runner, "run_agent_turn", side_effect=_fake_run_agent_turn):
        result = simulation_runner.run_human_simulation(
            session=session,
            guidance="Keep clusters coherent",
            max_turns=3,
            reporter=reporter,
            human_app=human_app,
        )

    assert result.stopped_reason == "clustering_saved"
    assert result.turns_run == 1
    assert result.human_agent_accounting["total_tokens"] == 10
    assert result.clustering_agent_accounting["total_tokens"] == 0
    assert reporter.simulation_end[0][0] == "clustering_saved"


def test_run_human_simulation_extracts_hidden_human_thought_and_keeps_it_out_of_messages(
    reporter: RecordingReporter,
) -> None:
    session = simulation_runner.SessionContext(
        app=FakeStreamingApp([]),
        data_loader=mock.Mock(),
        logger=logging.getLogger("test.run_human_simulation.thought_trace"),
        state=simulation_runner.build_initial_state(dataset_size=1),
        clustering_agent_backend="backend_v1",
        clustering_agent_backend_config=BackendV1ClusteringBackendConfig(),
        human_agent_prompt_id="baseline_v1",
        human_agent_system_prompt="Human prompt",
        topic_discovery_config=TopicDiscoveryConfig(),
        clustering_agent_usage_collector=_usage_collector(),
        human_agent_usage_collector=_usage_collector(),
    )
    human_app = FakeHumanApp(
        messages=[
            HumanMessage(
                content="Could you summarize by topic?",
                additional_kwargs={
                    "token_usage": {"input_tokens": 2, "output_tokens": 3},
                    "human_thought": "I want to check progress before asking for changes.",
                },
            )
        ]
    )

    def _fake_run_agent_turn(
        app: simulation_runner.StreamingApp,
        state: AgentState,
        logger: logging.Logger,
        clustering_agent_session_tokens: UsageCollector,
        seed_accounting: dict[str, object] | None = None,
        reporter: SimulationReporter | None = None,
        recursion_limit: int | None = None,
        allow_graph_recursion_exhaustion: bool = False,
        state_update_callback: Callable[[AgentState], None] | None = None,
    ) -> AgentState:
        _ = app
        _ = logger
        _ = clustering_agent_session_tokens
        _ = seed_accounting
        _ = reporter
        _ = state_update_callback
        assert recursion_limit == simulation_runner._FINAL_TURN_AGENT_RECURSION_LIMIT
        assert allow_graph_recursion_exhaustion is True
        state["messages"].append(AIMessage(content="Summary shared."))
        return state

    with mock.patch.object(simulation_runner, "run_agent_turn", side_effect=_fake_run_agent_turn):
        result = simulation_runner.run_human_simulation(
            session=session,
            guidance="Keep clusters coherent",
            max_turns=1,
            reporter=reporter,
            human_app=human_app,
        )

    assert result.stopped_reason == "max_turns"
    assert result.state["human_thought_trace"] == [
        {
            "turn_index": 1,
            "message": "Could you summarize by topic?",
            "thought": "I want to check progress before asking for changes.",
        }
    ]
    first_human_message = result.state["messages"][0]
    assert isinstance(first_human_message, HumanMessage)
    assert "human_thought" not in first_human_message.additional_kwargs
    assert first_human_message.additional_kwargs["token_usage"] == {
        "input_tokens": 2,
        "output_tokens": 3,
    }


def test_run_human_simulation_continues_human_thought_turn_index_for_seeded_state(
    reporter: RecordingReporter,
) -> None:
    session = simulation_runner.SessionContext(
        app=FakeStreamingApp([]),
        data_loader=mock.Mock(),
        logger=logging.getLogger("test.run_human_simulation.thought_trace.seeded"),
        state={
            "messages": [],
            "dataset_metadata": {"size": 1},
            "iteration_count": 2,
            "human_thought_trace": [
                {"turn_index": 1, "message": "Start by topic.", "thought": "Baseline first."},
                {"turn_index": 2, "message": "Now refine by sentiment.", "thought": "Need nuance."},
            ],
        },
        clustering_agent_backend="backend_v1",
        clustering_agent_backend_config=BackendV1ClusteringBackendConfig(),
        human_agent_prompt_id="baseline_v1",
        human_agent_system_prompt="Human prompt",
        topic_discovery_config=TopicDiscoveryConfig(),
        clustering_agent_usage_collector=_usage_collector(),
        human_agent_usage_collector=_usage_collector(),
    )
    human_app = FakeHumanApp(
        messages=[
            HumanMessage(
                content="Please split ambiguous items into a separate cluster.",
                additional_kwargs={"human_thought": "I want to avoid forced assignments."},
            )
        ]
    )

    def _fake_run_agent_turn(
        app: simulation_runner.StreamingApp,
        state: AgentState,
        logger: logging.Logger,
        clustering_agent_session_tokens: UsageCollector,
        seed_accounting: dict[str, object] | None = None,
        reporter: SimulationReporter | None = None,
        recursion_limit: int | None = None,
        allow_graph_recursion_exhaustion: bool = False,
        state_update_callback: Callable[[AgentState], None] | None = None,
    ) -> AgentState:
        _ = app
        _ = logger
        _ = clustering_agent_session_tokens
        _ = seed_accounting
        _ = reporter
        _ = recursion_limit
        _ = allow_graph_recursion_exhaustion
        _ = state_update_callback
        state["messages"].append(AIMessage(content="Updated clustering logic applied."))
        state["iteration_count"] = state["iteration_count"] + 1
        return state

    with mock.patch.object(simulation_runner, "run_agent_turn", side_effect=_fake_run_agent_turn):
        result = simulation_runner.run_human_simulation(
            session=session,
            guidance="Keep clusters coherent",
            max_turns=1,
            reporter=reporter,
            human_app=human_app,
        )

    assert result.state["human_thought_trace"][-1] == {
        "turn_index": 3,
        "message": "Please split ambiguous items into a separate cluster.",
        "thought": "I want to avoid forced assignments.",
    }


def test_run_interactive_session_records_hidden_judge_snapshot(
    tmp_path: Path,
) -> None:
    judge_path = tmp_path / "interactive.judge.json"
    numbered_judge_path = tmp_path / "interactive.judge.0001.json"
    session = simulation_runner.SessionContext(
        app=FakeStreamingApp([]),
        data_loader=mock.Mock(),
        logger=logging.getLogger("test.run_interactive_session.judge"),
        state=simulation_runner.build_initial_state(dataset_size=1),
        clustering_agent_backend="backend_v1",
        clustering_agent_backend_config=BackendV1ClusteringBackendConfig(),
        human_agent_prompt_id="baseline_v1",
        human_agent_system_prompt="Human prompt",
        topic_discovery_config=TopicDiscoveryConfig(),
        clustering_agent_usage_collector=_usage_collector(11, 13),
        human_agent_usage_collector=_usage_collector(),
        enable_intermittent_clustering=True,
        intermittent_clustering_judge_save_path=str(judge_path),
    )
    judge_payload = {
        "clusters": [
            {"name": "Topic A", "members": ["alpha"]},
        ]
    }
    inputs = iter(["Please cluster by topic."])

    def _input_fn(_prompt: str) -> str:
        try:
            return next(inputs)
        except StopIteration as exc:
            raise EOFError from exc

    def _fake_run_agent_turn(
        app: simulation_runner.StreamingApp,
        state: AgentState,
        logger: logging.Logger,
        clustering_agent_session_tokens: UsageCollector,
        seed_accounting: dict[str, object] | None = None,
        reporter: SimulationReporter | None = None,
    ) -> AgentState:
        _ = app
        _ = logger
        _ = clustering_agent_session_tokens
        _ = seed_accounting
        _ = reporter
        assert state["judge_clustering_trace"] == [{"turn_index": 1, "clustering": judge_payload}]
        state["messages"].append(AIMessage(content="Working on it."))
        state["iteration_count"] = state["iteration_count"] + 1
        return state

    with mock.patch.object(
        backend_v1_judge_module,
        "infer_backend_v1_intermittent_clustering_result",
        return_value=backend_v1_judge_module.IntermittentClusteringInferenceResult(
            clustering=judge_payload,
            token_usage=None,
        ),
    ) as infer_judge_mock:
        with mock.patch.object(
            simulation_runner, "run_agent_turn", side_effect=_fake_run_agent_turn
        ):
            result = simulation_runner.run_interactive_session(
                session,
                reporter=NullReporter(),
                input_fn=_input_fn,
            )

    assert result.state["judge_clustering_trace"] == [
        {"turn_index": 1, "clustering": judge_payload}
    ]
    assert [type(message).__name__ for message in result.state["messages"]] == [
        "HumanMessage",
        "AIMessage",
    ]
    assert "max_retries" not in infer_judge_mock.call_args.kwargs
    assert not judge_path.exists()
    assert numbered_judge_path.exists()
    assert numbered_judge_path.read_text(encoding="utf-8").strip().startswith("{")


def test_run_human_simulation_records_hidden_judge_snapshot_before_agent_turn(
    reporter: RecordingReporter,
    tmp_path: Path,
) -> None:
    judge_path = tmp_path / "simulate.judge.json"
    numbered_judge_path = tmp_path / "simulate.judge.0001.json"
    session = simulation_runner.SessionContext(
        app=FakeStreamingApp([]),
        data_loader=mock.Mock(),
        logger=logging.getLogger("test.run_human_simulation.judge"),
        state=simulation_runner.build_initial_state(dataset_size=1),
        clustering_agent_backend="backend_v1",
        clustering_agent_backend_config=BackendV1ClusteringBackendConfig(),
        human_agent_prompt_id="baseline_v1",
        human_agent_system_prompt="Human prompt",
        topic_discovery_config=TopicDiscoveryConfig(),
        clustering_agent_usage_collector=_usage_collector(),
        human_agent_usage_collector=_usage_collector(),
        enable_intermittent_clustering=True,
        intermittent_clustering_judge_save_path=str(judge_path),
    )
    human_app = FakeHumanApp(messages=[HumanMessage(content="Group by topic.")])
    judge_payload = {
        "clusters": [
            {"name": "Topic A", "members": ["alpha"]},
        ]
    }

    def _fake_run_agent_turn(
        app: simulation_runner.StreamingApp,
        state: AgentState,
        logger: logging.Logger,
        clustering_agent_session_tokens: UsageCollector,
        seed_accounting: dict[str, object] | None = None,
        reporter: SimulationReporter | None = None,
        recursion_limit: int | None = None,
        allow_graph_recursion_exhaustion: bool = False,
        state_update_callback: Callable[[AgentState], None] | None = None,
    ) -> AgentState:
        _ = app
        _ = logger
        _ = clustering_agent_session_tokens
        _ = seed_accounting
        _ = reporter
        _ = recursion_limit
        _ = allow_graph_recursion_exhaustion
        _ = state_update_callback
        assert state["judge_clustering_trace"] == [{"turn_index": 1, "clustering": judge_payload}]
        state["messages"].append(AIMessage(content="Topic grouping ready."))
        state["iteration_count"] = state["iteration_count"] + 1
        return state

    with mock.patch.object(
        backend_v1_judge_module,
        "infer_backend_v1_intermittent_clustering_result",
        return_value=backend_v1_judge_module.IntermittentClusteringInferenceResult(
            clustering=judge_payload,
            token_usage=None,
        ),
    ) as infer_judge_mock:
        with mock.patch.object(
            simulation_runner, "run_agent_turn", side_effect=_fake_run_agent_turn
        ):
            result = simulation_runner.run_human_simulation(
                session=session,
                guidance="Keep clusters coherent",
                max_turns=1,
                reporter=reporter,
                human_app=human_app,
            )

    assert result.state["judge_clustering_trace"] == [
        {"turn_index": 1, "clustering": judge_payload}
    ]
    assert "max_retries" not in infer_judge_mock.call_args.kwargs
    assert not judge_path.exists()
    assert numbered_judge_path.exists()


def test_run_human_simulation_skips_hidden_judge_snapshot_when_disabled(
    reporter: RecordingReporter,
) -> None:
    session = simulation_runner.SessionContext(
        app=FakeStreamingApp([]),
        data_loader=mock.Mock(),
        logger=logging.getLogger("test.run_human_simulation.judge.disabled"),
        state=simulation_runner.build_initial_state(dataset_size=1),
        clustering_agent_backend="backend_v1",
        clustering_agent_backend_config=BackendV1ClusteringBackendConfig(),
        human_agent_prompt_id="baseline_v1",
        human_agent_system_prompt="Human prompt",
        topic_discovery_config=TopicDiscoveryConfig(),
        clustering_agent_usage_collector=_usage_collector(),
        human_agent_usage_collector=_usage_collector(),
        enable_intermittent_clustering=False,
    )
    human_app = FakeHumanApp(messages=[HumanMessage(content="Group by topic.")])

    def _fake_run_agent_turn(
        app: simulation_runner.StreamingApp,
        state: AgentState,
        logger: logging.Logger,
        clustering_agent_session_tokens: UsageCollector,
        seed_accounting: dict[str, object] | None = None,
        reporter: SimulationReporter | None = None,
        recursion_limit: int | None = None,
        allow_graph_recursion_exhaustion: bool = False,
        state_update_callback: Callable[[AgentState], None] | None = None,
    ) -> AgentState:
        _ = app
        _ = logger
        _ = clustering_agent_session_tokens
        _ = seed_accounting
        _ = reporter
        _ = recursion_limit
        _ = allow_graph_recursion_exhaustion
        _ = state_update_callback
        assert state["judge_clustering_trace"] == []
        state["messages"].append(AIMessage(content="Topic grouping ready."))
        state["iteration_count"] = state["iteration_count"] + 1
        return state

    with mock.patch.object(
        backend_v1_judge_module,
        "infer_backend_v1_intermittent_clustering_result",
        side_effect=AssertionError("judge should not run when disabled"),
    ):
        with mock.patch.object(
            simulation_runner, "run_agent_turn", side_effect=_fake_run_agent_turn
        ):
            result = simulation_runner.run_human_simulation(
                session=session,
                guidance="Keep clusters coherent",
                max_turns=1,
                reporter=reporter,
                human_app=human_app,
            )

    assert result.state["judge_clustering_trace"] == []


def test_run_human_simulation_counts_hidden_judge_tokens_as_clustering_agent_usage(
    reporter: RecordingReporter,
) -> None:
    session = simulation_runner.SessionContext(
        app=FakeStreamingApp([]),
        data_loader=mock.Mock(),
        logger=logging.getLogger("test.run_human_simulation.judge.tokens"),
        state=simulation_runner.build_initial_state(dataset_size=1),
        clustering_agent_backend="backend_v1",
        clustering_agent_backend_config=BackendV1ClusteringBackendConfig(),
        human_agent_prompt_id="baseline_v1",
        human_agent_system_prompt="Human prompt",
        topic_discovery_config=TopicDiscoveryConfig(),
        clustering_agent_usage_collector=_usage_collector(11, 13),
        human_agent_usage_collector=_usage_collector(),
        enable_intermittent_clustering=True,
    )
    human_app = FakeHumanApp(messages=[HumanMessage(content="Group by topic.")])
    judge_payload = {
        "clusters": [
            {"name": "Topic A", "members": ["alpha"]},
        ]
    }

    def _fake_run_agent_turn(
        app: simulation_runner.StreamingApp,
        state: AgentState,
        logger: logging.Logger,
        clustering_agent_session_tokens: UsageCollector,
        seed_accounting: dict[str, object] | None = None,
        reporter: SimulationReporter | None = None,
        recursion_limit: int | None = None,
        allow_graph_recursion_exhaustion: bool = False,
        state_update_callback: Callable[[AgentState], None] | None = None,
    ) -> AgentState:
        _ = app
        _ = logger
        _ = seed_accounting
        _ = reporter
        _ = recursion_limit
        _ = allow_graph_recursion_exhaustion
        _ = state_update_callback
        assert state["judge_clustering_trace"] == [{"turn_index": 1, "clustering": judge_payload}]
        assert state["intermittent_clustering_token_usage"] == {}
        assert clustering_agent_session_tokens.snapshot(pricer=None).total_tokens == 24
        state["messages"].append(AIMessage(content="Topic grouping ready."))
        state["iteration_count"] = state["iteration_count"] + 1
        return state

    with mock.patch.object(
        backend_v1_judge_module,
        "infer_backend_v1_intermittent_clustering_result",
        return_value=backend_v1_judge_module.IntermittentClusteringInferenceResult(
            clustering=judge_payload,
            token_usage={"input_tokens": 11, "output_tokens": 13, "total_tokens": 24},
        ),
    ):
        with mock.patch.object(
            simulation_runner, "run_agent_turn", side_effect=_fake_run_agent_turn
        ):
            result = simulation_runner.run_human_simulation(
                session=session,
                guidance="Keep clusters coherent",
                max_turns=1,
                reporter=reporter,
                human_app=human_app,
            )

    assert result.clustering_agent_accounting["total_tokens"] == 24
    assert reporter.tokens == [
        (
            {"input_tokens": 11, "output_tokens": 13, "total_tokens": 24},
            {"input_tokens": 11, "output_tokens": 13, "total_tokens": 24},
        )
    ]


def test_run_human_simulation_records_topic_candidates_without_consuming_turn(
    reporter: RecordingReporter,
) -> None:
    session = simulation_runner.SessionContext(
        app=FakeStreamingApp([]),
        data_loader=mock.Mock(),
        logger=logging.getLogger("test.run_human_simulation.topic_candidates"),
        state=simulation_runner.build_initial_state(dataset_size=1),
        clustering_agent_backend="backend_v1",
        clustering_agent_backend_config=BackendV1ClusteringBackendConfig(),
        human_agent_prompt_id="baseline_v4",
        human_agent_system_prompt="Human prompt",
        topic_discovery_config=TopicDiscoveryConfig(
            enabled=True,
            trigger_mode="turn_budget",
            turn_budget=0 + 1,
        ),
        turn_injection=HumanTurnInjectionConfig(
            turn_index=2,
            edit_primitive="theme_switch",
            source_theme_id="fruit",
            target_theme_id="vehicle",
        ),
        clustering_agent_usage_collector=_usage_collector(),
        human_agent_usage_collector=_usage_collector(2, 3),
    )
    first_state = simulation_runner.build_initial_state(dataset_size=1)
    first_state["topic_candidates"] = [
        {
            "topic_id": "tone",
            "label": "Tone",
            "guidance": "Group items by tone.",
            "rationale": "Tone appears repeatedly in summaries.",
        },
        {
            "topic_id": "audience",
            "label": "Audience",
            "guidance": "Group items by audience.",
            "rationale": "Audience appears repeatedly in summaries.",
        },
    ]
    first_state["topic_candidates_metadata"] = {
        "trigger_mode": "turn_budget",
        "generated_at_turn": 1,
    }
    first_state["human_turn_token_usage"] = {
        "input_tokens": 2,
        "output_tokens": 3,
        "total_tokens": 5,
    }
    first_state["iteration_count"] = 1

    second_state = {
        **first_state,
        "messages": [
            HumanMessage(content="Could you compare the items by tone?"),
        ],
    }
    human_app = mock.Mock()
    human_app.invoke.side_effect = [first_state, second_state]

    def _fake_run_agent_turn(
        app: simulation_runner.StreamingApp,
        state: AgentState,
        logger: logging.Logger,
        clustering_agent_session_tokens: UsageCollector,
        seed_accounting: dict[str, object] | None = None,
        reporter: SimulationReporter | None = None,
        recursion_limit: int | None = None,
        allow_graph_recursion_exhaustion: bool = False,
        state_update_callback: Callable[[AgentState], None] | None = None,
    ) -> AgentState:
        _ = app
        _ = logger
        _ = clustering_agent_session_tokens
        _ = seed_accounting
        _ = reporter
        _ = recursion_limit
        _ = allow_graph_recursion_exhaustion
        _ = state_update_callback
        state["messages"].append(AIMessage(content="Here is a tone-based comparison."))
        state["iteration_count"] = state["iteration_count"] + 1
        return state

    with mock.patch.object(simulation_runner, "run_agent_turn", side_effect=_fake_run_agent_turn):
        result = simulation_runner.run_human_simulation(
            session=session,
            guidance=None,
            max_turns=1,
            reporter=reporter,
            human_app=human_app,
        )

    assert result.turns_run == 1
    assert result.state["topic_candidates_metadata"] == {
        "trigger_mode": "turn_budget",
        "generated_at_turn": 1,
    }
    assert "turn_injection_consumed_turn_index" not in human_app.invoke.call_args_list[1].args[0]
    assert result.state["turn_injection_consumed_turn_index"] == 2
    assert result.human_agent_accounting["total_tokens"] == 5
    assert reporter.system_messages[0] == "System: Human agent enumerated topic candidates."


def test_run_human_simulation_stops_at_topic_discovery_handoff(
    reporter: RecordingReporter,
) -> None:
    session = simulation_runner.SessionContext(
        app=FakeStreamingApp([]),
        data_loader=mock.Mock(),
        logger=logging.getLogger("test.run_human_simulation.topic_handoff"),
        state=simulation_runner.build_initial_state(dataset_size=1),
        clustering_agent_backend="backend_v1",
        clustering_agent_backend_config=BackendV1ClusteringBackendConfig(),
        human_agent_prompt_id="baseline_v4",
        human_agent_system_prompt="Human prompt",
        topic_discovery_config=TopicDiscoveryConfig(
            enabled=True,
            trigger_mode="turn_budget",
            turn_budget=1,
        ),
        turn_injection=HumanTurnInjectionConfig(
            turn_index=2,
            edit_primitive="theme_switch",
            source_theme_id="fruit",
            target_theme_id="vehicle",
        ),
        clustering_agent_usage_collector=_usage_collector(),
        human_agent_usage_collector=_usage_collector(),
    )
    handoff_state = simulation_runner.build_initial_state(dataset_size=1)
    handoff_state["topic_candidates"] = [
        {
            "topic_id": "tone",
            "label": "Tone",
            "guidance": "Group items by tone.",
            "rationale": "Tone appears repeatedly in summaries.",
        }
    ]
    handoff_state["topic_candidates_metadata"] = {
        "trigger_mode": "turn_budget",
        "generated_at_turn": 1,
    }
    handoff_state["human_turn_token_usage"] = {
        "input_tokens": 2,
        "output_tokens": 3,
        "total_tokens": 5,
    }
    handoff_state["iteration_count"] = 1
    human_app = mock.Mock()
    human_app.invoke.return_value = handoff_state

    result = simulation_runner.run_human_simulation(
        session=session,
        guidance=None,
        max_turns=3,
        reporter=reporter,
        human_app=human_app,
        stop_on_topic_candidates=True,
    )

    assert result.turns_run == 0
    assert result.stopped_reason == "topic_discovery_handoff"
    assert result.state["turn_injection_consumed_turn_index"] == 2
    assert human_app.invoke.call_count == 1
