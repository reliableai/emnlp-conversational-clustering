"""Session management for simulating agent-human conversations."""

import logging
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from ..accounting import (
    OpenRouterPricingClient,
    UsageCollector,
    build_pricing_client_from_configs,
    merge_usage_summaries,
)
from ..clustering_agent.backends import (
    DEFAULT_CLUSTERING_AGENT_BACKEND,
    ClusteringAgentBackendConfig,
    ConversationalClusteringMethod,
    build_conversational_clustering_method,
    default_clustering_agent_backend_config,
)
from ..clustering_agent.backends.pov_bank import coerce_clustering_pov_history
from ..clustering_agent.state import AgentState
from ..data_processing import DATASETS
from ..data_processing.csv_loader import CsvDataLoader
from ..data_processing.data_loader import DataLoader
from ..human_agent import build_human_app
from ..human_turn_injection import HumanTurnInjectionConfig
from ..logging_utils import configure_logger, extract_token_usage, log_json
from ..models import (
    EmbeddingModelConfig,
    ModelConfig,
    default_human_model_config,
    default_human_verifier_model_config,
)
from ..prompts import (
    get_human_agent_system_prompt,
    get_human_verifier_system_prompt,
    validate_human_agent_prompt_guidance,
)
from ..topic_discovery import TopicDiscoveryConfig
from .reporters import NullReporter, SimulationReporter

_HUMAN_THOUGHT_KEY = "human_thought"
_FINAL_TURN_AGENT_RECURSION_LIMIT = 15
_FINAL_TURN_AGENT_RECURSION_STOP_REASON = "final_turn_recursion_limit"


@dataclass
class SessionContext:
    app: "StreamingApp"
    data_loader: DataLoader
    logger: logging.Logger
    state: AgentState
    clustering_agent_backend: str
    clustering_agent_backend_config: ClusteringAgentBackendConfig
    human_agent_prompt_id: str
    human_agent_system_prompt: str
    topic_discovery_config: TopicDiscoveryConfig
    clustering_agent_usage_collector: UsageCollector = field(default_factory=UsageCollector)
    human_agent_usage_collector: UsageCollector = field(default_factory=UsageCollector)
    pricing_client: OpenRouterPricingClient = field(
        default_factory=lambda: OpenRouterPricingClient(api_key_envs=())
    )
    seed_clustering_agent_accounting: dict[str, Any] | None = None
    seed_human_agent_accounting: dict[str, Any] | None = None
    human_agent_persona: str | None = None
    turn_injection: HumanTurnInjectionConfig | None = None
    selected_topic_label: str | None = None
    selected_topic_guidance: str | None = None
    selected_topic_generated_at_turn: int | None = None
    human_verifier_prompt_id: str = "baseline_v1"
    max_verification_retries: int = 2
    human_verifier_system_prompt: str = ""
    human_agent_model_config: ModelConfig | None = None
    human_verifier_model_config: ModelConfig | None = None
    enable_intermittent_clustering: bool = False
    intermittent_clustering_judge_save_path: str | None = None
    clustering_method: ConversationalClusteringMethod | None = None


@dataclass
class SimulationResult:
    state: AgentState
    clustering_agent_accounting: dict[str, Any]
    human_agent_accounting: dict[str, Any]
    turns_run: int
    stopped_reason: str


class StreamingApp(Protocol):
    def stream(
        self,
        state: AgentState,
        config: dict[str, int] | None = None,
        *,
        stream_mode: list[str],
    ) -> Iterable[tuple[str, object]]: ...


class InvokableApp(Protocol):
    def invoke(self, state: AgentState) -> AgentState: ...


def build_data_loader(
    dataset_name: str,
    limit: int | None = None,
    random_seed: int | None = None,
    enable_embeddings: bool = False,
    embedding_model_config: EmbeddingModelConfig | None = None,
    embedding_force_recompute: bool = False,
    embedding_batch_size: int = 64,
    usage_collector: UsageCollector | None = None,
) -> CsvDataLoader:
    if dataset_name not in DATASETS:
        valid = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose one of: {valid}")
    config = DATASETS[dataset_name]
    dataset_path = config["path"]
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset file for '{dataset_name}' was not found at '{dataset_path}'."
        )
    return CsvDataLoader(
        dataset_path,
        item_column=config["item_column"],
        limit=limit,
        random_seed=random_seed,
        dataset_name=dataset_name,
        enable_embeddings=enable_embeddings,
        embedding_model_config=embedding_model_config,
        embedding_force_recompute=embedding_force_recompute,
        embedding_batch_size=embedding_batch_size,
        usage_collector=usage_collector,
    )


def build_initial_state(dataset_size: int) -> AgentState:
    return {
        "messages": [],
        "dataset_metadata": {"size": dataset_size},
        "iteration_count": 0,
        "human_thought_trace": [],
        "judge_clustering_trace": [],
    }


def _collect_pricing_configs(
    *,
    clustering_agent_backend_config: ClusteringAgentBackendConfig,
    embedding_model_config: EmbeddingModelConfig | None,
    human_agent_model_config: ModelConfig | None,
    human_verifier_model_config: ModelConfig | None,
) -> list[object]:
    model_configs: list[object] = []
    for maybe_model_config in (
        human_agent_model_config,
        human_verifier_model_config,
        embedding_model_config,
    ):
        if maybe_model_config is not None:
            model_configs.append(maybe_model_config)
    for value in vars(clustering_agent_backend_config).values():
        if isinstance(value, ModelConfig | EmbeddingModelConfig):
            model_configs.append(value)
    return model_configs


def build_intermittent_clustering_judge_save_path(
    clustering_save_path: str | None,
    *,
    clustering_agent_backend: str = DEFAULT_CLUSTERING_AGENT_BACKEND,
    clustering_agent_backend_config: ClusteringAgentBackendConfig | None = None,
) -> str:
    method = build_conversational_clustering_method(
        backend=clustering_agent_backend,
        backend_config=clustering_agent_backend_config,
    )
    resolved_path = method.build_intermittent_clustering_save_path(
        clustering_save_path=clustering_save_path
    )
    if resolved_path is None:
        raise ValueError(
            f"Backend '{clustering_agent_backend}' does not define intermittent snapshots."
        )
    return resolved_path


def maybe_build_intermittent_clustering_judge_save_path(
    clustering_save_path: str | None,
    *,
    clustering_agent_backend: str = DEFAULT_CLUSTERING_AGENT_BACKEND,
    clustering_agent_backend_config: ClusteringAgentBackendConfig | None = None,
) -> str | None:
    method = build_conversational_clustering_method(
        backend=clustering_agent_backend,
        backend_config=clustering_agent_backend_config,
    )
    return method.build_intermittent_clustering_save_path(clustering_save_path=clustering_save_path)


def create_session(
    dataset_name: str,
    limit: int | None = None,
    random_seed: int | None = None,
    clustering_agent_backend: str = DEFAULT_CLUSTERING_AGENT_BACKEND,
    clustering_agent_backend_config: ClusteringAgentBackendConfig | None = None,
    human_agent_model_config: ModelConfig | None = None,
    human_verifier_model_config: ModelConfig | None = None,
    enable_embeddings: bool = False,
    embedding_model_config: EmbeddingModelConfig | None = None,
    embedding_force_recompute: bool = False,
    embedding_batch_size: int = 64,
    human_agent_prompt_id: str = "baseline_v3",
    human_agent_persona: str | None = None,
    human_verifier_prompt_id: str = "baseline_v1",
    max_verification_retries: int = 2,
    topic_discovery_config: TopicDiscoveryConfig | None = None,
    turn_injection: HumanTurnInjectionConfig | None = None,
    selected_topic_label: str | None = None,
    selected_topic_guidance: str | None = None,
    selected_topic_generated_at_turn: int | None = None,
    enable_intermittent_clustering: bool = False,
    clustering_save_path: str | None = None,
    intermittent_clustering_judge_save_path: str | None = None,
    log_path: str | None = None,
    logger: logging.Logger | None = None,
) -> SessionContext:
    clustering_agent_usage_collector = UsageCollector()
    human_agent_usage_collector = UsageCollector()
    data_loader = build_data_loader(
        dataset_name,
        limit=limit,
        random_seed=random_seed,
        enable_embeddings=enable_embeddings,
        embedding_model_config=embedding_model_config,
        embedding_force_recompute=embedding_force_recompute,
        embedding_batch_size=embedding_batch_size,
        usage_collector=clustering_agent_usage_collector,
    )
    if clustering_agent_backend_config is None:
        clustering_agent_backend_config = default_clustering_agent_backend_config(
            clustering_agent_backend
        )
    clustering_method = build_conversational_clustering_method(
        backend=clustering_agent_backend,
        backend_config=clustering_agent_backend_config,
    )
    resolved_human_agent_model_config = human_agent_model_config or default_human_model_config()
    resolved_human_verifier_model_config = (
        human_verifier_model_config or default_human_verifier_model_config()
    )
    pricing_client = build_pricing_client_from_configs(
        _collect_pricing_configs(
            clustering_agent_backend_config=clustering_agent_backend_config,
            embedding_model_config=embedding_model_config,
            human_agent_model_config=resolved_human_agent_model_config,
            human_verifier_model_config=resolved_human_verifier_model_config,
        )
    )
    human_agent_system_prompt = get_human_agent_system_prompt(
        human_agent_prompt_id,
        persona=human_agent_persona,
    )
    human_verifier_system_prompt = get_human_verifier_system_prompt(human_verifier_prompt_id)
    app = cast(
        StreamingApp,
        clustering_method.resolve_app(
            data_loader=data_loader,
            save_path=clustering_save_path,
            usage_collector=clustering_agent_usage_collector,
        ),
    )
    if logger is None:
        logger = configure_logger(log_path) if log_path else configure_logger()
    dataset_size = len(data_loader.load())
    state = build_initial_state(dataset_size)
    resolved_topic_discovery_config = topic_discovery_config or TopicDiscoveryConfig()
    return SessionContext(
        app=app,
        data_loader=data_loader,
        logger=logger,
        state=state,
        clustering_agent_backend=clustering_agent_backend,
        clustering_agent_backend_config=clustering_agent_backend_config,
        clustering_method=clustering_method,
        human_agent_prompt_id=human_agent_prompt_id,
        human_agent_persona=human_agent_persona,
        human_verifier_prompt_id=human_verifier_prompt_id,
        max_verification_retries=max(0, max_verification_retries),
        human_agent_system_prompt=human_agent_system_prompt,
        topic_discovery_config=resolved_topic_discovery_config,
        turn_injection=turn_injection,
        selected_topic_label=selected_topic_label,
        selected_topic_guidance=selected_topic_guidance,
        selected_topic_generated_at_turn=selected_topic_generated_at_turn,
        human_verifier_system_prompt=human_verifier_system_prompt,
        human_agent_model_config=resolved_human_agent_model_config,
        human_verifier_model_config=resolved_human_verifier_model_config,
        enable_intermittent_clustering=enable_intermittent_clustering,
        clustering_agent_usage_collector=clustering_agent_usage_collector,
        human_agent_usage_collector=human_agent_usage_collector,
        pricing_client=pricing_client,
        intermittent_clustering_judge_save_path=intermittent_clustering_judge_save_path,
    )


def _coerce_reporter(
    reporter: SimulationReporter | None,
) -> SimulationReporter:
    return reporter or NullReporter()


def _should_stream_agent_messages(reporter: SimulationReporter) -> bool:
    return type(reporter) is not NullReporter


def _with_total(session_tokens: dict[str, Any]) -> dict[str, int]:
    input_tokens = int(session_tokens.get("input_tokens", 0))
    output_tokens = int(session_tokens.get("output_tokens", 0))
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _merged_accounting_snapshot(
    seed_accounting: dict[str, Any] | None,
    live_accounting: dict[str, Any],
) -> dict[str, Any]:
    return merge_usage_summaries((seed_accounting, live_accounting)).to_dict()


def _session_token_snapshot(
    seed_accounting: dict[str, Any] | None,
    collector: UsageCollector,
) -> dict[str, int]:
    merged_accounting = _merged_accounting_snapshot(
        seed_accounting,
        collector.snapshot(pricer=None).to_dict(),
    )
    return _with_total(merged_accounting)


def _session_accounting_snapshot(
    seed_accounting: dict[str, Any] | None,
    collector: UsageCollector,
    pricing_client: OpenRouterPricingClient,
) -> dict[str, Any]:
    return _merged_accounting_snapshot(
        seed_accounting,
        collector.snapshot(pricer=pricing_client).to_dict(),
    )


def _sync_session_state(session: SessionContext, state: AgentState) -> None:
    session.state = state


def run_agent_turn(
    app: StreamingApp,
    state: AgentState,
    logger: logging.Logger,
    clustering_agent_usage_collector: UsageCollector,
    seed_accounting: dict[str, Any] | None = None,
    reporter: SimulationReporter | None = None,
    recursion_limit: int | None = None,
    allow_graph_recursion_exhaustion: bool = False,
    state_update_callback: Callable[[AgentState], None] | None = None,
) -> AgentState:
    reporter = _coerce_reporter(reporter)
    stream_mode = ["messages", "values"] if _should_stream_agent_messages(reporter) else ["values"]
    latest_state = state
    latest_state.pop("final_turn_recursion_limit_reached", None)
    streaming_agent = False
    stream_config = {"recursion_limit": recursion_limit} if recursion_limit is not None else None
    try:
        stream = app.stream(state, config=stream_config, stream_mode=stream_mode)
        announced_tool_call_ids: set[str] = set()
        for mode, payload in stream:
            if mode == "messages":
                if not isinstance(payload, tuple) or len(payload) != 2:
                    continue
                message, _metadata = payload
                if isinstance(message, AIMessageChunk):
                    if getattr(message, "tool_call_chunks", None):
                        continue
                    content = message.content
                    if isinstance(content, str) and content:
                        reporter.on_agent_chunk(content, is_start=not streaming_agent)
                        streaming_agent = True
                continue

            if mode != "values" or not isinstance(payload, dict):
                continue

            latest_state = cast(AgentState, payload)
            if state_update_callback is not None:
                state_update_callback(latest_state)
            payload_messages = latest_state["messages"]
            if not isinstance(payload_messages, list) or not payload_messages:
                continue
            last_message = payload_messages[-1]
            if isinstance(last_message, ToolMessage):
                tool_name = last_message.name or "unknown"
                tool_call_id = str(last_message.tool_call_id or "").strip()
                if (
                    tool_call_id and tool_call_id not in announced_tool_call_ids
                ) or not tool_call_id:
                    reporter.on_tool_call(tool_name)
                    if tool_call_id:
                        announced_tool_call_ids.add(tool_call_id)
                tool_is_error = _tool_message_is_error(last_message)
                reporter.on_tool_result(
                    tool_name,
                    _content_to_text(last_message.content),
                    is_error=tool_is_error,
                    state=latest_state,
                )
                if tool_is_error:
                    reporter.on_tool_error(tool_name)
                log_json(
                    logger,
                    "tool_response",
                    {
                        "tool_call_id": last_message.tool_call_id,
                        "tool_name": last_message.name,
                        "content": last_message.content,
                    },
                )
            elif isinstance(last_message, AIMessage):
                tool_calls = getattr(last_message, "tool_calls", None)
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        tool_name = _extract_tool_call_name(tool_call)
                        if tool_name is None:
                            continue
                        tool_call_id = _extract_tool_call_id(tool_call)
                        if tool_call_id is not None and tool_call_id in announced_tool_call_ids:
                            continue
                        reporter.on_tool_call(tool_name)
                        if tool_call_id is not None:
                            announced_tool_call_ids.add(tool_call_id)
                reporter.on_agent_message(
                    _content_to_text(last_message.content), streamed=streaming_agent
                )
                token_usage = extract_token_usage(last_message)
                if token_usage:
                    session_token_usage = _session_token_snapshot(
                        seed_accounting,
                        clustering_agent_usage_collector,
                    )
                    reporter.on_tokens(token_usage, session_token_usage)
                log_json(
                    logger,
                    "agent_response",
                    {
                        "content": last_message.content,
                        "tool_calls": getattr(last_message, "tool_calls", None),
                        "token_usage": token_usage,
                        "session_token_usage": _session_token_snapshot(
                            seed_accounting,
                            clustering_agent_usage_collector,
                        ),
                    },
                )
    except GraphRecursionError:
        if not allow_graph_recursion_exhaustion:
            raise
        reporter.on_system_message(
            "System: Final clustering-agent turn reached the execution step limit. "
            "Ending the run with the latest available state."
        )
        latest_state["final_turn_recursion_limit_reached"] = True
        log_json(
            logger,
            "agent_turn_recursion_limit_reached",
            {
                "iteration": latest_state.get("iteration_count", 0),
                "recursion_limit": recursion_limit,
            },
        )

    latest_state["iteration_count"] = latest_state["iteration_count"] + 1
    if state_update_callback is not None:
        state_update_callback(latest_state)
    return latest_state


def _clustering_saved(messages) -> bool:
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == "save_clustering":
            content = message.content or ""
            return isinstance(content, str) and "Clustering saved" in content
    return False


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    return str(content)


def _extract_tool_call_name(raw_tool_call: object) -> str | None:
    if isinstance(raw_tool_call, Mapping):
        tool_call = cast(Mapping[str, object], raw_tool_call)
        raw_name = tool_call.get("name")
    else:
        raw_name = getattr(raw_tool_call, "name", None)
    if not isinstance(raw_name, str):
        return None
    stripped = raw_name.strip()
    return stripped or None


def _extract_tool_call_id(raw_tool_call: object) -> str | None:
    if isinstance(raw_tool_call, Mapping):
        tool_call = cast(Mapping[str, object], raw_tool_call)
        raw_id = tool_call.get("id")
    else:
        raw_id = getattr(raw_tool_call, "id", None)
    if not isinstance(raw_id, str):
        return None
    stripped = raw_id.strip()
    return stripped or None


def _tool_message_is_error(message: ToolMessage) -> bool:
    tool_status = getattr(message, "status", None)
    content = message.content or ""
    return tool_status in {"error", "failed"} or (
        isinstance(content, str)
        and content.strip()
        .lower()
        .startswith(("could not", "error", "exception", "failed", "traceback"))
    )


def _sanitize_simulated_human_message(
    message: HumanMessage,
) -> tuple[HumanMessage, str | None]:
    additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
    raw_human_thought = additional_kwargs.pop(_HUMAN_THOUGHT_KEY, None)
    human_thought: str | None = None
    if isinstance(raw_human_thought, str):
        stripped_human_thought = raw_human_thought.strip()
        if stripped_human_thought:
            human_thought = stripped_human_thought
    if raw_human_thought is None:
        return message, None
    if additional_kwargs:
        return HumanMessage(
            content=message.content, additional_kwargs=additional_kwargs
        ), human_thought
    return HumanMessage(content=message.content), human_thought


def _append_human_thought_trace_entry(
    state: AgentState,
    *,
    turn_index: int,
    message: str,
    thought: str,
) -> None:
    trace = state.setdefault("human_thought_trace", [])
    trace.append(
        {
            "turn_index": turn_index,
            "message": message,
            "thought": thought,
        }
    )


def _current_iteration_count(state: AgentState) -> int:
    raw_iteration_count = state.get("iteration_count", 0)
    if isinstance(raw_iteration_count, bool) or not isinstance(raw_iteration_count, int):
        return 0
    return max(0, raw_iteration_count)


def _mark_turn_injection_consumed(
    state: AgentState,
    *,
    turn_injection: HumanTurnInjectionConfig | None,
    turn_index: int,
) -> None:
    if turn_injection is None:
        return
    if turn_index != turn_injection.turn_index:
        return
    state["turn_injection_consumed_turn_index"] = turn_index


def _maybe_capture_intermittent_clustering_token_usage(
    session: SessionContext,
    state: AgentState,
    reporter: SimulationReporter,
) -> AgentState:
    token_usage = state.get("intermittent_clustering_token_usage")
    if not token_usage:
        return state
    session_token_usage = _session_token_snapshot(
        session.seed_clustering_agent_accounting,
        session.clustering_agent_usage_collector,
    )
    reporter.on_tokens(token_usage, session_token_usage)
    log_json(
        session.logger,
        "intermittent_clustering_judge_token_usage",
        {
            "token_usage": token_usage,
            "session_token_usage": session_token_usage,
        },
    )
    state["intermittent_clustering_token_usage"] = {}
    return state


def build_conversation_trace_artifact(state: AgentState) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    total_tool_calls_by_type: dict[str, int] = {}
    total_failed_tool_calls_by_type: dict[str, int] = {}

    for message in state.get("messages", []):
        if isinstance(message, ToolMessage):
            tool_name = str(message.name or "unknown").strip() or "unknown"
            total_tool_calls_by_type[tool_name] = total_tool_calls_by_type.get(tool_name, 0) + 1
            if _tool_message_is_error(message):
                total_failed_tool_calls_by_type[tool_name] = (
                    total_failed_tool_calls_by_type.get(tool_name, 0) + 1
                )
            continue

        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            role = "system"
        content = _content_to_text(getattr(message, "content", ""))
        if content.strip():
            messages.append({"role": role, "content": content})

    return {
        "messages": messages,
        "total_tool_calls_by_type": {
            name: total_tool_calls_by_type[name] for name in sorted(total_tool_calls_by_type)
        },
        "total_failed_tool_calls_by_type": {
            name: total_failed_tool_calls_by_type[name]
            for name in sorted(total_failed_tool_calls_by_type)
        },
    }


def build_human_thought_trace_artifact(state: AgentState) -> dict[str, Any]:
    messages: list[dict[str, str | int]] = []
    for raw_message in state.get("human_thought_trace", []):
        if not isinstance(raw_message, dict):
            continue
        raw_turn_index = raw_message.get("turn_index")
        raw_message_text = raw_message.get("message")
        raw_thought = raw_message.get("thought")
        if isinstance(raw_turn_index, bool) or not isinstance(raw_turn_index, int):
            continue
        if not isinstance(raw_message_text, str) or not raw_message_text.strip():
            continue
        if not isinstance(raw_thought, str) or not raw_thought.strip():
            continue
        messages.append(
            {
                "turn_index": raw_turn_index,
                "message": raw_message_text,
                "thought": raw_thought,
            }
        )
    return {
        "messages": messages,
        "num_messages": len(messages),
    }


def build_judge_clustering_trace_artifact(state: AgentState) -> dict[str, Any] | None:
    snapshots: list[dict[str, object]] = []
    for raw_entry in state.get("judge_clustering_trace", []):
        if not isinstance(raw_entry, dict):
            continue
        raw_turn_index = raw_entry.get("turn_index")
        raw_clustering = raw_entry.get("clustering")
        if isinstance(raw_turn_index, bool) or not isinstance(raw_turn_index, int):
            continue
        if not isinstance(raw_clustering, dict):
            continue
        normalized_clustering = cast(dict[str, object], raw_clustering)
        raw_clusters = normalized_clustering.get("clusters")
        if not isinstance(raw_clusters, list) or not raw_clusters:
            continue
        snapshots.append(
            {
                "turn_index": raw_turn_index,
                "clustering": normalized_clustering,
            }
        )
    if not snapshots:
        return None
    return {
        "clusterings": snapshots,
        "num_clusterings": len(snapshots),
    }


def build_clustering_pov_history_artifact(state: AgentState) -> dict[str, Any] | None:
    snapshots: list[dict[str, object]] = []
    for raw_entry in coerce_clustering_pov_history(state.get("clustering_pov_history")):
        raw_turn_index = raw_entry.get("turn_index")
        raw_povs = raw_entry.get("povs")
        if isinstance(raw_turn_index, bool) or not isinstance(raw_turn_index, int):
            continue
        if raw_turn_index < 1:
            continue
        if not isinstance(raw_povs, list):
            continue
        normalized_povs = [pov for pov in raw_povs if isinstance(pov, str)]
        if not normalized_povs:
            continue
        snapshots.append(
            {
                "turn_index": raw_turn_index,
                "povs": normalized_povs,
                "num_povs": len(normalized_povs),
            }
        )

    if not snapshots:
        return None
    return {
        "snapshots": snapshots,
        "num_snapshots": len(snapshots),
    }


def build_topic_candidates_artifact(state: AgentState) -> dict[str, Any] | None:
    raw_topics = state.get("topic_candidates")
    if not isinstance(raw_topics, list) or not raw_topics:
        return None

    topics: list[dict[str, str]] = []
    for raw_topic in raw_topics:
        if not isinstance(raw_topic, dict):
            continue
        raw_topic_id = raw_topic.get("topic_id")
        raw_label = raw_topic.get("label")
        raw_guidance = raw_topic.get("guidance")
        raw_rationale = raw_topic.get("rationale")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (raw_topic_id, raw_label, raw_guidance, raw_rationale)
        ):
            continue
        assert isinstance(raw_topic_id, str)
        assert isinstance(raw_label, str)
        assert isinstance(raw_guidance, str)
        assert isinstance(raw_rationale, str)
        topic_id = raw_topic_id.strip()
        label = raw_label.strip()
        guidance = raw_guidance.strip()
        rationale = raw_rationale.strip()
        topics.append(
            {
                "topic_id": topic_id,
                "label": label,
                "guidance": guidance,
                "rationale": rationale,
            }
        )

    if not topics:
        return None

    raw_metadata = state.get("topic_candidates_metadata", {})
    if isinstance(raw_metadata, dict):
        metadata = raw_metadata
    else:
        metadata = {}
    raw_trigger_mode = metadata.get("trigger_mode")
    trigger_mode = raw_trigger_mode if isinstance(raw_trigger_mode, str) else "unknown"
    raw_generated_at_turn = metadata.get("generated_at_turn")
    generated_at_turn = (
        raw_generated_at_turn
        if isinstance(raw_generated_at_turn, int) and not isinstance(raw_generated_at_turn, bool)
        else None
    )
    artifact: dict[str, Any] = {
        "trigger_mode": trigger_mode,
        "generated_at_turn": generated_at_turn,
        "num_topics": len(topics),
        "topics": topics,
    }
    raw_num_rounds = metadata.get("num_rounds")
    if isinstance(raw_num_rounds, int) and not isinstance(raw_num_rounds, bool):
        artifact["num_rounds"] = raw_num_rounds
    raw_new_topics_by_round = metadata.get("new_topics_by_round")
    if isinstance(raw_new_topics_by_round, list) and all(
        isinstance(value, int) and not isinstance(value, bool) for value in raw_new_topics_by_round
    ):
        artifact["new_topics_by_round"] = raw_new_topics_by_round
    return artifact


def _topic_candidates_generated(state: AgentState) -> bool:
    raw_topics = state.get("topic_candidates")
    return isinstance(raw_topics, list) and bool(raw_topics)


def _should_force_topic_generation(
    state: AgentState,
    topic_discovery_config: TopicDiscoveryConfig,
) -> bool:
    if not topic_discovery_config.enabled:
        return False
    if topic_discovery_config.trigger_mode != "turn_budget":
        return False
    if _topic_candidates_generated(state):
        return False
    turn_budget = topic_discovery_config.turn_budget
    if turn_budget is None:
        return False
    return _current_iteration_count(state) >= turn_budget


def _maybe_capture_intermittent_clustering_judge(
    session: SessionContext,
    state: AgentState,
) -> AgentState:
    if not session.enable_intermittent_clustering:
        return state
    return _resolve_clustering_method(session).capture_intermittent_clustering(
        state=state,
        data_loader=session.data_loader,
        logger=session.logger,
        output_path=session.intermittent_clustering_judge_save_path,
    )


def _resolve_clustering_method(session: SessionContext) -> ConversationalClusteringMethod:
    if session.clustering_method is None:
        session.clustering_method = build_conversational_clustering_method(
            backend=session.clustering_agent_backend,
            backend_config=session.clustering_agent_backend_config,
        )
    return session.clustering_method


def run_interactive_session(
    session: SessionContext,
    reporter: SimulationReporter | None = None,
    input_fn: Callable[[str], str] = input,
    prompt: str = "User: ",
) -> SessionContext:
    reporter = _coerce_reporter(reporter)
    state = session.state
    while True:
        try:
            user_input = input_fn(prompt).strip()
        except EOFError:
            reporter.on_system_message("")
            break

        if not user_input:
            continue

        state["messages"].append(HumanMessage(content=user_input))
        log_json(
            session.logger,
            "user_message",
            {"content": user_input, "iteration": state.get("iteration_count", 0)},
        )
        state = _maybe_capture_intermittent_clustering_judge(session, state)
        state = _maybe_capture_intermittent_clustering_token_usage(session, state, reporter)

        state = run_agent_turn(
            session.app,
            state,
            session.logger,
            session.clustering_agent_usage_collector,
            seed_accounting=session.seed_clustering_agent_accounting,
            reporter=reporter,
        )

    session.state = state
    return session


def run_human_simulation(
    session: SessionContext,
    guidance: str | None,
    max_turns: int = 8,
    reporter: SimulationReporter | None = None,
    human_app: InvokableApp | None = None,
    stop_on_topic_candidates: bool = False,
) -> SimulationResult:
    """
    Run conversational clustering system with simulated human agent.
    If `reporter` is not passed, there will be no terminal output
    """

    reporter = _coerce_reporter(reporter)
    if max_turns <= 0:
        max_turns = 1
    guidance = validate_human_agent_prompt_guidance(
        prompt_id=session.human_agent_prompt_id,
        guidance=guidance,
    )
    reporter.on_simulation_start(max_turns, guidance)

    if human_app is None:
        human_verifier_system_prompt = session.human_verifier_system_prompt or None
        human_app = cast(
            InvokableApp,
            build_human_app(
                guidance=guidance,
                human_agent_model_config=session.human_agent_model_config,
                system_prompt=session.human_agent_system_prompt,
                verifier_model_config=session.human_verifier_model_config,
                verifier_system_prompt=human_verifier_system_prompt,
                max_verification_retries=session.max_verification_retries,
                logger=session.logger,
                usage_collector=session.human_agent_usage_collector,
                topic_discovery_config=session.topic_discovery_config,
                turn_injection=session.turn_injection,
                selected_topic_label=session.selected_topic_label,
                selected_topic_guidance=session.selected_topic_guidance,
                selected_topic_generated_at_turn=session.selected_topic_generated_at_turn,
            ),
        )

    state = session.state
    stopped_reason = "max_turns"
    turns_run = 0

    while turns_run < max_turns:
        if _should_force_topic_generation(state, session.topic_discovery_config):
            state["topic_discovery_force_generation"] = True
        else:
            state["topic_discovery_force_generation"] = False
        topics_previously_generated = _topic_candidates_generated(state)
        human_state = human_app.invoke(state)
        human_messages = human_state.get("messages", [])
        state = human_state
        _sync_session_state(session, state)
        if _topic_candidates_generated(state) and not topics_previously_generated:
            topic_token_usage = state.get("human_turn_token_usage")
            if topic_token_usage:
                session_token_usage = _session_token_snapshot(
                    session.seed_human_agent_accounting,
                    session.human_agent_usage_collector,
                )
                reporter.on_human_tokens(topic_token_usage, session_token_usage)
                log_json(
                    session.logger,
                    "human_token_usage",
                    {
                        "token_usage": topic_token_usage,
                        "session_token_usage": session_token_usage,
                    },
                )
                state["human_turn_token_usage"] = {}
            reporter.on_system_message("System: Human agent enumerated topic candidates.")
            if stop_on_topic_candidates:
                _mark_turn_injection_consumed(
                    state,
                    turn_injection=session.turn_injection,
                    turn_index=_current_iteration_count(state) + 1,
                )
            _sync_session_state(session, state)
            if stop_on_topic_candidates:
                stopped_reason = "topic_discovery_handoff"
                break
            continue
        if not human_messages:
            reporter.on_system_message(
                "System: Human agent did not produce a message. Stopping simulation."
            )
            stopped_reason = "no_human_message"
            break
        human_message = human_messages[-1]
        if not isinstance(human_message, HumanMessage):
            human_message = HumanMessage(
                content=str(getattr(human_message, "content", human_message))
            )

        human_message, human_thought = _sanitize_simulated_human_message(human_message)
        human_message_text = _content_to_text(human_message.content)
        turn_index = _current_iteration_count(state) + 1
        reporter.on_human_message(human_message_text, simulated=True)
        if human_thought is not None:
            _append_human_thought_trace_entry(
                state,
                turn_index=turn_index,
                message=human_message_text,
                thought=human_thought,
            )
        state_messages = state.setdefault("messages", [])
        if state_messages and state_messages[-1] is human_messages[-1]:
            state_messages[-1] = human_message
        else:
            state_messages.append(human_message)
        token_usage = extract_token_usage(human_message)
        if token_usage:
            session_token_usage = _session_token_snapshot(
                session.seed_human_agent_accounting,
                session.human_agent_usage_collector,
            )
            reporter.on_human_tokens(token_usage, session_token_usage)
            log_json(
                session.logger,
                "human_token_usage",
                {
                    "token_usage": token_usage,
                    "session_token_usage": session_token_usage,
                },
            )
        else:
            reporter.on_system_message("System: Human token usage unavailable for last response.")

        log_json(
            session.logger,
            "simulated_user_message",
            {
                "content": human_message_text,
                "iteration": state.get("iteration_count", 0),
                "has_human_thought": human_thought is not None,
                "human_thought": human_thought,
            },
        )
        state = _maybe_capture_intermittent_clustering_judge(session, state)
        state = _maybe_capture_intermittent_clustering_token_usage(session, state, reporter)
        _sync_session_state(session, state)

        state = run_agent_turn(
            session.app,
            state,
            session.logger,
            session.clustering_agent_usage_collector,
            seed_accounting=session.seed_clustering_agent_accounting,
            reporter=reporter,
            recursion_limit=(
                _FINAL_TURN_AGENT_RECURSION_LIMIT if turns_run + 1 == max_turns else None
            ),
            allow_graph_recursion_exhaustion=turns_run + 1 == max_turns,
            state_update_callback=lambda latest_state: _sync_session_state(session, latest_state),
        )
        _mark_turn_injection_consumed(
            state,
            turn_injection=session.turn_injection,
            turn_index=turn_index,
        )
        _sync_session_state(session, state)
        turns_run += 1
        if _clustering_saved(state.get("messages", [])):
            reporter.on_system_message("System: Clustering saved. Ending simulation.")
            stopped_reason = "clustering_saved"
            break
        if state.get("final_turn_recursion_limit_reached"):
            stopped_reason = _FINAL_TURN_AGENT_RECURSION_STOP_REASON
            break

    _sync_session_state(session, state)
    clustering_agent_session_tokens = _session_token_snapshot(
        session.seed_clustering_agent_accounting,
        session.clustering_agent_usage_collector,
    )
    human_agent_session_tokens = _session_token_snapshot(
        session.seed_human_agent_accounting,
        session.human_agent_usage_collector,
    )
    reporter.on_simulation_end(
        stopped_reason,
        clustering_agent_session_tokens=clustering_agent_session_tokens,
        human_agent_session_tokens=human_agent_session_tokens,
    )
    return SimulationResult(
        state=state,
        clustering_agent_accounting=_session_accounting_snapshot(
            session.seed_clustering_agent_accounting,
            session.clustering_agent_usage_collector,
            session.pricing_client,
        ),
        human_agent_accounting=_session_accounting_snapshot(
            session.seed_human_agent_accounting,
            session.human_agent_usage_collector,
            session.pricing_client,
        ),
        turns_run=turns_run,
        stopped_reason=stopped_reason,
    )
