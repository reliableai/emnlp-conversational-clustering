"""LangGraph implementation for the oneshot_kmeans_v1 conversational clustering backend."""

from typing import Any, cast

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from ....accounting import UsageCollector, extract_response_model, extract_token_usage
from ....data_processing.data_loader import DataLoader
from ....models import build_clustering_agent_llm
from ...state import AgentState
from ..base import ConversationalClusteringMethod
from .config import ONESHOT_KMEANS_V1_IDENTIFIER, OneShotKMeansV1ClusteringBackendConfig
from .prompts import get_oneshot_kmeans_v1_system_prompt
from .tools import build_tools


class OneShotKMeansV1ConversationalClusteringMethod(ConversationalClusteringMethod):
    def __init__(self, config: OneShotKMeansV1ClusteringBackendConfig):
        self._config = config

    @property
    def backend(self) -> str:
        return ONESHOT_KMEANS_V1_IDENTIFIER

    def resolve_app(
        self,
        *,
        data_loader: DataLoader | None,
        save_path: str | None = None,
        usage_collector: UsageCollector | None = None,
    ) -> object:
        usage_collector = usage_collector or UsageCollector()
        if data_loader is None:
            raise ValueError("oneshot_kmeans_v1 requires a dataset-backed data loader.")
        if not data_loader.has_embeddings():
            raise ValueError("oneshot_kmeans_v1 requires embeddings to be enabled.")

        system_prompt = get_oneshot_kmeans_v1_system_prompt()
        llm = build_clustering_agent_llm(self._config.model_config)
        tools = build_tools(data_loader, save_path=save_path)
        llm_with_tools = llm.bind_tools(tools)

        def agent_node(state: AgentState) -> dict:
            messages = state["messages"]
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=system_prompt), *messages]
            dataset_metadata = state.get("dataset_metadata") or {}
            dataset_size = dataset_metadata.get("size")
            if dataset_size is not None:
                metadata_message = SystemMessage(
                    content=(
                        f"Dataset metadata: size={dataset_size} items, embeddings_loaded=true."
                    )
                )
                messages = [messages[0], metadata_message, *messages[1:]]
            response = llm_with_tools.invoke(messages)
            usage_collector.record(
                extract_response_model(response, fallback_model=self._config.model_config.model),
                extract_token_usage(response),
                source="oneshot_kmeans_v1_agent",
            )
            return {"messages": [response]}

        tool_node = ToolNode(tools)

        def should_continue(state: AgentState) -> str:
            last_message = state["messages"][-1]
            if getattr(last_message, "tool_calls", None):
                return "tools"
            return END

        graph = StateGraph(cast(type[Any], AgentState))
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", should_continue)
        graph.add_edge("tools", "agent")
        return graph.compile()
