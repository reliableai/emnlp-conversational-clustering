"""State for agents."""

from typing import Annotated, NotRequired

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class InstructionKMeansRuntimeStateSnapshot(TypedDict, total=False):
    latest_proposal: dict[str, object] | None
    latest_proposal_artifacts: dict[str, object] | None
    proposals_by_id: dict[str, dict[str, object]]
    clustering_pov_bank: list[str]
    next_proposal_index: int


class BackendV1RuntimeStateSnapshot(TypedDict, total=False):
    clustering_pov_bank: list[str]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    dataset_metadata: dict
    iteration_count: int
    final_turn_recursion_limit_reached: NotRequired[bool]
    latest_proposed_clustering: NotRequired[dict[str, object]]
    backend_v1_runtime_state: NotRequired[BackendV1RuntimeStateSnapshot]
    instruction_kmeans_runtime_state: NotRequired[InstructionKMeansRuntimeStateSnapshot]
    human_thought_trace: NotRequired[list[dict[str, str | int]]]
    judge_clustering_trace: NotRequired[list[dict[str, object]]]
    clustering_pov_history: NotRequired[list[dict[str, object]]]
    intermittent_clustering_unsupported_logged: NotRequired[bool]
    human_turn_token_usage: NotRequired[dict[str, int]]
    turn_injection_consumed_turn_index: NotRequired[int]
    intermittent_clustering_token_usage: NotRequired[dict[str, int]]
    topic_candidates: NotRequired[list[dict[str, str]]]
    topic_candidates_metadata: NotRequired[dict[str, str | int | list[int]]]
    topic_discovery_force_generation: NotRequired[bool]
