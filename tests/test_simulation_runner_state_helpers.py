"""Unit tests for simulation runner state and helper utilities."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import conversational_clustering.simulation.runner as simulation_runner
from conversational_clustering.clustering_agent.backends.backend_v1 import (
    method as backend_v1_method,
)
from conversational_clustering.clustering_agent.backends.oneshot_kmeans_v1 import (
    OneShotKMeansV1ClusteringBackendConfig,
)
from conversational_clustering.clustering_agent.state import AgentState
from conversational_clustering.simulation.reporters import NullReporter


def test_build_initial_state_sets_expected_defaults() -> None:
    state = simulation_runner.build_initial_state(dataset_size=12)

    assert state["messages"] == []
    assert state["dataset_metadata"] == {"size": 12}
    assert state["iteration_count"] == 0
    assert state["human_thought_trace"] == []
    assert state["judge_clustering_trace"] == []


def test_coerce_reporter_uses_null_reporter_when_none() -> None:
    coerced = simulation_runner._coerce_reporter(None)

    assert isinstance(coerced, NullReporter)


def test_coerce_reporter_returns_explicit_reporter() -> None:
    explicit_reporter = NullReporter()

    assert simulation_runner._coerce_reporter(explicit_reporter) is explicit_reporter


def test_with_total_adds_total_tokens() -> None:
    with_total = simulation_runner._with_total({"input_tokens": 5, "output_tokens": 7})

    assert with_total == {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}


def test_content_to_text_handles_str_and_non_str() -> None:
    assert simulation_runner._content_to_text("hello") == "hello"
    assert simulation_runner._content_to_text(["a", "b"]) == "['a', 'b']"


def test_clustering_saved_detects_successful_save_message() -> None:
    messages = [
        ToolMessage(name="read_dataset", content="ok", tool_call_id="1"),
        ToolMessage(name="save_clustering", content="Clustering saved to file", tool_call_id="2"),
    ]

    assert simulation_runner._clustering_saved(messages) is True


def test_clustering_saved_returns_false_when_no_success_message() -> None:
    messages = [ToolMessage(name="save_clustering", content="Could not save", tool_call_id="1")]

    assert simulation_runner._clustering_saved(messages) is False


def test_build_conversation_trace_artifact_excludes_tool_messages_and_tracks_counts() -> None:
    state: AgentState = {
        "messages": [
            HumanMessage(content="group by topic"),
            AIMessage(content="I'll analyze the dataset."),
            ToolMessage(name="read_dataset", content="ok", tool_call_id="1"),
            ToolMessage(name="save_clustering", content="Could not save", tool_call_id="2"),
        ],
        "dataset_metadata": {"size": 0},
        "iteration_count": 0,
    }

    artifact = simulation_runner.build_conversation_trace_artifact(state)

    assert artifact["messages"] == [
        {"role": "human", "content": "group by topic"},
        {"role": "assistant", "content": "I'll analyze the dataset."},
    ]
    assert artifact["total_tool_calls_by_type"] == {
        "read_dataset": 1,
        "save_clustering": 1,
    }
    assert artifact["total_failed_tool_calls_by_type"] == {"save_clustering": 1}


def test_build_human_thought_trace_artifact_returns_normalized_messages() -> None:
    state: AgentState = {
        "messages": [],
        "dataset_metadata": {"size": 0},
        "iteration_count": 0,
        "human_thought_trace": [
            {
                "turn_index": 1,
                "message": "Can we cluster by topic?",
                "thought": "I want the agent to start with topical buckets.",
            },
            {
                "turn_index": 2,
                "message": "Could you summarize the current clusters?",
                "thought": "I need a checkpoint before refining guidance.",
            },
        ],
    }

    artifact = simulation_runner.build_human_thought_trace_artifact(state)

    assert artifact == {
        "messages": [
            {
                "turn_index": 1,
                "message": "Can we cluster by topic?",
                "thought": "I want the agent to start with topical buckets.",
            },
            {
                "turn_index": 2,
                "message": "Could you summarize the current clusters?",
                "thought": "I need a checkpoint before refining guidance.",
            },
        ],
        "num_messages": 2,
    }


def test_build_human_thought_trace_artifact_ignores_invalid_entries() -> None:
    state: AgentState = {
        "messages": [],
        "dataset_metadata": {"size": 0},
        "iteration_count": 0,
        "human_thought_trace": [
            {"turn_index": True, "message": "x", "thought": "y"},
            {"turn_index": 1, "message": "", "thought": "y"},
            {"turn_index": 1, "message": "x", "thought": ""},
            {"turn_index": 3, "message": "valid", "thought": "valid thought"},
        ],
    }

    artifact = simulation_runner.build_human_thought_trace_artifact(state)

    assert artifact == {
        "messages": [{"turn_index": 3, "message": "valid", "thought": "valid thought"}],
        "num_messages": 1,
    }


def test_build_intermittent_clustering_judge_save_path_uses_sibling_filename() -> None:
    output_path = simulation_runner.build_intermittent_clustering_judge_save_path(
        "data/output/clustering.json"
    )

    assert output_path == "data/output/clustering.judge.json"


def test_maybe_build_intermittent_clustering_judge_save_path_returns_none_when_unsupported() -> (
    None
):
    output_path = simulation_runner.maybe_build_intermittent_clustering_judge_save_path(
        "data/output/clustering.json",
        clustering_agent_backend="oneshot_kmeans_v1",
        clustering_agent_backend_config=OneShotKMeansV1ClusteringBackendConfig(),
    )

    assert output_path is None


def test_build_numbered_judge_snapshot_path_appends_zero_padded_turn_index() -> None:
    output_path = backend_v1_method._build_numbered_judge_snapshot_path(
        "data/output/clustering.judge.json",
        turn_index=12,
    )

    assert output_path == "data/output/clustering.judge.0012.json"


def test_build_judge_clustering_trace_artifact_returns_normalized_payload() -> None:
    state: AgentState = {
        "messages": [],
        "dataset_metadata": {"size": 0},
        "iteration_count": 0,
        "judge_clustering_trace": [
            {
                "turn_index": 1,
                "clustering": {
                    "clusters": [
                        {"name": "Topic A", "members": ["alpha", "beta"]},
                        {"name": "Topic B", "members": ["gamma"]},
                    ]
                },
            }
        ],
    }

    artifact = simulation_runner.build_judge_clustering_trace_artifact(state)

    assert artifact == {
        "clusterings": [
            {
                "turn_index": 1,
                "clustering": {
                    "clusters": [
                        {"name": "Topic A", "members": ["alpha", "beta"]},
                        {"name": "Topic B", "members": ["gamma"]},
                    ]
                },
            }
        ],
        "num_clusterings": 1,
    }


def test_build_clustering_pov_history_artifact_returns_normalized_payload() -> None:
    state: AgentState = {
        "messages": [],
        "dataset_metadata": {"size": 0},
        "iteration_count": 0,
        "clustering_pov_history": [
            {"turn_index": 1, "povs": [" Genre ", "age   rating"]},
            {"turn_index": 2, "povs": ["genre", "tone"]},
        ],
    }

    artifact = simulation_runner.build_clustering_pov_history_artifact(state)

    assert artifact == {
        "snapshots": [
            {"turn_index": 1, "povs": ["Genre", "age rating"], "num_povs": 2},
            {"turn_index": 2, "povs": ["genre", "tone"], "num_povs": 2},
        ],
        "num_snapshots": 2,
    }


def test_build_clustering_pov_history_artifact_ignores_invalid_entries() -> None:
    state: AgentState = {
        "messages": [],
        "dataset_metadata": {"size": 0},
        "iteration_count": 0,
        "clustering_pov_history": [
            {"turn_index": True, "povs": ["invalid"]},
            {"turn_index": 0, "povs": ["invalid"]},
            {"turn_index": 2, "povs": ["   ", "\n"]},
            {"turn_index": 3, "povs": ["tone", "Tone", "audience"]},
        ],
    }

    artifact = simulation_runner.build_clustering_pov_history_artifact(state)

    assert artifact == {
        "snapshots": [
            {"turn_index": 3, "povs": ["tone", "audience"], "num_povs": 2},
        ],
        "num_snapshots": 1,
    }


def test_build_clustering_pov_history_artifact_returns_none_when_empty() -> None:
    state: AgentState = {
        "messages": [],
        "dataset_metadata": {"size": 0},
        "iteration_count": 0,
        "clustering_pov_history": [],
    }

    assert simulation_runner.build_clustering_pov_history_artifact(state) is None


def test_build_topic_candidates_artifact_returns_normalized_payload() -> None:
    state: AgentState = {
        "messages": [],
        "dataset_metadata": {"size": 0},
        "iteration_count": 0,
        "topic_candidates": [
            {
                "topic_id": "tone",
                "label": "Tone",
                "guidance": "Group items by tone.",
                "rationale": "Tone appears repeatedly in summaries.",
            },
            {
                "topic_id": "audience",
                "label": "Audience",
                "guidance": "Group items by intended audience.",
                "rationale": "The dataset seems to separate novice and expert material.",
            },
        ],
        "topic_candidates_metadata": {
            "trigger_mode": "turn_budget",
            "generated_at_turn": 3,
        },
    }

    artifact = simulation_runner.build_topic_candidates_artifact(state)

    assert artifact == {
        "trigger_mode": "turn_budget",
        "generated_at_turn": 3,
        "num_topics": 2,
        "topics": [
            {
                "topic_id": "tone",
                "label": "Tone",
                "guidance": "Group items by tone.",
                "rationale": "Tone appears repeatedly in summaries.",
            },
            {
                "topic_id": "audience",
                "label": "Audience",
                "guidance": "Group items by intended audience.",
                "rationale": "The dataset seems to separate novice and expert material.",
            },
        ],
    }


def test_build_topic_candidates_artifact_preserves_multi_round_metadata() -> None:
    state: AgentState = {
        "messages": [],
        "dataset_metadata": {"size": 0},
        "iteration_count": 0,
        "topic_candidates": [
            {
                "topic_id": "tone",
                "label": "Tone",
                "guidance": "Group items by tone.",
                "rationale": "Tone appears repeatedly in summaries.",
            }
        ],
        "topic_candidates_metadata": {
            "trigger_mode": "human_tool",
            "generated_at_turn": 2,
            "num_rounds": 3,
            "new_topics_by_round": [2, 2, 1],
        },
    }

    artifact = simulation_runner.build_topic_candidates_artifact(state)

    assert artifact == {
        "trigger_mode": "human_tool",
        "generated_at_turn": 2,
        "num_rounds": 3,
        "new_topics_by_round": [2, 2, 1],
        "num_topics": 1,
        "topics": [
            {
                "topic_id": "tone",
                "label": "Tone",
                "guidance": "Group items by tone.",
                "rationale": "Tone appears repeatedly in summaries.",
            }
        ],
    }
