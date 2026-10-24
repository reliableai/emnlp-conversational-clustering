"""Unit tests for batch runner slugging, path generation, and parallel result ordering."""

import io
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from conversational_clustering.accounting import (
    OpenRouterPricingClient,
    UsageCollector,
    empty_usage_summary,
)
from conversational_clustering.batch_runs.config import BatchRunConfig, HumanAgentConfig
from conversational_clustering.batch_runs.reporting import BatchProgressReporter
from conversational_clustering.batch_runs.runner import BatchRunner
from conversational_clustering.human_turn_injection import HumanTurnInjectionConfig
from conversational_clustering.models import EmbeddingModelConfig
from conversational_clustering.simulation.runner import SimulationResult
from conversational_clustering.topic_discovery import TopicDiscoveryConfig


def _simulation_result(turns_run: int) -> SimulationResult:
    return SimulationResult(
        state={"messages": [], "dataset_metadata": {"size": 0}, "iteration_count": 0},
        clustering_agent_accounting=empty_usage_summary().to_dict(),
        human_agent_accounting=empty_usage_summary().to_dict(),
        turns_run=turns_run,
        stopped_reason="max_turns",
    )


def _failed_simulation_result(reason: str = "run_error_exception") -> SimulationResult:
    return SimulationResult(
        state={"messages": [], "dataset_metadata": {"size": 0}, "iteration_count": 0},
        clustering_agent_accounting=empty_usage_summary().to_dict(),
        human_agent_accounting=empty_usage_summary().to_dict(),
        turns_run=0,
        stopped_reason=reason,
    )


def _usage_collector(input_tokens: int = 0, output_tokens: int = 0) -> UsageCollector:
    collector = UsageCollector()
    if input_tokens or output_tokens:
        collector.record(
            "test-model",
            {"input_tokens": input_tokens, "output_tokens": output_tokens},
        )
    return collector


def _session_stub(
    *,
    clustering_input_tokens: int = 0,
    clustering_output_tokens: int = 0,
    human_input_tokens: int = 0,
    human_output_tokens: int = 0,
    state: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        state=state or {"messages": [], "dataset_metadata": {"size": 0}, "iteration_count": 0},
        clustering_agent_usage_collector=_usage_collector(
            clustering_input_tokens,
            clustering_output_tokens,
        ),
        human_agent_usage_collector=_usage_collector(
            human_input_tokens,
            human_output_tokens,
        ),
        pricing_client=OpenRouterPricingClient(api_key_envs=()),
    )


def test_slugify_normalizes_and_applies_fallback() -> None:
    assert BatchRunner._slugify("My Run Name") == "my-run-name"
    assert BatchRunner._slugify("***", fallback="default") == "default"


def test_build_run_paths_uses_output_dir_and_expected_filenames() -> None:
    runner = BatchRunner(output_dir="data/batch_runs")

    clustering_path, log_path = runner._build_run_paths("slug", "001-abc")

    assert Path(clustering_path).as_posix().endswith("data/batch_runs/slug/001-abc/clustering.json")
    assert Path(log_path).as_posix().endswith("data/batch_runs/slug/001-abc/session.log")


def test_run_batch_parallel_orders_results_by_run_index() -> None:
    runner = BatchRunner(output_dir="unused")
    configs = [
        BatchRunConfig(name="Config One", dataset_name="synthetic", num_runs=2),
        BatchRunConfig(name="Config Two", dataset_name="synthetic", num_runs=1),
    ]

    with mock.patch.object(runner, "_build_config_slug", side_effect=["cfg-one", "cfg-two"]):
        with mock.patch.object(
            runner,
            "_run_single",
            side_effect=lambda config, run_index, config_slug: (
                f"{run_index:03d}-{config_slug}",
                _simulation_result(turns_run=run_index + 1),
            ),
        ):
            results = runner.run_batch_parallel(configs, max_workers=2)

    assert [result.config_slug for result in results] == ["cfg-one", "cfg-two"]
    assert results[0].run_ids == ["000-cfg-one", "001-cfg-one"]
    assert results[1].run_ids == ["000-cfg-two"]
    assert [res.turns_run for res in results[0].simulation_results] == [1, 2]


def test_run_batch_parallel_batch_progress_prints_failure_digest_and_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner = BatchRunner(output_dir="unused", batch_progress_enabled=True)
    configs = [BatchRunConfig(name="Config One", dataset_name="synthetic", num_runs=2)]

    with mock.patch.object(runner, "_build_config_slug", return_value="cfg-one"):
        with mock.patch.object(
            runner,
            "_run_single",
            side_effect=[
                ("000-cfg-one", _simulation_result(turns_run=1)),
                ("001-cfg-one", _failed_simulation_result()),
            ],
        ):
            _ = runner.run_batch_parallel(configs, max_workers=2)

    output = capsys.readouterr().out
    assert "FAIL cfg-one/001-cfg-one reason=run_error_exception" in output
    assert "Batch Progress Summary" in output
    assert "total=2 ok=1 fail=1" in output
    assert "failure_reasons=run_error_exception=1" in output
    assert "output_dir=" in output


def test_batch_progress_reporter_tty_path_renders_live_progress_line() -> None:
    class _TTYStream(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = _TTYStream()
    reporter = BatchProgressReporter(
        total_runs=1,
        output_dir="unused",
        stream=stream,
        now=lambda: 0.0,
    )

    reporter.start()
    reporter.record_run(
        config_slug="cfg-one",
        run_id="000-cfg-one",
        simulation_result=_simulation_result(turns_run=1),
    )
    reporter.finish(results=[])

    output = stream.getvalue()
    assert "\r[" in output
    assert "Batch Progress Summary" in output
    assert "total=1 ok=1 fail=0" in output


def test_run_single_passes_max_verification_retries_to_session_builder(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(
        name="Retry-config",
        dataset_name="synthetic",
        max_verification_retries=5,
    )
    fake_session = mock.Mock()
    fake_result = _simulation_result(turns_run=1)

    with mock.patch(
        "conversational_clustering.batch_runs.runner.create_session",
        return_value=fake_session,
    ) as create_session_mock:
        with mock.patch(
            "conversational_clustering.batch_runs.runner.run_human_simulation",
            return_value=fake_result,
        ):
            _run_id, result = runner._run_single(
                config=config,
                run_index=0,
                config_slug="slug",
            )

    assert result is fake_result
    assert create_session_mock.call_args.kwargs["max_verification_retries"] == 5


def test_run_single_passes_enable_intermittent_clustering_to_session_builder(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(
        name="Intermittent-config",
        dataset_name="synthetic",
        enable_intermittent_clustering=False,
    )
    fake_session = mock.Mock()
    fake_result = _simulation_result(turns_run=1)

    with mock.patch(
        "conversational_clustering.batch_runs.runner.create_session",
        return_value=fake_session,
    ) as create_session_mock:
        with mock.patch(
            "conversational_clustering.batch_runs.runner.run_human_simulation",
            return_value=fake_result,
        ):
            _run_id, result = runner._run_single(
                config=config,
                run_index=0,
                config_slug="slug",
            )

    assert result is fake_result
    assert create_session_mock.call_args.kwargs["enable_intermittent_clustering"] is False


def test_run_single_passes_turn_injection_to_session_builder(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    turn_injection = HumanTurnInjectionConfig(
        turn_index=2,
        edit_primitive="theme_switch",
        source_theme_id="fruit",
        target_theme_id="vehicle",
    )
    config = BatchRunConfig(
        name="Turn-injection-config",
        dataset_name="synthetic",
        max_turns=4,
        human_agent_config=HumanAgentConfig(turn_injection=turn_injection),
    )
    fake_session = mock.Mock()
    fake_result = _simulation_result(turns_run=1)

    with mock.patch(
        "conversational_clustering.batch_runs.runner.create_session",
        return_value=fake_session,
    ) as create_session_mock:
        with mock.patch(
            "conversational_clustering.batch_runs.runner.run_human_simulation",
            return_value=fake_result,
        ):
            _run_id, result = runner._run_single(
                config=config,
                run_index=0,
                config_slug="slug",
            )

    assert result is fake_result
    assert create_session_mock.call_args.kwargs["turn_injection"] == turn_injection


def test_run_single_writes_trace_artifacts_from_simulation_state(tmp_path: Path) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(name="Config One", dataset_name="synthetic", num_runs=1)
    simulation_result = SimulationResult(
        state={
            "messages": [
                HumanMessage(content="group by tone"),
                AIMessage(content="Working on it."),
                ToolMessage(name="save_clustering", content="Could not save", tool_call_id="1"),
            ],
            "dataset_metadata": {"size": 0},
            "iteration_count": 0,
            "human_thought_trace": [
                {
                    "turn_index": 1,
                    "message": "group by tone",
                    "thought": "I want clusters that reflect communication style.",
                }
            ],
            "judge_clustering_trace": [
                {
                    "turn_index": 1,
                    "clustering": {
                        "clusters": [
                            {
                                "name": "Tone",
                                "members": ["alpha", "beta"],
                            }
                        ]
                    },
                }
            ],
            "clustering_pov_history": [
                {"turn_index": 1, "povs": ["tone"]},
                {"turn_index": 2, "povs": ["tone", "audience"]},
            ],
            "topic_candidates": [
                {
                    "topic_id": "tone",
                    "label": "Tone",
                    "guidance": "Group items by tone and communication style.",
                    "rationale": "The conversation focused on stylistic differences.",
                }
            ],
            "topic_candidates_metadata": {
                "trigger_mode": "human_tool",
                "generated_at_turn": 1,
            },
        },
        clustering_agent_accounting={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        human_agent_accounting={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        turns_run=1,
        stopped_reason="max_turns",
    )

    with mock.patch(
        "conversational_clustering.batch_runs.runner.create_session",
        return_value=object(),
    ):
        with mock.patch(
            "conversational_clustering.batch_runs.runner.run_human_simulation",
            return_value=simulation_result,
        ):
            run_id, _ = runner._run_single(config=config, run_index=0, config_slug="cfg-one")

    trace_path = tmp_path / "cfg-one" / run_id / "conversation_trace.json"
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    thought_trace_path = tmp_path / "cfg-one" / run_id / "human_thought_trace.json"
    thought_payload = json.loads(thought_trace_path.read_text(encoding="utf-8"))
    judge_trace_path = tmp_path / "cfg-one" / run_id / "judge_clustering_trace.json"
    judge_trace_payload = json.loads(judge_trace_path.read_text(encoding="utf-8"))
    pov_history_path = tmp_path / "cfg-one" / run_id / "clustering_pov_history.json"
    pov_history_payload = json.loads(pov_history_path.read_text(encoding="utf-8"))
    topic_candidates_path = tmp_path / "cfg-one" / run_id / "topic_candidates.json"
    topic_candidates_payload = json.loads(topic_candidates_path.read_text(encoding="utf-8"))
    assert trace_path.exists()
    assert thought_trace_path.exists()
    assert judge_trace_path.exists()
    assert pov_history_path.exists()
    assert topic_candidates_path.exists()
    assert payload["messages"] == [
        {"role": "human", "content": "group by tone"},
        {"role": "assistant", "content": "Working on it."},
    ]
    assert payload["total_tool_calls_by_type"] == {"save_clustering": 1}
    assert payload["total_failed_tool_calls_by_type"] == {"save_clustering": 1}
    assert thought_payload == {
        "messages": [
            {
                "turn_index": 1,
                "message": "group by tone",
                "thought": "I want clusters that reflect communication style.",
            }
        ],
        "num_messages": 1,
    }
    assert judge_trace_payload == {
        "clusterings": [
            {
                "turn_index": 1,
                "clustering": {
                    "clusters": [
                        {
                            "name": "Tone",
                            "members": ["alpha", "beta"],
                        }
                    ]
                },
            }
        ],
        "num_clusterings": 1,
    }
    assert pov_history_payload == {
        "snapshots": [
            {"turn_index": 1, "povs": ["tone"], "num_povs": 1},
            {"turn_index": 2, "povs": ["tone", "audience"], "num_povs": 2},
        ],
        "num_snapshots": 2,
    }
    assert topic_candidates_payload == {
        "trigger_mode": "human_tool",
        "generated_at_turn": 1,
        "num_topics": 1,
        "topics": [
            {
                "topic_id": "tone",
                "label": "Tone",
                "guidance": "Group items by tone and communication style.",
                "rationale": "The conversation focused on stylistic differences.",
            }
        ],
    }


def test_run_single_retries_remote_protocol_error_then_succeeds(tmp_path: Path) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(name="Retry-config", dataset_name="synthetic")
    fake_session = mock.Mock()
    fake_result = _simulation_result(turns_run=1)

    with mock.patch(
        "conversational_clustering.batch_runs.runner.create_session",
        return_value=fake_session,
    ):
        with mock.patch(
            "conversational_clustering.batch_runs.runner.run_human_simulation",
            side_effect=[httpx.RemoteProtocolError("incomplete chunked read"), fake_result],
        ) as run_human_simulation_mock:
            with mock.patch("conversational_clustering.batch_runs.runner.time.sleep") as sleep_mock:
                _run_id, result = runner._run_single(
                    config=config,
                    run_index=0,
                    config_slug="slug",
                )

    assert result is fake_result
    assert run_human_simulation_mock.call_count == 2
    sleep_mock.assert_called_once_with(1.0)


def test_run_single_returns_failure_result_when_remote_protocol_retries_exhausted(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(name="Retry-config", dataset_name="synthetic")
    fake_session = _session_stub(
        clustering_input_tokens=2,
        clustering_output_tokens=3,
        human_input_tokens=5,
        human_output_tokens=7,
        state={"messages": [], "dataset_metadata": {"size": 3}, "iteration_count": 0},
    )

    with mock.patch(
        "conversational_clustering.batch_runs.runner.create_session",
        return_value=fake_session,
    ):
        with mock.patch(
            "conversational_clustering.batch_runs.runner.run_human_simulation",
            side_effect=httpx.RemoteProtocolError("incomplete chunked read"),
        ) as run_human_simulation_mock:
            with mock.patch("conversational_clustering.batch_runs.runner.time.sleep") as sleep_mock:
                _run_id, result = runner._run_single(
                    config=config,
                    run_index=0,
                    config_slug="slug",
                )

    assert result.stopped_reason == "run_error_remote_protocol"
    assert result.clustering_agent_accounting["total_tokens"] == 5
    assert result.human_agent_accounting["total_tokens"] == 12
    assert run_human_simulation_mock.call_count == 3
    assert [call.args[0] for call in sleep_mock.call_args_list] == [1.0, 2.0]


def test_run_single_returns_timeout_failure_result_when_request_times_out(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(name="Timeout-config", dataset_name="synthetic")
    fake_session = _session_stub(
        clustering_input_tokens=2,
        clustering_output_tokens=3,
        human_input_tokens=5,
        human_output_tokens=7,
        state={"messages": [], "dataset_metadata": {"size": 3}, "iteration_count": 0},
    )

    with mock.patch(
        "conversational_clustering.batch_runs.runner.create_session",
        return_value=fake_session,
    ):
        with mock.patch(
            "conversational_clustering.batch_runs.runner.run_human_simulation",
            side_effect=lambda **kwargs: (
                fake_session.state.update(
                    {
                        "messages": [HumanMessage(content="progress before timeout")],
                        "dataset_metadata": {"size": 3},
                        "iteration_count": 1,
                    }
                )
                or (_ for _ in ()).throw(httpx.ReadTimeout("timed out"))
            ),
        ) as run_human_simulation_mock:
            _run_id, result = runner._run_single(
                config=config,
                run_index=0,
                config_slug="slug",
            )

    assert result.stopped_reason == "run_error_timeout"
    assert result.turns_run == 1
    assert result.state["messages"] == [HumanMessage(content="progress before timeout")]
    assert result.clustering_agent_accounting["total_tokens"] == 5
    assert result.human_agent_accounting["total_tokens"] == 12
    assert run_human_simulation_mock.call_count == 1


def test_run_batch_parallel_continues_after_unexpected_run_exception() -> None:
    runner = BatchRunner(output_dir="unused")
    configs = [BatchRunConfig(name="Config One", dataset_name="synthetic", num_runs=2)]

    def _side_effect(config: BatchRunConfig, run_index: int, config_slug: str):
        _ = config
        if run_index == 1:
            raise RuntimeError("boom")
        return f"{run_index:03d}-{config_slug}", _simulation_result(turns_run=run_index + 1)

    with mock.patch.object(runner, "_build_config_slug", return_value="cfg-one"):
        with mock.patch.object(runner, "_run_single", side_effect=_side_effect):
            results = runner.run_batch_parallel(configs, max_workers=2)

    assert results[0].run_ids == ["000-cfg-one", "002-failed"]
    assert [res.stopped_reason for res in results[0].simulation_results] == [
        "max_turns",
        "run_error_exception",
    ]


def test_run_batch_parallel_executes_shared_prefix_once_per_branch_group_run(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    branch_metadata_common = {
        "branch_group": "human_agent_prompt_id_baseline_v1",
        "branch_shared_turns": 2,
        "branch_vary_by": "max_verification_retries",
        "branch_pre_vary_value": 2,
    }
    configs = [
        BatchRunConfig(
            name="branch_one",
            dataset_name="synthetic",
            max_turns=6,
            num_runs=1,
            max_verification_retries=1,
            metadata={**branch_metadata_common},
        ),
        BatchRunConfig(
            name="branch_two",
            dataset_name="synthetic",
            max_turns=6,
            num_runs=1,
            max_verification_retries=4,
            metadata={**branch_metadata_common},
        ),
    ]

    shared_result = SimulationResult(
        state={"messages": [], "dataset_metadata": {"size": 0}, "iteration_count": 2},
        clustering_agent_accounting={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        human_agent_accounting={"input_tokens": 5, "output_tokens": 6, "total_tokens": 11},
        turns_run=2,
        stopped_reason="max_turns",
    )
    branch_one_result = SimulationResult(
        state={"messages": [], "dataset_metadata": {"size": 0}, "iteration_count": 6},
        clustering_agent_accounting={
            "input_tokens": 10,
            "output_tokens": 10,
            "total_tokens": 20,
        },
        human_agent_accounting={"input_tokens": 8, "output_tokens": 8, "total_tokens": 16},
        turns_run=4,
        stopped_reason="max_turns",
    )
    branch_two_result = SimulationResult(
        state={"messages": [], "dataset_metadata": {"size": 0}, "iteration_count": 5},
        clustering_agent_accounting={
            "input_tokens": 9,
            "output_tokens": 9,
            "total_tokens": 18,
        },
        human_agent_accounting={"input_tokens": 7, "output_tokens": 7, "total_tokens": 14},
        turns_run=3,
        stopped_reason="clustering_saved",
    )

    shared_session = _session_stub()
    branch_session_one = _session_stub()
    branch_session_two = _session_stub()

    with mock.patch.object(runner, "_build_config_slug", side_effect=["cfg-one", "cfg-two"]):
        with mock.patch(
            "conversational_clustering.batch_runs.runner.create_session",
            side_effect=[shared_session, branch_session_one, branch_session_two],
        ) as create_session_mock:
            with mock.patch(
                "conversational_clustering.batch_runs.runner.run_human_simulation",
                side_effect=[shared_result, branch_one_result, branch_two_result],
            ) as run_human_simulation_mock:
                results = runner.run_batch_parallel(configs, max_workers=2)

    assert run_human_simulation_mock.call_count == 3
    assert [call.kwargs["max_turns"] for call in run_human_simulation_mock.call_args_list] == [
        2,
        4,
        4,
    ]
    assert create_session_mock.call_count == 3
    observed_retries = [
        call.kwargs["max_verification_retries"] for call in create_session_mock.call_args_list
    ]
    assert observed_retries[0] == 2
    assert sorted(observed_retries[1:]) == [1, 4]
    assert len(results) == 2
    assert results[0].run_ids and results[1].run_ids
    assert sorted(result.simulation_results[0].turns_run for result in results) == [5, 6]

    shared_root = tmp_path / "_shared" / "cfg-one"
    assert shared_root.exists()
    mapping_path = shared_root / "branch_mapping.json"
    assert mapping_path.exists()
    mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert mapping_payload == {
        "shared_config_slug": "cfg-one",
        "branch_config_slugs": ["cfg-one", "cfg-two"],
    }
    shared_run_dirs = [path for path in shared_root.iterdir() if path.is_dir()]
    assert len(shared_run_dirs) == 1
    shared_run_dir = shared_run_dirs[0]
    assert (shared_run_dir / "session.log").exists()
    assert (shared_run_dir / "conversation_trace.json").exists()
    assert (shared_run_dir / "human_thought_trace.json").exists()


def test_run_session_with_retries_preserves_live_collector_instances_for_seeded_runs(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(name="seeded", dataset_name="synthetic")
    branch_session = _session_stub()
    seed_result = SimulationResult(
        state={"messages": [], "dataset_metadata": {"size": 0}, "iteration_count": 2},
        clustering_agent_accounting={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        human_agent_accounting={"input_tokens": 5, "output_tokens": 6, "total_tokens": 11},
        turns_run=2,
        stopped_reason="max_turns",
    )

    def _fake_run_human_simulation(*, session: SimpleNamespace, **_: object) -> SimulationResult:
        assert (
            session.clustering_agent_usage_collector
            is branch_session.clustering_agent_usage_collector
        )
        assert session.human_agent_usage_collector is branch_session.human_agent_usage_collector
        assert session.seed_clustering_agent_accounting == seed_result.clustering_agent_accounting
        assert session.seed_human_agent_accounting == seed_result.human_agent_accounting
        return _simulation_result(turns_run=1)

    with mock.patch.object(runner, "_create_session_for_config", return_value=branch_session):
        with mock.patch(
            "conversational_clustering.batch_runs.runner.run_human_simulation",
            side_effect=_fake_run_human_simulation,
        ):
            result, ok = runner._run_session_with_retries(
                config=config,
                clustering_path=str(tmp_path / "clustering.json"),
                run_logger=logging.getLogger("test.seeded.collector_identity"),
                max_turns=1,
                reporter_factory=mock.Mock,
                run_id="001-seeded",
                run_label="run",
                seed_result=seed_result,
            )

    assert ok is True
    assert result.turns_run == 3


def test_run_session_with_retries_preserves_seed_result_when_session_creation_fails(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(name="seeded", dataset_name="synthetic")
    seed_result = SimulationResult(
        state={
            "messages": [HumanMessage(content="shared prefix")],
            "dataset_metadata": {"size": 2},
            "iteration_count": 2,
        },
        clustering_agent_accounting={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        human_agent_accounting={"input_tokens": 5, "output_tokens": 6, "total_tokens": 11},
        turns_run=2,
        stopped_reason="max_turns",
    )

    with mock.patch.object(
        runner,
        "_create_session_for_config",
        side_effect=RuntimeError("session creation failed"),
    ):
        result, ok = runner._run_session_with_retries(
            config=config,
            clustering_path=str(tmp_path / "clustering.json"),
            run_logger=logging.getLogger("test.seeded.session_creation_failure"),
            max_turns=1,
            reporter_factory=mock.Mock,
            run_id="001-seeded",
            run_label="branched run",
            seed_result=seed_result,
        )

    assert ok is False
    assert result.stopped_reason == "run_error_exception"
    assert result.turns_run == seed_result.turns_run
    assert result.state == seed_result.state
    assert result.state is not seed_result.state
    assert result.clustering_agent_accounting == seed_result.clustering_agent_accounting
    assert result.human_agent_accounting == seed_result.human_agent_accounting


def test_run_batch_parallel_uses_pre_branch_guidance_for_shared_turns(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    branch_metadata_common = {
        "branch_group": "human_agent_config_ablation",
        "branch_shared_turns": 2,
        "branch_vary_by": "human_agent_config",
        "branch_pre_vary_value": HumanAgentConfig(prompt_id="baseline_v3"),
    }
    configs = [
        BatchRunConfig(
            name="guidance_branch_one",
            dataset_name="synthetic",
            human_agent_config=HumanAgentConfig(
                prompt_id="baseline_v1",
                guidance="Group items by genre.",
            ),
            max_turns=4,
            num_runs=1,
            metadata={**branch_metadata_common},
        ),
        BatchRunConfig(
            name="guidance_branch_two",
            dataset_name="synthetic",
            human_agent_config=HumanAgentConfig(
                prompt_id="baseline_v1",
                guidance="Propose a data-driven clustering theme.",
            ),
            max_turns=4,
            num_runs=1,
            metadata={**branch_metadata_common},
        ),
    ]

    shared_result = _simulation_result(turns_run=2)
    branch_one_result = _simulation_result(turns_run=2)
    branch_two_result = _simulation_result(turns_run=2)

    shared_session = _session_stub()
    branch_session_one = _session_stub()
    branch_session_two = _session_stub()

    with mock.patch.object(runner, "_build_config_slug", side_effect=["cfg-one", "cfg-two"]):
        with mock.patch(
            "conversational_clustering.batch_runs.runner.create_session",
            side_effect=[shared_session, branch_session_one, branch_session_two],
        ):
            with mock.patch(
                "conversational_clustering.batch_runs.runner.run_human_simulation",
                side_effect=[shared_result, branch_one_result, branch_two_result],
            ) as run_human_simulation_mock:
                runner.run_batch_parallel(configs, max_workers=2)

    observed_guidance = [
        call.kwargs["guidance"] for call in run_human_simulation_mock.call_args_list
    ]
    assert observed_guidance[0] is None
    assert sorted(observed_guidance[1:]) == sorted(
        [
            "Group items by genre.",
            "Propose a data-driven clustering theme.",
        ]
    )


def test_run_batch_parallel_branch_group_preserves_successful_siblings_on_child_failure(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    branch_metadata_common = {
        "branch_group": "human_agent_prompt_id_baseline_v1",
        "branch_shared_turns": 2,
        "branch_vary_by": "max_verification_retries",
        "branch_pre_vary_value": 2,
    }
    configs = [
        BatchRunConfig(
            name="branch_one",
            dataset_name="synthetic",
            max_turns=6,
            num_runs=1,
            max_verification_retries=1,
            metadata={**branch_metadata_common},
        ),
        BatchRunConfig(
            name="branch_two",
            dataset_name="synthetic",
            max_turns=6,
            num_runs=1,
            max_verification_retries=4,
            metadata={**branch_metadata_common},
        ),
    ]
    shared_result = SimulationResult(
        state={
            "messages": [HumanMessage(content="shared branch context")],
            "dataset_metadata": {"size": 0},
            "iteration_count": 2,
        },
        clustering_agent_accounting={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        human_agent_accounting={"input_tokens": 5, "output_tokens": 6, "total_tokens": 11},
        turns_run=2,
        stopped_reason="max_turns",
    )

    with mock.patch.object(runner, "_build_config_slug", side_effect=["cfg-one", "cfg-two"]):
        with mock.patch.object(
            runner._branching,
            "run_shared_prefix",
            return_value=("001-shared", shared_result, str(tmp_path / "_shared-clustering.json")),
        ):
            with mock.patch.object(
                runner._branching,
                "run_single_from_shared_prefix",
                side_effect=[
                    ("001-cfg-one", _simulation_result(turns_run=4)),
                    RuntimeError("branch child failed"),
                ],
            ):
                results = runner.run_batch_parallel(configs, max_workers=1)

    assert [result.config_slug for result in results] == ["cfg-one", "cfg-two"]
    assert results[0].run_ids == ["001-cfg-one"]
    assert results[0].simulation_results[0].stopped_reason == "max_turns"
    assert results[1].run_ids == ["001-failed"]
    assert results[1].simulation_results[0].stopped_reason == "run_error_exception"
    assert results[1].simulation_results[0].turns_run == 2
    assert results[1].simulation_results[0].clustering_agent_accounting["total_tokens"] == 7
    assert results[1].simulation_results[0].human_agent_accounting["total_tokens"] == 11
    assert results[1].simulation_results[0].state["messages"] == [
        HumanMessage(content="shared branch context")
    ]


def test_plan_topic_branching_single_run_preserves_shared_prefix_exception(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(
        name="topic_branching",
        dataset_name="synthetic",
        max_turns=6,
        num_runs=1,
        human_agent_config=HumanAgentConfig(
            prompt_id="baseline_v4",
            topic_discovery=TopicDiscoveryConfig(enabled=True, turn_budget=2),
        ),
    )

    with mock.patch.object(
        runner._branching,
        "run_shared_prefix",
        side_effect=RuntimeError("shared prefix failed"),
    ):
        with pytest.raises(RuntimeError, match="shared prefix failed"):
            runner._branching.plan_topic_branching_single_run(
                source_config=config,
                execution_config=config,
                config_index=0,
                config_slug="cfg-topic",
                run_index=0,
            )


def test_create_session_for_config_passes_selected_topic_metadata(tmp_path: Path) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(
        name="topic_branch",
        dataset_name="synthetic",
        human_agent_config=HumanAgentConfig(
            prompt_id="baseline_v4",
            topic_discovery=TopicDiscoveryConfig(enabled=False),
        ),
        metadata={
            "topic_branch_topic_label": "Tone",
            "topic_branch_topic_guidance": "Group items by tone.",
            "topic_branch_generated_at_turn": 2,
        },
    )

    with mock.patch(
        "conversational_clustering.batch_runs.runner.create_session",
        return_value=mock.Mock(),
    ) as create_session_mock:
        runner._create_session_for_config(
            config=config,
            clustering_path=str(tmp_path / "clustering.json"),
            run_logger=logging.getLogger("test.create_session_for_config.topic"),
        )

    assert create_session_mock.call_args.kwargs["selected_topic_label"] == "Tone"
    assert create_session_mock.call_args.kwargs["selected_topic_guidance"] == (
        "Group items by tone."
    )
    assert create_session_mock.call_args.kwargs["selected_topic_generated_at_turn"] == 2


def test_create_session_for_config_preserves_inherited_shared_prefix_persona(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(
        name="persona_branch",
        dataset_name="synthetic",
        human_agent_config=HumanAgentConfig(prompt_id="baseline_v4"),
        metadata={
            "branch_pre_vary_value": HumanAgentConfig(
                prompt_id="baseline_v3",
                persona="casual_social",
            )
        },
    )

    with mock.patch(
        "conversational_clustering.batch_runs.runner.create_session",
        return_value=mock.Mock(),
    ) as create_session_mock:
        runner._create_session_for_config(
            config=config,
            clustering_path=str(tmp_path / "clustering.json"),
            run_logger=logging.getLogger("test.create_session_for_config.persona"),
        )

    assert create_session_mock.call_args.kwargs["human_agent_persona"] == "casual_social"


def test_create_session_for_config_preserves_inherited_late_turn_injection(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    late_turn_injection = HumanTurnInjectionConfig(
        turn_index=5,
        edit_primitive="theme_switch",
        source_theme_id="vehicle",
        target_theme_id="fruit",
    )
    config = BatchRunConfig(
        name="turn_injection_branch",
        dataset_name="synthetic",
        max_turns=6,
        human_agent_config=HumanAgentConfig(prompt_id="baseline_v4"),
        metadata={
            "branch_shared_turns": 3,
            "branch_pre_vary_value": HumanAgentConfig(
                prompt_id="baseline_v3",
                turn_injection=late_turn_injection,
            ),
        },
    )

    with mock.patch(
        "conversational_clustering.batch_runs.runner.create_session",
        return_value=mock.Mock(),
    ) as create_session_mock:
        runner._create_session_for_config(
            config=config,
            clustering_path=str(tmp_path / "clustering.json"),
            run_logger=logging.getLogger("test.create_session_for_config.turn_injection"),
        )

    assert create_session_mock.call_args.kwargs["turn_injection"] == late_turn_injection


def test_materialize_result_config_preserves_shared_prefix_human_agent_config_details() -> None:
    runner = BatchRunner()
    shared_turn_injection = HumanTurnInjectionConfig(
        turn_index=2,
        edit_primitive="theme_switch",
        source_theme_id="fruit",
        target_theme_id="vehicle",
    )
    late_turn_injection = HumanTurnInjectionConfig(
        turn_index=5,
        edit_primitive="theme_switch",
        source_theme_id="vehicle",
        target_theme_id="fruit",
    )
    config = BatchRunConfig(
        name="persona_branch",
        dataset_name="synthetic",
        max_turns=6,
        human_agent_config=HumanAgentConfig(
            prompt_id="baseline_v4",
            turn_injection=late_turn_injection,
        ),
        metadata={
            "branch_shared_turns": 3,
            "branch_vary_by": "human_agent_config",
            "branch_pre_vary_value": HumanAgentConfig(
                prompt_id="baseline_v3",
                persona="casual_social",
                turn_injection=shared_turn_injection,
            ),
        },
    )

    materialized = runner._materialize_result_config(config)
    serialized = materialized.to_dict()

    assert materialized.human_agent_config.persona == "casual_social"
    assert serialized["human_agent_config"]["persona"] == "casual_social"
    assert serialized["human_agent_config"]["turn_injection"] == {
        "turn_index": 5,
        "edit_primitive": "theme_switch",
        "source_theme_id": "vehicle",
        "target_theme_id": "fruit",
    }
    assert serialized["metadata"]["branch_pre_human_agent_config"] == {
        "prompt_id": "baseline_v3",
        "guidance": None,
        "persona": "casual_social",
        "turn_injection": {
            "turn_index": 2,
            "edit_primitive": "theme_switch",
            "source_theme_id": "fruit",
            "target_theme_id": "vehicle",
        },
    }


def test_run_batch_parallel_auto_topic_branching_materializes_realized_branch_results(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(
        name="topic_branching",
        dataset_name="synthetic",
        max_turns=6,
        num_runs=1,
        human_agent_config=HumanAgentConfig(
            prompt_id="baseline_v4",
            topic_discovery=TopicDiscoveryConfig(enabled=True, turn_budget=2),
        ),
    )
    shared_result = SimulationResult(
        state={
            "messages": [],
            "dataset_metadata": {"size": 0},
            "iteration_count": 2,
            "topic_candidates": [
                {
                    "topic_id": "tone",
                    "label": "Tone",
                    "guidance": "Group items by tone.",
                    "rationale": "Tone keeps appearing in summaries.",
                },
                {
                    "topic_id": "audience",
                    "label": "Audience",
                    "guidance": "Group items by audience.",
                    "rationale": "Audience keeps appearing in summaries.",
                },
            ],
            "topic_candidates_metadata": {
                "trigger_mode": "turn_budget",
                "generated_at_turn": 2,
            },
        },
        clustering_agent_accounting={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        human_agent_accounting={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        turns_run=2,
        stopped_reason="topic_discovery_handoff",
    )
    branch_result_one = _simulation_result(turns_run=4)
    branch_result_two = SimulationResult(
        state={"messages": [], "dataset_metadata": {"size": 0}, "iteration_count": 5},
        clustering_agent_accounting={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        human_agent_accounting={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        turns_run=5,
        stopped_reason="clustering_saved",
    )

    with mock.patch.object(runner, "_build_config_slug", return_value="cfg-topic"):
        with mock.patch.object(
            runner._branching,
            "run_shared_prefix",
            return_value=("001-shared", shared_result, str(tmp_path / "_shared-clustering.json")),
        ):
            with mock.patch(
                "conversational_clustering.batch_runs.branching.build_topic_branch_slug_suffix",
                side_effect=["a1b2c3d4e5f6", "b2c3d4e5f6a7"],
            ):
                with mock.patch.object(
                    runner._branching,
                    "run_single_from_shared_prefix",
                    side_effect=[
                        ("001-tone", branch_result_one),
                        ("001-audience", branch_result_two),
                    ],
                ) as branch_run_mock:
                    results = runner.run_batch_parallel([config], max_workers=1)

    assert [result.config_slug for result in results] == [
        "cfg-topic--topic-a1b2c3d4e5f6",
        "cfg-topic--topic-b2c3d4e5f6a7",
    ]
    assert all(result.run_ids for result in results)
    assert [result.config.metadata["topic_branch_topic_id"] for result in results] == [
        "tone",
        "audience",
    ]
    assert all(
        result.config.metadata["topic_branch_mode"] == "auto_topic_discovery" for result in results
    )
    assert all(
        result.config.human_agent_config.topic_discovery.enabled is False for result in results
    )
    assert [result.config.human_agent_config.guidance for result in results] == [None, None]
    assert all("topic_branch_source_run_id" not in result.config.metadata for result in results)
    assert all("topic_branch_source_run_index" not in result.config.metadata for result in results)
    assert all("topic_branch_generated_at_turn" not in result.config.metadata for result in results)
    assert [
        call.kwargs["config"].metadata["topic_branch_topic_guidance"]
        for call in branch_run_mock.call_args_list
    ] == [
        "Group items by tone.",
        "Group items by audience.",
    ]
    mapping_path = tmp_path / "_shared" / "cfg-topic" / "001-shared" / "topic_branch_mapping.json"
    assert mapping_path.exists()
    mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert mapping_payload["branches"][0]["topic_id"] == "tone"
    assert mapping_payload["branches"][1]["topic_id"] == "audience"
    assert mapping_payload["branches"][0]["generated_at_turn"] == 2
    assert mapping_payload["branches"][1]["generated_at_turn"] == 2


def test_run_batch_parallel_auto_topic_branching_groups_repeated_topic_runs_together(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(
        name="topic_branching",
        dataset_name="synthetic",
        max_turns=6,
        num_runs=2,
        human_agent_config=HumanAgentConfig(
            prompt_id="baseline_v4",
            topic_discovery=TopicDiscoveryConfig(enabled=True, turn_budget=2),
        ),
    )
    shared_result_one = SimulationResult(
        state={
            "messages": [],
            "dataset_metadata": {"size": 0},
            "iteration_count": 2,
            "topic_candidates": [
                {
                    "topic_id": "tone",
                    "label": "Tone",
                    "guidance": "Group items by tone.",
                    "rationale": "Tone keeps appearing in summaries.",
                }
            ],
            "topic_candidates_metadata": {
                "trigger_mode": "turn_budget",
                "generated_at_turn": 2,
            },
        },
        clustering_agent_accounting={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        human_agent_accounting={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        turns_run=2,
        stopped_reason="topic_discovery_handoff",
    )
    shared_result_two = SimulationResult(
        state={
            "messages": [],
            "dataset_metadata": {"size": 0},
            "iteration_count": 3,
            "topic_candidates": [
                {
                    "topic_id": "tone",
                    "label": "Tone",
                    "guidance": "Group items by tone.",
                    "rationale": "Tone keeps appearing in summaries.",
                }
            ],
            "topic_candidates_metadata": {
                "trigger_mode": "turn_budget",
                "generated_at_turn": 3,
            },
        },
        clustering_agent_accounting={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        human_agent_accounting={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        turns_run=3,
        stopped_reason="topic_discovery_handoff",
    )

    with mock.patch.object(runner, "_build_config_slug", return_value="cfg-topic"):
        with mock.patch.object(
            runner._branching,
            "run_shared_prefix",
            side_effect=[
                ("001-shared", shared_result_one, str(tmp_path / "_shared-clustering-one.json")),
                ("002-shared", shared_result_two, str(tmp_path / "_shared-clustering-two.json")),
            ],
        ):
            with mock.patch(
                "conversational_clustering.batch_runs.branching.build_topic_branch_slug_suffix",
                return_value="4d7f2b18c9a0",
            ):
                with mock.patch.object(
                    runner._branching,
                    "run_single_from_shared_prefix",
                    side_effect=[
                        ("001-tone", _simulation_result(turns_run=4)),
                        ("002-tone", _simulation_result(turns_run=5)),
                    ],
                ):
                    results = runner.run_batch_parallel([config], max_workers=1)

    assert [result.config_slug for result in results] == ["cfg-topic--topic-4d7f2b18c9a0"]
    assert results[0].run_ids == ["001-tone", "002-tone"]
    assert len(results[0].simulation_results) == 2
    assert results[0].config.num_runs == 2
    assert results[0].config.name == "topic_branching__topic_tone"


def test_run_batch_parallel_auto_topic_branching_preserves_original_embedding_force_recompute(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(
        name="topic_branching",
        dataset_name="synthetic",
        max_turns=6,
        num_runs=1,
        enable_embeddings=True,
        embedding_force_recompute=True,
        human_agent_config=HumanAgentConfig(
            prompt_id="baseline_v4",
            topic_discovery=TopicDiscoveryConfig(enabled=True, turn_budget=2),
        ),
    )
    shared_result = SimulationResult(
        state={
            "messages": [],
            "dataset_metadata": {"size": 0},
            "iteration_count": 2,
            "topic_candidates": [
                {
                    "topic_id": "tone",
                    "label": "Tone",
                    "guidance": "Group items by tone.",
                    "rationale": "Tone keeps appearing in summaries.",
                }
            ],
            "topic_candidates_metadata": {
                "trigger_mode": "turn_budget",
                "generated_at_turn": 2,
            },
        },
        clustering_agent_accounting={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        human_agent_accounting={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        turns_run=2,
        stopped_reason="topic_discovery_handoff",
    )

    with mock.patch(
        "conversational_clustering.batch_runs.runner.build_data_loader",
    ) as build_data_loader_mock:
        build_data_loader_mock.return_value.get_unique_item_texts_by_id.return_value = {
            "alpha": "Alpha"
        }
        with mock.patch(
            "conversational_clustering.batch_runs.runner.ensure_embeddings_for_items",
        ):
            with mock.patch.object(runner, "_build_config_slug", return_value="cfg-topic"):
                with mock.patch.object(
                    runner._branching,
                    "run_shared_prefix",
                    return_value=(
                        "001-shared",
                        shared_result,
                        str(tmp_path / "_shared-clustering.json"),
                    ),
                ):
                    with mock.patch(
                        "conversational_clustering.batch_runs.branching.build_topic_branch_slug_suffix",
                        return_value="4d7f2b18c9a0",
                    ):
                        with mock.patch.object(
                            runner._branching,
                            "run_single_from_shared_prefix",
                            return_value=("001-tone", _simulation_result(turns_run=4)),
                        ) as branch_run_mock:
                            results = runner.run_batch_parallel([config], max_workers=1)

    assert results[0].config.embedding_force_recompute is True
    assert branch_run_mock.call_args.kwargs["config"].embedding_force_recompute is False


def test_run_batch_parallel_auto_topic_branching_separates_divergent_repeated_topics(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(
        name="topic_branching",
        dataset_name="synthetic",
        max_turns=6,
        num_runs=2,
        human_agent_config=HumanAgentConfig(
            prompt_id="baseline_v4",
            topic_discovery=TopicDiscoveryConfig(enabled=True, turn_budget=2),
        ),
    )
    shared_result_one = SimulationResult(
        state={
            "messages": [],
            "dataset_metadata": {"size": 0},
            "iteration_count": 2,
            "topic_candidates": [
                {
                    "topic_id": "tone",
                    "label": "Tone",
                    "guidance": "Group items by tone.",
                    "rationale": "Tone keeps appearing in summaries.",
                }
            ],
            "topic_candidates_metadata": {
                "trigger_mode": "turn_budget",
                "generated_at_turn": 2,
            },
        },
        clustering_agent_accounting={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        human_agent_accounting={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        turns_run=2,
        stopped_reason="topic_discovery_handoff",
    )
    shared_result_two = SimulationResult(
        state={
            "messages": [],
            "dataset_metadata": {"size": 0},
            "iteration_count": 3,
            "topic_candidates": [
                {
                    "topic_id": "tone",
                    "label": "Mood",
                    "guidance": "Group items by mood shifts.",
                    "rationale": "Mood keeps appearing in summaries.",
                }
            ],
            "topic_candidates_metadata": {
                "trigger_mode": "turn_budget",
                "generated_at_turn": 3,
            },
        },
        clustering_agent_accounting={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        human_agent_accounting={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        turns_run=3,
        stopped_reason="topic_discovery_handoff",
    )

    with mock.patch.object(runner, "_build_config_slug", return_value="cfg-topic"):
        with mock.patch.object(
            runner._branching,
            "run_shared_prefix",
            side_effect=[
                ("001-shared", shared_result_one, str(tmp_path / "_shared-clustering-one.json")),
                ("002-shared", shared_result_two, str(tmp_path / "_shared-clustering-two.json")),
            ],
        ):
            with mock.patch(
                "conversational_clustering.batch_runs.branching.build_topic_branch_slug_suffix",
                side_effect=["4d7f2b18c9a0", "79afc2d4410e"],
            ):
                with mock.patch.object(
                    runner._branching,
                    "run_single_from_shared_prefix",
                    side_effect=[
                        ("001-tone", _simulation_result(turns_run=4)),
                        ("002-tone", _simulation_result(turns_run=5)),
                    ],
                ):
                    results = runner.run_batch_parallel([config], max_workers=1)

    assert [result.config_slug for result in results] == [
        "cfg-topic--topic-4d7f2b18c9a0",
        "cfg-topic--topic-79afc2d4410e",
    ]
    assert results[0].config.to_dict() != results[1].config.to_dict()
    assert results[0].config.metadata["topic_branch_topic_guidance"] == "Group items by tone."
    assert results[1].config.metadata["topic_branch_topic_guidance"] == (
        "Group items by mood shifts."
    )


def test_run_batch_parallel_preserves_input_order_for_auto_topic_and_ordinary_configs(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    auto_topic_config = BatchRunConfig(
        name="topic_branching",
        dataset_name="synthetic",
        max_turns=6,
        num_runs=1,
        human_agent_config=HumanAgentConfig(
            prompt_id="baseline_v4",
            topic_discovery=TopicDiscoveryConfig(enabled=True, turn_budget=2),
        ),
    )
    ordinary_config = BatchRunConfig(name="ordinary", dataset_name="synthetic", num_runs=1)
    shared_result = SimulationResult(
        state={
            "messages": [],
            "dataset_metadata": {"size": 0},
            "iteration_count": 2,
            "topic_candidates": [
                {
                    "topic_id": "tone",
                    "label": "Tone",
                    "guidance": "Group items by tone.",
                    "rationale": "Tone keeps appearing in summaries.",
                }
            ],
            "topic_candidates_metadata": {
                "trigger_mode": "turn_budget",
                "generated_at_turn": 2,
            },
        },
        clustering_agent_accounting={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        human_agent_accounting={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        turns_run=2,
        stopped_reason="topic_discovery_handoff",
    )

    with mock.patch.object(runner, "_build_config_slug", side_effect=["cfg-topic", "cfg-plain"]):
        with mock.patch.object(
            runner._branching,
            "run_shared_prefix",
            return_value=("001-shared", shared_result, str(tmp_path / "_shared-clustering.json")),
        ):
            with mock.patch(
                "conversational_clustering.batch_runs.branching.build_topic_branch_slug_suffix",
                return_value="4d7f2b18c9a0",
            ):
                with mock.patch.object(
                    runner._branching,
                    "run_single_from_shared_prefix",
                    return_value=("001-tone", _simulation_result(turns_run=4)),
                ):
                    with mock.patch.object(
                        runner,
                        "_run_single",
                        return_value=("001-cfg-plain", _simulation_result(turns_run=3)),
                    ):
                        results = runner.run_batch_parallel(
                            [auto_topic_config, ordinary_config],
                            max_workers=1,
                        )

    assert [result.config_slug for result in results] == [
        "cfg-topic--topic-4d7f2b18c9a0",
        "cfg-plain",
    ]
    assert results[0].config.name == "topic_branching__topic_tone"
    assert results[1].config.name == "ordinary"


def test_run_batch_parallel_auto_topic_branching_preserves_completed_branches_on_later_failure(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(
        name="topic_branching",
        dataset_name="synthetic",
        max_turns=6,
        num_runs=1,
        human_agent_config=HumanAgentConfig(
            prompt_id="baseline_v4",
            topic_discovery=TopicDiscoveryConfig(enabled=True, turn_budget=2),
        ),
    )
    shared_result = SimulationResult(
        state={
            "messages": [],
            "dataset_metadata": {"size": 0},
            "iteration_count": 2,
            "topic_candidates": [
                {
                    "topic_id": "tone",
                    "label": "Tone",
                    "guidance": "Group items by tone.",
                    "rationale": "Tone keeps appearing in summaries.",
                },
                {
                    "topic_id": "audience",
                    "label": "Audience",
                    "guidance": "Group items by audience.",
                    "rationale": "Audience keeps appearing in summaries.",
                },
                {
                    "topic_id": "genre",
                    "label": "Genre",
                    "guidance": "Group items by genre.",
                    "rationale": "Genre keeps appearing in summaries.",
                },
            ],
            "topic_candidates_metadata": {
                "trigger_mode": "turn_budget",
                "generated_at_turn": 2,
            },
        },
        clustering_agent_accounting={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        human_agent_accounting={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        turns_run=2,
        stopped_reason="topic_discovery_handoff",
    )

    with mock.patch.object(runner, "_build_config_slug", return_value="cfg-topic"):
        with mock.patch.object(
            runner._branching,
            "run_shared_prefix",
            return_value=("001-shared", shared_result, str(tmp_path / "_shared-clustering.json")),
        ):
            with mock.patch(
                "conversational_clustering.batch_runs.branching.build_topic_branch_slug_suffix",
                side_effect=["a1b2c3d4e5f6", "b2c3d4e5f6a7", "c3d4e5f6a7b8"],
            ):
                with mock.patch.object(
                    runner._branching,
                    "run_single_from_shared_prefix",
                    side_effect=[
                        ("001-tone", _simulation_result(turns_run=4)),
                        RuntimeError("artifact write failed"),
                        ("001-genre", _simulation_result(turns_run=5)),
                    ],
                ):
                    results = runner.run_batch_parallel([config], max_workers=1)

    assert [result.config_slug for result in results] == [
        "cfg-topic--topic-a1b2c3d4e5f6",
        "cfg-topic--topic-b2c3d4e5f6a7",
        "cfg-topic--topic-c3d4e5f6a7b8",
    ]
    assert results[0].run_ids == ["001-tone"]
    assert results[0].simulation_results[0].stopped_reason == "max_turns"
    assert results[1].run_ids == ["001-failed-branch-02"]
    assert results[1].simulation_results[0].stopped_reason == "run_error_exception"
    assert results[1].config.metadata["topic_branch_topic_id"] == "audience"
    assert results[2].run_ids == ["001-genre"]
    assert results[2].simulation_results[0].stopped_reason == "max_turns"
    assert results[2].config.metadata["topic_branch_topic_id"] == "genre"
    assert all(result.config.metadata["topic_branch_outcome"] == "branched" for result in results)
    mapping_path = tmp_path / "_shared" / "cfg-topic" / "001-shared" / "topic_branch_mapping.json"
    assert mapping_path.exists()
    mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert [branch["topic_id"] for branch in mapping_payload["branches"]] == [
        "tone",
        "audience",
        "genre",
    ]
    assert [branch["run_id"] for branch in mapping_payload["branches"]] == [
        "001-tone",
        "001-failed-branch-02",
        "001-genre",
    ]


def test_run_batch_parallel_auto_topic_branching_keeps_completed_branches_on_mapping_failure(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(
        name="topic_branching",
        dataset_name="synthetic",
        max_turns=6,
        num_runs=1,
        human_agent_config=HumanAgentConfig(
            prompt_id="baseline_v4",
            topic_discovery=TopicDiscoveryConfig(enabled=True, turn_budget=2),
        ),
    )
    shared_result = SimulationResult(
        state={
            "messages": [],
            "dataset_metadata": {"size": 0},
            "iteration_count": 2,
            "topic_candidates": [
                {
                    "topic_id": "tone",
                    "label": "Tone",
                    "guidance": "Group items by tone.",
                    "rationale": "Tone keeps appearing in summaries.",
                },
                {
                    "topic_id": "audience",
                    "label": "Audience",
                    "guidance": "Group items by audience.",
                    "rationale": "Audience keeps appearing in summaries.",
                },
            ],
            "topic_candidates_metadata": {
                "trigger_mode": "turn_budget",
                "generated_at_turn": 2,
            },
        },
        clustering_agent_accounting={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        human_agent_accounting={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        turns_run=2,
        stopped_reason="topic_discovery_handoff",
    )

    with mock.patch.object(runner, "_build_config_slug", return_value="cfg-topic"):
        with mock.patch.object(
            runner._branching,
            "run_shared_prefix",
            return_value=("001-shared", shared_result, str(tmp_path / "_shared-clustering.json")),
        ):
            with mock.patch(
                "conversational_clustering.batch_runs.branching.build_topic_branch_slug_suffix",
                side_effect=["a1b2c3d4e5f6", "b2c3d4e5f6a7"],
            ):
                with mock.patch.object(
                    runner._branching,
                    "run_single_from_shared_prefix",
                    side_effect=[
                        ("001-tone", _simulation_result(turns_run=4)),
                        ("001-audience", _simulation_result(turns_run=5)),
                    ],
                ):
                    with mock.patch.object(
                        runner._branching,
                        "write_topic_branch_mapping_artifact",
                        side_effect=RuntimeError("mapping write failed"),
                    ):
                        results = runner.run_batch_parallel([config], max_workers=1)

    assert [result.config_slug for result in results] == [
        "cfg-topic--topic-a1b2c3d4e5f6",
        "cfg-topic--topic-b2c3d4e5f6a7",
        "cfg-topic--run-001--shared_run_terminal",
    ]
    assert results[0].run_ids == ["001-tone"]
    assert results[1].run_ids == ["001-audience"]
    assert results[2].run_ids == ["001-failed-postprocess"]
    assert results[2].simulation_results[0].stopped_reason == "run_error_exception"


def test_run_batch_parallel_auto_topic_branching_uses_unique_failed_run_ids_per_shared_run(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(
        name="topic_branching",
        dataset_name="synthetic",
        max_turns=6,
        num_runs=1,
        human_agent_config=HumanAgentConfig(
            prompt_id="baseline_v4",
            topic_discovery=TopicDiscoveryConfig(enabled=True, turn_budget=2),
        ),
    )
    shared_result = SimulationResult(
        state={
            "messages": [],
            "dataset_metadata": {"size": 0},
            "iteration_count": 2,
            "topic_candidates": [
                {
                    "topic_id": "tone",
                    "label": "Tone",
                    "guidance": "Group items by tone.",
                    "rationale": "Tone keeps appearing in summaries.",
                },
                {
                    "topic_id": "audience",
                    "label": "Audience",
                    "guidance": "Group items by audience.",
                    "rationale": "Audience keeps appearing in summaries.",
                },
            ],
            "topic_candidates_metadata": {
                "trigger_mode": "turn_budget",
                "generated_at_turn": 2,
            },
        },
        clustering_agent_accounting={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        human_agent_accounting={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        turns_run=2,
        stopped_reason="topic_discovery_handoff",
    )

    with mock.patch.object(runner, "_build_config_slug", return_value="cfg-topic"):
        with mock.patch.object(
            runner._branching,
            "run_shared_prefix",
            return_value=("001-shared", shared_result, str(tmp_path / "_shared-clustering.json")),
        ):
            with mock.patch(
                "conversational_clustering.batch_runs.branching.build_topic_branch_slug_suffix",
                side_effect=["a1b2c3d4e5f6", "b2c3d4e5f6a7"],
            ):
                with mock.patch.object(
                    runner._branching,
                    "run_single_from_shared_prefix",
                    side_effect=[
                        RuntimeError("first branch failed"),
                        ("001-audience", _simulation_result(turns_run=5)),
                    ],
                ):
                    with mock.patch.object(
                        runner._branching,
                        "write_topic_branch_mapping_artifact",
                        side_effect=RuntimeError("mapping write failed"),
                    ):
                        results = runner.run_batch_parallel([config], max_workers=1)

    failed_run_ids = [
        run_id for result in results for run_id in result.run_ids if "failed" in run_id
    ]
    assert failed_run_ids == ["001-failed-branch-01", "001-failed-postprocess"]
    assert len(set(failed_run_ids)) == 2


def test_run_batch_parallel_auto_topic_branching_keeps_unbranched_result_when_no_topics(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(
        name="topic_branching",
        dataset_name="synthetic",
        max_turns=6,
        num_runs=1,
        human_agent_config=HumanAgentConfig(
            prompt_id="baseline_v4",
            topic_discovery=TopicDiscoveryConfig(enabled=True, turn_budget=2),
        ),
    )
    shared_result = _simulation_result(turns_run=6)
    shared_result.state = {
        "messages": [HumanMessage(content="Please cluster by tone.")],
        "dataset_metadata": {"size": 0},
        "iteration_count": 6,
    }
    shared_clustering_path = tmp_path / "_shared-clustering.json"
    shared_clustering_path.write_text("{}", encoding="utf-8")

    with mock.patch.object(runner, "_build_config_slug", return_value="cfg-topic"):
        with mock.patch.object(
            runner._branching,
            "run_shared_prefix",
            return_value=("001-shared", shared_result, str(shared_clustering_path)),
        ):
            with mock.patch.object(
                runner._branching, "run_single_from_shared_prefix"
            ) as branch_run_mock:
                results = runner.run_batch_parallel([config], max_workers=1)

    assert len(results) == 1
    assert results[0].config_slug == "cfg-topic--run-001--no_topics_generated"
    assert results[0].config.metadata["topic_branch_outcome"] == "no_topics_generated"
    branch_run_mock.assert_not_called()


def test_run_batch_parallel_auto_topic_branching_marks_final_turn_recursion_limit_as_terminal(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(
        name="topic_branching",
        dataset_name="synthetic",
        max_turns=6,
        num_runs=1,
        human_agent_config=HumanAgentConfig(
            prompt_id="baseline_v4",
            topic_discovery=TopicDiscoveryConfig(enabled=True, turn_budget=2),
        ),
    )
    shared_result = SimulationResult(
        state={
            "messages": [HumanMessage(content="Please cluster by tone.")],
            "dataset_metadata": {"size": 0},
            "iteration_count": 6,
        },
        clustering_agent_accounting={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        human_agent_accounting={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        turns_run=6,
        stopped_reason="final_turn_recursion_limit",
    )
    shared_clustering_path = tmp_path / "_shared-clustering.json"
    shared_clustering_path.write_text("{}", encoding="utf-8")

    with mock.patch.object(runner, "_build_config_slug", return_value="cfg-topic"):
        with mock.patch.object(
            runner._branching,
            "run_shared_prefix",
            return_value=("001-shared", shared_result, str(shared_clustering_path)),
        ):
            with mock.patch.object(
                runner._branching, "run_single_from_shared_prefix"
            ) as branch_run_mock:
                results = runner.run_batch_parallel([config], max_workers=1)

    assert len(results) == 1
    assert results[0].config_slug == "cfg-topic--run-001--shared_run_terminal"
    assert results[0].config.metadata["topic_branch_outcome"] == "shared_run_terminal"
    branch_run_mock.assert_not_called()


def test_run_single_from_shared_prefix_continues_after_topic_discovery_handoff(
    tmp_path: Path,
) -> None:
    runner = BatchRunner(output_dir=str(tmp_path))
    config = BatchRunConfig(
        name="topic_branch",
        dataset_name="synthetic",
        max_turns=6,
        human_agent_config=HumanAgentConfig(prompt_id="baseline_v4"),
        metadata={
            "topic_branch_topic_id": "tone",
            "topic_branch_topic_label": "Tone",
            "topic_branch_topic_guidance": "Group items by tone.",
        },
    )
    shared_result = SimulationResult(
        state={
            "messages": [],
            "dataset_metadata": {"size": 0},
            "iteration_count": 2,
            "topic_candidates_metadata": {"generated_at_turn": 2},
        },
        clustering_agent_accounting={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        human_agent_accounting={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        turns_run=2,
        stopped_reason="topic_discovery_handoff",
    )
    continued_result = SimulationResult(
        state={"messages": [], "dataset_metadata": {"size": 0}, "iteration_count": 5},
        clustering_agent_accounting={"input_tokens": 3, "output_tokens": 3, "total_tokens": 6},
        human_agent_accounting={"input_tokens": 4, "output_tokens": 4, "total_tokens": 8},
        turns_run=5,
        stopped_reason="clustering_saved",
    )

    with mock.patch.object(runner, "_build_run_id", return_value="001-branch"):
        with mock.patch.object(
            runner,
            "_run_session_with_retries",
            return_value=(continued_result, True),
        ) as run_session_mock:
            run_id, result = runner._branching.run_single_from_shared_prefix(
                config=config,
                config_slug="cfg-topic-001-topic-opaque-id",
                run_index=0,
                shared_result=shared_result,
                shared_clustering_path=str(tmp_path / "_shared-clustering.json"),
            )

    assert run_id == "001-branch"
    assert result is continued_result
    assert run_session_mock.call_args.kwargs["max_turns"] == 4
    assert run_session_mock.call_args.kwargs["seed_result"] is shared_result
    selected_topic_path = (
        tmp_path / "cfg-topic-001-topic-opaque-id" / "001-branch" / "selected_topic.json"
    )
    assert selected_topic_path.exists()
    assert json.loads(selected_topic_path.read_text(encoding="utf-8")) == {
        "topic_id": "tone",
        "label": "Tone",
        "guidance": "Group items by tone.",
        "generated_at_turn": 2,
    }
    topic_candidates_path = (
        tmp_path / "cfg-topic-001-topic-opaque-id" / "001-branch" / "topic_candidates.json"
    )
    assert not topic_candidates_path.exists()


def test_build_embedding_preparation_specs_merges_by_cache_identity() -> None:
    runner = BatchRunner(output_dir="unused")
    configs = [
        BatchRunConfig(
            name="Config One",
            dataset_name="synthetic",
            enable_embeddings=True,
            embedding_force_recompute=False,
            embedding_batch_size=16,
            embedding_model_config=EmbeddingModelConfig(
                model="openai/text-embedding-3-small",
                dimensions=256,
            ),
        ),
        BatchRunConfig(
            name="Config Two",
            dataset_name="synthetic",
            enable_embeddings=True,
            embedding_force_recompute=True,
            embedding_batch_size=64,
            embedding_model_config=EmbeddingModelConfig(
                model="openai/text-embedding-3-small",
                dimensions=256,
            ),
        ),
    ]

    loader_one = mock.Mock()
    loader_one.get_unique_item_texts_by_id.return_value = {"a": "Alpha"}
    loader_two = mock.Mock()
    loader_two.get_unique_item_texts_by_id.return_value = {"b": "Beta"}

    with mock.patch(
        "conversational_clustering.batch_runs.runner.build_data_loader",
        side_effect=[loader_one, loader_two],
    ):
        preparations = runner._build_embedding_preparation_specs(configs)

    assert len(preparations) == 1
    preparation = preparations[0]
    assert preparation.dataset_name == "synthetic"
    assert preparation.batch_size == 64
    assert preparation.force_recompute is True
    assert preparation.item_texts_by_id == {"a": "Alpha", "b": "Beta"}


def test_run_batch_parallel_prepares_embeddings_before_worker_execution() -> None:
    runner = BatchRunner(output_dir="unused")
    configs = [
        BatchRunConfig(
            name="Config One",
            dataset_name="synthetic",
            num_runs=1,
            enable_embeddings=True,
            embedding_force_recompute=True,
        )
    ]
    loader = mock.Mock()
    loader.get_unique_item_texts_by_id.return_value = {"alpha_id": "Alpha"}

    with mock.patch(
        "conversational_clustering.batch_runs.runner.build_data_loader",
        return_value=loader,
    ):
        with mock.patch(
            "conversational_clustering.batch_runs.runner.ensure_embeddings_for_items",
        ) as ensure_embeddings:

            def _run_single_side_effect(
                config: BatchRunConfig,
                run_index: int,
                config_slug: str,
            ) -> tuple[str, SimulationResult]:
                if not ensure_embeddings.called:
                    raise AssertionError("embeddings were not prepared")
                turns_run = 0 if config.embedding_force_recompute else run_index + 1
                return f"{run_index:03d}-{config_slug}", _simulation_result(turns_run=turns_run)

            with mock.patch.object(runner, "_build_config_slug", return_value="cfg-one"):
                with mock.patch.object(
                    runner,
                    "_run_single",
                    side_effect=_run_single_side_effect,
                ) as run_single:
                    results = runner.run_batch_parallel(configs, max_workers=1)

    ensure_embeddings.assert_called_once()
    assert ensure_embeddings.call_args.kwargs["force_recompute"] is True
    assert run_single.call_args.args[0].embedding_force_recompute is False
    assert results[0].simulation_results[0].turns_run == 1
