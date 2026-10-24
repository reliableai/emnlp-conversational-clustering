"""Unit and integration tests for evaluation runner config parsing and execution."""

import json
from pathlib import Path
from typing import cast

import pytest

from conversational_clustering.evaluation.base import EvaluationResult, Evaluator
from conversational_clustering.evaluation.batch_run_artifacts import (
    iter_batch_run_artifact_contexts,
)
from conversational_clustering.evaluation.runner import (
    EvaluationRunFailed,
    iter_batch_run_requests,
    parse_evaluation_run_config,
    run_evaluation_config,
)


def _write_clustering(
    input_dir: Path,
    *,
    config_slug: str,
    run_id: str,
    payload: dict,
    write_conversation_trace: bool = True,
    write_judge_trace: bool = False,
    write_clustering_pov_history: bool = False,
) -> Path:
    clustering_path = input_dir / config_slug / run_id / "clustering.json"
    clustering_path.parent.mkdir(parents=True, exist_ok=True)
    clustering_path.write_text(json.dumps(payload), encoding="utf-8")
    if write_conversation_trace:
        conversation_trace_path = clustering_path.with_name("conversation_trace.json")
        conversation_trace_path.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "human", "content": "hello"},
                        {"role": "assistant", "content": "hi"},
                    ],
                    "clarification_events": [],
                    "total_tool_calls_by_type": {"save_clustering": 1},
                    "total_failed_tool_calls_by_type": {},
                }
            ),
            encoding="utf-8",
        )
    if write_judge_trace:
        judge_trace_path = clustering_path.with_name("judge_clustering_trace.json")
        judge_trace_path.write_text(
            json.dumps(
                {
                    "clusterings": [
                        {
                            "turn_index": 1,
                            "clustering": payload,
                        }
                    ],
                    "num_clusterings": 1,
                }
            ),
            encoding="utf-8",
        )
    if write_clustering_pov_history:
        pov_history_path = clustering_path.with_name("clustering_pov_history.json")
        pov_history_path.write_text(
            json.dumps(
                {
                    "snapshots": [
                        {"turn_index": 1, "povs": ["topic"], "num_povs": 1},
                        {"turn_index": 2, "povs": ["topic", "tone"], "num_povs": 2},
                    ],
                    "num_snapshots": 2,
                }
            ),
            encoding="utf-8",
        )
    return clustering_path


def _write_results_jsonl(input_dir: Path, records: list[dict]) -> Path:
    path = input_dir / "results.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")
    return path


def _write_eval_config(
    directory: Path,
    *,
    input_dir: Path,
    evaluators: list[str],
    report_path: Path,
    vary_by: str | None = None,
    pairwise_scope: str | None = None,
    resume: bool = False,
) -> Path:
    evaluators_toml = ", ".join(f'"{name}"' for name in evaluators)
    # Use TOML literal strings so Windows backslashes are not treated as escapes.
    config_lines = [
        f"input_dir = '{input_dir}'",
        f"evaluators = [{evaluators_toml}]",
    ]
    if vary_by is not None:
        config_lines.append(f'vary_by = "{vary_by}"')
    if pairwise_scope is not None:
        config_lines.append(f'pairwise_scope = "{pairwise_scope}"')
    if resume:
        config_lines.append("resume = true")
    config_lines.append(f"report_path = '{report_path}'")
    config_text = "\n".join(config_lines) + "\n"
    config_path = directory / "evaluation.toml"
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


def test_parse_evaluation_run_config_infers_single_mode(tmp_path: Path) -> None:
    config_path = _write_eval_config(
        tmp_path,
        input_dir=tmp_path / "batch_output",
        evaluators=["basic_metrics"],
        report_path=tmp_path / "report.json",
    )

    parsed = parse_evaluation_run_config(config_path)

    assert parsed.evaluation_mode == "single"
    assert parsed.evaluators == ["basic_metrics"]
    assert parsed.evaluator_params == {}
    assert parsed.resume is False


def test_parse_evaluation_run_config_accepts_resume(tmp_path: Path) -> None:
    config_path = _write_eval_config(
        tmp_path,
        input_dir=tmp_path / "batch_output",
        evaluators=["basic_metrics"],
        report_path=tmp_path / "report.json",
        resume=True,
    )

    parsed = parse_evaluation_run_config(config_path)

    assert parsed.resume is True


def test_parse_evaluation_run_config_rejects_single_mode_vary_by_key(tmp_path: Path) -> None:
    config_path = _write_eval_config(
        tmp_path,
        input_dir=tmp_path / "batch_output",
        evaluators=["basic_metrics"],
        report_path=tmp_path / "report.json",
        vary_by="human_agent_config",
    )

    with pytest.raises(ValueError, match="unsupported top-level key"):
        parse_evaluation_run_config(config_path)


def test_parse_evaluation_run_config_rejects_mixed_evaluator_modes(tmp_path: Path) -> None:
    config_path = _write_eval_config(
        tmp_path,
        input_dir=tmp_path / "batch_output",
        evaluators=["basic_metrics", "pairwise_consistency"],
        report_path=tmp_path / "report.json",
    )

    with pytest.raises(ValueError, match="must be the same type"):
        parse_evaluation_run_config(config_path)


@pytest.mark.parametrize(
    ("relative_path", "expected_mode", "expected_resume"),
    [
        ("evaluation_configs/example_single_maximal.toml", "single", True),
        ("evaluation_configs/example_pairwise_maximal.toml", "pairwise", True),
    ],
)
def test_checked_in_example_evaluation_configs_parse(
    relative_path: str,
    expected_mode: str,
    expected_resume: bool,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parsed = parse_evaluation_run_config(repo_root / relative_path)

    assert parsed.evaluation_mode == expected_mode
    assert parsed.resume is expected_resume


def test_iter_batch_run_requests_yields_expected_metadata(tmp_path: Path) -> None:
    input_dir = tmp_path
    config_slug = "alpha-config"
    run_id = "001-run"
    _write_clustering(
        input_dir,
        config_slug=config_slug,
        run_id=run_id,
        payload={"clusters": [{"name": "A", "members": ["apple", "banana"]}]},
        write_judge_trace=True,
        write_clustering_pov_history=True,
    )
    results_path = _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Example"},
                "config_slug": config_slug,
                "run_ids": [run_id],
            }
        ],
    )

    requests = list(iter_batch_run_requests(input_dir))

    assert len(requests) == 1
    request = requests[0]
    assert request.metadata["config_slug"] == config_slug
    assert request.metadata["run_id"] == run_id
    assert request.artifacts["batch_run_config_path"] == str(results_path)
    assert "clustering.json" in request.artifacts["clustering_path"]
    assert "conversation_trace.json" in request.artifacts["conversation_trace_path"]
    assert "judge_clustering_trace.json" in request.artifacts["judge_clustering_trace_path"]
    assert "clustering_pov_history.json" in request.artifacts["clustering_pov_history_path"]
    assert request.artifacts["conversation_trace"]["messages"][0]["role"] == "human"
    assert request.artifacts["judge_clustering_trace"]["num_clusterings"] == 1
    assert request.artifacts["clustering_pov_history"]["num_snapshots"] == 2
    assert request.artifacts["batch_run_result"] is None
    assert request.artifacts["turns_run"] is None
    assert request.predictions.assignment == {"apple": 0, "banana": 0}


def test_iter_batch_run_requests_uses_results_jsonl_manifest_order(tmp_path: Path) -> None:
    input_dir = tmp_path
    manifest_runs = [
        ("second-config", "002-run", "pear"),
        ("first-config", "001-run", "apple"),
    ]
    for config_slug, run_id, item in manifest_runs:
        _write_clustering(
            input_dir,
            config_slug=config_slug,
            run_id=run_id,
            payload={"clusters": [{"name": "A", "members": [item]}]},
        )
    _write_clustering(
        input_dir,
        config_slug="stray-config",
        run_id="999-stray",
        payload={"clusters": [{"name": "A", "members": ["stray"]}]},
    )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Second"},
                "config_slug": "second-config",
                "run_ids": ["002-run"],
            },
            {
                "config": {"name": "First"},
                "config_slug": "first-config",
                "run_ids": ["001-run"],
            },
        ],
    )

    requests = list(iter_batch_run_requests(input_dir))

    assert [
        (request.metadata["config_slug"], request.metadata["run_id"]) for request in requests
    ] == [("second-config", "002-run"), ("first-config", "001-run")]
    assert [request.predictions.assignment for request in requests] == [
        {"pear": 0},
        {"apple": 0},
    ]


def test_iter_batch_run_requests_requires_manifest_listed_clustering_artifact(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path
    config_slug = "alpha-config"
    run_id = "001-run"
    run_dir = input_dir / config_slug / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "conversation_trace.json").write_text(
        json.dumps({"messages": [{"role": "human", "content": "hello"}]}),
        encoding="utf-8",
    )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Example"},
                "config_slug": config_slug,
                "run_ids": [run_id],
            }
        ],
    )

    with pytest.raises(ValueError, match="Missing required clustering artifact"):
        list(iter_batch_run_requests(input_dir))


def test_iter_batch_run_requests_requires_results_jsonl_config_slug(tmp_path: Path) -> None:
    input_dir = tmp_path
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Example"},
                "run_ids": ["001-run"],
            }
        ],
    )

    with pytest.raises(ValueError, match="missing non-empty string 'config_slug'"):
        list(iter_batch_run_requests(input_dir))


def test_iter_batch_run_artifact_contexts_can_discover_run_without_clustering(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path
    config_slug = "alpha-config"
    run_id = "001-run"
    run_dir = input_dir / config_slug / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "conversation_trace.json").write_text(
        json.dumps({"messages": [{"role": "human", "content": "hello"}]}),
        encoding="utf-8",
    )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Example"},
                "config_slug": config_slug,
                "run_ids": [run_id],
            }
        ],
    )

    contexts = list(iter_batch_run_artifact_contexts(input_dir, require_clustering=False))

    assert len(contexts) == 1
    assert contexts[0].run_dir == run_dir
    assert contexts[0].clustering_path == run_dir / "clustering.json"
    assert contexts[0].conversation_trace["messages"][0]["role"] == "human"


def test_iter_batch_run_requests_skips_failed_runs_without_artifacts(tmp_path: Path) -> None:
    input_dir = tmp_path
    _write_clustering(
        input_dir,
        config_slug="ok-config",
        run_id="001-ok",
        payload={"clusters": [{"name": "A", "members": ["apple"]}]},
    )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Failed"},
                "config_slug": "failed-config",
                "run_ids": ["000-failed"],
                "results": [{"turns_run": 0, "stopped_reason": "run_error_exception"}],
            },
            {
                "config": {"name": "Max turns"},
                "config_slug": "max-turns-config",
                "run_ids": ["000-max-turns"],
                "results": [{"turns_run": 8, "stopped_reason": "max_turns"}],
            },
            {
                "config": {"name": "Ok"},
                "config_slug": "ok-config",
                "run_ids": ["001-ok"],
                "results": [{"turns_run": 1, "stopped_reason": "clustering_saved"}],
            },
        ],
    )

    requests = list(iter_batch_run_requests(input_dir))

    assert [
        (request.metadata["config_slug"], request.metadata["run_id"]) for request in requests
    ] == [("ok-config", "001-ok")]


def test_iter_batch_run_requests_requires_conversation_trace_artifact(tmp_path: Path) -> None:
    input_dir = tmp_path
    config_slug = "alpha-config"
    run_id = "001-run"
    _write_clustering(
        input_dir,
        config_slug=config_slug,
        run_id=run_id,
        payload={"clusters": [{"name": "A", "members": ["apple", "banana"]}]},
        write_conversation_trace=False,
    )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Example"},
                "config_slug": config_slug,
                "run_ids": [run_id],
            }
        ],
    )

    with pytest.raises(ValueError, match="Missing required conversation trace artifact"):
        list(iter_batch_run_requests(input_dir))


def test_iter_batch_run_requests_leaves_missing_clustering_pov_history_optional(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path
    config_slug = "alpha-config"
    run_id = "001-run"
    _write_clustering(
        input_dir,
        config_slug=config_slug,
        run_id=run_id,
        payload={"clusters": [{"name": "A", "members": ["apple", "banana"]}]},
    )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Example"},
                "config_slug": config_slug,
                "run_ids": [run_id],
            }
        ],
    )

    requests = list(iter_batch_run_requests(input_dir))

    assert len(requests) == 1
    request = requests[0]
    assert request.artifacts["clustering_pov_history_path"] is None
    assert request.artifacts["clustering_pov_history"] is None
    assert request.artifacts["turns_run"] is None


def test_iter_batch_run_requests_exposes_turns_run_from_results_jsonl(tmp_path: Path) -> None:
    input_dir = tmp_path
    config_slug = "alpha-config"
    run_id = "001-run"
    _write_clustering(
        input_dir,
        config_slug=config_slug,
        run_id=run_id,
        payload={"clusters": [{"name": "A", "members": ["apple", "banana"]}]},
    )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Example"},
                "config_slug": config_slug,
                "run_ids": [run_id],
                "results": [{"turns_run": 3, "stopped_reason": "clustering_saved"}],
            }
        ],
    )

    requests = list(iter_batch_run_requests(input_dir))

    assert len(requests) == 1
    request = requests[0]
    assert request.artifacts["turns_run"] == 3
    assert request.artifacts["batch_run_result"] == {
        "turns_run": 3,
        "stopped_reason": "clustering_saved",
    }


def test_run_evaluation_config_single_mode_writes_report_and_details(tmp_path: Path) -> None:
    input_dir = tmp_path / "batch_output"
    config_slug = "single-config"
    run_id = "001-run"
    _write_clustering(
        input_dir,
        config_slug=config_slug,
        run_id=run_id,
        payload={
            "clusters": [
                {"name": "A", "members": ["alpha", "beta"]},
                {"name": "B", "members": ["gamma"]},
            ]
        },
    )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Single"},
                "config_slug": config_slug,
                "run_ids": [run_id],
            }
        ],
    )

    report_path = tmp_path / "single_report.json"
    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["basic_metrics"],
        report_path=report_path,
    )

    returned_report_path = run_evaluation_config(config_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    details_path = report_path.parent / f"{report_path.stem}_details" / "basic_metrics.jsonl"
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert returned_report_path == str(report_path)
    assert report["evaluation_mode"] == "single"
    assert report["num_artifacts"] == 1
    assert report["num_artifacts_by_config"] == {config_slug: 1}
    assert report["summary_by_config"][config_slug]["num_artifacts"] == 1
    assert (
        report["summary_by_config"][config_slug]["summary_by_evaluator"]["basic_metrics"][
            "details"
        ]["num_results"]
        == 1
    )
    assert (
        "accounting"
        not in report["summary_by_config"][config_slug]["summary_by_evaluator"]["basic_metrics"]
    )
    assert report["summary_by_evaluator"]["basic_metrics"]["details"]["num_results"] == 1
    assert report["summary_by_evaluator"]["basic_metrics"]["details"]["methods"] == [
        "basic_metrics"
    ]
    assert "accounting" not in report["summary_by_evaluator"]["basic_metrics"]
    assert detail_records[0]["run_id"] == run_id
    assert detail_records[0]["config_slug"] == config_slug
    assert "conversation_trace.json" in detail_records[0]["conversation_trace_path"]


def test_evaluation_result_summary_omits_accounting_when_no_inputs_provide_it() -> None:
    summary = EvaluationResult.summarize(
        [
            EvaluationResult(
                method="basic_metrics",
                metrics={"score": 1.0},
                details={"num_clusters": 2.0},
            )
        ]
    )

    assert summary.accounting is None


def test_run_evaluation_config_single_mode_resume_skips_completed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conversational_clustering.evaluation import runner as module

    class CountingEvaluator:
        name = "basic_metrics"
        resume_version = "v1"

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, request: object) -> EvaluationResult:
            self.calls += 1
            return EvaluationResult(
                method=self.name,
                metrics={"score": 1.0},
                details={"num_clusters": 2.0},
            )

    counting_evaluator = CountingEvaluator()
    monkeypatch.setattr(module, "get_evaluator", lambda name, params=None: counting_evaluator)

    input_dir = tmp_path / "batch_output"
    _write_clustering(
        input_dir,
        config_slug="single-config",
        run_id="001-run",
        payload={
            "clusters": [
                {"name": "A", "members": ["alpha", "beta"]},
                {"name": "B", "members": ["gamma"]},
            ]
        },
    )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Single"},
                "config_slug": "single-config",
                "run_ids": ["001-run"],
            }
        ],
    )

    report_path = tmp_path / "single_report.json"
    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["basic_metrics"],
        report_path=report_path,
    )
    run_evaluation_config(config_path)
    assert counting_evaluator.calls == 1

    _write_clustering(
        input_dir,
        config_slug="single-config",
        run_id="002-run",
        payload={
            "clusters": [
                {"name": "A", "members": ["alpha", "beta"]},
                {"name": "B", "members": ["gamma"]},
            ]
        },
    )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Single"},
                "config_slug": "single-config",
                "run_ids": ["001-run", "002-run"],
            }
        ],
    )
    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["basic_metrics"],
        report_path=report_path,
        resume=True,
    )
    run_evaluation_config(config_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    details_path = report_path.parent / f"{report_path.stem}_details" / "basic_metrics.jsonl"
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert counting_evaluator.calls == 2
    assert report["resume"] is True
    assert report["num_artifacts"] == 2
    assert report["summary_by_evaluator"]["basic_metrics"]["details"]["num_results"] == 2
    assert len(detail_records) == 2


def test_run_evaluation_config_single_mode_resume_recomputes_when_inputs_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conversational_clustering.evaluation import runner as module

    class CountingEvaluator:
        name = "basic_metrics"
        resume_version = "v1"

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, request: object) -> EvaluationResult:
            assert isinstance(request, module.EvaluationRequest)
            self.calls += 1
            return EvaluationResult(
                method=self.name,
                metrics={"score": float(len(request.predictions.cluster_names))},
                details={"num_clusters": float(len(request.predictions.cluster_names))},
            )

    counting_evaluator = CountingEvaluator()
    monkeypatch.setattr(module, "get_evaluator", lambda name, params=None: counting_evaluator)

    input_dir = tmp_path / "batch_output"
    report_path = tmp_path / "single_report.json"
    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["basic_metrics"],
        report_path=report_path,
    )

    _write_clustering(
        input_dir,
        config_slug="single-config",
        run_id="001-run",
        payload={"clusters": [{"name": "A", "members": ["alpha", "beta"]}]},
    )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Single"},
                "config_slug": "single-config",
                "run_ids": ["001-run"],
            }
        ],
    )
    run_evaluation_config(config_path)
    assert counting_evaluator.calls == 1

    _write_clustering(
        input_dir,
        config_slug="single-config",
        run_id="001-run",
        payload={
            "clusters": [
                {"name": "A", "members": ["alpha"]},
                {"name": "B", "members": ["beta"]},
            ]
        },
    )
    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["basic_metrics"],
        report_path=report_path,
        resume=True,
    )
    run_evaluation_config(config_path)

    details_path = report_path.parent / f"{report_path.stem}_details" / "basic_metrics.jsonl"
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert counting_evaluator.calls == 2
    assert len(detail_records) == 1
    assert detail_records[0]["details"]["num_clusters"] == pytest.approx(2.0)


def test_run_evaluation_config_single_mode_resume_recomputes_when_evaluator_version_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conversational_clustering.evaluation import runner as module

    class CountingEvaluator:
        name = "basic_metrics"

        def __init__(self, *, resume_version: str) -> None:
            self.resume_version = resume_version
            self.calls = 0

        def evaluate(self, request: object) -> EvaluationResult:
            self.calls += 1
            return EvaluationResult(
                method=self.name,
                metrics={"score": 1.0},
                details={"num_clusters": 1.0},
            )

    evaluator_holder = {"current": CountingEvaluator(resume_version="v1")}
    monkeypatch.setattr(
        module,
        "get_evaluator",
        lambda name, params=None: evaluator_holder["current"],
    )

    input_dir = tmp_path / "batch_output"
    report_path = tmp_path / "single_report.json"
    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["basic_metrics"],
        report_path=report_path,
    )
    _write_clustering(
        input_dir,
        config_slug="single-config",
        run_id="001-run",
        payload={"clusters": [{"name": "A", "members": ["alpha", "beta"]}]},
    )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Single"},
                "config_slug": "single-config",
                "run_ids": ["001-run"],
            }
        ],
    )

    run_evaluation_config(config_path)
    first_evaluator = evaluator_holder["current"]
    assert first_evaluator.calls == 1

    evaluator_holder["current"] = CountingEvaluator(resume_version="v2")
    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["basic_metrics"],
        report_path=report_path,
        resume=True,
    )
    run_evaluation_config(config_path)

    second_evaluator = evaluator_holder["current"]
    assert second_evaluator.calls == 1

    details_path = report_path.parent / f"{report_path.stem}_details" / "basic_metrics.jsonl"
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert len(detail_records) == 1


def test_run_evaluation_config_single_mode_resume_recomputes_pre_accounting_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conversational_clustering.evaluation import runner as module

    accounting_payload = {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
        "estimated_cost_usd": 0.25,
        "cost_estimate_status": "estimated",
        "unpriced_models": [],
        "by_model": {},
    }

    class CountingEvaluator:
        name = "basic_metrics"
        resume_version = "v2"

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, request: object) -> EvaluationResult:
            self.calls += 1
            return EvaluationResult(
                method=self.name,
                metrics={"score": 1.0},
                details={"num_clusters": 2.0},
                accounting=accounting_payload,
            )

    class LegacyEvaluator:
        name = "basic_metrics"
        resume_version = "v1"

    evaluator = CountingEvaluator()
    monkeypatch.setattr(module, "get_evaluator", lambda name, params=None: evaluator)

    input_dir = tmp_path / "batch_output"
    report_path = tmp_path / "single_report.json"
    _write_clustering(
        input_dir,
        config_slug="single-config",
        run_id="001-run",
        payload={
            "clusters": [
                {"name": "A", "members": ["alpha", "beta"]},
                {"name": "B", "members": ["gamma"]},
            ]
        },
    )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Single"},
                "config_slug": "single-config",
                "run_ids": ["001-run"],
            }
        ],
    )

    request = next(iter_batch_run_requests(input_dir))
    legacy_fingerprint = module._build_single_resume_fingerprint(
        request=request,
        evaluator=cast(Evaluator, LegacyEvaluator()),
        evaluator_name="basic_metrics",
        evaluator_params=None,
    )
    details_path = report_path.parent / f"{report_path.stem}_details" / "basic_metrics.jsonl"
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(
        json.dumps(
            {
                "clustering_path": request.artifacts["clustering_path"],
                "conversation_trace_path": request.artifacts["conversation_trace_path"],
                "batch_run_config_path": request.artifacts["batch_run_config_path"],
                "config_slug": request.metadata["config_slug"],
                "run_id": request.metadata["run_id"],
                "method": "basic_metrics",
                "metrics": {"score": 0.0},
                "details": {"num_clusters": 0.0},
                "resume_fingerprint": legacy_fingerprint,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["basic_metrics"],
        report_path=report_path,
        resume=True,
    )
    run_evaluation_config(config_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert evaluator.calls == 1
    assert detail_records == [
        {
            "clustering_path": request.artifacts["clustering_path"],
            "conversation_trace_path": request.artifacts["conversation_trace_path"],
            "batch_run_config_path": request.artifacts["batch_run_config_path"],
            "config_slug": request.metadata["config_slug"],
            "run_id": request.metadata["run_id"],
            "method": "basic_metrics",
            "metrics": {"score": 1.0},
            "details": {"num_clusters": 2.0},
            "accounting": accounting_payload,
            "resume_fingerprint": module._build_single_resume_fingerprint(
                request=request,
                evaluator=cast(Evaluator, evaluator),
                evaluator_name="basic_metrics",
                evaluator_params=None,
            ),
        }
    ]
    assert report["summary_by_evaluator"]["basic_metrics"]["accounting"] == accounting_payload


def test_run_evaluation_config_single_mode_writes_per_config_summaries(tmp_path: Path) -> None:
    input_dir = tmp_path / "batch_output"
    clustering_payload = {
        "clusters": [
            {"name": "A", "members": ["alpha", "beta"]},
            {"name": "B", "members": ["gamma"]},
        ]
    }
    _write_clustering(
        input_dir,
        config_slug="single-config-a",
        run_id="001-run",
        payload=clustering_payload,
    )
    _write_clustering(
        input_dir,
        config_slug="single-config-a",
        run_id="002-run",
        payload=clustering_payload,
    )
    _write_clustering(
        input_dir,
        config_slug="single-config-b",
        run_id="003-run",
        payload=clustering_payload,
    )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Single A"},
                "config_slug": "single-config-a",
                "run_ids": ["001-run", "002-run"],
            },
            {
                "config": {"name": "Single B"},
                "config_slug": "single-config-b",
                "run_ids": ["003-run"],
            },
        ],
    )

    report_path = tmp_path / "single_report.json"
    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["basic_metrics"],
        report_path=report_path,
    )

    run_evaluation_config(config_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["evaluation_mode"] == "single"
    assert report["num_artifacts"] == 3
    assert report["num_artifacts_by_config"] == {"single-config-a": 2, "single-config-b": 1}
    assert report["summary_by_config"]["single-config-a"]["num_artifacts"] == 2
    assert report["summary_by_config"]["single-config-b"]["num_artifacts"] == 1
    assert (
        report["summary_by_config"]["single-config-a"]["summary_by_evaluator"]["basic_metrics"][
            "details"
        ]["num_results"]
        == 2
    )
    assert (
        report["summary_by_config"]["single-config-b"]["summary_by_evaluator"]["basic_metrics"][
            "details"
        ]["num_results"]
        == 1
    )
    assert report["summary_by_evaluator"]["basic_metrics"]["details"]["num_results"] == 3


def test_run_evaluation_config_single_mode_resume_rewrites_superseded_error_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conversational_clustering.evaluation import runner as module

    class FlakyThenStableEvaluator:
        name = "basic_metrics"
        resume_version = "v1"

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, request: object) -> EvaluationResult:
            self.calls += 1
            if self.calls == 1:
                raise ValueError("transient failure")
            return EvaluationResult(
                method=self.name,
                metrics={"score": 1.0},
                details={"num_clusters": 2.0},
            )

    evaluator = FlakyThenStableEvaluator()
    monkeypatch.setattr(module, "get_evaluator", lambda name, params=None: evaluator)

    input_dir = tmp_path / "batch_output"
    _write_clustering(
        input_dir,
        config_slug="single-config",
        run_id="001-run",
        payload={
            "clusters": [
                {"name": "A", "members": ["alpha", "beta"]},
                {"name": "B", "members": ["gamma"]},
            ]
        },
    )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Single"},
                "config_slug": "single-config",
                "run_ids": ["001-run"],
            }
        ],
    )

    report_path = tmp_path / "single_report.json"
    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["basic_metrics"],
        report_path=report_path,
    )
    with pytest.raises(EvaluationRunFailed):
        run_evaluation_config(config_path)

    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["basic_metrics"],
        report_path=report_path,
        resume=True,
    )
    run_evaluation_config(config_path)

    details_path = report_path.parent / f"{report_path.stem}_details" / "basic_metrics.jsonl"
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert evaluator.calls == 2
    assert len(detail_records) == 1
    assert "status" not in detail_records[0]
    assert detail_records[0]["metrics"]["score"] == pytest.approx(1.0)


def test_run_evaluation_config_single_mode_resume_persists_partial_progress_before_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conversational_clustering.evaluation import runner as module

    class InterruptingEvaluator:
        name = "basic_metrics"
        resume_version = "v1"

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, request: object) -> EvaluationResult:
            assert isinstance(request, module.EvaluationRequest)
            self.calls += 1
            run_id = str(request.metadata.get("run_id"))
            if run_id == "003-run":
                raise KeyboardInterrupt("synthetic interruption")
            return EvaluationResult(
                method=self.name,
                metrics={"score": float(self.calls)},
                details={"num_clusters": float(len(request.predictions.cluster_names))},
            )

    evaluator = InterruptingEvaluator()
    monkeypatch.setattr(module, "get_evaluator", lambda name, params=None: evaluator)

    input_dir = tmp_path / "batch_output"
    report_path = tmp_path / "single_report.json"

    _write_clustering(
        input_dir,
        config_slug="single-config",
        run_id="001-run",
        payload={"clusters": [{"name": "A", "members": ["alpha", "beta"]}]},
    )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Single"},
                "config_slug": "single-config",
                "run_ids": ["001-run"],
            }
        ],
    )
    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["basic_metrics"],
        report_path=report_path,
    )
    run_evaluation_config(config_path)

    for run_id in ("002-run", "003-run"):
        _write_clustering(
            input_dir,
            config_slug="single-config",
            run_id=run_id,
            payload={"clusters": [{"name": "A", "members": ["alpha", "beta"]}]},
        )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Single"},
                "config_slug": "single-config",
                "run_ids": ["001-run", "002-run", "003-run"],
            }
        ],
    )
    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["basic_metrics"],
        report_path=report_path,
        resume=True,
    )
    with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
        run_evaluation_config(config_path)

    details_path = report_path.parent / f"{report_path.stem}_details" / "basic_metrics.jsonl"
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert [record["run_id"] for record in detail_records] == ["001-run", "002-run"]


def test_run_evaluation_config_single_mode_resume_prunes_obsolete_detail_rows(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "batch_output"
    clustering_payload = {
        "clusters": [
            {"name": "A", "members": ["alpha", "beta"]},
            {"name": "B", "members": ["gamma"]},
        ]
    }
    for run_id in ("001-run", "002-run"):
        _write_clustering(
            input_dir,
            config_slug="single-config",
            run_id=run_id,
            payload=clustering_payload,
        )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Single"},
                "config_slug": "single-config",
                "run_ids": ["001-run", "002-run"],
            }
        ],
    )

    report_path = tmp_path / "single_report.json"
    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["basic_metrics"],
        report_path=report_path,
    )
    run_evaluation_config(config_path)

    run_dir_to_remove = input_dir / "single-config" / "002-run"
    (run_dir_to_remove / "clustering.json").unlink()
    (run_dir_to_remove / "conversation_trace.json").unlink()
    run_dir_to_remove.rmdir()
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Single"},
                "config_slug": "single-config",
                "run_ids": ["001-run"],
            }
        ],
    )
    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["basic_metrics"],
        report_path=report_path,
        resume=True,
    )
    run_evaluation_config(config_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    details_path = report_path.parent / f"{report_path.stem}_details" / "basic_metrics.jsonl"
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert report["num_artifacts"] == 1
    assert len(detail_records) == 1
    assert detail_records[0]["run_id"] == "001-run"


def test_run_evaluation_config_single_mode_conversation_trace_metrics(tmp_path: Path) -> None:
    input_dir = tmp_path / "batch_output"
    config_slug = "trace-config"
    run_id = "001-run"
    _write_clustering(
        input_dir,
        config_slug=config_slug,
        run_id=run_id,
        payload={"clusters": [{"name": "A", "members": ["alpha"]}]},
    )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Trace"},
                "config_slug": config_slug,
                "run_ids": [run_id],
            }
        ],
    )

    report_path = tmp_path / "trace_report.json"
    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["conversation_trace_metrics"],
        report_path=report_path,
    )

    run_evaluation_config(config_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    details_path = (
        report_path.parent / f"{report_path.stem}_details" / "conversation_trace_metrics.jsonl"
    )
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert report["evaluation_mode"] == "single"
    assert (
        report["summary_by_evaluator"]["conversation_trace_metrics"]["details"]["num_results"] == 1
    )
    assert report["summary_by_evaluator"]["conversation_trace_metrics"]["details"][
        "mean_numeric_details"
    ] == {
        "conversation_length_messages": 2.0,
        "num_failed_tool_calls": 0.0,
        "num_successful_tool_calls": 1.0,
        "num_tool_calls": 1.0,
    }
    assert detail_records[0]["details"]["conversation_length_messages"] == 2
    assert detail_records[0]["details"]["num_tool_calls"] == 1
    assert detail_records[0]["details"]["num_successful_tool_calls"] == 1
    assert detail_records[0]["details"]["num_failed_tool_calls"] == 0


def test_run_evaluation_config_trace_metrics_allows_unsaved_run_without_clustering(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "batch_output"
    config_slug = "trace-config"
    run_id = "001-run"
    run_dir = input_dir / config_slug / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "conversation_trace.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "human", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
                "total_tool_calls_by_type": {"propose_clustering": 1},
                "total_failed_tool_calls_by_type": {},
            }
        ),
        encoding="utf-8",
    )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Trace"},
                "config_slug": config_slug,
                "run_ids": [run_id],
                "results": [{"turns_run": 2, "stopped_reason": "max_turns"}],
            }
        ],
    )

    report_path = tmp_path / "trace_report.json"
    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["conversation_trace_metrics"],
        report_path=report_path,
    )

    run_evaluation_config(config_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    details_path = (
        report_path.parent / f"{report_path.stem}_details" / "conversation_trace_metrics.jsonl"
    )
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert report["num_artifacts"] == 1
    assert detail_records[0]["clustering_path"] is None
    assert detail_records[0]["details"]["conversation_length_messages"] == 2


def test_run_evaluation_config_single_mode_human_effort_metrics(tmp_path: Path) -> None:
    input_dir = tmp_path / "batch_output"
    config_slug = "effort-config"
    run_id = "001-run"
    _write_clustering(
        input_dir,
        config_slug=config_slug,
        run_id=run_id,
        payload={"clusters": [{"name": "A", "members": ["alpha"]}]},
    )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {"name": "Effort"},
                "config_slug": config_slug,
                "run_ids": [run_id],
            }
        ],
    )

    report_path = tmp_path / "effort_report.json"
    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["human_effort_metrics"],
        report_path=report_path,
    )

    run_evaluation_config(config_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    details_path = report_path.parent / f"{report_path.stem}_details" / "human_effort_metrics.jsonl"
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert report["evaluation_mode"] == "single"
    assert report["summary_by_evaluator"]["human_effort_metrics"]["details"]["num_results"] == 1
    assert report["summary_by_evaluator"]["human_effort_metrics"]["details"][
        "mean_numeric_details"
    ] == {
        "agent_turn_count": 1.0,
        "avg_agent_message_length": 1.0,
        "avg_human_message_length": 1.0,
        "human_turn_count": 1.0,
        "human_to_agent_word_ratio": 1.0,
        "turn_count": 2.0,
    }
    assert detail_records[0]["details"]["turn_count"] == 2
    assert detail_records[0]["details"]["human_turn_count"] == 1
    assert detail_records[0]["details"]["agent_turn_count"] == 1
    assert detail_records[0]["details"]["avg_human_message_length"] == 1.0
    assert detail_records[0]["details"]["avg_agent_message_length"] == 1.0
    assert detail_records[0]["details"]["human_to_agent_word_ratio"] == 1.0


def test_run_evaluation_config_single_mode_uses_slug_when_configs_have_missing_keys(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "batch_output"
    clustering_payload = {
        "clusters": [
            {"name": "A", "members": ["alpha", "beta"]},
            {"name": "B", "members": ["gamma"]},
        ]
    }
    _write_clustering(
        input_dir,
        config_slug="single-config-a",
        run_id="001-run",
        payload=clustering_payload,
    )
    _write_clustering(
        input_dir,
        config_slug="single-config-b",
        run_id="002-run",
        payload=clustering_payload,
    )
    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {
                    "name": "Single A",
                    "model": "gpt-4o-mini",
                    "human_agent_config": {"prompt_id": "prompt-a", "guidance": None},
                },
                "config_slug": "single-config-a",
                "run_ids": ["001-run"],
            },
            {
                "config": {
                    "name": "Single B",
                    "model": "gpt-4o-mini",
                    "human_agent_config": {"prompt_id": "prompt-b", "guidance": None},
                    "temperature": 0.2,
                },
                "config_slug": "single-config-b",
                "run_ids": ["002-run"],
            },
        ],
    )

    report_path = tmp_path / "single_report.json"
    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["basic_metrics"],
        report_path=report_path,
    )

    run_evaluation_config(config_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["evaluation_mode"] == "single"
    assert "vary_by" not in report
    assert report["num_artifacts_by_config"] == {"single-config-a": 1, "single-config-b": 1}


def test_run_evaluation_config_pairwise_mode_writes_pairwise_counts(tmp_path: Path) -> None:
    input_dir = tmp_path / "batch_output"
    config_a_slug = "pairwise-config-a"
    config_b_slug = "pairwise-config-b"
    run_ids_a = ["001-run", "002-run"]
    run_ids_b = ["003-run", "004-run"]
    clustering_payload = {
        "clusters": [
            {"name": "A", "members": ["alpha", "beta"]},
            {"name": "B", "members": ["gamma"]},
        ]
    }
    for run_id in run_ids_a:
        _write_clustering(
            input_dir,
            config_slug=config_a_slug,
            run_id=run_id,
            payload=clustering_payload,
        )
    for run_id in run_ids_b:
        _write_clustering(
            input_dir,
            config_slug=config_b_slug,
            run_id=run_id,
            payload=clustering_payload,
        )

    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {
                    "name": "Pairwise A",
                    "dataset_name": "toy",
                    "human_agent_config": {"prompt_id": "prompt-a", "guidance": None},
                },
                "config_slug": config_a_slug,
                "run_ids": run_ids_a,
            },
            {
                "config": {
                    "name": "Pairwise B",
                    "dataset_name": "toy",
                    "human_agent_config": {"prompt_id": "prompt-b", "guidance": None},
                },
                "config_slug": config_b_slug,
                "run_ids": run_ids_b,
            },
        ],
    )

    report_path = tmp_path / "pairwise_report.json"
    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["pairwise_consistency"],
        report_path=report_path,
        pairwise_scope="inter",
    )

    run_evaluation_config(config_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    details_path = report_path.parent / f"{report_path.stem}_details" / "pairwise_consistency.jsonl"
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]
    pair_key = (
        "{'prompt_id': 'prompt-a', 'guidance': None}->{'prompt_id': 'prompt-b', 'guidance': None}"
    )

    assert report["evaluation_mode"] == "pairwise"
    assert report["pairwise_scope"] == "inter"
    assert report["vary_by"] == "human_agent_config"
    assert report["num_artifacts"] == 4
    assert report["num_configs"] == 2
    assert report["vary_values"] == [
        {"prompt_id": "prompt-a", "guidance": None},
        {"prompt_id": "prompt-b", "guidance": None},
    ]
    assert report["num_pairwise_requests"] == 4
    assert report["num_pairwise_requests_by_vary_value_pair"] == {pair_key: 4}
    assert report["summary_by_vary_value_pair"][pair_key]["num_pairwise_requests"] == 4
    assert (
        report["summary_by_vary_value_pair"][pair_key]["summary_by_evaluator"][
            "pairwise_consistency"
        ]["details"]["num_results"]
        == 4
    )
    assert "summary_by_vary_value" not in report
    assert report["summary_by_evaluator"]["pairwise_consistency"]["metrics"][
        "rand_index"
    ] == pytest.approx(1.0)
    assert {record["left_config_slug"] for record in detail_records} == {config_a_slug}
    assert {record["right_config_slug"] for record in detail_records} == {config_b_slug}
    assert {record["left_conversation_trace_path"] for record in detail_records} == {
        str(input_dir / config_a_slug / run_id / "conversation_trace.json") for run_id in run_ids_a
    }
    assert {record["right_conversation_trace_path"] for record in detail_records} == {
        str(input_dir / config_b_slug / run_id / "conversation_trace.json") for run_id in run_ids_b
    }


def test_run_evaluation_config_pairwise_mode_skips_unsaved_runs_without_clustering(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "batch_output"
    config_a_slug = "pairwise-config-a"
    config_b_slug = "pairwise-config-b"
    clustering_payload = {
        "clusters": [
            {"name": "A", "members": ["alpha", "beta"]},
            {"name": "B", "members": ["gamma"]},
        ]
    }
    _write_clustering(
        input_dir,
        config_slug=config_a_slug,
        run_id="001-saved",
        payload=clustering_payload,
    )
    _write_clustering(
        input_dir,
        config_slug=config_b_slug,
        run_id="002-saved",
        payload=clustering_payload,
    )

    _write_results_jsonl(
        input_dir,
        records=[
            {
                "config": {
                    "name": "Pairwise A",
                    "dataset_name": "toy",
                    "human_agent_config": {"prompt_id": "prompt-a", "guidance": None},
                },
                "config_slug": config_a_slug,
                "run_ids": ["001-saved", "001-unsaved"],
                "results": [
                    {"turns_run": 2, "stopped_reason": "clustering_saved"},
                    {"turns_run": 8, "stopped_reason": "max_turns"},
                ],
            },
            {
                "config": {
                    "name": "Pairwise B",
                    "dataset_name": "toy",
                    "human_agent_config": {"prompt_id": "prompt-b", "guidance": None},
                },
                "config_slug": config_b_slug,
                "run_ids": ["002-saved"],
                "results": [{"turns_run": 2, "stopped_reason": "clustering_saved"}],
            },
        ],
    )

    report_path = tmp_path / "pairwise_report.json"
    config_path = _write_eval_config(
        tmp_path,
        input_dir=input_dir,
        evaluators=["pairwise_consistency"],
        report_path=report_path,
        pairwise_scope="inter",
    )

    run_evaluation_config(config_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["num_artifacts"] == 2
    assert report["num_artifacts_by_config"] == {
        config_a_slug: 1,
        config_b_slug: 1,
    }
    assert report["num_pairwise_requests"] == 1
