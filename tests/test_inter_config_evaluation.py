"""Integration-focused tests for inter-config pairwise evaluation behavior."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from conversational_clustering.evaluation.base import EvaluationResult
from conversational_clustering.evaluation.pairwise import (
    iter_pairwise_requests,
    load_pairwise_inputs,
    validate_experiment_records,
)
from conversational_clustering.evaluation.runner import (
    _build_branch_inter_pairwise_plan,
    _build_sweep_inter_pairwise_plan,
    parse_evaluation_run_config,
    run_evaluation_config,
)


def _write_clustering(
    input_dir: Path,
    *,
    config_slug: str,
    run_id: str,
    write_clustering_pov_history: bool = False,
) -> None:
    clustering_path = input_dir / config_slug / run_id / "clustering.json"
    clustering_path.parent.mkdir(parents=True, exist_ok=True)
    clustering_path.write_text(
        json.dumps(
            {
                "clusters": [
                    {"name": "A", "members": ["alpha", "beta"]},
                    {"name": "B", "members": ["gamma"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    clustering_path.with_name("conversation_trace.json").write_text(
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
    if write_clustering_pov_history:
        clustering_path.with_name("clustering_pov_history.json").write_text(
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


def _write_results_jsonl(input_dir: Path, records: list[dict[str, object]]) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    results_path = input_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def _write_pairwise_eval_config(
    directory: Path,
    *,
    input_dir: Path,
    report_path: Path,
    pairwise_scope: str | None = None,
    resume: bool = False,
    extra_lines: list[str] | None = None,
) -> Path:
    # Use TOML literal strings so Windows backslashes are not treated as escapes.
    lines = [
        f"input_dir = '{input_dir}'",
        'evaluators = ["pairwise_consistency"]',
    ]
    if pairwise_scope:
        lines.append(f'pairwise_scope = "{pairwise_scope}"')
    if resume:
        lines.append("resume = true")
    lines.append(f"report_path = '{report_path}'")
    if extra_lines:
        lines.extend(extra_lines)

    config_path = directory / "pairwise_evaluation.toml"
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def _config_record(
    *,
    config_slug: str,
    run_ids: list[str],
    prompt_id: str,
    dataset_name: str = "toy",
    extra_config_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    config_payload: dict[str, object] = {
        "name": f"human_agent_config_{prompt_id}",
        "dataset_name": dataset_name,
        "human_agent_config": {"prompt_id": prompt_id, "guidance": None},
        "model": "gpt-4o-mini",
    }
    if extra_config_fields:
        config_payload.update(extra_config_fields)

    return {
        "config_slug": config_slug,
        "run_ids": run_ids,
        "config": config_payload,
    }


def test_parse_pairwise_config_defaults_and_rejects_invalid_fields(tmp_path: Path) -> None:
    valid_config = _write_pairwise_eval_config(
        tmp_path,
        input_dir=tmp_path / "batch_output",
        report_path=tmp_path / "report.json",
    )

    parsed = parse_evaluation_run_config(valid_config)
    assert parsed.evaluation_mode == "pairwise"
    assert parsed.pairwise_scope == "inter"

    invalid_vary_by = _write_pairwise_eval_config(
        tmp_path,
        input_dir=tmp_path / "batch_output",
        report_path=tmp_path / "report.json",
        extra_lines=['vary_by = "dataset_name"'],
    )
    with pytest.raises(ValueError, match="unsupported top-level key"):
        parse_evaluation_run_config(invalid_vary_by)

    invalid_key = _write_pairwise_eval_config(
        tmp_path,
        input_dir=tmp_path / "batch_output",
        report_path=tmp_path / "report.json",
        extra_lines=["unexpected = 123"],
    )
    with pytest.raises(ValueError, match="unsupported top-level key"):
        parse_evaluation_run_config(invalid_key)


def test_run_evaluation_config_pairwise_inter_writes_expected_report_shape(tmp_path: Path) -> None:
    input_dir = tmp_path / "batch_output"
    report_path = tmp_path / "inter_report.json"

    run_ids_by_config = {
        "cfg-a": ["a1", "a2"],
        "cfg-b": ["b1", "b2"],
    }
    for config_slug, run_ids in run_ids_by_config.items():
        for run_id in run_ids:
            _write_clustering(input_dir, config_slug=config_slug, run_id=run_id)

    _write_results_jsonl(
        input_dir,
        records=[
            _config_record(config_slug="cfg-a", run_ids=["a1", "a2"], prompt_id="prompt-a"),
            _config_record(config_slug="cfg-b", run_ids=["b1", "b2"], prompt_id="prompt-b"),
        ],
    )

    config_path = _write_pairwise_eval_config(
        tmp_path,
        input_dir=input_dir,
        report_path=report_path,
        pairwise_scope="inter",
    )
    run_evaluation_config(config_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    details_path = report_path.parent / f"{report_path.stem}_details" / "pairwise_consistency.jsonl"
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert report["evaluation_mode"] == "pairwise"
    assert report["dataset_name"] == "toy"
    assert report["pairwise_scope"] == "inter"
    assert report["vary_by"] == "human_agent_config"
    assert report["num_artifacts"] == 4
    assert report["num_configs"] == 2
    pair_key = (
        "{'prompt_id': 'prompt-a', 'guidance': None}->{'prompt_id': 'prompt-b', 'guidance': None}"
    )
    assert report["vary_values"] == [
        {"prompt_id": "prompt-a", "guidance": None},
        {"prompt_id": "prompt-b", "guidance": None},
    ]
    assert report["num_artifacts_by_config"] == {"cfg-a": 2, "cfg-b": 2}
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

    assert len(detail_records) == 4
    assert all(record["pairwise_scope"] == "inter" for record in detail_records)
    assert {record["left_config_slug"] for record in detail_records} == {"cfg-a"}
    assert {record["right_config_slug"] for record in detail_records} == {"cfg-b"}


def test_run_evaluation_config_pairwise_inter_resume_skips_completed_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conversational_clustering.evaluation import runner as module

    class CountingPairwiseEvaluator:
        name = "pairwise_consistency"
        resume_version = "v1"

        def __init__(self) -> None:
            self.calls = 0

        def evaluate(self, request: object) -> EvaluationResult:
            self.calls += 1
            return EvaluationResult(
                method=self.name,
                metrics={"adjusted_rand_index": 1.0},
                details={"num_common_items": 3.0},
            )

    counting_evaluator = CountingPairwiseEvaluator()
    monkeypatch.setattr(
        module,
        "get_pairwise_evaluator",
        lambda name, params=None: counting_evaluator,
    )

    input_dir = tmp_path / "batch_output"
    report_path = tmp_path / "inter_report.json"

    for config_slug, run_id in (("cfg-a", "a1"), ("cfg-b", "b1")):
        _write_clustering(input_dir, config_slug=config_slug, run_id=run_id)

    _write_results_jsonl(
        input_dir,
        records=[
            _config_record(config_slug="cfg-a", run_ids=["a1"], prompt_id="prompt-a"),
            _config_record(config_slug="cfg-b", run_ids=["b1"], prompt_id="prompt-b"),
        ],
    )

    config_path = _write_pairwise_eval_config(
        tmp_path,
        input_dir=input_dir,
        report_path=report_path,
        pairwise_scope="inter",
    )
    run_evaluation_config(config_path)
    assert counting_evaluator.calls == 1

    _write_clustering(input_dir, config_slug="cfg-c", run_id="c1")
    _write_results_jsonl(
        input_dir,
        records=[
            _config_record(config_slug="cfg-a", run_ids=["a1"], prompt_id="prompt-a"),
            _config_record(config_slug="cfg-b", run_ids=["b1"], prompt_id="prompt-b"),
            _config_record(config_slug="cfg-c", run_ids=["c1"], prompt_id="prompt-c"),
        ],
    )

    config_path = _write_pairwise_eval_config(
        tmp_path,
        input_dir=input_dir,
        report_path=report_path,
        pairwise_scope="inter",
        resume=True,
    )
    run_evaluation_config(config_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    details_path = report_path.parent / f"{report_path.stem}_details" / "pairwise_consistency.jsonl"
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert counting_evaluator.calls == 3
    assert report["resume"] is True
    assert report["num_pairwise_requests"] == 3
    assert report["summary_by_evaluator"]["pairwise_consistency"]["details"]["num_results"] == 3
    assert len(detail_records) == 3


def test_run_evaluation_config_pairwise_inter_resume_recomputes_when_evaluator_version_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conversational_clustering.evaluation import runner as module

    class CountingPairwiseEvaluator:
        name = "pairwise_consistency"

        def __init__(self, *, resume_version: str) -> None:
            self.resume_version = resume_version
            self.calls = 0

        def evaluate(self, request: object) -> EvaluationResult:
            self.calls += 1
            return EvaluationResult(
                method=self.name,
                metrics={"adjusted_rand_index": 1.0},
                details={"num_common_items": 3.0},
            )

    evaluator_holder = {"current": CountingPairwiseEvaluator(resume_version="v1")}
    monkeypatch.setattr(
        module,
        "get_pairwise_evaluator",
        lambda name, params=None: evaluator_holder["current"],
    )

    input_dir = tmp_path / "batch_output"
    report_path = tmp_path / "inter_report.json"

    for config_slug, run_id in (("cfg-a", "a1"), ("cfg-b", "b1")):
        _write_clustering(input_dir, config_slug=config_slug, run_id=run_id)

    _write_results_jsonl(
        input_dir,
        records=[
            _config_record(config_slug="cfg-a", run_ids=["a1"], prompt_id="prompt-a"),
            _config_record(config_slug="cfg-b", run_ids=["b1"], prompt_id="prompt-b"),
        ],
    )

    config_path = _write_pairwise_eval_config(
        tmp_path,
        input_dir=input_dir,
        report_path=report_path,
        pairwise_scope="inter",
    )
    run_evaluation_config(config_path)
    first_evaluator = evaluator_holder["current"]
    assert first_evaluator.calls == 1

    evaluator_holder["current"] = CountingPairwiseEvaluator(resume_version="v2")
    config_path = _write_pairwise_eval_config(
        tmp_path,
        input_dir=input_dir,
        report_path=report_path,
        pairwise_scope="inter",
        resume=True,
    )
    run_evaluation_config(config_path)
    second_evaluator = evaluator_holder["current"]
    assert second_evaluator.calls == 1

    details_path = report_path.parent / f"{report_path.stem}_details" / "pairwise_consistency.jsonl"
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert len(detail_records) == 1


def test_run_evaluation_config_pairwise_inter_resume_prunes_obsolete_detail_rows(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "batch_output"
    report_path = tmp_path / "inter_report.json"

    for config_slug, run_id in (("cfg-a", "a1"), ("cfg-b", "b1"), ("cfg-c", "c1")):
        _write_clustering(input_dir, config_slug=config_slug, run_id=run_id)

    _write_results_jsonl(
        input_dir,
        records=[
            _config_record(config_slug="cfg-a", run_ids=["a1"], prompt_id="prompt-a"),
            _config_record(config_slug="cfg-b", run_ids=["b1"], prompt_id="prompt-b"),
            _config_record(config_slug="cfg-c", run_ids=["c1"], prompt_id="prompt-c"),
        ],
    )

    config_path = _write_pairwise_eval_config(
        tmp_path,
        input_dir=input_dir,
        report_path=report_path,
        pairwise_scope="inter",
    )
    run_evaluation_config(config_path)

    run_dir_to_remove = input_dir / "cfg-c" / "c1"
    (run_dir_to_remove / "clustering.json").unlink()
    (run_dir_to_remove / "conversation_trace.json").unlink()
    run_dir_to_remove.rmdir()
    (input_dir / "cfg-c").rmdir()
    _write_results_jsonl(
        input_dir,
        records=[
            _config_record(config_slug="cfg-a", run_ids=["a1"], prompt_id="prompt-a"),
            _config_record(config_slug="cfg-b", run_ids=["b1"], prompt_id="prompt-b"),
        ],
    )

    config_path = _write_pairwise_eval_config(
        tmp_path,
        input_dir=input_dir,
        report_path=report_path,
        pairwise_scope="inter",
        resume=True,
    )
    run_evaluation_config(config_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    details_path = report_path.parent / f"{report_path.stem}_details" / "pairwise_consistency.jsonl"
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert report["num_pairwise_requests"] == 1
    assert len(detail_records) == 1
    assert {detail_records[0]["left_config_slug"], detail_records[0]["right_config_slug"]} == {
        "cfg-a",
        "cfg-b",
    }


def test_run_evaluation_config_pairwise_inter_rejects_inference_when_config_keys_differ(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "batch_output"
    report_path = tmp_path / "inter_report.json"

    for config_slug, run_id in [("cfg-a", "a1"), ("cfg-b", "b1")]:
        _write_clustering(input_dir, config_slug=config_slug, run_id=run_id)

    _write_results_jsonl(
        input_dir,
        records=[
            _config_record(
                config_slug="cfg-a",
                run_ids=["a1"],
                prompt_id="prompt-a",
                extra_config_fields={"temperature": 0.2, "name": "cfg-a"},
            ),
            _config_record(
                config_slug="cfg-b",
                run_ids=["b1"],
                prompt_id="prompt-b",
                extra_config_fields={"name": "cfg-b"},
            ),
        ],
    )

    config_path = _write_pairwise_eval_config(
        tmp_path,
        input_dir=input_dir,
        report_path=report_path,
        pairwise_scope="inter",
    )
    with pytest.raises(ValueError, match="Could not infer a single vary-by key"):
        run_evaluation_config(config_path)


def test_iter_pairwise_requests_includes_batch_run_configs_in_artifacts(tmp_path: Path) -> None:
    input_dir = tmp_path / "batch_output"
    for config_slug, run_id in [("cfg-a", "a1"), ("cfg-b", "b1")]:
        _write_clustering(
            input_dir,
            config_slug=config_slug,
            run_id=run_id,
            write_clustering_pov_history=True,
        )

    _write_results_jsonl(
        input_dir,
        records=[
            _config_record(
                config_slug="cfg-a",
                run_ids=["a1"],
                prompt_id="prompt-a",
                extra_config_fields={"temperature": 0.1},
            ),
            _config_record(
                config_slug="cfg-b",
                run_ids=["b1"],
                prompt_id="prompt-a",
                extra_config_fields={"temperature": 0.9},
            ),
        ],
    )

    records = load_pairwise_inputs(input_dir)
    requests = list(
        iter_pairwise_requests(
            records=records,
            vary_by="temperature",
            pairwise_scope="inter",
        )
    )

    assert len(requests) == 1
    request = requests[0]
    assert request.artifacts["left_batch_run_config"]["temperature"] == 0.1
    assert request.artifacts["right_batch_run_config"]["temperature"] == 0.9
    assert request.artifacts["left_clustering_pov_history"]["num_snapshots"] == 2
    assert request.artifacts["right_clustering_pov_history"]["num_snapshots"] == 2


def test_run_evaluation_config_pairwise_branch_inter_writes_expected_report_shape(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "batch_output"
    report_path = tmp_path / "branch_inter_report.json"

    branching_configs = [
        ("cfg-s1-low", "s1l1", "01-branch-group", "prompt-a", 0.1),
        ("cfg-s1-high", "s1h1", "01-branch-group", "prompt-a", 0.9),
        ("cfg-s2-low", "s2l1", "02-branch-group", "prompt-b", 0.1),
        ("cfg-s2-high", "s2h1", "02-branch-group", "prompt-b", 0.9),
    ]
    for config_slug, run_id, _, _, _ in branching_configs:
        _write_clustering(input_dir, config_slug=config_slug, run_id=run_id)

    _write_results_jsonl(
        input_dir,
        records=[
            _config_record(
                config_slug=config_slug,
                run_ids=[run_id],
                prompt_id=prompt_id,
                extra_config_fields={
                    "temperature": temperature,
                    "metadata": {
                        "branch_group": branch_group,
                        "branch_vary_by": "temperature",
                    },
                },
            )
            for config_slug, run_id, branch_group, prompt_id, temperature in branching_configs
        ],
    )

    config_path = _write_pairwise_eval_config(
        tmp_path,
        input_dir=input_dir,
        report_path=report_path,
        pairwise_scope="branch_inter",
    )
    run_evaluation_config(config_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    details_path = report_path.parent / f"{report_path.stem}_details" / "pairwise_consistency.jsonl"
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert report["evaluation_mode"] == "pairwise"
    assert report["dataset_name"] == "toy"
    assert report["pairwise_scope"] == "branch_inter"
    assert report["vary_by"] == "temperature"
    assert report["num_artifacts"] == 4
    assert report["num_configs"] == 4
    assert report["vary_values"] == [0.1, 0.9]
    assert report["num_artifacts_by_config"] == {
        "cfg-s1-high": 1,
        "cfg-s1-low": 1,
        "cfg-s2-high": 1,
        "cfg-s2-low": 1,
    }
    assert report["num_pairwise_requests"] == 2
    assert len(report["num_pairwise_requests_by_vary_value_pair"]) == 2
    assert all(count == 1 for count in report["num_pairwise_requests_by_vary_value_pair"].values())
    assert len(report["summary_by_vary_value_pair"]) == 2
    assert all(
        group["num_pairwise_requests"] == 1
        for group in report["summary_by_vary_value_pair"].values()
    )

    assert len(detail_records) == 2
    assert all(record["pairwise_scope"] == "branch_inter" for record in detail_records)
    assert all(record["vary_by"] == "temperature" for record in detail_records)
    assert {record["left_vary_value"]["branch_group"] for record in detail_records} == {
        "01-branch-group",
        "02-branch-group",
    }
    assert all(
        record["left_vary_value"]["branch_group"] == record["right_vary_value"]["branch_group"]
        for record in detail_records
    )
    assert all(
        record["left_vary_value"]["branch_value"] != record["right_vary_value"]["branch_value"]
        for record in detail_records
    )


def test_build_branch_inter_pairwise_plan_streams_requests(tmp_path: Path) -> None:
    input_dir = tmp_path / "batch_output"

    branching_configs = [
        ("cfg-s1-low", "s1l1", "01-branch-group", "prompt-a", 0.1),
        ("cfg-s1-high", "s1h1", "01-branch-group", "prompt-a", 0.9),
        ("cfg-s2-low", "s2l1", "02-branch-group", "prompt-b", 0.1),
        ("cfg-s2-high", "s2h1", "02-branch-group", "prompt-b", 0.9),
    ]
    for config_slug, run_id, _, _, _ in branching_configs:
        _write_clustering(input_dir, config_slug=config_slug, run_id=run_id)

    _write_results_jsonl(
        input_dir,
        records=[
            _config_record(
                config_slug=config_slug,
                run_ids=[run_id],
                prompt_id=prompt_id,
                extra_config_fields={
                    "temperature": temperature,
                    "metadata": {
                        "branch_group": branch_group,
                        "branch_vary_by": "temperature",
                    },
                },
            )
            for config_slug, run_id, branch_group, prompt_id, temperature in branching_configs
        ],
    )

    records = load_pairwise_inputs(input_dir)
    branch_plan = _build_branch_inter_pairwise_plan(records=records)

    assert isinstance(branch_plan.requests, Iterator)
    assert branch_plan.vary_values == [0.1, 0.9]
    requests = list(branch_plan.requests)
    assert len(requests) == 2
    assert all("left_batch_run_config" in request.artifacts for request in requests)
    assert all("right_batch_run_config" in request.artifacts for request in requests)
    assert all(
        request.artifacts["left_batch_run_config"]["temperature"] in {0.1, 0.9}
        for request in requests
    )
    assert all(
        request.artifacts["right_batch_run_config"]["temperature"] in {0.1, 0.9}
        for request in requests
    )


def test_run_evaluation_config_pairwise_sweep_inter_writes_expected_report_shape(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "batch_output"
    report_path = tmp_path / "sweep_inter_report.json"

    branching_configs = [
        ("cfg-s1-low", "s1l1", "01-branch-group", "prompt-a", 0.1),
        ("cfg-s1-high", "s1h1", "01-branch-group", "prompt-a", 0.9),
        ("cfg-s2-low", "s2l1", "02-branch-group", "prompt-b", 0.1),
        ("cfg-s2-high", "s2h1", "02-branch-group", "prompt-b", 0.9),
    ]
    for config_slug, run_id, _, _, _ in branching_configs:
        _write_clustering(input_dir, config_slug=config_slug, run_id=run_id)

    _write_results_jsonl(
        input_dir,
        records=[
            _config_record(
                config_slug=config_slug,
                run_ids=[run_id],
                prompt_id=prompt_id,
                extra_config_fields={
                    "temperature": temperature,
                    "metadata": {
                        "branch_group": branch_group,
                        "branch_vary_by": "temperature",
                    },
                },
            )
            for config_slug, run_id, branch_group, prompt_id, temperature in branching_configs
        ],
    )

    config_path = _write_pairwise_eval_config(
        tmp_path,
        input_dir=input_dir,
        report_path=report_path,
        pairwise_scope="sweep_inter",
    )
    run_evaluation_config(config_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    details_path = report_path.parent / f"{report_path.stem}_details" / "pairwise_consistency.jsonl"
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert report["evaluation_mode"] == "pairwise"
    assert report["dataset_name"] == "toy"
    assert report["pairwise_scope"] == "sweep_inter"
    assert report["vary_by"] == "branch_group"
    assert report["num_artifacts"] == 4
    assert report["num_configs"] == 4
    assert report["vary_values"] == ["01-branch-group", "02-branch-group"]
    assert report["num_artifacts_by_config"] == {
        "cfg-s1-high": 1,
        "cfg-s1-low": 1,
        "cfg-s2-high": 1,
        "cfg-s2-low": 1,
    }
    assert report["num_pairwise_requests"] == 2
    assert len(report["num_pairwise_requests_by_vary_value_pair"]) == 2
    assert all(count == 1 for count in report["num_pairwise_requests_by_vary_value_pair"].values())
    assert len(report["summary_by_vary_value_pair"]) == 2
    assert all(
        group["num_pairwise_requests"] == 1
        for group in report["summary_by_vary_value_pair"].values()
    )

    assert len(detail_records) == 2
    assert all(record["pairwise_scope"] == "sweep_inter" for record in detail_records)
    assert all(record["vary_by"] == "branch_group" for record in detail_records)
    assert all(
        record["left_vary_value"]["branch_value"] == record["right_vary_value"]["branch_value"]
        for record in detail_records
    )
    assert all(
        record["left_vary_value"]["branch_group"] != record["right_vary_value"]["branch_group"]
        for record in detail_records
    )


def test_build_sweep_inter_pairwise_plan_streams_requests(tmp_path: Path) -> None:
    input_dir = tmp_path / "batch_output"

    branching_configs = [
        ("cfg-s1-low", "s1l1", "01-branch-group", "prompt-a", 0.1),
        ("cfg-s1-high", "s1h1", "01-branch-group", "prompt-a", 0.9),
        ("cfg-s2-low", "s2l1", "02-branch-group", "prompt-b", 0.1),
        ("cfg-s2-high", "s2h1", "02-branch-group", "prompt-b", 0.9),
    ]
    for config_slug, run_id, _, _, _ in branching_configs:
        _write_clustering(input_dir, config_slug=config_slug, run_id=run_id)

    _write_results_jsonl(
        input_dir,
        records=[
            _config_record(
                config_slug=config_slug,
                run_ids=[run_id],
                prompt_id=prompt_id,
                extra_config_fields={
                    "temperature": temperature,
                    "metadata": {
                        "branch_group": branch_group,
                        "branch_vary_by": "temperature",
                    },
                },
            )
            for config_slug, run_id, branch_group, prompt_id, temperature in branching_configs
        ],
    )

    records = load_pairwise_inputs(input_dir)
    sweep_plan = _build_sweep_inter_pairwise_plan(records=records)

    assert isinstance(sweep_plan.requests, Iterator)
    assert sweep_plan.vary_values == ["01-branch-group", "02-branch-group"]
    requests = list(sweep_plan.requests)
    assert len(requests) == 2
    assert all("left_batch_run_config" in request.artifacts for request in requests)
    assert all("right_batch_run_config" in request.artifacts for request in requests)


def test_validate_experiment_records_rejects_branch_scope_with_consistent_valid_scopes(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "batch_output"
    for config_slug, run_id in [("cfg-a", "a1"), ("cfg-b", "b1")]:
        _write_clustering(input_dir, config_slug=config_slug, run_id=run_id)

    _write_results_jsonl(
        input_dir,
        records=[
            _config_record(config_slug="cfg-a", run_ids=["a1"], prompt_id="prompt-a"),
            _config_record(config_slug="cfg-b", run_ids=["b1"], prompt_id="prompt-b"),
        ],
    )

    records = load_pairwise_inputs(input_dir)
    with pytest.raises(ValueError, match="Expected one of: inter, intra"):
        validate_experiment_records(
            records=records,
            vary_by="human_agent_config",
            pairwise_scope="branch_inter",
        )


def test_run_evaluation_config_pairwise_intra_writes_expected_report_shape(tmp_path: Path) -> None:
    input_dir = tmp_path / "batch_output"
    report_path = tmp_path / "intra_report.json"

    run_ids = ["a1", "a2", "a3"]
    for run_id in run_ids:
        _write_clustering(input_dir, config_slug="cfg-a", run_id=run_id)

    _write_results_jsonl(
        input_dir,
        records=[_config_record(config_slug="cfg-a", run_ids=run_ids, prompt_id="prompt-a")],
    )

    config_path = _write_pairwise_eval_config(
        tmp_path,
        input_dir=input_dir,
        report_path=report_path,
        pairwise_scope="intra",
    )
    run_evaluation_config(config_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    details_path = report_path.parent / f"{report_path.stem}_details" / "pairwise_consistency.jsonl"
    detail_records = [
        json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert report["evaluation_mode"] == "pairwise"
    assert report["pairwise_scope"] == "intra"
    assert report["num_artifacts"] == 3
    assert report["num_configs"] == 1
    assert "vary_by" not in report
    assert "vary_values" not in report
    assert report["num_artifacts_by_config"] == {"cfg-a": 3}
    assert report["num_pairwise_requests"] == 3
    assert report["num_pairwise_requests_by_config_slug"] == {"cfg-a": 3}
    assert report["summary_by_config_slug"]["cfg-a"]["num_pairwise_requests"] == 3
    assert (
        report["summary_by_config_slug"]["cfg-a"]["summary_by_evaluator"]["pairwise_consistency"][
            "details"
        ]["num_results"]
        == 3
    )

    assert len(detail_records) == 3
    assert all(record["pairwise_scope"] == "intra" for record in detail_records)
    assert all("vary_by" not in record for record in detail_records)
    assert all("left_vary_value" not in record for record in detail_records)
    assert all("right_vary_value" not in record for record in detail_records)


def test_run_evaluation_config_pairwise_intra_single_config_non_sweep_name_does_not_require_vary_by(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "batch_output"
    report_path = tmp_path / "intra_report.json"

    run_ids = ["a1", "a2", "a3"]
    for run_id in run_ids:
        _write_clustering(input_dir, config_slug="cfg-a", run_id=run_id)

    _write_results_jsonl(
        input_dir,
        records=[
            _config_record(
                config_slug="cfg-a",
                run_ids=run_ids,
                prompt_id="prompt-a",
                extra_config_fields={"name": "cfg-a"},
            )
        ],
    )

    config_path = _write_pairwise_eval_config(
        tmp_path,
        input_dir=input_dir,
        report_path=report_path,
        pairwise_scope="intra",
    )
    run_evaluation_config(config_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["evaluation_mode"] == "pairwise"
    assert report["pairwise_scope"] == "intra"
    assert "vary_by" not in report
    assert report["num_pairwise_requests"] == 3


def test_run_evaluation_config_pairwise_intra_writes_per_vary_value_summaries(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "batch_output"
    report_path = tmp_path / "intra_report.json"

    run_ids_by_config = {
        "cfg-a": ["a1", "a2", "a3"],
        "cfg-b": ["b1", "b2", "b3"],
    }
    for config_slug, run_ids in run_ids_by_config.items():
        for run_id in run_ids:
            _write_clustering(input_dir, config_slug=config_slug, run_id=run_id)

    _write_results_jsonl(
        input_dir,
        records=[
            _config_record(config_slug="cfg-a", run_ids=["a1", "a2", "a3"], prompt_id="prompt-a"),
            _config_record(config_slug="cfg-b", run_ids=["b1", "b2", "b3"], prompt_id="prompt-b"),
        ],
    )

    config_path = _write_pairwise_eval_config(
        tmp_path,
        input_dir=input_dir,
        report_path=report_path,
        pairwise_scope="intra",
    )
    run_evaluation_config(config_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    prompt_a_key = "{'prompt_id': 'prompt-a', 'guidance': None}"
    prompt_b_key = "{'prompt_id': 'prompt-b', 'guidance': None}"
    prompt_a_pair_key = f"{prompt_a_key}->{prompt_a_key}"
    prompt_b_pair_key = f"{prompt_b_key}->{prompt_b_key}"

    assert report["num_pairwise_requests"] == 6
    assert report["num_pairwise_requests_by_vary_value_pair"] == {
        prompt_a_pair_key: 3,
        prompt_b_pair_key: 3,
    }
    assert report["summary_by_vary_value_pair"][prompt_a_pair_key]["num_pairwise_requests"] == 3
    assert report["summary_by_vary_value_pair"][prompt_b_pair_key]["num_pairwise_requests"] == 3
    assert report["summary_by_vary_value"][prompt_a_key]["num_pairwise_requests"] == 3
    assert (
        report["summary_by_vary_value"][prompt_a_key]["summary_by_evaluator"][
            "pairwise_consistency"
        ]["details"]["num_results"]
        == 3
    )
    assert report["summary_by_vary_value"][prompt_b_key]["num_pairwise_requests"] == 3
    assert (
        report["summary_by_vary_value"][prompt_b_key]["summary_by_evaluator"][
            "pairwise_consistency"
        ]["details"]["num_results"]
        == 3
    )


def test_run_evaluation_config_pairwise_intra_includes_zero_request_vary_value_summary(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "batch_output"
    report_path = tmp_path / "intra_report.json"

    run_ids_by_config = {
        "cfg-a": ["a1", "a2", "a3"],
        "cfg-b": ["b1"],
    }
    for config_slug, run_ids in run_ids_by_config.items():
        for run_id in run_ids:
            _write_clustering(input_dir, config_slug=config_slug, run_id=run_id)

    _write_results_jsonl(
        input_dir,
        records=[
            _config_record(config_slug="cfg-a", run_ids=["a1", "a2", "a3"], prompt_id="prompt-a"),
            _config_record(config_slug="cfg-b", run_ids=["b1"], prompt_id="prompt-b"),
        ],
    )

    config_path = _write_pairwise_eval_config(
        tmp_path,
        input_dir=input_dir,
        report_path=report_path,
        pairwise_scope="intra",
    )
    run_evaluation_config(config_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    prompt_a_key = "{'prompt_id': 'prompt-a', 'guidance': None}"
    prompt_b_key = "{'prompt_id': 'prompt-b', 'guidance': None}"
    prompt_a_pair_key = f"{prompt_a_key}->{prompt_a_key}"
    prompt_b_pair_key = f"{prompt_b_key}->{prompt_b_key}"

    assert report["vary_values"] == [
        {"prompt_id": "prompt-a", "guidance": None},
        {"prompt_id": "prompt-b", "guidance": None},
    ]
    assert report["num_pairwise_requests"] == 3
    assert report["num_pairwise_requests_by_vary_value_pair"] == {prompt_a_pair_key: 3}
    assert report["summary_by_vary_value_pair"][prompt_a_pair_key]["num_pairwise_requests"] == 3
    assert report["summary_by_vary_value_pair"][prompt_b_pair_key]["num_pairwise_requests"] == 0
    assert (
        report["summary_by_vary_value_pair"][prompt_b_pair_key]["summary_by_evaluator"][
            "pairwise_consistency"
        ]["metrics"]
        == {}
    )
    assert report["summary_by_vary_value_pair"][prompt_b_pair_key]["summary_by_evaluator"][
        "pairwise_consistency"
    ]["details"] == {
        "num_results": 0,
        "methods": [],
        "mean_numeric_details": {},
    }

    assert report["summary_by_vary_value"][prompt_a_key]["num_pairwise_requests"] == 3
    assert (
        report["summary_by_vary_value"][prompt_a_key]["summary_by_evaluator"][
            "pairwise_consistency"
        ]["details"]["num_results"]
        == 3
    )

    assert report["summary_by_vary_value"][prompt_b_key]["num_pairwise_requests"] == 0
    assert (
        report["summary_by_vary_value"][prompt_b_key]["summary_by_evaluator"][
            "pairwise_consistency"
        ]["metrics"]
        == {}
    )
    assert report["summary_by_vary_value"][prompt_b_key]["summary_by_evaluator"][
        "pairwise_consistency"
    ]["details"] == {
        "num_results": 0,
        "methods": [],
        "mean_numeric_details": {},
    }


def test_run_evaluation_config_pairwise_rejects_dataset_mismatch(tmp_path: Path) -> None:
    input_dir = tmp_path / "batch_output"
    report_path = tmp_path / "invalid_report.json"

    _write_clustering(input_dir, config_slug="cfg-a", run_id="a1")
    _write_clustering(input_dir, config_slug="cfg-b", run_id="b1")
    _write_results_jsonl(
        input_dir,
        records=[
            _config_record(
                config_slug="cfg-a",
                run_ids=["a1"],
                prompt_id="prompt-a",
                dataset_name="dataset-a",
            ),
            _config_record(
                config_slug="cfg-b",
                run_ids=["b1"],
                prompt_id="prompt-b",
                dataset_name="dataset-b",
            ),
        ],
    )

    config_path = _write_pairwise_eval_config(
        tmp_path,
        input_dir=input_dir,
        report_path=report_path,
        pairwise_scope="inter",
    )

    with pytest.raises(ValueError, match="exactly one dataset_name"):
        run_evaluation_config(config_path)


def test_run_evaluation_config_pairwise_inter_rejects_when_vary_by_inference_is_ambiguous(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "batch_output"
    report_path = tmp_path / "invalid_report.json"

    _write_clustering(input_dir, config_slug="cfg-a", run_id="a1")
    _write_clustering(input_dir, config_slug="cfg-b", run_id="b1")
    _write_results_jsonl(
        input_dir,
        records=[
            _config_record(
                config_slug="cfg-a",
                run_ids=["a1"],
                prompt_id="prompt-a",
                extra_config_fields={"max_turns": 4},
            ),
            _config_record(
                config_slug="cfg-b",
                run_ids=["b1"],
                prompt_id="prompt-b",
                extra_config_fields={"max_turns": 8},
            ),
        ],
    )

    config_path = _write_pairwise_eval_config(
        tmp_path,
        input_dir=input_dir,
        report_path=report_path,
        pairwise_scope="inter",
    )

    with pytest.raises(ValueError, match="Could not infer a single vary-by key"):
        run_evaluation_config(config_path)


def test_run_evaluation_config_pairwise_inter_rejects_when_vary_by_inference_has_no_variation(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "batch_output"
    report_path = tmp_path / "invalid_report.json"

    _write_clustering(input_dir, config_slug="cfg-a", run_id="a1")
    _write_clustering(input_dir, config_slug="cfg-b", run_id="b1")
    _write_results_jsonl(
        input_dir,
        records=[
            _config_record(config_slug="cfg-a", run_ids=["a1"], prompt_id="prompt-a"),
            _config_record(config_slug="cfg-b", run_ids=["b1"], prompt_id="prompt-a"),
        ],
    )

    config_path = _write_pairwise_eval_config(
        tmp_path,
        input_dir=input_dir,
        report_path=report_path,
        pairwise_scope="inter",
    )

    with pytest.raises(ValueError, match="Could not infer a single vary-by key"):
        run_evaluation_config(config_path)


def test_run_evaluation_config_pairwise_intra_requires_repeat_runs_per_group(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "batch_output"
    report_path = tmp_path / "invalid_report.json"

    _write_clustering(input_dir, config_slug="cfg-a", run_id="a1")
    _write_clustering(input_dir, config_slug="cfg-b", run_id="b1")
    _write_results_jsonl(
        input_dir,
        records=[
            _config_record(config_slug="cfg-a", run_ids=["a1"], prompt_id="prompt-a"),
            _config_record(config_slug="cfg-b", run_ids=["b1"], prompt_id="prompt-b"),
        ],
    )

    config_path = _write_pairwise_eval_config(
        tmp_path,
        input_dir=input_dir,
        report_path=report_path,
        pairwise_scope="intra",
    )

    with pytest.raises(ValueError, match="requires at least 2 artifacts in a shared config group"):
        run_evaluation_config(config_path)
