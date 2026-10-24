"""Unit tests for batch-run configuration validation and output serialization helpers."""

import csv
import json
from pathlib import Path
from typing import Any, cast
from unittest import mock

import pytest

from conversational_clustering.accounting import UsageCollector
from conversational_clustering.batch_runs.batch_runs import (
    _batch_runs_entry_require_non_negative_int,
    _batch_runs_entry_require_positive_int,
    _parse_batch_run_config_toml,
    _parse_embedding_model_config,
    _parse_model_config,
    run_batch_runs,
    validate_batch_run_datasets,
)
from conversational_clustering.batch_runs.config import (
    GENERATED_BENCHMARK_SPEC_METADATA_KEY,
    BatchRunConfig,
    BatchRunResult,
    HumanAgentConfig,
)
from conversational_clustering.batch_runs.io import write_jsonl, write_summary
from conversational_clustering.clustering_agent.backends import (
    DEFAULT_CLUSTERING_AGENT_BACKEND,
    BackendV1ClusteringBackendConfig,
    InstructionKMeansV1ClusteringBackendConfig,
    OneShotKMeansV1ClusteringBackendConfig,
    default_clustering_agent_backend_config,
)
from conversational_clustering.data_processing.datasets import DATASETS
from conversational_clustering.human_turn_injection import HumanTurnInjectionConfig
from conversational_clustering.models import (
    DEFAULT_BASE_URL,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingModelConfig,
    ModelConfig,
)
from conversational_clustering.simulation.runner import SimulationResult
from conversational_clustering.topic_discovery import TopicDiscoveryConfig


def _usage_summary(input_tokens: int, output_tokens: int, *, model_id: str) -> dict[str, object]:
    collector = UsageCollector()
    collector.record(
        model_id,
        {"input_tokens": input_tokens, "output_tokens": output_tokens},
    )
    return collector.snapshot(pricer=None).to_dict()


def _build_batch_run_result() -> BatchRunResult:
    config = BatchRunConfig(
        name="Example Run",
        dataset_name="synthetic",
        clustering_agent_backend="backend_v1",
        clustering_agent_backend_config=BackendV1ClusteringBackendConfig(
            model_config=ModelConfig(model="openai/gpt-5-mini")
        ),
        human_agent_config=HumanAgentConfig(prompt_id="baseline_v1", guidance="Group by topic"),
        max_turns=5,
        num_runs=1,
    )
    simulation_result = SimulationResult(
        state={"messages": [], "dataset_metadata": {"size": 0}, "iteration_count": 0},
        turns_run=3,
        stopped_reason="max_turns",
        clustering_agent_accounting=_usage_summary(
            10,
            6,
            model_id="openai/gpt-5-mini",
        ),
        human_agent_accounting=_usage_summary(
            8,
            3,
            model_id="openai/gpt-5-mini",
        ),
    )
    return BatchRunResult(
        config=config,
        simulation_results=[simulation_result],
        config_slug="example-run",
        run_ids=["001-abc12345"],
        timestamp="20260101T010101Z",
    )


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _theme_switch_turn_injection(
    turn_index: int,
    *,
    source_theme_id: str = "fruit",
    target_theme_id: str = "vehicle",
) -> HumanTurnInjectionConfig:
    return HumanTurnInjectionConfig(
        turn_index=turn_index,
        edit_primitive="theme_switch",
        source_theme_id=source_theme_id,
        target_theme_id=target_theme_id,
    )


def _theme_switch_turn_injection_toml(
    turn_index: int,
    *,
    source_theme_id: str = "fruit",
    target_theme_id: str = "vehicle",
) -> str:
    return (
        "{ turn_index = "
        f"{turn_index}, "
        'edit_primitive = "theme_switch", '
        f'source_theme_id = "{source_theme_id}", '
        f'target_theme_id = "{target_theme_id}" '
        "}"
    )


def _fruit_to_vehicle_turn_injection_toml(turn_index: int) -> str:
    return _theme_switch_turn_injection_toml(turn_index)


def _vehicle_to_fruit_turn_injection_toml(turn_index: int) -> str:
    return _theme_switch_turn_injection_toml(
        turn_index,
        source_theme_id="vehicle",
        target_theme_id="fruit",
    )


def test_batch_run_config_to_dict_omits_internal_branch_metadata_keys() -> None:
    config = BatchRunConfig(
        name="run",
        dataset_name="synthetic",
        metadata={
            "branch_group": "group-one",
            "branch_pre_vary_value": "base-guidance",
            "branch_pre_clustering_agent_backend_config": {"prompt_id": "baseline_v1"},
            "custom_label": "keep-me",
        },
    )

    metadata = config.to_dict()["metadata"]

    assert metadata == {
        "branch_group": "group-one",
        "custom_label": "keep-me",
    }


def test_batch_run_config_serializes_absent_guidance_as_none() -> None:
    config = BatchRunConfig(name="run", dataset_name="synthetic")

    assert config.to_dict()["human_agent_config"]["guidance"] is None


def test_batch_run_config_serializes_enable_intermittent_clustering() -> None:
    config = BatchRunConfig(
        name="run",
        dataset_name="synthetic",
        enable_intermittent_clustering=False,
    )

    assert config.to_dict()["enable_intermittent_clustering"] is False


def test_human_agent_config_serializes_topic_discovery_when_enabled() -> None:
    config = HumanAgentConfig(
        prompt_id="baseline_v4",
        topic_discovery=TopicDiscoveryConfig(
            enabled=True,
            trigger_mode="turn_budget",
            turn_budget=3,
            max_rounds=4,
            min_new_topics_per_round=1,
        ),
    )

    assert config.to_dict() == {
        "prompt_id": "baseline_v4",
        "guidance": None,
        "topic_discovery": {
            "enabled": True,
            "trigger_mode": "turn_budget",
            "turn_budget": 3,
            "max_rounds": 4,
            "min_new_topics_per_round": 1,
        },
    }


def test_human_agent_config_serializes_persona_when_present() -> None:
    config = HumanAgentConfig(persona="task_oriented")

    assert config.to_dict() == {
        "prompt_id": "baseline_v3",
        "guidance": None,
        "persona": "task_oriented",
    }


def test_human_agent_config_serializes_turn_injection_when_present() -> None:
    config = HumanAgentConfig(turn_injection=_theme_switch_turn_injection(2))

    assert config.to_dict() == {
        "prompt_id": "baseline_v3",
        "guidance": None,
        "turn_injection": {
            "turn_index": 2,
            "edit_primitive": "theme_switch",
            "source_theme_id": "fruit",
            "target_theme_id": "vehicle",
        },
    }


def test_human_agent_config_rejects_turn_injection_with_topic_discovery() -> None:
    with pytest.raises(
        ValueError,
        match="turn_injection cannot be combined with enabled topic_discovery",
    ):
        HumanAgentConfig(
            prompt_id="baseline_v4",
            topic_discovery=TopicDiscoveryConfig(
                enabled=True,
                trigger_mode="turn_budget",
                turn_budget=2,
            ),
            turn_injection=_theme_switch_turn_injection(3),
        )


def test_human_agent_config_serializes_theme_switch_theme_ids_when_present() -> None:
    config = HumanAgentConfig(
        turn_injection=HumanTurnInjectionConfig(
            turn_index=2,
            edit_primitive="theme_switch",
            source_theme_id="department",
            target_theme_id="ticket_category",
        )
    )

    assert config.to_dict() == {
        "prompt_id": "baseline_v3",
        "guidance": None,
        "turn_injection": {
            "turn_index": 2,
            "edit_primitive": "theme_switch",
            "source_theme_id": "department",
            "target_theme_id": "ticket_category",
        },
    }


def test_batch_run_config_allows_turn_injection_on_final_turn() -> None:
    config = BatchRunConfig(
        name="run",
        dataset_name="synthetic",
        max_turns=3,
        human_agent_config=HumanAgentConfig(turn_injection=_theme_switch_turn_injection(3)),
    )

    assert config.human_agent_config.turn_injection == _theme_switch_turn_injection(3)


def test_batch_run_config_rejects_turn_injection_beyond_max_turns() -> None:
    with pytest.raises(
        ValueError,
        match="turn_injection.turn_index must be less than or equal to max_turns",
    ):
        BatchRunConfig(
            name="run",
            dataset_name="synthetic",
            max_turns=3,
            human_agent_config=HumanAgentConfig(turn_injection=_theme_switch_turn_injection(4)),
        )


def test_human_agent_config_rejects_unknown_persona() -> None:
    with pytest.raises(ValueError, match="invalid persona"):
        HumanAgentConfig(persona="unknown_persona")


def test_parse_model_config_accepts_valid_fields() -> None:
    model_config = _parse_model_config(
        {
            "model": "openai/gpt-5-mini",
            "base_url": "https://example.com/v1",
            "provider_order": ["openai"],
            "allow_fallbacks": True,
            "api_key_env": "CUSTOM_API_KEY",
            "temperature": 0.3,
        },
        entry_index=1,
    )

    assert model_config == ModelConfig(
        model="openai/gpt-5-mini",
        base_url="https://example.com/v1",
        provider_order=["openai"],
        allow_fallbacks=True,
        api_key_env="CUSTOM_API_KEY",
        temperature=0.3,
    )


def test_parse_model_config_uses_defaults_for_optional_fields() -> None:
    model_config = _parse_model_config({"model": "openai/gpt-5-mini"}, entry_index=2)

    assert model_config.base_url == DEFAULT_BASE_URL
    assert model_config.provider_order is None
    assert model_config.allow_fallbacks is False
    assert model_config.api_key_env == "OPENROUTER_API_KEY"
    assert model_config.temperature is None


def test_parse_model_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        _parse_model_config({"model": "x", "unexpected": "y"}, entry_index=1)


def test_parse_embedding_model_config_accepts_valid_fields() -> None:
    embedding_model_config = _parse_embedding_model_config(
        {
            "model": "openai/text-embedding-3-large",
            "backend": "huggingface_local",
            "device": "metal",
            "provider_order": ["openai"],
            "allow_fallbacks": True,
            "dimensions": 512,
        },
        entry_index=1,
    )

    assert embedding_model_config == EmbeddingModelConfig(
        model="openai/text-embedding-3-large",
        backend="huggingface_local",
        device="mps",
        provider_order=["openai"],
        allow_fallbacks=True,
        dimensions=512,
    )


def test_parse_embedding_model_config_uses_defaults_for_optional_fields() -> None:
    embedding_model_config = _parse_embedding_model_config({}, entry_index=2)

    assert embedding_model_config.model == DEFAULT_EMBEDDING_MODEL
    assert embedding_model_config.base_url == DEFAULT_BASE_URL
    assert embedding_model_config.provider_order is None
    assert embedding_model_config.allow_fallbacks is False
    assert embedding_model_config.api_key_env == "OPENROUTER_API_KEY"
    assert embedding_model_config.dimensions is None
    assert embedding_model_config.input_type is None
    assert embedding_model_config.backend == "openrouter"
    assert embedding_model_config.device == "auto"


def test_parse_embedding_model_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        _parse_embedding_model_config({"model": "x", "unexpected": "y"}, entry_index=1)


def test_batch_runs_entry_require_positive_int_rejects_bool_and_non_positive_ints() -> None:
    with pytest.raises(ValueError, match="must be an integer >= 1"):
        _batch_runs_entry_require_positive_int(True, field_name="num_runs", entry_index=1)

    with pytest.raises(ValueError, match="must be an integer >= 1"):
        _batch_runs_entry_require_positive_int(0, field_name="num_runs", entry_index=1)


def test_batch_runs_entry_require_non_negative_int_rejects_bool_and_negative_ints() -> None:
    with pytest.raises(ValueError, match="must be an integer >= 0"):
        _batch_runs_entry_require_non_negative_int(
            True,
            field_name="random_seed",
            entry_index=1,
        )

    with pytest.raises(ValueError, match="must be an integer >= 0"):
        _batch_runs_entry_require_non_negative_int(
            -1,
            field_name="random_seed",
            entry_index=1,
        )


def test_batch_run_config_validates_numeric_constraints() -> None:
    with pytest.raises(ValueError, match="num_runs must be >= 1"):
        BatchRunConfig(name="run", dataset_name="synthetic", num_runs=0)

    with pytest.raises(ValueError, match="max_turns must be >= 1"):
        BatchRunConfig(name="run", dataset_name="synthetic", max_turns=0)

    with pytest.raises(ValueError, match="limit must be >= 1"):
        BatchRunConfig(name="run", dataset_name="synthetic", limit=-1)

    with pytest.raises(ValueError, match="max_verification_retries must be >= 0"):
        BatchRunConfig(name="run", dataset_name="synthetic", max_verification_retries=-1)

    with pytest.raises(ValueError, match="embedding_batch_size must be >= 1"):
        BatchRunConfig(name="run", dataset_name="synthetic", embedding_batch_size=0)

    with pytest.raises(ValueError, match="enable_intermittent_clustering must be a boolean"):
        invalid_kwargs: dict[str, Any] = {
            "name": "run",
            "dataset_name": "synthetic",
            "enable_intermittent_clustering": "false",
        }
        BatchRunConfig(**invalid_kwargs)


def test_batch_run_config_rejects_oneshot_kmeans_without_embeddings() -> None:
    with pytest.raises(ValueError, match="oneshot_kmeans_v1 requires enable_embeddings=true"):
        BatchRunConfig(
            name="run",
            dataset_name="synthetic",
            clustering_agent_backend="oneshot_kmeans_v1",
            enable_embeddings=False,
        )


def test_batch_run_config_resolves_default_backend_config_for_selected_backend() -> None:
    config = BatchRunConfig(
        name="run",
        dataset_name="synthetic",
        clustering_agent_backend="backend_v1",
        clustering_agent_backend_config=None,
    )

    assert config.clustering_agent_backend == "backend_v1"
    assert config.clustering_agent_backend_config == BackendV1ClusteringBackendConfig()


def test_batch_run_config_rejects_whitespace_padded_backend() -> None:
    with pytest.raises(ValueError, match="must not include leading/trailing whitespace"):
        BatchRunConfig(
            name="run",
            dataset_name="synthetic",
            clustering_agent_backend=" backend_v1 ",
            clustering_agent_backend_config=None,
        )


def test_validate_batch_run_datasets_reports_unknown_dataset_and_missing_file() -> None:
    configs = [
        BatchRunConfig(name="Unknown Dataset Run", dataset_name="missing"),
        BatchRunConfig(name="Missing File Run", dataset_name="known"),
    ]
    fake_datasets = {"known": {"path": "/tmp/does-not-exist.csv", "item_column": "item"}}

    with mock.patch(
        "conversational_clustering.batch_runs.batch_runs.DATASETS",
        fake_datasets,
    ):
        with mock.patch(
            "conversational_clustering.batch_runs.batch_runs.os.path.exists",
            return_value=False,
        ):
            with pytest.raises(ValueError, match="Invalid batch-run datasets"):
                validate_batch_run_datasets(configs)


def test_run_batch_runs_generates_sampled_benchmark_spec_for_supported_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "enterprise_ticket.csv"
    _write_csv(
        dataset_path,
        [
            {
                "ticket_text": "Reset my Okta password.",
                "department": "IT",
                "ticket_category": "account_issue",
                "intent_group": "access_identity",
                "issue_type": "password_reset",
                "sub_intent": "password_reset",
                "intent_label_coarse": "access_identity",
                "intent_label_normal": "access_identity__password_reset",
                "intent_label_fine": "access_identity__password_reset__okta_password_reset",
                "organizational_label_coarse": "it",
                "organizational_label_normal": "it__identity_access",
                "organizational_label_fine": "it__identity_access__okta_admin",
                "system_name": "Okta",
            },
            {
                "ticket_text": "Badge reader at side door is offline.",
                "department": "Facilities",
                "ticket_category": "badge_issue",
                "intent_group": "access_identity",
                "issue_type": "badge_not_working",
                "sub_intent": "badge_reader_outage",
                "intent_label_coarse": "access_identity",
                "intent_label_normal": "access_identity__badge_not_working",
                "intent_label_fine": "access_identity__badge_not_working__badge_reader_outage",
                "organizational_label_coarse": "facilities",
                "organizational_label_normal": "facilities__physical_security",
                "organizational_label_fine": "facilities__physical_security__badge_operations",
                "system_name": "Badge System",
            },
            {
                "ticket_text": "Invoice approval workflow is stuck.",
                "department": "Finance",
                "ticket_category": "invoice_issue",
                "intent_group": "finance_procurement",
                "issue_type": "workflow_blocked",
                "sub_intent": "invoice_approval",
                "intent_label_coarse": "finance_procurement",
                "intent_label_normal": "finance_procurement__workflow_blocked",
                "intent_label_fine": "finance_procurement__workflow_blocked__invoice_approval",
                "organizational_label_coarse": "finance",
                "organizational_label_normal": "finance__accounts_payable",
                "organizational_label_fine": "finance__accounts_payable__invoice_operations",
                "system_name": "SAP",
            },
        ],
    )
    monkeypatch.setitem(
        cast(dict[str, str], DATASETS["enterprise_ticket"]),
        "path",
        str(dataset_path),
    )

    config_path = tmp_path / "batch_runs.toml"
    output_dir = tmp_path / "batch_output"
    config_path.write_text(
        "\n".join(
            [
                f'output_dir = "{output_dir}"',
                "num_runners = 1",
                "",
                "[base]",
                'name = "enterprise_sample"',
                'dataset_name = "enterprise_ticket"',
                "limit = 2",
                "random_seed = 7",
            ]
        ),
        encoding="utf-8",
    )

    def mock_run_batch_parallel(
        self: object,
        batch_runs: list[BatchRunConfig],
        *,
        max_workers: int | None = None,
    ) -> list[BatchRunResult]:
        del self, max_workers
        return [BatchRunResult(config=batch_runs[0], simulation_results=[], config_slug="cfg-one")]

    with mock.patch(
        "conversational_clustering.batch_runs.batch_runs.BatchRunner.run_batch_parallel",
        autospec=True,
        side_effect=mock_run_batch_parallel,
    ):
        run_batch_runs(str(config_path))

    results = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(results) == 1
    benchmark_rel_path = results[0]["config"]["metadata"][GENERATED_BENCHMARK_SPEC_METADATA_KEY]
    benchmark_path = output_dir / benchmark_rel_path
    benchmark_payload = json.loads(benchmark_path.read_text(encoding="utf-8"))

    assert benchmark_payload["dataset_name"] == "enterprise_ticket"
    assert len(benchmark_payload["items"]) == 2
    assert set(benchmark_payload["items"]) <= {
        "Reset my Okta password.",
        "Badge reader at side door is offline.",
        "Invoice approval workflow is stuck.",
    }
    assert set(benchmark_payload["references"][0]["assignment"]) == set(benchmark_payload["items"])


def test_run_batch_runs_skips_generated_benchmark_spec_for_unlabeled_enterprise_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "enterprise_ticket_runtime.csv"
    _write_csv(
        dataset_path,
        [
            {"ticket_text": "Reset my Okta password."},
            {"ticket_text": "Badge reader at side door is offline."},
        ],
    )
    monkeypatch.setitem(
        cast(dict[str, str], DATASETS["enterprise_ticket"]),
        "path",
        str(dataset_path),
    )

    config_path = tmp_path / "batch_runs.toml"
    output_dir = tmp_path / "batch_output"
    config_path.write_text(
        "\n".join(
            [
                f'output_dir = "{output_dir}"',
                "num_runners = 1",
                "",
                "[base]",
                'name = "enterprise_runtime_only"',
                'dataset_name = "enterprise_ticket"',
                "limit = 2",
                "random_seed = 7",
            ]
        ),
        encoding="utf-8",
    )

    def mock_run_batch_parallel(
        self: object,
        batch_runs: list[BatchRunConfig],
        *,
        max_workers: int | None = None,
    ) -> list[BatchRunResult]:
        del self, max_workers
        return [BatchRunResult(config=batch_runs[0], simulation_results=[], config_slug="cfg-one")]

    with mock.patch(
        "conversational_clustering.batch_runs.batch_runs.BatchRunner.run_batch_parallel",
        autospec=True,
        side_effect=mock_run_batch_parallel,
    ):
        run_batch_runs(str(config_path))

    results = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(results) == 1
    assert GENERATED_BENCHMARK_SPEC_METADATA_KEY not in results[0]["config"]["metadata"]
    assert not (output_dir / "_benchmark_specs").exists()


def test_parse_batch_run_config_toml_parses_minimal_valid_config(tmp_path: Path) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 2
reporter = "null"

[base]
name = "Run One"
dataset_name = "synthetic"
"""
    config_path = tmp_path / "batch_runs.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    output_dir, batch_runs, reporter_factory, num_runners = _parse_batch_run_config_toml(
        str(config_path)
    )

    assert output_dir == "data/out"
    assert num_runners == 2
    assert len(batch_runs) == 1
    assert batch_runs[0].name == "Run One"
    assert batch_runs[0].dataset_name == "synthetic"
    assert batch_runs[0].clustering_agent_backend == DEFAULT_CLUSTERING_AGENT_BACKEND
    assert batch_runs[0].clustering_agent_backend_config == default_clustering_agent_backend_config(
        DEFAULT_CLUSTERING_AGENT_BACKEND
    )
    assert batch_runs[0].enable_intermittent_clustering is False


def test_parse_batch_run_config_toml_parses_enable_intermittent_clustering(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "batch_runs.toml"
    config_path.write_text(
        "\n".join(
            [
                'output_dir = "data/output"',
                "num_runners = 1",
                "",
                "[base]",
                'name = "run"',
                'dataset_name = "synthetic"',
                "enable_intermittent_clustering = false",
            ]
        ),
        encoding="utf-8",
    )

    _output_dir, batch_runs, reporter_factory, _num_runners = _parse_batch_run_config_toml(
        str(config_path)
    )

    assert batch_runs[0].enable_intermittent_clustering is False
    assert batch_runs[0].human_agent_prompt_id == "baseline_v3"
    assert batch_runs[0].human_verifier_prompt_id == "baseline_v1"
    assert batch_runs[0].max_verification_retries == 2
    assert batch_runs[0].enable_embeddings is False
    assert batch_runs[0].embedding_model_config == EmbeddingModelConfig(
        model=DEFAULT_EMBEDDING_MODEL
    )
    assert batch_runs[0].embedding_force_recompute is False
    assert batch_runs[0].embedding_batch_size == 64
    assert batch_runs[0].max_turns == 8
    assert batch_runs[0].num_runs == 1
    assert batch_runs[0].random_seed == 42
    assert batch_runs[0].guidance is None
    assert reporter_factory(batch_runs[0]).__class__.__name__ == "NullReporter"


def test_parse_batch_run_config_toml_parses_turn_injection(tmp_path: Path) -> None:
    config_path = tmp_path / "batch_runs_turn_injection.toml"
    config_path.write_text(
        "\n".join(
            [
                'output_dir = "data/output"',
                "num_runners = 1",
                "",
                "[base]",
                'name = "run"',
                'dataset_name = "synthetic"',
                "max_turns = 4",
                "",
                "[base.human_agent_config.turn_injection]",
                "turn_index = 2",
                'edit_primitive = "theme_switch"',
                'source_theme_id = "fruit"',
                'target_theme_id = "vehicle"',
            ]
        ),
        encoding="utf-8",
    )

    _output_dir, batch_runs, _reporter_factory, _num_runners = _parse_batch_run_config_toml(
        str(config_path)
    )

    assert batch_runs[0].human_agent_config.turn_injection == _theme_switch_turn_injection(2)


def test_parse_batch_run_config_toml_rejects_turn_injection_with_topic_discovery(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "batch_runs_turn_injection_topic_discovery.toml"
    config_path.write_text(
        "\n".join(
            [
                'output_dir = "data/output"',
                "num_runners = 1",
                "",
                "[base]",
                'name = "run"',
                'dataset_name = "synthetic"',
                "max_turns = 4",
                "",
                "[base.human_agent_config]",
                'prompt_id = "baseline_v4"',
                "",
                "[base.human_agent_config.topic_discovery]",
                "enabled = true",
                'trigger_mode = "turn_budget"',
                "turn_budget = 2",
                "",
                "[base.human_agent_config.turn_injection]",
                "turn_index = 3",
                'edit_primitive = "theme_switch"',
                'source_theme_id = "fruit"',
                'target_theme_id = "vehicle"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="turn_injection cannot be combined with enabled topic_discovery",
    ):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_parses_theme_switch_theme_ids(tmp_path: Path) -> None:
    config_path = tmp_path / "batch_runs_turn_injection_theme_switch.toml"
    config_path.write_text(
        "\n".join(
            [
                'output_dir = "data/output"',
                "num_runners = 1",
                "",
                "[base]",
                'name = "run"',
                'dataset_name = "synthetic"',
                "max_turns = 4",
                "",
                "[base.human_agent_config.turn_injection]",
                "turn_index = 2",
                'edit_primitive = "theme_switch"',
                'source_theme_id = "fruit"',
                'target_theme_id = "vehicle"',
            ]
        ),
        encoding="utf-8",
    )

    _output_dir, batch_runs, _reporter_factory, _num_runners = _parse_batch_run_config_toml(
        str(config_path)
    )

    assert batch_runs[0].human_agent_config.turn_injection == HumanTurnInjectionConfig(
        turn_index=2,
        edit_primitive="theme_switch",
        source_theme_id="fruit",
        target_theme_id="vehicle",
    )


def test_parse_batch_run_config_toml_rejects_blank_turn_injection_edit_primitive(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "batch_runs_blank_turn_injection.toml"
    config_path.write_text(
        "\n".join(
            [
                'output_dir = "data/output"',
                "num_runners = 1",
                "",
                "[base]",
                'name = "run"',
                'dataset_name = "synthetic"',
                "max_turns = 4",
                "",
                "[base.human_agent_config.turn_injection]",
                "turn_index = 2",
                'edit_primitive = "   "',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="edit_primitive must be a non-empty string"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_legacy_turn_injection_system_prompt(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "batch_runs_legacy_turn_injection.toml"
    config_path.write_text(
        "\n".join(
            [
                'output_dir = "data/output"',
                "num_runners = 1",
                "",
                "[base]",
                'name = "run"',
                'dataset_name = "synthetic"',
                "max_turns = 4",
                "",
                "[base.human_agent_config.turn_injection]",
                "turn_index = 2",
                'system_prompt = "Please broaden the clusters."',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported field\\(s\\): system_prompt"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_legacy_turn_injection_target_reference_id(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "batch_runs_legacy_turn_injection_target.toml"
    config_path.write_text(
        "\n".join(
            [
                'output_dir = "data/output"',
                "num_runners = 1",
                "",
                "[base]",
                'name = "run"',
                'dataset_name = "synthetic"',
                "max_turns = 4",
                "",
                "[base.human_agent_config.turn_injection]",
                "turn_index = 2",
                'edit_primitive = "theme_switch"',
                'source_theme_id = "fruit"',
                'target_theme_id = "vehicle"',
                'target_reference_id = "fruit__initial"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported field\\(s\\): target_reference_id"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_accepts_granularity_turn_injection_edit_primitives(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "batch_runs_turn_injection_granularity_primitive.toml"
    config_path.write_text(
        "\n".join(
            [
                'output_dir = "data/output"',
                "num_runners = 1",
                "",
                "[base]",
                'name = "run"',
                'dataset_name = "synthetic"',
                "max_turns = 4",
                "",
                "[base.human_agent_config.turn_injection]",
                "turn_index = 2",
                'edit_primitive = "coarsen"',
            ]
        ),
        encoding="utf-8",
    )

    _, batch_runs, _, _ = _parse_batch_run_config_toml(str(config_path))

    assert batch_runs[0].human_agent_config.turn_injection == HumanTurnInjectionConfig(
        turn_index=2,
        edit_primitive="coarsen",
    )


def test_parse_batch_run_config_toml_rejects_theme_ids_for_coarsen(tmp_path: Path) -> None:
    config_path = tmp_path / "batch_runs_turn_injection_coarsen_with_theme_ids.toml"
    config_path.write_text(
        "\n".join(
            [
                'output_dir = "data/output"',
                "num_runners = 1",
                "",
                "[base]",
                'name = "run"',
                'dataset_name = "synthetic"',
                "max_turns = 4",
                "",
                "[base.human_agent_config.turn_injection]",
                "turn_index = 2",
                'edit_primitive = "coarsen"',
                'source_theme_id = "fruit"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="source_theme_id is only supported when edit_primitive='theme_switch'",
    ):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_theme_switch_without_source_theme_id(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "batch_runs_turn_injection_missing_source_theme.toml"
    config_path.write_text(
        "\n".join(
            [
                'output_dir = "data/output"',
                "num_runners = 1",
                "",
                "[base]",
                'name = "run"',
                'dataset_name = "synthetic"',
                "max_turns = 4",
                "",
                "[base.human_agent_config.turn_injection]",
                "turn_index = 2",
                'edit_primitive = "theme_switch"',
                'target_theme_id = "vehicle"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="source_theme_id is required when edit_primitive='theme_switch'",
    ):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_theme_switch_without_target_theme_id(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "batch_runs_turn_injection_missing_target_theme.toml"
    config_path.write_text(
        "\n".join(
            [
                'output_dir = "data/output"',
                "num_runners = 1",
                "",
                "[base]",
                'name = "run"',
                'dataset_name = "synthetic"',
                "max_turns = 4",
                "",
                "[base.human_agent_config.turn_injection]",
                "turn_index = 2",
                'edit_primitive = "theme_switch"',
                'source_theme_id = "fruit"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="target_theme_id is required when edit_primitive='theme_switch'",
    ):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_theme_switch_with_identical_theme_ids(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "batch_runs_turn_injection_same_theme_ids.toml"
    config_path.write_text(
        "\n".join(
            [
                'output_dir = "data/output"',
                "num_runners = 1",
                "",
                "[base]",
                'name = "run"',
                'dataset_name = "synthetic"',
                "max_turns = 4",
                "",
                "[base.human_agent_config.turn_injection]",
                "turn_index = 2",
                'edit_primitive = "theme_switch"',
                'source_theme_id = "fruit"',
                'target_theme_id = "fruit"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="source_theme_id and target_theme_id must differ when edit_primitive='theme_switch'",
    ):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_parses_batch_progress_reporter(tmp_path: Path) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 2
reporter = "batch_progress"

[base]
name = "Run One"
dataset_name = "synthetic"
"""
    config_path = tmp_path / "batch_runs_progress.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _output_dir, batch_runs, reporter_factory, _num_runners = _parse_batch_run_config_toml(
        str(config_path)
    )

    assert reporter_factory(batch_runs[0]).__class__.__name__ == "NullReporter"
    assert reporter_factory.batch_progress_enabled is True


def test_parse_batch_run_config_toml_expands_sweep_values_into_multiple_configs(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 2
reporter = "null"

[base]
name = "prompt_ablation"
dataset_name = "synthetic"
max_turns = 3

[sweep]
vary_by = "human_agent_config"
values = [
    { prompt_id = "baseline_v1", guidance = "Group by topic" },
    { prompt_id = "baseline_v2", guidance = "Group by topic" },
]
"""
    config_path = tmp_path / "batch_runs.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _, batch_runs, _, _ = _parse_batch_run_config_toml(str(config_path))

    assert [config.name for config in batch_runs] == [
        "human_agent_config_guidance_Group by topic__prompt_id_baseline_v1",
        "human_agent_config_guidance_Group by topic__prompt_id_baseline_v2",
    ]
    assert [config.human_agent_prompt_id for config in batch_runs] == ["baseline_v1", "baseline_v2"]
    assert all(config.dataset_name == "synthetic" for config in batch_runs)
    assert all(config.max_turns == 3 for config in batch_runs)


def test_parse_batch_run_config_toml_rejects_whitespace_duplicate_sweep_values(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 2
reporter = "null"

[base]
name = "prompt_ablation"
dataset_name = "synthetic"

[sweep]
vary_by = "human_agent_config"
values = [
    { prompt_id = "baseline_v1", guidance = "Group by topic" },
    { prompt_id = " baseline_v1 ", guidance = "Group by topic" },
]
"""
    config_path = tmp_path / "batch_runs.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="must not contain duplicate values"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_unhashable_duplicate_sweep_values(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 2
reporter = "null"

[base]
name = "prompt_ablation"
dataset_name = "synthetic"

[sweep]
vary_by = "clustering_agent_backend_config"
values = [
    { prompt_id = "baseline_v1" },
    { prompt_id = "baseline_v1" },
]
"""
    config_path = tmp_path / "batch_runs.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="must not contain duplicate values"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_duplicate_human_agent_config_sweep_aliases(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 2
reporter = "null"

[base]
name = "prompt_ablation"
dataset_name = "synthetic"

[sweep]
vary_by = "human_agent_config"
values = [
    {},
    { prompt_id = "baseline_v3" },
    { prompt_id = "baseline_v3", guidance = "   " },
]
"""
    config_path = tmp_path / "batch_runs_human_agent_sweep_aliases.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="must not contain duplicate values"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_multifactor_human_agent_config_sweep(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 2
reporter = "null"

[base]
name = "prompt_ablation"
dataset_name = "synthetic"

[sweep]
vary_by = "human_agent_config"
values = [
    { prompt_id = "baseline_v4" },
    { prompt_id = "baseline_v1", guidance = "Group by topic" },
]
"""
    config_path = tmp_path / "batch_runs_multifactor_human_agent_sweep.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="may vary at most one top-level field"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_accepts_human_agent_persona_sweep(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/batch_run_results/persona_sweep"
num_runners = 1

[base]
name = "persona_sweep"
dataset_name = "synthetic"

[sweep]
vary_by = "human_agent_config"
values = [
  { persona = "task_oriented" },
  { persona = "casual_social" }
]
    """
    config_path = tmp_path / "batch_runs_persona_sweep.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _output_dir, batch_runs, _reporter_factory, _num_runners = _parse_batch_run_config_toml(
        str(config_path)
    )

    assert [config.human_agent_config.persona for config in batch_runs] == [
        "task_oriented",
        "casual_social",
    ]


def test_parse_batch_run_config_toml_rejects_multifactor_turn_injection_sweep(
    tmp_path: Path,
) -> None:
    config_toml = (
        """
output_dir = "data/out"
num_runners = 2
reporter = "null"

[base]
name = "turn_injection_ablation"
dataset_name = "synthetic"
max_turns = 5

[sweep]
vary_by = "human_agent_config"
values = [
"""
        f"    {{ turn_injection = {_fruit_to_vehicle_turn_injection_toml(2)} }},\n"
        '    { prompt_id = "baseline_v4", turn_injection = '
        f"{_vehicle_to_fruit_turn_injection_toml(3)} }},\n"
        "]\n"
    )
    config_path = tmp_path / "batch_runs_multifactor_turn_injection_sweep.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="may vary at most one top-level field"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_branching_on_turn_injection_during_shared_turns(
    tmp_path: Path,
) -> None:
    config_toml = (
        """
output_dir = "data/out"
num_runners = 1

[base]
name = "ablation"
dataset_name = "synthetic"
max_turns = 6

[base.human_agent_config]
prompt_id = "baseline_v4"

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
"""
        '    { prompt_id = "baseline_v4", turn_injection = '
        f"{_fruit_to_vehicle_turn_injection_toml(2)} }},\n"
        '    { prompt_id = "baseline_v4", turn_injection = '
        f"{_vehicle_to_fruit_turn_injection_toml(4)} }},\n"
        "]\n"
    )
    config_path = tmp_path / "batch_runs_branch_turn_injection_shared_turns.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="must not vary 'human_agent_config.turn_injection'"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_allows_branching_on_turn_injection_after_shared_turns(
    tmp_path: Path,
) -> None:
    config_toml = (
        """
output_dir = "data/out"
num_runners = 1

[base]
name = "ablation"
dataset_name = "synthetic"
max_turns = 6

[base.human_agent_config]
prompt_id = "baseline_v4"

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
"""
        '    { prompt_id = "baseline_v4", turn_injection = '
        f"{_fruit_to_vehicle_turn_injection_toml(4)} }},\n"
        '    { prompt_id = "baseline_v4", turn_injection = '
        f"{_vehicle_to_fruit_turn_injection_toml(5)} }},\n"
        "]\n"
    )
    config_path = tmp_path / "batch_runs_branch_turn_injection_late.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _output_dir, batch_runs, _reporter_factory, _num_runners = _parse_batch_run_config_toml(
        str(config_path)
    )

    assert [config.human_agent_config.turn_injection for config in batch_runs] == [
        _theme_switch_turn_injection(4),
        _theme_switch_turn_injection(5, source_theme_id="vehicle", target_theme_id="fruit"),
    ]


def test_parse_batch_run_config_toml_allows_branches_that_inherit_shared_turn_injection(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "ablation"
dataset_name = "synthetic"
max_turns = 6

[base.human_agent_config]
prompt_id = "baseline_v4"

[base.human_agent_config.turn_injection]
turn_index = 2
edit_primitive = "theme_switch"
source_theme_id = "fruit"
target_theme_id = "vehicle"

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
    { prompt_id = "baseline_v3" },
    { prompt_id = "baseline_v4" },
]
"""
    config_path = tmp_path / "batch_runs_branch_inherit_shared_turn_injection.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _output_dir, batch_runs, _reporter_factory, _num_runners = _parse_batch_run_config_toml(
        str(config_path)
    )

    assert [config.human_agent_prompt_id for config in batch_runs] == ["baseline_v3", "baseline_v4"]
    assert [config.human_agent_config.turn_injection for config in batch_runs] == [None, None]
    assert all(
        isinstance(config.metadata.get("branch_pre_vary_value"), HumanAgentConfig)
        for config in batch_runs
    )


def test_parse_batch_run_config_toml_allows_identical_shared_turn_injections_across_branches(
    tmp_path: Path,
) -> None:
    config_toml = (
        """
output_dir = "data/out"
num_runners = 1

[base]
name = "ablation"
dataset_name = "synthetic"
max_turns = 6

[base.human_agent_config]
prompt_id = "baseline_v3"

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
"""
        '    { prompt_id = "baseline_v3", turn_injection = '
        f"{_fruit_to_vehicle_turn_injection_toml(2)} }},\n"
        '    { prompt_id = "baseline_v4", turn_injection = '
        f"{_fruit_to_vehicle_turn_injection_toml(2)} }},\n"
        "]\n"
    )
    config_path = tmp_path / "batch_runs_branch_identical_shared_turn_injection.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _output_dir, batch_runs, _reporter_factory, _num_runners = _parse_batch_run_config_toml(
        str(config_path)
    )

    assert [config.human_agent_prompt_id for config in batch_runs] == ["baseline_v3", "baseline_v4"]
    assert [config.human_agent_config.turn_injection for config in batch_runs] == [
        _theme_switch_turn_injection(2),
        _theme_switch_turn_injection(2),
    ]
    assert all(
        config.metadata.get("branch_pre_vary_value")
        == HumanAgentConfig(
            prompt_id="baseline_v3",
            turn_injection=_theme_switch_turn_injection(2),
        )
        for config in batch_runs
    )


def test_parse_batch_run_config_toml_rejects_mixed_omitted_and_shared_turn_branch_injection(
    tmp_path: Path,
) -> None:
    config_toml = (
        """
output_dir = "data/out"
num_runners = 1

[base]
name = "ablation"
dataset_name = "synthetic"
max_turns = 6

[base.human_agent_config]
prompt_id = "baseline_v3"

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
    { prompt_id = "baseline_v3" },
"""
        '    { prompt_id = "baseline_v3", turn_injection = '
        f"{_fruit_to_vehicle_turn_injection_toml(2)} }},\n"
        "]\n"
    )
    config_path = tmp_path / "batch_runs_branch_mixed_shared_turn_injection.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="must not vary 'human_agent_config.turn_injection' across branch runs",
    ):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_preserves_base_shared_turn_injection_when_branch_common_injection_is_late(  # noqa: E501
    tmp_path: Path,
) -> None:
    config_toml = (
        """
output_dir = "data/out"
num_runners = 1

[base]
name = "ablation"
dataset_name = "synthetic"
max_turns = 6

[base.human_agent_config]
prompt_id = "baseline_v3"

[base.human_agent_config.turn_injection]
turn_index = 2
edit_primitive = "theme_switch"
source_theme_id = "fruit"
target_theme_id = "vehicle"

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
"""
        '    { prompt_id = "baseline_v3", turn_injection = '
        f"{_vehicle_to_fruit_turn_injection_toml(5)} }},\n"
        '    { prompt_id = "baseline_v4", turn_injection = '
        f"{_vehicle_to_fruit_turn_injection_toml(5)} }},\n"
        "]\n"
    )
    config_path = tmp_path / "batch_runs_branch_preserve_base_shared_turn_injection.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _output_dir, batch_runs, _reporter_factory, _num_runners = _parse_batch_run_config_toml(
        str(config_path)
    )

    assert [config.human_agent_config.turn_injection for config in batch_runs] == [
        _theme_switch_turn_injection(5, source_theme_id="vehicle", target_theme_id="fruit"),
        _theme_switch_turn_injection(5, source_theme_id="vehicle", target_theme_id="fruit"),
    ]
    assert all(
        config.metadata.get("branch_pre_vary_value")
        == HumanAgentConfig(
            prompt_id="baseline_v3",
            turn_injection=_theme_switch_turn_injection(2),
        )
        for config in batch_runs
    )


def test_parse_batch_run_config_toml_allows_identical_shared_turn_injection_in_base_and_branches(
    tmp_path: Path,
) -> None:
    config_toml = (
        """
output_dir = "data/out"
num_runners = 1

[base]
name = "ablation"
dataset_name = "synthetic"
max_turns = 6

[base.human_agent_config]
prompt_id = "baseline_v3"

[base.human_agent_config.turn_injection]
turn_index = 2
edit_primitive = "theme_switch"
source_theme_id = "fruit"
target_theme_id = "vehicle"

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
"""
        '    { prompt_id = "baseline_v3", turn_injection = '
        f"{_fruit_to_vehicle_turn_injection_toml(2)} }},\n"
        '    { prompt_id = "baseline_v4", turn_injection = '
        f"{_fruit_to_vehicle_turn_injection_toml(2)} }},\n"
        "]\n"
    )
    config_path = tmp_path / "batch_runs_branch_same_as_base_shared_turn_injection.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _output_dir, batch_runs, _reporter_factory, _num_runners = _parse_batch_run_config_toml(
        str(config_path)
    )

    assert [config.human_agent_prompt_id for config in batch_runs] == ["baseline_v3", "baseline_v4"]
    assert [config.human_agent_config.turn_injection for config in batch_runs] == [
        _theme_switch_turn_injection(2),
        _theme_switch_turn_injection(2),
    ]
    assert all(
        config.metadata.get("branch_pre_vary_value")
        == HumanAgentConfig(
            prompt_id="baseline_v3",
            turn_injection=_theme_switch_turn_injection(2),
        )
        for config in batch_runs
    )


def test_parse_batch_run_config_toml_rejects_overlapping_base_and_branch_shared_turn_injection(
    tmp_path: Path,
) -> None:
    config_toml = (
        """
output_dir = "data/out"
num_runners = 1

[base]
name = "ablation"
dataset_name = "synthetic"
max_turns = 6

[base.human_agent_config]
prompt_id = "baseline_v3"

[base.human_agent_config.turn_injection]
turn_index = 2
edit_primitive = "theme_switch"
source_theme_id = "fruit"
target_theme_id = "vehicle"

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
"""
        '    { prompt_id = "baseline_v3", turn_injection = '
        f"{_vehicle_to_fruit_turn_injection_toml(3)} }},\n"
        '    { prompt_id = "baseline_v4", turn_injection = '
        f"{_vehicle_to_fruit_turn_injection_toml(3)} }},\n"
        "]\n"
    )
    config_path = tmp_path / "batch_runs_branch_overlapping_shared_turn_injection.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="must not define both base and branch shared-prefix 'human_agent_config.turn_injection' values",  # noqa: E501
    ):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_base_late_and_branch_shared_turn_injection(
    tmp_path: Path,
) -> None:
    config_toml = (
        """
output_dir = "data/out"
num_runners = 1

[base]
name = "ablation"
dataset_name = "synthetic"
max_turns = 6

[base.human_agent_config]
prompt_id = "baseline_v3"

[base.human_agent_config.turn_injection]
turn_index = 5
edit_primitive = "theme_switch"
source_theme_id = "vehicle"
target_theme_id = "fruit"

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
"""
        '    { prompt_id = "baseline_v3", turn_injection = '
        f"{_fruit_to_vehicle_turn_injection_toml(2)} }},\n"
        '    { prompt_id = "baseline_v4", turn_injection = '
        f"{_fruit_to_vehicle_turn_injection_toml(2)} }},\n"
        "]\n"
    )
    config_path = tmp_path / "batch_runs_branch_shared_turn_injection_drops_late_base.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="must not combine a base late 'human_agent_config.turn_injection' value with a branch shared-prefix value",  # noqa: E501
    ):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_allows_single_factor_topic_discovery_sweep(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 2
reporter = "null"

[base]
name = "topic_discovery_ablation"
dataset_name = "synthetic"

[sweep]
vary_by = "human_agent_config"
values = [
    { prompt_id = "baseline_v4", topic_discovery = { enabled = false } },
    { prompt_id = "baseline_v4", topic_discovery = { enabled = true, turn_budget = 2 } },
]
"""
    config_path = tmp_path / "batch_runs_topic_discovery_sweep.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _, batch_runs, _, _ = _parse_batch_run_config_toml(str(config_path))

    assert len(batch_runs) == 2
    assert [config.human_agent_prompt_id for config in batch_runs] == ["baseline_v4", "baseline_v4"]


def test_parse_batch_run_config_toml_expands_random_seed_sweep_when_limit_is_defined(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 2
reporter = "null"

[base]
name = "seed_ablation"
dataset_name = "synthetic"
limit = 10

[sweep]
vary_by = "random_seed"
values = [7, 11]
"""
    config_path = tmp_path / "batch_runs.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _, batch_runs, _, _ = _parse_batch_run_config_toml(str(config_path))

    assert [config.name for config in batch_runs] == [
        "random_seed_7",
        "random_seed_11",
    ]
    assert [config.random_seed for config in batch_runs] == [7, 11]
    assert all(config.limit == 10 for config in batch_runs)


def test_parse_batch_run_config_toml_rejects_random_seed_sweep_without_limit(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 2
reporter = "null"

[base]
name = "seed_ablation"
dataset_name = "synthetic"

[sweep]
vary_by = "random_seed"
values = [7, 11]
"""
    config_path = tmp_path / "batch_runs.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="'base.limit' must be defined"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_expands_branch_values_on_top_of_sweep_values(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 2
reporter = "null"

[base]
name = "ablation"
dataset_name = "synthetic"
max_turns = 6

[sweep]
vary_by = "human_agent_config"
values = [
    { prompt_id = "baseline_v1", guidance = "Group by topic" },
    { prompt_id = "baseline_v2", guidance = "Group by topic" },
]

[branch]
shared_turns = 3
vary_by = "max_verification_retries"
values = [1, 4]
"""
    config_path = tmp_path / "batch_runs.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _, batch_runs, _, _ = _parse_batch_run_config_toml(str(config_path))

    assert len(batch_runs) == 4
    assert [config.human_agent_prompt_id for config in batch_runs] == [
        "baseline_v1",
        "baseline_v1",
        "baseline_v2",
        "baseline_v2",
    ]
    assert [config.max_verification_retries for config in batch_runs] == [1, 4, 1, 4]
    assert [config.metadata["branch_group"] for config in batch_runs] == [
        "human_agent_config_guidance_Group by topic__prompt_id_baseline_v1",
        "human_agent_config_guidance_Group by topic__prompt_id_baseline_v1",
        "human_agent_config_guidance_Group by topic__prompt_id_baseline_v2",
        "human_agent_config_guidance_Group by topic__prompt_id_baseline_v2",
    ]
    assert all(config.metadata["branch_shared_turns"] == 3 for config in batch_runs)
    assert all(
        config.metadata["branch_vary_by"] == "max_verification_retries" for config in batch_runs
    )
    assert all(config.metadata["branch_pre_vary_value"] == 2 for config in batch_runs)
    assert [config.name for config in batch_runs] == [
        (
            "human_agent_config_guidance_Group by topic__prompt_id_baseline_v1"
            "__max_verification_retries_1"
        ),
        (
            "human_agent_config_guidance_Group by topic__prompt_id_baseline_v1"
            "__max_verification_retries_4"
        ),
        (
            "human_agent_config_guidance_Group by topic__prompt_id_baseline_v2"
            "__max_verification_retries_1"
        ),
        (
            "human_agent_config_guidance_Group by topic__prompt_id_baseline_v2"
            "__max_verification_retries_4"
        ),
    ]


def test_parse_batch_run_config_toml_rejects_branch_when_shared_turns_not_less_than_max_turns(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "ablation"
dataset_name = "synthetic"
max_turns = 3

[branch]
shared_turns = 3
vary_by = "max_verification_retries"
values = [1, 2]
"""
    config_path = tmp_path / "batch_runs.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="shared_turns"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_branch_vary_by_matching_sweep_vary_by(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "ablation"
dataset_name = "synthetic"
max_turns = 6

[sweep]
vary_by = "human_agent_config"
values = [
    { prompt_id = "baseline_v1", guidance = "Group by topic" },
    { prompt_id = "baseline_v2", guidance = "Group by topic" },
]

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
    { prompt_id = "baseline_v3" },
    { prompt_id = "baseline_v2", guidance = "Refine by topic" },
]
"""
    config_path = tmp_path / "batch_runs.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="must differ"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_duplicate_branch_values(tmp_path: Path) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "ablation"
dataset_name = "synthetic"
max_turns = 6

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
    { prompt_id = "baseline_v1", guidance = "x" },
    { prompt_id = "baseline_v1", guidance = " x " },
]
"""
    config_path = tmp_path / "batch_runs.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="must not contain duplicate values"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_duplicate_human_agent_config_branch_aliases(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "ablation"
dataset_name = "synthetic"
max_turns = 6

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
    {},
    { prompt_id = "baseline_v3", guidance = "   " },
]
"""
    config_path = tmp_path / "batch_runs_human_agent_branch_aliases.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="must not contain duplicate values"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_multifactor_human_agent_config_branch(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "ablation"
dataset_name = "synthetic"
max_turns = 6

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
    { prompt_id = "baseline_v4" },
    { prompt_id = "baseline_v1", guidance = "x" },
]
"""
    config_path = tmp_path / "batch_runs_multifactor_human_agent_branch.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="may vary at most one top-level field"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_persona_branching_with_shared_turns(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "persona_branch"
dataset_name = "synthetic"
max_turns = 6

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
    { persona = "task_oriented" },
    { persona = "casual_social" },
]
"""
    config_path = tmp_path / "batch_runs_persona_branch.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="must not vary 'human_agent_config.persona'"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_allows_shared_persona_across_branch_values(
    tmp_path: Path,
) -> None:
    config_toml = (
        """
output_dir = "data/out"
num_runners = 1

[base]
name = "persona_branch"
dataset_name = "synthetic"
max_turns = 6

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
"""
        '    { persona = "task_oriented", turn_injection = { turn_index = 4, '
        'edit_primitive = "theme_switch", '
        'source_theme_id = "fruit", '
        'target_theme_id = "vehicle" } },\n'
        '    { persona = "task_oriented", turn_injection = { turn_index = 5, '
        'edit_primitive = "theme_switch", '
        'source_theme_id = "vehicle", '
        'target_theme_id = "fruit" } },\n'
        "]\n"
    )
    config_path = tmp_path / "batch_runs_shared_persona_branch.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _, batch_runs, _, _ = _parse_batch_run_config_toml(str(config_path))

    assert [config.human_agent_config.persona for config in batch_runs] == [
        "task_oriented",
        "task_oriented",
    ]
    assert all(
        config.metadata["branch_pre_vary_value"]
        == HumanAgentConfig(prompt_id="baseline_v3", persona="task_oriented")
        for config in batch_runs
    )


def test_parse_batch_run_config_toml_rejects_conflicting_base_and_shared_branch_personas(
    tmp_path: Path,
) -> None:
    config_toml = (
        """
output_dir = "data/out"
num_runners = 1

[base]
name = "persona_branch"
dataset_name = "synthetic"
max_turns = 6

[base.human_agent_config]
persona = "casual_social"

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
"""
        '    { persona = "task_oriented", turn_injection = { turn_index = 4, '
        'edit_primitive = "theme_switch", '
        'source_theme_id = "fruit", '
        'target_theme_id = "vehicle" } },\n'
        '    { persona = "task_oriented", turn_injection = { turn_index = 5, '
        'edit_primitive = "theme_switch", '
        'source_theme_id = "vehicle", '
        'target_theme_id = "fruit" } },\n'
        "]\n"
    )
    config_path = tmp_path / "batch_runs_conflicting_shared_persona_branch.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="must not define both base and branch shared-prefix "
        "'human_agent_config.persona' values",
    ):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_allows_branches_to_inherit_base_shared_persona(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "persona_branch"
dataset_name = "synthetic"
max_turns = 6

[base.human_agent_config]
persona = "casual_social"

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
    { prompt_id = "baseline_v3" },
    { prompt_id = "baseline_v4" },
]
"""
    config_path = tmp_path / "batch_runs_inherit_shared_persona_branch.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _, batch_runs, _, _ = _parse_batch_run_config_toml(str(config_path))

    assert [config.human_agent_prompt_id for config in batch_runs] == ["baseline_v3", "baseline_v4"]
    assert [config.human_agent_config.persona for config in batch_runs] == [None, None]
    assert all(
        config.metadata["branch_pre_vary_value"]
        == HumanAgentConfig(prompt_id="baseline_v3", persona="casual_social")
        for config in batch_runs
    )


def test_parse_batch_run_config_toml_rejects_branching_on_topic_discovery(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "ablation"
dataset_name = "synthetic"
max_turns = 6

[base.human_agent_config]
prompt_id = "baseline_v4"

[base.human_agent_config.topic_discovery]
enabled = false

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
    { prompt_id = "baseline_v4", topic_discovery = { enabled = false } },
    { prompt_id = "baseline_v4", topic_discovery = { enabled = true, turn_budget = 2 } },
]
"""
    config_path = tmp_path / "batch_runs_branch_topic_discovery.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="must not vary 'human_agent_config.topic_discovery'"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_branch_when_topic_discovery_is_enabled(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "ablation"
dataset_name = "synthetic"
max_turns = 6

[base.human_agent_config]
prompt_id = "baseline_v4"

[base.human_agent_config.topic_discovery]
enabled = true
trigger_mode = "turn_budget"
turn_budget = 2

[branch]
shared_turns = 3
vary_by = "max_verification_retries"
values = [1, 4]
"""
    config_path = tmp_path / "batch_runs_branch_with_topic_discovery.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="topic-discovery branching is automatic"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_reserved_branch_metadata_keys(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "ablation"
dataset_name = "synthetic"
max_turns = 6

[base.metadata]
branch_group = "custom"

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
    { prompt_id = "baseline_v1", guidance = "x" },
    { prompt_id = "baseline_v1", guidance = "y" },
]
"""
    config_path = tmp_path / "batch_runs.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="must not define reserved key"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_unsupported_branch_fields(tmp_path: Path) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "ablation"
dataset_name = "synthetic"
max_turns = 6

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
    { prompt_id = "baseline_v1", guidance = "x" },
    { prompt_id = "baseline_v1", guidance = "y" },
]
branch_turn = 3
"""
    config_path = tmp_path / "batch_runs.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported field"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_missing_guidance_for_guided_prompt(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "guided"
dataset_name = "synthetic"

[base.human_agent_config]
prompt_id = "baseline_v2"
"""
    config_path = tmp_path / "batch_runs_guided.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="requires guidance"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_guidance_for_guidance_free_prompt(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "exploratory"
dataset_name = "synthetic"

[base.human_agent_config]
prompt_id = "baseline_v3"
guidance = "Group by topic"
"""
    config_path = tmp_path / "batch_runs_exploratory.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="does not accept guidance"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_normalizes_blank_guidance_to_none_for_guidance_free_prompt(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "exploratory"
dataset_name = "synthetic"

[base.human_agent_config]
prompt_id = "baseline_v3"
guidance = "   "
"""
    config_path = tmp_path / "batch_runs_blank_guidance.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _, batch_runs, _, _ = _parse_batch_run_config_toml(str(config_path))

    assert batch_runs[0].guidance is None


def test_parse_batch_run_config_toml_parses_topic_discovery_config(tmp_path: Path) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "exploratory"
dataset_name = "synthetic"

[base.human_agent_config]
prompt_id = "baseline_v4"

[base.human_agent_config.topic_discovery]
enabled = true
trigger_mode = "turn_budget"
turn_budget = 3
max_rounds = 4
min_new_topics_per_round = 1
"""
    config_path = tmp_path / "batch_runs_topic_discovery.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _, batch_runs, _, _ = _parse_batch_run_config_toml(str(config_path))

    assert batch_runs[0].human_agent_config.topic_discovery == TopicDiscoveryConfig(
        enabled=True,
        trigger_mode="turn_budget",
        turn_budget=3,
        max_rounds=4,
        min_new_topics_per_round=1,
    )


def test_parse_batch_run_config_toml_uses_topic_discovery_defaults_for_optional_round_fields(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "exploratory"
dataset_name = "synthetic"

[base.human_agent_config]
prompt_id = "baseline_v4"

[base.human_agent_config.topic_discovery]
enabled = true
trigger_mode = "human_tool"
"""
    config_path = tmp_path / "batch_runs_topic_discovery_defaults.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _, batch_runs, _, _ = _parse_batch_run_config_toml(str(config_path))

    assert batch_runs[0].human_agent_config.topic_discovery == TopicDiscoveryConfig(
        enabled=True,
        trigger_mode="human_tool",
        turn_budget=None,
        max_rounds=3,
        min_new_topics_per_round=2,
    )


def test_parse_batch_run_config_toml_rejects_invalid_topic_discovery_config(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "exploratory"
dataset_name = "synthetic"

[base.human_agent_config]
prompt_id = "baseline_v4"

[base.human_agent_config.topic_discovery]
enabled = true
trigger_mode = "human_tool"
turn_budget = 3
"""
    config_path = tmp_path / "batch_runs_invalid_topic_discovery.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="turn_budget must be null"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_invalid_topic_discovery_round_config(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "exploratory"
dataset_name = "synthetic"

[base.human_agent_config]
prompt_id = "baseline_v4"

[base.human_agent_config.topic_discovery]
enabled = true
trigger_mode = "human_tool"
max_rounds = 0
"""
    config_path = tmp_path / "batch_runs_invalid_topic_discovery_rounds.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="max_rounds must be an integer >= 1"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_topic_discovery_for_unsupported_prompt(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "exploratory"
dataset_name = "synthetic"

[base.human_agent_config]
prompt_id = "baseline_v3"

[base.human_agent_config.topic_discovery]
enabled = true
trigger_mode = "turn_budget"
turn_budget = 3
"""
    config_path = tmp_path / "batch_runs_unsupported_topic_discovery.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="topic_discovery is only supported"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_topic_discovery_turn_budget_gte_max_turns(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "exploratory"
dataset_name = "synthetic"
max_turns = 3

[base.human_agent_config]
prompt_id = "baseline_v4"

[base.human_agent_config.topic_discovery]
enabled = true
trigger_mode = "turn_budget"
turn_budget = 3
"""
    config_path = tmp_path / "batch_runs_topic_discovery_turn_budget_exhausted.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="turn_budget must be less than max_turns"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_allows_branching_on_human_agent_config_without_base_config(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "prompt_branch"
dataset_name = "synthetic"
max_turns = 6

[branch]
shared_turns = 3
vary_by = "human_agent_config"
values = [
    { prompt_id = "baseline_v1", guidance = "Group by topic" },
    { prompt_id = "baseline_v2", guidance = "Group by topic" },
]
"""
    config_path = tmp_path / "batch_runs_branch_human_agent_config.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _, batch_runs, _, _ = _parse_batch_run_config_toml(str(config_path))

    assert [config.human_agent_prompt_id for config in batch_runs] == ["baseline_v1", "baseline_v2"]
    assert all(config.metadata["branch_vary_by"] == "human_agent_config" for config in batch_runs)
    assert all(
        config.metadata["branch_pre_vary_value"] == HumanAgentConfig(prompt_id="baseline_v3")
        for config in batch_runs
    )


def test_parse_batch_run_config_toml_parses_human_model_configs(tmp_path: Path) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1
reporter = "null"

[base]
name = "Run One"
dataset_name = "synthetic"
human_verifier_prompt_id = "baseline_v1"
clustering_agent_backend = "backend_v1"

[base.clustering_agent_backend_config]
prompt_id = "baseline_v1"

[base.clustering_agent_backend_config.intermittent_clustering_judge_model]
model = "openai/gpt-5-nano"

[base.clustering_agent_backend_config.model_config]
model = "openai/gpt-5-mini"

[base.human_agent_model_config]
model = "openai/gpt-5-mini"

[base.human_verifier_model_config]
model = "openai/gpt-5-nano"
"""
    config_path = tmp_path / "batch_runs_models.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _output_dir, batch_runs, _reporter_factory, _num_runners = _parse_batch_run_config_toml(
        str(config_path)
    )

    assert batch_runs[0].clustering_agent_backend_config == BackendV1ClusteringBackendConfig(
        prompt_id="baseline_v1",
        model_config=ModelConfig(model="openai/gpt-5-mini"),
        intermittent_clustering_judge_model=ModelConfig(model="openai/gpt-5-nano"),
    )
    assert batch_runs[0].human_agent_model_config == ModelConfig(model="openai/gpt-5-mini")
    assert batch_runs[0].human_verifier_model_config == ModelConfig(model="openai/gpt-5-nano")


def test_parse_batch_run_config_toml_parses_embedding_model_config(tmp_path: Path) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1
reporter = "null"

[base]
name = "Run One"
dataset_name = "synthetic"
enable_embeddings = true
embedding_force_recompute = true
embedding_batch_size = 32

[base.embedding_model_config]
backend = "huggingface_local"
device = "metal"
model = "openai/text-embedding-3-large"
dimensions = 512
"""
    config_path = tmp_path / "batch_runs_embeddings.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _output_dir, batch_runs, _reporter_factory, _num_runners = _parse_batch_run_config_toml(
        str(config_path)
    )

    assert batch_runs[0].enable_embeddings is True
    assert batch_runs[0].embedding_force_recompute is True
    assert batch_runs[0].embedding_batch_size == 32
    assert batch_runs[0].embedding_model_config == EmbeddingModelConfig(
        model="openai/text-embedding-3-large",
        backend="huggingface_local",
        device="mps",
        dimensions=512,
    )


def test_parse_batch_run_config_toml_parses_oneshot_kmeans_backend_config(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1
reporter = "null"

[base]
name = "Run One"
dataset_name = "synthetic"
clustering_agent_backend = "oneshot_kmeans_v1"
enable_embeddings = true

[base.clustering_agent_backend_config.model_config]
model = "google/gemini-3.1-flash-lite-preview"
temperature = 0.0
"""
    config_path = tmp_path / "batch_runs_oneshot_kmeans.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _output_dir, batch_runs, _reporter_factory, _num_runners = _parse_batch_run_config_toml(
        str(config_path)
    )

    assert batch_runs[0].clustering_agent_backend == "oneshot_kmeans_v1"
    assert batch_runs[0].enable_embeddings is True
    assert batch_runs[0].clustering_agent_backend_config == (
        OneShotKMeansV1ClusteringBackendConfig(
            model_config=ModelConfig(
                model="google/gemini-3.1-flash-lite-preview",
                temperature=0.0,
            )
        )
    )


def test_parse_batch_run_config_toml_parses_instruction_kmeans_backend_config(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1
reporter = "null"

[base]
name = "Run One"
dataset_name = "synthetic"
clustering_agent_backend = "instruction_kmeans_v1"

[base.clustering_agent_backend_config]
sample_window_size = 150
shuffle_seed = 11
cluster_summary_example_limit = 9
embedding_batch_size = 16

[base.clustering_agent_backend_config.model_config]
model = "openai/gpt-5.4"

[base.clustering_agent_backend_config.embedding_model_config]
backend = "huggingface_local"
device = "metal"
model = "Qwen/Qwen3-Embedding-8B"

[base.clustering_agent_backend_config.summary_model_config]
model = "openai/gpt-5.4-nano"

[base.clustering_agent_backend_config.intermittent_clustering_judge_model]
model = "openai/gpt-5.4"
"""
    config_path = tmp_path / "batch_runs_instruction_kmeans.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _output_dir, batch_runs, _reporter_factory, _num_runners = _parse_batch_run_config_toml(
        str(config_path)
    )

    assert batch_runs[0].clustering_agent_backend == "instruction_kmeans_v1"
    assert batch_runs[0].clustering_agent_backend_config == (
        InstructionKMeansV1ClusteringBackendConfig(
            model_config=ModelConfig(model="openai/gpt-5.4"),
            embedding_model_config=EmbeddingModelConfig(
                model="Qwen/Qwen3-Embedding-8B",
                backend="huggingface_local",
                device="mps",
            ),
            summary_model_config=ModelConfig(model="openai/gpt-5.4-nano"),
            intermittent_clustering_judge_model=ModelConfig(model="openai/gpt-5.4"),
            sample_window_size=150,
            shuffle_seed=11,
            cluster_summary_example_limit=9,
            embedding_batch_size=16,
        )
    )


def test_parse_batch_run_config_toml_rejects_oneshot_kmeans_without_embeddings(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1
reporter = "null"

[base]
name = "Run One"
dataset_name = "synthetic"
clustering_agent_backend = "oneshot_kmeans_v1"
"""
    config_path = tmp_path / "batch_runs_oneshot_kmeans_missing_embeddings.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="oneshot_kmeans_v1 requires enable_embeddings=true"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_embedding_model_config_sweep(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1
reporter = "null"

[base]
name = "Run One"
dataset_name = "synthetic"
enable_embeddings = true

[sweep]
vary_by = "embedding_model_config"
values = [
  { model = "openai/text-embedding-3-small" },
  { model = "openai/text-embedding-3-large" }
]
"""
    config_path = tmp_path / "batch_runs_embedding_sweep.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="'sweep.vary_by' must be one of"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_embedding_model_config_branch(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1

[base]
name = "ablation"
dataset_name = "synthetic"
max_turns = 6

[branch]
shared_turns = 3
vary_by = "embedding_model_config"
values = [
  { model = "openai/text-embedding-3-small" },
  { model = "openai/text-embedding-3-large" }
]
"""
    config_path = tmp_path / "batch_runs_bad_embedding_branch.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="'branch.vary_by' must be one of"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_legacy_model_config_key(tmp_path: Path) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1
reporter = "null"

[base]
name = "Run One"
dataset_name = "synthetic"

[base.model_config]
model = "openai/gpt-5-mini"
"""
    config_path = tmp_path / "batch_runs_legacy_model_config.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported field\\(s\\): model_config"):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_legacy_clustering_agent_fields(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1
reporter = "null"

[base]
name = "Run One"
dataset_name = "synthetic"
clustering_agent_prompt_id = "baseline_v1"

[base.clustering_agent_model_config]
model = "openai/gpt-5-mini"
"""
    config_path = tmp_path / "batch_runs_legacy_clustering_agent_fields.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=(
            "unsupported field\\(s\\): clustering_agent_model_config, clustering_agent_prompt_id"
        ),
    ):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_rejects_legacy_human_agent_fields(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1
reporter = "null"

[base]
name = "Run One"
dataset_name = "synthetic"
human_agent_prompt_id = "baseline_v1"
guidance = "Group by topic"
"""
    config_path = tmp_path / "batch_runs_legacy_human_agent_fields.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="unsupported field\\(s\\): guidance, human_agent_prompt_id",
    ):
        _parse_batch_run_config_toml(str(config_path))


def test_parse_batch_run_config_toml_parses_max_verification_retries(tmp_path: Path) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1
reporter = "null"

[base]
name = "Run One"
dataset_name = "synthetic"
max_verification_retries = 4
"""
    config_path = tmp_path / "batch_runs_retries.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    _output_dir, batch_runs, _reporter_factory, _num_runners = _parse_batch_run_config_toml(
        str(config_path)
    )

    assert batch_runs[0].max_verification_retries == 4


def test_parse_batch_run_config_toml_rejects_negative_max_verification_retries(
    tmp_path: Path,
) -> None:
    config_toml = """
output_dir = "data/out"
num_runners = 1
reporter = "null"

[base]
name = "Run One"
dataset_name = "synthetic"
max_verification_retries = -1
"""
    config_path = tmp_path / "batch_runs_bad_retries.toml"
    config_path.write_text(config_toml, encoding="utf-8")

    with pytest.raises(ValueError, match="max_verification_retries"):
        _parse_batch_run_config_toml(str(config_path))


def test_batch_run_result_to_dict_serializes_expected_fields() -> None:
    result = _build_batch_run_result()

    payload = result.to_dict()

    assert payload["run_ids"] == ["001-abc12345"]
    assert payload["timestamp"] == "20260101T010101Z"
    assert payload["config"]["name"] == "Example Run"
    assert payload["config"]["clustering_agent_backend"] == "backend_v1"
    assert (
        payload["config"]["clustering_agent_backend_config"]["model_config"]["model"]
        == "openai/gpt-5-mini"
    )
    assert (
        payload["config"]["clustering_agent_backend_config"]["intermittent_clustering_judge_model"][
            "model"
        ]
        == "openai/gpt-5.4"
    )
    assert payload["config"]["human_agent_config"] == {
        "prompt_id": "baseline_v1",
        "guidance": "Group by topic",
    }
    assert payload["config"]["human_verifier_prompt_id"] == "baseline_v1"
    assert payload["config"]["max_verification_retries"] == 2
    assert payload["results"][0]["turns_run"] == 3
    assert payload["results"][0]["stopped_reason"] == "max_turns"


def test_write_jsonl_writes_one_json_object_per_line(tmp_path: Path) -> None:
    result = _build_batch_run_result()

    output_path = tmp_path / "results.jsonl"
    write_jsonl([result], str(output_path))

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["config"]["name"] == "Example Run"
    assert parsed["results"][0]["turns_run"] == 3


def test_write_summary_aggregates_totals_and_stopped_reasons(tmp_path: Path) -> None:
    result_one = _build_batch_run_result()
    result_two = BatchRunResult(
        config=result_one.config,
        simulation_results=[
            SimulationResult(
                state={"messages": [], "dataset_metadata": {"size": 0}, "iteration_count": 0},
                turns_run=2,
                stopped_reason="clustering_saved",
                clustering_agent_accounting=_usage_summary(
                    4,
                    2,
                    model_id="openai/gpt-5-mini",
                ),
                human_agent_accounting=_usage_summary(
                    3,
                    1,
                    model_id="openai/gpt-5-mini",
                ),
            )
        ],
        config_slug="example-run-two",
    )

    output_path = tmp_path / "summary.json"
    write_summary([result_one, result_two], str(output_path))

    summary = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary["total_batch_run_configs"] == 2
    assert summary["total_runs"] == 2
    assert summary["total_turns"] == 5
    assert summary["components"] == ["clustering_agent", "human_agent"]
    assert summary["clustering_agent_accounting"]["total_tokens"] == 22
    assert summary["human_agent_accounting"]["total_tokens"] == 15
    assert summary["overall_accounting"]["total_tokens"] == 37
    assert summary["stopped_reasons"] == {"max_turns": 1, "clustering_saved": 1}
