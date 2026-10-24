"""Unit tests for simulation runner data loader construction."""

from unittest import mock

import pytest

import conversational_clustering.simulation.runner as simulation_runner


def test_build_data_loader_rejects_unknown_dataset() -> None:
    with mock.patch.dict(simulation_runner.DATASETS, {}, clear=True):
        with pytest.raises(ValueError, match="Unknown dataset"):
            simulation_runner.build_data_loader("missing")


def test_build_data_loader_rejects_missing_file() -> None:
    with mock.patch.dict(
        simulation_runner.DATASETS,
        {"demo": {"path": "/tmp/does-not-exist.csv", "item_column": "item"}},
        clear=True,
    ):
        with mock.patch(
            "conversational_clustering.simulation.runner.os.path.exists", return_value=False
        ):
            with pytest.raises(FileNotFoundError, match="was not found"):
                simulation_runner.build_data_loader("demo")


def test_build_data_loader_returns_csv_loader_for_known_dataset() -> None:
    with mock.patch.dict(
        simulation_runner.DATASETS,
        {"demo": {"path": "/tmp/demo.csv", "item_column": "item"}},
        clear=True,
    ):
        with mock.patch(
            "conversational_clustering.simulation.runner.os.path.exists", return_value=True
        ):
            with mock.patch(
                "conversational_clustering.simulation.runner.CsvDataLoader",
                autospec=True,
            ) as csv_loader_cls:
                sentinel_loader = object()
                csv_loader_cls.return_value = sentinel_loader

                loader = simulation_runner.build_data_loader(
                    "demo",
                    limit=4,
                    random_seed=9,
                )

    assert loader is sentinel_loader
    csv_loader_cls.assert_called_once_with(
        "/tmp/demo.csv",
        item_column="item",
        limit=4,
        random_seed=9,
        dataset_name="demo",
        enable_embeddings=False,
        embedding_model_config=None,
        embedding_force_recompute=False,
        embedding_batch_size=64,
        usage_collector=None,
    )
