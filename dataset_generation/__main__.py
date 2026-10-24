"""Run the structured dataset generator as a Python module."""

from .cli import run_structured_dataset_cli

if __name__ == "__main__":
    raise SystemExit(run_structured_dataset_cli())
