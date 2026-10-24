# Developer Guide: Working in `conversational-clustering`

This document is an initial guide for contributors who are newer to this repository or to software development. It explains how code is organized, where new logic should go, and how to add configuration cleanly.

## 1) Mental Model of the Repo

Think of the project as a pipeline with clear stages:

1. `main.py` chooses a run mode and routes control.
2. `simulation/` runs a single conversation session (interactive or simulated).
3. `batch_runs/` repeats many simulation runs from TOML config.
4. `evaluation/` reads saved clustering artifacts and computes evaluator outputs.

Core rule: keep each stage focused. Avoid mixing responsibilities across these boundaries.

## 2) Directory Ownership (What Goes Where)

- `main.py`
  - CLI argument parsing and mode selection only.
  - Thin orchestration layer. Do not place business logic here if it can live in a package module.

- `conversational_clustering/clustering_agent/`
  - Clustering state shape and backend registry.
  - Put concrete clustering methods and backend-specific behavior in `conversational_clustering/clustering_agent/backends/`.

- `conversational_clustering/human_agent.py`
  - Human-simulation LangGraph app and message sanitization logic.
  - Put simulated-human behavior here, not in `simulation/`.

- `conversational_clustering/simulation/`
  - Single-run session lifecycle: creating sessions, running turns, interactive loop, simulation loop.
  - Token accounting and reporter hooks for one conversation live here.

- `conversational_clustering/batch_runs/`
  - Multi-run orchestration over many simulation runs.
  - TOML parsing, batch config validation, run parallelization, and batch outputs (`results.jsonl`, `summary.json`) go here.
  - If you are running variations of human/clustering runs, this is the right home.

- `conversational_clustering/evaluation/`
  - Loading saved run artifacts and evaluating outputs.
  - Evaluator interface/registry and evaluation runners live here.
  - If logic is about scoring/analyzing outputs (not producing them), it belongs here.

- `conversational_clustering/data_processing/`
  - Dataset registration and loading (`DATASETS`, loader implementations).
  - New dataset definitions or loader mechanics belong here.

- `conversational_clustering/prompts/`
  - Prompt files and prompt ID registry.
  - Prompt versioning and required template-field validation belong here.

- `dataset_generation/`
  - Synthetic dataset presets and dataset generation CLI implementation.

- `demo_api/` and `demo_frontend/`
  - Local demo application.

- `batch_run_configs/` and `evaluation_configs/`
  - Example TOML configs for repeatable runs and evaluations.

- `scripts/`
  - Standalone helper scripts (keep package logic out of scripts).

## 3) Placement Rules: Quick Decision Checklist

Use this before writing code:

1. Is this about one conversation loop? Put it in `simulation/`.
2. Is this about running many variations? Put it in `batch_runs/`.
3. Is this about measuring/scoring saved outputs? Put it in `evaluation/`.
4. Is this about model prompts or prompt IDs? Put it in `prompts/`.
5. Is this about dataset registration/loading? Put it in `data_processing/`.
6. Is this clustering backend behavior? Put it in `clustering_agent/backends/`.
7. Is this simulated-human behavior? Put it in `human_agent.py`.
8. Is this only CLI wiring? Put it in `main.py`.

If a change touches multiple layers, keep each layer’s change minimal and focused.

## 4) How to Add Config Options (Recommended Pattern)

Treat config additions as contract changes. Add them end-to-end in one pass:

1. Add the field to the internal config type.
   - Batch runs: `conversational_clustering/batch_runs/config.py` (`BatchRunConfig`).
   - Evaluation: `conversational_clustering/evaluation/runner.py` (`EvaluationRunConfig`).

2. Parse and validate in one place.
   - Batch runs: `_parse_batch_run_config_toml(...)` in `conversational_clustering/batch_runs/batch_runs.py`.
   - Evaluation: parser helpers in `conversational_clustering/evaluation/runner.py`.
   - Prefer explicit type checks and clear `ValueError` messages.

3. Thread the value through execution.
   - Pass config values from parser -> runner -> called function(s).
   - Avoid reading TOML fields directly in deep modules.

4. Persist or report if relevant.
   - If a setting affects reproducibility, include it in artifacts/reports.
   - Example: batch run `to_dict()` output in `conversational_clustering/batch_runs/config.py`.

5. Update sample TOMLs.
   - `batch_run_configs/example_minimal.toml`
   - `batch_run_configs/example_maximal.toml`
   - `evaluation_configs/example_single_maximal.toml`
   - `evaluation_configs/example_pairwise_maximal.toml`

6. Expose CLI knobs only when needed.
   - If it is run-level behavior, add a `main.py` arg.
   - If it is batch/evaluation-specific, prefer TOML over extra CLI flags.

### Config Design Principles

- Keep one source of truth for defaults (ideally config dataclass/parser).
- Validate early at parse time, not late in runtime loops.
- Reject unknown or malformed fields when strictness improves safety (see `clustering_agent_model_config` parsing).
- Prefer additive options with safe defaults to avoid breaking existing configs.

## 5) Common Extension Workflows

### Add a new batch-run capability

1. Extend `BatchRunConfig` in `conversational_clustering/batch_runs/config.py`.
2. Parse/validate in `conversational_clustering/batch_runs/batch_runs.py`.
3. Use the option in `conversational_clustering/batch_runs/runner.py`.
4. Include it in serialized output (`BatchRunResult.to_dict`).
5. Update sample `batch_run_configs/`.

### Add a new evaluator

1. Create evaluator class in `conversational_clustering/evaluation/methods/`.
2. Implement `from_params(...)` and `evaluate(...)`.
3. Register in `conversational_clustering/evaluation/registry.py` or `conversational_clustering/evaluation/pairwise_registry.py`.

### Add a new prompt version

1. Add prompt file under the narrowest matching directory in `conversational_clustering/prompts/`.
2. Register prompt ID in `conversational_clustering/prompts/registry.py`.
3. Ensure required template fields are declared if needed.
4. Reference the new prompt ID in CLI args or TOML configs.

### Add a new dataset

1. Add dataset entry to `conversational_clustering/data_processing/datasets.py`.
2. Ensure file path and `item_column` are correct.
3. If the dataset is intended for benchmark-aware evaluation, treat it as a paired dataset artifact:
   - add the dataset CSV under `data/`
   - construct a matching benchmark spec JSON alongside it
   - register both the dataset path and `benchmark_spec_path` in `conversational_clustering/data_processing/datasets.py`
4. Use `--list-datasets` to verify visibility via CLI.
5. Add documentation in `README.md` describing the dataset contract and any required colocated files.

## 6) Code Conventions in This Repository

Baseline conventions:

- Python `3.11`, 4-space indentation, PEP 8 defaults.
- `snake_case` for functions/modules/variables, `PascalCase` for classes.
- Type hints on public functions and dataclass/Pydantic models.
- Keep modules focused and composable.
- Add a one-sentence file summary docstring at top of Python files with logic.

Tooling and checks:

- Dependency/env management: `uv`
- Optional Nix dev shell: `nix develop`
- Lint/format: `ruff`
- Type checks: `ty`
- Pre-commit hooks: `.pre-commit-config.yaml`

Use this command before opening or merging a PR:

```sh
uv run pre-commit run --all-files
```

## 7) Expandability and Modularity Expectations

- Prefer composition over special-case branching in one large function.
- Keep parsing/validation close to config boundaries.
- Keep side effects (file writes, logging, print output) in orchestration layers.
- Keep reusable logic importable from `conversational_clustering/` modules.
- Avoid turning `main.py` into a logic hub; it should stay mostly routing.

When unsure, choose the location based on lifecycle stage (single run vs multi-run vs post-run evaluation).

## 8) Suggested PR Checklist

1. New logic is in the correct module boundary.
2. Config changes are parsed, validated, threaded, and documented in sample TOML.
3. Public behavior changes include clear error messages and output updates where needed.
4. Top-of-file summary docstrings still match file behavior.
5. `uv run pre-commit run --all-files` passes.
