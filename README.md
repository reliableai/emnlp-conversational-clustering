# Conversational Clustering
Implementation repo for conversational clustering.

## Setup 
### Environment
Project dependencies are managed by [uv](https://github.com/astral-sh/uv).

To generate virtual environment at `.venv`, run: 
```sh
uv sync
```

To include development tools (linting/formatting hooks), run:
```sh
uv sync --dev
```

### Environment Variables
Add a `.env` file with 
```
OPENROUTER_API_KEY=sk-or-v1-...
```
(where this is your OpenRouter API key)

## Linting and formatting
This repository uses `ruff` for linting and formatting, with `pre-commit` hooks.

One-time setup (each developer, per clone):
```sh
uv sync --dev
uv run pre-commit install --install-hooks
```

After setup, hooks run automatically on `git commit`.
- If hooks auto-format files, the commit will fail once so you can review/stage changes.
- Re-run `git commit` after staging the hook changes.

Run checks manually:
```sh
uv run pre-commit run --all-files
```

## Running tests
The unit test suite uses `pytest`.

Run all unit tests:
```sh
uv run pytest
```

## Generating datasets
Structured dataset generation is available through the package entrypoints: 
```sh
uv run generate-structured-dataset --help
uv run generate-enterprise-ticket-dataset --help
```

Generate a social media dataset:
```sh
uv run generate-structured-dataset --preset social_media --rows 210 --output data/social_media_dataset.csv
```

Generate a restaurant review dataset:
```sh
uv run generate-structured-dataset --preset restaurant_review --rows 210 --output data/restaurant_review_dataset.csv
```

## Datasets
Registered datasets live under the repo-root `data/` directory and are configured in `conversational_clustering/data_processing/datasets.py`.

The baseline example dataset is `enterprise_ticket`:

- CSV path: `data/enterprise_ticket_dataset_1000.csv`
- Benchmark spec path: `data/enterprise_ticket_dataset.benchmark.json`

## Running the CLI
The main CLI supports interactive, simulated, batch-run, and evaluation modes:
```sh
uv run python main.py --help
```

### Running a batch run
Run the `enterprise_ticket_run` config:
```sh
uv run python main.py --batch-run-config batch_run_configs/enterprise_ticket_run.toml
```

This writes results to `data/batch_run_results/enterprise_ticket_run`.

### Running an evaluation
Most enterprise-ticket evaluation configs read from `data/batch_run_results/enterprise_ticket_run`, so run the batch example above first.

Run final clustering quality:
```sh
uv run python main.py --evaluation-config evaluation_configs/enterprise_ticket_evaluation/enterprise_ticket_final_clustering_quality.toml
```

Run discovery quality:
```sh
uv run python main.py --evaluation-config evaluation_configs/enterprise_ticket_evaluation/enterprise_ticket_discovery_quality.toml
```

Run efficiency:
```sh
uv run python main.py --evaluation-config evaluation_configs/enterprise_ticket_evaluation/enterprise_ticket_efficiency.toml
```

Run robustness:
```sh
uv run python main.py --evaluation-config evaluation_configs/enterprise_ticket_evaluation/enterprise_ticket_robustness.toml
```

### Demo frontend
The frontend defaults to `http://127.0.0.1:8000`; set `VITE_API_BASE_URL` if the API runs elsewhere.
Run the API:
```sh
uv run run-demo-api --reload
```

To force the demo API to avoid local embedding models:
```sh
uv run run-demo-api --reload --disable-local-embeddings
```

Run the frontend:
```sh
cd demo_frontend
pnpm install
bun run dev
```
