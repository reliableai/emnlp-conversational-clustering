# Repository Guidelines

If you make changes to any files that would change the context within this file, update this file.

## General Guidance 
- If an additional configuration field is added for the batch run config, ensure that `batch_run_configs/example_maximal.toml` still includes all configurable fields and `batch_run_configs/example_minimal.toml` includes all required fields. 
- If an additional single-evaluation configuration field is added, ensure that `evaluation_configs/example_single_maximal.toml` still includes all supported single evaluation fields.
- If an additional pairwise-evaluation configuration field is added, ensure that `evaluation_configs/example_pairwise_maximal.toml` still includes all supported pairwise evaluation fields.

## Coding Style & Naming Conventions
- Use type hints on public functions and dataclass/Pydantic models.
- Keep modules focused; place shared orchestration in package submodules rather than `main.py`.
- Include one-sentence summaries of functionality at the top of python files. These are only required if the file contains logic and is not needed for blank files or `__init__.py` files. If you ever change a file such that this summary is no longer accurate, update the summary as well.
- Prefer expanding `main.py` over adding new scripts. New scripts should be added to a dedicated `scripts/` directory. Keep `conversational_clustering/` importable/library focused. 
- Programmatically consumed LLM output must use validated structured output with an explicit typed schema. 
- Use a hard cutover approach and never implement backward compatibility. If a change is intentionally breaking (hard cutover), explicitly label it as a breaking change and list impacted workflows.

## Reviewing
- Always run `uv run pre-commit run --all-files` during a review to check linting, formatting, and types. 
- Run `uv run python -m pytest -v` to execute the unit test suite.
- In addition to enumerating issues, provide a short summary of the changes. 
- Explicitly assess the PR on these dimensions: (1) minimizes code churn, (2) aligns with existing architecture and coding patterns, (3) improves long-term maintainability, and (4) supports future extension without unnecessary rework.
- Output format for reviews:
    - `Findings` first, ordered by severity (`high`, `medium`, `low`), each with file/line reference and concrete impact.
    - Include a suggested fix for each finding.
    - If no findings, explicitly say so and list residual risks/testing gaps.
- For non-code file additions:
    - Ensure that any added example config file corresponds to a repeatable experiment intended for ongoing use, not a one-off config created only to test a single PR.
    - Verify that any new prompt file is placed in a directory whose scope matches its purpose. If the existing directory structure does not fit cleanly, suggest a clearer directory layout instead of placing the file in a loosely related location.


## Agent Notes
If you find something confusing or difficult, add notes on what you learned to solve the problem to this section.
- When tests mock `with_structured_output(...)` call sites, update the fake to accept newer keyword arguments such as `include_raw=True`; otherwise structured-output callers can silently fall back to default behavior in tests while production uses a different response shape.
- OpenAI/OpenRouter strict structured-output schemas can fail if nested generated objects contain derived/defaulted fields like `topic_id`; keep those fields out of the LLM-facing schema and derive them after validation instead.
