"""CLI entry point for running interactive, simulated, or batch-run clustering sessions."""

import argparse

from conversational_clustering.batch_runs import run_batch_runs
from conversational_clustering.clustering_agent.backends import (
    DEFAULT_CLUSTERING_AGENT_BACKEND,
    default_clustering_agent_backend_config,
)
from conversational_clustering.data_processing import DATASETS
from conversational_clustering.evaluation import EvaluationRunFailed, run_evaluation_config
from conversational_clustering.logging_utils import configure_console_logger
from conversational_clustering.prompts import (
    human_agent_prompt_supports_guidance,
    validate_human_agent_prompt_guidance,
)
from conversational_clustering.simulation import (
    ConsoleReporter,
    create_session,
    maybe_build_intermittent_clustering_judge_save_path,
    run_human_simulation,
    run_interactive_session,
)

DEFAULT_DATASET = "synthetic"
DEFAULT_MODE = "interactive"
DEFAULT_HUMAN_AGENT_PROMPT_ID = "baseline_v3"
DEFAULT_HUMAN_VERIFIER_PROMPT_ID = "baseline_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Conversational clustering CLI.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=sorted(DATASETS),
        help="Dataset key to load",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all dataset keys",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of items loaded from the dataset",
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "simulate", "batch_run", "evaluate"],
        help=(
            "Run mode: interactive (user talks to agent), simulate "
            "(single human agent/clustering agent conversation), batch_run "
            "(run provided batch-runs config), or evaluate (run evaluation config). "
            "Defaults to interactive, batch_run when --batch-run-config is passed,"
            "or evaluate when --evaluation-config is passed."
        ),
    )
    parser.add_argument(
        "--batch-run-config",
        type=str,
        help="Path to batch-runs config.toml. Required when `--mode batch_run`, otherwise ignored.",
    )
    parser.add_argument(
        "--human-guidance",
        default="",
        help="Guidance for the simulated human about clustering themes.",
    )
    parser.add_argument(
        "--simulation-turns",
        type=int,
        default=8,
        help="Maximum number of simulated turns to run.",
    )
    parser.add_argument(
        "--evaluation-config",
        type=str,
        dest="evaluation_config",
        help=(
            "Path to evaluation run config TOML for multi-evaluator evaluation of "
            "an existing batch-run output directory. Reports default to "
            "`data/evaluation_results/...` unless `report_path` is set in the config."
        ),
    )
    parser.add_argument(
        "--clustering-agent-backend",
        default=DEFAULT_CLUSTERING_AGENT_BACKEND,
        help=(
            "Clustering agent backend ID (used in interactive and simulate modes). "
            f"Default: {DEFAULT_CLUSTERING_AGENT_BACKEND}"
        ),
    )
    parser.add_argument(
        "--human-agent-prompt-id",
        default=DEFAULT_HUMAN_AGENT_PROMPT_ID,
        help=(
            "Prompt ID for the human agent system prompt (used in simulate mode). "
            "Also validated in interactive mode."
        ),
    )
    parser.add_argument(
        "--human-verifier-prompt-id",
        default=DEFAULT_HUMAN_VERIFIER_PROMPT_ID,
        help=(
            "Prompt ID for the human verifier system prompt (used in simulate mode). "
            "Also validated in interactive mode."
        ),
    )
    parser.add_argument(
        "--enable-embeddings",
        action="store_true",
        help="Enable eager embeddings loading/computation for dataset items.",
    )
    parser.add_argument(
        "--intermittent-clustering",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable hidden intermittent clustering capture in interactive and simulate modes. "
            "Disabled by default; use --intermittent-clustering to enable it."
        ),
    )
    args = parser.parse_args()

    if args.evaluation_config and args.batch_run_config:
        parser.error("`--evaluation-config` cannot be combined with `--batch-run-config`.")

    if args.mode is None:
        if args.evaluation_config:
            args.mode = "evaluate"
        elif args.batch_run_config:
            args.mode = "batch_run"
        else:
            args.mode = DEFAULT_MODE

    if args.evaluation_config:
        if args.mode != "evaluate":
            parser.error("`--evaluation-config` is only supported with `--mode evaluate`.")
    elif args.mode == "evaluate":
        parser.error("When running with `--mode evaluate`, pass `--evaluation-config`.")

    if args.batch_run_config:
        if args.mode != "batch_run":
            parser.error("`--batch-run-config` is only supported with `--mode batch_run`.")
    elif args.mode == "batch_run":
        parser.error("When running with `--mode batch_run`, pass `--batch-run-config`.")
    if args.list_datasets:
        print("Available datasets:")
        for name in sorted(DATASETS):
            config = DATASETS[name]
            print(f" {name}: path={config['path']} item_column={config['item_column']}")
        return

    if args.mode in ["interactive", "simulate"]:
        try:
            clustering_agent_backend_config = default_clustering_agent_backend_config(
                args.clustering_agent_backend
            )
        except (TypeError, ValueError) as exc:
            parser.error(f"Invalid clustering-agent backend config: {exc}")
        session = create_session(
            args.dataset,
            limit=args.limit,
            clustering_agent_backend=args.clustering_agent_backend,
            clustering_agent_backend_config=clustering_agent_backend_config,
            human_agent_prompt_id=args.human_agent_prompt_id,
            human_verifier_prompt_id=args.human_verifier_prompt_id,
            enable_embeddings=args.enable_embeddings,
            enable_intermittent_clustering=args.intermittent_clustering,
            intermittent_clustering_judge_save_path=(
                maybe_build_intermittent_clustering_judge_save_path(
                    None,
                    clustering_agent_backend=args.clustering_agent_backend,
                    clustering_agent_backend_config=clustering_agent_backend_config,
                )
                if args.intermittent_clustering
                else None
            ),
        )
        reporter = ConsoleReporter()

        if args.mode == "interactive":
            run_interactive_session(session, reporter=reporter)
        elif args.mode == "simulate":
            try:
                guidance = args.human_guidance
                if human_agent_prompt_supports_guidance(args.human_agent_prompt_id):
                    guidance = guidance.strip()
                    if not guidance:
                        guidance = input(
                            "Simulated human guidance (themes or what to look for): "
                        ).strip()
                guidance = validate_human_agent_prompt_guidance(
                    prompt_id=args.human_agent_prompt_id,
                    guidance=guidance,
                )
            except ValueError as exc:
                parser.error(str(exc))

            run_human_simulation(
                session,
                guidance=guidance,
                max_turns=args.simulation_turns,
                reporter=reporter,
            )
        return

    elif args.mode == "batch_run":
        run_batch_runs(args.batch_run_config)
        return

    elif args.mode == "evaluate":
        configure_console_logger()
        try:
            run_evaluation_config(args.evaluation_config)
        except EvaluationRunFailed as exc:
            parser.exit(status=1, message=f"{exc}\n")
        return


if __name__ == "__main__":
    main()
