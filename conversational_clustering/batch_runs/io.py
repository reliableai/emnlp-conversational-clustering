"""File output helpers for batch-run artifacts and aggregate summaries."""

import json
import os

from ..accounting import merge_usage_summaries
from .config import BatchRunResult


def write_jsonl(results: list[BatchRunResult], output_path: str) -> None:
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result.to_dict()) + "\n")


def write_summary(results: list[BatchRunResult], output_path: str) -> None:
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    all_simulation_results = [
        simulation_result
        for batch_run_result in results
        for simulation_result in batch_run_result.simulation_results
    ]
    summary = {
        "total_batch_run_configs": len(results),
        "total_runs": len(all_simulation_results),
        "total_turns": sum(r.turns_run for r in all_simulation_results),
        "components": ["clustering_agent", "human_agent"],
        "clustering_agent_accounting": merge_usage_summaries(
            simulation_result.clustering_agent_accounting
            for simulation_result in all_simulation_results
        ).to_dict(),
        "human_agent_accounting": merge_usage_summaries(
            simulation_result.human_agent_accounting for simulation_result in all_simulation_results
        ).to_dict(),
        "overall_accounting": merge_usage_summaries(
            accounting
            for simulation_result in all_simulation_results
            for accounting in (
                simulation_result.clustering_agent_accounting,
                simulation_result.human_agent_accounting,
            )
        ).to_dict(),
        "stopped_reasons": {},
    }

    for r in all_simulation_results:
        reason = r.stopped_reason
        summary["stopped_reasons"][reason] = summary["stopped_reasons"].get(reason, 0) + 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
