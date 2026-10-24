#!/usr/bin/env python3
"""CLI utility for building an evaluation benchmark spec from enterprise ticket CSV labels."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conversational_clustering.evaluation.benchmark_builders import (
    DEFAULT_ENTERPRISE_TICKET_THEME_DEFINITIONS,
    build_enterprise_ticket_benchmark_spec,
    load_labeled_csv_rows,
    write_benchmark_spec,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a benchmark spec JSON from the generated enterprise ticket dataset CSV."
    )
    parser.add_argument(
        "--input",
        default="data/enterprise_ticket_dataset_1000.csv",
        help="Input enterprise ticket CSV path.",
    )
    parser.add_argument(
        "--output",
        default="data/enterprise_ticket_dataset.benchmark.json",
        help="Output benchmark spec JSON path.",
    )
    parser.add_argument(
        "--dataset-name",
        default="enterprise_ticket",
        help="Dataset registry name stored in the benchmark spec.",
    )
    parser.add_argument(
        "--item-column",
        default="ticket_text",
        help="CSV column to use as the benchmark item text.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample size to match a limited batch run.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed to use when --limit is provided.",
    )
    parser.add_argument(
        "--theme-column",
        action="append",
        dest="theme_columns",
        default=None,
        help=(
            "Optional benchmark theme column. Repeat to override the default theme set. "
            "Defaults to: "
            + ", ".join(defn.theme_id for defn in DEFAULT_ENTERPRISE_TICKET_THEME_DEFINITIONS)
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    rows = load_labeled_csv_rows(
        args.input,
        limit=args.limit,
        random_seed=args.random_seed,
    )
    payload = build_enterprise_ticket_benchmark_spec(
        rows,
        dataset_name=args.dataset_name,
        item_column=args.item_column,
        theme_columns=args.theme_columns,
    )
    write_benchmark_spec(args.output, payload)
    print(
        "Wrote benchmark spec with "
        f"{len(payload['items'])} items, {len(payload['themes'])} themes, "
        f"and {len(payload['references'])} references to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
