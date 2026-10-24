"""CLI runner for the standalone demo API with optional local embedding overrides."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

import uvicorn

from .config import DemoSettings, load_demo_settings


def _build_parser(defaults: DemoSettings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the conversational clustering demo API.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable code reload while developing the demo API.",
    )
    parser.add_argument(
        "--dataset",
        default=defaults.dataset_name,
        help="Dataset name to load for new demo sessions.",
    )
    parser.add_argument(
        "--backend",
        default=defaults.clustering_agent_backend,
        help="Clustering backend to use for new demo sessions.",
    )
    parser.add_argument(
        "--cors-origin",
        action="append",
        dest="cors_origins",
        help="Allowed CORS origin. Pass multiple times to allow multiple origins.",
    )
    parser.add_argument(
        "--disable-local-embeddings",
        action="store_true",
        help=(
            "Force the demo API to avoid local Hugging Face embeddings and use the backend's "
            "remote embedding defaults instead."
        ),
    )
    return parser


def _apply_cli_overrides(args: argparse.Namespace, defaults: DemoSettings) -> None:
    os.environ["DEMO_DATASET"] = str(args.dataset).strip() or defaults.dataset_name
    os.environ["DEMO_CLUSTERING_BACKEND"] = str(args.backend).strip() or (
        defaults.clustering_agent_backend
    )
    cors_origins = args.cors_origins or list(defaults.cors_origins)
    os.environ["DEMO_CORS_ORIGINS"] = ",".join(cors_origins)
    os.environ["DEMO_DISABLE_LOCAL_EMBEDDINGS"] = "1" if args.disable_local_embeddings else "0"


def run_demo_api_cli(argv: Sequence[str] | None = None) -> None:
    defaults = load_demo_settings()
    parser = _build_parser(defaults)
    args = parser.parse_args(argv)
    _apply_cli_overrides(args, defaults)
    uvicorn.run(
        "demo_api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    run_demo_api_cli()
