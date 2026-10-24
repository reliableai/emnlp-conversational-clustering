"""CLI to run step-one discovery on a CSV corpus: profile, cluster two ways, propose dimensions.

Example:
    uv run python scripts/run_discovery.py \
        --path data/enteprise_dataset_1000/enterprise_ticket_dataset_1000.csv \
        --item-column ticket_text --dataset-name enterprise_ticket --k 20
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from conversational_clustering.discovery.pipeline import run_discovery


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-one discovery on a CSV corpus.")
    parser.add_argument("--path", required=True, help="Path to the CSV corpus.")
    parser.add_argument("--item-column", required=True, help="Column holding the item text.")
    parser.add_argument("--dataset-name", required=True, help="Name used in reports and caches.")
    parser.add_argument("--k", type=int, default=20, help="Number of K-means clusters per view.")
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip the semantic (embedding) clustering arm; run TF-IDF only.",
    )
    parser.add_argument(
        "--json-out", default=None, help="Optional path to write the full result JSON."
    )
    args = parser.parse_args()

    df = pd.read_csv(args.path, low_memory=False).drop_duplicates().reset_index(drop=True)
    result = run_discovery(
        df=df,
        dataset_name=args.dataset_name,
        item_column=args.item_column,
        k=args.k,
        use_embeddings=not args.no_embeddings,
    )

    print(
        f"\n=== {result.profile.dataset_name}: {result.profile.n_rows} items, "
        f"{result.profile.n_columns} columns ==="
    )
    print(f"clusterings: {', '.join(c.method for c in result.clusterings)}")
    print(f"\n=== proposed dimensions ({len(result.proposals.dimensions)}) ===")
    for dim in result.proposals.dimensions:
        flag = "metadata" if dim.already_in_metadata else "latent"
        print(f"\n• {dim.name}  [{dim.shape}, {dim.grounding}, {flag}]")
        print(f"    {dim.description}")
        print(f"    e.g. {', '.join(dim.example_labels)}")
        print(f"    why: {dim.rationale}")
    if result.proposals.notes:
        print(f"\nnotes: {result.proposals.notes}")

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(result.model_dump(), handle, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
