#!/usr/bin/env python3
"""CLI utility for generating synthetic text datasets and saving them as CSV."""

import argparse
import csv
import json
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr

SYSTEM_PROMPT = (
    "You generate synthetic text datasets. "
    "Always maximize the number of distinct themes within the dataset, "
    "provided these still adhere to the user's prompt. "
    "Return a JSON array of strings only, no extra text."
)


def _parse_items(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.startswith("["):
        return json.loads(raw)
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return a JSON array.")
    return json.loads(raw[start : end + 1])


def _generate_batch(llm: ChatOpenRouter, prompt: str, count: int) -> list[str]:
    user_prompt = (
        f"Create {count} unique items for a dataset.\n"
        f"Prompt: {prompt}\n"
        "Return a JSON array of strings."
    )
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)])
    if not isinstance(response.content, str):
        raise TypeError("Expected string response content from model")
    return _parse_items(response.content)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic text dataset using OpenRouter."
    )
    parser.add_argument("--prompt", required=False, help="Describe the dataset you want.")
    parser.add_argument("--count", type=int, default=200, help="Total number of items to generate.")
    parser.add_argument("--batch-size", type=int, default=50, help="Items per API call.")
    parser.add_argument("--out", default="data/synthetic_dataset.csv", help="Output CSV path.")
    parser.add_argument(
        "--model", default="google/gemini-3-flash-preview", help="OpenRouter model name."
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    args = parser.parse_args()

    prompt = args.prompt or input("Dataset prompt: ").strip()
    if not prompt:
        print("Prompt is required.", file=sys.stderr)
        return 2

    load_dotenv()

    api_key_env = os.getenv("OPENROUTER_API_KEY")
    if not api_key_env:
        print("OPENROUTER_API_KEY is not set.", file=sys.stderr)
        return 2

    llm = ChatOpenRouter(
        openrouter_api_base="https://openrouter.ai/api/v1",
        openrouter_api_key=SecretStr(api_key_env),
        model=args.model,
        temperature=args.temperature,
    )

    total = args.count
    batch_size = max(1, args.batch_size)
    items: list[str] = []

    while len(items) < total:
        remaining = total - len(items)
        take = min(batch_size, remaining)
        batch = _generate_batch(llm, prompt, take)
        for item in batch:
            if isinstance(item, str) and item.strip():
                items.append(item.strip())
        items = items[:total]
        print(f"Generated {len(items)}/{total}", file=sys.stderr)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["item"])
        writer.writeheader()
        for item in items:
            writer.writerow({"item": item})

    print(f"Wrote {len(items)} items to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
