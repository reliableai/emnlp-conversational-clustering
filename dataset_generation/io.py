"""Generic file output helpers for structured dataset generation."""

from __future__ import annotations

import csv
import os


def default_summary_path(output_path: str) -> str:
    directory, filename = os.path.split(output_path)
    stem, _ = os.path.splitext(filename)
    summary_filename = f"{stem}_summary.txt"
    if directory:
        return os.path.join(directory, summary_filename)
    return summary_filename


def write_text(path: str, text: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def write_csv(path: str, rows: list[dict[str, str]], fieldnames: tuple[str, ...]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
