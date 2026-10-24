"""Terminal progress helpers for batch dataset generation."""

from __future__ import annotations

import sys

_LAST_PROGRESS_MESSAGE_LENGTH = 0


def format_progress_bar(completed: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return f"[{'-' * width}]"
    ratio = min(max(completed / total, 0.0), 1.0)
    filled = int(ratio * width)
    if completed > 0 and filled == 0:
        filled = 1
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def render_generation_progress(
    completed_rows: int,
    total_rows: int,
    *,
    status: str,
    batch_index: int | None = None,
    batch_total: int | None = None,
) -> str:
    percentage = 0.0 if total_rows <= 0 else (completed_rows / total_rows) * 100
    progress = (
        f"{status}: {format_progress_bar(completed_rows, total_rows)} "
        f"{completed_rows}/{total_rows} rows ({percentage:5.1f}%)"
    )
    if batch_index is not None and batch_total is not None:
        progress += f" | batch {batch_index}/{batch_total}"
    return progress


def write_generation_progress(message: str, *, final: bool = False) -> None:
    global _LAST_PROGRESS_MESSAGE_LENGTH

    stream = sys.stderr
    if stream.isatty():
        padded = message + (" " * max(0, _LAST_PROGRESS_MESSAGE_LENGTH - len(message)))
        stream.write(f"\r{padded}")
        if final:
            stream.write("\n")
            _LAST_PROGRESS_MESSAGE_LENGTH = 0
        else:
            _LAST_PROGRESS_MESSAGE_LENGTH = len(message)
    else:
        stream.write(f"{message}\n")
    stream.flush()
