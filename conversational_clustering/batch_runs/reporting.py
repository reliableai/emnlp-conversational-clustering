"""Terminal progress reporting helpers for batch-run execution status."""

import os
import sys
import time
from collections.abc import Callable
from typing import TextIO

from ..simulation.runner import SimulationResult
from .config import BatchRunResult

_DEFAULT_BAR_WIDTH = 20


class BatchProgressReporter:
    def __init__(
        self,
        *,
        total_runs: int,
        output_dir: str,
        bar_width: int = _DEFAULT_BAR_WIDTH,
        stream: TextIO | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        self.total_runs = total_runs
        self.output_dir = output_dir
        self.bar_width = bar_width
        self._stream = stream or sys.stdout
        self._now = now or time.monotonic

        self.completed_runs = 0
        self.successful_runs = 0
        self.failed_runs = 0
        self.failure_reasons: dict[str, int] = {}

        self._start_time = self._now()
        self._is_tty = self._stream.isatty()
        self._last_tty_line_len = 0
        self._last_snapshot_completed = 0
        self._last_snapshot_percent = 0.0

    def start(self) -> None:
        self._render_progress_line()

    def tick(self) -> None:
        self._render_progress_line()

    def record_run(
        self,
        *,
        config_slug: str,
        run_id: str,
        simulation_result: SimulationResult,
    ) -> None:
        self.completed_runs += 1
        stopped_reason = simulation_result.stopped_reason
        if self._is_run_failure(stopped_reason):
            self.failed_runs += 1
            self.failure_reasons[stopped_reason] = self.failure_reasons.get(stopped_reason, 0) + 1
            self._print_event_line(f"FAIL {config_slug}/{run_id} reason={stopped_reason}")
        else:
            self.successful_runs += 1
        self._render_progress_line()

    def add_expected_runs(self, additional_runs: int) -> None:
        if additional_runs <= 0:
            return
        self.total_runs += additional_runs
        self._render_progress_line()

    def finish(self, results: list[BatchRunResult]) -> None:
        self._flush_live_tty_line()
        failure_reasons_payload = (
            ", ".join(f"{reason}={count}" for reason, count in sorted(self.failure_reasons.items()))
            if self.failure_reasons
            else "none"
        )
        print("Batch Progress Summary", file=self._stream, flush=True)
        print(
            f"duration={self._format_duration(self._now() - self._start_time)} "
            f"total={self.total_runs} ok={self.successful_runs} fail={self.failed_runs}",
            file=self._stream,
            flush=True,
        )
        print(f"failure_reasons={failure_reasons_payload}", file=self._stream, flush=True)
        print(
            f"output_dir={os.path.abspath(self.output_dir)}",
            file=self._stream,
            flush=True,
        )

    @staticmethod
    def _is_run_failure(stopped_reason: str) -> bool:
        return stopped_reason.startswith("run_error_")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        clamped = max(0, int(seconds))
        minutes, secs = divmod(clamped, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours}h{minutes:02d}m"
        if minutes > 0:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"

    def _build_status_line(self, *, now: float) -> str:
        elapsed = max(0.0, now - self._start_time)
        percent_complete = (
            100.0 if self.total_runs == 0 else (self.completed_runs / self.total_runs) * 100.0
        )
        pending_runs = max(self.total_runs - self.completed_runs, 0)
        bar = self._build_progress_bar()
        return (
            f"{bar} {percent_complete:5.1f}% ({self.completed_runs}/{self.total_runs}) "
            f"| ok={self.successful_runs} fail={self.failed_runs} pending={pending_runs} "
            f"| elapsed {self._format_duration(elapsed)}"
        )

    def _build_progress_bar(self) -> str:
        if self.total_runs <= 0:
            return "[" + ("-" * self.bar_width) + "]"
        ratio = min(max(self.completed_runs / self.total_runs, 0.0), 1.0)
        filled = int(round(ratio * self.bar_width))
        return "[" + ("#" * filled) + ("-" * (self.bar_width - filled)) + "]"

    def _render_progress_line(self, *, final: bool = False, force_snapshot: bool = False) -> None:
        now = self._now()
        line = self._build_status_line(now=now)
        if self._is_tty and not final and not force_snapshot:
            padded = line.ljust(self._last_tty_line_len)
            print(f"\r{padded}", end="", file=self._stream, flush=True)
            self._last_tty_line_len = max(self._last_tty_line_len, len(line))
            return

        if self._is_tty and self._last_tty_line_len > 0:
            print("", file=self._stream, flush=True)
            self._last_tty_line_len = 0

        if not self._is_tty and not final and not force_snapshot:
            percent_complete = (
                100.0 if self.total_runs == 0 else (self.completed_runs / self.total_runs) * 100.0
            )
            should_emit = (
                self.completed_runs == self.total_runs
                or self.completed_runs - self._last_snapshot_completed >= 20
                or percent_complete - self._last_snapshot_percent >= 5.0
            )
            if not should_emit:
                return
            self._last_snapshot_completed = self.completed_runs
            self._last_snapshot_percent = percent_complete

        print(line, file=self._stream, flush=True)
        if not self._is_tty:
            self._last_snapshot_completed = self.completed_runs
            self._last_snapshot_percent = (
                100.0 if self.total_runs == 0 else (self.completed_runs / self.total_runs) * 100.0
            )

    def _print_event_line(self, message: str) -> None:
        self._flush_live_tty_line()
        print(message, file=self._stream, flush=True)
        if self._is_tty:
            self._render_progress_line()

    def _flush_live_tty_line(self) -> None:
        if self._is_tty and self._last_tty_line_len > 0:
            print("", file=self._stream, flush=True)
            self._last_tty_line_len = 0
