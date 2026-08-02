"""Logging setup and timestamped run-directory management.

A "run" is one invocation of the experiment driver. Every run gets its own
directory named ``<timestamp>_<tag>``. Scripts usually place those run
directories under an experiment root such as ``outputs/exp_<timestamp>/``.
Each run directory contains:

* ``run.log``           - human-readable text log (rotating).
* ``trace.jsonl``       - one JSON object per node-level event.
* ``config.yaml``       - a copy of the resolved configuration.
* ``results.jsonl``     - one JSON object per (example, condition, seed,
                          perm_seed) tuple.
* ``transcripts.jsonl`` - one JSON object per (example, condition, seed,
                          perm_seed) tuple, carrying the ordered debate
                          transcript with each agent's full message text per
                          turn. Convenient when you want to re-read a
                          specific run's dialogue without parsing
                          ``trace.jsonl``.
* ``summary.json``      - aggregated metrics written at the end.

The helpers here only set up the directory and the loggers; the JSONL writer
itself lives in :mod:`utils.tracing`.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

_DEFAULT_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
UK_TZ = ZoneInfo("Europe/London")


@dataclass(frozen=True)
class RunPaths:
    """Filesystem layout for a single experiment run.

    Attributes
    ----------
    run_dir:
        Top-level directory for the run.
    log_file:
        Plain-text rotating log file.
    trace_file:
        JSONL stream of node-level events.
    results_file:
        JSONL stream of per-example results (one row per dataset example).
    transcript_file:
        JSONL stream of per-example debate transcripts (one row each).
    routing_file:
        JSONL stream of routing decisions, one row per round. A flat,
        analysis-ready projection of the ``topology`` trace events joined with
        the state that produced them, so routing figures do not require
        parsing (and filtering) the full event trace.
    config_file:
        Resolved YAML config (copy of inputs).
    summary_file:
        Aggregated metrics, written at end of run.
    """

    run_dir: Path
    log_file: Path
    trace_file: Path
    results_file: Path
    transcript_file: Path
    routing_file: Path
    config_file: Path
    summary_file: Path


def _uk_timestamp() -> str:
    """Return a filesystem-safe Europe/London timestamp like ``20260503194512``."""
    return datetime.now(UK_TZ).strftime("%Y%m%d%H%M%S")


def _safe_slug(value: str) -> str:
    """Return a filesystem-safe slug using the same convention as run tags."""
    return "".join(c if c.isalnum() or c in "_-" else "-" for c in value)


class _UKFormatter(logging.Formatter):
    """Formatter whose ``asctime`` is rendered in Europe/London."""

    def formatTime(self, record, datefmt=None):  # noqa: N802 - logging API name
        dt = datetime.fromtimestamp(record.created, UK_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat(timespec="milliseconds")


def _uk_formatter() -> logging.Formatter:
    """Create a formatter whose ``asctime`` is rendered in UK local time."""
    return _UKFormatter(_DEFAULT_FMT, datefmt="%Y-%m-%d %H:%M:%S%z")


def setup_run_logging(
    output_root: str | os.PathLike,
    *,
    tag: Optional[str] = None,
    level: int | str = "INFO",
    console: bool = True,
) -> RunPaths:
    """Create a timestamped run directory and configure root logging.

    Parameters
    ----------
    output_root:
        Parent directory under which the run directory is created.
    tag:
        Optional short slug appended to the timestamp (e.g.
        ``"gsm8k_pear_star"``). Sanitised to ``[A-Za-z0-9_-]``.
    level:
        Log level for the root logger. Strings like ``"DEBUG"`` or numeric
        levels are both accepted.
    console:
        Mirror logs to ``stderr`` if ``True``.

    Returns
    -------
    RunPaths
        File paths for the new run directory.
    """
    run_timestamp = _uk_timestamp()
    output_root = Path(output_root)
    if output_root.name == "outputs" and os.environ.get("PEAR_DISABLE_EXP_SUBDIR") != "1":
        exp_timestamp = os.environ.get("PEAR_EXP_TIMESTAMP") or run_timestamp
        output_root = output_root / f"exp_{_safe_slug(exp_timestamp)}"
    output_root.mkdir(parents=True, exist_ok=True)

    safe_tag = ""
    if tag:
        safe_tag = "_" + _safe_slug(tag)
    # ``exist_ok=False`` on purpose: a run must never write into a directory
    # that already holds another run's results. The timestamp has one-second
    # resolution, though, and a benchmark sweep launches runs back to back, so
    # a collision is disambiguated with a suffix rather than crashing the run.
    run_dir = output_root / f"{run_timestamp}{safe_tag}"
    attempt = 1
    while True:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            attempt += 1
            if attempt > 1000:  # pragma: no cover - pathological
                raise
            run_dir = output_root / f"{run_timestamp}-{attempt}{safe_tag}"

    paths = RunPaths(
        run_dir=run_dir,
        log_file=run_dir / "run.log",
        trace_file=run_dir / "trace.jsonl",
        results_file=run_dir / "results.jsonl",
        transcript_file=run_dir / "transcripts.jsonl",
        routing_file=run_dir / "routing.jsonl",
        config_file=run_dir / "config.yaml",
        summary_file=run_dir / "summary.json",
    )

    # Configure the root logger. We deliberately replace existing handlers so
    # that successive runs in a single process do not double-log.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level if isinstance(level, int) else level.upper())

    formatter = _uk_formatter()

    file_handler = RotatingFileHandler(
        paths.log_file, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler(stream=sys.stderr)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    logging.getLogger("pear").info("Run directory: %s", run_dir)
    return paths


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper for ``logging.getLogger`` with a stable namespace.

    All package-internal loggers use the ``<module>`` namespace so that
    the user can adjust verbosity for a single module if needed.
    """
    if not name.startswith("pear"):
        name = f"{name}"
    return logging.getLogger(name)
