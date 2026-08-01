"""Shared plumbing for the figure scripts: outputs, provenance, mock marking.

Every headline figure writes the same four things -- a CSV of exactly the
plotted values, a PNG, a PDF and a metadata JSON -- and every one of them has
to make mock data visibly mock. Doing that once here keeps the figure modules
about their own quantity, and keeps the five of them consistent.

Nothing benchmark-, model- or method-specific belongs in this file. Labels are
built from the data.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: figures are files, never windows
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.common import analysis_git_commit  # noqa: E402

#: Appended to the title of anything drawn from canned model outputs.
MOCK_TITLE_SUFFIX = "  [MOCK DATA - pipeline diagnostic, not a result]"

#: Marker cycle for grouped line plots. Deliberately colour-independent as
#: well, so a printed or colour-blind-viewed figure still separates the curves.
MARKERS: Tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")
LINESTYLES: Tuple[str, ...] = ("-", "--", "-.", ":")


class FigureError(RuntimeError):
    """Raised when a figure cannot be drawn from the available data."""


@dataclass
class FigurePaths:
    """Where one figure's four artefacts go."""

    table: Path
    png: Path
    pdf: Path
    metadata: Path

    def as_dict(self) -> Dict[str, str]:
        return {
            "table": str(self.table),
            "png": str(self.png),
            "pdf": str(self.pdf),
            "metadata": str(self.metadata),
        }


def figure_paths(analysis_dir: str | Path, slug: str) -> FigurePaths:
    """Standard output layout, alongside (never overwriting) the input tables."""
    analysis_dir = Path(analysis_dir)
    tables = analysis_dir / "tables"
    figures = analysis_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return FigurePaths(
        table=tables / f"{slug}.csv",
        png=figures / f"{slug}.png",
        pdf=figures / f"{slug}.pdf",
        metadata=analysis_dir / f"{slug}.meta.json",
    )


def mock_status_of(frame, column: str = "mock") -> str:
    """``"mock"``, ``"real"``, ``"mixed"`` or ``"unknown"`` for a table."""
    if column not in getattr(frame, "columns", []) or len(frame) == 0:
        return "unknown"
    flags = {bool(v) for v in frame[column].dropna().unique()}
    if flags == {True}:
        return "mock"
    if flags == {False}:
        return "real"
    if len(flags) > 1:
        return "mixed"
    return "unknown"


def require_separable_mock_status(status: str, *, allow_mock: bool, what: str) -> None:
    """Refuse to draw a figure over an indefensible mixture of inputs."""
    if status == "mixed":
        raise FigureError(
            f"{what}: inputs mix mock and real rows. A curve averaging canned "
            "outputs with model outputs describes neither; separate them into "
            "two analysis directories."
        )
    if status == "mock" and not allow_mock:
        raise FigureError(
            f"{what}: every input row is mock (canned model outputs). Pass "
            "--allow-mock to produce the figure as a pipeline diagnostic."
        )


def annotate_mock(fig, status: str) -> None:
    """Stamp a mock figure so a screenshot of it cannot be mistaken for data."""
    if status != "mock":
        return
    fig.text(
        0.5,
        0.5,
        "MOCK",
        fontsize=90,
        color="0.85",
        alpha=0.45,
        ha="center",
        va="center",
        rotation=30,
        zorder=0,
    )


def title_for(base: str, status: str) -> str:
    return base + (MOCK_TITLE_SUFFIX if status == "mock" else "")


def style_axes(ax) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def series_label(key: Mapping[str, Any], *, skip_constant: Mapping[str, Any] = {}) -> str:
    """Human label for one curve, dropping keys that are the same everywhere."""
    parts = [f"{k}={v}" for k, v in key.items() if k not in skip_constant]
    return ", ".join(parts) if parts else "all"


def constant_columns(frame, columns: Sequence[str]) -> Dict[str, Any]:
    """Columns with a single value across the frame; they belong in the title."""
    out: Dict[str, Any] = {}
    for column in columns:
        if column in frame.columns:
            values = frame[column].dropna().unique()
            if len(values) == 1:
                out[column] = values[0]
    return out


def save_figure(fig, paths: FigurePaths, *, dpi: int = 200) -> None:
    fig.savefig(paths.png, dpi=dpi, bbox_inches="tight")
    fig.savefig(paths.pdf, bbox_inches="tight")
    plt.close(fig)


def write_metadata(paths: FigurePaths, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Write the figure's metadata sidecar, with provenance filled in."""
    enriched = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # The analysis code's commit. The runner does not record the commit the
        # experiments ran at, so this is not the generating commit.
        "analysis_git_commit": analysis_git_commit(),
        "experiment_git_commit": None,
        "experiment_git_commit_note": (
            "not recorded by the experiment runner; the commit above is the "
            "analysis code's"
        ),
        **dict(payload),
        "outputs": paths.as_dict(),
    }
    paths.metadata.write_text(json.dumps(enriched, indent=2, default=str), encoding="utf-8")
    return enriched


def command_line(script: str, argv: Optional[Sequence[str]] = None) -> str:
    return " ".join([f"python analysis/{script}", *(argv if argv is not None else sys.argv[1:])])


def add_common_arguments(parser) -> None:
    """CLI flags every figure script shares, so they cannot drift apart."""
    from analysis.statistics import DEFAULT_ANALYSIS_SEED, DEFAULT_BOOTSTRAP_REPETITIONS

    parser.add_argument("analysis_dir", help="Directory written by analysis/collect_runs.py")
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Draw the figure from mock runs, marked diagnostic_only.",
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPETITIONS,
        help="Cluster-bootstrap replications (default: %(default)s).",
    )
    parser.add_argument(
        "--analysis-seed",
        type=int,
        default=DEFAULT_ANALYSIS_SEED,
        help="Seed for the bootstrap, recorded in the outputs (default: %(default)s).",
    )


def print_summary(name: str, table, paths: FigurePaths, *, diagnostic_only: bool) -> None:
    print(f"\n{name}: {len(table)} plotted point(s)")
    if diagnostic_only:
        print("  diagnostic_only=True -- mock inputs; this shows the pipeline works, nothing else")
    for path in (paths.table, paths.png, paths.pdf, paths.metadata):
        print(f"  wrote {path}")


def bool_series(frame, column: str):
    """Read a possibly-object-dtype boolean column without inventing values."""
    import pandas as pd

    if column not in frame.columns:
        return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="boolean")
    return frame[column].map(
        lambda v: pd.NA if v is None or (isinstance(v, float) and pd.isna(v)) else bool(v)
    ).astype("boolean")


def stable_group_keys(frame, columns: Sequence[str]) -> List[Tuple[Any, ...]]:
    return sorted({tuple(row) for row in frame[list(columns)].itertuples(index=False)}, key=str)
