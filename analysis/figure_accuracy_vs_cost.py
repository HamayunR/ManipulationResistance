"""Figure 4 -- clean accuracy against token cost.

Only clean conditions (``attacker_count == 0``) appear: this is what each
method buys and costs when nobody is attacking it. A clean run *with* verified
confidence stays eligible and is labelled as such -- pricing the defence on
clean debates is the point of running it there.

Two rules keep the x-axis honest:

* a method that does not report token usage is **excluded**, by name, with a
  readiness message. It is not placed at zero cost, which is where a
  ``fillna(0)`` would put it -- the free end of the axis;
* accuracy is never pooled across datasets or models; each pair gets a panel.

Methods that log no routing are fully eligible; this figure never touches
routing data.

Usage
-----
    python analysis/figure_accuracy_vs_cost.py \\
        analysis_artifacts/my_analysis --allow-mock
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.collect_runs import load_tables  # noqa: E402
from analysis.common import RunLoadError  # noqa: E402
from analysis.figure_utils import (  # noqa: E402
    FigureError,
    MARKERS,
    add_common_arguments,
    annotate_mock,
    command_line,
    figure_paths,
    mock_status_of,
    print_summary,
    require_separable_mock_status,
    save_figure,
    style_axes,
    title_for,
    write_metadata,
)
from analysis.statistics import (  # noqa: E402
    DEFAULT_ANALYSIS_SEED,
    DEFAULT_BOOTSTRAP_REPETITIONS,
    cluster_bootstrap_mean,
)

SLUG = "figure4_accuracy_vs_cost"
FIGURE_NAME = "Clean accuracy vs token cost"

GROUP_COLS: Tuple[str, ...] = (
    "dataset",
    "model",
    "method",
    "mechanism",
    "verification_mode",
)

OUTPUT_COLUMNS: Tuple[str, ...] = (
    *GROUP_COLS,
    "mean_accuracy",
    "ci_lower",
    "ci_upper",
    "mean_prompt_tokens",
    "mean_completion_tokens",
    "mean_total_tokens",
    "n_examples",
    "n_replications",
    "n_runs",
    "analysis_seed",
    "bootstrap_repetitions",
    "mock",
    "diagnostic_only",
)


def prepare(results: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Clean, priced, accuracy-bearing rows; plus who was excluded and why."""
    for column in ("attacker_count", "correct", "total_tokens", "method"):
        if column not in results.columns:
            raise FigureError(f"results.csv is missing required column {column!r}")

    clean = results[results["attacker_count"].fillna(0) == 0]
    if clean.empty:
        raise FigureError(
            "no clean (attacker_count == 0) condition in the inputs; this figure "
            "prices the methods when nobody is attacking them"
        )
    scored = clean[clean["correct"].notna()]
    priced = scored[scored["total_tokens"].notna()].copy()

    excluded: List[Dict[str, Any]] = []
    for method in sorted({str(m) for m in scored["method"].dropna().unique()}):
        if method not in {str(m) for m in priced["method"].dropna().unique()}:
            excluded.append(
                {
                    "method": method,
                    "reason": (
                        "no token usage recorded; excluded from the cost axis "
                        "rather than plotted at zero cost"
                    ),
                }
            )
    if priced.empty:
        raise FigureError(
            "no clean condition reports token usage. Missing usage is a "
            "capability gap, never a cost of zero. Methods affected: "
            + ", ".join(e["method"] for e in excluded)
        )

    for column in GROUP_COLS:
        if column not in priced.columns:
            priced[column] = "n/a"
        priced[column] = (
            priced[column].astype(object).where(priced[column].notna(), "n/a").astype(str)
        )
    priced["accuracy"] = priced["correct"].astype(bool).astype(float)
    return priced, excluded


def aggregate(
    frame: pd.DataFrame,
    *,
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    analysis_seed: int = DEFAULT_ANALYSIS_SEED,
) -> pd.DataFrame:
    stats = cluster_bootstrap_mean(
        frame,
        value_col="accuracy",
        cluster_col="example_id",
        group_cols=list(GROUP_COLS),
        repetitions=repetitions,
        analysis_seed=analysis_seed,
    ).rename(columns={"mean": "mean_accuracy", "n_raw_rows": "n_replications"})

    costs = (
        frame.groupby(list(GROUP_COLS), dropna=False, sort=True)
        .agg(
            mean_prompt_tokens=("prompt_tokens", "mean"),
            mean_completion_tokens=("completion_tokens", "mean"),
            mean_total_tokens=("total_tokens", "mean"),
            n_runs=("run_id", "nunique"),
        )
        .reset_index()
    )
    return stats.merge(costs, on=list(GROUP_COLS), how="left")


def plot(table: pd.DataFrame, *, mock_status: str):
    panel_cols = ["dataset", "model"]
    panels = sorted({tuple(r) for r in table[panel_cols].itertuples(index=False)}, key=str)
    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.6 * len(panels), 4.4), squeeze=False, sharey=True
    )
    for ax, panel in zip(axes[0], panels):
        subset = table
        for column, value in zip(panel_cols, panel):
            subset = subset[subset[column] == value]
        for index, (_, row) in enumerate(subset.iterrows()):
            label = f"{row['method']}/{row['mechanism']}"
            if str(row["verification_mode"]) not in {"none", "n/a"}:
                label += f" +{row['verification_mode']}"
            ax.errorbar(
                [row["mean_total_tokens"]],
                [row["mean_accuracy"]],
                yerr=[
                    [max(0.0, row["mean_accuracy"] - row["ci_lower"])
                     if pd.notna(row["ci_lower"]) else 0.0],
                    [max(0.0, row["ci_upper"] - row["mean_accuracy"])
                     if pd.notna(row["ci_upper"]) else 0.0],
                ],
                marker=MARKERS[index % len(MARKERS)],
                markersize=9,
                capsize=3,
                linestyle="none",
                label=label,
            )
        ax.set_xlabel("Mean total tokens per example")
        ax.set_title(", ".join(f"{c}={v}" for c, v in zip(panel_cols, panel)), fontsize=10)
        style_axes(ax)
        ax.legend(fontsize=8, frameon=False)

    axes[0][0].set_ylabel("Clean accuracy (95% cluster bootstrap CI)")
    fig.suptitle(title_for(FIGURE_NAME, mock_status), fontsize=12)
    fig.tight_layout()
    annotate_mock(fig, mock_status)
    return fig


def build_figure(
    analysis_dir: str | Path,
    *,
    allow_mock: bool = False,
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    analysis_seed: int = DEFAULT_ANALYSIS_SEED,
    command: Optional[str] = None,
) -> Dict[str, Any]:
    analysis_dir = Path(analysis_dir)
    results = load_tables(analysis_dir)["results"]

    mock_status = mock_status_of(results)
    require_separable_mock_status(mock_status, allow_mock=allow_mock, what="Figure 4")
    diagnostic_only = mock_status != "real"

    frame, excluded = prepare(results)
    table = aggregate(frame, repetitions=repetitions, analysis_seed=analysis_seed)
    table["mock"] = mock_status == "mock"
    table["diagnostic_only"] = diagnostic_only
    table = table[list(OUTPUT_COLUMNS)]

    paths = figure_paths(analysis_dir, SLUG)
    table.to_csv(paths.table, index=False)
    save_figure(plot(table, mock_status=mock_status), paths)

    metadata = write_metadata(
        paths,
        {
            "figure": 4,
            "name": FIGURE_NAME,
            "analysis_dir": str(analysis_dir),
            "filters": {"attacker_count": 0, "total_tokens": "not null"},
            "grouping_columns": list(GROUP_COLS),
            "cluster_column": "example_id",
            "analysis_seed": analysis_seed,
            "bootstrap_repetitions": repetitions,
            "mock_status": mock_status,
            "diagnostic_only": diagnostic_only,
            "methods_plotted": sorted({str(m) for m in table["method"].unique()}),
            "methods_excluded": excluded,
            "verification_modes": sorted({str(v) for v in table["verification_mode"].unique()}),
            "note": (
                "a clean verified-confidence condition is included on purpose: it "
                "measures the defence's clean-performance cost"
            ),
            "command": command or command_line("figure_accuracy_vs_cost.py"),
        },
    )
    return {"table": table, "paths": paths, "metadata": metadata, "excluded": excluded}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    try:
        result = build_figure(
            args.analysis_dir,
            allow_mock=args.allow_mock,
            repetitions=args.bootstrap_repetitions,
            analysis_seed=args.analysis_seed,
            command=command_line("figure_accuracy_vs_cost.py", argv),
        )
    except (FigureError, RunLoadError) as exc:
        print(f"Figure 4 not produced: {exc}", file=sys.stderr)
        return 1
    print_summary(
        FIGURE_NAME, result["table"], result["paths"], diagnostic_only=result["metadata"]["diagnostic_only"]
    )
    for method in result["excluded"]:
        print(f"  excluded: {method['method']} -- {method['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
