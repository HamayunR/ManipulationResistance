"""Figure 5 -- routing temperature x influence-weight heatmap.

Usage
-----
    python analysis/figure_routing_heatmap.py \\
        analysis_artifacts/my_analysis --metric accuracy

``--metric`` is required. Picking one automatically would let the heatmap
answer whichever question the available data happened to suit.

Supported metrics
-----------------
``accuracy``                   mean accuracy over the cell's runs.
``robustness_gap``             clean accuracy - attacked accuracy, from matched
                               clean and attacked conditions in the same cell.
``adversary_routing_exposure`` the adversary's mean out-degree share, from the
                               normalised routing table.
``empirical_ic_regret``        the best fixed-report utility gain, read from
                               Figure 2's table when it has been generated.

Rules that apply to every metric:

* a cell is a parameter pair that was actually run. Missing cells are reported
  and left blank -- never interpolated, because an interpolated cell is an
  experiment nobody performed;
* at least a 2 x 2 grid, or there is no surface to look at;
* datasets, models, mechanisms, verification modes and routing modes are kept
  apart.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.collect_runs import load_tables  # noqa: E402
from analysis.common import RunLoadError  # noqa: E402
from analysis.figure_utils import (  # noqa: E402
    FigureError,
    add_common_arguments,
    annotate_mock,
    bool_series,
    command_line,
    figure_paths,
    mock_status_of,
    print_summary,
    require_separable_mock_status,
    save_figure,
    title_for,
    write_metadata,
)
from analysis.statistics import DEFAULT_ANALYSIS_SEED, DEFAULT_BOOTSTRAP_REPETITIONS  # noqa: E402

METRICS: Tuple[str, ...] = (
    "accuracy",
    "robustness_gap",
    "adversary_routing_exposure",
    "empirical_ic_regret",
)

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "robustness_gap": "Robustness gap (clean - attacked accuracy)",
    "adversary_routing_exposure": "Adversary routing-exposure share",
    "empirical_ic_regret": "Empirical best fixed-report utility gain",
}

GROUP_COLS: Tuple[str, ...] = (
    "dataset",
    "model",
    "mechanism",
    "verification_mode",
    "routing_mode",
)
CELL_COLS: Tuple[str, ...] = ("routing_temperature", "alpha_influence")

FIGURE_NAME = "Routing temperature x influence-weight heatmap"


def slug_for(metric: str) -> str:
    return f"figure5_heatmap_{metric}"


def _label_groups(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in GROUP_COLS:
        if column not in out.columns:
            out[column] = "n/a"
        out[column] = out[column].astype(object).where(out[column].notna(), "n/a").astype(str)
    return out


def _condition_parameters(runs: pd.DataFrame) -> pd.DataFrame:
    """Per run x condition parameter coordinates, from the resolved config."""
    for column in CELL_COLS:
        if column not in runs.columns or runs[column].isna().all():
            raise FigureError(
                f"no run records {column!r}; the heatmap axes come from the "
                "resolved condition metadata, not from a guess"
            )
    keep = ["run_id", "condition", *GROUP_COLS, *CELL_COLS, "attacker_count"]
    frame = _label_groups(runs)
    return frame[[c for c in keep if c in frame.columns]].dropna(subset=list(CELL_COLS))


def _join_parameters(rows: pd.DataFrame, parameters: pd.DataFrame) -> pd.DataFrame:
    return rows.merge(
        parameters[["run_id", "condition", *CELL_COLS]], on=["run_id", "condition"], how="inner"
    )


def cell_values(
    metric: str,
    runs: pd.DataFrame,
    results: pd.DataFrame,
    routing: pd.DataFrame,
    analysis_dir: Path,
) -> pd.DataFrame:
    """One value per group x parameter cell, for the chosen metric."""
    parameters = _condition_parameters(runs)
    labelled_results = _label_groups(results)

    if metric == "accuracy":
        scored = labelled_results[labelled_results["correct"].notna()]
        joined = _join_parameters(scored, parameters)
        if joined.empty:
            raise FigureError("no accuracy row could be joined to a parameter cell")
        joined["value"] = joined["correct"].astype(bool).astype(float)
        return (
            joined.groupby([*GROUP_COLS, *CELL_COLS], dropna=False, sort=True)
            .agg(value=("value", "mean"), n_examples=("example_id", "nunique"),
                 n_rows=("value", "size"))
            .reset_index()
        )

    if metric == "robustness_gap":
        scored = labelled_results[labelled_results["correct"].notna()].copy()
        scored["value"] = scored["correct"].astype(bool).astype(float)
        joined = _join_parameters(scored, parameters)
        clean = joined[joined["attacker_count"].fillna(0) == 0]
        attacked = joined[joined["attacker_count"].fillna(0) > 0]
        if clean.empty or attacked.empty:
            raise FigureError(
                "robustness_gap needs matched clean and attacked conditions in "
                f"the same parameter cells; found {len(clean)} clean and "
                f"{len(attacked)} attacked row(s)"
            )
        keys = [*GROUP_COLS, *CELL_COLS]
        clean_mean = clean.groupby(keys, dropna=False, sort=True).agg(
            clean_accuracy=("value", "mean"), n_examples=("example_id", "nunique")
        )
        attacked_mean = attacked.groupby(keys, dropna=False, sort=True).agg(
            attacked_accuracy=("value", "mean"), n_rows=("value", "size")
        )
        merged = clean_mean.join(attacked_mean, how="inner").reset_index()
        if merged.empty:
            raise FigureError(
                "no parameter cell has both a clean and an attacked condition; "
                "the gap cannot be formed from unmatched cells"
            )
        merged["value"] = merged["clean_accuracy"] - merged["attacked_accuracy"]
        return merged

    if metric == "adversary_routing_exposure":
        if routing.empty:
            raise FigureError(
                "adversary_routing_exposure needs routing rows; no input method "
                "logs routing (a capability limitation, not a validation failure)"
            )
        labelled_routing = _label_groups(routing)
        attackers = labelled_routing[bool_series(labelled_routing, "is_attacker").fillna(False)]
        attackers = attackers[attackers["out_degree_share"].notna()]
        if attackers.empty:
            raise FigureError("no attacker routing rows with an out-degree share")
        joined = _join_parameters(attackers, parameters)
        if joined.empty:
            raise FigureError("no routing row could be joined to a parameter cell")
        return (
            joined.groupby([*GROUP_COLS, *CELL_COLS], dropna=False, sort=True)
            .agg(
                value=("out_degree_share", "mean"),
                n_examples=("example_id", "nunique"),
                n_rows=("out_degree_share", "size"),
            )
            .reset_index()
        )

    if metric == "empirical_ic_regret":
        table_path = analysis_dir / "tables" / "figure2_empirical_ic_regret.csv"
        if not table_path.is_file():
            raise FigureError(
                f"{table_path} not found. Generate Figure 2 first (it needs "
                "explicit --w-rho and --w-adoption weights), then re-run this "
                "heatmap; the regret is not recomputed here with invented weights."
            )
        regret = _label_groups(pd.read_csv(table_path))
        best = regret[bool_series(regret, "is_best_report").fillna(False)]
        if best.empty:
            raise FigureError("figure2 table carries no best-report row")
        # The Figure 2 table is per group, not per run, so the parameter
        # coordinates come from the runs of the same group.
        coordinates = (
            parameters.groupby(list(GROUP_COLS), dropna=False, sort=True)[list(CELL_COLS)]
            .agg(lambda s: s.iloc[0])
            .reset_index()
        )
        merged = best.merge(coordinates, on=list(GROUP_COLS), how="inner")
        if merged.empty:
            raise FigureError(
                "the Figure 2 groups do not line up with any parameter cell"
            )
        merged["value"] = merged["utility_gain"]
        merged["n_rows"] = merged["n_pairs"]
        return merged

    raise FigureError(f"unknown metric {metric!r}; expected one of {list(METRICS)}")


def check_grid(table: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Keep only groups with a complete grid of at least 2 x 2."""
    kept: List[pd.DataFrame] = []
    rejected: List[Dict[str, Any]] = []
    for key, subset in table.groupby(list(GROUP_COLS), dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        label = dict(zip(GROUP_COLS, [str(k) for k in key_tuple]))
        temperatures = sorted({float(t) for t in subset["routing_temperature"].unique()})
        alphas = sorted({float(a) for a in subset["alpha_influence"].unique()})
        present = {
            (float(r["routing_temperature"]), float(r["alpha_influence"]))
            for _, r in subset.iterrows()
        }
        holes = sorted({(t, a) for t in temperatures for a in alphas} - present)
        if len(temperatures) < 2 or len(alphas) < 2:
            rejected.append(
                {
                    **label,
                    "grid": f"{len(temperatures)} x {len(alphas)}",
                    "reason": "grid is smaller than 2 x 2",
                }
            )
        elif holes:
            rejected.append(
                {
                    **label,
                    "grid": f"{len(temperatures)} x {len(alphas)}",
                    "missing_cells": [list(c) for c in holes],
                    "reason": "incomplete grid; missing cells are never interpolated",
                }
            )
        else:
            kept.append(subset)
    if not kept:
        raise FigureError(
            "no group forms a complete grid of at least 2 x 2. "
            + "; ".join(f"{r['reason']} ({r['grid']})" for r in rejected)
        )
    return pd.concat(kept, ignore_index=True), rejected


def plot(table: pd.DataFrame, *, metric: str, mock_status: str):
    groups = sorted({tuple(r) for r in table[list(GROUP_COLS)].itertuples(index=False)}, key=str)
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 4.4), squeeze=False)
    for ax, group in zip(axes[0], groups):
        subset = table
        for column, value in zip(GROUP_COLS, group):
            subset = subset[subset[column] == value]
        grid = subset.pivot_table(
            index="alpha_influence", columns="routing_temperature", values="value", aggfunc="mean"
        ).sort_index(ascending=False)
        image = ax.imshow(grid.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(grid.columns)), [f"{c:g}" for c in grid.columns])
        ax.set_yticks(range(len(grid.index)), [f"{i:g}" for i in grid.index])
        ax.set_xlabel("routing_temperature")
        ax.set_ylabel("alpha_influence")
        for row_index in range(len(grid.index)):
            for column_index in range(len(grid.columns)):
                value = grid.to_numpy(dtype=float)[row_index, column_index]
                ax.text(
                    column_index,
                    row_index,
                    "-" if np.isnan(value) else f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )
        ax.set_title(
            ", ".join(
                f"{c}={v}" for c, v in zip(GROUP_COLS, group)
                if c in {"dataset", "model", "routing_mode"}
            ),
            fontsize=9,
        )
        fig.colorbar(image, ax=ax, label=METRIC_LABELS[metric])

    fig.suptitle(title_for(f"{FIGURE_NAME}: {METRIC_LABELS[metric]}", mock_status), fontsize=12)
    fig.tight_layout()
    annotate_mock(fig, mock_status)
    return fig


def build_figure(
    analysis_dir: str | Path,
    *,
    metric: str,
    allow_mock: bool = False,
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    analysis_seed: int = DEFAULT_ANALYSIS_SEED,
    command: Optional[str] = None,
) -> Dict[str, Any]:
    if metric not in METRICS:
        raise FigureError(f"unknown metric {metric!r}; expected one of {list(METRICS)}")
    analysis_dir = Path(analysis_dir)
    tables = load_tables(analysis_dir)
    runs, results, routing = tables["runs"], tables["results"], tables["routing"]

    mock_status = mock_status_of(runs)
    require_separable_mock_status(mock_status, allow_mock=allow_mock, what="Figure 5")
    diagnostic_only = mock_status != "real"

    values = cell_values(metric, runs, results, routing, analysis_dir)
    table, rejected = check_grid(values)
    table = table.copy()
    table["metric"] = metric
    table["mock"] = mock_status == "mock"
    table["diagnostic_only"] = diagnostic_only

    columns = [
        *GROUP_COLS,
        *CELL_COLS,
        "metric",
        "value",
        *[c for c in ("clean_accuracy", "attacked_accuracy", "n_examples", "n_rows") if c in table],
        "mock",
        "diagnostic_only",
    ]
    table = table[columns]

    paths = figure_paths(analysis_dir, slug_for(metric))
    table.to_csv(paths.table, index=False)
    save_figure(plot(table, metric=metric, mock_status=mock_status), paths)

    metadata = write_metadata(
        paths,
        {
            "figure": 5,
            "name": f"{FIGURE_NAME}: {METRIC_LABELS[metric]}",
            "metric": metric,
            "analysis_dir": str(analysis_dir),
            "cell_columns": list(CELL_COLS),
            "grouping_columns": list(GROUP_COLS),
            "grid_policy": "cells are never interpolated; missing cells are reported and left blank",
            "analysis_seed": analysis_seed,
            "bootstrap_repetitions": repetitions,
            "mock_status": mock_status,
            "diagnostic_only": diagnostic_only,
            "rejected_groups": rejected,
            "n_cells": int(len(table)),
            "command": command or command_line("figure_routing_heatmap.py"),
        },
    )
    return {"table": table, "paths": paths, "metadata": metadata, "rejected": rejected}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--metric",
        required=True,
        choices=list(METRICS),
        help="Which quantity to colour the grid by. Required: no metric is chosen automatically.",
    )
    args = parser.parse_args(argv)
    try:
        result = build_figure(
            args.analysis_dir,
            metric=args.metric,
            allow_mock=args.allow_mock,
            repetitions=args.bootstrap_repetitions,
            analysis_seed=args.analysis_seed,
            command=command_line("figure_routing_heatmap.py", argv),
        )
    except (FigureError, RunLoadError) as exc:
        print(f"Figure 5 ({args.metric}) not produced: {exc}", file=sys.stderr)
        return 1
    print_summary(
        f"{FIGURE_NAME} [{args.metric}]",
        result["table"],
        result["paths"],
        diagnostic_only=result["metadata"]["diagnostic_only"],
    )
    for group in result["rejected"]:
        print(f"  rejected: {group}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
