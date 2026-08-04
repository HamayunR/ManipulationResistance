"""Figure 1 -- accuracy against the fraction of adversarial agents.

Each curve is one mechanism under one attack type, on one dataset and model.
The x-axis is the fraction of the debate's agents under attacker control; the
clean run is that curve's ``0.0`` point, not a separate experiment.

What this figure does not do:

* it does not pool datasets or models into one unexplained accuracy number;
* it does not treat decoding or permutation seeds as extra questions -- the
  bootstrap clusters on ``example_id``;
* it does not invent multi-agent attack semantics. The fraction comes from the
  attacker set the run actually recorded, so a sweep that only ever ran one
  attacker produces one non-zero point and says so.

Methods that log no routing are fully eligible here: accuracy and the attacker
set are all this figure needs.

Usage
-----
    python analysis/figure_accuracy_vs_adversaries.py \\
        analysis_artifacts/my_analysis
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
    LINESTYLES,
    MARKERS,
    add_common_arguments,
    command_line,
    constant_columns,
    figure_paths,
    print_summary,
    save_figure,
    series_label,
    style_axes,
    title_for,
    write_metadata,
)
from analysis.statistics import (  # noqa: E402
    DEFAULT_ANALYSIS_SEED,
    DEFAULT_BOOTSTRAP_REPETITIONS,
    broadcast_clean_arm,
    cluster_bootstrap_mean,
)

SLUG = "figure1_accuracy_vs_adversaries"
FIGURE_NAME = "Accuracy vs adversarial fraction"

BASE_GROUP_COLS: Tuple[str, ...] = (
    "dataset",
    "model",
    "method",
    "mechanism",
    "verification_mode",
    "routing_mode",
)
CURVE_COLS: Tuple[str, ...] = (*BASE_GROUP_COLS, "attack_type")

OUTPUT_COLUMNS: Tuple[str, ...] = (
    *CURVE_COLS,
    "adversarial_fraction",
    "mean_accuracy",
    "ci_lower",
    "ci_upper",
    "n_examples",
    "n_replications",
    "n_seeds",
    "n_perm_seeds",
    "analysis_seed",
    "bootstrap_repetitions",
)


def prepare(results: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for this figure, with the clean arm attached to each curve."""
    for column in ("correct", "adversarial_fraction", "example_id"):
        if column not in results.columns:
            raise FigureError(f"results.csv is missing required column {column!r}")

    frame = results[results["correct"].notna() & results["adversarial_fraction"].notna()].copy()
    if frame.empty:
        raise FigureError(
            "no result row carries both accuracy and an adversarial fraction. A "
            "method without an attacker set cannot be placed on this x-axis; it "
            "remains eligible for the accuracy/cost figure."
        )
    for column in CURVE_COLS:
        if column not in frame.columns:
            frame[column] = "n/a"
        frame[column] = frame[column].astype(object).where(frame[column].notna(), "n/a").astype(str)
    frame["accuracy"] = frame["correct"].astype(bool).astype(float)
    return broadcast_clean_arm(frame, base_group_cols=list(BASE_GROUP_COLS))


def aggregate(
    frame: pd.DataFrame,
    *,
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    analysis_seed: int = DEFAULT_ANALYSIS_SEED,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """One point per curve x adversarial fraction; drop curves that are a point."""
    stats = cluster_bootstrap_mean(
        frame,
        value_col="accuracy",
        cluster_col="example_id",
        group_cols=[*CURVE_COLS, "adversarial_fraction"],
        repetitions=repetitions,
        analysis_seed=analysis_seed,
    ).rename(columns={"mean": "mean_accuracy", "n_raw_rows": "n_replications"})

    counts = (
        frame.groupby([*CURVE_COLS, "adversarial_fraction"], dropna=False, sort=True)
        .agg(n_seeds=("seed", "nunique"), n_perm_seeds=("perm_seed", "nunique"))
        .reset_index()
    )
    table = stats.merge(counts, on=[*CURVE_COLS, "adversarial_fraction"], how="left")

    plotted: List[pd.DataFrame] = []
    excluded: List[Dict[str, Any]] = []
    for key, subset in table.groupby(list(CURVE_COLS), dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        label = dict(zip(CURVE_COLS, [str(k) for k in key_tuple]))
        fractions = sorted(float(f) for f in subset["adversarial_fraction"].unique())
        if len(fractions) < 2:
            excluded.append(
                {**label, "adversarial_fractions": fractions, "reason": "only one adversarial fraction"}
            )
        elif 0.0 not in fractions:
            excluded.append(
                {
                    **label,
                    "adversarial_fractions": fractions,
                    "reason": "no clean adversarial_fraction=0.0 arm to anchor the curve",
                }
            )
        else:
            plotted.append(subset)

    if not plotted:
        raise FigureError(
            "no curve has a clean 0.0 arm and at least one attacked fraction. "
            "Excluded: " + "; ".join(f"{g['reason']} {g['adversarial_fractions']}" for g in excluded)
        )
    result = pd.concat(plotted, ignore_index=True).sort_values(
        [*CURVE_COLS, "adversarial_fraction"]
    )
    return result.reset_index(drop=True), excluded


def plot(table: pd.DataFrame):
    panel_cols = ["dataset", "model"]
    curve_cols = [c for c in CURVE_COLS if c not in panel_cols]
    panels = sorted({tuple(r) for r in table[panel_cols].itertuples(index=False)}, key=str)
    constants = constant_columns(table, curve_cols)

    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.8 * len(panels), 4.4), squeeze=False, sharey=True
    )
    for ax, panel in zip(axes[0], panels):
        subset = table
        for column, value in zip(panel_cols, panel):
            subset = subset[subset[column] == value]
        curves = sorted({tuple(r) for r in subset[curve_cols].itertuples(index=False)}, key=str)
        for index, curve in enumerate(curves):
            line = subset
            for column, value in zip(curve_cols, curve):
                line = line[line[column] == value]
            line = line.sort_values("adversarial_fraction")
            ax.errorbar(
                line["adversarial_fraction"],
                line["mean_accuracy"],
                yerr=[
                    (line["mean_accuracy"] - line["ci_lower"]).clip(lower=0).fillna(0),
                    (line["ci_upper"] - line["mean_accuracy"]).clip(lower=0).fillna(0),
                ],
                marker=MARKERS[index % len(MARKERS)],
                linestyle=LINESTYLES[index % len(LINESTYLES)],
                capsize=3,
                linewidth=1.6,
                label=series_label(dict(zip(curve_cols, curve)), skip_constant=constants),
            )
        ax.set_xlabel("Adversarial fraction (attacked agents / agents)")
        ax.set_title(", ".join(f"{c}={v}" for c, v in zip(panel_cols, panel)), fontsize=10)
        style_axes(ax)
        ax.legend(fontsize=8, frameon=False)

    axes[0][0].set_ylabel("Accuracy (95% cluster bootstrap CI)")
    suffix = "  (" + ", ".join(f"{k}={v}" for k, v in constants.items()) + ")" if constants else ""
    fig.suptitle(title_for(FIGURE_NAME + suffix), fontsize=12)
    fig.tight_layout()
    return fig


def build_figure(
    analysis_dir: str | Path,
    *,
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    analysis_seed: int = DEFAULT_ANALYSIS_SEED,
    command: Optional[str] = None,
) -> Dict[str, Any]:
    analysis_dir = Path(analysis_dir)
    tables = load_tables(analysis_dir)
    results = tables["results"]

    frame = prepare(results)
    table, excluded = aggregate(frame, repetitions=repetitions, analysis_seed=analysis_seed)
    table = table[list(OUTPUT_COLUMNS)]

    paths = figure_paths(analysis_dir, SLUG)
    table.to_csv(paths.table, index=False)
    save_figure(plot(table), paths)

    metadata = write_metadata(
        paths,
        {
            "figure": 1,
            "name": FIGURE_NAME,
            "analysis_dir": str(analysis_dir),
            "input_tables": {
                "runs": str(analysis_dir / "tables" / "runs.csv"),
                "results": str(analysis_dir / "tables" / "results.csv"),
            },
            "grouping_columns": list(CURVE_COLS) + ["adversarial_fraction"],
            "clean_arm_handling": (
                "a clean (attacker_count == 0) arm anchors every attack curve in "
                "its group at fraction 0.0; it is not a separate curve"
            ),
            "cluster_column": "example_id",
            "analysis_seed": analysis_seed,
            "bootstrap_repetitions": repetitions,
            "datasets": sorted({str(v) for v in table["dataset"].unique()}),
            "models": sorted({str(v) for v in table["model"].unique()}),
            "methods": sorted({str(v) for v in table["method"].unique()}),
            "attack_types": sorted({str(v) for v in table["attack_type"].unique()}),
            "routing_modes": sorted({str(v) for v in table["routing_mode"].unique()}),
            "adversarial_fractions": sorted({float(v) for v in table["adversarial_fraction"].unique()}),
            "excluded_curves": excluded,
            "n_plotted_points": int(len(table)),
            "command": command or command_line("figure_accuracy_vs_adversaries.py"),
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
            repetitions=args.bootstrap_repetitions,
            analysis_seed=args.analysis_seed,
            command=command_line("figure_accuracy_vs_adversaries.py", argv),
        )
    except (FigureError, RunLoadError) as exc:
        print(f"Figure 1 not produced: {exc}", file=sys.stderr)
        return 1
    print_summary(
        FIGURE_NAME, result["table"], result["paths"]
    )
    for group in result["excluded"]:
        print(f"  excluded: {group}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
