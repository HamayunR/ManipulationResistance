"""Figure 3 -- adversary routing-exposure share vs reported confidence.

What is plotted
---------------
For the agent under a confidence-reporting attack, the share of a round's
out-edges that originate at it -- how much of the debate it gets to talk at --
against the confidence it *reported*.

That share is **routing exposure**. It is not PEAR's accumulated influence:
rho is a separate state variable, carried in the normalised tables as
``influence_before`` and never averaged into this quantity.

Three confidences are kept distinct throughout:

``reported_confidence``   what the agent claimed, after any attack. The x-axis.
``verified_confidence``   ``min(reported, g_i)`` under an active verification
                          mode.
``routing_confidence``    what the router actually scored: the reported value
                          with verification off, the verified value with it on.

The x-axis stays *reported* confidence even for oracle curves, because the
question is what a manipulated report buys. Verification changes the answer,
not the question, so ``verification_mode`` stays a visible curve label.

Usage
-----
    python analysis/figure_exposure_vs_confidence.py \\
        analysis_artifacts/my_analysis --allow-mock

Reads ``tables/runs.csv`` and ``tables/routing.csv``; writes
``tables/figure3_exposure_vs_confidence.csv``, PNG, PDF and a metadata JSON.
The normalised ``routing.csv`` is never modified.
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
    annotate_mock,
    bool_series,
    command_line,
    constant_columns,
    figure_paths,
    mock_status_of,
    print_summary,
    require_separable_mock_status,
    save_figure,
    series_label,
    style_axes,
    title_for,
    write_metadata,
)
from analysis.statistics import (  # noqa: E402
    DEFAULT_ANALYSIS_SEED,
    DEFAULT_BOOTSTRAP_REPETITIONS,
    average_within,
    cluster_bootstrap_mean,
)

SLUG = "figure3_exposure_vs_confidence"
FIGURE_NAME = "Adversary routing-exposure share vs reported confidence"

#: Nothing is pooled across these. Sampled and enumerated routing are different
#: mechanisms; so are two datasets, two models, two debate mechanisms and two
#: verification modes.
GROUP_COLS: Tuple[str, ...] = (
    "dataset",
    "model",
    "mechanism",
    "verification_mode",
    "routing_mode",
)

#: Columns a routing row must carry for this figure to mean anything.
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "is_attacker",
    "reported_confidence",
    "out_degree_share",
    "source_eligible",
    "target_eligible",
    "example_id",
    "routing_mode",
    "dataset",
    "model",
    "mechanism",
    "verification_mode",
)

#: One observation per debate, before clustering: the replication coordinates.
REPLICATION_COLS: Tuple[str, ...] = ("run_id", "condition", "seed", "perm_seed", "agent_id")

#: The full rubric. A complete diagnostic sweep covers all of it.
REPORT_VALUES: Tuple[int, ...] = (1, 2, 3, 4, 5)

OUTPUT_COLUMNS: Tuple[str, ...] = (
    "dataset",
    "model",
    "mechanism",
    "verification_mode",
    "routing_mode",
    "reported_confidence",
    "mean_out_degree_share",
    "ci_lower",
    "ci_upper",
    "source_eligibility_rate",
    "target_eligibility_rate",
    "n_examples",
    "n_raw_rows",
    "n_seeds",
    "n_perm_seeds",
    "analysis_seed",
    "bootstrap_repetitions",
    "mock",
    "diagnostic_only",
)


def select_attacker_rows(routing: pd.DataFrame) -> pd.DataFrame:
    """Attacker rows with everything the figure needs, and nothing invented."""
    missing = [c for c in REQUIRED_COLUMNS if c not in routing.columns]
    if missing:
        raise FigureError(
            f"routing.csv is missing required column(s) {missing}; re-run "
            "analysis/collect_runs.py with a current checkout"
        )
    frame = routing.copy()
    frame["is_attacker"] = bool_series(frame, "is_attacker").fillna(False).astype(bool)
    attackers = frame[frame["is_attacker"]]
    if attackers.empty:
        raise FigureError(
            "no routing row is flagged is_attacker. This figure measures what a "
            "manipulated report buys the adversary, so it needs at least one "
            "confidence-attacker arm."
        )
    usable = attackers[
        attackers["reported_confidence"].notna() & attackers["out_degree_share"].notna()
    ]
    if usable.empty:
        raise FigureError(
            "attacker rows carry no reported_confidence / out_degree_share. "
            "Rows where the round's total out-degree was zero are flagged and "
            "left missing rather than divided by zero."
        )
    return usable


def aggregate(
    attackers: pd.DataFrame,
    *,
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    analysis_seed: int = DEFAULT_ANALYSIS_SEED,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Aggregate to one plotted point per group x reported confidence.

    Rounds are averaged first, within each example x replication x reported
    confidence, so a three-round debate contributes one observation per debate
    rather than three. The bootstrap then resamples ``example_id``: repeated
    rounds and repeated seeds of one question are never counted as independent
    questions.

    Returns ``(plotted points, groups excluded with a reason)``.
    """
    group_cols = list(GROUP_COLS)
    replication_cols = [c for c in REPLICATION_COLS if c in attackers.columns]

    # 1. Collapse rounds within one debate.
    per_debate = average_within(
        attackers,
        value_col="out_degree_share",
        within_cols=[*group_cols, "reported_confidence", "example_id", *replication_cols],
        keep_cols=["source_eligible", "target_eligible"],
    )
    # Eligibility flags are per round; average them over the same unit so the
    # rates describe the same observations as the exposure share.
    for flag in ("source_eligible", "target_eligible"):
        rates = average_within(
            attackers.assign(**{f"_{flag}": bool_series(attackers, flag).astype("float")}),
            value_col=f"_{flag}",
            within_cols=[*group_cols, "reported_confidence", "example_id", *replication_cols],
        )
        per_debate = per_debate.merge(
            rates,
            on=[*group_cols, "reported_confidence", "example_id", *replication_cols],
            how="left",
        )

    # 2. Cluster bootstrap over examples.
    stats = cluster_bootstrap_mean(
        per_debate,
        value_col="out_degree_share",
        cluster_col="example_id",
        group_cols=[*group_cols, "reported_confidence"],
        repetitions=repetitions,
        analysis_seed=analysis_seed,
    ).rename(columns={"mean": "mean_out_degree_share"})

    # 3. Eligibility rates and replication counts alongside each point.
    extras = (
        per_debate.groupby([*group_cols, "reported_confidence"], dropna=False, sort=True)
        .agg(
            source_eligibility_rate=("_source_eligible", "mean"),
            target_eligibility_rate=("_target_eligible", "mean"),
            n_seeds=("seed", "nunique") if "seed" in per_debate.columns else ("example_id", "size"),
            n_perm_seeds=(
                ("perm_seed", "nunique") if "perm_seed" in per_debate.columns else ("example_id", "size")
            ),
        )
        .reset_index()
    )
    table = stats.merge(extras, on=[*group_cols, "reported_confidence"], how="left")

    # 4. A line needs at least two x values; a single point is not a curve.
    plotted: List[pd.DataFrame] = []
    excluded: List[Dict[str, Any]] = []
    for key, subset in table.groupby(group_cols, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        label = dict(zip(group_cols, [str(k) for k in key_tuple]))
        values = sorted(float(v) for v in subset["reported_confidence"].unique())
        if len(values) < 2:
            excluded.append(
                {
                    **label,
                    "reported_confidence_values": values,
                    "reason": (
                        "only one reported-confidence value; a curve needs at "
                        f"least two (complete sweep: {list(REPORT_VALUES)})"
                    ),
                }
            )
            continue
        plotted.append(subset)

    if not plotted:
        raise FigureError(
            "no analysis group has two or more distinct reported-confidence "
            f"values. Run the fixed-report sweep over values {list(REPORT_VALUES)} "
            "and collect it again. Excluded groups: "
            + "; ".join(f"{g['reason']}" for g in excluded)
        )

    result = pd.concat(plotted, ignore_index=True).sort_values(
        [*group_cols, "reported_confidence"]
    )
    return result.reset_index(drop=True), excluded


def plot(table: pd.DataFrame, *, mock_status: str):
    """One panel per dataset/model/mechanism; one curve per verification x routing."""
    panel_cols = ["dataset", "model", "mechanism"]
    curve_cols = ["verification_mode", "routing_mode"]
    panels = sorted({tuple(row) for row in table[panel_cols].itertuples(index=False)}, key=str)

    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.6 * len(panels), 4.4), squeeze=False, sharey=True
    )
    constants = constant_columns(table, curve_cols)

    for ax, panel in zip(axes[0], panels):
        subset = table
        for column, value in zip(panel_cols, panel):
            subset = subset[subset[column] == value]
        curves = sorted({tuple(row) for row in subset[curve_cols].itertuples(index=False)}, key=str)
        for index, curve in enumerate(curves):
            line = subset
            for column, value in zip(curve_cols, curve):
                line = line[line[column] == value]
            line = line.sort_values("reported_confidence")
            key = dict(zip(curve_cols, curve))
            ax.errorbar(
                line["reported_confidence"],
                line["mean_out_degree_share"],
                yerr=[
                    (line["mean_out_degree_share"] - line["ci_lower"]).clip(lower=0).fillna(0),
                    (line["ci_upper"] - line["mean_out_degree_share"]).clip(lower=0).fillna(0),
                ],
                marker=MARKERS[index % len(MARKERS)],
                linestyle=LINESTYLES[index % len(LINESTYLES)],
                capsize=3,
                linewidth=1.6,
                label=series_label(key, skip_constant=constants),
            )
        ax.set_xlabel("Reported confidence (1-5 rubric)")
        ax.set_xticks(sorted({float(v) for v in subset["reported_confidence"].unique()}))
        ax.set_title(", ".join(f"{c}={v}" for c, v in zip(panel_cols, panel)), fontsize=10)
        style_axes(ax)
        if len(curves) > 1 or not constants:
            ax.legend(fontsize=8, frameon=False)

    axes[0][0].set_ylabel("Adversary routing-exposure share\n(out-degree share, 95% cluster CI)")
    suffix = (
        "  (" + ", ".join(f"{k}={v}" for k, v in constants.items()) + ")" if constants else ""
    )
    fig.suptitle(title_for(FIGURE_NAME + suffix, mock_status), fontsize=12)
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
    """Produce the CSV, PNG, PDF and metadata for Figure 3."""
    analysis_dir = Path(analysis_dir)
    tables = load_tables(analysis_dir)
    runs, routing = tables["runs"], tables["routing"]

    if routing.empty:
        no_routing = sorted(
            {
                str(m)
                for m, subset in runs.groupby("method", dropna=False)
                if not subset.get("has_routing", pd.Series(dtype=bool)).fillna(False).any()
            }
        )
        raise FigureError(
            "routing.csv has no rows: no input method logs routing decisions. "
            f"Methods without routing data (a capability limitation, not a "
            f"validation failure): {no_routing or 'unknown'}. They remain "
            "eligible for the accuracy and cost figures."
        )

    mock_status = mock_status_of(routing)
    require_separable_mock_status(mock_status, allow_mock=allow_mock, what="Figure 3")
    diagnostic_only = mock_status != "real"

    attackers = select_attacker_rows(routing)
    table, excluded = aggregate(
        attackers, repetitions=repetitions, analysis_seed=analysis_seed
    )
    table["mock"] = mock_status == "mock"
    table["diagnostic_only"] = diagnostic_only
    table = table[list(OUTPUT_COLUMNS)]

    paths = figure_paths(analysis_dir, SLUG)
    table.to_csv(paths.table, index=False)
    save_figure(plot(table, mock_status=mock_status), paths)

    values_present = sorted({int(v) for v in table["reported_confidence"].unique()})
    metadata = write_metadata(
        paths,
        {
            "figure": 3,
            "name": FIGURE_NAME,
            "quantity": (
                "adversary out-degree share per routing decision (routing "
                "exposure); PEAR accumulated influence rho is a separate "
                "quantity and is not plotted here"
            ),
            "x_axis": "reported_confidence (pre-verification report)",
            "analysis_dir": str(analysis_dir),
            "input_tables": {
                "runs": str(analysis_dir / "tables" / "runs.csv"),
                "routing": str(analysis_dir / "tables" / "routing.csv"),
            },
            "filters": {
                "is_attacker": True,
                "reported_confidence": "not null",
                "out_degree_share": "not null (zero-total-degree rounds excluded and flagged)",
            },
            "grouping_columns": list(GROUP_COLS),
            "round_handling": (
                "rounds averaged within example x replication before "
                "bootstrapping; rounds are not independent questions"
            ),
            "analysis_seed": analysis_seed,
            "bootstrap_repetitions": repetitions,
            "cluster_column": "example_id",
            "mock_status": mock_status,
            "diagnostic_only": diagnostic_only,
            "routing_modes": sorted({str(v) for v in table["routing_mode"].unique()}),
            "datasets": sorted({str(v) for v in table["dataset"].unique()}),
            "models": sorted({str(v) for v in table["model"].unique()}),
            "mechanisms": sorted({str(v) for v in table["mechanism"].unique()}),
            "verification_modes": sorted({str(v) for v in table["verification_mode"].unique()}),
            "reported_confidence_values": values_present,
            "complete_sweep": values_present == list(REPORT_VALUES),
            "excluded_groups": excluded,
            "n_plotted_points": int(len(table)),
            "command": command or command_line("figure_exposure_vs_confidence.py"),
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
            command=command_line("figure_exposure_vs_confidence.py", argv),
        )
    except (FigureError, RunLoadError) as exc:
        print(f"Figure 3 not produced: {exc}", file=sys.stderr)
        return 1

    print_summary(
        FIGURE_NAME,
        result["table"],
        result["paths"],
        diagnostic_only=result["metadata"]["diagnostic_only"],
    )
    for group in result["excluded"]:
        print(f"  excluded: {group}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
