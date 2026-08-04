"""Figure 2 -- empirical best fixed-report utility gain (end-to-end IC-regret).

What this measures
------------------
Paired runs. The same debates are run twice: once with the agent reporting
truthfully, once with its report pinned to a fixed value v. The difference in
the agent's utility is what deviating bought it:

    U_i = w_rho * rho_i + w_adoption * 1[final group answer == attacker's
                                          initial answer]
    P_i = 0            (no explicit penalty term is implemented yet)

    gain(v) = U_i(fixed report v) - U_i(truthful)

and the reported number is ``max_v gain(v)`` over v in 1..5.

What it is not
--------------
This is **not** exact strategyproofness regret, and not an offline replay of
sampled PEAR's candidate pool -- the realised pool is not logged, so it cannot
be replayed. Concretely:

* changing the report changes the whole subsequent debate, so the difference
  includes every downstream effect, not just the routing one;
* the report is fixed across all applicable rounds rather than optimised per
  round;
* only five fixed reports are searched, so this is a lower bound on the best
  deviation.

Call it the *empirical best fixed-report utility gain* or *empirical
end-to-end IC-regret*. Do not call it universal or exact.

Usage
-----
    python analysis/figure_empirical_ic_regret.py \\
        analysis_artifacts/my_analysis --w-rho 1.0 --w-adoption 1.0

The utility weights are required: there is no defensible default, and this
script does not search over them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.collect_runs import load_tables  # noqa: E402
from analysis.common import RunLoadError  # noqa: E402
from analysis.figure_utils import (  # noqa: E402
    FigureError,
    add_common_arguments,
    command_line,
    figure_paths,
    print_summary,
    save_figure,
    style_axes,
    title_for,
    write_metadata,
)
from analysis.statistics import (  # noqa: E402
    DEFAULT_ANALYSIS_SEED,
    DEFAULT_BOOTSTRAP_REPETITIONS,
    paired_cluster_bootstrap_difference,
)

SLUG = "figure2_empirical_ic_regret"
FIGURE_NAME = "Empirical best fixed-report utility gain"

GROUP_COLS: Tuple[str, ...] = (
    "dataset",
    "model",
    "method",
    "mechanism",
    "verification_mode",
    "routing_mode",
)
PAIR_COLS: Tuple[str, ...] = ("example_id", "seed", "perm_seed", "attacker_id")
REPORT_VALUES: Tuple[int, ...] = (1, 2, 3, 4, 5)
TRUTHFUL_ARM = "truthful"

OUTPUT_COLUMNS: Tuple[str, ...] = (
    *GROUP_COLS,
    "attacker_id",
    "report_value",
    "mean_utility_truthful",
    "mean_utility_fixed_report",
    "utility_gain",
    "ci_lower",
    "ci_upper",
    "n_examples",
    "n_pairs",
    "is_best_report",
    "w_rho",
    "w_adoption",
    "penalty",
    "analysis_seed",
    "bootstrap_repetitions",
)


# ------------------------------------------------------------ answer parsing --
def canonical_answer_parser(dataset: Optional[str]) -> Tuple[Callable[[str], str], str]:
    """The benchmark's own answer parser, or a documented fallback.

    Adoption compares the group's final answer with the attacker's initial one.
    Raw string equality gets that wrong whenever the benchmark's canonical form
    differs from the surface string (``"7."`` vs ``"7"``), so the dataset's
    ``parse_answer`` is used when the dataset is registered.

    The task is not instantiated: constructing one loads a whole benchmark
    split. ``parse_answer`` is a pure text function in every registered task,
    so an uninitialised instance is enough, and any failure falls back to a
    normalised string comparison rather than pretending.
    """
    try:
        from data.tasks import TASK_REGISTRY

        task_cls = TASK_REGISTRY.get(str(dataset))
        if task_cls is None:
            return (lambda text: str(text or "").strip()), "normalised string (dataset not registered)"
        instance = object.__new__(task_cls)
        parser = instance.parse_answer
        parser("probe 1")  # fail here rather than mid-analysis
        return (lambda text: str(parser(str(text or "")) or "").strip()), f"{task_cls.name}.parse_answer"
    except Exception:  # pragma: no cover - defensive: any task that needs state
        return (lambda text: str(text or "").strip()), "normalised string (parser unavailable)"


# ------------------------------------------------------------------ utility --
def _history(cell: Any) -> Optional[list]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None
    if isinstance(cell, list):
        return cell
    try:
        value = json.loads(cell)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, list) else None


def final_rho(row: pd.Series, agent_id: int) -> Optional[float]:
    """Accumulated influence rho of the agent at the end of the debate."""
    history = _history(row.get("influence_history"))
    if not history:
        return None
    last = history[-1]
    if not isinstance(last, dict):
        return None
    value = last.get(str(agent_id), last.get(agent_id))
    return None if value is None else float(value)


def initial_answer(row: pd.Series, agent_id: int) -> Optional[str]:
    """The attacker's own first answer, before any debate."""
    history = _history(row.get("answer_history"))
    if not history:
        return None
    first = history[0]
    if not isinstance(first, dict):
        return None
    value = first.get(str(agent_id), first.get(agent_id))
    return None if value is None else str(value)


def utility(
    row: pd.Series,
    agent_id: int,
    *,
    w_rho: float,
    w_adoption: float,
    parse: Callable[[str], str],
) -> Optional[float]:
    """``w_rho * rho + w_adoption * adoption``; ``None`` when inputs are missing."""
    rho = final_rho(row, agent_id)
    answer = initial_answer(row, agent_id)
    if rho is None or answer is None:
        return None
    decision = row.get("prediction")
    adopted = float(parse(str(decision)) == parse(answer)) if decision is not None else 0.0
    return w_rho * rho + w_adoption * adopted


# ------------------------------------------------------------------- arms --
def label_arms(results: pd.DataFrame) -> pd.DataFrame:
    """Tag every row with its arm: truthful, or ``report_<v>``."""
    frame = results.copy()
    attacked = frame["attacker_count"].fillna(0) > 0
    fixed = attacked & (frame["attack_mode"].astype(str) == "fixed_report")
    frame["arm"] = TRUTHFUL_ARM
    frame.loc[attacked, "arm"] = "other_attack"
    frame.loc[fixed, "arm"] = frame.loc[fixed, "attack_value"].map(
        lambda v: f"report_{int(v)}" if pd.notna(v) else "other_attack"
    )
    return frame


def attacker_of(subset: pd.DataFrame) -> int:
    """The single confidence attacker of a group, or a clear refusal."""
    ids: set[int] = set()
    counts: set[int] = set()
    for _, row in subset.iterrows():
        if (row.get("attacker_count") or 0) <= 0:
            continue
        counts.add(int(row["attacker_count"]))
        for value in json.loads(row["attacker_ids"]) if isinstance(row.get("attacker_ids"), str) else []:
            ids.add(int(value))
    if counts - {1}:
        raise FigureError(
            f"this analysis assumes exactly one confidence attacker; the group "
            f"contains arms with attacker_count in {sorted(counts)}. A multi-attacker "
            "utility needs a stated joint-deviation model, which is not implemented."
        )
    if len(ids) != 1:
        raise FigureError(
            f"the attacked arms name {sorted(ids) or 'no'} attacker(s); pairing "
            "needs one agent whose utility is compared across arms"
        )
    return ids.pop()


def build_utilities(
    frame: pd.DataFrame,
    attacker_id: int,
    *,
    w_rho: float,
    w_adoption: float,
    parse: Callable[[str], str],
) -> pd.DataFrame:
    values = frame.apply(
        lambda row: utility(row, attacker_id, w_rho=w_rho, w_adoption=w_adoption, parse=parse),
        axis=1,
    )
    out = frame.copy()
    out["attacker_id"] = attacker_id
    out["utility"] = values
    missing = int(out["utility"].isna().sum())
    if missing:
        raise FigureError(
            f"{missing} row(s) lack influence_history or answer_history, so the "
            "attacker's utility cannot be computed. Missing history is not zero "
            "utility -- re-run those arms or exclude them explicitly."
        )
    return out


# -------------------------------------------------------------- aggregation --
def analyse_group(
    subset: pd.DataFrame,
    label: Dict[str, str],
    *,
    w_rho: float,
    w_adoption: float,
    repetitions: int,
    analysis_seed: int,
) -> List[Dict[str, Any]]:
    """One row per report value for a single analysis group."""
    parse, _ = canonical_answer_parser(label.get("dataset"))
    attacker_id = attacker_of(subset)
    with_utility = build_utilities(
        subset, attacker_id, w_rho=w_rho, w_adoption=w_adoption, parse=parse
    )

    rows: List[Dict[str, Any]] = []
    for value in REPORT_VALUES:
        arm = f"report_{value}"
        difference = paired_cluster_bootstrap_difference(
            with_utility,
            value_col="utility",
            arm_col="arm",
            arm_a=TRUTHFUL_ARM,
            arm_b=arm,
            pair_cols=list(PAIR_COLS),
            cluster_col="example_id",
            repetitions=repetitions,
            analysis_seed=analysis_seed,
        )
        rows.append(
            {
                **label,
                "attacker_id": attacker_id,
                "report_value": value,
                "mean_utility_truthful": difference.mean_arm_a,
                "mean_utility_fixed_report": difference.mean_arm_b,
                "utility_gain": difference.mean_difference,
                "ci_lower": difference.ci_lower,
                "ci_upper": difference.ci_upper,
                "n_examples": difference.n_examples,
                "n_pairs": difference.n_pairs,
                "w_rho": w_rho,
                "w_adoption": w_adoption,
                "penalty": 0.0,
                "analysis_seed": analysis_seed,
                "bootstrap_repetitions": repetitions,
            }
        )

    best = max(rows, key=lambda r: r["utility_gain"])
    for row in rows:
        row["is_best_report"] = row is best
    return rows


def aggregate(
    results: pd.DataFrame,
    *,
    w_rho: float,
    w_adoption: float,
    repetitions: int,
    analysis_seed: int,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    frame = label_arms(results)
    for column in GROUP_COLS:
        if column not in frame.columns:
            frame[column] = "n/a"
        frame[column] = frame[column].astype(object).where(frame[column].notna(), "n/a").astype(str)

    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for key, subset in frame.groupby(list(GROUP_COLS), dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        label = dict(zip(GROUP_COLS, [str(k) for k in key_tuple]))
        arms = set(subset["arm"].unique())
        present = sorted(int(a.split("_")[1]) for a in arms if a.startswith("report_"))
        absent = [v for v in REPORT_VALUES if v not in present]
        if TRUTHFUL_ARM not in arms or absent:
            reasons = []
            if TRUTHFUL_ARM not in arms:
                reasons.append("no truthful arm (confidence_inflation disabled)")
            if absent:
                reasons.append(f"missing fixed_report values {absent}")
            skipped.append({**label, "reason": "; ".join(reasons), "report_values_present": present})
            continue
        usable = subset[subset["arm"].isin([TRUTHFUL_ARM, *[f"report_{v}" for v in REPORT_VALUES]])]
        try:
            rows.extend(
                analyse_group(
                    usable,
                    label,
                    w_rho=w_rho,
                    w_adoption=w_adoption,
                    repetitions=repetitions,
                    analysis_seed=analysis_seed,
                )
            )
        except FigureError as exc:
            skipped.append({**label, "reason": str(exc)})
        except ValueError as exc:  # unmatched pairing keys from the statistics layer
            skipped.append({**label, "reason": f"arms are not paired: {exc}"})

    if not rows:
        raise FigureError(
            "no analysis group has a truthful arm plus fixed reports 1-5 with "
            "matching coverage. "
            + "; ".join(f"{r.get('reason')}" for r in skipped)
        )
    return pd.DataFrame(rows), skipped


def plot(table: pd.DataFrame, *, w_rho: float, w_adoption: float):
    groups = sorted({tuple(r) for r in table[list(GROUP_COLS)].itertuples(index=False)}, key=str)
    fig, axes = plt.subplots(
        1, len(groups), figsize=(5.4 * len(groups), 4.2), squeeze=False, sharey=True
    )
    for ax, group in zip(axes[0], groups):
        subset = table
        for column, value in zip(GROUP_COLS, group):
            subset = subset[subset[column] == value]
        subset = subset.sort_values("report_value")
        colours = ["tab:orange" if best else "tab:blue" for best in subset["is_best_report"]]
        ax.bar(subset["report_value"], subset["utility_gain"], color=colours, width=0.6)
        ax.errorbar(
            subset["report_value"],
            subset["utility_gain"],
            yerr=[
                (subset["utility_gain"] - subset["ci_lower"]).clip(lower=0).fillna(0),
                (subset["ci_upper"] - subset["utility_gain"]).clip(lower=0).fillna(0),
            ],
            fmt="none",
            ecolor="0.2",
            capsize=3,
        )
        ax.axhline(0.0, color="0.3", linewidth=0.9)
        ax.set_xticks(list(REPORT_VALUES))
        ax.set_xlabel("Fixed report value")
        ax.set_title(
            ", ".join(f"{c}={v}" for c, v in zip(GROUP_COLS, group) if c in {"dataset", "model", "verification_mode"}),
            fontsize=9,
        )
        style_axes(ax)

    axes[0][0].set_ylabel("Utility gain vs truthful\n(paired, 95% cluster CI)")
    fig.suptitle(
        title_for(
            f"{FIGURE_NAME}  (w_rho={w_rho}, w_adoption={w_adoption}, penalty=0)"
        ),
        fontsize=12,
    )
    fig.tight_layout()
    return fig


def build_figure(
    analysis_dir: str | Path,
    *,
    w_rho: float,
    w_adoption: float,
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    analysis_seed: int = DEFAULT_ANALYSIS_SEED,
    command: Optional[str] = None,
) -> Dict[str, Any]:
    analysis_dir = Path(analysis_dir)
    results = load_tables(analysis_dir)["results"]

    table, skipped = aggregate(
        results,
        w_rho=w_rho,
        w_adoption=w_adoption,
        repetitions=repetitions,
        analysis_seed=analysis_seed,
    )
    table = table[list(OUTPUT_COLUMNS)]

    paths = figure_paths(analysis_dir, SLUG)
    table.to_csv(paths.table, index=False)
    save_figure(plot(table, w_rho=w_rho, w_adoption=w_adoption), paths)

    best = table[table["is_best_report"]]
    _, parser_name = canonical_answer_parser(
        str(table["dataset"].iloc[0]) if len(table) else None
    )
    metadata = write_metadata(
        paths,
        {
            "figure": 2,
            "name": FIGURE_NAME,
            "alternative_name": "Empirical end-to-end IC-regret",
            "not": (
                "not exact or universal strategyproofness regret; not an offline "
                "replay of sampled PEAR's candidate pool, which is not logged"
            ),
            "analysis_dir": str(analysis_dir),
            "utility": "U_i = w_rho * rho_i + w_adoption * 1[final answer == attacker initial answer]",
            "w_rho": w_rho,
            "w_adoption": w_adoption,
            "penalty": 0.0,
            "penalty_note": "P_i = 0: no explicit penalty value is implemented",
            "answer_parser": parser_name,
            "pairing_columns": list(PAIR_COLS),
            "grouping_columns": list(GROUP_COLS),
            "report_values": list(REPORT_VALUES),
            "cluster_column": "example_id",
            "analysis_seed": analysis_seed,
            "bootstrap_repetitions": repetitions,
            "best_reports": best[["report_value", "utility_gain", *GROUP_COLS]].to_dict("records"),
            "skipped_groups": skipped,
            "limitations": [
                "changing a report changes the whole subsequent debate, so the "
                "difference is end-to-end and not attributable to routing alone",
                "the report is fixed across all applicable rounds",
                "only the five fixed reports are searched, so this is a lower "
                "bound on the best deviation",
            ],
            "command": command or command_line("figure_empirical_ic_regret.py"),
        },
    )
    return {"table": table, "paths": paths, "metadata": metadata, "skipped": skipped}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--w-rho",
        type=float,
        required=True,
        help="Weight on the attacker's final accumulated influence rho. Required: "
        "there is no defensible default and the script does not search weights.",
    )
    parser.add_argument(
        "--w-adoption",
        type=float,
        required=True,
        help="Weight on the group adopting the attacker's initial answer.",
    )
    args = parser.parse_args(argv)

    try:
        result = build_figure(
            args.analysis_dir,
            w_rho=args.w_rho,
            w_adoption=args.w_adoption,
            repetitions=args.bootstrap_repetitions,
            analysis_seed=args.analysis_seed,
            command=command_line("figure_empirical_ic_regret.py", argv),
        )
    except (FigureError, RunLoadError) as exc:
        print(f"Figure 2 not produced: {exc}", file=sys.stderr)
        return 1

    print_summary(
        FIGURE_NAME, result["table"], result["paths"]
    )
    for row in result["metadata"]["best_reports"]:
        print(f"  best fixed report: value={row['report_value']} gain={row['utility_gain']:+.4f}")
    for group in result["skipped"]:
        print(f"  skipped: {group}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
