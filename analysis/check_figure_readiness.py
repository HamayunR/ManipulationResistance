"""Which headline figures can be drawn from the collected runs, and why not.

Usage
-----
    python analysis/check_figure_readiness.py analysis_artifacts/my_analysis

Reads ``tables/{runs,results,routing}.csv``, writes ``readiness.json`` next to
them and prints the same thing in prose.

The report exists so that "the figure is missing" is answered with an
experiment to run rather than a guess. For every planned figure it names the
arms that are absent, the capabilities no input method provides, and whether
what *is* present can support the claim a figure would make.

Two rules run through all of it:

* a missing capability is a property of the method, not a zero. A competitor
  with no routing log is excluded from routing figures and stays fully
  eligible for accuracy and cost;
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.common import PEAR_ROUTING_MODES, RunLoadError, analysis_git_commit  # noqa: E402
from analysis.collect_runs import load_tables  # noqa: E402
from analysis.statistics import broadcast_clean_arm  # noqa: E402

#: The report values the fixed-report sweep must cover for Figure 2/3 to be a
#: complete diagnostic. They are the whole confidence rubric.
REPORT_VALUES: Tuple[int, ...] = (1, 2, 3, 4, 5)

#: Grouping keys. Nothing is pooled across these: different datasets, models,
#: mechanisms, verification modes or routing modes are different experiments.
BASE_GROUP_COLS: Tuple[str, ...] = ("dataset", "model", "mechanism", "verification_mode")

FIGURE_NAMES: Dict[int, str] = {
    1: "Accuracy vs adversarial fraction",
    2: "Empirical best fixed-report utility gain",
    3: "Adversary routing-exposure share vs reported confidence",
    4: "Clean accuracy vs token cost",
    5: "Routing temperature x influence-weight heatmap",
}

NA = "n/a"


@dataclass
class FigureReadiness:
    """One figure's verdict."""

    figure: int
    name: str
    ready: bool
    reason: str
    required_groups: List[str] = field(default_factory=list)
    available_groups: List[Dict[str, Any]] = field(default_factory=list)
    missing_arms: List[str] = field(default_factory=list)
    missing_capabilities: List[str] = field(default_factory=list)
    routing_modes: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------ helpers --
def _labelled(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Copy with grouping columns filled, so a null key is a visible group."""
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = NA
        out[column] = out[column].astype(object).where(out[column].notna(), NA).astype(str)
    return out


def _group_label(key: Mapping[str, Any]) -> str:
    return " | ".join(f"{k}={v}" for k, v in key.items())


def _routing_modes(frame: pd.DataFrame) -> List[str]:
    if "routing_mode" not in frame:
        return []
    return sorted({str(v) for v in frame["routing_mode"].dropna().unique()})


def _verdict(
    figure: int,
    *,
    ready: bool,
    reason: str,
    **kwargs: Any,
) -> FigureReadiness:
    return FigureReadiness(
        figure=figure,
        name=FIGURE_NAMES[figure],
        ready=ready,
        reason=reason,
        **kwargs,
    )


def _capability_methods(runs: pd.DataFrame, capability: str) -> Tuple[List[str], List[str]]:
    """``(methods with the capability, methods without it)``."""
    if capability not in runs or runs.empty:
        return [], sorted({str(m) for m in runs.get("method", [])})
    have, lack = [], []
    for method, subset in runs.groupby("method", dropna=False):
        (have if subset[capability].fillna(False).astype(bool).any() else lack).append(str(method))
    return sorted(have), sorted(lack)


# --------------------------------------------------------------- figure 1 --
def check_figure1(runs: pd.DataFrame, results: pd.DataFrame, routing: pd.DataFrame) -> FigureReadiness:
    """Accuracy vs adversarial fraction."""
    group_cols = [*BASE_GROUP_COLS, "attack_type", "routing_mode"]
    required = [
        "accuracy (correct) on every arm",
        "a clean adversarial_fraction=0.0 arm",
        "at least two distinct adversarial fractions in a group",
    ]

    have_accuracy, lack_accuracy = _capability_methods(runs, "has_accuracy")
    if not have_accuracy:
        return _verdict(
            1,
            ready=False,
            reason="no input run reports per-example correctness",
            required_groups=required,
            missing_capabilities=["has_accuracy"],
        )

    frame = results[results["correct"].notna() & results["adversarial_fraction"].notna()]
    if frame.empty:
        return _verdict(
            1,
            ready=False,
            reason=(
                "no rows carry both accuracy and an adversarial fraction; the "
                "attacked-arm metadata is what identifies the x-axis"
            ),
            required_groups=required,
            missing_capabilities=["supports_adversarial_fraction"],
        )

    frame = _labelled(frame, group_cols)
    # The clean arm anchors every attack curve in its group at fraction 0.
    frame = broadcast_clean_arm(
        frame, base_group_cols=[*BASE_GROUP_COLS, "routing_mode"]
    )
    available: List[Dict[str, Any]] = []
    missing: List[str] = []
    for key, subset in frame.groupby(group_cols, dropna=False, sort=True):
        label = dict(zip(group_cols, [str(k) for k in key]))
        fractions = sorted({float(v) for v in subset["adversarial_fraction"].dropna().unique()})
        entry = {
            **label,
            "adversarial_fractions": fractions,
            "has_clean_arm": 0.0 in fractions,
            "n_examples": int(subset["example_id"].nunique()),
            "n_rows": int(len(subset)),
        }
        if len(fractions) >= 2 and 0.0 in fractions:
            available.append(entry)
        else:
            reason = "only one adversarial fraction" if len(fractions) < 2 else "no clean 0.0 arm"
            missing.append(f"{_group_label(label)}: {reason} (present: {fractions})")

    ready = bool(available)
    reason = (
        f"{len(available)} group(s) have a clean arm and at least two adversarial fractions"
        if ready
        else "no group has both a clean 0.0 arm and a second adversarial fraction"
    )
    return _verdict(
        1,
        ready=ready,
        reason=reason,
        required_groups=required,
        available_groups=available,
        missing_arms=missing,
        missing_capabilities=[] if ready else ["adversarial_fraction sweep"],
        routing_modes=_routing_modes(frame),
        notes=(
            [f"methods without accuracy data, excluded: {lack_accuracy}"]
            if lack_accuracy
            else []
        ),
    )


# --------------------------------------------------------------- figure 2 --
def _fixed_report_arms(subset: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split a group into the truthful arm and the fixed-report arms."""
    truthful = subset[(subset["attacker_count"].fillna(0) == 0)]
    attacked = subset[
        (subset["attacker_count"].fillna(0) > 0)
        & (subset["attack_mode"].astype(str) == "fixed_report")
    ]
    arms = {"truthful": truthful}
    for value, rows in attacked.groupby("attack_value", dropna=True):
        arms[f"fixed_report={int(value)}"] = rows
    return arms


def check_figure2(runs: pd.DataFrame, results: pd.DataFrame, routing: pd.DataFrame) -> FigureReadiness:
    """Empirical best fixed-report utility gain (paired truthful vs report v)."""
    group_cols = [*BASE_GROUP_COLS, "routing_mode"]
    required = [
        "one truthful arm with confidence_inflation disabled",
        "fixed_report arms for values 1, 2, 3, 4 and 5",
        "exactly one confidence attacker per attacked arm",
        "influence_history, answer_history and a decision on every row",
        "identical (example_id, seed, perm_seed) coverage across arms",
    ]

    missing_capabilities: List[str] = []
    if not runs.get("has_influence_history", pd.Series(dtype=bool)).fillna(False).any():
        missing_capabilities.append("has_influence_history")
    if "influence_history" not in results.columns or results["influence_history"].isna().all():
        missing_capabilities.append("influence_history")
    if "answer_history" not in results.columns or results["answer_history"].isna().all():
        missing_capabilities.append("answer_history")
    if "prediction" not in results.columns or results["prediction"].isna().all():
        missing_capabilities.append("prediction")
    if missing_capabilities:
        return _verdict(
            2,
            ready=False,
            reason=(
                "the utility needs rho from influence_history, the attacker's "
                f"initial answer from answer_history and the group decision; "
                f"missing: {missing_capabilities}"
            ),
            required_groups=required,
            missing_capabilities=missing_capabilities,
        )

    frame = _labelled(results, group_cols)
    available: List[Dict[str, Any]] = []
    missing: List[str] = []
    notes: List[str] = []

    for key, subset in frame.groupby(group_cols, dropna=False, sort=True):
        label = dict(zip(group_cols, [str(k) for k in key]))
        arms = _fixed_report_arms(subset)
        present_values = sorted(
            int(name.split("=")[1]) for name in arms if name.startswith("fixed_report=")
        )
        gaps: List[str] = []
        if arms["truthful"].empty:
            gaps.append("truthful arm (confidence_inflation disabled)")
        absent_values = [v for v in REPORT_VALUES if v not in present_values]
        if absent_values:
            gaps.append(f"fixed_report values {absent_values}")

        multi_attacker = sorted(
            {
                int(v)
                for name, rows in arms.items()
                if name != "truthful"
                for v in rows["attacker_count"].dropna().unique()
                if int(v) != 1
            }
        )
        if multi_attacker:
            gaps.append(
                f"exactly one confidence attacker (found attacker_count={multi_attacker})"
            )

        # Identical pairing-key coverage: an arm covering different examples is
        # not a paired comparison of the same questions.
        keys_by_arm = {
            name: set(map(tuple, rows[["example_id", "seed", "perm_seed"]].astype(str).values))
            for name, rows in arms.items()
            if not rows.empty
        }
        if len(keys_by_arm) > 1:
            reference = keys_by_arm.get("truthful") or next(iter(keys_by_arm.values()))
            mismatched = [name for name, keys in keys_by_arm.items() if keys != reference]
            if mismatched:
                gaps.append(f"identical paired key coverage (differs for {sorted(mismatched)})")

        entry = {
            **label,
            "report_values_present": present_values,
            "report_values_missing": absent_values,
            "has_truthful_arm": not arms["truthful"].empty,
            "n_examples": int(subset["example_id"].nunique()),
        }
        if gaps:
            missing.append(f"{_group_label(label)}: missing {'; '.join(gaps)}")
        else:
            available.append(entry)

    ready = bool(available)
    reason = (
        f"{len(available)} group(s) carry a truthful arm and fixed reports 1-5"
        if ready
        else "no group carries a truthful arm plus all five fixed-report arms"
    )
    return _verdict(
        2,
        ready=ready,
        reason=reason,
        required_groups=required,
        available_groups=available,
        missing_arms=missing,
        routing_modes=_routing_modes(frame),
        notes=notes,
    )


# --------------------------------------------------------------- figure 3 --
def check_figure3(runs: pd.DataFrame, results: pd.DataFrame, routing: pd.DataFrame) -> FigureReadiness:
    """Adversary routing-exposure share vs reported confidence."""
    group_cols = [*BASE_GROUP_COLS, "routing_mode"]
    required = [
        "routing rows (has_routing)",
        "attacker identity (is_attacker)",
        "reported_confidence and out_degree_share",
        "at least two distinct reported-confidence values per group",
        "one routing mode per curve",
    ]

    have_routing, lack_routing = _capability_methods(runs, "has_routing")
    notes = (
        [
            f"methods without routing data, excluded from this figure only: "
            f"{lack_routing} (capability limitation, not a validation failure)"
        ]
        if lack_routing
        else []
    )
    if routing.empty or not have_routing:
        return _verdict(
            3,
            ready=False,
            reason="no input run logs routing decisions",
            required_groups=required,
            missing_capabilities=["has_routing"],
            notes=notes,
        )

    attackers = routing[routing["is_attacker"].fillna(False).astype(bool)]
    if attackers.empty:
        return _verdict(
            3,
            ready=False,
            reason=(
                "routing rows exist but none is flagged is_attacker; the figure "
                "plots the adversary's exposure, so a clean-only sweep cannot "
                "produce it"
            ),
            required_groups=required,
            missing_arms=["at least one confidence-attacker arm"],
            routing_modes=_routing_modes(routing),
            notes=notes,
        )

    usable = attackers[
        attackers["reported_confidence"].notna() & attackers["out_degree_share"].notna()
    ]
    if usable.empty:
        return _verdict(
            3,
            ready=False,
            reason="attacker rows carry no reported_confidence / out_degree_share",
            required_groups=required,
            missing_capabilities=["has_confidence_reports"],
            notes=notes,
        )

    frame = _labelled(usable, group_cols)
    available: List[Dict[str, Any]] = []
    missing: List[str] = []
    for key, subset in frame.groupby(group_cols, dropna=False, sort=True):
        label = dict(zip(group_cols, [str(k) for k in key]))
        values = sorted({float(v) for v in subset["reported_confidence"].unique()})
        entry = {
            **label,
            "reported_confidence_values": values,
            "n_examples": int(subset["example_id"].nunique()),
            "n_rows": int(len(subset)),
            "complete_sweep": sorted(int(v) for v in values) == list(REPORT_VALUES),
        }
        if len(values) >= 2:
            available.append(entry)
        else:
            missing.append(
                f"{_group_label(label)}: only one reported-confidence value {values}; "
                f"a curve needs at least two (full sweep: {list(REPORT_VALUES)})"
            )

    ready = bool(available)
    complete = [g for g in available if g["complete_sweep"]]
    reason = (
        f"{len(available)} group(s) have at least two reported-confidence values"
        + ("" if complete else "; none covers the full 1-5 sweep yet")
        if ready
        else "every group has a single reported-confidence value"
    )
    return _verdict(
        3,
        ready=ready,
        reason=reason,
        required_groups=required,
        available_groups=available,
        missing_arms=missing,
        routing_modes=_routing_modes(frame),
        notes=notes,
    )


# --------------------------------------------------------------- figure 4 --
def check_figure4(runs: pd.DataFrame, results: pd.DataFrame, routing: pd.DataFrame) -> FigureReadiness:
    """Clean accuracy vs token cost."""
    group_cols = ["dataset", "model"]
    required = [
        "clean conditions (attacker_count == 0)",
        "accuracy",
        "non-missing token usage",
        "an identifiable method",
    ]

    clean = results[results["attacker_count"].fillna(0) == 0]
    if clean.empty:
        return _verdict(
            4,
            ready=False,
            reason="no clean (attacker_count == 0) condition in the inputs",
            required_groups=required,
            missing_arms=["a clean arm"],
        )

    clean = clean[clean["correct"].notna()]
    with_tokens = clean[clean["total_tokens"].notna()]
    methods_all = sorted({str(m) for m in clean["method"].dropna().unique()})
    methods_priced = sorted({str(m) for m in with_tokens["method"].dropna().unique()})
    excluded = [m for m in methods_all if m not in methods_priced]

    frame = _labelled(with_tokens, group_cols)
    available: List[Dict[str, Any]] = []
    for key, subset in frame.groupby(group_cols, dropna=False, sort=True):
        label = dict(zip(group_cols, [str(k) for k in key]))
        available.append(
            {
                **label,
                "methods": sorted({str(m) for m in subset["method"].unique()}),
                "verification_modes": sorted({str(v) for v in subset["verification_mode"].unique()}),
                "n_examples": int(subset["example_id"].nunique()),
            }
        )

    notes: List[str] = []
    if excluded:
        notes.append(
            f"methods excluded for missing token usage (never counted as zero): {excluded}"
        )
    _, lack_routing = _capability_methods(runs, "has_routing")
    if lack_routing:
        notes.append(f"methods without routing remain eligible here: {lack_routing}")

    ready = bool(available)
    return _verdict(
        4,
        ready=ready,
        reason=(
            f"{len(methods_priced)} method(s) have clean accuracy and token usage"
            if ready
            else "no clean condition reports token usage"
        ),
        required_groups=required,
        available_groups=available,
        missing_arms=[f"token usage for method {m}" for m in excluded],
        missing_capabilities=["has_token_usage"] if not ready else [],
        notes=notes,
    )


# --------------------------------------------------------------- figure 5 --
HEATMAP_METRICS = ("accuracy", "robustness_gap", "adversary_routing_exposure", "empirical_ic_regret")


def check_figure5(
    runs: pd.DataFrame,
    results: pd.DataFrame,
    routing: pd.DataFrame,
    *,
    metric: str = "accuracy",
) -> FigureReadiness:
    """Routing temperature x influence-weight heatmap."""
    group_cols = [*BASE_GROUP_COLS, "routing_mode"]
    required = [
        "routing_temperature and alpha_influence on the condition metadata",
        "at least a 2 x 2 parameter grid",
        "every grid cell present (cells are never interpolated)",
        "a metric chosen explicitly",
    ]

    if "routing_temperature" not in runs or runs["routing_temperature"].isna().all():
        return _verdict(
            5,
            ready=False,
            reason="no run records routing_temperature",
            required_groups=required,
            missing_capabilities=["routing_temperature"],
        )
    if "alpha_influence" not in runs or runs["alpha_influence"].isna().all():
        return _verdict(
            5,
            ready=False,
            reason="no run records alpha_influence",
            required_groups=required,
            missing_capabilities=["alpha_influence"],
        )

    frame = _labelled(runs, group_cols)
    available: List[Dict[str, Any]] = []
    missing: List[str] = []
    for key, subset in frame.groupby(group_cols, dropna=False, sort=True):
        label = dict(zip(group_cols, [str(k) for k in key]))
        temperatures = sorted({float(t) for t in subset["routing_temperature"].dropna().unique()})
        alphas = sorted({float(a) for a in subset["alpha_influence"].dropna().unique()})
        cells = {
            (float(r["routing_temperature"]), float(r["alpha_influence"]))
            for _, r in subset.dropna(subset=["routing_temperature", "alpha_influence"]).iterrows()
        }
        expected = {(t, a) for t in temperatures for a in alphas}
        holes = sorted(expected - cells)
        entry = {
            **label,
            "routing_temperatures": temperatures,
            "alpha_influence_values": alphas,
            "grid": f"{len(temperatures)} x {len(alphas)}",
            "missing_cells": [list(c) for c in holes],
        }
        if len(temperatures) >= 2 and len(alphas) >= 2 and not holes:
            available.append(entry)
        else:
            if len(temperatures) < 2 or len(alphas) < 2:
                missing.append(
                    f"{_group_label(label)}: grid is {len(temperatures)} x {len(alphas)}, "
                    "needs at least 2 x 2"
                )
            if holes:
                missing.append(f"{_group_label(label)}: missing grid cell(s) {holes}")

    notes = [f"metric requested: {metric}"]
    if metric == "robustness_gap":
        clean = results[results["attacker_count"].fillna(0) == 0]
        attacked = results[results["attacker_count"].fillna(0) > 0]
        if clean.empty or attacked.empty:
            missing.append(
                "robustness_gap needs matched clean and attacked conditions; "
                f"clean rows={len(clean)}, attacked rows={len(attacked)}"
            )
            available = []
    elif metric == "adversary_routing_exposure" and routing.empty:
        missing.append("adversary_routing_exposure needs routing rows; none present")
        available = []

    ready = bool(available)
    return _verdict(
        5,
        ready=ready,
        reason=(
            f"{len(available)} group(s) form a complete parameter grid"
            if ready
            else "no group forms a complete 2 x 2 (or larger) parameter grid"
        ),
        required_groups=required,
        available_groups=available,
        missing_arms=missing,
        routing_modes=_routing_modes(frame),
        notes=notes,
    )


CHECKS = {
    1: check_figure1,
    2: check_figure2,
    3: check_figure3,
    4: check_figure4,
    5: check_figure5,
}


# ------------------------------------------------------------------ driver --
def check_readiness(
    analysis_dir: str | Path,
    *,
    heatmap_metric: str = "accuracy",
    figures: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Build the readiness payload for an analysis artifact directory."""
    analysis_dir = Path(analysis_dir)
    tables = load_tables(analysis_dir)
    runs, results, routing = tables["runs"], tables["results"], tables["routing"]

    wanted = sorted(figures) if figures else sorted(CHECKS)
    reports: Dict[str, Any] = {}
    for number in wanted:
        check = CHECKS[number]
        if number == 5:
            report = check(runs, results, routing, metric=heatmap_metric)
        else:
            report = check(runs, results, routing)
        reports[str(number)] = report.as_dict()

    pear_modes = sorted(set(_routing_modes(routing)) & set(PEAR_ROUTING_MODES))
    payload = {
        "analysis_dir": str(analysis_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_git_commit": analysis_git_commit(),
        "n_runs": int(runs["run_id"].nunique()) if "run_id" in runs else 0,
        "n_run_conditions": int(len(runs)),
        "n_results_rows": int(len(results)),
        "n_routing_rows": int(len(routing)),
        "routing_modes": _routing_modes(routing),
        "state_aware_routing_modes": pear_modes,
        "methods": sorted({str(m) for m in runs.get("method", pd.Series(dtype=str)).dropna().unique()}),
        "heatmap_metric": heatmap_metric,
        "figures": reports,
    }
    return payload


def print_readiness(payload: Mapping[str, Any]) -> None:
    print(f"Figure readiness for {payload['analysis_dir']}")
    print(f"  runs: {payload['n_runs']}   methods: {payload['methods']}")
    print(
        f"  rows: results={payload['n_results_rows']} routing={payload['n_routing_rows']}   "
        f"routing modes: {payload['routing_modes'] or 'none'}"
    )
    for number, report in sorted(payload["figures"].items()):
        status = "READY" if report["ready"] else "NOT READY"
        print(f"\n  Figure {number}: {report['name']}")
        print(f"    {status} -- {report['reason']}")
        if report["missing_capabilities"]:
            print(f"    missing capabilities : {report['missing_capabilities']}")
        if report["missing_arms"]:
            print("    missing arms:")
            for arm in report["missing_arms"]:
                print(f"      - {arm}")
        if report["available_groups"]:
            print(f"    available groups ({len(report['available_groups'])}):")
            for group in report["available_groups"]:
                print(f"      - {json.dumps(group, sort_keys=True)}")
        if report["routing_modes"]:
            print(f"    routing modes        : {report['routing_modes']}")
        for note in report["notes"]:
            print(f"    note: {note}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("analysis_dir", help="Directory written by analysis/collect_runs.py")
    parser.add_argument(
        "--heatmap-metric",
        default="accuracy",
        choices=list(HEATMAP_METRICS),
        help="Metric assumed when judging Figure 5 readiness (default: %(default)s).",
    )
    parser.add_argument(
        "--figures",
        type=int,
        nargs="+",
        choices=sorted(CHECKS),
        help="Only report on these figures.",
    )
    args = parser.parse_args(argv)

    try:
        payload = check_readiness(
            args.analysis_dir, heatmap_metric=args.heatmap_metric, figures=args.figures
        )
    except RunLoadError as exc:
        print(f"readiness check failed: {exc}", file=sys.stderr)
        return 2

    print_readiness(payload)
    out = Path(args.analysis_dir) / "readiness.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
