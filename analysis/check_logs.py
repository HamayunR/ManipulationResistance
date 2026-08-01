"""Per-round routing report from one or more run directories.

Reads ``routing.jsonl`` and prints, per round, the quantities needed to see
whether a confidence-reporting attack bought its agent any routing influence:

* total out-degree, and each agent's out-degree and share of it;
* each agent's clean / reported / verified confidence;
* each agent's targeted-cross source and target eligibility.

Out-degree is the quantity to watch: ``core.topology.score_state_permutation``
scores *source out-degree* as structural exposure, so it is what an inflated
report is trying to buy.

Rows are aggregated across seeds and perm_seeds, because PEAR samples a
topology per round -- a single perm_seed says nothing about expected influence.
Numeric columns are means over the aggregated rows; boolean columns print as a
share of rows when they are not constant, so a flag that varies is visible
rather than silently averaged away.

Usage
-----
    python analysis/check_logs.py RUN_DIR [RUN_DIR ...]
    python analysis/check_logs.py outputs/exp_*/  --condition pear_full
    python analysis/check_logs.py RUN_DIR --example dummy-2 --label attacked

Multiple run directories are pooled into one report, so a 20-perm-seed sweep
spread over several runs reads the same as a single run containing all of them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ROUTING_FILE = "routing.jsonl"


# ------------------------------------------------------------------ loading --
def load_rows(run_dirs: Sequence[str]) -> List[Dict[str, Any]]:
    """Read every routing row from every run directory.

    A directory holding several runs (e.g. ``outputs/exp_.../``) is walked, so
    both a single run dir and an experiment dir work.
    """
    rows: List[Dict[str, Any]] = []
    seen: List[Path] = []
    for raw in run_dirs:
        base = Path(raw)
        if not base.exists():
            raise SystemExit(f"No such path: {base}")
        candidates = [base / ROUTING_FILE] if (base / ROUTING_FILE).exists() else sorted(
            base.rglob(ROUTING_FILE)
        )
        if not candidates:
            raise SystemExit(f"No {ROUTING_FILE} found under {base}")
        for path in candidates:
            seen.append(path)
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    if not rows:
        raise SystemExit("No routing rows found.")
    print(f"Read {len(rows)} routing rows from {len(seen)} file(s):")
    for path in seen:
        print(f"  {path}")
    versions = sorted({row.get("schema_version") for row in rows})
    print(f"schema_version(s): {versions}")
    if len(versions) > 1:
        print("  WARNING: pooling rows across schema versions; field meanings may differ.")
    return rows


def filter_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    condition: Optional[str],
    example: Optional[str],
) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        if condition and row.get("condition") != condition:
            continue
        if example and row.get("example_id") != example:
            continue
        out.append(dict(row))
    return out


# ---------------------------------------------------------------- accessors --
def agent_ids(row: Mapping[str, Any]) -> List[int]:
    n = int(row.get("n_agents") or len(row.get("out_degree") or []))
    return list(range(1, n + 1))


def out_degree_of(row: Mapping[str, Any], agent_id: int) -> Optional[float]:
    degrees = row.get("out_degree")
    if not degrees or agent_id > len(degrees):
        return None
    return float(degrees[agent_id - 1])


def confidence_of(row: Mapping[str, Any], field: str, agent_id: int) -> Optional[float]:
    """Read one confidence view. Returns None when the view does not apply.

    ``g_i`` and ``verified_confidence`` are null on runs with no verification;
    that is a real distinction from zero and is preserved as None here.
    """
    view = row.get(field)
    if not isinstance(view, Mapping):
        return None
    value = view.get(str(agent_id))
    return None if value is None else float(value)


def eligibility_of(row: Mapping[str, Any], field: str, agent_id: int) -> Optional[bool]:
    elig = row.get("targeted_cross_eligibility")
    if not isinstance(elig, Mapping):
        return None
    entry = elig.get(str(agent_id))
    if not isinstance(entry, Mapping) or field not in entry:
        return None
    return bool(entry[field])


# --------------------------------------------------------------- formatting --
def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def fmt_num(values: Sequence[Optional[float]], places: int = 3) -> str:
    present = [v for v in values if v is not None]
    if not present:
        return "-"
    return f"{mean(present):.{places}f}"


def fmt_bool(values: Sequence[Optional[bool]]) -> str:
    """Constant flags print as yes/no; varying ones print as a share."""
    present = [v for v in values if v is not None]
    if not present:
        return "-"
    if all(present):
        return "yes"
    if not any(present):
        return "no"
    return f"{sum(present) / len(present):.0%} yes"


def describe_scope(rows: Sequence[Mapping[str, Any]]) -> str:
    seeds = sorted({row.get("seed") for row in rows})
    perms = sorted({row.get("perm_seed") for row in rows})
    modes = sorted({str(row.get("routing_mode")) for row in rows})
    scope = (
        f"{len(rows)} rows | seeds={seeds} | {len(perms)} perm_seeds | "
        f"routing_mode={','.join(modes)}"
    )
    if len(modes) > 1:
        scope += "  <-- WARNING: mixed routing modes are different mechanisms"
    return scope


# ------------------------------------------------------------------ report ---
HEADER = (
    f"{'agent':>5}  {'out-deg':>8}  {'share':>7}  {'clean':>7}  {'reported':>8}  "
    f"{'verified':>8}  {'src_elig':>9}  {'tgt_elig':>9}"
)


def print_agent_table(rows: Sequence[Mapping[str, Any]], indent: str = "  ") -> None:
    """One line per agent, aggregated over the rows handed in."""
    if not rows:
        return
    ids = agent_ids(rows[0])

    per_row_total = []
    for row in rows:
        degrees = [out_degree_of(row, i) for i in ids]
        present = [d for d in degrees if d is not None]
        per_row_total.append(sum(present) if present else math.nan)
    total_mean = mean([t for t in per_row_total if not math.isnan(t)])

    print(f"{indent}total out-degree: {total_mean:.3f}")
    print(indent + HEADER)
    for agent in ids:
        degrees = [out_degree_of(row, agent) for row in rows]
        shares = []
        for row, total in zip(rows, per_row_total):
            degree = out_degree_of(row, agent)
            if degree is not None and total and not math.isnan(total):
                shares.append(degree / total)
        print(
            f"{indent}{agent:>5}  {fmt_num(degrees):>8}  {fmt_num(shares):>7}  "
            f"{fmt_num([confidence_of(r, 'clean_confidence', agent) for r in rows], 2):>7}  "
            f"{fmt_num([confidence_of(r, 'reported_confidence', agent) for r in rows], 2):>8}  "
            f"{fmt_num([confidence_of(r, 'verified_confidence', agent) for r in rows], 2):>8}  "
            f"{fmt_bool([eligibility_of(r, 'source_eligible', agent) for r in rows]):>9}  "
            f"{fmt_bool([eligibility_of(r, 'target_eligible', agent) for r in rows]):>9}"
        )


def report(rows: Sequence[Mapping[str, Any]], label: str = "") -> None:
    title = f"RUN SET{': ' + label if label else ''}"
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")

    attack_modes = sorted({str(row.get("confidence_inflation_mode")) for row in rows})
    attacked = sorted({tuple(row.get("confidence_inflation_agent_ids") or []) for row in rows})
    verified_modes = sorted({str(row.get("verified_confidence_mode")) for row in rows})
    print(f"confidence_inflation: mode={attack_modes} agent_ids={[list(a) for a in attacked]}")
    print(f"verified_confidence : mode={verified_modes}")

    groups: Dict[Any, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("condition"), row.get("example_id"), row.get("round"))].append(row)

    conditions = sorted({key[0] for key in groups}, key=str)
    for condition in conditions:
        examples = sorted({key[1] for key in groups if key[0] == condition}, key=str)
        for example in examples:
            rounds = sorted(
                key[2] for key in groups if key[0] == condition and key[1] == example
            )
            subset = [
                row
                for row in rows
                if row.get("condition") == condition and row.get("example_id") == example
            ]
            print(f"\n--- condition={condition}  example={example} ---")
            print(f"    {describe_scope(subset)}")
            for round_idx in rounds:
                bucket = groups[(condition, example, round_idx)]
                print(f"\n  round {round_idx}   ({len(bucket)} rows)")
                print_agent_table(bucket, indent="    ")
            print(f"\n  ALL ROUNDS   ({len(subset)} rows)")
            print_agent_table(subset, indent="    ")

        pooled = [row for row in rows if row.get("condition") == condition]
        if len({key[1] for key in groups if key[0] == condition}) > 1:
            print(f"\n### condition={condition}: ALL EXAMPLES, ALL ROUNDS ###")
            print(f"    {describe_scope(pooled)}")
            print_agent_table(pooled, indent="    ")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dirs", nargs="+", help="Run directories (or dirs containing them).")
    parser.add_argument("--condition", help="Keep only this condition name.")
    parser.add_argument("--example", help="Keep only this example id.")
    parser.add_argument("--label", default="", help="Label for the report header.")
    args = parser.parse_args(argv)

    rows = filter_rows(
        load_rows(args.run_dirs), condition=args.condition, example=args.example
    )
    if not rows:
        raise SystemExit("No rows left after filtering.")
    report(rows, label=args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
