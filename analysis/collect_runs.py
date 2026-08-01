"""Build the normalised analysis tables from validated run directories.

Usage
-----
    python analysis/collect_runs.py \\
        outputs/exp_a outputs/exp_b \\
        --output analysis_artifacts/my_analysis \\
        --allow-mock

Writes::

    analysis_artifacts/my_analysis/
    |-- manifest.json
    `-- tables/
        |-- runs.csv      one row per run x condition
        |-- results.csv   one row per run x condition x example x seed x perm_seed
        `-- routing.csv   one row per routing decision x agent

Validation runs first and hard-fails the collection: a table built from runs
that should not be pooled is worse than no table, because it looks usable.

These three CSVs are the only thing the figure scripts read. Nothing downstream
opens a JSONL file, so a new benchmark or model changes nothing below this
point, and a competitor system needs one adapter in ``analysis/common.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.common import (  # noqa: E402
    ANALYSIS_SCHEMA_VERSION,
    CAPABILITY_COLUMNS,
    ConditionMeta,
    NormalizedRun,
    RunLoadError,
    agent_ids_of,
    analysis_git_commit,
    attacker_ids_of,
    confidence_map,
    json_dumps_stable,
    routing_confidence_field,
)
from analysis.validate_runs import ValidationReport, print_report, validate_runs  # noqa: E402

RUNS_TABLE = "runs.csv"
RESULTS_TABLE = "results.csv"
ROUTING_TABLE = "routing.csv"
MANIFEST = "manifest.json"

RUNS_COLUMNS: Sequence[str] = (
    "run_id",
    "source_run_dir",
    "schema_version",
    "mock",
    "adapter",
    "method",
    "model",
    "dataset",
    "dataset_split",
    "condition",
    "mechanism",
    "base_topology",
    "n_agents",
    "rounds",
    "k_regular_degree",
    "routing_mode",
    "routing_temperature",
    "alpha_targeted_cross",
    "alpha_influence",
    "alpha_low_confidence",
    "influence_beta",
    "verification_mode",
    "attack_type",
    "attack_mode",
    "attack_value",
    "attacker_ids",
    "attacker_count",
    "adversarial_fraction",
    "seed_count",
    "perm_seed_count",
    "n_results",
    "n_routing_rows",
    *CAPABILITY_COLUMNS,
)

RESULTS_COLUMNS: Sequence[str] = (
    "run_id",
    "source_run_dir",
    "schema_version",
    "mock",
    "method",
    "model",
    "dataset",
    "dataset_split",
    "condition",
    "mechanism",
    "routing_mode",
    "verification_mode",
    "attack_type",
    "attack_mode",
    "attack_value",
    "attacker_ids",
    "attacker_count",
    "adversarial_fraction",
    "example_id",
    "seed",
    "perm_seed",
    "prediction",
    "correct",
    "parse_failures",
    "calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "n_messages",
    "answer_history",
    "confidence_history",
    "influence_history",
)

ROUTING_COLUMNS: Sequence[str] = (
    "source_run_dir",
    "run_id",
    "mock",
    "dataset",
    "dataset_split",
    "model",
    "method",
    "mechanism",
    "condition",
    "example_id",
    "seed",
    "perm_seed",
    "round",
    "agent_id",
    "is_attacker",
    "attack_type",
    "attack_mode",
    "attack_value",
    "clean_confidence",
    "reported_confidence",
    "g_i",
    "verified_confidence",
    "routing_confidence",
    "verification_mode",
    "out_degree",
    "in_degree",
    "out_degree_total",
    "out_degree_share",
    "zero_out_degree_total",
    "influence_before",
    "source_eligible",
    "target_eligible",
    "routing_mode",
    "candidate_count",
    "selected_score",
    "routing_temperature",
    "alpha_influence",
    "n_agents",
)


def load_tables(analysis_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Read the three normalised tables written by :func:`collect`.

    The single entry point for every figure script, so no figure ever opens a
    raw JSONL file and no two figures disagree about a column's dtype.
    """
    analysis_dir = Path(analysis_dir)
    tables_dir = analysis_dir / "tables"
    frames: Dict[str, pd.DataFrame] = {}
    for key, name in (("runs", RUNS_TABLE), ("results", RESULTS_TABLE), ("routing", ROUTING_TABLE)):
        path = tables_dir / name
        if not path.is_file():
            raise RunLoadError(
                f"missing normalised table {path}. Run analysis/collect_runs.py "
                f"first: python analysis/collect_runs.py <run dirs> --output {analysis_dir}"
            )
        frames[key] = pd.read_csv(path)
    return frames


@dataclass
class CollectionResult:
    """What the collector produced, for the master command and the tests."""

    output_dir: Path
    runs: pd.DataFrame
    results: pd.DataFrame
    routing: pd.DataFrame
    manifest: Dict[str, Any]
    validation: ValidationReport


# --------------------------------------------------------------- table builders --
def _condition_routing_rows(run: NormalizedRun, condition: str) -> List[Mapping[str, Any]]:
    return [r for r in (run.routing or []) if str(r.get("condition")) == condition]


def build_runs_table(runs: Sequence[NormalizedRun]) -> pd.DataFrame:
    """One row per run x condition, carrying the resolved condition metadata."""
    rows: List[Dict[str, Any]] = []
    for run in runs:
        for condition, meta in sorted(run.conditions.items()):
            routing_rows = _condition_routing_rows(run, condition)
            modes = sorted({str(r.get("routing_mode")) for r in routing_rows if r.get("routing_mode")})
            results_rows = [r for r in run.results if str(r.get("condition")) == condition]
            capabilities = dict(run.capabilities.as_dict())
            # Capabilities are per condition where they can differ: a run may
            # hold both a routed condition and a single-shot baseline.
            capabilities["has_routing"] = bool(routing_rows)
            capabilities["has_confidence_reports"] = any(
                isinstance(r.get("reported_confidence"), Mapping) for r in routing_rows
            )
            capabilities["has_accuracy"] = any(r.get("correct") is not None for r in results_rows)
            capabilities["has_token_usage"] = any(
                r.get("total_tokens") is not None for r in results_rows
            )
            capabilities["has_influence_history"] = any(
                r.get("influence_history") for r in results_rows
            )
            rows.append(
                {
                    "run_id": run.run_id,
                    "source_run_dir": str(run.run_dir),
                    "schema_version": run.schema_version,
                    "mock": run.mock,
                    "adapter": run.adapter,
                    "method": meta.method,
                    "model": meta.model or run.model,
                    "dataset": meta.dataset or run.dataset,
                    "dataset_split": meta.dataset_split or run.dataset_split,
                    "condition": condition,
                    "mechanism": meta.mechanism,
                    "base_topology": meta.base_topology,
                    "n_agents": meta.n_agents,
                    "rounds": meta.rounds,
                    "k_regular_degree": meta.k_regular_degree,
                    # Only when unambiguous: several modes in one condition is a
                    # validation error, so a blank here means "no routing logged".
                    "routing_mode": modes[0] if len(modes) == 1 else None,
                    "routing_temperature": meta.routing_temperature,
                    "alpha_targeted_cross": meta.alpha_targeted_cross,
                    "alpha_influence": meta.alpha_influence,
                    "alpha_low_confidence": meta.alpha_low_confidence,
                    "influence_beta": meta.influence_beta,
                    "verification_mode": meta.verification_mode,
                    "attack_type": meta.attack_type,
                    "attack_mode": meta.attack_mode,
                    "attack_value": meta.attack_value,
                    "attacker_ids": json_dumps_stable(list(meta.attacker_ids)),
                    "attacker_count": meta.attacker_count,
                    "adversarial_fraction": meta.adversarial_fraction,
                    "seed_count": len({r.get("seed") for r in results_rows}),
                    "perm_seed_count": len({r.get("perm_seed") for r in results_rows}),
                    "n_results": len(results_rows),
                    "n_routing_rows": len(routing_rows),
                    **capabilities,
                }
            )
    return pd.DataFrame(rows, columns=list(RUNS_COLUMNS))


def build_results_table(runs: Sequence[NormalizedRun]) -> pd.DataFrame:
    """One row per run x condition x example x seed x perm_seed."""
    rows: List[Dict[str, Any]] = []
    for run in runs:
        for record in run.results:
            row = {name: record.get(name) for name in RESULTS_COLUMNS}
            row["run_id"] = record.get("run_id") or run.run_id
            row["source_run_dir"] = str(run.run_dir)
            row["schema_version"] = run.schema_version
            for history in ("answer_history", "confidence_history", "influence_history"):
                row[history] = json_dumps_stable(record.get(history))
            # Kept beside attacker_count so a paired analysis knows *which*
            # agent deviated, not just how many did.
            row["attacker_ids"] = json_dumps_stable(record.get("attacker_ids") or [])
            rows.append(row)
    frame = pd.DataFrame(rows, columns=list(RESULTS_COLUMNS))
    # Nullable integer dtype: a run without token usage keeps <NA>, never 0.
    for column in ("prompt_tokens", "completion_tokens", "total_tokens", "calls", "n_messages"):
        if column in frame:
            frame[column] = frame[column].astype("Int64")
    return frame


def _eligibility(row: Mapping[str, Any], agent: int, field: str) -> Optional[bool]:
    block = row.get("targeted_cross_eligibility")
    if not isinstance(block, Mapping):
        return None
    entry = block.get(str(agent))
    if not isinstance(entry, Mapping) or field not in entry:
        return None
    return bool(entry[field])


def _degree(row: Mapping[str, Any], key: str, agent: int) -> Optional[float]:
    degrees = row.get(key)
    if not isinstance(degrees, (list, tuple)) or agent > len(degrees):
        return None
    value = degrees[agent - 1]
    return None if value is None else float(value)


def build_routing_table(runs: Sequence[NormalizedRun]) -> pd.DataFrame:
    """Expand every routing decision into one row per agent.

    ``out_degree_share`` is *routing exposure*: the share of the round's edges
    that originate at this agent, i.e. how much of the debate it gets to talk
    at. It is not accumulated influence -- PEAR's rho is carried separately as
    ``influence_before``, the value the router penalised when it chose this
    topology.
    """
    rows: List[Dict[str, Any]] = []
    for run in runs:
        for raw in run.routing or []:
            condition = str(raw.get("condition"))
            meta: Optional[ConditionMeta] = run.conditions.get(condition)
            verification_mode = str(raw.get("verified_confidence_mode") or "none")
            confidence_field = routing_confidence_field(verification_mode)
            attackers = set(attacker_ids_of(raw, meta))
            objective = raw.get("objective") if isinstance(raw.get("objective"), Mapping) else {}

            agents = agent_ids_of(raw)
            out_degrees = {a: _degree(raw, "out_degree", a) for a in agents}
            present = [d for d in out_degrees.values() if d is not None]
            total_out = sum(present) if present else None
            zero_total = bool(total_out is not None and total_out == 0)

            views = {
                name: confidence_map(raw, name)
                for name in ("clean_confidence", "reported_confidence", "g_i", "verified_confidence")
            }
            influence = confidence_map(raw, "influence") or {}

            for agent in agents:
                out_degree = out_degrees[agent]
                if total_out and out_degree is not None:
                    share = out_degree / total_out
                else:
                    # Zero (or unknown) total: flagged, never 0/0 -> 0.
                    share = None
                routing_confidence = (views.get(confidence_field) or {}).get(agent)
                rows.append(
                    {
                        "source_run_dir": str(run.run_dir),
                        "run_id": run.run_id,
                        "mock": bool(raw.get("mock")) if raw.get("mock") is not None else run.mock,
                        "dataset": (meta.dataset if meta else None) or run.dataset,
                        "dataset_split": (meta.dataset_split if meta else None) or run.dataset_split,
                        "model": (meta.model if meta else None) or run.model,
                        "method": meta.method if meta else run.method,
                        "mechanism": (meta.mechanism if meta else None) or raw.get("mode"),
                        "condition": condition,
                        "example_id": str(raw.get("example_id")),
                        "seed": raw.get("seed"),
                        "perm_seed": raw.get("perm_seed"),
                        "round": raw.get("round"),
                        "agent_id": agent,
                        "is_attacker": agent in attackers,
                        "attack_type": meta.attack_type if meta else None,
                        "attack_mode": raw.get("confidence_inflation_mode")
                        or (meta.attack_mode if meta else None),
                        "attack_value": raw.get("confidence_inflation_value")
                        if raw.get("confidence_inflation_value") is not None
                        else (meta.attack_value if meta else None),
                        "clean_confidence": (views["clean_confidence"] or {}).get(agent),
                        "reported_confidence": (views["reported_confidence"] or {}).get(agent),
                        "g_i": (views["g_i"] or {}).get(agent),
                        "verified_confidence": (views["verified_confidence"] or {}).get(agent),
                        "routing_confidence": routing_confidence,
                        "verification_mode": verification_mode,
                        "out_degree": out_degree,
                        "in_degree": _degree(raw, "in_degree", agent),
                        "out_degree_total": total_out,
                        "out_degree_share": share,
                        "zero_out_degree_total": zero_total,
                        "influence_before": influence.get(agent),
                        "source_eligible": _eligibility(raw, agent, "source_eligible"),
                        "target_eligible": _eligibility(raw, agent, "target_eligible"),
                        "routing_mode": raw.get("routing_mode"),
                        "candidate_count": raw.get("candidate_count"),
                        "selected_score": raw.get("selected_score"),
                        "routing_temperature": objective.get("routing_temperature")
                        if objective.get("routing_temperature") is not None
                        else (meta.routing_temperature if meta else None),
                        "alpha_influence": objective.get("alpha_influence")
                        if objective.get("alpha_influence") is not None
                        else (meta.alpha_influence if meta else None),
                        "n_agents": len(agents),
                    }
                )
    return pd.DataFrame(rows, columns=list(ROUTING_COLUMNS))


# ------------------------------------------------------------------- collect --
def collect(
    paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    allow_mock: bool = False,
    allowed_schema_versions: Optional[Iterable[int]] = None,
    command: Optional[str] = None,
    filters: Optional[Mapping[str, Any]] = None,
    quiet: bool = False,
) -> CollectionResult:
    """Validate, normalise and write the three tables plus the manifest."""
    report = validate_runs(
        paths, allow_mock=allow_mock, allowed_schema_versions=allowed_schema_versions
    )
    if not quiet:
        print_report(report)
    if not report.ok:
        raise RunLoadError(
            f"{len(report.errors())} validation error(s); refusing to build tables. "
            "Fix the runs, or narrow the input paths."
        )

    runs = report.valid_runs
    runs_table = build_runs_table(runs)
    results_table = build_results_table(runs)
    routing_table = build_routing_table(runs)

    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    runs_table.to_csv(tables_dir / RUNS_TABLE, index=False)
    results_table.to_csv(tables_dir / RESULTS_TABLE, index=False)
    routing_table.to_csv(tables_dir / ROUTING_TABLE, index=False)

    manifest = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "analysis_command": command or " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        # The commit of the *analysis* code. The runner does not record the
        # commit it ran at, so this must not be read as the commit that
        # generated the experiments.
        "analysis_git_commit": analysis_git_commit(),
        "experiment_git_commit": None,
        "experiment_git_commit_note": (
            "not recorded: the experiment runner does not log its own commit, so "
            "the analysis commit above is not the generating commit"
        ),
        "input_paths": [str(p) for p in paths],
        "discovered_run_dirs": [str(r.run_dir) for r in runs],
        "n_runs": len(runs),
        "n_run_conditions": int(len(runs_table)),
        "log_schema_versions": report.schema_versions,
        "mock_status": report.mock_status,
        "allow_mock": bool(allow_mock),
        "routing_modes": report.routing_modes,
        "n_results_rows": int(len(results_table)),
        "n_routing_rows": int(len(routing_table)),
        "filters": dict(filters or {}),
        "validation": {
            "ok": report.ok,
            "n_errors": len(report.errors()),
            "n_warnings": len(report.warnings()),
            "warnings": [i.render() for i in report.warnings()],
        },
        "tables": {
            "runs": str(tables_dir / RUNS_TABLE),
            "results": str(tables_dir / RESULTS_TABLE),
            "routing": str(tables_dir / ROUTING_TABLE),
        },
    }
    (output_dir / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not quiet:
        print(f"\nwrote {tables_dir / RUNS_TABLE} ({len(runs_table)} rows)")
        print(f"wrote {tables_dir / RESULTS_TABLE} ({len(results_table)} rows)")
        print(f"wrote {tables_dir / ROUTING_TABLE} ({len(routing_table)} rows)")
        print(f"wrote {output_dir / MANIFEST}")

    return CollectionResult(
        output_dir=output_dir,
        runs=runs_table,
        results=results_table,
        routing=routing_table,
        manifest=manifest,
        validation=report,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dirs", nargs="+", help="Run directories, or directories containing them.")
    parser.add_argument("--output", required=True, help="Analysis artifact directory to write.")
    parser.add_argument("--allow-mock", action="store_true", help="Permit mock runs.")
    parser.add_argument(
        "--allow-schema-version", type=int, action="append", default=[], metavar="N"
    )
    args = parser.parse_args(argv)

    from analysis.common import SUPPORTED_SCHEMA_VERSIONS

    allowed = set(SUPPORTED_SCHEMA_VERSIONS) | set(args.allow_schema_version)
    try:
        collect(
            args.run_dirs,
            args.output,
            allow_mock=args.allow_mock,
            allowed_schema_versions=allowed,
            command=" ".join(["python analysis/collect_runs.py", *(argv or sys.argv[1:])]),
        )
    except RunLoadError as exc:
        print(f"\ncollection failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
