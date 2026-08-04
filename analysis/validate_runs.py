"""Integrity checks for run directories, before anything is aggregated.

Everything here answers one question: *can these runs be pooled into a single
analysis without changing what a number means?* A file that parses is not
enough -- pooling a v1 run with a v2 run, or sampled routing with enumerated
routing, produces a plausible-looking table that answers a question nobody
asked.

Usage
-----
    python analysis/validate_runs.py RUN_DIR [RUN_DIR ...]
    python analysis/validate_runs.py outputs/exp_a outputs/exp_b
    python analysis/validate_runs.py outputs/ --json report.json

Exit code is 0 only when there are no errors. Warnings never fail the run;
they mark things a figure will have to split on or skip.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.common import (  # noqa: E402
    ROUTING_FILE,
    ROUTING_MODES,
    REQUIRED_RUN_FILES,
    SUPPORTED_SCHEMA_VERSIONS,
    NormalizedRun,
    RunLoadError,
    agent_ids_of,
    condition_names,
    discover_runs,
    is_finite_number,
    load_normalized_run,
    run_id_for,
)

ERROR = "error"
WARNING = "warning"

#: Modes for which the runner forces ``rounds = 0``: single-shot baselines that
#: never route, so expecting routing rows for them would be a false alarm.
NO_ROUND_MECHANISMS = {"cot", "cot_sc"}


@dataclass
class Issue:
    """One finding. ``code`` is stable enough to assert on in tests."""

    level: str
    code: str
    message: str
    run_id: Optional[str] = None

    def render(self) -> str:
        where = f"[{self.run_id}] " if self.run_id else ""
        return f"{self.level.upper():7} {where}{self.code}: {self.message}"


@dataclass
class RunReport:
    """Per-run outcome, plus the loaded run when it could be loaded at all."""

    run_dir: str
    run_id: str
    adapter: Optional[str] = None
    schema_version: Optional[int] = None
    routing_modes: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    n_results: int = 0
    n_routing_rows: int = 0
    capabilities: Dict[str, bool] = field(default_factory=dict)
    issues: List[Issue] = field(default_factory=list)
    run: Optional[NormalizedRun] = None

    @property
    def ok(self) -> bool:
        return not any(i.level == ERROR for i in self.issues)

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            k: v for k, v in asdict(self).items() if k not in {"run", "issues"}
        }
        payload["issues"] = [asdict(i) for i in self.issues]
        payload["ok"] = self.ok
        return payload


@dataclass
class ValidationReport:
    """Aggregate outcome across every validated run."""

    runs: List[RunReport] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    schema_versions: List[int] = field(default_factory=list)
    routing_modes: List[str] = field(default_factory=list)

    @property
    def all_issues(self) -> List[Issue]:
        return [i for r in self.runs for i in r.issues] + list(self.issues)

    @property
    def ok(self) -> bool:
        return not any(i.level == ERROR for i in self.all_issues)

    @property
    def valid_runs(self) -> List[NormalizedRun]:
        return [r.run for r in self.runs if r.ok and r.run is not None]

    def errors(self) -> List[Issue]:
        return [i for i in self.all_issues if i.level == ERROR]

    def warnings(self) -> List[Issue]:
        return [i for i in self.all_issues if i.level == WARNING]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "schema_versions": self.schema_versions,
            "routing_modes": self.routing_modes,
            "n_runs": len(self.runs),
            "n_errors": len(self.errors()),
            "n_warnings": len(self.warnings()),
            "runs": [r.as_dict() for r in self.runs],
            "global_issues": [asdict(i) for i in self.issues],
        }


# ------------------------------------------------------------------ helpers --
def _key(row: Mapping[str, Any], fields: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(row.get(f) for f in fields)


def _duplicate_keys(
    rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> List[Tuple[Tuple[Any, ...], int]]:
    counts = Counter(_key(row, fields) for row in rows)
    return sorted(
        ((key, n) for key, n in counts.items() if n > 1), key=lambda kv: str(kv[0])
    )


def _effective_rounds(mechanism: Optional[str], rounds: Optional[int]) -> int:
    if str(mechanism or "") in NO_ROUND_MECHANISMS:
        return 0
    return int(rounds or 0)


# -------------------------------------------------------------- single run --
def validate_run(run_dir: Path, *, allowed_schema_versions: Set[int]) -> RunReport:
    """Validate one run directory in isolation."""
    report = RunReport(run_dir=str(run_dir), run_id=run_id_for(run_dir))
    add = report.issues.append

    for name in REQUIRED_RUN_FILES:
        if not (run_dir / name).is_file():
            add(Issue(ERROR, "missing_file", f"required file not found: {name}", report.run_id))
    if not report.ok:
        return report

    try:
        run = load_normalized_run(run_dir)
    except RunLoadError as exc:
        add(Issue(ERROR, "load_failed", str(exc), report.run_id))
        return report

    report.run = run
    report.adapter = run.adapter
    report.schema_version = run.schema_version
    report.routing_modes = run.routing_modes
    report.conditions = sorted(run.conditions)
    report.n_results = len(run.results)
    report.n_routing_rows = len(run.routing or [])
    report.capabilities = run.capabilities.as_dict()

    _check_schema(run, report, allowed_schema_versions)
    _check_required_fields(run, report)
    _check_duplicates(run, report)
    _check_routing(run, report, allowed_schema_versions)
    _check_coverage(run, report)
    return report


def _check_schema(
    run: NormalizedRun, report: RunReport, allowed: Set[int]
) -> None:
    add = report.issues.append
    if run.schema_version is None:
        add(
            Issue(
                ERROR,
                "schema_version_missing",
                "summary.json carries no schema_version; this run predates log "
                "versioning and its fields cannot be interpreted with confidence. "
                "Re-run it, or opt in explicitly with --allow-schema-version.",
                report.run_id,
            )
        )
    elif run.schema_version not in allowed:
        add(
            Issue(
                ERROR,
                "schema_version_unsupported",
                f"schema_version={run.schema_version} is not supported by this "
                f"analysis code (supported: {sorted(allowed)}). Field meanings "
                "differ between versions; opt in with --allow-schema-version "
                "only after checking the readers.",
                report.run_id,
            )
        )


def _check_required_fields(run: NormalizedRun, report: RunReport) -> None:
    """Required numeric fields must be real, finite numbers.

    Strict JSON already rejects bare NaN tokens; this catches the other way a
    non-number arrives -- a null or a string where a count belongs.
    """
    add = report.issues.append
    bad_tokens = 0
    missing_ids = 0
    for row in run.results:
        if row.get("example_id") in (None, "", "None"):
            missing_ids += 1
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = row.get(name)
            if value is not None and not is_finite_number(value):
                bad_tokens += 1
    if missing_ids:
        add(
            Issue(
                ERROR,
                "result_missing_example_id",
                f"{missing_ids} result row(s) carry no example_id; they cannot "
                "be joined, paired or clustered",
                report.run_id,
            )
        )
    if bad_tokens:
        add(
            Issue(
                ERROR,
                "token_usage_non_finite",
                f"{bad_tokens} token-usage field(s) are present but not finite "
                "numbers",
                report.run_id,
            )
        )


def _check_duplicates(run: NormalizedRun, report: RunReport) -> None:
    add = report.issues.append
    result_keys = ("condition", "example_id", "seed", "perm_seed")
    duplicates = _duplicate_keys(run.results, result_keys)
    if duplicates:
        preview = ", ".join(f"{k}x{n}" for k, n in duplicates[:5])
        add(
            Issue(
                ERROR,
                "duplicate_results",
                f"{len(duplicates)} duplicated (run, condition, example_id, seed, "
                f"perm_seed) key(s) in results.jsonl: {preview}"
                + (" ..." if len(duplicates) > 5 else ""),
                report.run_id,
            )
        )
    if run.routing:
        routing_keys = ("condition", "example_id", "seed", "perm_seed", "round")
        duplicates = _duplicate_keys(run.routing, routing_keys)
        if duplicates:
            preview = ", ".join(f"{k}x{n}" for k, n in duplicates[:5])
            add(
                Issue(
                    ERROR,
                    "duplicate_routing",
                    f"{len(duplicates)} duplicated (run, condition, example_id, "
                    f"seed, perm_seed, round) key(s) in routing.jsonl: {preview}"
                    + (" ..." if len(duplicates) > 5 else ""),
                    report.run_id,
                )
            )


def _check_routing(
    run: NormalizedRun, report: RunReport, allowed: Set[int]
) -> None:
    """Routing-log checks: schema agreement, mode separation, confidence maps."""
    add = report.issues.append

    if run.routing is None:
        expects_routing = any(
            _effective_rounds(meta.mechanism, meta.rounds) > 0
            for meta in run.conditions.values()
        )
        if run.adapter == "pear" and expects_routing:
            add(
                Issue(
                    WARNING,
                    "routing_log_absent",
                    f"no {ROUTING_FILE}: accuracy and cost analysis still work, "
                    "routing analysis is unavailable for this run "
                    "(has_routing=False)",
                    report.run_id,
                )
            )
        return

    if not run.routing:
        add(
            Issue(
                ERROR,
                "routing_log_empty",
                f"{ROUTING_FILE} exists but contains no rows",
                report.run_id,
            )
        )
        return

    versions = sorted({r.get("schema_version") for r in run.routing})
    if versions != [run.schema_version]:
        add(
            Issue(
                ERROR,
                "routing_schema_mismatch",
                f"routing rows carry schema_version(s) {versions} but the run "
                f"declares {run.schema_version}",
                report.run_id,
            )
        )

    unknown_modes = sorted(
        {
            str(r.get("routing_mode"))
            for r in run.routing
            if r.get("routing_mode") not in (None, "")
        }
        - set(ROUTING_MODES)
    )
    if unknown_modes:
        add(
            Issue(
                WARNING,
                "routing_mode_unknown",
                f"unrecognised routing_mode(s) {unknown_modes}; analysis will "
                "keep them separate but cannot say what mechanism they are",
                report.run_id,
            )
        )
    if any(r.get("routing_mode") in (None, "") for r in run.routing):
        add(
            Issue(
                ERROR,
                "routing_mode_missing",
                "routing rows without a routing_mode: a state-aware decision "
                "cannot be told from an identity relabelling",
                report.run_id,
            )
        )

    per_condition: Dict[str, Set[str]] = defaultdict(set)
    for row in run.routing:
        per_condition[str(row.get("condition"))].add(str(row.get("routing_mode")))
    for condition, modes in sorted(per_condition.items()):
        if len(modes) > 1:
            add(
                Issue(
                    ERROR,
                    "routing_mode_mixed_within_condition",
                    f"condition {condition!r} mixes routing modes {sorted(modes)}; "
                    "these are different mechanisms and cannot be pooled or split "
                    "apart after the fact",
                    report.run_id,
                )
            )

    _check_confidence_maps(run, report)


def _check_confidence_maps(run: NormalizedRun, report: RunReport) -> None:
    """Every schema-v2 routing row must carry a complete confidence picture.

    An incomplete map is worse than a missing one: the mean over the agents
    that happen to be present looks like a real number.
    """
    add = report.issues.append
    problems: Counter[str] = Counter()
    examples: Dict[str, str] = {}

    for row in run.routing or []:
        if row.get("schema_version") != 2:
            continue
        agents = agent_ids_of(row)
        where = (
            f"condition={row.get('condition')} example={row.get('example_id')} "
            f"seed={row.get('seed')} perm_seed={row.get('perm_seed')} "
            f"round={row.get('round')}"
        )
        for name in ("clean_confidence", "reported_confidence"):
            view = row.get(name)
            if not isinstance(view, Mapping):
                problems[f"{name}_absent"] += 1
                examples.setdefault(f"{name}_absent", where)
                continue
            missing = [a for a in agents if view.get(str(a)) is None]
            if missing:
                problems[f"{name}_incomplete"] += 1
                examples.setdefault(f"{name}_incomplete", f"{where} agents={missing}")
            non_finite = [
                a for a in agents if a not in missing and not is_finite_number(view.get(str(a)))
            ]
            if non_finite:
                problems[f"{name}_non_finite"] += 1
                examples.setdefault(f"{name}_non_finite", f"{where} agents={non_finite}")

        mode = str(row.get("verified_confidence_mode") or "none").strip().lower()
        for name in ("g_i", "verified_confidence"):
            view = row.get(name)
            if mode == "none":
                if view is not None:
                    problems[f"{name}_present_without_verification"] += 1
                    examples.setdefault(f"{name}_present_without_verification", where)
                continue
            if not isinstance(view, Mapping):
                problems[f"{name}_absent_under_verification"] += 1
                examples.setdefault(f"{name}_absent_under_verification", where)
                continue
            missing = [a for a in agents if view.get(str(a)) is None]
            if missing:
                problems[f"{name}_incomplete"] += 1
                examples.setdefault(f"{name}_incomplete", f"{where} agents={missing}")

    for code, count in sorted(problems.items()):
        add(
            Issue(
                ERROR,
                f"confidence_{code}",
                f"{count} routing row(s); first at {examples[code]}",
                report.run_id,
            )
        )


def _check_coverage(run: NormalizedRun, report: RunReport) -> None:
    """Expected condition x example x seed x perm_seed (x round) combinations.

    Example ids come from the rows, because the config records only how many
    examples to take, not which ones. Seeds and perm seeds come from the config:
    those the config does pin down, so a missing cell is a real gap rather than
    an inference.
    """
    add = report.issues.append
    config = run.config
    replication = config.get("replication") or {}
    seeds = list(replication.get("seeds") or [])
    perm_seeds = list(replication.get("agent_perm_seeds") or [])
    declared = condition_names(config) if config.get("conditions") is not None else []
    if not declared:
        declared = sorted(run.conditions)

    if not seeds or not perm_seeds:
        add(
            Issue(
                WARNING,
                "coverage_unverifiable",
                "config does not pin down replication.seeds / "
                "replication.agent_perm_seeds, so expected coverage cannot be "
                "checked; only duplicates were verified",
                report.run_id,
            )
        )
        return

    example_ids = sorted({str(r.get("example_id")) for r in run.results})
    if not example_ids:
        add(Issue(ERROR, "no_results", "results.jsonl has no rows", report.run_id))
        return

    expected = {
        (c, e, s, p)
        for c, e, s, p in product(declared, example_ids, seeds, perm_seeds)
    }
    observed = {
        (
            str(r.get("condition")),
            str(r.get("example_id")),
            r.get("seed"),
            r.get("perm_seed"),
        )
        for r in run.results
    }
    _report_set_difference(
        add,
        report.run_id,
        expected,
        observed,
        missing_code="coverage_missing_results",
        extra_code="coverage_unexpected_results",
        label="condition x example x seed x perm_seed",
    )

    num_examples = (config.get("dataset") or {}).get("num_examples")
    if isinstance(num_examples, int) and num_examples > 0 and len(example_ids) != num_examples:
        add(
            Issue(
                WARNING,
                "coverage_example_count",
                f"config asks for {num_examples} examples but results carry "
                f"{len(example_ids)}",
                report.run_id,
            )
        )

    if not run.routing:
        return

    routed_conditions = {str(row.get("condition")) for row in run.routing}
    expected_routing = set()
    for condition in declared:
        meta = run.conditions.get(condition)
        if meta is None:
            continue
        # A condition that logged no routing at all does not route -- a method
        # with a fixed communication graph has no decision to record (see
        # baselines/). That is a capability difference, not missing coverage,
        # and demanding rows from it would make every mixed run invalid. A
        # condition that logged *some* rows is still held to all of them.
        if condition not in routed_conditions:
            continue
        rounds = _effective_rounds(meta.mechanism, meta.rounds)
        for e, s, p, r in product(example_ids, seeds, perm_seeds, range(1, rounds + 1)):
            expected_routing.add((condition, e, s, p, r))
    observed_routing = {
        (
            str(row.get("condition")),
            str(row.get("example_id")),
            row.get("seed"),
            row.get("perm_seed"),
            row.get("round"),
        )
        for row in run.routing
    }
    _report_set_difference(
        add,
        report.run_id,
        expected_routing,
        observed_routing,
        missing_code="coverage_missing_routing",
        extra_code="coverage_unexpected_routing",
        label="condition x example x seed x perm_seed x round",
    )


def _report_set_difference(
    add,
    run_id: str,
    expected: Set[Tuple[Any, ...]],
    observed: Set[Tuple[Any, ...]],
    *,
    missing_code: str,
    extra_code: str,
    label: str,
) -> None:
    missing = sorted(expected - observed, key=str)
    extra = sorted(observed - expected, key=str)
    if missing:
        preview = "; ".join(str(k) for k in missing[:5])
        add(
            Issue(
                ERROR,
                missing_code,
                f"{len(missing)} missing {label} combination(s): {preview}"
                + (" ..." if len(missing) > 5 else ""),
                run_id,
            )
        )
    if extra:
        preview = "; ".join(str(k) for k in extra[:5])
        add(
            Issue(
                ERROR,
                extra_code,
                f"{len(extra)} unexpected {label} combination(s) not implied by "
                f"the config: {preview}" + (" ..." if len(extra) > 5 else ""),
                run_id,
            )
        )


# ------------------------------------------------------------- across runs --
def _cross_run_checks(report: ValidationReport) -> None:
    add = report.issues.append
    loaded = [r for r in report.runs if r.run is not None]

    versions = sorted({r.schema_version for r in loaded if r.schema_version is not None})
    report.schema_versions = versions
    if len(versions) > 1:
        add(
            Issue(
                ERROR,
                "schema_versions_mixed",
                f"inputs mix log schema versions {versions}. Field meanings "
                "differ between versions -- analyse them separately.",
            )
        )

    # Same benchmark split, different corpus bytes: the runs did not answer the
    # same questions. The checksum is stamped on each run at load time, so this
    # is detectable here even though nothing in the results rows would show it.
    corpora: Dict[Tuple[Any, Any], Set[str]] = defaultdict(set)
    for entry in loaded:
        run = entry.run
        if run is not None and run.dataset_sha256:
            corpora[(run.dataset, run.dataset_split)].add(run.dataset_sha256)
    for (dataset, split), checksums in sorted(corpora.items(), key=str):
        if len(checksums) > 1:
            add(
                Issue(
                    ERROR,
                    "dataset_corpus_mismatch",
                    f"runs on {dataset}/{split} scored {len(checksums)} different "
                    f"corpus files (sha256 {sorted(c[:12] for c in checksums)}). "
                    "They did not answer the same questions, so their accuracies "
                    "are not comparable. Re-fetch the split and re-run, or "
                    "analyse the groups separately.",
                )
            )

    modes = sorted({m for r in loaded for m in r.routing_modes})
    report.routing_modes = modes
    if len(modes) > 1:
        add(
            Issue(
                WARNING,
                "routing_modes_multiple",
                f"inputs contain several routing modes {modes}; figures must "
                "split on routing_mode rather than pool them",
            )
        )


def validate_runs(
    paths: Sequence[str | Path],
    *,
    allowed_schema_versions: Optional[Iterable[int]] = None,
) -> ValidationReport:
    """Discover and validate every run under ``paths``."""
    allowed = set(allowed_schema_versions or SUPPORTED_SCHEMA_VERSIONS)
    report = ValidationReport()
    run_dirs = discover_runs(paths)
    if not run_dirs:
        report.issues.append(
            Issue(
                ERROR,
                "no_runs_found",
                f"no run directories found under {[str(p) for p in paths]}. A run "
                f"directory is one containing {REQUIRED_RUN_FILES}.",
            )
        )
        return report
    for run_dir in run_dirs:
        report.runs.append(validate_run(run_dir, allowed_schema_versions=allowed))
    _cross_run_checks(report)
    return report


# ------------------------------------------------------------------ printing --
def print_report(report: ValidationReport) -> None:
    print(f"Validated {len(report.runs)} run director{'y' if len(report.runs) == 1 else 'ies'}")
    print(f"  schema versions : {report.schema_versions or 'none'}")
    print(f"  routing modes   : {report.routing_modes or 'none logged'}")
    print()
    for run in report.runs:
        status = "OK   " if run.ok else "FAIL "
        caps = ",".join(sorted(k for k, v in run.capabilities.items() if v)) or "none"
        print(f"  {status} {run.run_id}")
        print(
            f"        adapter={run.adapter} schema={run.schema_version} "
            f"conditions={run.conditions}"
        )
        print(
            f"        results={run.n_results} routing_rows={run.n_routing_rows} "
            f"routing_modes={run.routing_modes or '-'}"
        )
        print(f"        capabilities: {caps}")
        for issue in run.issues:
            print(f"        {issue.render()}")
    if report.issues:
        print("\n  across runs:")
        for issue in report.issues:
            print(f"        {issue.render()}")
    print()
    print(f"{len(report.errors())} error(s), {len(report.warnings())} warning(s)")
    print("VALID" if report.ok else "INVALID")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dirs", nargs="+", help="Run directories, or directories containing them.")
    parser.add_argument(
        "--allow-schema-version",
        type=int,
        action="append",
        default=[],
        metavar="N",
        help=(
            "Additionally accept this log schema version. Repeatable. Use only "
            "after checking that the readers interpret that version correctly."
        ),
    )
    parser.add_argument("--json", dest="json_out", help="Also write the report as JSON here.")
    args = parser.parse_args(argv)

    allowed = set(SUPPORTED_SCHEMA_VERSIONS) | set(args.allow_schema_version)
    try:
        report = validate_runs(
            args.run_dirs, allowed_schema_versions=allowed
        )
    except RunLoadError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 2

    print_report(report)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
