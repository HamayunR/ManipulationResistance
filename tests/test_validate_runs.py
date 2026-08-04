"""Tests for ``analysis/validate_runs.py``.

Each test builds a run directory that is broken in exactly one way, because
the validator's job is to name *which* thing is wrong; a check that only says
"invalid" does not help anyone fix a sweep.
"""

from __future__ import annotations

import json
from pathlib import Path

from analysis.validate_runs import main, validate_runs
from tests.analysis_fixtures import (
    corrupt_jsonl,
    duplicate_last_line,
    make_generic_run,
    make_pear_run,
    patch_summary,
    read_jsonl_raw,
    rewrite_jsonl,
)


def codes(report) -> set[str]:
    return {issue.code for issue in report.all_issues}


def error_codes(report) -> set[str]:
    return {issue.code for issue in report.errors()}


# ---------------------------------------------------------------- happy path --
def test_clean_pear_run_validates(tmp_path: Path) -> None:
    make_pear_run(tmp_path / "exp", "clean")

    report = validate_runs([tmp_path])

    assert report.ok, [i.render() for i in report.errors()]
    assert report.schema_versions == [2]
    assert report.routing_modes == ["sampled"]
    assert report.runs[0].capabilities["has_routing"] is True


def test_attacked_and_oracle_runs_validate_together(tmp_path: Path) -> None:
    root = tmp_path / "exp"
    make_pear_run(root, "clean")
    make_pear_run(root, "attacked", attacker_ids=(4,), attack_value=4)
    make_pear_run(
        root, "oracle", attacker_ids=(4,), attack_value=4, verification_mode="oracle"
    )

    report = validate_runs([root])

    assert report.ok, [i.render() for i in report.errors()]
    assert len(report.runs) == 3


def test_generic_no_routing_run_is_accepted(tmp_path: Path) -> None:
    make_generic_run(tmp_path / "exp", "competitor")

    report = validate_runs([tmp_path])

    assert report.ok, [i.render() for i in report.errors()]
    run = report.runs[0]
    assert run.capabilities["has_routing"] is False
    assert run.capabilities["has_accuracy"] is True
    # Absent routing is not an error for a method that never routes.
    assert "routing_log_absent" not in codes(report)


# ------------------------------------------------------------- broken files --
def test_missing_required_file_is_named(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "broken")
    (run / "config.yaml").unlink()

    report = validate_runs([tmp_path])

    assert not report.ok
    assert "missing_file" in error_codes(report)


def test_malformed_jsonl_fails(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "broken")
    corrupt_jsonl(run / "results.jsonl", "{oops")

    report = validate_runs([tmp_path])

    assert "load_failed" in error_codes(report)


def test_bare_nan_fails_strict_json(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "nan")
    rows = read_jsonl_raw(run / "results.jsonl")
    text = "\n".join(json.dumps(r) for r in rows[:-1])
    (run / "results.jsonl").write_text(
        text + '\n{"example_id": "ex-3", "correct": NaN}\n', encoding="utf-8"
    )

    report = validate_runs([tmp_path])

    assert "load_failed" in error_codes(report)
    assert any("NaN" in i.message for i in report.errors())


def test_empty_results_file_fails(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "empty")
    (run / "results.jsonl").write_text("", encoding="utf-8")

    report = validate_runs([tmp_path])

    assert not report.ok


# ------------------------------------------------------------------- schema --
def test_mixed_schema_versions_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "exp"
    make_pear_run(root, "v2")
    make_pear_run(root, "v3", schema_version=3)

    report = validate_runs([root])

    assert "schema_versions_mixed" in error_codes(report)
    assert "schema_version_unsupported" in error_codes(report)


def test_unsupported_schema_version_can_be_opted_into(tmp_path: Path) -> None:
    make_pear_run(tmp_path / "exp", "v3", schema_version=3)

    rejected = validate_runs([tmp_path])
    accepted = validate_runs([tmp_path], allowed_schema_versions={2, 3})

    assert "schema_version_unsupported" in error_codes(rejected)
    assert "schema_version_unsupported" not in error_codes(accepted)


def test_routing_rows_must_agree_with_the_run_schema(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "mismatch")
    rows = read_jsonl_raw(run / "routing.jsonl")
    rows[0]["schema_version"] = 1
    rewrite_jsonl(run / "routing.jsonl", rows)

    report = validate_runs([tmp_path])

    assert "routing_schema_mismatch" in error_codes(report)


def test_missing_schema_version_is_rejected(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "legacy")
    payload = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    payload.pop("schema_version")
    (run / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

    report = validate_runs([tmp_path])

    assert "schema_version_missing" in error_codes(report)


# ----------------------------------------------------------- routing modes --
def test_mixed_routing_modes_across_runs_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "exp"
    make_pear_run(root, "sampled", routing_mode="sampled")
    make_pear_run(root, "enumerated", routing_mode="enumerated")

    report = validate_runs([root])

    assert report.ok  # separable, so not fatal ...
    assert "routing_modes_multiple" in codes(report)  # ... but never silent
    assert report.routing_modes == ["enumerated", "sampled"]


def test_mixed_routing_modes_within_one_condition_is_fatal(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "mixed")
    rows = read_jsonl_raw(run / "routing.jsonl")
    rows[0]["routing_mode"] = "enumerated"
    rewrite_jsonl(run / "routing.jsonl", rows)

    report = validate_runs([tmp_path])

    assert "routing_mode_mixed_within_condition" in error_codes(report)


def test_missing_routing_mode_is_fatal(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "nomode")
    rows = read_jsonl_raw(run / "routing.jsonl")
    for row in rows:
        row["routing_mode"] = None
    rewrite_jsonl(run / "routing.jsonl", rows)

    report = validate_runs([tmp_path])

    assert "routing_mode_missing" in error_codes(report)


# ------------------------------------------------------------- duplicates --
def test_duplicate_result_rows_are_rejected(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "dupes")
    duplicate_last_line(run / "results.jsonl")

    report = validate_runs([tmp_path])

    assert "duplicate_results" in error_codes(report)


def test_duplicate_routing_rows_are_rejected(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "dupes")
    duplicate_last_line(run / "routing.jsonl")

    report = validate_runs([tmp_path])

    assert "duplicate_routing" in error_codes(report)


# ------------------------------------------------------ confidence maps --
def test_incomplete_reported_confidence_is_rejected(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "incomplete", n_agents=4)
    rows = read_jsonl_raw(run / "routing.jsonl")
    rows[0]["reported_confidence"].pop("4")
    rewrite_jsonl(run / "routing.jsonl", rows)

    report = validate_runs([tmp_path])

    assert "confidence_reported_confidence_incomplete" in error_codes(report)


def test_incomplete_clean_confidence_is_rejected(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "incomplete", n_agents=4)
    rows = read_jsonl_raw(run / "routing.jsonl")
    rows[0]["clean_confidence"]["2"] = None
    rewrite_jsonl(run / "routing.jsonl", rows)

    report = validate_runs([tmp_path])

    assert "confidence_clean_confidence_incomplete" in error_codes(report)


def test_verification_off_must_leave_g_i_null(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "stray_g")
    rows = read_jsonl_raw(run / "routing.jsonl")
    rows[0]["g_i"] = {"1": 5.0, "2": 5.0, "3": 5.0, "4": 5.0}
    rewrite_jsonl(run / "routing.jsonl", rows)

    report = validate_runs([tmp_path])

    assert "confidence_g_i_present_without_verification" in error_codes(report)


def test_verification_on_requires_complete_g_i(tmp_path: Path) -> None:
    run = make_pear_run(
        tmp_path / "exp", "oracle", attacker_ids=(4,), attack_value=4, verification_mode="oracle"
    )
    rows = read_jsonl_raw(run / "routing.jsonl")
    rows[0]["g_i"].pop("2")
    rewrite_jsonl(run / "routing.jsonl", rows)

    report = validate_runs([tmp_path])

    assert "confidence_g_i_incomplete" in error_codes(report)


def test_oracle_run_with_complete_maps_is_valid(tmp_path: Path) -> None:
    make_pear_run(
        tmp_path / "exp", "oracle", attacker_ids=(4,), attack_value=4, verification_mode="oracle"
    )

    report = validate_runs([tmp_path])

    assert report.ok, [i.render() for i in report.errors()]


# ----------------------------------------------------------- corpus bytes --
def test_runs_scoring_different_corpora_are_not_pooled(tmp_path: Path) -> None:
    """Same benchmark split, different corpus bytes: not the same questions."""
    root = tmp_path / "exp"
    make_pear_run(root, "before")
    make_pear_run(root, "after")
    patch_summary(root / "before", dataset="gsm8k", dataset_split="test", dataset_sha256="a" * 64)
    patch_summary(root / "after", dataset="gsm8k", dataset_split="test", dataset_sha256="b" * 64)

    report = validate_runs([root])

    assert "dataset_corpus_mismatch" in error_codes(report)
    assert any("not comparable" in i.message for i in report.errors())


def test_matching_corpora_pool_fine(tmp_path: Path) -> None:
    root = tmp_path / "exp"
    make_pear_run(root, "one")
    make_pear_run(root, "two")
    for name in ("one", "two"):
        patch_summary(root / name, dataset="gsm8k", dataset_split="test", dataset_sha256="a" * 64)

    report = validate_runs([root])

    assert "dataset_corpus_mismatch" not in error_codes(report)


# --------------------------------------------------------------- coverage --
def test_missing_seed_combination_is_reported(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "gap", perm_seeds=(10, 11))
    rows = [r for r in read_jsonl_raw(run / "results.jsonl") if r["perm_seed"] != 11]
    rewrite_jsonl(run / "results.jsonl", rows)

    report = validate_runs([tmp_path])

    assert "coverage_missing_results" in error_codes(report)


def test_missing_round_is_reported(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "gap", rounds=2)
    rows = [r for r in read_jsonl_raw(run / "routing.jsonl") if r["round"] != 2]
    rewrite_jsonl(run / "routing.jsonl", rows)

    report = validate_runs([tmp_path])

    assert "coverage_missing_routing" in error_codes(report)


def test_a_condition_that_does_not_route_is_not_missing_coverage(tmp_path: Path) -> None:
    # A run comparing PEAR against a baseline with a fixed communication graph
    # has routing rows from one condition and none from the other. Demanding
    # them from both would make every such comparison invalid.
    import yaml

    run = make_pear_run(tmp_path / "exp", "mixed", rounds=2)
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    config["conditions"].append({"name": "debunc_prompt", "debate": {"mode": "debunc"}})
    (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    # The baseline's results are present; only its routing rows are absent.
    rows = read_jsonl_raw(run / "results.jsonl")
    baseline = [{**r, "condition": "debunc_prompt", "mode": "debunc"} for r in rows]
    rewrite_jsonl(run / "results.jsonl", rows + baseline)

    report = validate_runs([tmp_path])

    assert "coverage_missing_routing" not in error_codes(report)
    assert report.ok


def test_a_routing_condition_is_still_held_to_full_coverage(tmp_path: Path) -> None:
    # The exemption is for conditions that log *no* routing at all; one that
    # logged some rows and lost others is still a gap.
    run = make_pear_run(tmp_path / "exp", "partial", rounds=2)
    rows = [r for r in read_jsonl_raw(run / "routing.jsonl") if r["round"] != 2]
    rewrite_jsonl(run / "routing.jsonl", rows)

    report = validate_runs([tmp_path])

    assert "coverage_missing_routing" in error_codes(report)


def test_unexpected_combination_is_reported(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "extra")
    rows = read_jsonl_raw(run / "results.jsonl")
    stray = dict(rows[0])
    stray["perm_seed"] = 999
    rewrite_jsonl(run / "results.jsonl", rows + [stray])

    report = validate_runs([tmp_path])

    assert "coverage_unexpected_results" in error_codes(report)


def test_coverage_is_not_guessed_without_replication_config(tmp_path: Path) -> None:
    import yaml

    run = make_pear_run(tmp_path / "exp", "noreps")
    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    config.pop("replication")
    (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    report = validate_runs([tmp_path])

    assert report.ok
    assert "coverage_unverifiable" in codes(report)


# ------------------------------------------------------------------- CLI --
def test_cli_exit_codes(tmp_path: Path, capsys) -> None:
    clean = tmp_path / "clean"
    make_pear_run(clean / "exp", "run")
    assert main([str(clean)]) == 0
    assert "routing modes" in capsys.readouterr().out

    broken = tmp_path / "broken"
    run = make_pear_run(broken / "exp", "run")
    (run / "config.yaml").unlink()
    assert main([str(broken)]) == 1


def test_cli_writes_json_report(tmp_path: Path, capsys) -> None:
    make_pear_run(tmp_path / "exp", "clean")
    out_path = tmp_path / "report.json"

    assert main([str(tmp_path), "--json", str(out_path)]) == 0

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["runs"][0]["capabilities"]["has_routing"] is True


def test_no_runs_found_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    report = validate_runs([tmp_path])

    assert "no_runs_found" in error_codes(report)
