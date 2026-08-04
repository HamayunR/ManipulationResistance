"""Tests for ``analysis/collect_runs.py``: the normalised tables.

Everything downstream reads these three CSVs, so what matters is that a value
in them means exactly one thing: a missing capability stays missing, a share is
a share of the round it came from, and every row still names where it came
from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.collect_runs import (
    RESULTS_COLUMNS,
    ROUTING_COLUMNS,
    RUNS_COLUMNS,
    collect,
    main,
)
from analysis.common import RunLoadError
from tests.analysis_fixtures import (
    corrupt_jsonl,
    make_generic_run,
    make_pear_run,
    read_jsonl_raw,
    rewrite_jsonl,
)


@pytest.fixture()
def artifacts(tmp_path: Path):
    def _collect(paths, **kwargs):
        kwargs.setdefault("quiet", True)
        return collect(paths, tmp_path / "artifacts", **kwargs)

    return _collect


# ------------------------------------------------------------------ runs.csv --
def test_runs_table_has_one_row_per_run_and_condition(tmp_path: Path, artifacts) -> None:
    root = tmp_path / "exp"
    make_pear_run(root, "clean")
    make_pear_run(root, "attacked", attacker_ids=(4,), attack_value=4)

    result = artifacts([root])

    assert list(result.runs.columns) == list(RUNS_COLUMNS)
    assert len(result.runs) == 2
    assert set(result.runs["run_id"]) == {"exp/clean", "exp/attacked"}


def test_runs_table_records_condition_metadata(tmp_path: Path, artifacts) -> None:
    make_pear_run(
        tmp_path / "exp",
        "attacked",
        n_agents=4,
        rounds=2,
        attacker_ids=(4,),
        attack_value=3,
        verification_mode="oracle",
        routing_temperature=0.5,
        alpha_influence=0.9,
    )

    row = artifacts([tmp_path]).runs.iloc[0]

    assert row["mechanism"] == "pear_full"
    assert row["base_topology"] == "k_regular"
    assert row["n_agents"] == 4
    assert row["rounds"] == 2
    assert row["routing_mode"] == "sampled"
    assert row["routing_temperature"] == 0.5
    assert row["alpha_influence"] == 0.9
    assert row["verification_mode"] == "oracle"
    assert row["attack_type"] == "confidence_inflation"
    assert row["attack_mode"] == "fixed_report"
    assert row["attack_value"] == 3
    assert json.loads(row["attacker_ids"]) == [4]
    assert row["attacker_count"] == 1
    assert row["adversarial_fraction"] == 0.25
    assert row["seed_count"] == 1
    assert row["perm_seed_count"] == 2


def test_runs_table_reports_capabilities(tmp_path: Path, artifacts) -> None:
    root = tmp_path / "exp"
    make_pear_run(root, "pear")
    make_generic_run(root, "competitor")

    runs = artifacts([root]).runs.set_index("method")

    assert bool(runs.loc["pear", "has_routing"]) is True
    assert bool(runs.loc["single_agent_baseline", "has_routing"]) is False
    assert bool(runs.loc["single_agent_baseline", "has_accuracy"]) is True
    assert bool(runs.loc["single_agent_baseline", "has_token_usage"]) is True


# --------------------------------------------------------------- results.csv --
def test_results_table_row_count_and_keys(tmp_path: Path, artifacts) -> None:
    make_pear_run(
        tmp_path / "exp", "run", examples=("a", "b"), seeds=(0, 1), perm_seeds=(10, 11)
    )

    results = artifacts([tmp_path]).results

    assert list(results.columns) == list(RESULTS_COLUMNS)
    assert len(results) == 2 * 2 * 2
    assert results.duplicated(
        ["source_run_dir", "condition", "example_id", "seed", "perm_seed"]
    ).sum() == 0


def test_total_tokens_is_the_sum_of_parts(tmp_path: Path, artifacts) -> None:
    make_pear_run(
        tmp_path / "exp",
        "run",
        rounds=1,
        prompt_tokens_per_round=100,
        completion_tokens_per_round=25,
    )

    results = artifacts([tmp_path]).results

    assert (results["total_tokens"] == results["prompt_tokens"] + results["completion_tokens"]).all()
    assert results["total_tokens"].iloc[0] == 250


def test_missing_token_usage_is_not_zero(tmp_path: Path, artifacts) -> None:
    make_pear_run(tmp_path / "exp", "run", include_tokens=False)

    results = artifacts([tmp_path]).results

    assert results["total_tokens"].isna().all()
    assert not (results["total_tokens"].fillna(-1) == 0).any()


def test_histories_round_trip_as_json(tmp_path: Path, artifacts) -> None:
    make_pear_run(tmp_path / "exp", "run", rounds=2, n_agents=3)

    row = artifacts([tmp_path]).results.iloc[0]
    influence = json.loads(row["influence_history"])
    answers = json.loads(row["answer_history"])

    assert len(influence) == 3  # initial + one per round
    assert set(influence[0]) == {"1", "2", "3"}
    assert len(answers) == 3


def test_provenance_columns_are_never_dropped(tmp_path: Path, artifacts) -> None:
    make_pear_run(tmp_path / "exp", "run")

    result = artifacts([tmp_path])

    for frame in (result.runs, result.results, result.routing):
        assert "source_run_dir" in frame.columns
        assert "run_id" in frame.columns
        assert frame["source_run_dir"].notna().all()
        assert frame["run_id"].notna().all()


# --------------------------------------------------------------- routing.csv --
def test_routing_expands_to_one_row_per_agent(tmp_path: Path, artifacts) -> None:
    make_pear_run(
        tmp_path / "exp",
        "run",
        n_agents=4,
        rounds=2,
        examples=("a", "b"),
        seeds=(0,),
        perm_seeds=(10,),
    )

    routing = artifacts([tmp_path]).routing

    assert list(routing.columns) == list(ROUTING_COLUMNS)
    assert len(routing) == 4 * 2 * 2  # agents x rounds x examples
    per_decision = routing.groupby(["example_id", "seed", "perm_seed", "round"]).size()
    assert (per_decision == 4).all()
    assert sorted(routing["agent_id"].unique()) == [1, 2, 3, 4]


def test_out_degree_shares_sum_to_one_per_decision(tmp_path: Path, artifacts) -> None:
    make_pear_run(tmp_path / "exp", "run", n_agents=4, rounds=2)

    routing = artifacts([tmp_path]).routing
    sums = routing.groupby(["run_id", "condition", "example_id", "seed", "perm_seed", "round"])[
        "out_degree_share"
    ].sum()

    assert sums.between(1 - 1e-9, 1 + 1e-9).all()


def test_zero_total_out_degree_leaves_share_missing(tmp_path: Path, artifacts) -> None:
    run = make_pear_run(tmp_path / "exp", "run", n_agents=3, rounds=1)
    rows = read_jsonl_raw(run / "routing.jsonl")
    rows[0]["out_degree"] = [0, 0, 0]
    rewrite_jsonl(run / "routing.jsonl", rows)

    routing = artifacts([tmp_path]).routing
    flagged = routing[routing["zero_out_degree_total"]]

    assert len(flagged) == 3
    assert flagged["out_degree_share"].isna().all()


def test_routing_confidence_follows_verification_mode(tmp_path: Path, artifacts) -> None:
    root = tmp_path / "exp"
    make_pear_run(root, "attacked", n_agents=4, attacker_ids=(4,), attack_value=4)
    make_pear_run(
        root,
        "oracle",
        n_agents=4,
        attacker_ids=(4,),
        attack_value=4,
        verification_mode="oracle",
    )

    routing = collect([root], tmp_path / "art", quiet=True).routing
    attacker = routing[routing["is_attacker"]]
    no_verify = attacker[attacker["verification_mode"] == "none"]
    oracle = attacker[attacker["verification_mode"] == "oracle"]

    assert (no_verify["reported_confidence"] == 4.0).all()
    assert (no_verify["routing_confidence"] == no_verify["reported_confidence"]).all()
    assert no_verify["verified_confidence"].isna().all()
    # min(reported 4, g_i 1) -- the router scored 1, not the reported 4.
    assert (oracle["routing_confidence"] == 1.0).all()
    assert (oracle["reported_confidence"] == 4.0).all()


def test_attacker_flag_marks_only_the_attacked_agent(tmp_path: Path, artifacts) -> None:
    make_pear_run(tmp_path / "exp", "run", n_agents=4, attacker_ids=(3,), attack_value=5)

    routing = artifacts([tmp_path]).routing

    assert set(routing.loc[routing["is_attacker"], "agent_id"]) == {3}
    assert set(routing.loc[~routing["is_attacker"], "agent_id"]) == {1, 2, 4}


def test_influence_before_is_kept_separate_from_exposure(tmp_path: Path, artifacts) -> None:
    make_pear_run(tmp_path / "exp", "run", n_agents=4, rounds=2)

    routing = artifacts([tmp_path]).routing
    first_round = routing[routing["round"] == 1]

    # rho starts uniform; the exposure share does not, so the two columns are
    # measuring different things and neither is a copy of the other.
    assert (first_round["influence_before"] == 0.25).all()
    assert first_round["out_degree_share"].nunique() > 1


def test_generic_competitor_contributes_no_routing_rows(tmp_path: Path, artifacts) -> None:
    root = tmp_path / "exp"
    make_pear_run(root, "pear")
    make_generic_run(root, "competitor")

    result = artifacts([root])

    assert set(result.routing["method"]) == {"pear"}
    assert "single_agent_baseline" in set(result.results["method"])


# ---------------------------------------------------------------- manifest --
def test_manifest_records_provenance(tmp_path: Path) -> None:
    root = tmp_path / "exp"
    make_pear_run(root, "run")
    out = tmp_path / "artifacts"

    result = collect([root], out, quiet=True, command="pytest")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["n_runs"] == 1
    assert manifest["n_results_rows"] == len(result.results)
    assert manifest["n_routing_rows"] == len(result.routing)
    assert manifest["log_schema_versions"] == [2]
    assert manifest["routing_modes"] == ["sampled"]
    assert manifest["analysis_command"] == "pytest"
    assert manifest["input_paths"] == [str(root)]
    assert manifest["discovered_run_dirs"] == [str(root / "run")]
    # The runner does not log its own commit, so the analysis commit must not
    # be presented as the commit that generated the data.
    assert manifest["experiment_git_commit"] is None
    assert "not recorded" in manifest["experiment_git_commit_note"]


def test_tables_are_written_to_disk(tmp_path: Path) -> None:
    make_pear_run(tmp_path / "exp", "run")
    out = tmp_path / "artifacts"

    collect([tmp_path], out, quiet=True)

    for name in ("runs.csv", "results.csv", "routing.csv"):
        path = out / "tables" / name
        assert path.is_file()
        assert len(pd.read_csv(path)) > 0


# ------------------------------------------------------------- refusals --
def test_collection_refuses_invalid_runs(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "run")
    corrupt_jsonl(run / "results.jsonl", "{bad")

    with pytest.raises(RunLoadError, match="validation error"):
        collect([tmp_path], tmp_path / "artifacts", quiet=True)


def test_cli_returns_nonzero_on_invalid_input(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "run")
    corrupt_jsonl(run / "routing.jsonl", "{bad")

    code = main([str(tmp_path), "--output", str(tmp_path / "artifacts")])

    assert code == 1


def test_cli_writes_tables(tmp_path: Path) -> None:
    make_pear_run(tmp_path / "exp", "run")
    out = tmp_path / "artifacts"

    assert main([str(tmp_path), "--output", str(out)]) == 0
    assert (out / "tables" / "runs.csv").is_file()
