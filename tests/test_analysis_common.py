"""Tests for the shared analysis loading/normalisation layer.

The three things worth pinning here are the ones that silently corrupt every
downstream figure if they drift: how a condition's effective config is
resolved, what a missing capability turns into, and whether a file that is not
valid JSON is allowed through.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.common import (
    Capabilities,
    RunLoadError,
    StrictJsonError,
    deep_merge,
    detect_run_adapter,
    discover_runs,
    load_normalized_run,
    load_routing,
    load_run_config,
    load_summary,
    read_jsonl,
    resolve_condition_config,
    resolve_condition_metadata,
    routing_confidence_field,
    strict_json_loads,
)
from tests.analysis_fixtures import corrupt_jsonl, make_generic_run, make_pear_run


# ------------------------------------------------------------------ discovery --
def test_discover_runs_is_recursive(tmp_path: Path) -> None:
    make_pear_run(tmp_path / "exp_a", "run1")
    make_pear_run(tmp_path / "exp_a", "run2")
    make_pear_run(tmp_path / "exp_b" / "nested" / "deeper", "run3")

    found = discover_runs([tmp_path])

    assert [p.name for p in found] == ["run1", "run2", "run3"]


def test_parent_experiment_dir_is_not_a_run(tmp_path: Path) -> None:
    exp_root = tmp_path / "exp_a"
    make_pear_run(exp_root, "run1")
    # Experiment roots often carry stray report files next to the run dirs.
    (exp_root / "check_logs_report.txt").write_text("...", encoding="utf-8")

    found = discover_runs([tmp_path])

    assert [str(p) for p in found] == [str(exp_root / "run1")]


def test_discover_runs_accepts_a_run_dir_directly_and_dedupes(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "run1")

    assert discover_runs([run]) == [run]
    assert len(discover_runs([run, tmp_path])) == 1


def test_discover_runs_rejects_a_missing_path(tmp_path: Path) -> None:
    with pytest.raises(RunLoadError, match="no such path"):
        discover_runs([tmp_path / "nope"])


# --------------------------------------------------------------- strict JSON --
@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_strict_json_rejects_non_finite_tokens(token: str) -> None:
    with pytest.raises(StrictJsonError):
        strict_json_loads('{"value": %s}' % token)


def test_strict_json_keeps_null_as_null() -> None:
    assert strict_json_loads('{"g_i": null}') == {"g_i": None}


def test_read_jsonl_reports_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text('{"a": 1}\n{"b": NaN}\n', encoding="utf-8")

    with pytest.raises(StrictJsonError) as excinfo:
        read_jsonl(path)

    assert "results.jsonl:2" in str(excinfo.value)


def test_bare_nan_in_routing_fails_to_load(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "run1")
    corrupt_jsonl(run / "routing.jsonl", '{"schema_version": 2, "selected_score": NaN}')

    with pytest.raises(StrictJsonError):
        load_routing(run)


def test_malformed_jsonl_fails_clearly(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "run1")
    corrupt_jsonl(run / "results.jsonl", "{not json at all")

    with pytest.raises(StrictJsonError, match="results.jsonl"):
        load_normalized_run(run)


def test_empty_summary_fails_clearly(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "run1")
    (run / "summary.json").write_text("", encoding="utf-8")

    with pytest.raises(StrictJsonError, match="empty"):
        load_summary(run)


# ---------------------------------------------------------- condition merging --
def test_deep_merge_matches_the_runner() -> None:
    """The analysis copy of the merge must behave like the runner's."""
    from runner.experiment import _deep_merge  # imported here: heavy dependency

    base = {
        "n_agents": 5,
        "robustness": {
            "enabled": True,
            "type": "confidence_inflation",
            "confidence_inflation": {"agent_ids": [5], "mode": "fixed_report", "value": 4},
        },
    }
    override = {"robustness": {"confidence_inflation": {"value": 2}}}

    ours = deep_merge(json.loads(json.dumps(base)), override)
    theirs = _deep_merge(json.loads(json.dumps(base)), override)

    assert ours == theirs
    # Sibling keys of the nested block survive: a shallow update would drop them.
    assert ours["robustness"]["confidence_inflation"] == {
        "agent_ids": [5],
        "mode": "fixed_report",
        "value": 2,
    }


def test_resolve_condition_config_deep_merges_nested_overrides(tmp_path: Path) -> None:
    run = make_pear_run(
        tmp_path / "exp",
        "run1",
        attacker_ids=(3,),
        attack_value=4,
        condition_debate={
            "mode": "pear_full",
            "robustness": {"confidence_inflation": {"value": 2}},
        },
    )
    config = load_run_config(run)

    effective = resolve_condition_config(config, "pear_full")
    inflation = effective["debate"]["robustness"]["confidence_inflation"]

    assert inflation["value"] == 2
    assert inflation["agent_ids"] == [3]
    assert inflation["mode"] == "fixed_report"
    assert effective["debate"]["robustness"]["enabled"] is True


def test_resolve_condition_config_rejects_an_unknown_condition(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "run1")
    config = load_run_config(run)

    with pytest.raises(RunLoadError, match="unknown condition"):
        resolve_condition_config(config, "not_a_condition")


def test_condition_metadata_carries_effective_values(tmp_path: Path) -> None:
    run = make_pear_run(
        tmp_path / "exp",
        "run1",
        n_agents=5,
        rounds=3,
        attacker_ids=(5,),
        attack_value=4,
        verification_mode="oracle",
        dataset_split="test",
    )
    meta = resolve_condition_metadata(load_run_config(run), "pear_full")

    assert meta.dataset == "dummy"
    assert meta.dataset_split == "test"
    assert meta.model == "mock-agent"
    assert meta.mechanism == "pear_full"
    assert meta.base_topology == "k_regular"
    assert (meta.n_agents, meta.rounds) == (5, 3)
    assert meta.routing_temperature == 0.7
    assert meta.alpha_influence == 0.7
    assert meta.verification_mode == "oracle"
    assert meta.attack_type == "confidence_inflation"
    assert meta.attack_mode == "fixed_report"
    assert meta.attack_value == 4


def test_adversarial_fraction_from_attacker_count(tmp_path: Path) -> None:
    run = make_pear_run(
        tmp_path / "exp", "attacked", n_agents=4, attacker_ids=(3, 4), attack_value=5
    )
    meta = resolve_condition_metadata(load_run_config(run), "pear_full")

    assert meta.attacker_ids == (3, 4)
    assert meta.attacker_count == 2
    assert meta.adversarial_fraction == 0.5


def test_clean_run_has_zero_adversarial_fraction(tmp_path: Path) -> None:
    run = make_pear_run(tmp_path / "exp", "clean", n_agents=4)
    meta = resolve_condition_metadata(load_run_config(run), "pear_full")

    assert meta.attacker_count == 0
    assert meta.adversarial_fraction == 0.0
    assert meta.attack_type == "none"
    assert meta.attack_mode is None


def test_disabled_attack_block_is_not_an_attack(tmp_path: Path) -> None:
    """A fully configured attack under ``enabled: false`` is a clean arm."""
    run = make_pear_run(
        tmp_path / "exp",
        "disabled",
        n_agents=4,
        attacker_ids=(4,),
        attack_value=5,
        attack_enabled=False,
    )
    meta = resolve_condition_metadata(load_run_config(run), "pear_full")

    assert meta.attacker_count == 0
    assert meta.adversarial_fraction == 0.0


def test_attack_type_is_preserved_not_flattened(tmp_path: Path) -> None:
    run = make_pear_run(
        tmp_path / "exp", "targeted", attacker_ids=(2,), attack_mode="targeted_wrong", attack_value=5
    )
    meta = resolve_condition_metadata(load_run_config(run), "pear_full")

    assert meta.attack_type == "confidence_inflation"
    assert meta.attack_mode == "targeted_wrong"


# --------------------------------------------------------------- adapters --
def test_detect_adapter_pear_and_generic(tmp_path: Path) -> None:
    pear = make_pear_run(tmp_path / "exp", "pear_run")
    generic = make_generic_run(tmp_path / "exp", "generic_run")

    assert detect_run_adapter(pear) == "pear"
    assert detect_run_adapter(generic) == "generic"


def test_generic_competitor_loads_without_routing(tmp_path: Path) -> None:
    run_dir = make_generic_run(tmp_path / "exp", "competitor", method="single_agent_cot")

    run = load_normalized_run(run_dir)

    assert run.adapter == "generic"
    assert run.method == "single_agent_cot"
    assert run.routing is None
    assert run.capabilities.has_routing is False
    assert run.capabilities.has_accuracy is True
    assert run.capabilities.has_token_usage is True
    assert all(row["method"] == "single_agent_cot" for row in run.results)


def test_missing_routing_is_a_capability_not_a_zero(tmp_path: Path) -> None:
    run_dir = make_generic_run(tmp_path / "exp", "competitor")

    run = load_normalized_run(run_dir)

    assert run.capabilities.has_routing is False
    assert run.routing is None
    assert all(row.get("routing_mode") is None for row in run.results)


def test_pear_run_without_routing_log_keeps_accuracy(tmp_path: Path) -> None:
    run_dir = make_pear_run(tmp_path / "exp", "no_routing", include_routing=False)

    run = load_normalized_run(run_dir)

    assert run.capabilities.has_routing is False
    assert run.capabilities.has_accuracy is True
    assert len(run.results) == 6


def test_total_tokens_is_the_sum(tmp_path: Path) -> None:
    run_dir = make_pear_run(
        tmp_path / "exp",
        "tokens",
        rounds=1,
        prompt_tokens_per_round=100,
        completion_tokens_per_round=25,
    )

    run = load_normalized_run(run_dir)
    row = run.results[0]

    assert row["prompt_tokens"] == 200
    assert row["completion_tokens"] == 50
    assert row["total_tokens"] == 250


def test_missing_token_usage_stays_missing(tmp_path: Path) -> None:
    run_dir = make_pear_run(tmp_path / "exp", "no_tokens", include_tokens=False)

    run = load_normalized_run(run_dir)

    assert run.capabilities.has_token_usage is False
    assert all(row["total_tokens"] is None for row in run.results)
    assert all(row["prompt_tokens"] is None for row in run.results)


def test_provenance_columns_are_present_on_every_row(tmp_path: Path) -> None:
    run_dir = make_pear_run(tmp_path / "exp", "prov")

    run = load_normalized_run(run_dir)

    for row in run.results:
        assert row["source_run_dir"] == str(run_dir)
        assert row["run_id"].endswith("prov")
        assert row["schema_version"] == 2
        assert row["mock"] is True
        assert row["dataset"] == "dummy"
        assert row["model"] == "mock-agent"
        assert row["condition"] == "pear_full"


def test_routing_confidence_field_depends_on_verification() -> None:
    assert routing_confidence_field("none") == "reported_confidence"
    assert routing_confidence_field(None) == "reported_confidence"
    assert routing_confidence_field("oracle") == "verified_confidence"


def test_oracle_run_logs_g_i_and_verified_confidence(tmp_path: Path) -> None:
    run_dir = make_pear_run(
        tmp_path / "exp",
        "oracle",
        attacker_ids=(4,),
        attack_value=4,
        verification_mode="oracle",
    )

    run = load_normalized_run(run_dir)
    row = run.routing[0]

    assert row["verified_confidence_mode"] == "oracle"
    assert row["g_i"]["4"] == 1.0
    assert row["verified_confidence"]["4"] == 1.0  # min(reported 4, g_i 1)
    assert run.capabilities.has_confidence_reports is True


def test_capabilities_default_to_false() -> None:
    caps = Capabilities()

    assert not any(caps.as_dict().values())
