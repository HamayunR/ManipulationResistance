"""Tests for ``analysis/check_figure_readiness.py``.

A readiness report is only useful if "not ready" comes with the experiment
that would fix it. These tests therefore assert on the *named* missing arm or
capability, not just on the boolean.
"""

from __future__ import annotations

import json
from pathlib import Path

from analysis.check_figure_readiness import check_readiness, main
from analysis.collect_runs import collect
from tests.analysis_fixtures import make_generic_run, make_pear_run

REPORT_VALUES = (1, 2, 3, 4, 5)


def build(tmp_path: Path, builder) -> dict:
    """Run a fixture builder, collect the tables, return the readiness payload."""
    root = tmp_path / "runs"
    builder(root)
    collect([root], tmp_path / "artifacts", quiet=True)
    return check_readiness(tmp_path / "artifacts")


def figure(payload: dict, number: int) -> dict:
    return payload["figures"][str(number)]


def full_sweep(root: Path, **kwargs) -> None:
    """Truthful arm plus fixed reports 1-5 over matching keys."""
    make_pear_run(root, "truthful", **kwargs)
    for value in REPORT_VALUES:
        make_pear_run(
            root, f"report_{value}", attacker_ids=(4,), attack_value=value, **kwargs
        )


# ---------------------------------------------------------------- figure 1 --
def test_figure1_ready_with_two_adversarial_fractions(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_pear_run(root, "clean", n_agents=4)
        make_pear_run(root, "attacked", n_agents=4, attacker_ids=(4,), attack_value=5)

    report = figure(build(tmp_path, builder), 1)

    assert report["ready"] is True
    fractions = {f for g in report["available_groups"] for f in g["adversarial_fractions"]}
    assert fractions == {0.0, 0.25}


def test_figure1_not_ready_with_a_single_fraction(tmp_path: Path) -> None:
    report = figure(build(tmp_path, lambda root: make_pear_run(root, "clean")), 1)

    assert report["ready"] is False
    assert any("only one adversarial fraction" in arm for arm in report["missing_arms"])


def test_figure1_not_ready_without_a_clean_arm(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_pear_run(root, "one", n_agents=4, attacker_ids=(4,), attack_value=5)
        make_pear_run(root, "two", n_agents=4, attacker_ids=(3, 4), attack_value=5)

    report = figure(build(tmp_path, builder), 1)

    assert report["ready"] is False
    assert any("no clean 0.0 arm" in arm for arm in report["missing_arms"])


def test_figure1_accepts_a_method_without_routing(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_generic_run(root, "competitor_clean", adversarial_fraction=0.0)
        make_generic_run(root, "competitor_attacked", adversarial_fraction=0.25)

    report = figure(build(tmp_path, builder), 1)

    assert report["ready"] is True
    assert report["missing_capabilities"] == []


# ---------------------------------------------------------------- figure 2 --
def test_figure2_ready_with_the_full_sweep(tmp_path: Path) -> None:
    report = figure(build(tmp_path, full_sweep), 2)

    assert report["ready"] is True
    assert report["available_groups"][0]["report_values_present"] == list(REPORT_VALUES)
    assert report["available_groups"][0]["has_truthful_arm"] is True


def test_figure2_missing_truthful_arm_is_named(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        for value in REPORT_VALUES:
            make_pear_run(root, f"report_{value}", attacker_ids=(4,), attack_value=value)

    report = figure(build(tmp_path, builder), 2)

    assert report["ready"] is False
    assert any("truthful arm" in arm for arm in report["missing_arms"])


def test_figure2_names_the_missing_report_values(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_pear_run(root, "truthful")
        for value in (1, 2, 5):
            make_pear_run(root, f"report_{value}", attacker_ids=(4,), attack_value=value)

    report = figure(build(tmp_path, builder), 2)

    assert report["ready"] is False
    assert any("fixed_report values [3, 4]" in arm for arm in report["missing_arms"])


def test_figure2_rejects_mismatched_pairing_keys(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_pear_run(root, "truthful", perm_seeds=(10, 11))
        for value in REPORT_VALUES:
            # One arm was run over a different perm-seed set, so the arms do not
            # describe the same set of debates.
            perm_seeds = (10,) if value == 3 else (10, 11)
            make_pear_run(
                root,
                f"report_{value}",
                attacker_ids=(4,),
                attack_value=value,
                perm_seeds=perm_seeds,
            )

    report = figure(build(tmp_path, builder), 2)

    assert report["ready"] is False
    assert any("identical paired key coverage" in arm for arm in report["missing_arms"])


def test_figure2_rejects_multiple_confidence_attackers(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_pear_run(root, "truthful")
        for value in REPORT_VALUES:
            make_pear_run(
                root, f"report_{value}", attacker_ids=(3, 4), attack_value=value
            )

    report = figure(build(tmp_path, builder), 2)

    assert report["ready"] is False
    assert any("exactly one confidence attacker" in arm for arm in report["missing_arms"])


# ---------------------------------------------------------------- figure 3 --
def test_figure3_ready_across_report_values(tmp_path: Path) -> None:
    report = figure(build(tmp_path, full_sweep), 3)

    assert report["ready"] is True
    group = report["available_groups"][0]
    assert group["reported_confidence_values"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert group["complete_sweep"] is True


def test_figure3_not_ready_with_one_confidence_value(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_pear_run(root, "attacked", attacker_ids=(4,), attack_value=4)

    report = figure(build(tmp_path, builder), 3)

    assert report["ready"] is False
    assert any("only one reported-confidence value" in arm for arm in report["missing_arms"])


def test_figure3_not_ready_without_routing_capability(tmp_path: Path) -> None:
    report = figure(build(tmp_path, lambda root: make_generic_run(root, "competitor")), 3)

    assert report["ready"] is False
    assert report["missing_capabilities"] == ["has_routing"]


def test_figure3_reports_no_attacker_arm(tmp_path: Path) -> None:
    report = figure(build(tmp_path, lambda root: make_pear_run(root, "clean")), 3)

    assert report["ready"] is False
    assert "confidence-attacker arm" in report["missing_arms"][0]


def test_figure3_names_the_excluded_no_routing_method(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        full_sweep(root)
        make_generic_run(root, "competitor")

    report = figure(build(tmp_path, builder), 3)

    assert report["ready"] is True
    assert any("single_agent_baseline" in note for note in report["notes"])
    assert any("capability limitation" in note for note in report["notes"])


# ---------------------------------------------------------------- figure 4 --
def test_figure4_ready_and_includes_a_generic_competitor(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_pear_run(root, "clean")
        make_generic_run(root, "competitor")

    report = figure(build(tmp_path, builder), 4)

    assert report["ready"] is True
    methods = {m for g in report["available_groups"] for m in g["methods"]}
    assert methods == {"pear", "single_agent_baseline"}


def test_figure4_excludes_a_method_without_token_usage(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_pear_run(root, "clean")
        make_generic_run(root, "competitor", include_tokens=False)

    report = figure(build(tmp_path, builder), 4)

    assert report["ready"] is True
    assert any("single_agent_baseline" in arm for arm in report["missing_arms"])
    assert any("never counted as zero" in note for note in report["notes"])


def test_figure4_not_ready_when_nothing_reports_tokens(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_pear_run(root, "clean", include_tokens=False)

    report = figure(build(tmp_path, builder), 4)

    assert report["ready"] is False
    assert report["missing_capabilities"] == ["has_token_usage"]


def test_figure4_needs_a_clean_condition(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_pear_run(root, "attacked", attacker_ids=(4,), attack_value=4)

    report = figure(build(tmp_path, builder), 4)

    assert report["ready"] is False
    assert report["missing_arms"] == ["a clean arm"]


# ---------------------------------------------------------------- figure 5 --
def test_figure5_ready_on_a_full_two_by_two_grid(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        for temperature in (0.3, 0.7):
            for alpha in (0.4, 0.8):
                make_pear_run(
                    root,
                    f"t{temperature}_a{alpha}",
                    routing_temperature=temperature,
                    alpha_influence=alpha,
                )

    report = figure(build(tmp_path, builder), 5)

    assert report["ready"] is True
    assert report["available_groups"][0]["grid"] == "2 x 2"
    assert report["available_groups"][0]["missing_cells"] == []


def test_figure5_refuses_a_one_by_two_grid(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        for alpha in (0.4, 0.8):
            make_pear_run(root, f"a{alpha}", routing_temperature=0.7, alpha_influence=alpha)

    report = figure(build(tmp_path, builder), 5)

    assert report["ready"] is False
    assert any("needs at least 2 x 2" in arm for arm in report["missing_arms"])


def test_figure5_refuses_an_incomplete_grid(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        for temperature, alpha in ((0.3, 0.4), (0.3, 0.8), (0.7, 0.4)):
            make_pear_run(
                root,
                f"t{temperature}_a{alpha}",
                routing_temperature=temperature,
                alpha_influence=alpha,
            )

    report = figure(build(tmp_path, builder), 5)

    assert report["ready"] is False
    assert any("missing grid cell" in arm for arm in report["missing_arms"])


def test_figure5_robustness_gap_needs_matched_arms(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    for temperature in (0.3, 0.7):
        for alpha in (0.4, 0.8):
            make_pear_run(
                root,
                f"t{temperature}_a{alpha}",
                routing_temperature=temperature,
                alpha_influence=alpha,
            )
    collect([root], tmp_path / "artifacts", quiet=True)

    payload = check_readiness(tmp_path / "artifacts", heatmap_metric="robustness_gap")
    report = figure(payload, 5)

    assert report["ready"] is False
    assert any("matched clean and attacked" in arm for arm in report["missing_arms"])


# ------------------------------------------------------- global behaviour --
def test_mixed_routing_modes_form_separate_groups(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        for mode in ("sampled", "enumerated"):
            make_pear_run(root, f"clean_{mode}", routing_mode=mode)
            make_pear_run(
                root, f"attacked_{mode}", routing_mode=mode, attacker_ids=(4,), attack_value=5
            )

    payload = build(tmp_path, builder)
    report = figure(payload, 3)

    # Each routing mode is judged on its own; neither is pooled into the other.
    modes = {g["routing_mode"] for g in report["available_groups"]} | {
        arm.split("routing_mode=")[1].split(":")[0]
        for arm in report["missing_arms"]
        if "routing_mode=" in arm
    }
    assert modes == {"sampled", "enumerated"}
    assert payload["routing_modes"] == ["enumerated", "sampled"]
    assert payload["state_aware_routing_modes"] == ["enumerated", "sampled"]


def test_readiness_json_is_written_by_the_cli(tmp_path: Path, capsys) -> None:
    root = tmp_path / "runs"
    full_sweep(root)
    collect([root], tmp_path / "artifacts", quiet=True)

    assert main([str(tmp_path / "artifacts")]) == 0

    payload = json.loads((tmp_path / "artifacts" / "readiness.json").read_text(encoding="utf-8"))
    assert set(payload["figures"]) == {"1", "2", "3", "4", "5"}
    assert payload["figures"]["3"]["ready"] is True
    out = capsys.readouterr().out
    assert "Figure 3" in out
    assert "READY" in out


def test_readiness_requires_collected_tables(tmp_path: Path) -> None:
    assert main([str(tmp_path / "missing")]) == 2
