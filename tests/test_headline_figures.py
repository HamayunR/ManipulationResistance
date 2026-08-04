"""Tests for Figures 1, 2, 4, 5 and the master analysis command.

The recurring risk in this layer is a figure that quietly turns a missing
capability into a number -- a method with no token usage plotted at zero cost,
an unmatched arm dropped instead of refused, a heatmap cell interpolated. Each
of those has a test that fails if the silence returns.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import analysis.figure_accuracy_vs_adversaries as figure1
import analysis.figure_accuracy_vs_cost as figure4
import analysis.figure_empirical_ic_regret as figure2
import analysis.figure_routing_heatmap as figure5
from analysis.collect_runs import collect
from analysis.figure_utils import FigureError
from analysis.run_headline_analysis import main as headline_main, run as headline_run
from tests.analysis_fixtures import make_generic_run, make_pear_run

REPORT_VALUES = (1, 2, 3, 4, 5)
EXAMPLES = ("a", "b", "c")


def collected(tmp_path: Path, builder) -> Path:
    root = tmp_path / "runs"
    builder(root)
    collect([root], tmp_path / "artifacts", quiet=True)
    return tmp_path / "artifacts"


def full_sweep(root: Path, **kwargs) -> None:
    # The truthful arm is the same dissenting agent reporting honestly: same
    # answers, no confidence attack. Anything else would compare two different
    # agents rather than two reports.
    make_pear_run(root, "truthful", examples=EXAMPLES, dissenting_ids=(4,), **kwargs)
    for value in REPORT_VALUES:
        make_pear_run(
            root, f"report_{value}", examples=EXAMPLES, attacker_ids=(4,), attack_value=value, **kwargs
        )


def artifacts_exist(paths) -> None:
    for path in (paths.table, paths.png, paths.pdf, paths.metadata):
        assert path.is_file(), path
    assert paths.png.stat().st_size > 1000
    assert paths.pdf.stat().st_size > 1000


# =============================================================== figure 1 ==
def test_figure1_computes_the_accuracy_points(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_pear_run(root, "clean", n_agents=4, examples=EXAMPLES, accuracy=1.0)
        make_pear_run(
            root,
            "attacked",
            n_agents=4,
            examples=EXAMPLES,
            accuracy=1 / 3,
            attacker_ids=(4,),
            attack_value=5,
        )

    result = figure1.build_figure(collected(tmp_path, builder), repetitions=200)
    table = result["table"].set_index("adversarial_fraction")

    assert table.loc[0.0, "mean_accuracy"] == pytest.approx(1.0)
    assert table.loc[0.25, "mean_accuracy"] == pytest.approx(1 / 3)
    assert table.loc[0.0, "n_examples"] == 3
    assert table.loc[0.0, "n_replications"] == 6  # 3 examples x 2 perm seeds
    artifacts_exist(result["paths"])


def test_figure1_uses_the_clean_arm_as_the_zero_point(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_pear_run(root, "clean", n_agents=4, examples=EXAMPLES)
        make_pear_run(root, "one", n_agents=4, examples=EXAMPLES, attacker_ids=(4,), attack_value=5)
        make_pear_run(
            root, "two", n_agents=4, examples=EXAMPLES, attacker_ids=(3, 4), attack_value=5
        )

    table = figure1.build_figure(
        collected(tmp_path, builder), repetitions=100
    )["table"]

    assert sorted(table["adversarial_fraction"].unique()) == [0.0, 0.25, 0.5]
    # The clean row belongs to the attack curve, not to a curve of its own.
    assert set(table["attack_type"].unique()) == {"confidence_inflation"}


def test_figure1_refuses_a_single_fraction(tmp_path: Path) -> None:
    analysis_dir = collected(tmp_path, lambda root: make_pear_run(root, "clean"))

    with pytest.raises(FigureError, match="clean 0.0 arm and at least one attacked"):
        figure1.build_figure(analysis_dir)


def test_figure1_includes_a_method_without_routing(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_generic_run(root, "gen_clean", examples=EXAMPLES, adversarial_fraction=0.0)
        make_generic_run(root, "gen_attacked", examples=EXAMPLES, adversarial_fraction=0.5)

    table = figure1.build_figure(
        collected(tmp_path, builder), repetitions=100
    )["table"]

    assert set(table["method"]) == {"single_agent_baseline"}
    assert sorted(table["adversarial_fraction"].unique()) == [0.0, 0.5]


def test_figure1_keeps_datasets_apart(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        for dataset in ("bench_a", "bench_b"):
            make_pear_run(root, f"{dataset}_clean", dataset=dataset, n_agents=4, examples=EXAMPLES)
            make_pear_run(
                root,
                f"{dataset}_attacked",
                dataset=dataset,
                n_agents=4,
                examples=EXAMPLES,
                attacker_ids=(4,),
                attack_value=5,
            )

    table = figure1.build_figure(
        collected(tmp_path, builder), repetitions=100
    )["table"]

    assert sorted(table["dataset"].unique()) == ["bench_a", "bench_b"]
    assert len(table) == 4  # two datasets x two fractions, never averaged together


# =============================================================== figure 2 ==
def test_figure2_pairs_arms_and_finds_the_best_report(tmp_path: Path) -> None:
    analysis_dir = collected(tmp_path, full_sweep)

    result = figure2.build_figure(
        analysis_dir, w_rho=1.0, w_adoption=1.0, repetitions=200
    )
    table = result["table"]

    assert sorted(table["report_value"]) == list(REPORT_VALUES)
    assert (table["n_pairs"] == 6).all()  # 3 examples x 2 perm seeds
    assert (table["n_examples"] == 3).all()
    assert (table["attacker_id"] == 4).all()
    # Exposure rises with the report in the fixture, so the highest report wins.
    best = table[table["is_best_report"]].iloc[0]
    assert best["report_value"] == 5
    assert best["utility_gain"] > 0
    artifacts_exist(result["paths"])


def test_figure2_records_the_utility_weights(tmp_path: Path) -> None:
    analysis_dir = collected(tmp_path, full_sweep)

    result = figure2.build_figure(
        analysis_dir, w_rho=2.0, w_adoption=0.5, repetitions=100
    )

    assert (result["table"]["w_rho"] == 2.0).all()
    assert (result["table"]["w_adoption"] == 0.5).all()
    assert (result["table"]["penalty"] == 0.0).all()
    metadata = json.loads(result["paths"].metadata.read_text(encoding="utf-8"))
    assert metadata["w_rho"] == 2.0
    assert metadata["w_adoption"] == 0.5
    assert metadata["penalty_note"].startswith("P_i = 0")
    assert "not exact or universal" in metadata["not"]
    assert len(metadata["limitations"]) >= 3


def test_figure2_weights_change_the_gain(tmp_path: Path) -> None:
    analysis_dir = collected(tmp_path, full_sweep)

    single = figure2.build_figure(
        analysis_dir, w_rho=1.0, w_adoption=0.0, repetitions=100
    )["table"]
    double = figure2.build_figure(
        analysis_dir, w_rho=2.0, w_adoption=0.0, repetitions=100
    )["table"]

    assert double["utility_gain"].sum() == pytest.approx(2 * single["utility_gain"].sum())


def test_figure2_refuses_unmatched_coverage(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_pear_run(
            root, "truthful", examples=EXAMPLES, dissenting_ids=(4,), perm_seeds=(10, 11)
        )
        for value in REPORT_VALUES:
            make_pear_run(
                root,
                f"report_{value}",
                examples=EXAMPLES,
                attacker_ids=(4,),
                attack_value=value,
                perm_seeds=(10,) if value == 2 else (10, 11),
            )

    analysis_dir = collected(tmp_path, builder)

    with pytest.raises(FigureError) as excinfo:
        figure2.build_figure(analysis_dir, w_rho=1.0, w_adoption=1.0)

    assert "not paired" in str(excinfo.value)


def test_figure2_refuses_multiple_attackers(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_pear_run(root, "truthful", examples=EXAMPLES, dissenting_ids=(3, 4))
        for value in REPORT_VALUES:
            make_pear_run(
                root, f"report_{value}", examples=EXAMPLES, attacker_ids=(3, 4), attack_value=value
            )

    analysis_dir = collected(tmp_path, builder)

    with pytest.raises(FigureError, match="exactly one confidence attacker"):
        figure2.build_figure(analysis_dir, w_rho=1.0, w_adoption=1.0)


def test_figure2_names_the_missing_report_values(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_pear_run(root, "truthful", examples=EXAMPLES, dissenting_ids=(4,))
        for value in (1, 2):
            make_pear_run(
                root, f"report_{value}", examples=EXAMPLES, attacker_ids=(4,), attack_value=value
            )

    analysis_dir = collected(tmp_path, builder)

    with pytest.raises(FigureError, match=r"missing fixed_report values \[3, 4, 5\]"):
        figure2.build_figure(analysis_dir, w_rho=1.0, w_adoption=1.0)


def test_figure2_adoption_term_uses_the_canonical_parser(tmp_path: Path) -> None:
    """The group adopting the attacker's answer is worth w_adoption."""

    def builder(root: Path) -> None:
        make_pear_run(root, "truthful", dataset="gsm8k", examples=EXAMPLES, dissenting_ids=(4,))
        for value in REPORT_VALUES:
            make_pear_run(
                root,
                f"report_{value}",
                dataset="gsm8k",
                examples=EXAMPLES,
                attacker_ids=(4,),
                attack_value=value,
                # The debate ends on the attacker's answer in the attacked arms.
                adopts_attacker_answer=value == 5,
            )

    result = figure2.build_figure(
        collected(tmp_path, builder), w_rho=0.0, w_adoption=1.0, repetitions=100
    )
    table = result["table"].set_index("report_value")

    assert table.loc[5, "utility_gain"] == pytest.approx(1.0)
    assert table.loc[1, "utility_gain"] == pytest.approx(0.0)
    metadata = json.loads(result["paths"].metadata.read_text(encoding="utf-8"))
    assert metadata["answer_parser"] == "gsm8k.parse_answer"


def test_figure2_missing_history_is_not_zero_utility(tmp_path: Path) -> None:
    analysis_dir = collected(tmp_path, full_sweep)
    results_path = analysis_dir / "tables" / "results.csv"
    results = pd.read_csv(results_path)
    results.loc[0, "influence_history"] = None
    results.to_csv(results_path, index=False)

    with pytest.raises(FigureError, match="Missing history is not zero utility"):
        figure2.build_figure(analysis_dir, w_rho=1.0, w_adoption=1.0)


# =============================================================== figure 4 ==
def test_figure4_includes_a_generic_competitor(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_pear_run(root, "pear_clean", examples=EXAMPLES, rounds=1)
        make_generic_run(root, "competitor", examples=EXAMPLES, prompt_tokens=300, completion_tokens=60)

    result = figure4.build_figure(collected(tmp_path, builder), repetitions=100)
    table = result["table"].set_index("method")

    assert set(table.index) == {"pear", "single_agent_baseline"}
    assert table.loc["single_agent_baseline", "mean_total_tokens"] == 360
    assert table.loc["pear", "mean_total_tokens"] == 1000 + 240  # two turns of tokens
    artifacts_exist(result["paths"])


def test_figure4_excludes_methods_without_token_usage(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_pear_run(root, "pear_clean", examples=EXAMPLES)
        make_generic_run(root, "competitor", examples=EXAMPLES, include_tokens=False)

    result = figure4.build_figure(collected(tmp_path, builder), repetitions=100)

    assert set(result["table"]["method"]) == {"pear"}
    assert result["excluded"] == [
        {
            "method": "single_agent_baseline",
            "reason": (
                "no token usage recorded; excluded from the cost axis rather "
                "than plotted at zero cost"
            ),
        }
    ]
    # The excluded method must not appear anywhere on the cost axis, at zero or
    # otherwise.
    assert (result["table"]["mean_total_tokens"] > 0).all()


def test_figure4_keeps_a_clean_verified_condition(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_pear_run(root, "plain", examples=EXAMPLES)
        make_pear_run(root, "verified", examples=EXAMPLES, verification_mode="oracle")

    table = figure4.build_figure(
        collected(tmp_path, builder), repetitions=100
    )["table"]

    assert sorted(table["verification_mode"].unique()) == ["none", "oracle"]


def test_figure4_needs_a_clean_arm(tmp_path: Path) -> None:
    analysis_dir = collected(
        tmp_path,
        lambda root: make_pear_run(root, "attacked", attacker_ids=(4,), attack_value=4),
    )

    with pytest.raises(FigureError, match="no clean"):
        figure4.build_figure(analysis_dir)


def test_figure4_refuses_when_nothing_is_priced(tmp_path: Path) -> None:
    analysis_dir = collected(
        tmp_path, lambda root: make_pear_run(root, "clean", include_tokens=False)
    )

    with pytest.raises(FigureError, match="never a cost of zero"):
        figure4.build_figure(analysis_dir)


# =============================================================== figure 5 ==
def grid_builder(cells):
    def builder(root: Path) -> None:
        for temperature, alpha in cells:
            make_pear_run(
                root,
                f"t{temperature}_a{alpha}",
                examples=EXAMPLES,
                routing_temperature=temperature,
                alpha_influence=alpha,
            )

    return builder


def test_figure5_draws_a_complete_grid(tmp_path: Path) -> None:
    cells = [(0.3, 0.4), (0.3, 0.8), (0.7, 0.4), (0.7, 0.8)]
    analysis_dir = collected(tmp_path, grid_builder(cells))

    result = figure5.build_figure(analysis_dir, metric="accuracy")

    assert len(result["table"]) == 4
    assert sorted(result["table"]["routing_temperature"].unique()) == [0.3, 0.7]
    assert sorted(result["table"]["alpha_influence"].unique()) == [0.4, 0.8]
    assert result["rejected"] == []
    artifacts_exist(result["paths"])
    assert result["paths"].table.name == "figure5_heatmap_accuracy.csv"


def test_figure5_refuses_an_incomplete_grid(tmp_path: Path) -> None:
    analysis_dir = collected(tmp_path, grid_builder([(0.3, 0.4), (0.3, 0.8), (0.7, 0.4)]))

    with pytest.raises(FigureError, match="complete grid of at least 2 x 2"):
        figure5.build_figure(analysis_dir, metric="accuracy")


def test_figure5_refuses_a_one_by_two_grid(tmp_path: Path) -> None:
    analysis_dir = collected(tmp_path, grid_builder([(0.7, 0.4), (0.7, 0.8)]))

    with pytest.raises(FigureError, match="2 x 2"):
        figure5.build_figure(analysis_dir, metric="accuracy")


def test_figure5_robustness_gap_needs_matched_arms(tmp_path: Path) -> None:
    cells = [(0.3, 0.4), (0.3, 0.8), (0.7, 0.4), (0.7, 0.8)]
    analysis_dir = collected(tmp_path, grid_builder(cells))

    with pytest.raises(FigureError, match="matched clean and attacked"):
        figure5.build_figure(analysis_dir, metric="robustness_gap")


def test_figure5_robustness_gap_on_matched_arms(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        for temperature in (0.3, 0.7):
            for alpha in (0.4, 0.8):
                make_pear_run(
                    root,
                    f"clean_t{temperature}_a{alpha}",
                    examples=EXAMPLES,
                    n_agents=4,
                    accuracy=1.0,
                    routing_temperature=temperature,
                    alpha_influence=alpha,
                )
                make_pear_run(
                    root,
                    f"att_t{temperature}_a{alpha}",
                    examples=EXAMPLES,
                    n_agents=4,
                    accuracy=1 / 3,
                    attacker_ids=(4,),
                    attack_value=5,
                    routing_temperature=temperature,
                    alpha_influence=alpha,
                )

    result = figure5.build_figure(collected(tmp_path, builder), metric="robustness_gap")

    assert len(result["table"]) == 4
    assert result["table"]["value"].round(4).unique().tolist() == [pytest.approx(0.6667, abs=1e-3)]


def test_figure5_rejects_an_unknown_metric(tmp_path: Path) -> None:
    analysis_dir = collected(tmp_path, grid_builder([(0.3, 0.4), (0.7, 0.8)]))

    with pytest.raises(FigureError, match="unknown metric"):
        figure5.build_figure(analysis_dir, metric="vibes")


def test_figure5_ic_regret_requires_figure2_first(tmp_path: Path) -> None:
    cells = [(0.3, 0.4), (0.3, 0.8), (0.7, 0.4), (0.7, 0.8)]
    analysis_dir = collected(tmp_path, grid_builder(cells))

    with pytest.raises(FigureError, match="Generate Figure 2 first"):
        figure5.build_figure(analysis_dir, metric="empirical_ic_regret")


# ========================================================= master command ==
def test_master_generates_available_figures_and_names_the_skips(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    full_sweep(root)

    summary = headline_run(
        [root],
        tmp_path / "artifacts",
        w_rho=1.0,
        w_adoption=1.0,
        heatmap_metric="accuracy",
        repetitions=100,
        quiet=True,
    )

    generated = {entry["figure"] for entry in summary["generated"]}
    skipped = {entry["figure"]: entry["reason"] for entry in summary["skipped"]}
    assert {1, 2, 3, 4} <= generated
    assert 5 in skipped
    assert "2 x 2" in skipped[5]
    assert (tmp_path / "artifacts" / "readiness.json").is_file()
    assert (tmp_path / "artifacts" / "headline_analysis.json").is_file()


def test_master_can_select_figures(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    full_sweep(root)

    summary = headline_run(
        [root], tmp_path / "artifacts", figures=[3, 4], repetitions=100, quiet=True
    )

    assert {entry["figure"] for entry in summary["generated"]} == {3, 4}
    assert summary["requested_figures"] == [3, 4]


def test_master_requires_utility_weights_for_figure2(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    full_sweep(root)

    summary = headline_run(
        [root], tmp_path / "artifacts", figures=[2], repetitions=100, quiet=True
    )

    assert summary["generated"] == []
    assert "--w-rho" in summary["skipped"][0]["reason"]


def test_master_requires_a_heatmap_metric(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    full_sweep(root)

    summary = headline_run(
        [root], tmp_path / "artifacts", figures=[5], repetitions=100, quiet=True
    )

    assert "No metric is selected automatically" in summary["skipped"][0]["reason"]


def test_master_strict_exits_non_zero(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    full_sweep(root)
    base = [
        str(root),
        "--output",
        str(tmp_path / "artifacts"),
        "--bootstrap-repetitions",
        "100",
        "--heatmap-metric",
        "accuracy",
    ]
    with_missing_figure = [*base, "--figures", "3", "5"]
    only_available = [*base, "--figures", "3"]

    # Figure 5 has no parameter grid here, so it is skipped ...
    assert headline_main(with_missing_figure) == 0
    # ... which --strict turns into a CI failure.
    assert headline_main([*with_missing_figure, "--strict"]) == 1
    assert headline_main([*only_available, "--strict"]) == 0


def test_master_prints_skip_reasons(tmp_path: Path, capsys) -> None:
    root = tmp_path / "runs"
    make_pear_run(root, "clean", examples=EXAMPLES)

    code = headline_main(
        [
            str(root),
            "--output",
            str(tmp_path / "artifacts"),
            "--bootstrap-repetitions",
            "50",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0  # nothing requested was mandatory
    assert "SKIPPED    Figure 3" in out
    assert "is_attacker" in out
    assert "SKIPPED    Figure 1" in out


def test_master_keeps_a_no_routing_competitor_usable(tmp_path: Path) -> None:
    def builder(root: Path) -> None:
        make_generic_run(root, "gen_clean", examples=EXAMPLES, adversarial_fraction=0.0)
        make_generic_run(root, "gen_attacked", examples=EXAMPLES, adversarial_fraction=0.5)

    root = tmp_path / "runs"
    builder(root)

    summary = headline_run(
        [root], tmp_path / "artifacts", repetitions=100, quiet=True
    )

    generated = {entry["figure"] for entry in summary["generated"]}
    skipped = {entry["figure"]: entry["reason"] for entry in summary["skipped"]}
    assert {1, 4} <= generated  # accuracy and cost still work
    assert "routing" in skipped[3]
    assert "single_agent_baseline" in skipped[3]
