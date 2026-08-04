"""Tests for Figure 3 -- adversary routing exposure vs reported confidence.

Two kinds of test: hand-written tables where every plotted number can be
worked out by hand, and end-to-end runs through the real fixtures so the
script is exercised against the normalised tables it will actually read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.collect_runs import collect
from analysis.figure_exposure_vs_confidence import SLUG, build_figure, main
from analysis.figure_utils import FigureError
from tests.analysis_fixtures import make_generic_run, make_pear_run

REPORT_VALUES = (1, 2, 3, 4, 5)


# ------------------------------------------------------- hand-built tables --
def write_tables(
    analysis_dir: Path, routing_rows: list[dict], runs_rows: list[dict] | None = None
) -> Path:
    """Write a minimal analysis artifact directory with exact values."""
    tables = analysis_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    routing = pd.DataFrame(routing_rows)
    routing.to_csv(tables / "routing.csv", index=False)
    runs = pd.DataFrame(
        runs_rows
        or [
            {
                "run_id": "exp/run",
                "method": "pear",
                "has_routing": True,
                "has_accuracy": True,
                "has_token_usage": True,
            }
        ]
    )
    runs.to_csv(tables / "runs.csv", index=False)
    pd.DataFrame(columns=["example_id"]).to_csv(tables / "results.csv", index=False)
    return analysis_dir


def routing_row(**overrides) -> dict:
    row = {
        "run_id": "exp/run",
        "condition": "pear_full",
        "dataset": "bench",
        "model": "some-model",
        "mechanism": "pear_full",
        "verification_mode": "none",
        "routing_mode": "sampled",
        "example_id": "ex-1",
        "seed": 0,
        "perm_seed": 10,
        "round": 1,
        "agent_id": 5,
        "is_attacker": True,
        "reported_confidence": 3.0,
        "out_degree_share": 0.2,
        "source_eligible": False,
        "target_eligible": True,
    }
    row.update(overrides)
    return row


def test_exact_plotted_values(tmp_path: Path) -> None:
    """Two questions x two reports; every number below is arithmetic by hand."""
    rows = []
    for example, shares in (("ex-1", (0.2, 0.4)), ("ex-2", (0.3, 0.5))):
        for confidence, share in zip((1.0, 5.0), shares):
            rows.append(
                routing_row(example_id=example, reported_confidence=confidence, out_degree_share=share)
            )
    write_tables(tmp_path, rows)

    table = build_figure(tmp_path, repetitions=200)["table"]

    at_one = table[table["reported_confidence"] == 1.0].iloc[0]
    at_five = table[table["reported_confidence"] == 5.0].iloc[0]
    assert at_one["mean_out_degree_share"] == pytest.approx(0.25)
    assert at_five["mean_out_degree_share"] == pytest.approx(0.45)
    assert at_one["n_examples"] == 2
    assert at_one["n_raw_rows"] == 2
    assert at_one["source_eligibility_rate"] == 0.0
    assert at_one["target_eligibility_rate"] == 1.0
    assert at_one["analysis_seed"] == 20260801
    assert at_one["bootstrap_repetitions"] == 200


def test_only_attacker_rows_are_used(tmp_path: Path) -> None:
    rows = [
        routing_row(example_id="ex-1", reported_confidence=1.0, out_degree_share=0.2),
        routing_row(example_id="ex-2", reported_confidence=5.0, out_degree_share=0.4),
        # Non-attackers in the same decisions, with wildly different exposure.
        routing_row(
            example_id="ex-1",
            agent_id=1,
            is_attacker=False,
            reported_confidence=1.0,
            out_degree_share=0.9,
        ),
        routing_row(
            example_id="ex-2",
            agent_id=2,
            is_attacker=False,
            reported_confidence=5.0,
            out_degree_share=0.9,
        ),
    ]
    write_tables(tmp_path, rows)

    table = build_figure(tmp_path, repetitions=100)["table"]

    assert table["mean_out_degree_share"].tolist() == [0.2, 0.4]


def test_rounds_are_averaged_not_counted_as_examples(tmp_path: Path) -> None:
    """Three rounds of one debate are one observation, not three."""
    rows = []
    for example in ("ex-1", "ex-2"):
        for confidence in (1.0, 5.0):
            for round_idx, share in enumerate((0.1, 0.2, 0.3), start=1):
                rows.append(
                    routing_row(
                        example_id=example,
                        reported_confidence=confidence,
                        round=round_idx,
                        out_degree_share=share,
                    )
                )
    write_tables(tmp_path, rows)

    table = build_figure(tmp_path, repetitions=100)

    row = table["table"].iloc[0]
    assert row["mean_out_degree_share"] == pytest.approx(0.2)  # mean of the rounds
    assert row["n_examples"] == 2
    assert row["n_raw_rows"] == 2  # two debates, not six rounds
    assert table["metadata"]["round_handling"].startswith("rounds averaged")


def test_replications_of_one_example_share_a_cluster(tmp_path: Path) -> None:
    """Opposite answers on two questions give the widest possible interval."""
    rows = []
    for example, share in (("ex-1", 0.0), ("ex-2", 1.0)):
        for perm_seed in (10, 11, 12, 13):
            for confidence in (1.0, 5.0):
                rows.append(
                    routing_row(
                        example_id=example,
                        perm_seed=perm_seed,
                        reported_confidence=confidence,
                        out_degree_share=share,
                    )
                )
    write_tables(tmp_path, rows)

    row = build_figure(tmp_path, repetitions=500)["table"].iloc[0]

    assert row["mean_out_degree_share"] == pytest.approx(0.5)
    assert row["n_examples"] == 2
    assert row["n_raw_rows"] == 8
    assert (row["ci_lower"], row["ci_upper"]) == (0.0, 1.0)
    assert row["n_perm_seeds"] == 4


def test_refuses_a_single_confidence_value(tmp_path: Path) -> None:
    write_tables(
        tmp_path,
        [
            routing_row(example_id="ex-1", reported_confidence=4.0),
            routing_row(example_id="ex-2", reported_confidence=4.0),
        ],
    )

    with pytest.raises(FigureError, match="two or more distinct reported-confidence"):
        build_figure(tmp_path)


def test_routing_modes_are_separated_not_pooled(tmp_path: Path) -> None:
    rows = []
    for mode, share in (("sampled", 0.2), ("enumerated", 0.6)):
        for example in ("ex-1", "ex-2"):
            for confidence in (1.0, 5.0):
                rows.append(
                    routing_row(
                        routing_mode=mode,
                        example_id=example,
                        reported_confidence=confidence,
                        out_degree_share=share,
                    )
                )
    write_tables(tmp_path, rows)

    table = build_figure(tmp_path, repetitions=100)["table"]

    assert sorted(table["routing_mode"].unique()) == ["enumerated", "sampled"]
    assert len(table) == 4  # two modes x two confidence values, never merged
    assert set(table[table["routing_mode"] == "enumerated"]["mean_out_degree_share"]) == {0.6}


def test_verification_modes_are_separate_curves(tmp_path: Path) -> None:
    rows = []
    for mode, share in (("none", 0.4), ("oracle", 0.1)):
        for example in ("ex-1", "ex-2"):
            for confidence in (1.0, 5.0):
                rows.append(
                    routing_row(
                        verification_mode=mode,
                        example_id=example,
                        reported_confidence=confidence,
                        out_degree_share=share,
                    )
                )
    write_tables(tmp_path, rows)

    table = build_figure(tmp_path, repetitions=100)["table"]

    assert sorted(table["verification_mode"].unique()) == ["none", "oracle"]
    # The x-axis stays the *reported* report even when the oracle overrode it.
    assert sorted(table["reported_confidence"].unique()) == [1.0, 5.0]


# ----------------------------------------------------------- end-to-end --
def collected(tmp_path: Path, builder) -> Path:
    root = tmp_path / "runs"
    builder(root)
    collect([root], tmp_path / "artifacts", quiet=True)
    return tmp_path / "artifacts"


def full_sweep(root: Path) -> None:
    make_pear_run(root, "truthful")
    for value in REPORT_VALUES:
        make_pear_run(root, f"report_{value}", attacker_ids=(4,), attack_value=value)


def test_end_to_end_writes_all_four_artifacts(tmp_path: Path) -> None:
    analysis_dir = collected(tmp_path, full_sweep)

    result = build_figure(analysis_dir, repetitions=200)

    for path in (
        result["paths"].table,
        result["paths"].png,
        result["paths"].pdf,
        result["paths"].metadata,
    ):
        assert path.is_file(), path
    assert result["paths"].png.stat().st_size > 1000
    assert result["paths"].pdf.stat().st_size > 1000


def test_end_to_end_covers_the_whole_rubric(tmp_path: Path) -> None:
    analysis_dir = collected(tmp_path, full_sweep)

    metadata = build_figure(analysis_dir, repetitions=200)["metadata"]

    assert metadata["reported_confidence_values"] == list(REPORT_VALUES)
    assert metadata["complete_sweep"] is True
    assert metadata["routing_modes"] == ["sampled"]


def test_every_plotted_point_traces_back_to_routing_rows(tmp_path: Path) -> None:
    analysis_dir = collected(tmp_path, full_sweep)

    table = build_figure(analysis_dir, repetitions=200)["table"]
    routing = pd.read_csv(analysis_dir / "tables" / "routing.csv")
    attackers = routing[routing["is_attacker"]]

    for _, point in table.iterrows():
        matching = attackers[
            (attackers["reported_confidence"] == point["reported_confidence"])
            & (attackers["routing_mode"] == point["routing_mode"])
            & (attackers["verification_mode"] == point["verification_mode"])
        ]
        assert not matching.empty
        assert matching["example_id"].nunique() == point["n_examples"]


def test_csv_is_written_and_routing_table_untouched(tmp_path: Path) -> None:
    analysis_dir = collected(tmp_path, full_sweep)
    before = (analysis_dir / "tables" / "routing.csv").read_bytes()

    build_figure(analysis_dir, repetitions=100)

    assert (analysis_dir / "tables" / f"{SLUG}.csv").is_file()
    assert (analysis_dir / "tables" / "routing.csv").read_bytes() == before


def test_generic_competitor_is_excluded_with_a_capability_reason(tmp_path: Path) -> None:
    analysis_dir = collected(tmp_path, lambda root: make_generic_run(root, "competitor"))

    with pytest.raises(FigureError) as excinfo:
        build_figure(analysis_dir)

    message = str(excinfo.value)
    assert "no input method logs routing decisions" in message
    assert "single_agent_baseline" in message
    assert "capability limitation" in message
    assert "eligible for the accuracy and cost figures" in message


def test_clean_only_sweep_is_refused(tmp_path: Path) -> None:
    analysis_dir = collected(tmp_path, lambda root: make_pear_run(root, "clean"))

    with pytest.raises(FigureError, match="is_attacker"):
        build_figure(analysis_dir)


def test_cli_reports_failure_without_crashing(tmp_path: Path, capsys) -> None:
    analysis_dir = collected(tmp_path, lambda root: make_pear_run(root, "clean"))

    assert main([str(analysis_dir)]) == 1
    assert "Figure 3 not produced" in capsys.readouterr().err


def test_cli_writes_metadata_with_the_command(tmp_path: Path) -> None:
    analysis_dir = collected(tmp_path, full_sweep)

    assert main([str(analysis_dir), "--bootstrap-repetitions", "100"]) == 0

    metadata = json.loads((analysis_dir / f"{SLUG}.meta.json").read_text(encoding="utf-8"))
    assert metadata["bootstrap_repetitions"] == 100
    assert metadata["grouping_columns"][0] == "dataset"
    assert "rho" in metadata["quantity"]
