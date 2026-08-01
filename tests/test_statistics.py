"""Tests for the shared cluster-bootstrap helpers.

The property worth protecting is the clustering itself: replications of one
question must never look like independent evidence. The tests below detect
that by construction rather than by inspecting internals -- a bootstrap that
resamples rows instead of questions cannot produce the intervals asserted
here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.statistics import (
    BOOTSTRAP_COLUMNS,
    DEFAULT_ANALYSIS_SEED,
    StatisticsError,
    average_within,
    cluster_bootstrap_mean,
    describe_replication,
    paired_cluster_bootstrap_difference,
)


def two_example_frame() -> pd.DataFrame:
    """Two questions, ten replications each; the answer differs per question."""
    rows = []
    for seed in range(10):
        rows.append({"example_id": "a", "seed": seed, "value": 0.0, "arm": "x"})
        rows.append({"example_id": "b", "seed": seed, "value": 1.0, "arm": "x"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ determinism --
def test_bootstrap_is_deterministic_under_a_fixed_seed() -> None:
    frame = two_example_frame()

    first = cluster_bootstrap_mean(frame, value_col="value", repetitions=200)
    second = cluster_bootstrap_mean(frame, value_col="value", repetitions=200)

    pd.testing.assert_frame_equal(first, second)


def test_a_different_analysis_seed_changes_the_interval() -> None:
    rows = [
        {"example_id": f"ex-{i}", "value": float(i % 4), "seed": 0} for i in range(24)
    ]
    frame = pd.DataFrame(rows)

    a = cluster_bootstrap_mean(frame, value_col="value", repetitions=200, analysis_seed=1)
    b = cluster_bootstrap_mean(frame, value_col="value", repetitions=200, analysis_seed=2)

    assert a["mean"].iloc[0] == b["mean"].iloc[0]
    assert (a["ci_lower"].iloc[0], a["ci_upper"].iloc[0]) != (
        b["ci_lower"].iloc[0],
        b["ci_upper"].iloc[0],
    )


def test_group_intervals_do_not_depend_on_the_other_groups() -> None:
    """Adding a condition must not move an existing condition's interval."""
    frame = pd.DataFrame(
        [
            {"example_id": f"ex-{i}", "arm": arm, "value": float((i + len(arm)) % 3)}
            for i in range(12)
            for arm in ("a", "b")
        ]
    )

    together = cluster_bootstrap_mean(
        frame, value_col="value", group_cols=["arm"], repetitions=200
    )
    alone = cluster_bootstrap_mean(
        frame[frame["arm"] == "a"], value_col="value", group_cols=["arm"], repetitions=200
    )

    pd.testing.assert_frame_equal(
        together[together["arm"] == "a"].reset_index(drop=True), alone.reset_index(drop=True)
    )


# ------------------------------------------------------------- clustering --
def test_replications_of_one_example_stay_in_one_cluster() -> None:
    """With two questions the resampled mean can only be 0, 0.5 or 1.

    A row-level bootstrap over twenty rows would concentrate near 0.5 and give
    a much narrower interval; the wide interval is the evidence that seeds were
    not counted as questions.
    """
    frame = two_example_frame()

    result = cluster_bootstrap_mean(frame, value_col="value", repetitions=500).iloc[0]

    assert result["mean"] == 0.5
    assert result["n_examples"] == 2
    assert result["n_raw_rows"] == 20
    assert result["ci_lower"] == 0.0
    assert result["ci_upper"] == 1.0


def test_cluster_column_can_be_overridden() -> None:
    frame = two_example_frame().rename(columns={"example_id": "question_id"})

    result = cluster_bootstrap_mean(
        frame, value_col="value", cluster_col="question_id", repetitions=100
    ).iloc[0]

    assert result["n_examples"] == 2


def test_single_cluster_reports_no_interval() -> None:
    frame = pd.DataFrame([{"example_id": "a", "value": v} for v in (0.0, 1.0, 0.5)])

    result = cluster_bootstrap_mean(frame, value_col="value", repetitions=100).iloc[0]

    assert result["mean"] == 0.5
    assert pd.isna(result["ci_lower"]) and pd.isna(result["ci_upper"])
    assert result["n_examples"] == 1


def test_result_columns_and_provenance() -> None:
    frame = two_example_frame()

    result = cluster_bootstrap_mean(
        frame, value_col="value", group_cols=["arm"], repetitions=123
    )

    assert list(result.columns) == ["arm", *BOOTSTRAP_COLUMNS]
    assert result["bootstrap_repetitions"].iloc[0] == 123
    assert result["analysis_seed"].iloc[0] == DEFAULT_ANALYSIS_SEED
    assert result["confidence_level"].iloc[0] == 0.95


def test_missing_values_are_dropped_not_zero_filled() -> None:
    frame = pd.DataFrame(
        [
            {"example_id": "a", "value": 1.0},
            {"example_id": "b", "value": None},
            {"example_id": "c", "value": 1.0},
        ]
    )

    result = cluster_bootstrap_mean(frame, value_col="value", repetitions=50).iloc[0]

    assert result["mean"] == 1.0  # a zero-fill would make this 2/3
    assert result["n_examples"] == 2


def test_missing_column_raises() -> None:
    with pytest.raises(StatisticsError, match="missing column"):
        cluster_bootstrap_mean(pd.DataFrame({"a": [1]}), value_col="value")


# ---------------------------------------------------------------- paired --
def paired_frame() -> pd.DataFrame:
    rows = []
    for example in ("a", "b", "c"):
        for seed in (0, 1):
            rows.append(
                {"example_id": example, "seed": seed, "arm": "truthful", "utility": 0.1}
            )
            rows.append(
                {"example_id": example, "seed": seed, "arm": "report_5", "utility": 0.4}
            )
    return pd.DataFrame(rows)


def test_paired_difference_basic() -> None:
    result = paired_cluster_bootstrap_difference(
        paired_frame(),
        value_col="utility",
        arm_col="arm",
        arm_a="truthful",
        arm_b="report_5",
        pair_cols=["example_id", "seed"],
        repetitions=200,
    )

    assert result.mean_difference == pytest.approx(0.3)
    assert result.n_pairs == 6
    assert result.n_examples == 3
    assert result.bootstrap_repetitions == 200
    assert result.mean_arm_a == pytest.approx(0.1)
    assert result.mean_arm_b == pytest.approx(0.4)


def test_paired_difference_rejects_unmatched_coverage() -> None:
    frame = paired_frame()
    frame = frame[~((frame["example_id"] == "c") & (frame["arm"] == "report_5"))]

    with pytest.raises(StatisticsError, match="present in only one"):
        paired_cluster_bootstrap_difference(
            frame,
            value_col="utility",
            arm_col="arm",
            arm_a="truthful",
            arm_b="report_5",
            pair_cols=["example_id", "seed"],
        )


def test_paired_difference_rejects_a_missing_arm() -> None:
    frame = paired_frame()

    with pytest.raises(StatisticsError, match="absent from column"):
        paired_cluster_bootstrap_difference(
            frame,
            value_col="utility",
            arm_col="arm",
            arm_a="truthful",
            arm_b="report_3",
            pair_cols=["example_id", "seed"],
        )


def test_paired_difference_rejects_duplicate_pairs() -> None:
    frame = pd.concat([paired_frame(), paired_frame().head(2)], ignore_index=True)

    with pytest.raises(StatisticsError, match="duplicate row"):
        paired_cluster_bootstrap_difference(
            frame,
            value_col="utility",
            arm_col="arm",
            arm_a="truthful",
            arm_b="report_5",
            pair_cols=["example_id", "seed"],
        )


def test_paired_difference_clusters_by_example() -> None:
    """Two questions with opposite effects give the widest possible interval."""
    rows = []
    for seed in range(5):
        rows.append({"example_id": "a", "seed": seed, "arm": "truthful", "u": 0.0})
        rows.append({"example_id": "a", "seed": seed, "arm": "report_5", "u": 1.0})
        rows.append({"example_id": "b", "seed": seed, "arm": "truthful", "u": 0.0})
        rows.append({"example_id": "b", "seed": seed, "arm": "report_5", "u": -1.0})

    result = paired_cluster_bootstrap_difference(
        pd.DataFrame(rows),
        value_col="u",
        arm_col="arm",
        arm_a="truthful",
        arm_b="report_5",
        pair_cols=["example_id", "seed"],
        repetitions=500,
    )

    assert result.mean_difference == pytest.approx(0.0)
    assert result.n_examples == 2
    assert (result.ci_lower, result.ci_upper) == (-1.0, 1.0)


def test_paired_difference_requires_cluster_in_pair_cols() -> None:
    with pytest.raises(StatisticsError, match="must be one of the pairing columns"):
        paired_cluster_bootstrap_difference(
            paired_frame(),
            value_col="utility",
            arm_col="arm",
            arm_a="truthful",
            arm_b="report_5",
            pair_cols=["seed"],
        )


# ---------------------------------------------------------------- helpers --
def test_average_within_collapses_rounds() -> None:
    rows = [
        {"example_id": "a", "seed": 0, "round": r, "share": 0.2 * r} for r in (1, 2, 3)
    ]

    collapsed = average_within(
        pd.DataFrame(rows), value_col="share", within_cols=["example_id", "seed"]
    )

    assert len(collapsed) == 1
    assert collapsed["share"].iloc[0] == pytest.approx(0.4)


def test_describe_replication_counts() -> None:
    frame = pd.DataFrame(
        [
            {"example_id": "a", "seed": 0, "perm_seed": 10},
            {"example_id": "a", "seed": 0, "perm_seed": 11},
            {"example_id": "b", "seed": 1, "perm_seed": 10},
        ]
    )

    assert describe_replication(frame) == {
        "n_examples": 2,
        "n_raw_rows": 3,
        "n_seeds": 2,
        "n_perm_seeds": 2,
    }
