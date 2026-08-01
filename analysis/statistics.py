"""Cluster bootstrap helpers shared by every figure script.

One idea underlies both functions: **the question is the unit of evidence**.

A single benchmark example is replicated across decoding seeds, agent
permutation seeds and debate rounds. Those replications are not independent
observations of "how often is this system right" -- they are repeated looks at
the same question. Treating them as independent shrinks every interval by
roughly the square root of the replication count, which is how a mock sweep of
three questions ends up with error bars that look like a study of sixty.

So resampling is over *clusters* (``example_id`` by default), and every row
belonging to a resampled example travels with it.

The intervals are engineering artefacts on small or mock datasets: they
describe the variability of the numbers in the table, not scientific
uncertainty about any model. Figure scripts label mock output
``diagnostic_only`` for exactly that reason.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

#: Defaults, recorded in every figure's metadata so a table can be reproduced.
DEFAULT_BOOTSTRAP_REPETITIONS = 2000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_ANALYSIS_SEED = 20260801
DEFAULT_CLUSTER_COLUMN = "example_id"

BOOTSTRAP_COLUMNS: Tuple[str, ...] = (
    "mean",
    "ci_lower",
    "ci_upper",
    "n_examples",
    "n_raw_rows",
    "bootstrap_repetitions",
    "analysis_seed",
    "confidence_level",
)


class StatisticsError(ValueError):
    """Raised when the data cannot support the requested statistic."""


def _group_rng(analysis_seed: int, key: Any) -> np.random.Generator:
    """A generator that depends on the group, not on iteration order.

    Deriving from the group key rather than advancing one shared stream keeps
    a group's interval identical whether it is computed alone or alongside
    twenty others -- otherwise adding a condition silently changes every
    previously published number.
    """
    digest = zlib.crc32(repr(key).encode("utf-8"))
    return np.random.default_rng([int(analysis_seed), int(digest)])


def _percentile_bounds(confidence_level: float) -> Tuple[float, float]:
    if not 0 < confidence_level < 1:
        raise StatisticsError(f"confidence_level must be in (0, 1), got {confidence_level}")
    alpha = (1.0 - confidence_level) / 2.0
    return 100.0 * alpha, 100.0 * (1.0 - alpha)


def _cluster_sums(
    frame: pd.DataFrame, value_col: str, cluster_col: str
) -> Tuple[np.ndarray, np.ndarray]:
    grouped = frame.groupby(cluster_col, sort=True)[value_col]
    return grouped.sum().to_numpy(dtype=float), grouped.count().to_numpy(dtype=float)


def _bootstrap_ci(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    repetitions: int,
    rng: np.random.Generator,
    confidence_level: float,
) -> Tuple[float, float]:
    """Percentile interval of the cluster-resampled weighted mean."""
    n_clusters = len(sums)
    if n_clusters < 2 or repetitions < 1:
        # One cluster carries no between-question variation to resample. An
        # interval here would be an artefact of the replications inside a
        # single example, so none is reported.
        return float("nan"), float("nan")
    idx = rng.integers(0, n_clusters, size=(repetitions, n_clusters))
    boot_sums = sums[idx].sum(axis=1)
    boot_counts = counts[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(boot_counts > 0, boot_sums / boot_counts, np.nan)
    low, high = _percentile_bounds(confidence_level)
    return float(np.nanpercentile(means, low)), float(np.nanpercentile(means, high))


def cluster_bootstrap_mean(
    frame: pd.DataFrame,
    *,
    value_col: str,
    cluster_col: str = DEFAULT_CLUSTER_COLUMN,
    group_cols: Sequence[str] = (),
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    analysis_seed: int = DEFAULT_ANALYSIS_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> pd.DataFrame:
    """Mean of ``value_col`` with a cluster-bootstrap interval, per group.

    Parameters
    ----------
    frame:
        Long-format rows. Every row is one replication.
    value_col:
        The quantity to average (accuracy indicator, exposure share, ...).
    cluster_col:
        Resampling unit. ``example_id`` by default: seeds, permutation seeds
        and rounds belonging to one question are resampled together.
    group_cols:
        Columns defining the separate curves/points. Nothing is pooled across
        them.

    Returns one row per group with the columns in :data:`BOOTSTRAP_COLUMNS`.
    ``n_examples`` counts clusters, ``n_raw_rows`` counts replications, so a
    reader can always see how much of the apparent sample size is replication.
    """
    for column in [value_col, cluster_col, *group_cols]:
        if column not in frame.columns:
            raise StatisticsError(f"missing column {column!r} in the input frame")

    usable = frame[frame[value_col].notna()]
    dropped = len(frame) - len(usable)

    group_cols = list(group_cols)
    groups: Iterable[Tuple[Any, pd.DataFrame]]
    if group_cols:
        groups = usable.groupby(group_cols, dropna=False, sort=True)
    else:
        groups = [((), usable)]

    rows: List[Dict[str, Any]] = []
    for key, subset in groups:
        if len(subset) == 0:
            continue
        key_tuple = key if isinstance(key, tuple) else (key,)
        sums, counts = _cluster_sums(subset, value_col, cluster_col)
        rng = _group_rng(analysis_seed, key_tuple)
        lower, upper = _bootstrap_ci(
            sums,
            counts,
            repetitions=repetitions,
            rng=rng,
            confidence_level=confidence_level,
        )
        row = dict(zip(group_cols, key_tuple))
        row.update(
            {
                "mean": float(subset[value_col].mean()),
                "ci_lower": lower,
                "ci_upper": upper,
                "n_examples": int(len(sums)),
                "n_raw_rows": int(len(subset)),
                "bootstrap_repetitions": int(repetitions),
                "analysis_seed": int(analysis_seed),
                "confidence_level": float(confidence_level),
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows, columns=[*group_cols, *BOOTSTRAP_COLUMNS])
    result.attrs["n_rows_dropped_missing"] = int(dropped)
    return result


@dataclass
class PairedDifference:
    """Result of :func:`paired_cluster_bootstrap_difference`."""

    mean_difference: float
    ci_lower: float
    ci_upper: float
    n_examples: int
    n_pairs: int
    bootstrap_repetitions: int
    analysis_seed: int
    confidence_level: float
    arm_a: str
    arm_b: str
    mean_arm_a: float
    mean_arm_b: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def paired_cluster_bootstrap_difference(
    frame: pd.DataFrame,
    *,
    value_col: str,
    arm_col: str,
    arm_a: str,
    arm_b: str,
    pair_cols: Sequence[str],
    cluster_col: str = DEFAULT_CLUSTER_COLUMN,
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    analysis_seed: int = DEFAULT_ANALYSIS_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> PairedDifference:
    """Paired ``arm_b - arm_a`` difference with a cluster-bootstrap interval.

    Pairing is explicit: ``pair_cols`` must identify exactly one row per arm.
    Rows that fail to pair are an error, never a silent drop -- an arm that
    covers a different set of examples is not a comparison, and dropping the
    difference would hide precisely the runs that behaved differently enough
    to fail or be re-run.

    Clusters (``example_id``) are resampled whole, so the two arms of a
    question always move together.
    """
    for column in [value_col, arm_col, cluster_col, *pair_cols]:
        if column not in frame.columns:
            raise StatisticsError(f"missing column {column!r} in the input frame")

    pair_cols = list(pair_cols)
    if cluster_col not in pair_cols:
        raise StatisticsError(
            f"cluster column {cluster_col!r} must be one of the pairing columns "
            f"{pair_cols} so that pairs can be grouped by question"
        )

    arms = frame[frame[arm_col].isin([arm_a, arm_b])]
    missing_arms = {arm_a, arm_b} - set(arms[arm_col].unique())
    if missing_arms:
        raise StatisticsError(
            f"arm(s) {sorted(missing_arms)} are absent from column {arm_col!r}; "
            "a paired comparison needs both"
        )

    duplicated = arms.duplicated([*pair_cols, arm_col]).sum()
    if duplicated:
        raise StatisticsError(
            f"{duplicated} duplicate row(s) for the same ({pair_cols}, {arm_col}) "
            "key; pairing requires exactly one observation per arm per key"
        )

    wide = arms.pivot_table(
        index=pair_cols,
        columns=arm_col,
        values=value_col,
        aggfunc="first",
        dropna=False,
    )
    unmatched = wide[wide[arm_a].isna() | wide[arm_b].isna()]
    if len(unmatched):
        preview = unmatched.head(5).index.tolist()
        raise StatisticsError(
            f"{len(unmatched)} pairing key(s) are present in only one of "
            f"{arm_a!r} / {arm_b!r}; the arms do not have identical coverage. "
            f"First unmatched: {preview}"
        )

    paired = wide.reset_index()
    paired["_difference"] = paired[arm_b] - paired[arm_a]

    sums, counts = _cluster_sums(paired, "_difference", cluster_col)
    rng = _group_rng(analysis_seed, (arm_a, arm_b, tuple(pair_cols)))
    lower, upper = _bootstrap_ci(
        sums, counts, repetitions=repetitions, rng=rng, confidence_level=confidence_level
    )

    return PairedDifference(
        mean_difference=float(paired["_difference"].mean()),
        ci_lower=lower,
        ci_upper=upper,
        n_examples=int(len(sums)),
        n_pairs=int(len(paired)),
        bootstrap_repetitions=int(repetitions),
        analysis_seed=int(analysis_seed),
        confidence_level=float(confidence_level),
        arm_a=str(arm_a),
        arm_b=str(arm_b),
        mean_arm_a=float(paired[arm_a].mean()),
        mean_arm_b=float(paired[arm_b].mean()),
    )


def broadcast_clean_arm(
    frame: pd.DataFrame,
    *,
    base_group_cols: Sequence[str],
    attack_type_col: str = "attack_type",
    attacker_count_col: str = "attacker_count",
    clean_labels: Sequence[str] = ("none",),
) -> pd.DataFrame:
    """Make the clean arm the zero point of every attack curve in its group.

    A clean run carries ``attack_type == "none"``, so grouping naively by
    attack type puts it in a curve of its own and every attacked curve loses
    its ``adversarial_fraction = 0`` anchor. The clean arm is not a separate
    experiment: it is what each attack is measured against.

    Clean rows are therefore copied once per attack type present in the same
    base group (dataset, model, mechanism, ...). Attacked rows are untouched,
    and a group with no attack at all keeps its single ``none`` curve.
    """
    base_group_cols = list(base_group_cols)
    if attack_type_col not in frame.columns or attacker_count_col not in frame.columns:
        return frame
    clean_mask = frame[attacker_count_col].fillna(0) == 0
    clean, attacked = frame[clean_mask], frame[~clean_mask]
    if attacked.empty or clean.empty:
        return frame

    pieces: List[pd.DataFrame] = [attacked]
    for key, subset in clean.groupby(base_group_cols, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        match = attacked
        for column, value in zip(base_group_cols, key_tuple):
            match = match[match[column] == value]
        types = sorted({str(t) for t in match[attack_type_col].dropna().unique()})
        types = [t for t in types if t not in set(clean_labels)]
        if not types:
            pieces.append(subset)
            continue
        for attack_type in types:
            copy = subset.copy()
            copy[attack_type_col] = attack_type
            pieces.append(copy)
    return pd.concat(pieces, ignore_index=True)


def describe_replication(
    frame: pd.DataFrame,
    *,
    cluster_col: str = DEFAULT_CLUSTER_COLUMN,
    seed_col: str = "seed",
    perm_seed_col: str = "perm_seed",
) -> Dict[str, int]:
    """Replication counts to print next to any aggregate.

    Kept beside every figure table so nobody has to guess whether an n of 180
    is 180 questions or 3 questions seen 60 times.
    """
    out = {
        "n_examples": int(frame[cluster_col].nunique()) if cluster_col in frame else 0,
        "n_raw_rows": int(len(frame)),
    }
    out["n_seeds"] = int(frame[seed_col].nunique()) if seed_col in frame else 0
    out["n_perm_seeds"] = int(frame[perm_seed_col].nunique()) if perm_seed_col in frame else 0
    return out


def average_within(
    frame: pd.DataFrame,
    *,
    value_col: str,
    within_cols: Sequence[str],
    keep_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Collapse repeated observations before bootstrapping.

    Used to average the rounds of one example x replication into a single
    observation. Without it, a three-round debate contributes three rows per
    question and the bootstrap sees three times the evidence it has.
    """
    within_cols = list(within_cols)
    keep_cols = [c for c in keep_cols if c not in within_cols]
    aggregation: Dict[str, Any] = {value_col: "mean"}
    for column in keep_cols:
        aggregation[column] = "first"
    collapsed = (
        frame.groupby(within_cols, dropna=False, sort=True)
        .agg(aggregation)
        .reset_index()
    )
    return collapsed
