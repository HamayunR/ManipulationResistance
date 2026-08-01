"""ExpPlan_v3 diagnostic metrics for two-phase PEAR debates."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence


ScoreFn = Callable[[str], bool]


def safe_div(num: float, den: float) -> float:
    return float("nan") if den == 0 else float(num) / float(den)


def _agent_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        import re

        match = re.search(r"\d+", str(value))
        return int(match.group(0)) if match else None


def trajectory_event_rates(
    initial_answers: Mapping[int, str],
    final_answers: Mapping[int, str],
    is_correct: ScoreFn,
) -> Dict[str, float]:
    """Compute W2R/R2W/W2W/R2R rates over agent answer trajectories."""
    counts = Counter()
    wrong0 = 0
    right0 = 0
    for agent_id, initial in initial_answers.items():
        final = final_answers.get(agent_id, "")
        init_ok = bool(is_correct(initial))
        final_ok = bool(is_correct(final))
        if init_ok:
            right0 += 1
            counts["R2R" if final_ok else "R2W"] += 1
        else:
            wrong0 += 1
            counts["W2R" if final_ok else "W2W"] += 1
    return {
        "w2r_rate": safe_div(counts["W2R"], wrong0),
        "r2w_rate": safe_div(counts["R2W"], right0),
        "w2w_rate": safe_div(counts["W2W"], wrong0),
        "r2r_rate": safe_div(counts["R2R"], right0),
        "w2r_count": float(counts["W2R"]),
        "r2w_count": float(counts["R2W"]),
        "w2w_count": float(counts["W2W"]),
        "r2r_count": float(counts["R2R"]),
    }


def critique_precision(update_events: Iterable[Mapping[str, Any]], is_correct: ScoreFn) -> float:
    """Fraction of accepted critiques associated with a W2R update."""
    accepted = 0
    accepted_w2r = 0
    for event in update_events:
        before = str(event.get("previous_answer", ""))
        after = str(event.get("answer", ""))
        w2r = (not is_correct(before)) and is_correct(after)
        for response in (event.get("critique_response") or {}).values():
            decision = str((response or {}).get("decision", "")).upper()
            if decision == "ACCEPT":
                accepted += 1
                if w2r:
                    accepted_w2r += 1
    return safe_div(accepted_w2r, accepted)


def critique_acceptance_rate(update_events: Iterable[Mapping[str, Any]]) -> float:
    accepted = 0
    total = 0
    for event in update_events:
        for response in (event.get("critique_response") or {}).values():
            decision = str((response or {}).get("decision", "")).upper()
            if decision in {"ACCEPT", "REJECT"}:
                total += 1
                accepted += int(decision == "ACCEPT")
    return safe_div(accepted, total)


def cross_cluster_critique_rate(edge_events: Iterable[Mapping[str, Any]]) -> float:
    total = 0
    cross = 0
    for event in edge_events:
        if event.get("event") != "critique_edge":
            continue
        total += 1
        cross += int(bool(event.get("cross_cluster")))
    return safe_div(cross, total)


def targeted_cross_critique_rate(edge_events: Iterable[Mapping[str, Any]]) -> float:
    """Fraction of critique edges matching the targeted-cross routing gate."""
    total = 0
    targeted = 0
    for event in edge_events:
        if event.get("event") != "critique_edge":
            continue
        total += 1
        targeted += int(bool(event.get("targeted_cross")))
    return safe_div(targeted, total)


def confidence_calibration(
    edge_events: Iterable[Mapping[str, Any]],
    update_events: Iterable[Mapping[str, Any]],
    is_correct: ScoreFn,
) -> Dict[str, Dict[str, float]]:
    """Bucket source confidence and estimate associated W2R rate."""
    accepted_sources_by_round_target: Dict[tuple[int, int], set[int]] = {}
    w2r_by_round_target: Dict[tuple[int, int], bool] = {}
    for event in update_events:
        key = (int(event.get("round", 0)), int(event.get("agent_id", 0)))
        before = str(event.get("previous_answer", ""))
        after = str(event.get("answer", ""))
        w2r_by_round_target[key] = (not is_correct(before)) and is_correct(after)
        accepted_sources = set()
        for src, response in (event.get("critique_response") or {}).items():
            parsed_src = _agent_id(src)
            if parsed_src is None:
                continue
            if str((response or {}).get("decision", "")).upper() == "ACCEPT":
                accepted_sources.add(parsed_src)
        accepted_sources_by_round_target[key] = accepted_sources

    buckets = {
        "low_1_2": {"edges": 0.0, "accepted": 0.0, "accepted_w2r": 0.0},
        "mid_3": {"edges": 0.0, "accepted": 0.0, "accepted_w2r": 0.0},
        "high_4_5": {"edges": 0.0, "accepted": 0.0, "accepted_w2r": 0.0},
    }
    for event in edge_events:
        if event.get("event") != "critique_edge":
            continue
        conf = float(event.get("source_confidence", 3))
        if conf <= 2:
            bucket = buckets["low_1_2"]
        elif conf < 4:
            bucket = buckets["mid_3"]
        else:
            bucket = buckets["high_4_5"]
        bucket["edges"] += 1.0
        key = (int(event.get("round", 0)), int(event.get("target", 0)))
        source = int(event.get("source", 0))
        accepted = source in accepted_sources_by_round_target.get(key, set())
        bucket["accepted"] += float(accepted)
        bucket["accepted_w2r"] += float(accepted and w2r_by_round_target.get(key, False))

    return {
        name: {
            **vals,
            "accept_rate": safe_div(vals["accepted"], vals["edges"]),
            "accepted_w2r_rate": safe_div(vals["accepted_w2r"], vals["accepted"]),
        }
        for name, vals in buckets.items()
    }


def entropy(values: Sequence[Any]) -> float:
    cleaned = [v for v in values if v is not None and v != ""]
    if not cleaned:
        return 0.0
    counts = Counter(cleaned)
    total = sum(counts.values())
    out = 0.0
    for count in counts.values():
        p = count / total
        out -= p * math.log2(p)
    return out


def normalized_entropy(weights: Sequence[float]) -> float:
    vals = [max(0.0, float(v)) for v in weights]
    total = sum(vals)
    if total <= 0 or len(vals) <= 1:
        return 0.0
    probs = [v / total for v in vals if v > 0]
    h = -sum(p * math.log(p) for p in probs)
    return h / math.log(len(vals))


def aggregate_diagnostics(rows: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    """Mean diagnostic values across per-example rows."""
    keys = [
        "w2r_rate",
        "r2w_rate",
        "w2w_rate",
        "r2r_rate",
        "critique_precision",
        "critique_acceptance_rate",
        "cross_cluster_critique_rate",
        "targeted_cross_critique_rate",
        "influence_entropy",
        "confidence_perturbation_rate",
        "confidence_mean_abs_delta",
        "critique_noise_rate",
        "critique_drop_rate",
        "critique_corrupt_rate",
        "critique_source_swap_rate",
        "malicious_agent_present",
        "adversary_final_correct",
        "adversary_final_influence",
        "adversary_wrong_adoption_rate",
        "non_adversary_r2w_rate_under_attack",
        "attack_success_rate",
        # Mean count of model outputs per example from which no JSON object
        # could be recovered. Non-zero means the parser is silently falling
        # back, so any metric downstream of it is suspect.
        "parse_failures",
    ]
    vals: Dict[str, List[float]] = {key: [] for key in keys}
    for row in rows:
        diag = row.get("diagnostics") or {}
        for key in keys:
            value = diag.get(key)
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                vals[key].append(float(value))
    return {
        key: (sum(items) / len(items) if items else float("nan"))
        for key, items in vals.items()
    }
