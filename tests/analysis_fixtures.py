"""Synthetic run directories for the analysis-layer tests.

Real runs are slow, mock-provider-bound and only cover the scenarios that
happen to have been run. The analysis layer instead needs runs that are
deliberately *wrong* -- mixed schema versions, duplicate rows, incomplete
confidence maps, a competitor with no routing at all -- so the fixtures build
run directories directly.

They write exactly what ``runner.experiment`` writes (same file names, same
field names, same strict-JSON serialisation), so a test that passes here is a
test about the real log format. ``tests/test_analysis_common.py`` pins the
fixture writer against a genuine runner-produced run to keep that true.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

DEFAULT_EXAMPLES = ("ex-1", "ex-2", "ex-3")


def _stable_jitter(*parts: Any) -> random.Random:
    """A reproducible RNG keyed on the row coordinates.

    Fixtures must be deterministic across runs and machines -- bootstrap tests
    assert on exact numbers -- but not constant, or every confidence interval
    would collapse to a point.
    """
    return random.Random("|".join(str(p) for p in parts))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
            fh.write("\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(payload), fh, sort_keys=False)


def make_pear_run(
    root: Path,
    name: str,
    *,
    dataset: str = "gsm8k",
    dataset_split: Optional[str] = None,
    model: str = "test-model",
    mock: bool = True,
    schema_version: int = 2,
    mechanism: str = "pear_full",
    base_topology: str = "k_regular",
    n_agents: int = 4,
    rounds: int = 2,
    examples: Sequence[str] = DEFAULT_EXAMPLES,
    seeds: Sequence[int] = (0,),
    perm_seeds: Sequence[int] = (10, 11),
    condition: str = "pear_full",
    condition_debate: Optional[Mapping[str, Any]] = None,
    routing_mode: str = "sampled",
    include_routing: bool = True,
    include_tokens: bool = True,
    attacker_ids: Sequence[int] = (),
    dissenting_ids: Optional[Sequence[int]] = None,
    attack_mode: str = "fixed_report",
    attack_value: Optional[int] = None,
    attack_enabled: bool = True,
    verification_mode: str = "none",
    clean_confidence: int = 3,
    attacker_clean_confidence: int = 2,
    attacker_answer: str = "7",
    group_answer: str = "42",
    accuracy: float = 0.5,
    adopts_attacker_answer: bool = False,
    routing_temperature: float = 0.7,
    alpha_influence: float = 0.7,
    alpha_targeted_cross: float = 0.2,
    alpha_low_confidence: float = 0.7,
    influence_beta: float = 0.6,
    prompt_tokens_per_round: int = 500,
    completion_tokens_per_round: int = 120,
    extra_debate: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write one PEAR-shaped run directory and return its path.

    The generated numbers are not a simulation of PEAR. They are internally
    consistent -- out-degrees sum to the edge count, verified confidence is
    ``min(reported, g_i)``, influence is monotone in routing exposure -- which
    is all the analysis layer can be tested against.
    """
    run_dir = Path(root) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    attackers = sorted(int(a) for a in attacker_ids)
    attack_active = bool(attackers) and attack_enabled and attack_value is not None
    # Who dissents from the majority, which is a property of the *agent*, not of
    # the attack: the truthful arm of a paired sweep is the same dissenting
    # agent reporting honestly. Defaults to the attacked set so a single
    # attacked run needs no extra argument.
    dissenters = sorted(int(a) for a in (dissenting_ids if dissenting_ids is not None else attackers))
    degree = min(2, max(1, n_agents - 1))

    debate: Dict[str, Any] = {
        "n_agents": n_agents,
        "rounds": rounds,
        "base_topology": base_topology,
        "k_regular_degree": degree,
        "mode": mechanism,
        "agg_mode": "majority_vote",
        "mc_permutations": 50,
        "routing_temperature": routing_temperature,
        "alpha_targeted_cross": alpha_targeted_cross,
        "alpha_influence": alpha_influence,
        "alpha_low_confidence": alpha_low_confidence,
        "influence_beta": influence_beta,
        "low_confidence_threshold": 3,
        "targeted_cross_source_confidence_min": 4,
        "targeted_cross_target_confidence_max": 3,
    }
    if extra_debate:
        debate.update(dict(extra_debate))
    if attackers:
        debate["robustness"] = {
            "enabled": True,
            "type": "confidence_inflation",
            "confidence_inflation": {
                "enabled": bool(attack_enabled),
                "agent_ids": list(attackers),
                "mode": attack_mode,
                "value": attack_value,
            },
        }
    if verification_mode != "none":
        debate["verified_confidence"] = {"mode": verification_mode}

    dataset_cfg: Dict[str, Any] = {"name": dataset, "num_examples": len(examples)}
    if dataset_split is not None:
        dataset_cfg["split"] = dataset_split

    config = {
        "agents": {"model": model},
        "dataset": dataset_cfg,
        "debate": debate,
        "conditions": [{"name": condition, "debate": dict(condition_debate or {"mode": mechanism})}],
        "replication": {"seeds": list(seeds), "agent_perm_seeds": list(perm_seeds)},
        "paths": {"data_dir": "data", "output_dir": str(root)},
    }
    _write_yaml(run_dir / "config.yaml", config)

    effective = dict(debate)
    effective.update(dict(condition_debate or {}))
    effective_mechanism = str(effective.get("mode", mechanism))
    effective_rounds = 0 if effective_mechanism in {"cot", "cot_sc"} else int(effective.get("rounds", rounds))
    effective_agents = int(effective.get("n_agents", n_agents))

    n_correct = int(round(accuracy * len(examples)))
    correct_examples = set(examples[:n_correct])

    def reported_of(agent: int) -> float:
        if attack_active and agent in attackers:
            return float(attack_value)
        return float(attacker_clean_confidence if agent in dissenters else clean_confidence)

    def clean_of(agent: int) -> float:
        return float(attacker_clean_confidence if agent in dissenters else clean_confidence)

    def g_of(agent: int, example: str) -> float:
        # Oracle corroboration: 5 when the agent's answer is right, else 1.
        # The dissenter's answer is wrong, so it is never corroborated.
        if agent in dissenters:
            return 1.0
        return 5.0 if example in correct_examples else 1.0

    results: List[Dict[str, Any]] = []
    routing: List[Dict[str, Any]] = []

    for example in examples:
        for seed in seeds:
            for perm_seed in perm_seeds:
                answers = {
                    str(a): (attacker_answer if a in dissenters else group_answer)
                    for a in range(1, effective_agents + 1)
                }
                influence = {str(a): 1.0 / effective_agents for a in range(1, effective_agents + 1)}
                influence_history = [dict(influence)]
                answer_history = [dict(answers)]
                confidence_history = [
                    {str(a): reported_of(a) for a in range(1, effective_agents + 1)}
                ]

                for round_idx in range(1, effective_rounds + 1):
                    rng = _stable_jitter(name, condition, example, seed, perm_seed, round_idx)
                    out_degree: List[int] = []
                    for agent in range(1, effective_agents + 1):
                        reported = reported_of(agent)
                        verified = min(reported, g_of(agent, example))
                        routing_conf = verified if verification_mode != "none" else reported
                        # Routing exposure rises with the confidence the router
                        # actually scored: that is the effect Figure 3 measures.
                        base = 1 + int(round(routing_conf)) if agent in dissenters else 2
                        out_degree.append(max(0, base + rng.randint(0, 1)))
                    in_degree = [degree] * effective_agents
                    total_out = sum(out_degree) or 1
                    influence = {
                        str(a): round(
                            influence_beta * influence[str(a)]
                            + (1 - influence_beta) * out_degree[a - 1] / total_out,
                            6,
                        )
                        for a in range(1, effective_agents + 1)
                    }
                    influence_history.append(dict(influence))
                    answer_history.append(dict(answers))
                    confidence_history.append(
                        {str(a): reported_of(a) for a in range(1, effective_agents + 1)}
                    )

                    if not include_routing:
                        continue
                    verified_map = {
                        str(a): min(reported_of(a), g_of(a, example))
                        for a in range(1, effective_agents + 1)
                    }
                    g_map = {str(a): g_of(a, example) for a in range(1, effective_agents + 1)}
                    routing.append(
                        {
                            "schema_version": schema_version,
                            "condition": condition,
                            "example_id": example,
                            "seed": seed,
                            "perm_seed": perm_seed,
                            "mock": mock,
                            "mode": effective_mechanism,
                            "base_topology": base_topology,
                            "n_agents": effective_agents,
                            "rounds": effective_rounds,
                            "round": round_idx,
                            "adversary_ids": [],
                            "answers": dict(answers),
                            "clean_confidence": {
                                str(a): clean_of(a) for a in range(1, effective_agents + 1)
                            },
                            "reported_confidence": {
                                str(a): reported_of(a) for a in range(1, effective_agents + 1)
                            },
                            "g_i": g_map if verification_mode != "none" else None,
                            "verified_confidence": (
                                verified_map if verification_mode != "none" else None
                            ),
                            "verified_confidence_mode": verification_mode,
                            "confidence_inflation_agent_ids": list(attackers) if attack_active else None,
                            "confidence_inflation_mode": attack_mode if attack_active else None,
                            "confidence_inflation_value": attack_value if attack_active else None,
                            "in_degree": in_degree,
                            "out_degree": out_degree,
                            "influence": {
                                str(a): influence_history[round_idx - 1][str(a)]
                                for a in range(1, effective_agents + 1)
                            },
                            "targeted_cross_eligibility": {
                                str(a): {
                                    "confidence": reported_of(a),
                                    "source_eligible": reported_of(a) >= 4,
                                    "target_eligible": reported_of(a) <= 3,
                                    "source_edges": 0,
                                    "target_edges": 0,
                                }
                                for a in range(1, effective_agents + 1)
                            },
                            "candidate_count": 50,
                            "selected_score": -0.25,
                            "routing_mode": routing_mode,
                            "objective": {
                                "alpha_targeted_cross": alpha_targeted_cross,
                                "alpha_influence": alpha_influence,
                                "alpha_low_confidence": alpha_low_confidence,
                                "routing_temperature": routing_temperature,
                            },
                        }
                    )

                decision = attacker_answer if adopts_attacker_answer else group_answer
                row: Dict[str, Any] = {
                    "example_id": example,
                    "decision": decision,
                    "correct": example in correct_examples and not adopts_attacker_answer,
                    "parse_failures": 0,
                    "n_messages": effective_agents * (effective_rounds + 1),
                    "seed": seed,
                    "perm_seed": perm_seed,
                    "answer_history": answer_history,
                    "confidence_history": confidence_history,
                    "influence_history": influence_history,
                    "condition": condition,
                    "mock": mock,
                    "mode": effective_mechanism,
                    "topology": base_topology,
                    "base_topology": base_topology,
                    "n_agents": effective_agents,
                    "rounds": effective_rounds,
                }
                if include_tokens:
                    row["budget"] = {
                        "calls": effective_agents * (effective_rounds + 1),
                        "prompt_tokens": prompt_tokens_per_round * (effective_rounds + 1),
                        "completion_tokens": completion_tokens_per_round * (effective_rounds + 1),
                        "max_calls": 0,
                        "max_tokens": 0,
                        "extras": {},
                    }
                results.append(row)

    _write_jsonl(run_dir / "results.jsonl", results)
    if include_routing:
        _write_jsonl(run_dir / "routing.jsonl", routing)

    accuracy_value = (
        sum(1 for r in results if r["correct"]) / len(results) if results else 0.0
    )
    _write_json(
        run_dir / "summary.json",
        {
            "schema_version": schema_version,
            "run_dir": str(run_dir),
            "mock": mock,
            "model": model,
            "judge_model": None,
            "parallel_examples": 1,
            "conditions": [
                {
                    "name": condition,
                    "accuracy": accuracy_value,
                    "n_runs": len(results),
                    "n_examples": len(examples),
                    "mock": mock,
                    "mode": effective_mechanism,
                    "base_topology": base_topology,
                    "n_agents": effective_agents,
                    "rounds": effective_rounds,
                }
            ],
        },
    )
    return run_dir


def make_generic_run(
    root: Path,
    name: str,
    *,
    method: str = "single_agent_baseline",
    dataset: str = "gsm8k",
    dataset_split: Optional[str] = None,
    model: str = "test-model",
    mock: bool = True,
    schema_version: int = 2,
    examples: Sequence[str] = DEFAULT_EXAMPLES,
    seeds: Sequence[int] = (0,),
    accuracy: float = 0.6,
    include_tokens: bool = True,
    prompt_tokens: int = 300,
    completion_tokens: int = 60,
    condition: str = "default",
    n_agents: int = 1,
    adversarial_fraction: Optional[float] = 0.0,
) -> Path:
    """Write a competitor-shaped run: accuracy and tokens, no routing at all.

    This is the shape an adapter for someone else's system produces. It must
    survive validation, appear in the accuracy and cost figures, and be
    excluded from the routing figures with a stated capability reason -- never
    with zeros.
    """
    run_dir = Path(root) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    n_correct = int(round(accuracy * len(examples)))
    correct_examples = set(examples[:n_correct])

    rows: List[Dict[str, Any]] = []
    for example in examples:
        for seed in seeds:
            row: Dict[str, Any] = {
                "example_id": example,
                "prediction": "42" if example in correct_examples else "0",
                "correct": example in correct_examples,
                "seed": seed,
                "perm_seed": None,
                "condition": condition,
                "mock": mock,
                "mechanism": method,
                "n_agents": n_agents,
                "attacker_count": 0,
                "adversarial_fraction": adversarial_fraction,
                "verification_mode": "none",
                "attack_type": "none",
            }
            if include_tokens:
                row["prompt_tokens"] = prompt_tokens
                row["completion_tokens"] = completion_tokens
            rows.append(row)
    _write_jsonl(run_dir / "results.jsonl", rows)

    dataset_cfg: Dict[str, Any] = {"name": dataset, "num_examples": len(examples)}
    if dataset_split is not None:
        dataset_cfg["split"] = dataset_split
    _write_yaml(
        run_dir / "config.yaml",
        {
            "method": method,
            "agents": {"model": model},
            "dataset": dataset_cfg,
            "replication": {"seeds": list(seeds), "agent_perm_seeds": [None]},
        },
    )
    _write_json(
        run_dir / "summary.json",
        {
            "schema_version": schema_version,
            "adapter": "generic",
            "method": method,
            "model": model,
            "mock": mock,
            "dataset": dataset,
            "dataset_split": dataset_split,
            "accuracy": len(correct_examples) / len(examples) if examples else 0.0,
        },
    )
    return run_dir


def corrupt_jsonl(path: Path, text: str) -> None:
    """Append a raw line to a JSONL file (malformed JSON, bare NaN, ...)."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(text.rstrip("\n") + "\n")


def duplicate_last_line(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(lines[-1] + "\n")


def rewrite_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_jsonl(path, rows)


def read_jsonl_raw(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def patch_summary(path: Path, **updates: Any) -> None:
    payload = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    payload.update(updates)
    _write_json(path / "summary.json", payload)
