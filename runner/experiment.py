"""High-level experiment runner.

The runner is deliberately decoupled from any CLI: ``scripts/run_experiment``
imports :func:`run_experiment` and provides the argument parsing.
"""

from __future__ import annotations

import json
import math
import os
import re
import string
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import yaml

try:  # tqdm is optional; vLLM environments usually already include it.
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    _tqdm = None

from data.tasks import Task, load_task
from core.topology import (
    apply_perm_to_topology,
    edge_list,
    make_base_topology,
    out_neighbors,
    state_aware_permutation,
    subgroup_permutation,
    topology_hash,
    uniform_permutation,
)
from metrics.diagnostics import (
    aggregate_diagnostics,
    confidence_calibration,
    critique_acceptance_rate,
    critique_precision,
    cross_cluster_critique_rate,
    normalized_entropy,
    targeted_cross_critique_rate,
    trajectory_event_rates,
)
from metrics.stability import summarise_runs
from models.base import BaseLLM
from models.factory import build_llm, load_model_registry
from nodes.aggregator import _majority_vote
from prompts import (
    AGENT_SYSTEM,
    ANSWER_UPDATE_TEMPLATE,
    CRITIQUE_GENERATION_TEMPLATE,
    EMPTY_TRANSCRIPT,
    INITIAL_ANSWER_TEMPLATE,
    MALICIOUS_CRITIQUE_GENERATION_TEMPLATE,
    MALICIOUS_INITIAL_ANSWER_TEMPLATE,
)
from utils.budget import Budget
from utils.logging import RunPaths, get_logger, setup_run_logging
from utils.seed import seeded_rng, set_global_seeds
from utils.tracing import JsonlTracer

_log = get_logger("runner")


class _NullProgress:
    """Tiny context-manager shim used when tqdm is unavailable or disabled."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def update(self, n: int = 1) -> None:
        return None


def _progress_bar(*, total: int, desc: str, enabled: bool):
    if not enabled or _tqdm is None:
        return _NullProgress()
    return _tqdm(total=total, desc=desc, unit="run", dynamic_ncols=True)


# Configuration handling
def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (mutating ``base``)."""
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = deepcopy(v)
    return base


def _load_config(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML config, resolving the ``extends:`` chain (if any)."""
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    parent_rel = cfg.pop("extends", None)
    if parent_rel:
        parent_path = (path.parent / parent_rel).resolve()
        parent = _load_config(parent_path)
        merged = _deep_merge(parent, cfg)
        return merged
    return cfg


# Public dataclasses
@dataclass
class ExperimentConfig:
    """Resolved experiment configuration ready for :func:`run_experiment`."""

    raw: Dict[str, Any]
    run_tag: str = ""

    @classmethod
    def from_file(cls, path: str | os.PathLike, *, run_tag: str = "") -> "ExperimentConfig":
        return cls(raw=_load_config(path), run_tag=run_tag)


@dataclass
class RunResult:
    """One condition's results across all seeds and examples."""

    condition: str
    accuracy: float
    n_examples: int
    n_runs: int
    summary: Dict[str, Any] = field(default_factory=dict)


# Two-phase PEAR execution helpers
def _extract_json_object(text: str) -> Dict[str, Any]:
    """Best-effort parse of a model JSON object."""
    if not text:
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    candidate = text[start : end + 1]
    try:
        payload = json.loads(candidate)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        # Common local-model failure: trailing commas in JSON-ish output.
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            payload = json.loads(cleaned)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}


def _clamp_confidence(value: Any) -> int:
    try:
        return max(1, min(5, int(round(float(value)))))
    except (TypeError, ValueError):
        return 3


def _parse_answer_payload(text: str, parse_answer: Callable[[str], str]) -> Dict[str, Any]:
    payload = _extract_json_object(text)
    answer = str(payload.get("answer") or "").strip()
    if not answer:
        answer = parse_answer(text)
    return {
        "answer": answer,
        "confidence": _clamp_confidence(payload.get("confidence", 3)),
        "reasoning": str(payload.get("reasoning") or text).strip(),
        "critique_response": _normalize_critique_response(
            payload.get("critique_response") or {}
        ),
        "raw": text,
    }


def _normalize_critique_response(raw: Any) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    if not isinstance(raw, Mapping):
        return out
    for source, value in raw.items():
        source_text = str(source)
        source_match = re.search(r"\d+", source_text)
        source_key = source_match.group(0) if source_match else source_text
        if isinstance(value, Mapping):
            decision = str(value.get("decision") or "").upper()
            reason = str(value.get("reason") or "").strip()
        else:
            text = str(value)
            decision = "ACCEPT" if "ACCEPT" in text.upper() else "REJECT"
            reason = text
        if decision not in {"ACCEPT", "REJECT"}:
            decision = "REJECT"
        out[source_key] = {"decision": decision, "reason": reason}
    return out


def _parse_critiques(text: str, expected_targets: Iterable[int]) -> List[Dict[str, Any]]:
    payload = _extract_json_object(text)
    reviews = payload.get("reviews") if isinstance(payload, dict) else None
    parsed: List[Dict[str, Any]] = []
    if isinstance(reviews, list):
        for item in reviews:
            if not isinstance(item, Mapping):
                continue
            try:
                target = int(item.get("target"))
            except (TypeError, ValueError):
                continue
            assessment = str(item.get("assessment") or "Acceptable").strip().title()
            if assessment not in {"Strong", "Acceptable", "Flawed"}:
                assessment = "Acceptable"
            parsed.append(
                {
                    "target": target,
                    "step_loc": str(item.get("step_loc") or "").strip(),
                    "correction": str(item.get("correction") or "").strip(),
                    "assessment": assessment,
                    "raw": text,
                }
            )

    by_target = {int(item["target"]): item for item in parsed}
    for target in expected_targets:
        if target not in by_target:
            by_target[int(target)] = {
                "target": int(target),
                "step_loc": "No structured review parsed.",
                "correction": "",
                "assessment": "Acceptable",
                "raw": text,
            }
    return [by_target[int(target)] for target in expected_targets]


def _charge_budget(budget: Budget, generations) -> None:
    for gen in generations:
        budget.charge(
            calls=1,
            prompt_tokens=getattr(gen, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(gen, "completion_tokens", 0) or 0,
        )


def _generate_many(
    llm: BaseLLM,
    prompts: List[str],
    *,
    max_tokens: int,
    system: str,
    agent_ids: List[int],
    budget: Budget,
):
    generations = llm.generate_batch(
        prompts,
        max_tokens=max_tokens,
        system=system,
        agent_ids=agent_ids,
    )
    if len(generations) != len(prompts):
        raise RuntimeError(
            f"Batch generation returned {len(generations)} outputs for {len(prompts)} prompts"
        )
    _charge_budget(budget, generations)
    return generations


def _render_critiques_for_agent(
    source_critiques: List[Dict[str, Any]],
    previous_answers: Mapping[int, Dict[str, Any]],
) -> str:
    if not source_critiques:
        return EMPTY_TRANSCRIPT
    blocks = []
    for critique in source_critiques:
        source = int(critique["source"])
        source_answer = previous_answers.get(source, {}).get("answer", "")
        blocks.append(
            "\n".join(
                [
                    f"[Critique from Agent {source}]",
                    f"Their answer: {source_answer}",
                    f"Error identified: {critique.get('step_loc', '')}",
                    f"Correction suggested: {critique.get('correction', '')}",
                    f"Assessment: {critique.get('assessment', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _render_targets(
    targets: Iterable[int],
    answers: Mapping[int, Dict[str, Any]],
) -> str:
    blocks = []
    for target in targets:
        current = answers[int(target)]
        blocks.append(
            "\n".join(
                [
                    f"[Participant {target}]",
                    f"Answer: {current.get('answer', '')}",
                    f"Confidence: {current.get('confidence', 0)}",
                    f"Reasoning: {current.get('reasoning', '')}",
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else EMPTY_TRANSCRIPT


def _routing_weights(mode: str, debate_cfg: Mapping[str, Any]) -> tuple[float, float, float]:
    """Resolve targeted-cross, influence, and low-confidence routing weights."""
    route_cfg = debate_cfg.get("routing", {}) if isinstance(debate_cfg.get("routing"), Mapping) else {}
    default_targeted_cross = float(
        route_cfg.get(
            "alpha_targeted_cross",
            debate_cfg.get("alpha_targeted_cross", 1.0),
        )
    )
    default_inf = float(route_cfg.get("alpha_influence", debate_cfg.get("alpha_influence", 1.0)))
    default_low_conf = float(
        route_cfg.get("alpha_low_confidence", debate_cfg.get("alpha_low_confidence", 0.0))
    )
    if mode in {"fixed", "cot", "cot_sc", "mad", "random_k_regular"}:
        return 0.0, 0.0, 0.0
    if mode == "pear_uniform":
        return 0.0, 0.0, 0.0
    if mode in {"pear_targeted_cross", "targeted_cross"}:
        return default_targeted_cross, 0.0, 0.0
    if mode in {"pear_influence", "influence"}:
        return 0.0, default_inf, 0.0
    if mode in {"pear_low_confidence", "low_confidence"}:
        return 0.0, 0.0, default_low_conf
    if mode in {"pear_targeted_influence", "targeted_influence"}:
        return default_targeted_cross, default_inf, 0.0
    if mode in {"pear_targeted_low_confidence", "targeted_low_confidence"}:
        return default_targeted_cross, 0.0, default_low_conf
    if mode in {"pear_influence_low_confidence", "influence_low_confidence"}:
        return 0.0, default_inf, default_low_conf
    if mode in {"pear_full", "state_aware"}:
        return default_targeted_cross, default_inf, default_low_conf
    raise ValueError(f"Unknown debate mode: {mode!r}")


def _routing_thresholds(debate_cfg: Mapping[str, Any]) -> tuple[float, float, float]:
    """Resolve confidence thresholds used by routing."""
    route_cfg = debate_cfg.get("routing", {}) if isinstance(debate_cfg.get("routing"), Mapping) else {}
    low_confidence_threshold = float(
        route_cfg.get(
            "low_confidence_threshold",
            debate_cfg.get("low_confidence_threshold", 3.0),
        )
    )
    targeted_source_min = float(
        route_cfg.get(
            "targeted_cross_source_confidence_min",
            debate_cfg.get("targeted_cross_source_confidence_min", low_confidence_threshold + 1.0),
        )
    )
    targeted_target_max = float(
        route_cfg.get(
            "targeted_cross_target_confidence_max",
            debate_cfg.get("targeted_cross_target_confidence_max", low_confidence_threshold),
        )
    )
    return low_confidence_threshold, targeted_source_min, targeted_target_max


def _routing_enumeration(debate_cfg: Mapping[str, Any]) -> tuple[bool, int, int]:
    """Resolve the candidate-pool construction for state-aware routing.

    Returns ``(enumerate_permutations, candidates, enumeration_max_factorial)``.

    Exact enumeration of :math:`S_n` is opt-in, selected by either
    ``mc_permutations: null`` or an explicit ``enumerate_permutations: true``.
    Anything else keeps the vanilla-PEAR sampled pool (identity-seeded, drawn
    with replacement) at ``mc_permutations`` candidates, default 100.
    """
    route_cfg = debate_cfg.get("routing", {}) if isinstance(debate_cfg.get("routing"), Mapping) else {}
    sentinel = object()
    raw = route_cfg.get("mc_permutations", sentinel)
    if raw is sentinel:
        raw = debate_cfg.get("mc_permutations", 100)
    explicit = route_cfg.get(
        "enumerate_permutations",
        debate_cfg.get("enumerate_permutations", False),
    )
    enumerate_permutations = bool(explicit) or raw is None
    candidates = 100 if raw is None else int(raw)
    enumeration_max_factorial = int(
        route_cfg.get(
            "enumeration_max_factorial",
            debate_cfg.get("enumeration_max_factorial", 5040),
        )
    )
    return enumerate_permutations, candidates, enumeration_max_factorial


def _routing_terms_normalized(debate_cfg: Mapping[str, Any]) -> bool:
    route_cfg = debate_cfg.get("routing", {}) if isinstance(debate_cfg.get("routing"), Mapping) else {}
    value = route_cfg.get(
        "normalize_routing_terms",
        debate_cfg.get("normalize_routing_terms", True),
    )
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off"}



def _robustness_config(debate_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = debate_cfg.get("robustness") or {}
    if not isinstance(cfg, Mapping) or not cfg.get("enabled", False):
        return {}
    return dict(cfg)


def _robustness_component(robust_cfg: Mapping[str, Any], name: str) -> Dict[str, Any]:
    if not robust_cfg:
        return {}
    kind = str(robust_cfg.get("type", "")).strip().lower()
    nested = robust_cfg.get(name) or {}
    explicit = isinstance(nested, Mapping) and bool(nested.get("enabled", False))
    if kind not in {name, "all"} and not explicit:
        return {}
    merged = dict(robust_cfg)
    if isinstance(nested, Mapping):
        merged.update(dict(nested))
    return merged


def _adversary_candidates(malicious_cfg: Mapping[str, Any], n_agents: int) -> List[int]:
    """Normalise ``adversary_agent_id`` into an ordered candidate list.

    Accepts either a single int (the default, ``1``) or a list/tuple of ints.
    Values are clamped into ``[1, n_agents]`` and de-duplicated with order
    preserved; sets are sorted first so the result stays reproducible.
    """
    if not malicious_cfg or n_agents <= 1:
        return []
    raw = malicious_cfg.get("adversary_agent_id", 1)
    if isinstance(raw, (set, frozenset)):
        values = sorted(raw)
    elif isinstance(raw, (list, tuple)):
        values = list(raw)
    else:
        values = [raw]

    seen: set[int] = set()
    candidates: List[int] = []
    for value in values:
        agent_id = max(1, min(n_agents, int(value)))
        if agent_id not in seen:
            seen.add(agent_id)
            candidates.append(agent_id)
    return candidates


def _adversary_id(
    malicious_cfg: Mapping[str, Any],
    n_agents: int,
    *,
    rng: Any = None,
) -> Optional[int]:
    """Resolve which single agent plays the adversary.

    ``adversary_agent_id`` may be a single int or a list. A list is a *candidate
    set*: exactly one agent is drawn from it, uniformly at random when ``rng`` is
    supplied and otherwise the first entry. This lets the adversary slot be
    randomised across examples instead of pinned to agent 1, which under sampled
    routing is confounded with both the identity-seeding weight and the star hub
    (``make_base_topology("star", ...)`` places the hub at role 1).

    No RNG draw is consumed when the candidate set has one element, so
    single-int configs stay bit-identical to the previous behaviour.
    """
    candidates = _adversary_candidates(malicious_cfg, n_agents)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if rng is None:
        return candidates[0]
    return candidates[rng.randrange(len(candidates))]


def _default_wrong_answer(task: Task, example) -> str:
    choices = list(example.choices or [])
    if choices:
        for letter in string.ascii_uppercase[: len(choices)]:
            if not task.score(letter, example):
                return letter
    gold = str(example.answer).strip()
    try:
        value = float(gold.replace(",", ""))
        if value.is_integer():
            return str(int(value) + 1)
        return str(value + 1.0)
    except Exception:
        pass
    candidates = [f"{gold} + 1", "0", "1", "-1", "A", "unknown"]
    for cand in candidates:
        if not task.score(task.parse_answer(cand) or cand, example):
            return cand
    return "__intentionally_wrong__"


def _ensure_wrong_answer(task: Task, example, answer: str) -> str:
    parsed = task.parse_answer(answer) or answer
    if not task.score(parsed, example):
        return parsed
    return _default_wrong_answer(task, example)


def _apply_malicious_payload(
    *,
    task: Task,
    example,
    payload: Dict[str, Any],
    fallback_answer: Optional[str],
    confidence: int,
    phase: str,
) -> tuple[str, Dict[str, Any]]:
    chosen = str(payload.get("answer") or fallback_answer or "").strip()
    wrong = _ensure_wrong_answer(task, example, chosen)
    payload["answer"] = wrong
    payload["confidence"] = _clamp_confidence(confidence)
    reasoning = str(payload.get("reasoning") or "").strip()
    if not reasoning:
        reasoning = "Robustness adversary: persuasive but intentionally flawed reasoning."
    payload["reasoning"] = reasoning
    return wrong, {
        "event": "robustness_malicious_agent",
        "phase": phase,
        "answer": wrong,
        "confidence": payload["confidence"],
        "gold": example.answer,
    }


def _apply_confidence_perturbation(
    current: Dict[int, Dict[str, Any]],
    *,
    phase: str,
    round_idx: int,
    cfg: Mapping[str, Any],
    rng,
    is_correct: Callable[[str], bool],
) -> tuple[List[Dict[str, Any]], Dict[str, float]]:
    if not cfg:
        return [], {"confidence_total": 0.0, "confidence_changed": 0.0, "confidence_abs_delta": 0.0}
    rate = max(0.0, min(1.0, float(cfg.get("rate", 1.0))))
    strategy = str(cfg.get("strategy", "miscalibrate")).strip().lower()
    magnitude = max(0, int(cfg.get("magnitude", 2)))
    before = {i: _clamp_confidence(v.get("confidence", 3)) for i, v in current.items()}
    after = dict(before)
    selected = [i for i in sorted(current) if rng.random() <= rate]

    if strategy == "shuffle" and len(selected) > 1:
        values = [after[i] for i in selected]
        rng.shuffle(values)
        for i, value in zip(selected, values):
            after[i] = value
    else:
        for i in selected:
            old = before[i]
            answer = str(current[i].get("answer", ""))
            correct = bool(is_correct(answer))
            if strategy == "flip":
                new = 6 - old
            elif strategy == "noise":
                new = old + rng.randint(-magnitude, magnitude)
            elif strategy == "overconfident_wrong":
                new = 5 if not correct else old
            elif strategy == "underconfident_correct":
                new = 2 if correct else old
            elif strategy == "miscalibrate":
                new = 2 if correct else 5
            else:
                new = old
            after[i] = _clamp_confidence(new)

    changes = []
    abs_delta = 0.0
    for i, old in before.items():
        new = after[i]
        current[i]["confidence"] = new
        if new != old:
            changes.append({"agent_id": i, "old": old, "new": new})
            abs_delta += abs(new - old)
    event = {
        "event": "robustness_confidence_perturbation",
        "phase": phase,
        "round": round_idx,
        "strategy": strategy,
        "rate": rate,
        "changes": changes,
    }
    return [event], {
        "confidence_total": float(len(before)),
        "confidence_changed": float(len(changes)),
        "confidence_abs_delta": float(abs_delta),
    }


def _apply_critique_noise(
    critiques: List[Dict[str, Any]],
    *,
    source: int,
    n_agents: int,
    round_idx: int,
    cfg: Mapping[str, Any],
    rng,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    if not cfg or not critiques:
        return critiques, [], {
            "critique_original": float(len(critiques)),
            "critique_dropped": 0.0,
            "critique_corrupted": 0.0,
            "critique_source_swapped": 0.0,
        }
    rate = max(0.0, min(1.0, float(cfg.get("rate", 0.3))))
    strategy = str(cfg.get("strategy", "mixed")).strip().lower()
    noisy_text = str(
        cfg.get(
            "corrupt_correction",
            "This solution may contain a hidden flaw; reconsider the conclusion and compare alternatives.",
        )
    )
    out: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    stats = {
        "critique_original": float(len(critiques)),
        "critique_dropped": 0.0,
        "critique_corrupted": 0.0,
        "critique_source_swapped": 0.0,
    }
    for critique in critiques:
        if rng.random() > rate:
            out.append(critique)
            continue
        action = strategy
        if action == "mixed":
            action = rng.choice(["drop", "corrupt", "swap_source"])
        mutated = dict(critique)
        event = {
            "event": "robustness_critique_noise",
            "round": round_idx,
            "source": source,
            "target": critique.get("target"),
            "action": action,
        }
        if action == "drop":
            stats["critique_dropped"] += 1.0
            events.append(event)
            continue
        if action == "corrupt":
            mutated["step_loc"] = "Noisy injected critique."
            mutated["correction"] = noisy_text
            mutated["assessment"] = str(cfg.get("corrupt_assessment", "Strong"))
            stats["critique_corrupted"] += 1.0
        elif action in {"swap_source", "source_swap"}:
            target = int(critique.get("target", 0) or 0)
            choices = [i for i in range(1, n_agents + 1) if i not in {source, target}]
            if choices:
                mutated["source_override"] = rng.choice(choices)
                event["source_override"] = mutated["source_override"]
                stats["critique_source_swapped"] += 1.0
        out.append(mutated)
        events.append(event)
    return out, events, stats


def _add_robustness_stats(total: Dict[str, float], update: Mapping[str, float]) -> None:
    for key, value in update.items():
        try:
            total[key] = total.get(key, 0.0) + float(value)
        except (TypeError, ValueError):
            continue


def _robustness_diagnostics(
    *,
    task: Task,
    example,
    robustness_stats: Mapping[str, float],
    malicious_cfg: Mapping[str, Any],
    adversary_id: Optional[int],
    answer_history: List[Dict[int, str]],
    final_candidates: Mapping[int, str],
    influence: Mapping[int, float],
    decision: Optional[str],
    correct: bool,
    is_correct: Callable[[str], bool],
) -> Dict[str, float]:
    diag: Dict[str, float] = {}
    conf_total = robustness_stats.get("confidence_total", 0.0)
    diag["confidence_perturbation_rate"] = safe_div_local(
        robustness_stats.get("confidence_changed", 0.0), conf_total
    )
    diag["confidence_mean_abs_delta"] = safe_div_local(
        robustness_stats.get("confidence_abs_delta", 0.0), conf_total
    )
    crit_total = robustness_stats.get("critique_original", 0.0)
    changed = (
        robustness_stats.get("critique_dropped", 0.0)
        + robustness_stats.get("critique_corrupted", 0.0)
        + robustness_stats.get("critique_source_swapped", 0.0)
    )
    diag["critique_noise_rate"] = safe_div_local(changed, crit_total)
    diag["critique_drop_rate"] = safe_div_local(robustness_stats.get("critique_dropped", 0.0), crit_total)
    diag["critique_corrupt_rate"] = safe_div_local(robustness_stats.get("critique_corrupted", 0.0), crit_total)
    diag["critique_source_swap_rate"] = safe_div_local(robustness_stats.get("critique_source_swapped", 0.0), crit_total)

    if malicious_cfg and adversary_id is not None and answer_history:
        adv_final = str(final_candidates.get(adversary_id, ""))
        non_adv = [i for i in final_candidates if i != adversary_id]
        adopted = [i for i in non_adv if str(final_candidates.get(i, "")) == adv_final]
        initially_correct = [
            i for i in non_adv if bool(is_correct(str(answer_history[0].get(i, ""))))
        ]
        r2w = [i for i in initially_correct if not bool(is_correct(str(final_candidates.get(i, ""))))]
        diag.update(
            {
                "malicious_agent_present": 1.0,
                "adversary_final_correct": 1.0 if is_correct(adv_final) else 0.0,
                "adversary_final_influence": float(influence.get(adversary_id, 0.0)),
                "adversary_wrong_adoption_rate": safe_div_local(len(adopted), len(non_adv)),
                "non_adversary_r2w_rate_under_attack": safe_div_local(len(r2w), len(initially_correct)),
                "attack_success_rate": 1.0
                if (not correct and str(decision or "") == adv_final and not is_correct(adv_final))
                else 0.0,
            }
        )
    else:
        diag["malicious_agent_present"] = 0.0
    return diag


def _targeted_cross_eligibility(
    topo: List[List[int]],
    current_answers: Mapping[int, Dict[str, Any]],
    *,
    n_agents: int,
    source_confidence_min: float,
    target_confidence_max: float,
) -> Dict[str, Dict[str, Any]]:
    """Per-agent targeted-cross source/target eligibility, for one round.

    ``source_eligible`` / ``target_eligible`` are the two confidence gates in
    ``core.topology.score_state_permutation``: an agent may act as a
    targeted-cross *source* only at confidence >= ``source_confidence_min``,
    and may be *targeted* only at confidence <= ``target_confidence_max``.
    Inflating a report therefore moves an agent across both gates at once --
    gaining broadcast eligibility while shedding scrutiny.

    ``source_edges`` / ``target_edges`` count the edges on which each gate
    actually fired, i.e. the full predicate including answer disagreement.

    Pure diagnostic: derived from the already-selected topology and the
    confidences that produced it. It does not feed the routing computation.
    """
    confidences = {
        i: _clamp_confidence(current_answers[i].get("confidence", 0))
        for i in range(1, n_agents + 1)
    }
    answers = {
        i: str(current_answers[i].get("answer", "")) for i in range(1, n_agents + 1)
    }
    out: Dict[str, Dict[str, Any]] = {
        str(i): {
            "confidence": confidences[i],
            "source_eligible": confidences[i] >= float(source_confidence_min),
            "target_eligible": confidences[i] <= float(target_confidence_max),
            "source_edges": 0,
            "target_edges": 0,
        }
        for i in range(1, n_agents + 1)
    }
    for source, target in edge_list(topo):
        answers_differ = bool(
            answers[source] and answers[target] and answers[source] != answers[target]
        )
        if (
            answers_differ
            and confidences[source] >= float(source_confidence_min)
            and confidences[target] <= float(target_confidence_max)
        ):
            out[str(source)]["source_edges"] += 1
            out[str(target)]["target_edges"] += 1
    return out


def _install_mock_log_marker() -> None:
    """Prefix every subsequent log record with ``mock=true``.

    Attached to the root logger, so it covers the run-directory ``run.log``
    and the console alike. Idempotent: re-installing does not double-prefix.
    """
    import logging

    marker = "mock=true | "

    class _MockMarkerFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            message = str(record.getMessage())
            if not message.startswith(marker):
                record.msg = marker + message
                record.args = ()
            return True

    root = logging.getLogger()
    if any(isinstance(f, _MockMarkerFilter) for f in root.filters):
        return
    root.addFilter(_MockMarkerFilter())
    for handler in root.handlers:
        if not any(isinstance(f, _MockMarkerFilter) for f in handler.filters):
            handler.addFilter(_MockMarkerFilter())


def safe_div_local(num: float, den: float) -> float:
    return float("nan") if den == 0 else float(num) / float(den)

def _select_topology(
    *,
    mode: str,
    n_agents: int,
    perm_rng,
    base_role_topo,
    current_answers: Mapping[int, Dict[str, Any]],
    influence: Mapping[int, float],
    debate_cfg: Mapping[str, Any],
) -> tuple[List[int], List[List[int]], Dict[str, Any]]:
    if mode in {"fixed", "cot", "cot_sc", "mad"}:
        perm = list(range(1, n_agents + 1))
        info = {"candidate_count": 1, "selected_score": 0.0, "routing_mode": "identity"}
    elif mode == "random_k_regular":
        degree = int(debate_cfg.get("k_regular_degree", debate_cfg.get("degree", 2)))
        topo = make_base_topology("k_regular", n_agents, rng=perm_rng, degree=degree)
        info = {
            "candidate_count": 1,
            "selected_score": 0.0,
            "random_k_regular": True,
            "k_regular_degree": degree,
            "routing_mode": "random_k_regular",
        }
        return list(range(1, n_agents + 1)), topo, info
    elif mode == "pear_subgroup":
        perm = subgroup_permutation(n_agents, perm_rng)
        info = {"candidate_count": 1, "selected_score": 0.0, "routing_mode": "subgroup"}
    else:
        alpha_targeted_cross, alpha_inf, alpha_low_conf = _routing_weights(mode, debate_cfg)
        low_conf_threshold, targeted_source_min, targeted_target_max = _routing_thresholds(debate_cfg)
        normalize_terms = _routing_terms_normalized(debate_cfg)
        enumerate_perms, mc_candidates, enumeration_cap = _routing_enumeration(debate_cfg)
        answers = {i: current_answers[i].get("answer", "") for i in range(1, n_agents + 1)}
        confidences = {
            i: float(current_answers[i].get("confidence", 0))
            for i in range(1, n_agents + 1)
        }
        if mode == "pear_uniform":
            perm = uniform_permutation(n_agents, perm_rng)
            info = {"candidate_count": 1, "selected_score": 0.0, "routing_mode": "uniform"}
        else:
            perm, info = state_aware_permutation(
                n_agents,
                perm_rng,
                base_role_topo,
                answers=answers,
                confidences=confidences,
                influence=influence,
                alpha_targeted_cross=alpha_targeted_cross,
                alpha_influence=alpha_inf,
                alpha_low_confidence=alpha_low_conf,
                low_confidence_threshold=low_conf_threshold,
                targeted_cross_source_confidence_min=targeted_source_min,
                targeted_cross_target_confidence_max=targeted_target_max,
                normalize_terms=normalize_terms,
                candidates=mc_candidates,
                temperature=float(debate_cfg.get("routing_temperature", 1.0)),
                enumerate_permutations=enumerate_perms,
                enumeration_max_factorial=enumeration_cap,
            )
    topo = apply_perm_to_topology(base_role_topo, perm)
    return perm, topo, info


def _aggregate_candidates(cands: Dict[int, str], mode: str) -> Optional[str]:
    if mode != "majority_vote":
        # The direct runner keeps LLM-judge unavailable by design unless a
        # future version supplies a structured judge prompt with full traces.
        _log.warning("agg_mode=%s is not implemented in the two-phase runner; using majority_vote.", mode)
    return _majority_vote(cands)


def _is_correct_fn(task: Task, example) -> Callable[[str], bool]:
    return lambda answer: bool(task.score(task.parse_answer(answer) or answer, example))


# Per-example execution
def _make_initial_state(
    task: Task,
    example,
    *,
    debate_cfg: Mapping[str, Any],
    seed: int,
    perm_seed: int,
    judge_llm: Optional[BaseLLM] = None,
) -> Dict[str, Any]:
    """Create the pre-init state shared by serial and batched runners."""
    return {
        "x": task.format_question(example),
        "y_star": example.answer,
        "n_agents": int(debate_cfg.get("n_agents", 4)),
        "rounds": int(debate_cfg.get("rounds", 2)),
        "turns_per_round": int(debate_cfg.get("turns_per_round", 4)),
        "mode": debate_cfg.get("mode", "fixed"),
        "agg_mode": debate_cfg.get("agg_mode", "majority_vote"),
        "base_topology": debate_cfg.get("base_topology", "clique"),
        "max_tokens_per_call": int(debate_cfg.get("max_tokens_per_call", 256)),
        "shuffle_frequency": debate_cfg.get("shuffle_frequency", "per_round"),
        "shuffle_every_k": int(debate_cfg.get("shuffle_every_k", 2)),
        "seed": int(seed),
        "perm_seed": int(perm_seed),
        # Runtime-only channels consumed by the scorer / aggregator.
        "task_obj": task,
        "example_obj": example,
        "judge_llm_obj": judge_llm,
    }


def _row_from_final_state(
    final_state: Mapping[str, Any],
    example,
    *,
    seed: int,
    perm_seed: int,
) -> Dict[str, Any]:
    messages = list(final_state.get("messages", []))
    return {
        "example_id": example.id,
        "decision": final_state.get("decision"),
        "correct": bool(final_state.get("correct")) if final_state.get("correct") is not None else None,
        "n_messages": len(messages),
        "budget": final_state.get("budget", {}),
        "trace_events": final_state.get("metrics_trace", []),
        "messages": messages,
        "final_candidates": dict(final_state.get("final_candidates") or {}),
        "seed": int(seed),
        "perm_seed": int(perm_seed),
    }


def _random_baseline_decision(task: Task, example, rng) -> str:
    if example.choices:
        idx = rng.randrange(len(example.choices))
        return string.ascii_uppercase[idx]
    answer_pool = [str(ex.answer) for ex in task.examples if str(ex.answer).strip()]
    if answer_pool:
        return answer_pool[rng.randrange(len(answer_pool))]
    return ""


def run_one(
    task: Task,
    example,
    *,
    debate_cfg: Mapping[str, Any],
    llm: Optional[BaseLLM],
    seed: int,
    perm_seed: int,
    judge_llm: Optional[BaseLLM] = None,
) -> Dict[str, Any]:
    """Run one example through the ExpPlan_v3 two-phase PEAR loop."""
    mode = str(debate_cfg.get("mode", "pear_full"))
    n_agents = int(debate_cfg.get("n_agents", 6))
    if mode == "cot":
        n_agents = 1
    rounds = int(debate_cfg.get("rounds", 3))
    if mode in {"cot", "cot_sc"}:
        rounds = 0

    max_tokens = int(debate_cfg.get("max_tokens_per_call", 512))
    rng = seeded_rng(int(seed))
    perm_rng = seeded_rng(int(perm_seed))
    budget = Budget()
    question = task.format_question(example)
    is_correct = _is_correct_fn(task, example)

    robustness_cfg = _robustness_config(debate_cfg)
    malicious_cfg = _robustness_component(robustness_cfg, "malicious_agent")
    confidence_perturbation_cfg = _robustness_component(
        robustness_cfg, "confidence_perturbation"
    )
    critique_noise_cfg = _robustness_component(robustness_cfg, "critique_noise")
    # Independent stream so that randomising the adversary slot cannot shift the
    # robustness_rng draws consumed by perturbation/critique-noise components.
    adversary_rng = seeded_rng(int(seed) * 7_919 + int(perm_seed) * 104_729 + 41)
    adversary_candidates = _adversary_candidates(malicious_cfg, n_agents)
    adversary_id = _adversary_id(malicious_cfg, n_agents, rng=adversary_rng)
    adversary_confidence = _clamp_confidence(malicious_cfg.get("adversary_confidence", 5))
    adversary_sticky = bool(malicious_cfg.get("sticky", True))
    adversary_answer: Optional[str] = None
    robustness_rng = seeded_rng(
        int(seed) * 1_000_003 + int(perm_seed) * 9_176 + 13
    )
    robustness_stats: Dict[str, float] = {}

    trace_events: List[Dict[str, Any]] = [
        {
            "event": "init",
            "n_agents": n_agents,
            "rounds": rounds,
            "mode": mode,
            "seed": int(seed),
            "perm_seed": int(perm_seed),
            # Which agent was adversarial, and the set it was drawn from, so a
            # randomised adversary slot is attributable per example offline.
            "adversary_id": adversary_id,
            "adversary_candidates": adversary_candidates,
        }
    ]
    messages: List[Dict[str, Any]] = []

    if mode == "random":
        decision = _random_baseline_decision(task, example, rng)
        trace_events.append(
            {
                "event": "random_baseline",
                "answer": decision,
                "correct": is_correct(decision),
            }
        )
        return {
            "example_id": example.id,
            "decision": decision,
            "correct": is_correct(decision),
            "n_messages": 0,
            "budget": budget.to_dict(),
            "trace_events": trace_events,
            "messages": messages,
            "final_candidates": {1: decision},
            "seed": int(seed),
            "perm_seed": int(perm_seed),
            "answer_history": [{1: decision}],
            "confidence_history": [],
            "influence_history": [],
            "diagnostics": {},
        }

    # Round 0: independent answers.
    prompts = []
    for i in range(1, n_agents + 1):
        if adversary_id is not None and i == adversary_id:
            prompts.append(
                MALICIOUS_INITIAL_ANSWER_TEMPLATE.format(
                    agent_id=i,
                    question=question,
                    gold_answer=example.answer,
                    adversary_confidence=adversary_confidence,
                )
            )
        else:
            prompts.append(INITIAL_ANSWER_TEMPLATE.format(agent_id=i, question=question))
    gens = _generate_many(
        llm,
        prompts,
        max_tokens=max_tokens,
        system=AGENT_SYSTEM,
        agent_ids=list(range(1, n_agents + 1)),
        budget=budget,
    )
    current: Dict[int, Dict[str, Any]] = {}
    answer_history: List[Dict[int, str]] = []
    confidence_history: List[Dict[int, int]] = []
    for agent_id, gen in enumerate(gens, start=1):
        parsed = _parse_answer_payload(gen.text, task.parse_answer)
        if adversary_id is not None and agent_id == adversary_id:
            adversary_answer, event = _apply_malicious_payload(
                task=task,
                example=example,
                payload=parsed,
                fallback_answer=adversary_answer,
                confidence=adversary_confidence,
                phase="initial",
            )
            event["agent_id"] = agent_id
            trace_events.append(event)
        current[agent_id] = parsed
        messages.append(
            {
                "speaker": agent_id,
                "round": 0,
                "phase": "initial_answer",
                "content": gen.text,
            }
        )
        trace_events.append(
            {
                "event": "answer",
                "phase": "initial",
                "round": 0,
                "agent_id": agent_id,
                "answer": parsed["answer"],
                "confidence": parsed["confidence"],
                "reasoning": parsed["reasoning"],
                "correct": is_correct(parsed["answer"]),
                "prompt_tokens": gen.prompt_tokens,
                "completion_tokens": gen.completion_tokens,
            }
        )

    if confidence_perturbation_cfg:
        perturb_events, perturb_stats = _apply_confidence_perturbation(
            current,
            phase="initial",
            round_idx=0,
            cfg=confidence_perturbation_cfg,
            rng=robustness_rng,
            is_correct=is_correct,
        )
        trace_events.extend(perturb_events)
        _add_robustness_stats(robustness_stats, perturb_stats)

    answer_history.append({i: current[i]["answer"] for i in range(1, n_agents + 1)})
    confidence_history.append({i: current[i]["confidence"] for i in range(1, n_agents + 1)})

    base_topology_name = str(debate_cfg.get("base_topology", "k_regular"))
    base_role_topo = make_base_topology(
        base_topology_name,
        n_agents,
        rng=perm_rng,
        degree=int(debate_cfg.get("k_regular_degree", debate_cfg.get("degree", 3))),
    )
    influence = {i: 1.0 / max(1, n_agents) for i in range(1, n_agents + 1)}
    beta = float(debate_cfg.get("influence_beta", 0.7))
    _, targeted_source_min, targeted_target_max = _routing_thresholds(debate_cfg)

    update_events: List[Dict[str, Any]] = []
    edge_events: List[Dict[str, Any]] = []
    influence_history = [{i: influence[i] for i in range(1, n_agents + 1)}]

    for round_idx in range(1, rounds + 1):
        previous = deepcopy(current)
        perm, topo, route_info = _select_topology(
            mode=mode,
            n_agents=n_agents,
            perm_rng=perm_rng,
            base_role_topo=base_role_topo,
            current_answers=previous,
            influence=influence,
            debate_cfg=debate_cfg,
        )
        trace_events.append(
            {
                "event": "topology",
                "round": round_idx,
                "perm": perm,
                "topology": topo,
                "topology_hash": topology_hash(topo),
                "in_degree": [len(row) for row in topo],
                "targeted_cross_eligibility": _targeted_cross_eligibility(
                    topo,
                    previous,
                    n_agents=n_agents,
                    source_confidence_min=targeted_source_min,
                    target_confidence_max=targeted_target_max,
                ),
                **route_info,
            }
        )

        outgoing = out_neighbors(topo, n_agents)
        critique_prompts: List[str] = []
        critique_sources: List[int] = []
        source_targets: Dict[int, List[int]] = {}
        for source in range(1, n_agents + 1):
            targets = outgoing.get(source, [])
            source_targets[source] = targets
            if not targets:
                continue
            critique_template = (
                MALICIOUS_CRITIQUE_GENERATION_TEMPLATE
                if adversary_id is not None and source == adversary_id
                else CRITIQUE_GENERATION_TEMPLATE
            )
            critique_prompts.append(
                critique_template.format(
                    agent_id=source,
                    question=question,
                    own_answer=previous[source]["answer"],
                    own_confidence=previous[source]["confidence"],
                    own_reasoning=previous[source]["reasoning"],
                    targets=_render_targets(targets, previous),
                )
            )
            critique_sources.append(source)

        critiques_by_target: Dict[int, List[Dict[str, Any]]] = {
            i: [] for i in range(1, n_agents + 1)
        }
        if critique_prompts:
            critique_gens = _generate_many(
                llm,
                critique_prompts,
                max_tokens=max_tokens,
                system=AGENT_SYSTEM,
                agent_ids=critique_sources,
                budget=budget,
            )
            for source, gen in zip(critique_sources, critique_gens):
                messages.append(
                    {
                        "speaker": source,
                        "round": round_idx,
                        "phase": "critique",
                        "content": gen.text,
                    }
                )
                parsed_critiques = _parse_critiques(gen.text, source_targets[source])
                if critique_noise_cfg:
                    parsed_critiques, noise_events, noise_stats = _apply_critique_noise(
                        parsed_critiques,
                        source=source,
                        n_agents=n_agents,
                        round_idx=round_idx,
                        cfg=critique_noise_cfg,
                        rng=robustness_rng,
                    )
                    trace_events.extend(noise_events)
                    _add_robustness_stats(robustness_stats, noise_stats)
                for critique in parsed_critiques:
                    target = int(critique["target"])
                    actual_source = int(critique.pop("source_override", source))
                    enriched = {
                        **critique,
                        "source": actual_source,
                        "original_source": source,
                        "round": round_idx,
                    }
                    critiques_by_target[target].append(enriched)
                    cross_cluster = previous[actual_source]["answer"] != previous[target]["answer"]
                    source_confidence = previous[actual_source]["confidence"]
                    target_confidence = previous[target]["confidence"]
                    targeted_cross = (
                        cross_cluster
                        and float(source_confidence) >= targeted_source_min
                        and float(target_confidence) <= targeted_target_max
                    )
                    edge_event = {
                        "event": "critique_edge",
                        "round": round_idx,
                        "source": actual_source,
                        "original_source": source,
                        "target": target,
                        "cross_cluster": bool(cross_cluster),
                        "targeted_cross": bool(targeted_cross),
                        "source_confidence": source_confidence,
                        "target_confidence": target_confidence,
                        "source_answer": previous[actual_source]["answer"],
                        "target_answer": previous[target]["answer"],
                        "assessment": enriched.get("assessment"),
                    }
                    edge_events.append(edge_event)
                    trace_events.append(edge_event)

        update_prompts = []
        update_agents = []
        for agent_id in range(1, n_agents + 1):
            update_prompts.append(
                ANSWER_UPDATE_TEMPLATE.format(
                    agent_id=agent_id,
                    question=question,
                    previous_answer=previous[agent_id]["answer"],
                    previous_confidence=previous[agent_id]["confidence"],
                    previous_reasoning=previous[agent_id]["reasoning"],
                    critiques=_render_critiques_for_agent(
                        critiques_by_target[agent_id], previous
                    ),
                )
            )
            update_agents.append(agent_id)

        update_gens = _generate_many(
            llm,
            update_prompts,
            max_tokens=max_tokens,
            system=AGENT_SYSTEM,
            agent_ids=update_agents,
            budget=budget,
        )
        for agent_id, gen in zip(update_agents, update_gens):
            parsed = _parse_answer_payload(gen.text, task.parse_answer)
            if adversary_id is not None and agent_id == adversary_id and adversary_sticky:
                adversary_answer, malicious_event = _apply_malicious_payload(
                    task=task,
                    example=example,
                    payload=parsed,
                    fallback_answer=adversary_answer,
                    confidence=adversary_confidence,
                    phase="update",
                )
                malicious_event["agent_id"] = agent_id
                malicious_event["round"] = round_idx
                trace_events.append(malicious_event)
            if not parsed["critique_response"]:
                parsed["critique_response"] = {
                    str(c["source"]): {"decision": "REJECT", "reason": "No structured response parsed."}
                    for c in critiques_by_target[agent_id]
                }
            current[agent_id] = parsed
            event = {
                "event": "answer_update",
                "round": round_idx,
                "agent_id": agent_id,
                "previous_answer": previous[agent_id]["answer"],
                "answer": parsed["answer"],
                "confidence": parsed["confidence"],
                "reasoning": parsed["reasoning"],
                "critique_response": parsed["critique_response"],
                "flipped": previous[agent_id]["answer"] != parsed["answer"],
                "correct_before": is_correct(previous[agent_id]["answer"]),
                "correct_after": is_correct(parsed["answer"]),
                "prompt_tokens": gen.prompt_tokens,
                "completion_tokens": gen.completion_tokens,
            }
            update_events.append(event)
            trace_events.append(event)
            messages.append(
                {
                    "speaker": agent_id,
                    "round": round_idx,
                    "phase": "answer_update",
                    "content": gen.text,
                }
            )

        if confidence_perturbation_cfg:
            perturb_events, perturb_stats = _apply_confidence_perturbation(
                current,
                phase="update",
                round_idx=round_idx,
                cfg=confidence_perturbation_cfg,
                rng=robustness_rng,
                is_correct=is_correct,
            )
            trace_events.extend(perturb_events)
            _add_robustness_stats(robustness_stats, perturb_stats)

        outgoing_current = out_neighbors(topo, n_agents)
        new_influence: Dict[int, float] = {}
        for source in range(1, n_agents + 1):
            targets = outgoing_current.get(source, [])
            if targets:
                adopted = sum(
                    1
                    for target in targets
                    if current[target]["answer"] == previous[source]["answer"]
                )
                content = adopted / len(targets)
            else:
                content = 0.0
            raw = content
            new_influence[source] = beta * influence[source] + (1.0 - beta) * raw
        influence = new_influence
        influence_history.append({i: influence[i] for i in range(1, n_agents + 1)})

        answer_history.append({i: current[i]["answer"] for i in range(1, n_agents + 1)})
        confidence_history.append({i: current[i]["confidence"] for i in range(1, n_agents + 1)})

    final_candidates = {i: current[i]["answer"] for i in range(1, n_agents + 1)}
    decision = _aggregate_candidates(final_candidates, str(debate_cfg.get("agg_mode", "majority_vote")))
    prediction = task.parse_answer(decision or "") or (decision or "")
    correct = task.score(prediction, example)

    diagnostics = trajectory_event_rates(answer_history[0], final_candidates, is_correct)
    diagnostics.update(
        {
            "critique_precision": critique_precision(update_events, is_correct),
            "critique_acceptance_rate": critique_acceptance_rate(update_events),
            "cross_cluster_critique_rate": cross_cluster_critique_rate(edge_events),
            "targeted_cross_critique_rate": targeted_cross_critique_rate(edge_events),
            "influence_entropy": normalized_entropy(list(influence.values())),
            "answer_entropy_initial": float(math.nan) if not answer_history else 0.0,
            "answer_entropy_final": 0.0,
            "confidence_calibration": confidence_calibration(edge_events, update_events, is_correct),
        }
    )
    diagnostics.update(
        _robustness_diagnostics(
            task=task,
            example=example,
            robustness_stats=robustness_stats,
            malicious_cfg=malicious_cfg,
            adversary_id=adversary_id,
            answer_history=answer_history,
            final_candidates=final_candidates,
            influence=influence,
            decision=decision,
            correct=bool(correct),
            is_correct=is_correct,
        )
    )
    try:
        from metrics.diagnostics import entropy

        diagnostics["answer_entropy_initial"] = entropy(list(answer_history[0].values()))
        diagnostics["answer_entropy_final"] = entropy(list(final_candidates.values()))
    except Exception:
        pass

    trace_events.append(
        {
            "event": "aggregate",
            "agg_mode": debate_cfg.get("agg_mode", "majority_vote"),
            "decision": decision,
            "candidates": final_candidates,
        }
    )
    trace_events.append(
        {
            "event": "score",
            "prediction": prediction,
            "gold": example.answer,
            "correct": bool(correct),
        }
    )

    return {
        "example_id": example.id,
        "decision": decision,
        "correct": bool(correct),
        "n_messages": len(messages),
        "budget": budget.to_dict(),
        "trace_events": trace_events,
        "messages": messages,
        "final_candidates": final_candidates,
        "seed": int(seed),
        "perm_seed": int(perm_seed),
        "answer_history": answer_history,
        "confidence_history": confidence_history,
        "influence_history": influence_history,
        "diagnostics": diagnostics,
    }


def run_batch(
    task: Task,
    examples: List[Any],
    *,
    debate_cfg: Mapping[str, Any],
    llm: BaseLLM,
    seed: int,
    perm_seed: int,
    judge_llm: Optional[BaseLLM] = None,
) -> List[Dict[str, Any]]:
    """Run independent examples.

    The v3 runner batches agents inside each phase for one example. Cross-example
    lockstep batching can be reintroduced later without changing semantics.
    """
    return [
        run_one(
            task,
            ex,
            debate_cfg=debate_cfg,
            llm=llm,
            seed=seed,
            perm_seed=perm_seed,
            judge_llm=judge_llm,
        )
        for ex in examples
    ]


# Per-condition / experiment loops
def run_condition(
    *,
    name: str,
    debate_cfg: Mapping[str, Any],
    task: Task,
    llm: BaseLLM,
    seeds: Iterable[int],
    perm_seeds: Iterable[int],
    paths: RunPaths,
    tracer: JsonlTracer,
    judge_llm: Optional[BaseLLM] = None,
    parallel_examples: int = 1,
    show_progress: bool = True,
) -> RunResult:
    """Execute one debate condition across (seed x perm_seed x example)."""
    results: List[Dict[str, Any]] = []
    examples = list(task.examples)
    seed_list = list(seeds)
    perm_seed_list = list(perm_seeds)
    parallel_examples = max(1, int(parallel_examples))
    condition_meta = {
        # Tags every results row, trace event and transcript line produced by
        # this condition, so a mock number can never be read as a real one.
        "mock": bool(getattr(llm, "is_mock", False)),
        "mode": str(debate_cfg.get("mode", "")),
        "topology": str(debate_cfg.get("base_topology", "")),
        "base_topology": str(debate_cfg.get("base_topology", "")),
        "n_agents": int(debate_cfg.get("n_agents", 0) or 0),
        "rounds": int(debate_cfg.get("rounds", 0) or 0),
    }
    if "k_regular_degree" in debate_cfg or "degree" in debate_cfg:
        condition_meta["k_regular_degree"] = int(
            debate_cfg.get("k_regular_degree", debate_cfg.get("degree", 0)) or 0
        )

    def record_row(row: Dict[str, Any], ex) -> None:
        row["condition"] = name
        row.update(condition_meta)
        results.append(row)

        # Pull heavy fields out of the result row before writing the
        # compact summary; they get their own files below.
        trace_events = row.pop("trace_events", [])
        messages = row.pop("messages", [])
        final_cands = row.pop("final_candidates", {})

        # Per-event trace
        for ev in trace_events:
            tracer.write({"condition": name, **condition_meta, "example": ex.id, **ev})

        # Per-example metrics summary
        _append_jsonl(paths.results_file, {**row, "condition": name})

        # Per-example transcript: one JSON line carrying the full
        # ordered debate so a reader can reconstruct each agent's
        # turn-by-turn utterance without parsing trace.jsonl.
        _append_jsonl(
            paths.transcript_file,
            {
                "condition": name,
                **condition_meta,
                "example_id": ex.id,
                "seed": int(row["seed"]),
                "perm_seed": int(row["perm_seed"]),
                "question": ex.question,
                "gold": ex.answer,
                "decision": row.get("decision"),
                "correct": row.get("correct"),
                "final_candidates": {
                    str(k): v for k, v in final_cands.items()
                },
                "transcript": [
                    {
                        "agent_id": m["speaker"],
                        "round": m["round"],
                        "phase": m.get("phase", "turn"),
                        "turn": m.get("turn"),
                        "content": m["content"],
                    }
                    for m in messages
                ],
                "answer_history": row.get("answer_history", []),
                "confidence_history": row.get("confidence_history", []),
                "influence_history": row.get("influence_history", []),
                "diagnostics": row.get("diagnostics", {}),
            },
        )
        progress.update(1)

    total_runs = len(seed_list) * len(perm_seed_list) * len(examples)
    desc = f"{name}:{condition_meta['topology']}:{task.name}"
    with _progress_bar(total=total_runs, desc=desc, enabled=show_progress) as progress:
        for seed in seed_list:
            for perm_seed in perm_seed_list:
                if parallel_examples > 1:
                    for start in range(0, len(examples), parallel_examples):
                        chunk = examples[start : start + parallel_examples]
                        rows = run_batch(
                            task,
                            chunk,
                            debate_cfg=debate_cfg,
                            llm=llm,
                            seed=seed,
                            perm_seed=perm_seed,
                            judge_llm=judge_llm,
                        )
                        for ex, row in zip(chunk, rows):
                            record_row(row, ex)
                else:
                    for ex in examples:
                        row = run_one(
                            task,
                            ex,
                            debate_cfg=debate_cfg,
                            llm=llm,
                            seed=seed,
                            perm_seed=perm_seed,
                            judge_llm=judge_llm,
                        )
                        record_row(row, ex)

    # Aggregate.
    correct_flags = [bool(r.get("correct")) for r in results if r.get("correct") is not None]
    summary = summarise_runs([1.0 if c else 0.0 for c in correct_flags])
    diagnostic_summary = aggregate_diagnostics(results)
    out = RunResult(
        condition=name,
        accuracy=summary.mean,
        n_examples=len(examples),
        n_runs=len(results),
        summary={
            "mean": summary.mean,
            "std": summary.std,
            "iqr": list(summary.iqr),
            "min": summary.min,
            "max": summary.max,
            "n": summary.n,
            **condition_meta,
            **diagnostic_summary,
        },
    )
    _log.info(
        "[%s] mean=%.3f std=%.3f over %d runs (%d examples)",
        name, summary.mean, summary.std, len(results), len(examples),
    )
    return out


def _append_jsonl(path, row: Mapping[str, Any]) -> None:
    """Append a JSON line to ``path``; creates parents on first call."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str))
        fh.write("\n")


def run_experiment(config: ExperimentConfig) -> List[RunResult]:
    """Run all conditions in a config; return one :class:`RunResult` per condition.

    Side effects: creates a run directory, writes ``trace.jsonl``,
    ``results.jsonl``, ``config.yaml``, and ``summary.json``.
    """
    cfg = config.raw
    set_global_seeds(int(cfg.get("seed", 0)))

    # Output directory and logging
    paths_root = cfg.get("paths", {}).get("output_dir", "outputs")
    log_cfg = cfg.get("logging", {})
    paths = setup_run_logging(
        paths_root,
        tag=config.run_tag or cfg.get("agents", {}).get("model", "pear"),
        level=log_cfg.get("level", "INFO"),
        console=bool(log_cfg.get("console", True)),
    )

    # Snapshot config for reproducibility.
    with open(paths.config_file, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    tracer = JsonlTracer(paths.trace_file)
    _log.info("Starting experiment; outputs -> %s", paths.run_dir)

    # Dataset
    # ``split`` is left as ``None`` when the user didn't pick one so each
    # Task class can apply its own per-benchmark default (e.g. HotpotQA
    # defaults to validation because its public test split has null gold).
    ds_cfg = cfg.get("dataset", {})
    task = load_task(
        ds_cfg.get("name", "gsm8k"),
        split=ds_cfg.get("split"),
        num_examples=int(ds_cfg.get("num_examples", 0)),
        data_dir=cfg.get("paths", {}).get("data_dir", "data"),
    )
    _log.info(
        "Loaded task %s (split=%s, %d examples)",
        task.name, task.split, len(task),
    )

    # Conditions are resolved before model construction so random-only
    # baselines can avoid loading a real LLM backend.
    base_debate = cfg.get("debate", {})
    conditions = cfg.get("conditions") or [{"name": "default", "debate": {}}]
    condition_modes: List[str] = []
    for cond in conditions:
        merged = deepcopy(base_debate)
        _deep_merge(merged, cond.get("debate", {}))
        condition_modes.append(str(merged.get("mode", "")))
    random_only = bool(condition_modes) and all(mode == "random" for mode in condition_modes)

    # LLM
    model_name = cfg.get("agents", {}).get("model", "stub")
    registry_path = cfg.get("paths", {}).get("models_yaml", "configs/models.yaml")
    registry = load_model_registry(registry_path)
    model_overrides = cfg.get("agents", {}).get("model_overrides") or {}
    if random_only:
        llm = None
        _log.info("Skipping LLM construction for random-only baseline run")
    else:
        if model_overrides:
            models = registry.get("models", {})
            if model_name not in models:
                raise KeyError(f"Unknown model: {model_name!r}. Known: {sorted(models)}")
            model_spec = dict(models[model_name])
            model_spec.update(dict(model_overrides))
            model_spec["_name"] = model_name
            _log.info(
                "Applying model overrides for %s: %s",
                model_name,
                ", ".join(sorted(model_overrides)),
            )
        else:
            model_spec = model_name
        llm = build_llm(model_spec, registry=registry)

    if getattr(llm, "is_mock", False):
        _install_mock_log_marker()
        _log.warning(
            "MOCK PROVIDER ACTIVE - all outputs of this run are canned, not "
            "model-generated. Results are tagged mock: true."
        )

    judge_llm = None
    judge_name = cfg.get("agents", {}).get("judge_model")
    if judge_name and not random_only:
        judge_llm = build_llm(judge_name, registry=registry)

    # Conditions
    seeds = cfg.get("replication", {}).get("seeds", [0])
    perm_seeds = cfg.get("replication", {}).get("agent_perm_seeds", [0])

    runner_cfg = cfg.get("runner", {})
    parallel_examples = max(1, int(runner_cfg.get("parallel_examples", 1)))
    show_progress = bool(runner_cfg.get("show_progress", True))
    if parallel_examples > 1:
        _log.info("Using batched runner with parallel_examples=%d", parallel_examples)

    summaries: List[RunResult] = []
    for cond in conditions:
        merged = deepcopy(base_debate)
        _deep_merge(merged, cond.get("debate", {}))
        result = run_condition(
            name=cond.get("name", "default"),
            debate_cfg=merged,
            task=task,
            llm=llm,
            seeds=seeds,
            perm_seeds=perm_seeds,
            paths=paths,
            tracer=tracer,
            judge_llm=judge_llm,
            parallel_examples=parallel_examples,
            show_progress=show_progress,
        )
        summaries.append(result)

    # Final summary
    payload = {
        "run_dir": str(paths.run_dir),
        "mock": bool(getattr(llm, "is_mock", False)),
        "model": model_name,
        "judge_model": judge_name,
        "parallel_examples": parallel_examples,
        "conditions": [
            {
                "name": s.condition,
                "accuracy": s.accuracy,
                "n_runs": s.n_runs,
                "n_examples": s.n_examples,
                **s.summary,
            }
            for s in summaries
        ],
    }
    with open(paths.summary_file, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)

    tracer.close()
    _log.info("Experiment complete; summary -> %s", paths.summary_file)
    return summaries
