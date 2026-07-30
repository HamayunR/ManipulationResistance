"""Topology factory and permutation utilities.

A *role graph* is the abstract communication structure (e.g. "role 1 is the
hub of a star"). A *permutation* :math:`\\sigma` maps roles to agents, which
yields the *agent graph* that actually drives visibility during a debate
round.

The functions here are pure: they take inputs, return outputs, and never read
state, the file system, or RNGs that they did not receive as arguments.
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple


# Public type aliases

#: Adjacency expressed as incoming-neighbor lists, 1-indexed.
#: ``Adjacency[i-1]`` is the list of node IDs whose messages node ``i`` can
#: see. We keep node IDs 1-indexed throughout the codebase for human
#: readability; the index into Python lists is therefore always ``id - 1``.
Adjacency = List[List[int]]
Edge = Tuple[int, int]


# Base topology factory
def make_base_topology(
    name: str,
    n: int,
    *,
    rng: random.Random | None = None,
    degree: int | None = None,
) -> Adjacency:
    """Construct the role-space adjacency for a given topology name.

    Parameters
    ----------
    name:
        One of ``"clique"``, ``"star"``, ``"ring"``, ``"chain"``,
        ``"random_sparse"`` or ``"k_regular"``.
    n:
        Number of agents/roles.
    rng:
        Optional :class:`random.Random` instance used by stochastic topologies.
        A fresh ``Random`` is used if omitted; this is acceptable for "draw
        once, freeze" semantics described in ExpPlan.md, but callers that
        need reproducibility should pass an explicit RNG.

    Returns
    -------
    Adjacency
        Incoming-neighbor lists in role space (1-indexed IDs).

    Raises
    ------
    ValueError
        If ``name`` is unknown or ``n < 1``.

    Examples
    --------
    >>> sorted(make_base_topology("clique", 3)[0])
    [2, 3]
    >>> make_base_topology("ring", 3)
    [[3], [1], [2]]
    """
    if n < 1:
        raise ValueError(f"n_agents must be >= 1, got {n}")

    if name == "clique":
        return [[j for j in range(1, n + 1) if j != i] for i in range(1, n + 1)]

    if name == "ring":
        # Each node i sees node i-1 (with wrap-around).
        return [[((i - 2) % n) + 1] for i in range(1, n + 1)]

    if name == "chain":
        # Like a ring but without the wrap-around edge.
        topo: Adjacency = [[] for _ in range(n)]
        for i in range(2, n + 1):
            topo[i - 1] = [i - 1]
        return topo

    if name == "star":
        # Hub role = 1: hub sees everyone, everyone sees the hub.
        hub = 1
        topo = [[] for _ in range(n)]
        for i in range(1, n + 1):
            if i == hub:
                topo[i - 1] = [j for j in range(1, n + 1) if j != i]
            else:
                topo[i - 1] = [hub]
        return topo

    if name in {"random_sparse", "k_regular", "regular", "expander"}:
        rng_local = rng if rng is not None else random.Random()
        topo: Adjacency = []
        if name == "random_sparse" and degree is None:
            k = min(2, max(0, n - 1))
        else:
            k = min(max(0, int(degree if degree is not None else min(3, n - 1))), n - 1)
        for i in range(1, n + 1):
            others = [j for j in range(1, n + 1) if j != i]
            topo.append(sorted(rng_local.sample(others, k=k)))
        return topo

    raise ValueError(f"Unknown base_topology: {name!r}")


# Permutation applier
def apply_perm_to_topology(
    base_role_topo: Adjacency,
    perm_roles_to_agents: Sequence[int],
) -> Adjacency:
    """Convert a role-space adjacency into an agent-space adjacency via ``perm``.

    Let :math:`\\sigma` be the permutation such that ``perm[r-1] = sigma(r)``;
    that is, role ``r`` is played by agent ``sigma(r)`` this round. The
    induced agent graph has an edge ``(sigma(r), sigma(r'))`` for every
    role-edge ``(r, r')`` in the base graph.

    Parameters
    ----------
    base_role_topo:
        Role-space adjacency from :func:`make_base_topology`.
    perm_roles_to_agents:
        A permutation of ``[1, ..., n]`` (no validation beyond length is
        performed for speed; callers should ensure the input is a permutation).

    Returns
    -------
    Adjacency
        Agent-space adjacency; ``out[i-1]`` is the list of agents visible to
        agent ``i``.
    """
    n = len(base_role_topo)
    if len(perm_roles_to_agents) != n:
        raise ValueError(
            f"perm length {len(perm_roles_to_agents)} != n_agents {n}"
        )

    # Build agent -> role lookup (inverse of perm).
    agent_to_role = {agent: role for role, agent in enumerate(perm_roles_to_agents, start=1)}
    topo_agent: Adjacency = [[] for _ in range(n)]
    for agent_i in range(1, n + 1):
        role_i = agent_to_role[agent_i]
        incoming_roles = base_role_topo[role_i - 1]
        topo_agent[agent_i - 1] = [perm_roles_to_agents[r - 1] for r in incoming_roles]
    return topo_agent


def edge_list(topo: Adjacency) -> List[Edge]:
    """Return ``(source, target)`` edges from an incoming-neighbor adjacency."""
    edges: List[Edge] = []
    for target, incoming in enumerate(topo, start=1):
        for source in incoming:
            edges.append((int(source), target))
    return edges


def out_neighbors(topo: Adjacency, n: int | None = None) -> Dict[int, List[int]]:
    """Map each source agent to the target agents that can hear it."""
    n = n or len(topo)
    out = {i: [] for i in range(1, n + 1)}
    for source, target in edge_list(topo):
        out.setdefault(source, []).append(target)
    for targets in out.values():
        targets.sort()
    return out


def _answer_at(answers: Mapping[int, str] | Sequence[str], agent_id: int) -> str:
    if isinstance(answers, Mapping):
        return str(answers.get(agent_id, ""))
    idx = agent_id - 1
    return str(answers[idx]) if 0 <= idx < len(answers) else ""


def _value_at(values: Mapping[int, float] | Sequence[float], agent_id: int) -> float:
    if isinstance(values, Mapping):
        return float(values.get(agent_id, 0.0))
    idx = agent_id - 1
    return float(values[idx]) if 0 <= idx < len(values) else 0.0


def score_state_permutation(
    base_role_topo: Adjacency,
    perm_roles_to_agents: Sequence[int],
    *,
    answers: Mapping[int, str] | Sequence[str],
    confidences: Mapping[int, float] | Sequence[float],
    influence: Mapping[int, float] | Sequence[float],
    alpha_targeted_cross: float = 0.0,
    alpha_influence: float = 0.0,
    alpha_low_confidence: float = 0.0,
    low_confidence_threshold: float = 3.0,
    targeted_cross_source_confidence_min: float = 4.0,
    targeted_cross_target_confidence_max: float = 3.0,
    normalize_terms: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """Score one candidate permutation under the targeted PEAR objective.

    The code uses source out-degree as structural exposure because the local
    adjacency convention stores incoming neighbors by target. This is the
    operational quantity that decides how many agents can receive a source's
    critique.
    """
    topo = apply_perm_to_topology(base_role_topo, perm_roles_to_agents)
    edges = edge_list(topo)

    targeted_cross = 0.0
    answer_disagreement = 0.0
    low_confidence_penalty = 0.0
    out_degree = {agent: 0 for agent in range(1, len(topo) + 1)}
    for source, target in edges:
        src_answer = _answer_at(answers, source)
        tgt_answer = _answer_at(answers, target)
        source_confidence = _value_at(confidences, source)
        target_confidence = _value_at(confidences, target)
        answers_differ = bool(src_answer and tgt_answer and src_answer != tgt_answer)
        if answers_differ:
            answer_disagreement += 1.0
            if (
                source_confidence >= float(targeted_cross_source_confidence_min)
                and target_confidence <= float(targeted_cross_target_confidence_max)
            ):
                targeted_cross += 1.0
        low_confidence_penalty += max(
            0.0,
            float(low_confidence_threshold) + 1.0 - source_confidence,
        )
        out_degree[source] = out_degree.get(source, 0) + 1

    influence_term = sum(
        _value_at(influence, source) * degree
        for source, degree in out_degree.items()
    )

    edge_count = max(1.0, float(len(edges)))
    max_low_confidence_penalty = max(1e-6, float(low_confidence_threshold))
    targeted_cross_objective = targeted_cross / edge_count
    influence_objective = influence_term / edge_count
    low_confidence_objective = low_confidence_penalty / (
        edge_count * max_low_confidence_penalty
    )
    if not normalize_terms:
        targeted_cross_objective = targeted_cross
        influence_objective = influence_term
        low_confidence_objective = low_confidence_penalty

    score = (
        float(alpha_targeted_cross) * targeted_cross_objective
        - float(alpha_influence) * influence_objective
        - float(alpha_low_confidence) * low_confidence_objective
    )
    return score, {
        "targeted_cross_edges": targeted_cross,
        "answer_disagreement_edges": answer_disagreement,
        "influence_penalty": influence_term,
        "low_confidence_penalty": low_confidence_penalty,
        "targeted_cross_rate": targeted_cross / edge_count,
        "answer_disagreement_rate": answer_disagreement / edge_count,
        "influence_penalty_rate": influence_term / edge_count,
        "low_confidence_penalty_rate": low_confidence_penalty
        / (edge_count * max_low_confidence_penalty),
        "targeted_cross_objective": targeted_cross_objective,
        "influence_objective": influence_objective,
        "low_confidence_objective": low_confidence_objective,
    }


def state_aware_permutation(
    n: int,
    rng: random.Random,
    base_role_topo: Adjacency,
    *,
    answers: Mapping[int, str] | Sequence[str],
    confidences: Mapping[int, float] | Sequence[float],
    influence: Mapping[int, float] | Sequence[float],
    alpha_targeted_cross: float = 0.0,
    alpha_influence: float = 0.0,
    alpha_low_confidence: float = 0.0,
    low_confidence_threshold: float = 3.0,
    targeted_cross_source_confidence_min: float = 4.0,
    targeted_cross_target_confidence_max: float = 3.0,
    normalize_terms: bool = True,
    candidates: int = 100,
    temperature: float = 1.0,
    enumerate_permutations: bool = False,
    enumeration_max_factorial: int = 5040,
) -> Tuple[List[int], Dict[str, Any]]:
    """Sample a state-aware role permutation by Monte Carlo softmax.

    Two candidate-pool constructions are available.

    ``enumerate_permutations=False`` (the default) is the original sampled pool:
    the identity permutation, then i.i.d. uniform draws *with replacement* until
    the pool holds ``candidates`` entries. Duplicates are kept, so multiplicity
    acts as an importance weight in the softmax normalisation and the identity
    carries one extra unit of weight relative to every other permutation. This
    is the vanilla-PEAR baseline and is deliberately left bit-identical.

    ``enumerate_permutations=True`` builds the pool by exact enumeration of
    :math:`S_n`. Every permutation appears exactly once, the identity is not
    seeded a second time, and the softmax is therefore taken over the whole
    symmetric group -- which makes the sampled distribution the one that an
    offline expectation over all ``n!`` permutations actually computes. Refuses
    with :class:`ValueError` when ``n! > enumeration_max_factorial``.

    The returned info dict reports ``routing_mode`` (``"sampled"`` or
    ``"enumerated"``) and ``identity_seeded`` so every round is attributable to
    the pool construction that produced it.
    """
    if n <= 1:
        return [1], {
            "candidate_count": 1,
            "selected_score": 0.0,
            "routing_mode": "enumerated" if enumerate_permutations else "sampled",
            "identity_seeded": False,
            "permutation_space_size": 1,
        }

    if enumerate_permutations:
        space_size = math.factorial(n)
        cap = int(enumeration_max_factorial)
        if space_size > cap:
            raise ValueError(
                f"enumerate_permutations=True requires n! <= "
                f"enumeration_max_factorial, but n={n} gives n!={space_size} > "
                f"{cap}. Raise enumeration_max_factorial explicitly, or use "
                f"sampled routing by setting mc_permutations to an integer."
            )
        # Exact pool over S_n. itertools yields the identity first; it is not
        # seeded again, so no permutation carries extra softmax multiplicity.
        perms: List[List[int]] = [list(p) for p in itertools.permutations(range(1, n + 1))]
        routing_mode = "enumerated"
        identity_seeded = False
    else:
        # Vanilla PEAR baseline: identity-seeded, sampled with replacement, no
        # dedupe. Do not alter -- harness validation depends on this being
        # bit-identical, including the number of RNG draws consumed.
        candidates = max(1, int(candidates))
        perms = [list(range(1, n + 1))]
        while len(perms) < candidates:
            perms.append(uniform_permutation(n, rng))
        routing_mode = "sampled"
        identity_seeded = True

    pool_meta = {
        "routing_mode": routing_mode,
        "identity_seeded": identity_seeded,
        "permutation_space_size": math.factorial(n),
    }

    if not any([alpha_targeted_cross, alpha_influence, alpha_low_confidence]):
        selected = perms[rng.randrange(len(perms))]
        return selected, {
            "candidate_count": len(perms),
            "selected_score": 0.0,
            **pool_meta,
            "objective": {
                "alpha_targeted_cross": alpha_targeted_cross,
                "alpha_influence": alpha_influence,
                "alpha_low_confidence": alpha_low_confidence,
                "low_confidence_threshold": low_confidence_threshold,
                "targeted_cross_source_confidence_min": targeted_cross_source_confidence_min,
                "targeted_cross_target_confidence_max": targeted_cross_target_confidence_max,
                "normalize_routing_terms": bool(normalize_terms),
            },
        }

    scored = []
    for perm in perms:
        score, terms = score_state_permutation(
            base_role_topo,
            perm,
            answers=answers,
            confidences=confidences,
            influence=influence,
            alpha_targeted_cross=alpha_targeted_cross,
            alpha_influence=alpha_influence,
            alpha_low_confidence=alpha_low_confidence,
            low_confidence_threshold=low_confidence_threshold,
            targeted_cross_source_confidence_min=targeted_cross_source_confidence_min,
            targeted_cross_target_confidence_max=targeted_cross_target_confidence_max,
            normalize_terms=normalize_terms,
        )
        scored.append((perm, score, terms))

    max_score = max(item[1] for item in scored)
    temp = max(1e-6, float(temperature))
    weights = [math.exp((item[1] - max_score) / temp) for item in scored]
    total = sum(weights)
    pick = rng.random() * total
    cursor = 0.0
    selected_idx = len(scored) - 1
    for idx, weight in enumerate(weights):
        cursor += weight
        if cursor >= pick:
            selected_idx = idx
            break

    selected_perm, selected_score, selected_terms = scored[selected_idx]
    return list(selected_perm), {
        "candidate_count": len(perms),
        "selected_score": selected_score,
        **pool_meta,
        "selected_terms": selected_terms,
        "score_min": min(item[1] for item in scored),
        "score_max": max(item[1] for item in scored),
        "score_mean": sum(item[1] for item in scored) / len(scored),
        "objective": {
            "alpha_targeted_cross": alpha_targeted_cross,
            "alpha_influence": alpha_influence,
            "alpha_low_confidence": alpha_low_confidence,
            "low_confidence_threshold": low_confidence_threshold,
            "targeted_cross_source_confidence_min": targeted_cross_source_confidence_min,
            "targeted_cross_target_confidence_max": targeted_cross_target_confidence_max,
            "normalize_routing_terms": bool(normalize_terms),
        },
    }


# Edge dropout (mechanism-isolation control)
def edge_dropout(topo: Adjacency, drop_p: float, rng: random.Random) -> Adjacency:
    """Randomly drop edges from an adjacency with i.i.d. probability ``drop_p``.

    Used as a mechanism-isolation control (ExpPlan.md section 3.3): preserves
    expected graph sparsity but breaks the *permutation averaging* effect that
    is hypothesised to drive PEAR's gains.
    """
    if not 0.0 <= drop_p <= 1.0:
        raise ValueError(f"drop_p must be in [0, 1], got {drop_p}")
    return [[j for j in incoming if rng.random() > drop_p] for incoming in topo]


# Sample permutations
def uniform_permutation(n: int, rng: random.Random) -> List[int]:
    """Draw a uniform permutation of ``[1..n]`` using ``rng``."""
    perm = list(range(1, n + 1))
    rng.shuffle(perm)
    return perm


def subgroup_permutation(
    n: int,
    rng: random.Random,
    *,
    blocks: List[int] | None = None,
) -> List[int]:
    """Draw a permutation that only mixes *within* the given blocks.

    Parameters
    ----------
    n:
        Number of agents.
    rng:
        Random source.
    blocks:
        List of block sizes summing to ``n``. When omitted, the agents are
        split into two roughly equal blocks.

    Returns
    -------
    List[int]
        A 1-indexed permutation that fixes block membership but shuffles
        agents within each block.
    """
    if blocks is None:
        mid = n // 2
        blocks = [mid, n - mid]
    if sum(blocks) != n:
        raise ValueError(f"block sizes {blocks} must sum to n={n}")

    perm = list(range(1, n + 1))
    cursor = 0
    for size in blocks:
        chunk = perm[cursor:cursor + size]
        rng.shuffle(chunk)
        perm[cursor:cursor + size] = chunk
        cursor += size
    return perm


def topology_hash(topo: Adjacency) -> str:
    """Deterministic short hash for a topology, useful for trace logs."""
    import hashlib

    canon = ";".join(",".join(str(j) for j in sorted(row)) for row in topo)
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()[:12]
