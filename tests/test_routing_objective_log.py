"""The routing objective must be self-describing on every routing row.

Figure 5 (ablation heatmap over tau x alpha_I) reads both axes off
``routing.jsonl``. Joining back to the run's config works only while tau is
constant within a run, and fails silently the moment tau is swept across
conditions -- so both axes belong on the row itself.
"""

from __future__ import annotations

import random

import pytest

from core.topology import make_base_topology, state_aware_permutation
from data.tasks import DummyTask
from models.model import MockLLM
from runner.experiment import run_one

N = 5
ANSWERS = {1: "42", 2: "42", 3: "42", 4: "42", 5: "7"}
CONFIDENCES = {1: 3, 2: 3, 3: 3, 4: 3, 5: 4}
INFLUENCE = {i: 0.2 for i in range(1, N + 1)}


def route(*, temperature, alphas):
    rng = random.Random(10)
    base = make_base_topology("k_regular", N, rng=rng, degree=3)
    _, info = state_aware_permutation(
        N,
        rng,
        base,
        answers=ANSWERS,
        confidences=CONFIDENCES,
        influence=INFLUENCE,
        temperature=temperature,
        **alphas,
    )
    return info


STATE_AWARE = {
    "alpha_targeted_cross": 0.2,
    "alpha_influence": 0.7,
    "alpha_low_confidence": 0.7,
}
UNIFORM = {
    "alpha_targeted_cross": 0.0,
    "alpha_influence": 0.0,
    "alpha_low_confidence": 0.0,
}


@pytest.mark.parametrize("temperature", [0.1, 0.7, 1.0, 3.0])
def test_state_aware_objective_records_the_temperature(temperature):
    objective = route(temperature=temperature, alphas=STATE_AWARE)["objective"]
    assert objective["routing_temperature"] == temperature
    # The other heatmap axis was already there; assert both, since the figure
    # needs the pair on one row.
    assert objective["alpha_influence"] == 0.7


def test_configured_zero_is_recorded_as_zero_not_the_floor():
    """The softmax floors at 1e-6; the log reports the cell you configured."""
    objective = route(temperature=0.0, alphas=STATE_AWARE)["objective"]
    assert objective["routing_temperature"] == 0.0


def test_uniform_branch_records_null_temperature():
    """No softmax runs when every alpha is zero, so no temperature applied."""
    objective = route(temperature=0.7, alphas=UNIFORM)["objective"]
    assert objective["routing_temperature"] is None


def test_temperature_reaches_the_routing_event_end_to_end():
    task = DummyTask()
    result = run_one(
        task,
        task.examples[0],
        debate_cfg={
            "n_agents": N,
            "rounds": 2,
            "mode": "pear_full",
            "base_topology": "k_regular",
            "k_regular_degree": 3,
            "agg_mode": "majority_vote",
            "max_tokens_per_call": 128,
            "mc_permutations": 20,
            "routing_temperature": 0.35,
            "alpha_targeted_cross": 0.2,
            "alpha_influence": 0.7,
            "alpha_low_confidence": 0.7,
            "influence_beta": 0.6,
        },
        llm=MockLLM(mock={"default": {"answer": "42", "confidence": 3}}),
        seed=0,
        perm_seed=10,
    )
    events = [e for e in result["trace_events"] if e.get("event") == "topology"]
    assert events
    for event in events:
        assert event["objective"]["routing_temperature"] == 0.35
        assert event["objective"]["alpha_influence"] == 0.7
