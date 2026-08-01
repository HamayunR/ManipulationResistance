"""Tests for the verified-confidence defence (``debate.verified_confidence``).

The router scores ``min(reported, g_i)``: an agent may talk itself down, but
cannot report its way above the level the evidence supports (proposal
Theorem 1). These tests pin the no-op mode, the clamp arithmetic, the
provenance the logs must carry, and -- importantly -- that a missing
corroboration score is never defaulted.
"""

from __future__ import annotations

import pytest

from data.tasks import DummyTask
from models.model import MockLLM
from runner.experiment import (
    _apply_verified_confidence,
    _confidence_log_maps,
    _corroboration_scores,
    _verified_confidence_config,
    run_one,
)

N_AGENTS = 5

# dummy-3 is "What is 2 * 3?", gold "6". Agents 1-4 are correct; agent 5 is
# wrong and inflates, so the oracle separates attacker from honest agents.
EXAMPLE_IDX = 2
ISOLATING_MOCK = {
    1: {"answer": "6", "confidence": 3},
    2: {"answer": "6", "confidence": 3},
    3: {"answer": "6", "confidence": 3},
    4: {"answer": "6", "confidence": 3},
    5: {"answer": "7", "confidence": 2},
}

ATTACK = {
    "enabled": True,
    "type": "confidence_inflation",
    "confidence_inflation": {
        "enabled": True,
        "agent_ids": [5],
        "mode": "fixed_report",
        "value": 4,
    },
}


def make_debate_cfg(robustness=None, verified=None):
    cfg = {
        "n_agents": N_AGENTS,
        "rounds": 2,
        "mode": "pear_full",
        "base_topology": "k_regular",
        "k_regular_degree": 3,
        "agg_mode": "majority_vote",
        "max_tokens_per_call": 128,
        "mc_permutations": 20,
        "routing_temperature": 0.7,
        "alpha_targeted_cross": 0.2,
        "alpha_influence": 0.7,
        "alpha_low_confidence": 0.7,
        "low_confidence_threshold": 3,
        "targeted_cross_source_confidence_min": 4,
        "targeted_cross_target_confidence_max": 3,
        "influence_beta": 0.6,
    }
    if robustness:
        cfg["robustness"] = robustness
    if verified:
        cfg["verified_confidence"] = verified
    return cfg


def run(robustness=None, verified=None):
    task = DummyTask()
    return run_one(
        task,
        task.examples[EXAMPLE_IDX],
        debate_cfg=make_debate_cfg(robustness, verified),
        llm=MockLLM(
            mock={"agents": ISOLATING_MOCK, "default": {"answer": "42", "confidence": 3}}
        ),
        seed=0,
        perm_seed=10,
    )


def topology_events(result):
    return [e for e in result["trace_events"] if e.get("event") == "topology"]


def verification_events(result):
    return [e for e in result["trace_events"] if e.get("event") == "verified_confidence"]


def states(confidence=3, answer="6"):
    return {
        i: {
            "answer": answer,
            "confidence": confidence,
            "clean_confidence": confidence,
            "reasoning": "r",
        }
        for i in range(1, N_AGENTS + 1)
    }


# ------------------------------------------------------------------ config ----
def test_absent_block_is_noop():
    assert _verified_confidence_config({}) is None


def test_null_block_is_noop():
    assert _verified_confidence_config({"verified_confidence": None}) is None


def test_mode_none_is_noop():
    assert _verified_confidence_config({"verified_confidence": {"mode": "none"}}) is None


def test_mode_defaults_to_none():
    assert _verified_confidence_config({"verified_confidence": {}}) is None


def test_oracle_mode_resolves():
    assert _verified_confidence_config({"verified_confidence": {"mode": "oracle"}}) == {
        "mode": "oracle"
    }


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown debate.verified_confidence.mode"):
        _verified_confidence_config({"verified_confidence": {"mode": "min"}})


def test_non_mapping_block_raises():
    with pytest.raises(ValueError, match="must be a mapping"):
        _verified_confidence_config({"verified_confidence": "oracle"})


def test_block_under_robustness_raises():
    """It is a mechanism, not an attack; silently ignoring it would be worse."""
    with pytest.raises(ValueError, match="move it"):
        _verified_confidence_config(
            {"robustness": {"enabled": True, "verified_confidence": {"mode": "oracle"}}}
        )


def test_agreement_mode_is_not_silently_a_noop():
    # Resolves fine ...
    assert _verified_confidence_config({"verified_confidence": {"mode": "agreement"}}) == {
        "mode": "agreement"
    }
    # ... but refuses to run rather than behaving like `none`.
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        _corroboration_scores(
            states(), n_agents=N_AGENTS, mode="agreement", is_correct=lambda a: True
        )


# ------------------------------------------------------------------- clamp ----
def test_oracle_scores_are_five_when_correct_one_when_wrong():
    current = states()
    current[5]["answer"] = "wrong"
    scores = _corroboration_scores(
        current,
        n_agents=N_AGENTS,
        mode="oracle",
        is_correct=lambda answer: answer != "wrong",
    )
    assert scores == {1: 5.0, 2: 5.0, 3: 5.0, 4: 5.0, 5: 1.0}


def test_clamp_reduces_an_inflated_report():
    current = states()
    current[5].update(answer="wrong", confidence=4)
    _apply_verified_confidence(
        current,
        phase="initial",
        round_idx=0,
        cfg={"mode": "oracle"},
        n_agents=N_AGENTS,
        is_correct=lambda answer: answer != "wrong",
    )
    assert current[5]["reported_confidence"] == 4
    assert current[5]["g_i"] == 1.0
    assert current[5]["verified_confidence"] == 1.0
    assert current[5]["confidence"] == 1  # what the router reads


def test_clamp_never_raises_a_low_report():
    """min(), not replacement: a corroborated agent keeps its modest report."""
    current = states(confidence=2)
    _apply_verified_confidence(
        current,
        phase="initial",
        round_idx=0,
        cfg={"mode": "oracle"},
        n_agents=N_AGENTS,
        is_correct=lambda answer: True,
    )
    for agent_id in range(1, N_AGENTS + 1):
        assert current[agent_id]["g_i"] == 5.0
        assert current[agent_id]["verified_confidence"] == 2.0
        assert current[agent_id]["confidence"] == 2


def test_clamp_applies_to_every_agent_not_just_attacked_ones():
    current = states()
    current[2]["answer"] = "wrong"
    _apply_verified_confidence(
        current,
        phase="initial",
        round_idx=0,
        cfg={"mode": "oracle"},
        n_agents=N_AGENTS,
        is_correct=lambda answer: answer != "wrong",
    )
    assert current[2]["confidence"] == 1
    assert current[1]["confidence"] == 3


def test_inactive_cfg_is_a_complete_noop():
    current = states()
    before = {i: dict(state) for i, state in current.items()}
    events = _apply_verified_confidence(
        current,
        phase="initial",
        round_idx=0,
        cfg=None,
        n_agents=N_AGENTS,
        is_correct=lambda answer: False,
    )
    assert events == []
    assert current == before  # no g_i / verified_confidence keys added either


def test_clean_confidence_survives_the_clamp():
    current = states()
    current[5].update(answer="wrong", confidence=4, clean_confidence=2)
    _apply_verified_confidence(
        current,
        phase="initial",
        round_idx=0,
        cfg={"mode": "oracle"},
        n_agents=N_AGENTS,
        is_correct=lambda answer: answer != "wrong",
    )
    assert current[5]["clean_confidence"] == 2  # model's own report, untouched


def test_missing_g_i_raises_and_is_never_defaulted(monkeypatch):
    """A hole in the corroboration map must not become 0.0 (or anything else)."""
    import runner.experiment as experiment

    monkeypatch.setattr(
        experiment,
        "_corroboration_scores",
        lambda *a, **k: {1: 5.0, 2: 5.0},  # agents 3-5 missing
    )
    with pytest.raises(ValueError, match=r"no g_i for agents \[3, 4, 5\]"):
        experiment._apply_verified_confidence(
            states(),
            phase="initial",
            round_idx=0,
            cfg={"mode": "oracle"},
            n_agents=N_AGENTS,
            is_correct=lambda answer: True,
        )


# ----------------------------------------------------------------- logging ----
def test_trace_event_carries_all_four_values_per_agent():
    result = run(robustness=ATTACK, verified={"mode": "oracle"})
    events = verification_events(result)
    assert events, "expected a verified_confidence event"
    for event in events:
        assert event["mode"] == "oracle"
        assert {a["agent_id"] for a in event["agents"]} == set(range(1, N_AGENTS + 1))
        for entry in event["agents"]:
            assert set(entry) >= {
                "agent_id",
                "clean_confidence",
                "reported_confidence",
                "g_i",
                "verified_confidence",
                "clamped",
            }
    attacker = [a for a in events[0]["agents"] if a["agent_id"] == 5][0]
    assert (attacker["reported_confidence"], attacker["g_i"]) == (4.0, 1.0)
    assert attacker["verified_confidence"] == 1.0
    assert attacker["clamped"] is True


def test_routing_events_carry_complete_g_and_verified_maps():
    result = run(robustness=ATTACK, verified={"mode": "oracle"})
    expected = {str(i) for i in range(1, N_AGENTS + 1)}
    events = topology_events(result)
    assert events
    for event in events:
        assert event["verified_confidence_mode"] == "oracle"
        assert set(event["g_i"]) == expected
        assert set(event["verified_confidence"]) == expected
        # The report is preserved alongside the value actually scored.
        assert event["reported_confidence"]["5"] == 4.0
        assert event["verified_confidence"]["5"] == 1.0


def test_routing_events_null_the_maps_when_verification_is_off():
    """None, not {} or zeros: 'no verification ran' must be unambiguous."""
    result = run(robustness=ATTACK)
    for event in topology_events(result):
        assert event["g_i"] is None
        assert event["verified_confidence"] is None
        assert event["verified_confidence_mode"] == "none"


def test_incomplete_verified_maps_raise():
    current = states()
    for state in current.values():
        state.update(reported_confidence=3, g_i=5.0, verified_confidence=3.0)

    del current[4]["g_i"]
    with pytest.raises(ValueError, match="Incomplete g_i routing log"):
        _confidence_log_maps(current, N_AGENTS, verified_active=True)

    current[4]["g_i"] = 5.0
    del current[4]["verified_confidence"]
    with pytest.raises(ValueError, match="Incomplete verified_confidence routing log"):
        _confidence_log_maps(current, N_AGENTS, verified_active=True)


def test_reported_falls_back_to_confidence_only_when_verification_is_off():
    """With no verification the router reads `confidence`, so that IS the report."""
    current = states()
    clean, reported, g_scores, verified = _confidence_log_maps(current, N_AGENTS)
    assert reported == {str(i): 3.0 for i in range(1, N_AGENTS + 1)}
    assert (g_scores, verified) == (None, None)

    # With verification active the pre-clamp report is required, not inferred.
    with pytest.raises(ValueError, match="Incomplete reported_confidence routing log"):
        _confidence_log_maps(current, N_AGENTS, verified_active=True)


# ------------------------------------------------------------ end to end ----
def test_oracle_reverses_the_attack_on_the_gates():
    clean = topology_events(run())[0]["targeted_cross_eligibility"]["5"]
    attacked = topology_events(run(robustness=ATTACK))[0]["targeted_cross_eligibility"]["5"]
    defended = topology_events(run(robustness=ATTACK, verified={"mode": "oracle"}))[0][
        "targeted_cross_eligibility"
    ]["5"]

    assert (clean["source_eligible"], clean["target_eligible"]) == (False, True)
    assert (attacked["source_eligible"], attacked["target_eligible"]) == (True, False)
    assert (defended["source_eligible"], defended["target_eligible"]) == (False, True)
    assert defended["source_edges"] == clean["source_edges"]


def test_defence_runs_without_any_attack_configured():
    """Needed to price the clean-accuracy cost of the defence (RQ6)."""
    result = run(verified={"mode": "oracle"})
    assert verification_events(result)
    for event in topology_events(result):
        assert event["verified_confidence_mode"] == "oracle"
        # Honest agents corroborate, so their reports survive untouched.
        assert event["verified_confidence"]["1"] == event["reported_confidence"]["1"]
