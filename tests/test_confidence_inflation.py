"""Tests for the scoped confidence-reporting attack (``confidence_inflation``).

The component overwrites the *reported* confidence of explicitly listed agents
and nothing else. These tests pin three things:

* the no-op cases really are no-ops, so an inactive component cannot perturb a
  clean run;
* scoping and field preservation, so a routing change is attributable to the
  report alone;
* every malformed config raises instead of silently clamping or degrading to a
  no-op.

The end-to-end cases drive :func:`runner.experiment.run_one` against the
offline mock backend, so they assert on the events analysis code actually
reads rather than on internal state.
"""

from __future__ import annotations

import pytest

from data.tasks import DummyTask
from models.model import MockLLM
from runner.experiment import (
    _apply_confidence_inflation,
    _confidence_inflation_config,
    run_one,
)

N_AGENTS = 5

# Agents 1-4 agree on "42" at confidence 3; agent 5 dissents with "7" at 2.
# Mirrors configs/models.yaml: mock-agent.
MOCK_AGENTS = {
    1: {"answer": "42", "confidence": 3},
    2: {"answer": "42", "confidence": 3},
    3: {"answer": "42", "confidence": 3},
    4: {"answer": "42", "confidence": 3},
    5: {"answer": "7", "confidence": 2},
}


def make_llm() -> MockLLM:
    return MockLLM(mock={"agents": MOCK_AGENTS, "default": {"answer": "42", "confidence": 3}})


def make_debate_cfg(robustness=None) -> dict:
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
    if robustness is not None:
        cfg["robustness"] = robustness
    return cfg


def run(robustness=None, example_idx: int = 0) -> dict:
    """Run one dummy example end to end on the mock backend."""
    task = DummyTask()
    return run_one(
        task,
        task.examples[example_idx],
        debate_cfg=make_debate_cfg(robustness),
        llm=make_llm(),
        seed=0,
        perm_seed=10,
    )


def states(n=N_AGENTS, answer="42", confidence=3):
    """Fresh per-agent state as the parser would produce it."""
    return {
        i: {
            "answer": answer,
            "confidence": confidence,
            "clean_confidence": confidence,
            "reasoning": f"reasoning for agent {i}",
            "critique_response": {"1": {"decision": "REJECT", "reason": "no"}},
        }
        for i in range(1, n + 1)
    }


def inflation_events(result) -> list:
    return [
        e
        for e in result["trace_events"]
        if e.get("event") == "robustness_confidence_inflation"
    ]


def topology_events(result) -> list:
    return [e for e in result["trace_events"] if e.get("event") == "topology"]


# ---------------------------------------------------------------- no-ops ----
def test_missing_robustness_block_is_noop():
    assert _confidence_inflation_config({}, N_AGENTS) is None

    result = run(robustness=None)
    assert inflation_events(result) == []
    for event in topology_events(result):
        assert event["clean_confidence"] == event["reported_confidence"]
        assert event["confidence_inflation_agent_ids"] == []
        assert event["confidence_inflation_mode"] is None
        assert event["confidence_inflation_value"] is None


def test_missing_confidence_inflation_block_is_noop():
    robust = {"enabled": True, "type": "confidence_inflation"}
    assert _confidence_inflation_config(robust, N_AGENTS) is None

    result = run({"enabled": True, "type": "confidence_inflation"})
    assert inflation_events(result) == []


def test_null_confidence_inflation_block_is_noop():
    robust = {"enabled": True, "type": "all", "confidence_inflation": None}
    assert _confidence_inflation_config(robust, N_AGENTS) is None


def test_enabled_false_is_noop_even_when_type_matches():
    robust = {
        "enabled": True,
        "type": "confidence_inflation",
        "confidence_inflation": {
            "enabled": False,
            "agent_ids": [5],
            "mode": "fixed_report",
            "value": 4,
        },
    }
    assert _confidence_inflation_config(robust, N_AGENTS) is None

    result = run(robust)
    assert inflation_events(result) == []


def test_robustness_master_switch_off_is_noop():
    # _robustness_config returns {} when robustness.enabled is falsy.
    assert _confidence_inflation_config({}, N_AGENTS) is None


# -------------------------------------------------------------- behaviour ----
def test_only_configured_agents_are_modified():
    current = states()
    _apply_confidence_inflation(
        current,
        phase="initial",
        round_idx=0,
        cfg={"agent_ids": [5], "mode": "fixed_report", "value": 4},
        is_correct=lambda answer: False,
    )
    assert current[5]["confidence"] == 4
    for agent_id in (1, 2, 3, 4):
        assert current[agent_id]["confidence"] == 3


def test_fixed_report_changes_confidence_only():
    current = states()
    before = {i: dict(state) for i, state in current.items()}
    _apply_confidence_inflation(
        current,
        phase="initial",
        round_idx=0,
        cfg={"agent_ids": [5], "mode": "fixed_report", "value": 4},
        is_correct=lambda answer: False,
    )
    changed_keys = {
        key for key in current[5] if current[5][key] != before[5][key]
    }
    assert changed_keys == {"confidence"}


def test_fixed_report_preserves_answer_reasoning_and_critique():
    current = states()
    _apply_confidence_inflation(
        current,
        phase="initial",
        round_idx=0,
        cfg={"agent_ids": [5], "mode": "fixed_report", "value": 4},
        is_correct=lambda answer: False,
    )
    assert current[5]["answer"] == "42"
    assert current[5]["reasoning"] == "reasoning for agent 5"
    assert current[5]["critique_response"] == {"1": {"decision": "REJECT", "reason": "no"}}


def test_fixed_report_is_a_replacement_not_an_increase():
    """Named 'inflation', but the value replaces the report in both directions."""
    current = states(confidence=5)
    _apply_confidence_inflation(
        current,
        phase="initial",
        round_idx=0,
        cfg={"agent_ids": [1], "mode": "fixed_report", "value": 4},
        is_correct=lambda answer: False,
    )
    assert current[1]["confidence"] == 4


def test_fixed_report_is_gold_blind():
    """Same outcome whether the agent is right or wrong."""
    outcomes = []
    for correct in (True, False):
        current = states()
        _apply_confidence_inflation(
            current,
            phase="initial",
            round_idx=0,
            cfg={"agent_ids": [5], "mode": "fixed_report", "value": 4},
            is_correct=lambda answer, c=correct: c,
        )
        outcomes.append(current[5]["confidence"])
    assert outcomes == [4, 4]


def test_targeted_wrong_changes_only_incorrect_configured_agents():
    current = states()
    current[2]["answer"] = "wrong"
    current[5]["answer"] = "wrong"
    _apply_confidence_inflation(
        current,
        phase="initial",
        round_idx=0,
        cfg={"agent_ids": [2, 5], "mode": "targeted_wrong", "value": 5},
        is_correct=lambda answer: answer != "wrong",
    )
    assert current[2]["confidence"] == 5
    assert current[5]["confidence"] == 5
    # Unlisted agents untouched even though they are "correct".
    assert current[1]["confidence"] == 3


def test_targeted_wrong_preserves_genuine_report_for_correct_agents():
    current = states()
    current[5]["answer"] = "wrong"
    events = _apply_confidence_inflation(
        current,
        phase="initial",
        round_idx=0,
        cfg={"agent_ids": [1, 5], "mode": "targeted_wrong", "value": 5},
        is_correct=lambda answer: answer != "wrong",
    )
    assert current[1]["confidence"] == 3  # correct -> genuine report kept
    assert current[1]["clean_confidence"] == 3
    assert current[5]["confidence"] == 5  # wrong -> overclaims

    # A configured-but-unchanged agent is still reported, so the log shows the
    # attack ran and declined to act.
    changes = {c["agent_id"]: c for c in events[0]["changes"]}
    assert set(changes) == {1, 5}
    assert changes[1]["changed"] is False
    assert changes[5]["changed"] is True


# ------------------------------------------------------------- provenance ----
def test_clean_confidence_preserved_for_every_agent():
    result = run(
        {
            "enabled": True,
            "type": "confidence_inflation",
            "confidence_inflation": {
                "enabled": True,
                "agent_ids": [5],
                "mode": "fixed_report",
                "value": 4,
            },
        }
    )
    expected = {str(i) for i in range(1, N_AGENTS + 1)}
    for event in topology_events(result):
        assert set(event["clean_confidence"]) == expected
        assert set(event["reported_confidence"]) == expected
        # Genuine reports survive the attack.
        assert event["clean_confidence"]["5"] == 2.0
        assert event["reported_confidence"]["5"] == 4.0


def test_unattacked_agents_have_equal_clean_and_reported_confidence():
    result = run(
        {
            "enabled": True,
            "type": "confidence_inflation",
            "confidence_inflation": {
                "enabled": True,
                "agent_ids": [5],
                "mode": "fixed_report",
                "value": 4,
            },
        }
    )
    for event in topology_events(result):
        for agent_id in ("1", "2", "3", "4"):
            assert event["clean_confidence"][agent_id] == event["reported_confidence"][agent_id]


def test_routing_events_carry_complete_confidence_maps():
    result = run()
    expected = {str(i) for i in range(1, N_AGENTS + 1)}
    events = topology_events(result)
    assert events, "expected at least one routing decision"
    for event in events:
        assert set(event["clean_confidence"]) == expected
        assert set(event["reported_confidence"]) == expected


def test_incomplete_confidence_map_raises():
    from runner.experiment import _confidence_log_maps

    partial = states()
    del partial[3]["clean_confidence"]
    with pytest.raises(ValueError, match="Incomplete clean_confidence routing log"):
        _confidence_log_maps(partial, N_AGENTS)

    partial = states()
    del partial[3]["confidence"]
    with pytest.raises(ValueError, match="Incomplete reported_confidence routing log"):
        _confidence_log_maps(partial, N_AGENTS)


# ------------------------------------------------------------- validation ----
def base_robustness(**overrides):
    block = {"enabled": True, "agent_ids": [5], "mode": "fixed_report", "value": 4}
    block.update(overrides)
    return {"enabled": True, "type": "confidence_inflation", "confidence_inflation": block}


@pytest.mark.parametrize("bad_ids", [[0], [6], [1, 99], [-1]])
def test_out_of_range_agent_ids_raise(bad_ids):
    with pytest.raises(ValueError, match="outside the valid range"):
        _confidence_inflation_config(base_robustness(agent_ids=bad_ids), N_AGENTS)


def test_out_of_range_agent_ids_are_not_clamped():
    """Regression guard: the neighbouring adversary path clamps; this must not."""
    with pytest.raises(ValueError):
        _confidence_inflation_config(base_robustness(agent_ids=[99]), N_AGENTS)


def test_empty_agent_ids_raises():
    with pytest.raises(ValueError, match="empty"):
        _confidence_inflation_config(base_robustness(agent_ids=[]), N_AGENTS)


def test_missing_agent_ids_raises():
    block = base_robustness()
    del block["confidence_inflation"]["agent_ids"]
    with pytest.raises(ValueError, match="agent_ids is required"):
        _confidence_inflation_config(block, N_AGENTS)


def test_non_integer_agent_ids_raise():
    with pytest.raises(ValueError, match="must be integers"):
        _confidence_inflation_config(base_robustness(agent_ids=["5"]), N_AGENTS)


@pytest.mark.parametrize("bad_mode", ["inflate", "noise", "miscalibrate", ""])
def test_unknown_mode_raises(bad_mode):
    with pytest.raises(ValueError, match="Unknown robustness.confidence_inflation.mode"):
        _confidence_inflation_config(base_robustness(mode=bad_mode), N_AGENTS)


def test_missing_mode_raises():
    block = base_robustness()
    del block["confidence_inflation"]["mode"]
    with pytest.raises(ValueError, match="mode is required"):
        _confidence_inflation_config(block, N_AGENTS)


@pytest.mark.parametrize("bad_value", [0, 6, -1, 42])
def test_out_of_range_value_raises(bad_value):
    with pytest.raises(ValueError, match="outside the confidence rubric"):
        _confidence_inflation_config(base_robustness(value=bad_value), N_AGENTS)


def test_non_integer_value_raises():
    with pytest.raises(ValueError, match="must be an integer"):
        _confidence_inflation_config(base_robustness(value="4"), N_AGENTS)


def test_missing_value_raises():
    block = base_robustness()
    del block["confidence_inflation"]["value"]
    with pytest.raises(ValueError, match="value is required"):
        _confidence_inflation_config(block, N_AGENTS)


def test_both_confidence_components_enabled_raises():
    robust = base_robustness()
    robust["confidence_perturbation"] = {"enabled": True, "strategy": "miscalibrate"}
    with pytest.raises(ValueError, match="Enable exactly one"):
        _confidence_inflation_config(robust, N_AGENTS)


def test_both_components_via_type_all_raises():
    """`type: all` activates the existing perturbation component too."""
    robust = {
        "enabled": True,
        "type": "all",
        "confidence_perturbation": {"strategy": "miscalibrate"},
        "confidence_inflation": {"agent_ids": [5], "mode": "fixed_report", "value": 4},
    }
    with pytest.raises(ValueError, match="Enable exactly one"):
        _confidence_inflation_config(robust, N_AGENTS)


def test_perturbation_alone_is_unaffected():
    """The existing component still resolves when inflation is absent."""
    robust = {
        "enabled": True,
        "type": "confidence_perturbation",
        "confidence_perturbation": {"strategy": "miscalibrate"},
    }
    assert _confidence_inflation_config(robust, N_AGENTS) is None


def test_non_mapping_block_raises():
    robust = {"enabled": True, "type": "confidence_inflation", "confidence_inflation": 4}
    with pytest.raises(ValueError, match="must be a mapping"):
        _confidence_inflation_config(robust, N_AGENTS)


def test_top_level_keys_do_not_leak_into_the_component():
    """Sibling-leak guard: only the nested block configures this component."""
    robust = {
        "enabled": True,
        "type": "confidence_inflation",
        # These would leak in under _robustness_component's merge semantics.
        "mode": "targeted_wrong",
        "value": 1,
        "agent_ids": [1, 2, 3],
        "confidence_inflation": {"agent_ids": [5], "mode": "fixed_report", "value": 4},
    }
    resolved = _confidence_inflation_config(robust, N_AGENTS)
    assert resolved == {"agent_ids": [5], "mode": "fixed_report", "value": 4}


def test_duplicate_agent_ids_are_deduped_in_order():
    resolved = _confidence_inflation_config(
        base_robustness(agent_ids=[5, 2, 5]), N_AGENTS
    )
    assert resolved["agent_ids"] == [5, 2]
