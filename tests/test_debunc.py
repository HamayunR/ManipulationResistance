"""Tests for the DebUnc baseline (``baselines/debunc``).

The point of a reimplementation is that it does what the paper did, so the
tests here pin it against the reference rather than against itself:

* the uncertainty-to-confidence conversion is checked against a transcription
  of the authors' ``unc_to_confidence``; and
* the prompt strings are checked against the constants in their
  ``debate/*/common.py``.

Everything runs on a stub backend. No model weights, no network.
"""

from __future__ import annotations

import math
import re

import pytest

from baselines.debunc.answers import parse_answer
from baselines.debunc.config import DebUncConfig
from baselines.debunc.confidence import confidences_from_uncertainties
from baselines.debunc.prompts import build_debate_message, prompts_for
from baselines.debunc.runner import run_debunc_example
from data.tasks import Example, GSM8KTask, MMLUProTask, TruthfulQATask
from models.whitebox import WhiteboxGeneration, WhiteboxLLM


# ------------------------------------------------------------- reference --
def reference_unc_to_confidence(uncertainties):
    """Transcription of ``debate/gen_utils.py::unc_to_confidence``.

    Kept verbatim (including the hard-coded 14, which is ``5n - 1`` for the
    paper's three agents) so a change to our implementation has to be justified
    against the thing it reproduces.
    """
    import numpy as np

    confidences = 1 / np.array(uncertainties, dtype=float)
    confidences = confidences * 14 / np.sum(confidences) + 1 / len(uncertainties)
    confidences = np.clip(confidences, 1, 10)
    return np.round(confidences).astype(int).tolist()


# ------------------------------------------------------------ confidence --
@pytest.mark.parametrize(
    "uncertainties",
    [
        [0.295, 0.146, 0.314],
        [1.0, 1.0, 1.0],
        [0.01, 5.0, 0.5],
        [0.9, 0.9, 0.91],
        [2.5, 0.001, 1.25],
    ],
)
def test_prompt_confidence_matches_reference_for_three_agents(uncertainties):
    assert list(confidences_from_uncertainties(uncertainties)) == (
        reference_unc_to_confidence(uncertainties)
    )


def test_prompt_confidence_averages_five_before_clamping():
    # Section 3.2: the conversion is defined so the mean confidence is 5,
    # whatever scale the metric happens to be on.
    uncertainties = [0.2, 0.4, 0.6, 0.8]
    inverse = [1 / u for u in uncertainties]
    n = len(inverse)
    scaled = [c / sum(inverse) * (5 * n - 1) + 1 / n for c in inverse]
    assert sum(scaled) == pytest.approx(5 * n)
    assert list(confidences_from_uncertainties(uncertainties)) == [
        int(round(min(max(s, 1), 10))) for s in scaled
    ]


def test_lower_uncertainty_means_higher_confidence():
    levels = confidences_from_uncertainties([0.1, 0.5, 2.0])
    assert levels[0] > levels[1] > levels[2]


def test_zero_uncertainty_is_floored_not_infinite():
    # Complete certainty is a legitimate outcome for a short answer; inverting
    # it must not put an infinity into the prompt.
    levels = confidences_from_uncertainties([0.0, 1.0])
    assert all(1 <= level <= 10 for level in levels)
    assert levels[0] > levels[1]


# --------------------------------------------------------------- prompts --
def _task(cls):
    task = cls.__new__(cls)
    task.split = "test"
    task.data_dir = "data"
    task.examples = []
    return task


def _example(choices=None):
    return Example(id="q1", question="What is 2+2?", answer="B", choices=choices)


def test_multiple_choice_prompts_match_the_reference_strings():
    prompts = prompts_for(_task(MMLUProTask), _example(["3", "4", "5", "6"]))
    assert prompts.start == (
        "These are solutions and confidence values from 1 to 10 (higher means "
        "more confident) to the problem from other agents: "
    )
    assert prompts.end == (
        "\n\nBased off the opinion of other agents, can you give an updated "
        "response? Do not mention your confidence. Think step by step before "
        "answering. The last line of your response should be of the following "
        "format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD."
    )
    assert prompts.initial.startswith("Answer the following multiple choice question.")
    assert prompts.initial.endswith("What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6")


def test_letter_instruction_follows_the_option_count():
    prompts = prompts_for(_task(MMLUProTask), _example(list("abcdefghij")))
    assert "LETTER is one of ABCDEFGHIJ." in prompts.initial


def test_gsm8k_debate_turn_restates_the_problem():
    prompts = prompts_for(
        _task(GSM8KTask), Example(id="g1", question="How many?", answer="4")
    )
    assert prompts.end == (
        "\n\nBased off the opinion of other agents, can you provide an updated "
        "response? The original problem is:\n\nHow many?\n\nDo not mention your "
        "confidence. The last line of your response should be of the following "
        "format: 'Answer: $INTEGER' (without quotes) where INTEGER is the "
        "integer answer."
    )


def test_truthfulqa_debate_turn_is_a_double_check():
    prompts = prompts_for(_task(TruthfulQATask), _example(["a", "b", "c", "d"]))
    assert prompts.start == (
        "These are the selections and confidence values from 1 to 10 (higher "
        "means more confident) from other agents: "
    )
    assert prompts.end.startswith(
        "\n\nCan you double check that your response is correct? Do not mention "
        "your confidence."
    )


def test_debate_message_labels_each_solution_with_its_confidence():
    prompts = prompts_for(_task(MMLUProTask), _example(["3", "4", "5", "6"]))
    message = build_debate_message(prompts, ["alpha", "beta"], [3, 9])
    assert "One agent solution (confidence level is 3): ```alpha```" in message
    assert "One agent solution (confidence level is 9): ```beta```" in message
    assert message.startswith("These are solutions and confidence values from 1 to 10")
    # The paper suppresses a self-reported confidence whenever the prompt
    # already carries a measured one.
    assert "Do not mention your confidence." in message


def test_confidence_count_must_match_response_count():
    prompts = prompts_for(_task(MMLUProTask), _example(["3", "4", "5", "6"]))
    with pytest.raises(ValueError):
        build_debate_message(prompts, ["a", "b"], [1])


# --------------------------------------------------------------- answers --
def test_letter_answers_are_read_from_the_answer_line():
    task, example = _task(MMLUProTask), _example(["3", "4", "5", "6"])
    assert parse_answer("Reasoning...\nAnswer: B", task=task, example=example) == "B"
    assert parse_answer("answer:  c", task=task, example=example) == "C"
    assert parse_answer("no answer here", task=task, example=example) == ""
    # Beyond this question's options is not an answer to this question.
    assert parse_answer("Answer: J", task=task, example=example) == ""


def test_letter_parser_takes_the_first_match_like_the_reference():
    # The reference uses re.search, so an answer stated mid-reasoning wins over
    # the one on the last line. Preserved deliberately; see answers.py.
    task, example = _task(MMLUProTask), _example(["3", "4", "5", "6"])
    assert parse_answer("Answer: A ... actually\nAnswer: C", task=task, example=example) == "A"


def test_numeric_answers_drop_thousands_separators():
    task = _task(GSM8KTask)
    example = Example(id="g1", question="How many?", answer="1024")
    assert parse_answer("Answer: 1,024", task=task, example=example) == "1024"
    assert parse_answer("Answer: $18", task=task, example=example) == "18"
    assert parse_answer("I do not know", task=task, example=example) == ""


# ---------------------------------------------------------------- config --
@pytest.mark.parametrize("mode", ["debunc", "debunc_prompt"])
def test_both_mode_spellings_are_accepted(mode):
    config = DebUncConfig.from_debate_cfg({"mode": mode})
    assert config.temperature == 1.0
    assert config.min_new_tokens == 2


def test_a_non_debunc_mode_is_refused():
    with pytest.raises(ValueError, match="not a DebUnc mode"):
        DebUncConfig.from_debate_cfg({"mode": "pear_full"})


@pytest.mark.parametrize(
    "block",
    [
        # The variants this port does not implement must not look honoured.
        {"communication": "attention_all"},
        {"uncertainty": "token_sar"},
        {"oracle_prompt_confidence": [1, 10]},
        {"typo_here": 1},
    ],
)
def test_settings_this_port_does_not_implement_are_rejected(block):
    with pytest.raises(ValueError, match="unknown debate.debunc key"):
        DebUncConfig.from_debate_cfg({"mode": "debunc", "debunc": block})


def test_decoding_parameters_are_read_from_the_block():
    config = DebUncConfig.from_debate_cfg(
        {"mode": "debunc", "debunc": {"temperature": 0.7, "top_k": 20}}
    )
    assert (config.temperature, config.top_k) == (0.7, 20)


# ----------------------------------------------------------- debate loop --
class StubWhiteboxLLM(WhiteboxLLM):
    """A whitebox backend that answers from a script and reports fixed scores.

    ``answers`` is indexed ``[round][agent]``; ``entropies`` likewise, which is
    what makes the confidence ordering checkable.
    """

    name = "stub"

    def __init__(self, answers, entropies):
        self.answers = answers
        self.entropies = entropies
        self.calls = []

    def generate(self, prompt, **kwargs):  # pragma: no cover - unused here
        raise NotImplementedError

    @property
    def tokenizer(self):  # pragma: no cover - unused here
        return None

    def chat_generate(self, messages, *, max_tokens, **kwargs):
        conversation = [dict(m) for m in messages]
        round_idx = sum(1 for m in conversation if m["role"] == "assistant")
        agent_index = self._agent_index(conversation, round_idx)
        self.calls.append(
            {
                "round": round_idx,
                "agent": agent_index,
                "prompt": conversation[-1]["content"],
            }
        )
        text = self.answers[round_idx][agent_index]
        entropy = self.entropies[round_idx][agent_index]
        return WhiteboxGeneration(
            text=text,
            prompt_tokens=sum(len(m["content"]) for m in conversation),
            completion_tokens=len(text),
            token_ids=list(range(len(text))),
            token_logprobs=[-0.5] * len(text),
            token_entropies=[entropy] * max(1, len(text)),
        )

    def _agent_index(self, conversation, round_idx):
        """Recover which agent this is from the answer it gave last round."""
        assistants = [m["content"] for m in conversation if m["role"] == "assistant"]
        if not assistants:
            return sum(1 for c in self.calls if c["round"] == 0)
        return self.answers[len(assistants) - 1].index(assistants[-1])


@pytest.fixture
def mcq_task():
    return _task(MMLUProTask)


@pytest.fixture
def mcq_example():
    return Example(id="q1", question="What is 2+2?", answer="B", choices=["3", "4", "5", "6"])


def _answers(rounds):
    return [[f"Reasoning.\nAnswer: {letter}" for letter in row] for row in rounds]


def test_debate_runs_the_configured_number_of_rounds(mcq_task, mcq_example):
    llm = StubWhiteboxLLM(
        answers=_answers([["A", "B", "C"], ["B", "B", "C"], ["B", "B", "B"]]),
        entropies=[[0.1, 0.2, 0.3]] * 3,
    )
    row = run_debunc_example(
        mcq_task,
        mcq_example,
        debate_cfg={"mode": "debunc", "n_agents": 3, "rounds": 2},
        llm=llm,
        seed=0,
        perm_seed=10,
    )
    # One independent answer plus two debate rounds, for three agents.
    assert len(llm.calls) == 9
    assert len(row["answer_history"]) == 3
    assert row["decision"] == "B"
    assert row["correct"] is True
    assert row["budget"]["calls"] == 9


def test_each_agent_sees_every_other_agent_and_not_itself(mcq_task, mcq_example):
    llm = StubWhiteboxLLM(
        answers=_answers([["A", "B", "C"], ["A", "B", "C"]]),
        entropies=[[0.1, 0.2, 0.3]] * 2,
    )
    run_debunc_example(
        mcq_task,
        mcq_example,
        debate_cfg={"mode": "debunc", "n_agents": 3, "rounds": 1},
        llm=llm,
        seed=0,
        perm_seed=10,
    )
    round_two = [c for c in llm.calls if c["round"] == 1]
    assert len(round_two) == 3
    for call in round_two:
        quoted = re.findall(r"```(.*?)```", call["prompt"], flags=re.S)
        assert len(quoted) == 2, "a debate turn quotes every other agent exactly once"
        own = llm.answers[0][call["agent"]]
        assert own not in quoted, "an agent is not shown its own solution as an 'other agent'"


def test_each_quoted_solution_carries_its_own_agents_confidence(mcq_task, mcq_example):
    llm = StubWhiteboxLLM(
        answers=_answers([["A", "B", "C"], ["A", "B", "C"]]),
        entropies=[[0.1, 0.5, 1.0], [0.1, 0.5, 1.0]],
    )
    row = run_debunc_example(
        mcq_task,
        mcq_example,
        debate_cfg={"mode": "debunc", "n_agents": 3, "rounds": 1},
        llm=llm,
        seed=0,
        perm_seed=10,
    )
    expected = confidences_from_uncertainties([0.1, 0.5, 1.0])
    turn = next(c for c in llm.calls if c["round"] == 1 and c["agent"] == 0)
    assert f"confidence level is {expected[1]}" in turn["prompt"]
    assert f"confidence level is {expected[2]}" in turn["prompt"]
    # Agent 1 is the most certain, so it must be quoted at the highest level.
    assert expected[0] > expected[1] > expected[2]
    assert row["confidence_history"][0] == {1: expected[0], 2: expected[1], 3: expected[2]}


def test_uncertainty_is_recorded_for_every_response(mcq_task, mcq_example):
    llm = StubWhiteboxLLM(
        answers=_answers([["A", "B", "C"], ["A", "B", "C"]]),
        entropies=[[0.1, 0.5, 1.0]] * 2,
    )
    row = run_debunc_example(
        mcq_task,
        mcq_example,
        debate_cfg={"mode": "debunc", "n_agents": 3, "rounds": 1},
        llm=llm,
        seed=0,
        perm_seed=10,
    )
    responses = [e for e in row["trace_events"] if e["event"] == "debunc_response"]
    assert len(responses) == 6
    assert all(math.isfinite(e["uncertainty"]) for e in responses)
    assert all(e["uncertainty_metric"] == "mean_token_entropy" for e in responses)


def test_result_row_matches_the_runner_contract(mcq_task, mcq_example):
    llm = StubWhiteboxLLM(
        answers=_answers([["A", "B", "C"], ["B", "B", "B"]]),
        entropies=[[0.2, 0.3, 0.4]] * 2,
    )
    row = run_debunc_example(
        mcq_task,
        mcq_example,
        debate_cfg={"mode": "debunc", "n_agents": 3, "rounds": 1},
        llm=llm,
        seed=3,
        perm_seed=11,
    )
    for key in (
        "example_id",
        "decision",
        "correct",
        "parse_failures",
        "n_messages",
        "budget",
        "trace_events",
        "messages",
        "final_candidates",
        "seed",
        "perm_seed",
        "answer_history",
        "confidence_history",
        "influence_history",
        "diagnostics",
    ):
        assert key in row, f"missing result key {key}"
    assert row["seed"] == 3 and row["perm_seed"] == 11
    # DebUnc has no routing, so it must not emit routing rows that the analysis
    # layer would read as a routing decision.
    assert not [e for e in row["trace_events"] if e["event"] == "topology"]
    assert row["influence_history"] == []
    assert row["diagnostics"]["w2r_rate"] == pytest.approx(1.0)


def test_unparseable_answers_are_counted_not_hidden(mcq_task, mcq_example):
    llm = StubWhiteboxLLM(
        answers=[["I have no idea.", "Answer: B", "Answer: B"]],
        entropies=[[0.2, 0.3, 0.4]],
    )
    row = run_debunc_example(
        mcq_task,
        mcq_example,
        debate_cfg={"mode": "debunc", "n_agents": 3, "rounds": 0},
        llm=llm,
        seed=0,
        perm_seed=10,
    )
    assert row["parse_failures"] == 1
    assert row["final_candidates"][1] == ""
    assert row["decision"] == "B"


def test_a_non_whitebox_backend_is_refused(mcq_task, mcq_example):
    class ApiOnly:
        pass

    with pytest.raises(TypeError, match="whitebox"):
        run_debunc_example(
            mcq_task,
            mcq_example,
            debate_cfg={"mode": "debunc", "n_agents": 3, "rounds": 1},
            llm=ApiOnly(),
            seed=0,
            perm_seed=10,
        )
