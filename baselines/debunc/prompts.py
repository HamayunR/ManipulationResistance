"""DebUnc's prompts, reproduced from the paper's Appendix C and its repository.

These are deliberately *not* the templates in the top-level :mod:`prompts`
module. PEAR asks for JSON with a self-reported 1-5 confidence; DebUnc asks for
prose ending in an ``Answer:`` line and never asks the model about its own
confidence, because in DebUnc confidence comes from token probabilities. Using
PEAR's prompts here would measure a different system.

Each benchmark family gets its own wording, as in the reference implementation:

* multiple choice (``debate/mmlu/common.py``),
* free-form numeric (``debate/gsm8k/common.py``), and
* TruthfulQA, which phrases the debate turn as a double-check rather than an
  update (``debate/truthfulqa/common.py``).

Only the Confidence-in-Prompt wording is reproduced -- the reference's
``START_PREFIX_PROMPT`` and ``END_PREFIX_NO_CONF`` -- because that is the only
communication method this port implements.

Two adaptations to this harness, both documented in ``README.md``: the letter
set in the answer instruction is derived from the question's own option count
rather than hard-coded to ``ABCD``, and MATH-500 (which DebUnc does not
evaluate) reuses the free-form shape with a mathematical answer instruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from data.normalize import LETTERS, format_options
from data.tasks import Example, Task

#: One critiqued solution with its confidence level. The leading space after
#: the newlines is in the reference implementation; it is kept so the prompt is
#: byte-identical to the one the paper's numbers came from.
SOLUTION_WITH_CONFIDENCE = (
    "\n\n One agent solution (confidence level is {confidence}): ```{response}```"
)


@dataclass(frozen=True)
class DebUncPrompts:
    """Fully resolved prompt pieces for one example.

    Attributes
    ----------
    initial:
        Round-1 prompt, answered independently by every agent.
    start:
        Opening line of a debate turn, naming the 1-10 confidence scale.
    end:
        Closing instruction of a debate turn. It adds "Do not mention your
        confidence", which the paper uses whenever confidence appears in the
        prompt, so the model reports an answer rather than echoing a number.
    """

    initial: str
    start: str
    end: str


def answer_letters(example: Example) -> str:
    """Letters valid for ``example``, e.g. ``"ABCD"`` or ``"ABCDEFGHIJ"``."""
    return LETTERS[: len(example.choices or [])]


def prompts_for(task: Task, example: Example) -> DebUncPrompts:
    """Select and resolve the prompt family for ``task``."""
    builder = _FAMILIES.get(task.name)
    if builder is None:
        raise ValueError(
            f"no DebUnc prompt family for dataset {task.name!r}; "
            f"known: {sorted(_FAMILIES)}"
        )
    return builder(example)


def build_debate_message(
    prompts: DebUncPrompts,
    responses: Sequence[str],
    confidences: Sequence[int],
) -> str:
    """Assemble one debate turn quoting ``responses`` from the other agents.

    Each solution is labelled with its confidence level, and the closing
    instruction tells the model not to restate a confidence of its own.
    """
    if len(confidences) != len(responses):
        raise ValueError(
            f"{len(confidences)} confidences for {len(responses)} responses"
        )

    parts: List[str] = [prompts.start]
    for response, confidence in zip(responses, confidences):
        parts.append(
            SOLUTION_WITH_CONFIDENCE.format(response=response, confidence=confidence)
        )
    parts.append(prompts.end)
    return "".join(parts)


# ------------------------------------------------------------- families --
def _multiple_choice(example: Example) -> DebUncPrompts:
    """MMLU-style letter answers (``debate/mmlu/common.py``)."""
    last_line = (
        "The last line of your response should be of the following format: "
        f"'Answer: $LETTER' (without quotes) where LETTER is one of {answer_letters(example)}."
    )
    body = f"{example.question}\n{format_options(example.choices or [])}"
    return DebUncPrompts(
        initial=(
            f"Answer the following multiple choice question. {last_line} "
            f"Think step by step before answering.\n\n{body}"
        ).strip(),
        start=(
            "These are solutions and confidence values from 1 to 10 (higher means "
            "more confident) to the problem from other agents: "
        ),
        end=(
            "\n\nBased off the opinion of other agents, can you give an updated "
            "response? Do not mention your confidence. Think step by step before "
            f"answering. {last_line}"
        ),
    )


def _truthful_qa(example: Example) -> DebUncPrompts:
    """TruthfulQA (``debate/truthfulqa/common.py``): the debate turn is a double-check."""
    last_line = (
        "The last line of your response should be of the following format: "
        f"'Answer: $LETTER' (without quotes) where LETTER is one of {answer_letters(example)}."
    )
    options = "".join(
        f"{letter}. {choice}\n"
        for letter, choice in zip(LETTERS, example.choices or [])
    )
    return DebUncPrompts(
        initial=(
            "Answer the following multiple choice question:\n\n"
            f"{example.question}\n {options}"
            f"\n\nThink step by step before answering. {last_line}\n\n"
        ),
        start=(
            "These are the selections and confidence values from 1 to 10 (higher "
            "means more confident) from other agents: "
        ),
        end=(
            "\n\nCan you double check that your response is correct? Do not "
            f"mention your confidence. {last_line}"
        ),
    )


def _free_form(example: Example, *, subject: str, last_line: str) -> DebUncPrompts:
    """GSM8k-style free-form answers (``debate/gsm8k/common.py``).

    The debate turn restates the problem, because by round three the original
    question is far enough back in the agent's history to be worth repeating.
    """
    return DebUncPrompts(
        initial=(
            f"Answer the following {subject}. {last_line} "
            f"Think step by step before answering.\n\n{example.question}"
        ).strip(),
        start=(
            "These are solutions and confidence values from 1 to 10 (higher means "
            "more confident) to the problem from other agents: "
        ),
        end=(
            "\n\nBased off the opinion of other agents, can you provide an updated "
            f"response? The original problem is:\n\n{example.question}\n\n"
            f"Do not mention your confidence. {last_line}"
        ),
    )


def _gsm8k(example: Example) -> DebUncPrompts:
    return _free_form(
        example,
        subject="math problem",
        last_line=(
            "The last line of your response should be of the following format: "
            "'Answer: $INTEGER' (without quotes) where INTEGER is the integer answer."
        ),
    )


def _math_500(example: Example) -> DebUncPrompts:
    # DebUnc does not evaluate MATH-500. The free-form shape carries over; only
    # the answer instruction changes, so that answers are extractable the same way.
    return _free_form(
        example,
        subject="math problem",
        last_line=(
            "The last line of your response should be of the following format: "
            "'Answer: $ANSWER' (without quotes) where ANSWER is the final "
            "mathematical expression, in the form the problem asks for."
        ),
    )


_FAMILIES = {
    "mmlu_pro": _multiple_choice,
    "gpqa": _multiple_choice,
    "truthful_qa": _truthful_qa,
    "gsm8k": _gsm8k,
    "math_500": _math_500,
}


__all__ = [
    "SOLUTION_WITH_CONFIDENCE",
    "DebUncPrompts",
    "answer_letters",
    "build_debate_message",
    "prompts_for",
]
