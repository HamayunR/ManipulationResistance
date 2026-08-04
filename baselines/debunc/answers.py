"""DebUnc's answer extraction.

The reference implementation pulls the answer out with one regular expression
per benchmark (``debate/*/eval_*.py``). Those rules are reproduced here rather
than delegating to :meth:`data.tasks.Task.parse_answer`, because extraction is
part of the method being reproduced: DebUnc prompts for an ``Answer:`` line and
scores whatever that line says, and a more forgiving parser would credit the
method with answers its own evaluation would have thrown away.

*Scoring* still goes through the harness's ``Task``, so "correct" means the same
thing for DebUnc as for every other method in the repository.

One quirk is preserved on purpose: the reference uses ``re.search``, so it takes
the *first* ``Answer:`` in the response even though the prompt asks for it on
the last line. A model that reasons out loud about "Answer: B" before settling
on C is scored on B.
"""

from __future__ import annotations

import re
from typing import Callable, Dict

from data.normalize import LETTERS
from data.tasks import Example, Task

_NUMERIC = re.compile(r"(?i)Answer\s*:\s*\$?(\d+\.?\d*)")
_FREE_FORM = re.compile(r"(?i)Answer\s*:\s*(.+)")


def parse_answer(text: str, *, task: Task, example: Example) -> str:
    """Extract the answer token from one agent response; ``""`` if there is none."""
    parser = _PARSERS.get(task.name)
    if parser is None:
        raise ValueError(
            f"no DebUnc answer parser for dataset {task.name!r}; known: {sorted(_PARSERS)}"
        )
    return parser(str(text or ""), example, task)


def _letter(text: str, example: Example, _task: Task) -> str:
    valid = LETTERS[: len(example.choices or [])] or LETTERS
    match = re.search(rf"(?i)Answer\s*:\s*([{valid}])", text)
    return match.group(1).upper() if match else ""


def _integer(text: str, _example: Example, _task: Task) -> str:
    # The reference strips thousands separators before matching, so "1,024"
    # is read as 1024 rather than truncated at the comma.
    match = _NUMERIC.search(text.replace(",", ""))
    return match.group(1) if match else ""


def _math(text: str, _example: Example, task: Task) -> str:
    # MATH-500 is outside DebUnc's benchmark set, so there is no reference rule
    # to copy. Take the same ``Answer:`` line, then hand the expression to the
    # task's normaliser, which knows about \boxed{} and LaTeX spacing.
    match = _FREE_FORM.search(text)
    if not match:
        return ""
    return task.parse_answer(match.group(1).strip()) or match.group(1).strip()


_PARSERS: Dict[str, Callable[[str, Example, Task], str]] = {
    "mmlu_pro": _letter,
    "gpqa": _letter,
    "truthful_qa": _letter,
    "gsm8k": _integer,
    "math_500": _math,
}


__all__ = ["parse_answer"]
