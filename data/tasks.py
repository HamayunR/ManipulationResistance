"""Dataset adapters: formatting, answer parsing, scoring.

A *task* wraps one benchmark split and exposes the small interface that
:mod:`runner.experiment` and :mod:`nodes.scorer` are written against. The
interface is exactly what the call sites use, no more:

``Task``
    ``name``, ``split``, ``examples``, ``len(task)``,
    ``format_question(example)``, ``parse_answer(text)``,
    ``score(prediction, example)``.

``Example``
    ``id``, ``question``, ``answer``, ``choices``.

Two conventions matter and are easy to get wrong:

* :meth:`Task.score` takes ``(prediction, example)`` -- the *whole* example, not
  the gold string (see ``nodes/scorer.py``).
* ``choices`` carries letter semantics. When it is non-empty the runner treats
  answers as option letters, indexing ``string.ascii_uppercase`` positionally
  (``runner/experiment.py`` ``_random_baseline_decision`` /
  ``_default_wrong_answer``). Free-form tasks must leave it ``None``.

:func:`parse_answer` is also handed to the runner as a *bare callable*, so it
must not rely on anything beyond its argument, and it must tolerate ``""``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Type


@dataclass
class Example:
    """One benchmark item, normalised across datasets.

    Attributes
    ----------
    id:
        Stable identifier; used as the per-example key in traces and
        transcripts.
    question:
        Raw question text. Written verbatim to the transcript; the *prompted*
        form is produced separately by :meth:`Task.format_question`.
    answer:
        Gold answer as a string. For multiple-choice tasks this is the option
        letter.
    choices:
        Option strings for multiple-choice tasks, or ``None`` for free-form
        answers. Non-empty enables the runner's letter-based paths.
    """

    id: str
    question: str
    answer: str
    choices: Optional[List[str]] = None


class Task:
    """Base class for benchmark adapters.

    Subclasses supply ``name``, ``default_split``, and a way to load examples.
    The default :meth:`parse_answer` and :meth:`score` implement exact match on
    stripped strings; override them per benchmark as needed.
    """

    #: Registry key, also reported in logs and run metadata.
    name: str = "task"
    #: Split used when the config leaves ``dataset.split`` unset.
    default_split: str = "test"

    def __init__(
        self,
        *,
        split: Optional[str] = None,
        num_examples: int = 0,
        data_dir: str = "data",
    ) -> None:
        self.split = split or self.default_split
        self.data_dir = data_dir
        examples = list(self.load_examples())
        # ``num_examples`` is a cap for compute-light runs; 0 means "use all".
        if num_examples and num_examples > 0:
            examples = examples[:num_examples]
        self.examples: List[Example] = examples

    # Subclass hook
    def load_examples(self) -> Sequence[Example]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.examples)

    # Prompting
    def format_question(self, example: Example) -> str:
        """Render ``example`` as the question body handed to the agents.

        Real benchmarks should wrap this in the matching template from
        :mod:`prompts` (``GSM8K_TEMPLATE`` and friends) so the answer-format
        instructions travel with the question.
        """
        text = example.question
        if example.choices:
            options = "\n".join(
                f"{letter}. {choice}"
                for letter, choice in zip(_LETTERS, example.choices)
            )
            text = f"{text}\n\nOptions:\n{options}"
        return text

    # Answer handling
    def parse_answer(self, text: str) -> str:
        """Extract a bare answer token from free-form model output.

        Returns ``""`` when nothing can be extracted; every call site in the
        runner supplies its own fallback in that case.
        """
        return str(text or "").strip()

    def score(self, prediction: str, example: Example) -> bool:
        """Return whether ``prediction`` matches ``example``'s gold answer."""
        return str(prediction or "").strip() == str(example.answer).strip()


class DummyTask(Task):
    """Three hardcoded single-digit arithmetic items, for pipeline smoke tests.

    Free-form numeric answers, so ``choices`` stays ``None``. Integer golds also
    keep ``_default_wrong_answer``'s numeric ``+1`` branch working, which the
    robustness adversary depends on.
    """

    name = "dummy"
    default_split = "test"

    _ITEMS = [
        ("dummy-1", "What is 3 + 4?", "7"),
        ("dummy-2", "What is 9 - 5?", "4"),
        ("dummy-3", "What is 2 * 3?", "6"),
    ]

    def load_examples(self) -> Sequence[Example]:
        return [
            Example(id=item_id, question=question, answer=answer, choices=None)
            for item_id, question, answer in self._ITEMS
        ]

    def format_question(self, example: Example) -> str:
        return (
            "Task type: single-digit arithmetic.\n"
            "Answer format: provide only the final integer in the JSON answer "
            "field. No units, commas, or explanatory text.\n\n"
            f"Problem:\n{example.question}"
        )

    def parse_answer(self, text: str) -> str:
        """Return the last integer in ``text``, or ``""`` if there is none.

        Taking the *last* match is the usual convention for chain-of-thought
        output, where the final line restates the answer.
        """
        matches = re.findall(r"-?\d+", str(text or "").replace(",", ""))
        return matches[-1] if matches else ""

    def score(self, prediction: str, example: Example) -> bool:
        gold = str(example.answer).strip()
        pred = str(prediction or "").strip()
        if not pred:
            return False
        try:
            return int(pred.replace(",", "")) == int(gold)
        except ValueError:
            return pred == gold


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: Registry of available datasets, keyed by ``dataset.name`` in the config.
TASK_REGISTRY: Dict[str, Type[Task]] = {
    DummyTask.name: DummyTask,
}


def load_task(
    name: str,
    split: Optional[str] = None,
    num_examples: int = 0,
    data_dir: str = "data",
) -> Task:
    """Instantiate the task registered under ``name``.

    Parameters
    ----------
    name:
        Registry key, from ``dataset.name``.
    split:
        Split to load. ``None`` lets the task apply its own default, which is
        why the runner passes the config value through unchanged.
    num_examples:
        Cap on the number of examples; ``0`` means use all.
    data_dir:
        Local dataset root. Ignored by datasets that carry their items inline.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    try:
        task_cls = TASK_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset: {name!r}. Registered: {sorted(TASK_REGISTRY)}"
        ) from None
    return task_cls(split=split, num_examples=num_examples, data_dir=data_dir)


def available_tasks() -> List[str]:
    """Sorted list of registered dataset names."""
    return sorted(TASK_REGISTRY)


__all__ = [
    "Example",
    "Task",
    "DummyTask",
    "TASK_REGISTRY",
    "load_task",
    "available_tasks",
]
