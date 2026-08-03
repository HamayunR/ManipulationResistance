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

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Type

from data.local import CorpusRecord, check_corpus, corpus_path, load_corpus
from data.normalize import (
    format_options,
    math_equal,
    numbers_equal,
    parse_choice_letter,
    parse_math_answer,
    parse_number,
)
from prompts import (
    GPQA_TEMPLATE,
    GSM8K_TEMPLATE,
    MATH_500_TEMPLATE,
    MMLU_PRO_TEMPLATE,
    TRUTHFUL_QA_TEMPLATE,
)


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


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# --------------------------------------------------------------- real benchmarks --
class LocalCorpusTask(Task):
    """Base class for benchmarks read from a frozen local corpus.

    Subclasses declare their name, default split, prompt template and answer
    type; the loading, the "you have not fetched this yet" refusal and the
    provenance handling are shared.

    The corpus is authoritative. Nothing here re-derives an option order, a
    gold label or an id at run time -- those were fixed once at fetch time and
    recorded with a checksum, so two runs a month apart score the same items in
    the same order (see :mod:`data.local`).
    """

    #: Prompt wrapper from :mod:`prompts`; ``{question}`` and, for
    #: multiple-choice tasks, ``{options}`` are substituted.
    template: str = "{question}"
    #: Human-readable hint appended to the "not fetched" error.
    fetch_hint: str = ""

    #: Checksum of the corpus this task loaded, and how it compared to the
    #: manifest. Written into the run's summary.json so every result records
    #: which bytes it was scored against.
    corpus_sha256: Optional[str] = None
    corpus_status: Optional[str] = None
    corpus_file: Optional[str] = None

    def load_examples(self) -> Sequence[Example]:
        records = load_corpus(
            name=self.name,
            split=self.split,
            data_dir=self.data_dir,
            fetch_hint=self.fetch_hint,
        )
        # Verified on every load, not in a command someone has to remember.
        # ~13 ms for the largest corpus; the failure it catches is silent and
        # unrecoverable once runs exist.
        path = corpus_path(self.data_dir, self.name, self.split)
        if path is not None:
            integrity = check_corpus(
                path, name=self.name, split=self.split, data_dir=self.data_dir
            )
            self.corpus_sha256 = integrity.sha256
            self.corpus_status = integrity.status
            self.corpus_file = str(integrity.path)
        return [self._to_example(record) for record in records]

    def _to_example(self, record: CorpusRecord) -> Example:
        return Example(
            id=record.id,
            question=record.question,
            answer=record.answer,
            choices=list(record.choices) if record.choices else None,
        )

    def format_question(self, example: Example) -> str:
        if example.choices:
            return self.template.format(
                question=example.question, options=format_options(example.choices)
            )
        return self.template.format(question=example.question)


class MultipleChoiceTask(LocalCorpusTask):
    """Letter-answer benchmarks: the gold answer is an option letter.

    ``score`` compares letters, never option *text*: the runner's adversary and
    random-baseline paths both emit letters, and accepting text here would make
    them look wrong for the wrong reason.
    """

    def parse_answer(self, text: str) -> str:
        """Extract an option letter, bounded by this benchmark's option count.

        ``parse_answer`` is handed to the runner as a bare callable, so it
        cannot see the example. It can still see the corpus: a letter beyond
        the widest question in the split is not an option in this benchmark at
        all, and admitting it would put a non-option into ``answer_history``,
        the routing state and the aggregator, where nothing checks it again.
        """
        return parse_choice_letter(text, n_options=self._max_options())

    def _max_options(self) -> int:
        """Widest option list in the loaded split; 26 before examples load."""
        cached = getattr(self, "_max_options_cache", None)
        if cached is None:
            counts = [len(e.choices or []) for e in getattr(self, "examples", [])]
            cached = max(counts) if counts else len(_LETTERS)
            self._max_options_cache = cached
        return cached

    def score(self, prediction: str, example: Example) -> bool:
        n_options = len(example.choices or []) or len(_LETTERS)
        predicted = parse_choice_letter(prediction, n_options=n_options)
        if not predicted:
            return False
        return predicted == str(example.answer).strip().upper()


class GSM8KTask(LocalCorpusTask):
    """GSM8K grade-school word problems. Free-form numeric answers."""

    name = "gsm8k"
    default_split = "test"
    template = GSM8K_TEMPLATE
    fetch_hint = "Source: openai/gsm8k (config 'main'). Splits: train, test."

    def parse_answer(self, text: str) -> str:
        return parse_number(text)

    def score(self, prediction: str, example: Example) -> bool:
        return numbers_equal(parse_number(prediction) or prediction, example.answer)


class MATH500Task(LocalCorpusTask):
    """MATH-500 competition problems. Free-form LaTeX answers."""

    name = "math_500"
    default_split = "test"
    template = MATH_500_TEMPLATE
    fetch_hint = "Source: HuggingFaceH4/MATH-500. Split: test (500 items)."

    def parse_answer(self, text: str) -> str:
        return parse_math_answer(text)

    def score(self, prediction: str, example: Example) -> bool:
        return math_equal(parse_math_answer(prediction) or prediction, example.answer)


class MMLUProTask(MultipleChoiceTask):
    """MMLU-Pro. Up to ten options per question (A-J)."""

    name = "mmlu_pro"
    default_split = "test"
    template = MMLU_PRO_TEMPLATE
    fetch_hint = "Source: TIGER-Lab/MMLU-Pro. Splits: test, validation."


class GPQATask(MultipleChoiceTask):
    """GPQA expert-level science. Four options, order frozen at fetch time.

    The upstream release stores the correct answer in its own column rather
    than among the distractors, so the option order is created during the
    fetch, from a recorded seed. Re-deriving it per run would let a refactor
    quietly change which letter is correct.
    """

    name = "gpqa"
    default_split = "diamond"
    template = GPQA_TEMPLATE
    fetch_hint = (
        "Source: Idavidrein/gpqa (gated). Accept the conditions on the dataset "
        "page and export HF_TOKEN before fetching. Subsets: diamond, main, "
        "extended. The authors ask that GPQA items are not published in "
        "plain text, so no sample of it is committed to this repository."
    )


class TruthfulQATask(MultipleChoiceTask):
    """TruthfulQA MC1: one truthful option among several plausible falsehoods.

    Option order is frozen at fetch time for the same reason as GPQA -- the
    raw MC1 targets always list the correct answer first.
    """

    name = "truthful_qa"
    default_split = "validation"
    template = TRUTHFUL_QA_TEMPLATE
    fetch_hint = (
        "Source: truthfulqa/truthful_qa (config 'multiple_choice'). "
        "Split: validation (817 items)."
    )


#: Registry of available datasets, keyed by ``dataset.name`` in the config.
TASK_REGISTRY: Dict[str, Type[Task]] = {
    GSM8KTask.name: GSM8KTask,
    MATH500Task.name: MATH500Task,
    MMLUProTask.name: MMLUProTask,
    GPQATask.name: GPQATask,
    TruthfulQATask.name: TruthfulQATask,
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
    "LocalCorpusTask",
    "MultipleChoiceTask",
    "GSM8KTask",
    "MATH500Task",
    "MMLUProTask",
    "GPQATask",
    "TruthfulQATask",
    "TASK_REGISTRY",
    "load_task",
    "available_tasks",
]
