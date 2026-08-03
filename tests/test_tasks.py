"""Tests for the benchmark adapters and the local corpus format.

Answer parsing is where benchmark accuracy silently goes wrong: a parser that
returns the wrong token makes a working system look broken, and one that is too
generous makes a broken system look fine. These tests pin both directions --
what must parse, and what must *not*.

They never touch the network or the fetched corpora. Real data is fetched into
gitignored paths; everything here uses temporary corpora and the packaged
synthetic ``sample`` splits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from data.local import (
    CorpusMismatch,
    CorpusRecord,
    DatasetNotAvailable,
    SAMPLE_SPLIT,
    SKIP_CHECK_ENV,
    check_corpus,
    corpus_sha256,
    available_splits,
    load_corpus,
    read_records,
    write_manifest,
    write_records,
)
from data.normalize import (
    extract_boxed,
    math_equal,
    normalize_math,
    numbers_equal,
    parse_choice_letter,
    parse_math_answer,
    parse_number,
)
from data.tasks import (
    TASK_REGISTRY,
    Example,
    GPQATask,
    GSM8KTask,
    MATH500Task,
    MMLUProTask,
    TruthfulQATask,
    available_tasks,
    load_task,
)

BENCHMARKS = ("gsm8k", "math_500", "mmlu_pro", "gpqa", "truthful_qa")


# ------------------------------------------------------------------ registry --
def test_every_planned_benchmark_is_registered() -> None:
    assert set(BENCHMARKS) <= set(available_tasks())
    for name in BENCHMARKS:
        assert TASK_REGISTRY[name].name == name


def test_unknown_dataset_names_the_registered_ones() -> None:
    with pytest.raises(ValueError, match="Registered:"):
        load_task("not_a_benchmark")


def test_default_splits_match_the_upstream_releases() -> None:
    assert GSM8KTask.default_split == "test"
    assert MATH500Task.default_split == "test"
    assert MMLUProTask.default_split == "test"
    assert GPQATask.default_split == "diamond"
    # TruthfulQA publishes only a validation split; defaulting to "test" would
    # fail at load time on a dataset that has no test split at all.
    assert TruthfulQATask.default_split == "validation"


# ------------------------------------------------------- multiple choice --
@pytest.mark.parametrize(
    "text,expected",
    [
        ("C", "C"),
        ("(C)", "C"),
        ("C.", "C"),
        ("The answer is C", "C"),
        ("the answer is (D).", "D"),
        ("Final answer: B", "B"),
        ("\\boxed{E}", "E"),
        ("**A**", "A"),
        ("After weighing the options, I conclude the answer is B.", "B"),
        ("Option G is correct", "G"),
    ],
)
def test_choice_letter_is_extracted(text: str, expected: str) -> None:
    assert parse_choice_letter(text, n_options=10) == expected


def test_choice_letter_prefers_the_explicit_statement() -> None:
    text = "Option A looks tempting and D is plausible, but the answer is C."
    assert parse_choice_letter(text, n_options=4) == "C"


def test_choice_letter_rejects_a_letter_beyond_the_options() -> None:
    # A four-option question has no option J; scoring that as merely wrong
    # would hide a prompting or option-count bug.
    assert parse_choice_letter("The answer is J", n_options=4) == ""


def test_choice_letter_returns_empty_for_no_answer() -> None:
    assert parse_choice_letter("", n_options=4) == ""
    assert parse_choice_letter("I cannot determine this.", n_options=4) == ""


def test_multiple_choice_scoring_uses_letters(tmp_path: Path) -> None:
    example = Example(id="q1", question="?", answer="C", choices=["w", "x", "y", "z"])
    task = _mmlu_task(tmp_path, [example])

    assert task.score("C", example) is True
    assert task.score("The answer is C", example) is True
    assert task.score("B", example) is False
    # Option *text* is not an answer: the runner's baselines emit letters.
    assert task.score("y", example) is False


def test_multiple_choice_prompt_lists_lettered_options(tmp_path: Path) -> None:
    example = Example(id="q1", question="Which?", answer="B", choices=["first", "second"])
    task = _mmlu_task(tmp_path, [example])

    prompt = task.format_question(example)

    assert "A. first" in prompt
    assert "B. second" in prompt
    assert "MMLU-Pro" in prompt


# ------------------------------------------------------------------ numbers --
@pytest.mark.parametrize(
    "text,expected",
    [
        ("18", "18"),
        ("The answer is 18.", "18"),
        ("$1,234", "1234"),
        ("She makes $18 every day", "18"),
        ("-5", "-5"),
        ("3.50", "3.50"),
        ("First 12 then 7, so the total is 19", "19"),
        ("no digits here", ""),
        ("", ""),
    ],
)
def test_number_parsing(text: str, expected: str) -> None:
    assert parse_number(text) == expected


def test_numeric_equality_tolerates_formatting() -> None:
    assert numbers_equal("18", "18") is True
    assert numbers_equal("18.0", "18") is True
    assert numbers_equal("1,000", "1000") is True
    assert numbers_equal("19", "18") is False
    assert numbers_equal("", "18") is False
    # A numeric benchmark is not scored on prose.
    assert numbers_equal("eighteen", "18") is False


def test_gsm8k_scoring(tmp_path: Path) -> None:
    example = Example(id="g1", question="?", answer="18")
    task = _gsm8k_task(tmp_path, [example])

    assert task.score("18", example) is True
    assert task.score("So she makes $18 per day.", example) is True
    assert task.score("The answer is 17.", example) is False
    assert task.parse_answer("She earns 18 dollars") == "18"


# --------------------------------------------------------------------- math --
def test_extract_boxed_handles_nesting() -> None:
    assert extract_boxed("so \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert extract_boxed("\\boxed{1} then \\boxed{2}") == "2"
    assert extract_boxed("no box here") == ""


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("\\dfrac{1}{2}", "\\frac{1}{2}"),
        ("\\frac12", "\\frac{1}{2}"),
        ("\\left( 3, \\frac{\\pi}{2} \\right)", "(3,\\frac{\\pi}{2})"),
        ("x = 5", "5"),
        ("1,000", "1000"),
        (".5", "0.5"),
        ("5.50", "5.5"),
        ("50\\%", "50"),
        ("$12$", "12"),
        ("90^\\circ", "90"),
        ("\\text{Evelyn}", "Evelyn"),
        ("\\sqrt3", "\\sqrt{3}"),
    ],
)
def test_math_normalisation(raw: str, expected: str) -> None:
    assert normalize_math(raw) == expected


def test_math_normalisation_keeps_meaning() -> None:
    # An equation with a compound left-hand side must not be truncated.
    assert normalize_math("x + y = 5") == "x+y=5"


def test_math_equality() -> None:
    assert math_equal("\\dfrac{1}{2}", "\\frac{1}{2}") is True
    assert math_equal("0.50", "0.5") is True
    assert math_equal("\\frac{1}{2}", "\\frac{1}{3}") is False
    assert math_equal("", "5") is False
    # Symbolic equivalence is explicitly not claimed, so this stays False
    # rather than being quietly counted as correct.
    assert math_equal("\\frac{1}{2}", "0.5") is False


def test_math_answer_parsing_prefers_the_box() -> None:
    assert parse_math_answer("Working... therefore \\boxed{42}.") == "42"
    assert parse_math_answer("blah\nThe final answer is 42.") == "42"
    assert parse_math_answer("") == ""


@pytest.mark.parametrize(
    "text,expected",
    [
        # An answer statement mid-sentence, which is how models actually write.
        ("... therefore the answer is \\text{Evelyn}", "\\text{Evelyn}"),
        ("Hence, the final answer is $5\\sqrt{2}$.", "5\\sqrt{2}"),
        ("So the answer is 42.", "42"),
        ("After simplifying we get\n\\frac{1}{2}", "\\frac{1}{2}"),
    ],
)
def test_math_answer_parsing_finds_a_mid_sentence_statement(text: str, expected: str) -> None:
    assert parse_math_answer(text) == expected


def test_math_scoring_accepts_a_stated_gold(tmp_path: Path) -> None:
    """Regression: a gold answer restated in a sentence must still score."""
    example = Example(id="m1", question="?", answer="\\text{Evelyn}")
    task = _math_task(tmp_path, [example])

    assert task.score("... therefore the answer is \\text{Evelyn}", example) is True


def test_math_500_scoring(tmp_path: Path) -> None:
    example = Example(id="m1", question="?", answer="5\\sqrt{2}")
    task = _math_task(tmp_path, [example])

    assert task.score("\\boxed{5\\sqrt{2}}", example) is True
    assert task.score("5\\sqrt2", example) is True
    assert task.score("\\boxed{10}", example) is False


# ----------------------------------------------------------- corpus format --
def test_corpus_round_trip(tmp_path: Path) -> None:
    records = [
        CorpusRecord("a-1", "Q1", "B", ["x", "y"], {"category": "test"}),
        CorpusRecord("a-2", "Q2", "7", None),
    ]
    info = write_records(tmp_path / "gsm8k" / "test.jsonl", records)

    assert info["n_examples"] == 2
    assert len(info["sha256"]) == 64
    loaded = read_records(tmp_path / "gsm8k" / "test.jsonl")
    assert loaded[0].choices == ["x", "y"]
    assert loaded[0].metadata == {"category": "test"}
    assert loaded[1].choices is None


def test_corpus_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        '{"id": "a", "question": "q", "answer": "1"}\n'
        '{"id": "a", "question": "q2", "answer": "2"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate example ids"):
        read_records(path)


def test_corpus_rejects_a_missing_gold(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"id": "a", "question": "q"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing required field 'answer'"):
        read_records(path)


def test_corpus_accepts_a_json_array(tmp_path: Path) -> None:
    path = tmp_path / "test.json"
    path.write_text(json.dumps([{"id": "a", "question": "q", "answer": "1"}]), encoding="utf-8")

    assert len(read_records(path)) == 1


def test_missing_corpus_names_the_fetch_command(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotAvailable) as excinfo:
        load_corpus(name="gsm8k", split="test", data_dir=tmp_path, fetch_hint="hint here")

    message = str(excinfo.value)
    assert "python scripts/fetch_datasets.py gsm8k --split test" in message
    assert "hint here" in message
    assert SAMPLE_SPLIT in message


def test_manifest_is_not_listed_as_a_split(tmp_path: Path) -> None:
    """manifest.json sits beside the splits and shares their suffix."""
    write_records(tmp_path / "gsm8k" / "test.jsonl", [CorpusRecord("a", "q", "1")])
    write_manifest(tmp_path / "gsm8k", {"dataset": "gsm8k", "splits": {}})

    splits = available_splits(tmp_path, "gsm8k")

    assert "manifest" not in splits
    # "test" is on disk; "sample" is always offered from the packaged fixtures.
    assert splits == ["sample", "test"]


def test_manifest_merges_splits(tmp_path: Path) -> None:
    write_manifest(tmp_path, {"dataset": "gsm8k", "splits": {"test": {"n_examples": 3}}})
    write_manifest(tmp_path, {"dataset": "gsm8k", "splits": {"train": {"n_examples": 5}}})

    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert sorted(payload["splits"]) == ["test", "train"]


# ------------------------------------------------------- packaged samples --
@pytest.mark.parametrize("name", ["gsm8k", "math_500", "mmlu_pro", "truthful_qa"])
def test_sample_split_loads_offline(name: str, tmp_path: Path) -> None:
    """The synthetic split must work with no fetched data present."""
    task = load_task(name, split=SAMPLE_SPLIT, data_dir=str(tmp_path))

    assert len(task) == 3
    for example in task.examples:
        assert example.question
        assert example.answer
        assert task.format_question(example)
        # Gold answers must be scored as correct by the task's own scorer, or
        # the corpus and the parser disagree about what an answer looks like.
        assert task.score(example.answer, example) is True


def test_gpqa_has_no_packaged_sample() -> None:
    """GPQA items are deliberately not committed to this repository."""
    assert available_splits("data", "gpqa") == [] or SAMPLE_SPLIT not in available_splits(
        "data", "gpqa"
    )


def test_sample_multiple_choice_golds_are_letters(tmp_path: Path) -> None:
    for name in ("mmlu_pro", "truthful_qa"):
        task = load_task(name, split=SAMPLE_SPLIT, data_dir=str(tmp_path))
        for example in task.examples:
            assert example.choices, f"{name}: MC task must carry choices"
            assert example.answer in "ABCDEFGHIJ"
            assert len(example.answer) == 1


def test_num_examples_caps_the_split(tmp_path: Path) -> None:
    task = load_task("gsm8k", split=SAMPLE_SPLIT, num_examples=2, data_dir=str(tmp_path))

    assert len(task) == 2


# --------------------------------------------------- runner compatibility --
def test_wrong_answer_helper_works_for_each_task(tmp_path: Path) -> None:
    """The robustness adversary needs a wrong answer it can actually produce."""
    from runner.experiment import _default_wrong_answer

    for name in ("gsm8k", "math_500", "mmlu_pro", "truthful_qa"):
        task = load_task(name, split=SAMPLE_SPLIT, data_dir=str(tmp_path))
        for example in task.examples:
            wrong = _default_wrong_answer(task, example)
            assert wrong, f"{name}: no wrong answer could be constructed"
            assert task.score(task.parse_answer(wrong) or wrong, example) is False


def test_parse_answer_is_usable_as_a_bare_callable(tmp_path: Path) -> None:
    """The runner passes ``task.parse_answer`` around unbound from its call site."""
    for name in ("gsm8k", "math_500", "mmlu_pro", "truthful_qa"):
        task = load_task(name, split=SAMPLE_SPLIT, data_dir=str(tmp_path))
        parser = task.parse_answer
        assert parser("") == ""
        assert isinstance(parser("some model output"), str)


# --------------------------------------------------- corpus integrity --
def _fetched_corpus(tmp_path: Path, records) -> Path:
    """A corpus plus the manifest a real fetch would have written."""
    info = write_records(tmp_path / "gsm8k" / "test.jsonl", records)
    write_manifest(
        tmp_path / "gsm8k",
        {"dataset": "gsm8k", "splits": {"test": {"sha256": info["sha256"], "n_examples": info["n_examples"]}}},
    )
    return tmp_path / "gsm8k" / "test.jsonl"


def test_intact_corpus_verifies(tmp_path: Path) -> None:
    _fetched_corpus(tmp_path, [CorpusRecord("a", "q", "1")])

    integrity = check_corpus(
        tmp_path / "gsm8k" / "test.jsonl", name="gsm8k", split="test", data_dir=tmp_path
    )

    assert integrity.verified is True
    assert integrity.sha256 == integrity.expected


def test_edited_corpus_is_refused_at_load_time(tmp_path: Path) -> None:
    """The check must fire when the data is loaded, not only in a command.

    A corpus that changed after the fetch silently changes what every run is
    scored against, and nothing downstream can recover the difference.
    """
    path = _fetched_corpus(tmp_path, [CorpusRecord("a", "q", "1")])
    path.write_text('{"answer": "2", "id": "a", "question": "q"}\n', encoding="utf-8")

    with pytest.raises(CorpusMismatch) as excinfo:
        load_task("gsm8k", split="test", data_dir=str(tmp_path))

    message = str(excinfo.value)
    assert "does not match the checksum" in message
    assert "--force" in message  # tells you how to fix it
    assert SKIP_CHECK_ENV in message  # ... and how to override, knowingly


def test_the_override_is_explicit(tmp_path: Path, monkeypatch) -> None:
    path = _fetched_corpus(tmp_path, [CorpusRecord("a", "q", "1")])
    path.write_text('{"answer": "2", "id": "a", "question": "q"}\n', encoding="utf-8")
    monkeypatch.setenv(SKIP_CHECK_ENV, "1")

    task = load_task("gsm8k", split="test", data_dir=str(tmp_path))

    assert len(task) == 1
    # The run still records that the corpus was not the fetched one.
    assert task.corpus_status == "no_manifest"


def test_corpus_without_a_manifest_loads_but_says_so(tmp_path: Path) -> None:
    write_records(tmp_path / "gsm8k" / "test.jsonl", [CorpusRecord("a", "q", "1")])

    task = load_task("gsm8k", split="test", data_dir=str(tmp_path))

    assert task.corpus_status == "no_manifest"
    assert task.corpus_sha256


def test_sample_split_is_marked_synthetic(tmp_path: Path) -> None:
    task = load_task("gsm8k", split=SAMPLE_SPLIT, data_dir=str(tmp_path))

    assert task.corpus_status == "synthetic_sample"


def test_write_and_check_agree_on_the_checksum(tmp_path: Path) -> None:
    """One definition of the checksum, used by the writer and the reader."""
    info = write_records(tmp_path / "x" / "test.jsonl", [CorpusRecord("a", "q", "1")])

    assert corpus_sha256(tmp_path / "x" / "test.jsonl") == info["sha256"]


# ------------------------------------------------------------ gated access --
def test_gated_dataset_gives_an_instruction_not_a_stack_trace(monkeypatch) -> None:
    """A gated dataset lists its parquet files to anyone but refuses the download.

    Guarding only the listing call turns an access problem into a traceback at
    the point where the fetch would otherwise have started.
    """
    import urllib.error
    import urllib.request

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import fetch_datasets

    monkeypatch.setattr(
        fetch_datasets, "_get_json", lambda url, token=None: ["https://example/0.parquet"]
    )

    def _refuse(*args, **kwargs):
        raise urllib.error.HTTPError("https://example/0.parquet", 401, "Unauthorized", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        fetch_datasets.load_split("Idavidrein/gpqa", "gpqa_diamond", "train")

    message = str(excinfo.value)
    assert "gated" in message
    assert "HF_TOKEN" in message
    assert "downloading the data" in message


def test_gated_message_differs_when_a_token_is_present(monkeypatch) -> None:
    import urllib.error

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import fetch_datasets

    monkeypatch.setenv("HF_TOKEN", "hf_fake")
    error = urllib.error.HTTPError("https://example", 403, "Forbidden", {}, None)

    message = str(fetch_datasets._access_error("Idavidrein/gpqa", error, "downloading the data"))

    assert "does not grant access" in message
    assert "accept the conditions" in message


# ---------------------------------------------------------------- helpers --
def _corpus_task(cls, tmp_path: Path, examples):
    """Write examples to a temporary corpus and load them through the task."""
    records = [
        CorpusRecord(e.id, e.question, e.answer, e.choices, None) for e in examples
    ]
    write_records(tmp_path / cls.name / f"{cls.default_split}.jsonl", records)
    return cls(data_dir=str(tmp_path))


def _mmlu_task(tmp_path: Path, examples):
    return _corpus_task(MMLUProTask, tmp_path, examples)


def _gsm8k_task(tmp_path: Path, examples):
    return _corpus_task(GSM8KTask, tmp_path, examples)


def _math_task(tmp_path: Path, examples):
    return _corpus_task(MATH500Task, tmp_path, examples)
