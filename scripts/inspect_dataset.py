"""Look at benchmark data the way the pipeline sees it.

Usage
-----
    python scripts/inspect_dataset.py                      # what is on disk
    python scripts/inspect_dataset.py gsm8k                # one example, in full
    python scripts/inspect_dataset.py mmlu_pro --n 3
    python scripts/inspect_dataset.py truthful_qa --stats
    python scripts/inspect_dataset.py math_500 --id test/precalculus/807.json
    python scripts/inspect_dataset.py gsm8k --split sample --raw

Reading the JSONL directly tells you what was stored. This tells you what the
run will actually do with it, which is a different question and the one that
usually matters:

* the exact prompt body the agents receive (template + options + letters);
* what the scorer accepts, checked by round-tripping the gold answer through
  ``parse_answer`` and ``score`` the way the runner does;
* what the robustness adversary would emit as a deliberately wrong answer.

Nothing here writes anything.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.local import (  # noqa: E402
    SAMPLE_SPLIT,
    available_splits,
    corpus_path,
    read_manifest,
)
from data.tasks import TASK_REGISTRY, Example, Task, load_task  # noqa: E402

RULE = "=" * 78


def _print_catalogue(data_dir: str) -> int:
    """What is fetched, what is not, and how to get the rest."""
    print(f"Corpus root: {Path(data_dir).resolve()}\n")
    print(f"{'dataset':<14}{'answer':<9}{'splits on disk':<28}{'items':>8}  source revision")
    for name, task_cls in sorted(TASK_REGISTRY.items()):
        splits = available_splits(data_dir, name)
        manifest = read_manifest(data_dir, name) or {}
        entries = manifest.get("splits") or {}
        total = sum(int(entry.get("n_examples") or 0) for entry in entries.values())
        revision = str(manifest.get("source_revision") or "-")[:12]
        answer_type = manifest.get("answer_type") or _guess_answer_type(name)
        shown = ",".join(splits) if splits else "(none fetched)"
        print(f"{name:<14}{answer_type:<9}{shown:<28}{total or '-':>8}  {revision}")

    print(
        "\n'sample' is the packaged synthetic split (invented items, offline "
        "smoke tests only).\nFetch real data with:  python scripts/fetch_datasets.py all"
    )
    return 0


def _guess_answer_type(name: str) -> str:
    task_cls = TASK_REGISTRY.get(name)
    from data.tasks import MultipleChoiceTask

    if task_cls and issubclass(task_cls, MultipleChoiceTask):
        return "letter"
    return {"gsm8k": "number", "math_500": "math"}.get(name, "?")


def _print_provenance(name: str, split: str, data_dir: str) -> None:
    manifest = read_manifest(data_dir, name) or {}
    entry = (manifest.get("splits") or {}).get(split) or {}
    path = corpus_path(data_dir, name, split)
    print(f"file          : {path}")
    if split == SAMPLE_SPLIT:
        print("provenance    : packaged synthetic items -- NOT a real benchmark result")
        return
    if not entry:
        print("provenance    : no manifest entry (corpus written by hand?)")
        return
    print(f"source        : {manifest.get('source_repo')} [{manifest.get('source_config')}]")
    print(f"revision      : {manifest.get('source_revision')}")
    print(f"upstream split: {entry.get('upstream_split')}   fetched: {entry.get('fetched_at')}")
    print(f"items         : {entry.get('n_examples')} of {entry.get('n_upstream_rows')} upstream rows")
    if entry.get("limit"):
        print(f"NOTE          : capped at {entry['limit']} items -- a subset, not the full split")
    if manifest.get("option_shuffle_seed"):
        print(
            f"option order  : shuffled once at fetch time, seed "
            f"{manifest['option_shuffle_seed']} (frozen in the corpus)"
        )
    print(f"licence       : {manifest.get('license')}")


def _show_example(task: Task, example: Example, index: int, raw: bool, data_dir: str) -> None:
    print(f"\n{RULE}\nEXAMPLE {index}   id={example.id}\n{RULE}")

    print("\n--- what the agents are shown (format_question) " + "-" * 30)
    print(task.format_question(example))

    print("\n--- gold " + "-" * 68)
    print(f"answer        : {example.answer!r}")
    if example.choices:
        letter = example.answer.strip().upper()
        position = ord(letter) - ord("A")
        if 0 <= position < len(example.choices):
            print(f"which is      : {example.choices[position]!r}")
        print(f"n_options     : {len(example.choices)}")

    print("\n--- scorer round-trip " + "-" * 55)
    _round_trip(task, example)

    if raw:
        print("\n--- raw corpus record " + "-" * 55)
        payload = {
            "id": example.id,
            "question": example.question,
            "answer": example.answer,
            "choices": example.choices,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False)[:2000])


def _round_trip(task: Task, example: Example) -> None:
    """Check the parser and scorer agree with the stored gold answer.

    A mismatch here is the failure that quietly costs a whole sweep: the corpus
    and the parser disagree about what an answer looks like, so a correct model
    is scored wrong.
    """
    stated = (
        f"The answer is {example.answer}."
        if example.choices
        else f"... therefore the answer is {example.answer}."
    )
    cases = [
        ("gold as stored", example.answer),
        ("gold as a model would state it", stated),
    ]
    for label, text in cases:
        parsed = task.parse_answer(text)
        ok = task.score(parsed or text, example)
        flag = "OK " if ok else "*** MISMATCH ***"
        print(f"  {flag} {label:<32} parse_answer -> {parsed!r}")

    try:
        from runner.experiment import _default_wrong_answer

        wrong = _default_wrong_answer(task, example)
        scored = task.score(task.parse_answer(wrong) or wrong, example)
        print(f"  {'OK ' if not scored else '*** BAD ***'} adversary's wrong answer{'':<10} {wrong!r}")
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"  (could not build an adversarial wrong answer: {exc})")


def _print_stats(task: Task) -> None:
    examples = task.examples
    print(f"\n{RULE}\nSPLIT STATISTICS   {task.name}/{task.split}   n={len(examples)}\n{RULE}")
    lengths = sorted(len(e.question) for e in examples)
    print(f"question length (chars): min {lengths[0]}  median {lengths[len(lengths)//2]}  max {lengths[-1]}")

    if examples[0].choices:
        counts = collections.Counter(len(e.choices or []) for e in examples)
        golds = collections.Counter(e.answer for e in examples)
        print(f"options per question   : {dict(sorted(counts.items()))}")
        print(f"gold letter frequency  : {dict(sorted(golds.items()))}")
        top = max(golds.values()) / len(examples)
        print(
            f"most common gold letter: {top:.1%} of items "
            + ("(balanced)" if top < 0.35 else "(skewed -- check the option shuffle)")
        )
    else:
        sample = [e.answer for e in examples[:8]]
        print(f"example gold answers   : {sample}")

    scored = sum(task.score(e.answer, e) for e in examples)
    print(f"gold answers the scorer accepts: {scored}/{len(examples)} ({scored/len(examples):.1%})")
    if scored != len(examples):
        print("  *** the corpus and the parser disagree; fix before running anything ***")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", nargs="?", help="Dataset name. Omit to list what is on disk.")
    parser.add_argument("--split", help="Split (default: the benchmark's own default).")
    parser.add_argument("--n", type=int, default=1, help="How many examples to show.")
    parser.add_argument("--index", type=int, default=0, help="Start at this position.")
    parser.add_argument("--id", help="Show one specific example id instead.")
    parser.add_argument("--raw", action="store_true", help="Also print the raw corpus record.")
    parser.add_argument("--stats", action="store_true", help="Print split-level statistics.")
    parser.add_argument("--data-dir", default="data", help="Corpus root (default: %(default)s).")
    args = parser.parse_args(argv)

    if not args.dataset:
        return _print_catalogue(args.data_dir)

    try:
        task = load_task(args.dataset, split=args.split, data_dir=args.data_dir)
    except Exception as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    print(f"{RULE}\n{task.name}  /  split={task.split}  /  {len(task)} examples\n{RULE}")
    _print_provenance(task.name, task.split, args.data_dir)

    if args.stats:
        _print_stats(task)
        return 0

    if args.id:
        selected: List[Example] = [e for e in task.examples if e.id == args.id]
        if not selected:
            print(f"no example with id {args.id!r} in {task.name}/{task.split}", file=sys.stderr)
            return 1
    else:
        selected = task.examples[args.index : args.index + max(1, args.n)]

    for offset, example in enumerate(selected, start=args.index):
        _show_example(task, example, offset, args.raw, args.data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
