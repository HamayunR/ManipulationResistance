"""Fetch benchmark data and freeze it into the local corpus format.

Usage
-----
    python scripts/fetch_datasets.py --list
    python scripts/fetch_datasets.py gsm8k
    python scripts/fetch_datasets.py mmlu_pro --split test --limit 500
    python scripts/fetch_datasets.py all --limit 200      # pilot-sized copies
    HF_TOKEN=hf_... python scripts/fetch_datasets.py gpqa --split diamond

Writes ``data/<name>/<split>.jsonl`` plus ``data/<name>/manifest.json``.

Three things this script does that a one-line dataset load does not:

* **Freezes the option order.** GPQA and TruthfulQA store the correct answer
  outside (or at the head of) the distractor list. The shuffle happens here,
  once, seeded per item, and is written into the corpus -- so the gold letter
  is a property of the data rather than of whatever code last ran.
* **Records provenance.** Source repo, resolved commit sha, config, split, row
  count, sha256 of the written file, the shuffle seed and the fetch time all
  land in the manifest. A run can be tied to the exact bytes it scored.
* **Keeps a subset honest.** ``--limit`` takes a deterministic prefix and
  records the cap, so a 200-item pilot is never mistaken for the full split.

Downloads the parquet conversion of each dataset over plain HTTP, so no
dataset library is needed on the machine that runs it (pandas + pyarrow only).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from data.local import CorpusRecord, write_manifest, write_records  # noqa: E402
from data.normalize import LETTERS, letter_for_index  # noqa: E402

HF_API = "https://huggingface.co/api/datasets"

#: Seed base for the per-item option shuffle. Changing it changes every gold
#: letter in the affected corpora, so it is a constant, not an argument.
SHUFFLE_SEED = 20260802


# ------------------------------------------------------------------ downloads --
def _request(url: str, token: Optional[str] = None) -> urllib.request.Request:
    headers = {"User-Agent": "pear-fetch-datasets/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def _get_json(url: str, token: Optional[str] = None) -> Any:
    with urllib.request.urlopen(_request(url, token), timeout=60) as response:
        return json.loads(response.read())


def repo_revision(dataset: str, token: Optional[str] = None) -> Optional[str]:
    """Resolved commit sha of the dataset repo, for the manifest."""
    try:
        return _get_json(f"{HF_API}/{dataset}", token).get("sha")
    except Exception:  # pragma: no cover - provenance is best effort
        return None


def load_split(
    dataset: str, config: str, split: str, token: Optional[str] = None
) -> pd.DataFrame:
    """Download one split via the parquet conversion and return it as a frame."""
    listing_url = f"{HF_API}/{dataset}/parquet/{config}/{split}"
    try:
        urls = _get_json(listing_url, token)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise SystemExit(
                f"{dataset} is gated or private (HTTP {exc.code}).\n"
                "Accept the dataset's conditions on its Hugging Face page, then\n"
                "export a token with read access:\n"
                "    export HF_TOKEN=hf_xxx\n"
                "and re-run this command."
            ) from exc
        raise SystemExit(f"could not list parquet files for {dataset}: {exc}") from exc
    if not isinstance(urls, list) or not urls:
        raise SystemExit(f"no parquet files listed for {dataset} [{config}/{split}]")

    frames = []
    for url in urls:
        print(f"    downloading {url}")
        with urllib.request.urlopen(_request(url, token), timeout=300) as response:
            payload = response.read()
        import io

        frames.append(pd.read_parquet(io.BytesIO(payload)))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------- normalisation --
def _shuffled_options(
    example_key: str, correct: str, distractors: Sequence[str]
) -> tuple[List[str], str]:
    """Deterministically interleave the correct answer with its distractors.

    Seeded per item, so adding or removing items elsewhere in the split cannot
    change any other item's option order.
    """
    options = [str(correct).strip(), *[str(d).strip() for d in distractors]]
    order = list(range(len(options)))
    random.Random(f"{SHUFFLE_SEED}:{example_key}").shuffle(order)
    shuffled = [options[i] for i in order]
    return shuffled, letter_for_index(order.index(0))


def normalize_gsm8k(frame: pd.DataFrame, split: str) -> List[CorpusRecord]:
    """Gold is the value after ``####``; the rest of the field is a rationale."""
    records = []
    for index, row in frame.iterrows():
        answer_field = str(row["answer"])
        if "####" not in answer_field:
            raise ValueError(f"gsm8k row {index}: no '####' marker in the answer field")
        gold = answer_field.split("####")[-1].strip().replace(",", "")
        records.append(
            CorpusRecord(
                id=f"gsm8k-{split}-{index:05d}",
                question=str(row["question"]).strip(),
                answer=gold,
                choices=None,
                metadata={"rationale": answer_field.split("####")[0].strip()},
            )
        )
    return records


def normalize_math_500(frame: pd.DataFrame, split: str) -> List[CorpusRecord]:
    records = []
    for index, row in frame.iterrows():
        records.append(
            CorpusRecord(
                id=str(row.get("unique_id") or f"math500-{split}-{index:05d}"),
                question=str(row["problem"]).strip(),
                answer=str(row["answer"]).strip(),
                choices=None,
                metadata={
                    "subject": row.get("subject"),
                    "level": int(row["level"]) if pd.notna(row.get("level")) else None,
                },
            )
        )
    return records


def normalize_mmlu_pro(frame: pd.DataFrame, split: str) -> List[CorpusRecord]:
    """Options are already ordered and the gold letter is given.

    Upstream pads short questions with ``"N/A"`` options. They are dropped, but
    only from the tail and only when the gold index still points inside what is
    left -- a padded option that the gold refers to would be a data bug worth
    failing on, not worth silently reindexing.
    """
    records = []
    for index, row in frame.iterrows():
        options = [str(option) for option in list(row["options"])]
        answer_index = int(row["answer_index"])
        while len(options) > answer_index + 1 and options[-1].strip().upper() in {"N/A", ""}:
            options.pop()
        if not 0 <= answer_index < len(options):
            raise ValueError(
                f"mmlu_pro row {index}: answer_index {answer_index} outside "
                f"{len(options)} options"
            )
        letter = letter_for_index(answer_index)
        declared = str(row.get("answer") or "").strip().upper()
        if declared and declared != letter:
            raise ValueError(
                f"mmlu_pro row {index}: answer {declared!r} disagrees with "
                f"answer_index {answer_index} ({letter})"
            )
        records.append(
            CorpusRecord(
                id=f"mmlu_pro-{split}-{int(row.get('question_id', index)):06d}",
                question=str(row["question"]).strip(),
                answer=letter,
                choices=options,
                metadata={"category": row.get("category"), "src": row.get("src")},
            )
        )
    return records


def normalize_truthful_qa(frame: pd.DataFrame, split: str) -> List[CorpusRecord]:
    """MC1: exactly one label is 1, and upstream always lists it first."""
    records = []
    for index, row in frame.iterrows():
        targets = row["mc1_targets"]
        choices = [str(c) for c in list(targets["choices"])]
        labels = [int(v) for v in list(targets["labels"])]
        if sum(labels) != 1:
            raise ValueError(f"truthful_qa row {index}: expected exactly one correct MC1 option")
        correct = choices[labels.index(1)]
        distractors = [c for c, label in zip(choices, labels) if label == 0]
        if len(choices) > len(LETTERS):
            raise ValueError(f"truthful_qa row {index}: more options than letters")
        example_id = f"truthful_qa-{split}-{index:05d}"
        options, letter = _shuffled_options(example_id, correct, distractors)
        records.append(
            CorpusRecord(
                id=example_id,
                question=str(row["question"]).strip(),
                answer=letter,
                choices=options,
                metadata={"n_options": len(options), "option_order": "shuffled_at_fetch"},
            )
        )
    return records


def normalize_gpqa(frame: pd.DataFrame, split: str) -> List[CorpusRecord]:
    """Correct answer and distractors live in separate columns upstream."""
    question_col = _first_column(frame, ("Question", "question"))
    correct_col = _first_column(frame, ("Correct Answer", "correct_answer"))
    incorrect_cols = [
        _first_column(frame, (f"Incorrect Answer {n}", f"incorrect_answer_{n}"))
        for n in (1, 2, 3)
    ]
    id_col = _first_column(frame, ("Record ID", "record_id"), required=False)

    records = []
    for index, row in frame.iterrows():
        example_id = f"gpqa-{split}-{str(row[id_col]).strip() if id_col else f'{index:05d}'}"
        options, letter = _shuffled_options(
            example_id, row[correct_col], [row[column] for column in incorrect_cols]
        )
        records.append(
            CorpusRecord(
                id=example_id,
                question=str(row[question_col]).strip(),
                answer=letter,
                choices=options,
                metadata={
                    "subdomain": row.get("Subdomain"),
                    "high_level_domain": row.get("High-level domain"),
                    "option_order": "shuffled_at_fetch",
                },
            )
        )
    return records


def _first_column(
    frame: pd.DataFrame, candidates: Sequence[str], required: bool = True
) -> Optional[str]:
    for name in candidates:
        if name in frame.columns:
            return name
    if required:
        raise ValueError(
            f"none of the expected columns {list(candidates)} are present; "
            f"got {list(frame.columns)[:12]}"
        )
    return None


# ------------------------------------------------------------------ registry --
@dataclass(frozen=True)
class Source:
    """Where one benchmark comes from and how to normalise it."""

    name: str
    dataset: str
    config: str
    splits: Dict[str, str]  # local split name -> upstream split name
    normalize: Callable[[pd.DataFrame, str], List[CorpusRecord]]
    default_split: str
    answer_type: str
    license_note: str
    gated: bool = False
    notes: str = ""
    redistributable_sample: bool = True


SOURCES: Dict[str, Source] = {
    "gsm8k": Source(
        name="gsm8k",
        dataset="openai/gsm8k",
        config="main",
        splits={"test": "test", "train": "train"},
        normalize=normalize_gsm8k,
        default_split="test",
        answer_type="number",
        license_note="MIT",
    ),
    "math_500": Source(
        name="math_500",
        dataset="HuggingFaceH4/MATH-500",
        config="default",
        splits={"test": "test"},
        normalize=normalize_math_500,
        default_split="test",
        answer_type="math",
        license_note="MIT",
        notes="The 500-problem subset of MATH used by the process-supervision line of work.",
    ),
    "mmlu_pro": Source(
        name="mmlu_pro",
        dataset="TIGER-Lab/MMLU-Pro",
        config="default",
        splits={"test": "test", "validation": "validation"},
        normalize=normalize_mmlu_pro,
        default_split="test",
        answer_type="letter",
        license_note="MIT",
        notes="Up to ten options per question; option order is upstream's.",
    ),
    "truthful_qa": Source(
        name="truthful_qa",
        dataset="truthfulqa/truthful_qa",
        config="multiple_choice",
        splits={"validation": "validation"},
        normalize=normalize_truthful_qa,
        default_split="validation",
        answer_type="letter",
        license_note="Apache-2.0",
        notes="MC1 single-true-answer variant; options shuffled at fetch time.",
    ),
    "gpqa": Source(
        name="gpqa",
        dataset="Idavidrein/gpqa",
        config="gpqa_diamond",
        splits={"diamond": "train"},
        normalize=normalize_gpqa,
        default_split="diamond",
        answer_type="letter",
        license_note="CC-BY-4.0, gated",
        gated=True,
        notes=(
            "Gated: accept the conditions on the dataset page and set HF_TOKEN. "
            "The authors ask that items are not reproduced in public text, so no "
            "sample of GPQA is committed to this repository."
        ),
        redistributable_sample=False,
    ),
}

#: Subsets selectable with --config, for the datasets that have more than one.
GPQA_CONFIGS = {"diamond": "gpqa_diamond", "main": "gpqa_main", "extended": "gpqa_extended"}


# --------------------------------------------------------------------- fetch --
@dataclass
class FetchResult:
    name: str
    split: str
    path: str
    n_examples: int
    sha256: str
    manifest: str
    warnings: List[str] = field(default_factory=list)


def fetch(
    name: str,
    *,
    split: Optional[str] = None,
    limit: int = 0,
    data_dir: str | Path = "data",
    token: Optional[str] = None,
    config: Optional[str] = None,
    force: bool = False,
) -> FetchResult:
    """Download one benchmark split and write the frozen corpus."""
    if name not in SOURCES:
        raise SystemExit(f"unknown dataset {name!r}; known: {sorted(SOURCES)}")
    source = SOURCES[name]
    split = split or source.default_split
    if split not in source.splits:
        raise SystemExit(
            f"{name}: unknown split {split!r}; available: {sorted(source.splits)}"
        )

    upstream_config = config or source.config
    if name == "gpqa" and config in GPQA_CONFIGS:
        upstream_config = GPQA_CONFIGS[config]
    upstream_split = source.splits[split]

    out_dir = Path(data_dir) / name
    out_path = out_dir / f"{split}.jsonl"
    if out_path.exists() and not force:
        raise SystemExit(
            f"{out_path} already exists. Pass --force to overwrite it.\n"
            "Overwriting changes what every future run is scored against, so it "
            "is never the default."
        )

    print(f"  {name}: {source.dataset} [{upstream_config}/{upstream_split}]")
    frame = load_split(source.dataset, upstream_config, upstream_split, token)
    print(f"    {len(frame)} upstream rows")

    records = source.normalize(frame, split)
    warnings: List[str] = []
    if limit and limit > 0 and limit < len(records):
        records = records[:limit]
        warnings.append(
            f"capped at the first {limit} items of {split}: a subset, not the full split"
        )

    written = write_records(out_path, records)
    manifest_payload = {
        "dataset": name,
        "source_repo": source.dataset,
        "source_revision": repo_revision(source.dataset, token),
        "source_config": upstream_config,
        "answer_type": source.answer_type,
        "license": source.license_note,
        "notes": source.notes,
        "option_shuffle_seed": SHUFFLE_SEED if name in {"gpqa", "truthful_qa"} else None,
        "fetch_script_version": 1,
        "splits": {
            split: {
                "upstream_split": upstream_split,
                "n_examples": written["n_examples"],
                "n_upstream_rows": int(len(frame)),
                "limit": int(limit) if limit else None,
                "sha256": written["sha256"],
                "file": str(out_path),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        },
    }
    manifest_path = write_manifest(out_dir, manifest_payload)
    print(f"    wrote {out_path} ({written['n_examples']} items)")
    for warning in warnings:
        print(f"    note: {warning}")
    return FetchResult(
        name=name,
        split=split,
        path=str(out_path),
        n_examples=written["n_examples"],
        sha256=written["sha256"],
        manifest=str(manifest_path),
        warnings=warnings,
    )


def verify(data_dir: str | Path = "data") -> int:
    """Re-check every fetched corpus against its manifest checksum.

    Worth running before a long sweep: a truncated download or an edited
    corpus changes what the whole experiment is scored against, and nothing
    downstream can detect it once the runs exist.
    """
    from data.local import read_manifest, read_records
    import hashlib

    data_dir = Path(data_dir)
    problems = 0
    checked = 0
    for name in sorted(SOURCES):
        manifest = read_manifest(data_dir, name)
        if manifest is None:
            print(f"  {name:<14} not fetched")
            continue
        for split, info in sorted((manifest.get("splits") or {}).items()):
            path = Path(info.get("file") or (data_dir / name / f"{split}.jsonl"))
            if not path.is_file():
                print(f"  {name}/{split:<12} MISSING {path}")
                problems += 1
                continue
            digest = hashlib.sha256()
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        digest.update(line.rstrip("\n").encode("utf-8"))
            actual = digest.hexdigest()
            records = read_records(path)
            checked += 1
            status = "ok"
            if actual != info.get("sha256"):
                status = "CHECKSUM MISMATCH -- the file changed after it was fetched"
                problems += 1
            elif len(records) != info.get("n_examples"):
                status = "ROW COUNT MISMATCH"
                problems += 1
            limit = f", capped at {info['limit']}" if info.get("limit") else ""
            print(
                f"  {name}/{split:<12} {len(records):>6} items  "
                f"rev={str(manifest.get('source_revision'))[:12]}{limit}  {status}"
            )
    print(f"\n{checked} corpus file(s) checked, {problems} problem(s)")
    return 1 if problems else 0


def print_catalogue() -> None:
    print(f"{'dataset':<14}{'answer':<9}{'splits':<24}{'source':<32}{'license'}")
    for source in SOURCES.values():
        splits = ",".join(sorted(source.splits)) + (" (gated)" if source.gated else "")
        print(
            f"{source.name:<14}{source.answer_type:<9}{splits:<24}"
            f"{source.dataset:<32}{source.license_note}"
        )
    print(
        "\nEvery dataset also has an offline 'sample' split of invented items "
        "(data/fixtures/),\nexcept gpqa, whose items must not be republished."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset names, or 'all' for every non-gated dataset.",
    )
    parser.add_argument("--split", help="Split to fetch (default: the dataset's own default).")
    parser.add_argument("--config", help="Upstream config/subset override (e.g. gpqa main).")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Keep only the first N items. Recorded in the manifest. 0 = full split.",
    )
    parser.add_argument("--data-dir", default="data", help="Corpus root (default: %(default)s).")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing corpus file.")
    parser.add_argument("--list", action="store_true", help="Show the catalogue and exit.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-check fetched corpora against their manifest checksums and exit.",
    )
    args = parser.parse_args(argv)

    if args.verify:
        return verify(args.data_dir)

    if args.list or not args.datasets:
        print_catalogue()
        return 0

    names = list(args.datasets)
    if "all" in names:
        names = [name for name, source in SOURCES.items() if not source.gated]
        print(f"fetching non-gated datasets: {names}")
        print("(gpqa is gated; fetch it explicitly once HF_TOKEN is set)\n")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    results: List[FetchResult] = []
    failures: List[str] = []
    for name in names:
        try:
            results.append(
                fetch(
                    name,
                    split=args.split,
                    limit=args.limit,
                    data_dir=args.data_dir,
                    token=token,
                    config=args.config,
                    force=args.force,
                )
            )
        except SystemExit as exc:
            failures.append(f"{name}: {exc}")
            print(f"  {name}: FAILED -- {exc}\n", file=sys.stderr)

    print("\nfetched:")
    for result in results:
        print(f"  {result.name}/{result.split}: {result.n_examples} items -> {result.path}")
    if failures:
        print(f"\n{len(failures)} dataset(s) failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
