"""The on-disk corpus format: one frozen JSONL file per benchmark split.

Why the pipeline reads local files instead of calling a dataset library at run
time:

* **Reproducibility.** A run scores against the exact bytes on disk. Upstream
  datasets are revised; a silent revision between two runs would move a
  headline number with nothing in the logs to explain it. Every corpus carries
  a ``manifest.json`` recording the source, revision, row count and checksum.
* **Frozen option order.** GPQA and TruthfulQA store the correct answer first
  in their raw form. The shuffle that fixes that is applied *once*, at fetch
  time, with a recorded seed -- so it is part of the data, not something each
  run re-derives (and could re-derive differently after a refactor).
* **Offline and air-gapped runs.** Cluster jobs and vLLM nodes do not need
  network access or dataset-library versions to agree.

Format -- ``<data_dir>/<name>/<split>.jsonl``, one object per line::

    {"id": "gsm8k-test-0001",
     "question": "...",
     "answer": "18",              # option LETTER for multiple choice
     "choices": ["...", "..."],   # null for free-form answers
     "metadata": {...}}           # provenance; never used for scoring

``choices`` carries letter semantics (see :mod:`data.tasks`): non-empty means
the runner treats answers as option letters.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

CORPUS_SUFFIXES = (".jsonl", ".json")
MANIFEST_NAME = "manifest.json"

#: Packaged synthetic samples. Structurally identical to a real corpus, so the
#: pipeline can be exercised offline, but the items are invented. They are
#: reachable only through the explicit ``sample`` split, so a real evaluation
#: can never fall back to them by accident.
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_SPLIT = "sample"


class DatasetNotAvailable(FileNotFoundError):
    """Raised when a benchmark split has not been fetched yet."""


@dataclass(frozen=True)
class CorpusRecord:
    """One normalised item as stored on disk."""

    id: str
    question: str
    answer: str
    choices: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def as_json(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "choices": list(self.choices) if self.choices else None,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


def corpus_path(data_dir: str | os.PathLike, name: str, split: str) -> Optional[Path]:
    """Locate ``<data_dir>/<name>/<split>.jsonl``, or the packaged sample.

    ``.json`` is accepted as well: some users drop a hand-made JSON array in
    place while prototyping a new benchmark.
    """
    if str(split) == SAMPLE_SPLIT:
        for suffix in CORPUS_SUFFIXES:
            candidate = FIXTURES_DIR / name / f"{SAMPLE_SPLIT}{suffix}"
            if candidate.is_file():
                return candidate
        return None
    base = Path(data_dir) / name
    for suffix in CORPUS_SUFFIXES:
        candidate = base / f"{split}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def read_records(path: str | os.PathLike) -> List[CorpusRecord]:
    """Read a corpus file. Accepts JSONL or a JSON array of the same objects."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"{path}: corpus file is empty")

    if path.suffix == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected a JSON array of items")
        rows = list(enumerate(payload, start=1))
    else:
        rows = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                rows.append((lineno, json.loads(line)))

    records: List[CorpusRecord] = []
    for lineno, row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{lineno}: expected a JSON object")
        for field in ("id", "question", "answer"):
            if row.get(field) in (None, ""):
                raise ValueError(f"{path}:{lineno}: missing required field {field!r}")
        choices = row.get("choices")
        if choices is not None and (
            not isinstance(choices, list) or not all(isinstance(c, str) for c in choices)
        ):
            raise ValueError(f"{path}:{lineno}: choices must be a list of strings or null")
        records.append(
            CorpusRecord(
                id=str(row["id"]),
                question=str(row["question"]),
                answer=str(row["answer"]),
                choices=list(choices) if choices else None,
                metadata=row.get("metadata") or None,
            )
        )

    ids = [record.id for record in records]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise ValueError(f"{path}: duplicate example ids, e.g. {duplicates}")
    return records


def write_records(
    path: str | os.PathLike, records: Iterable[CorpusRecord]
) -> Dict[str, Any]:
    """Write a corpus file; return ``{"n_examples", "sha256", "path"}``.

    The checksum goes into the manifest so a run can be tied to the exact bytes
    it scored against.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record.as_json(), ensure_ascii=False, sort_keys=True)
            handle.write(line + "\n")
            digest.update(line.encode("utf-8"))
            count += 1
    return {"path": str(path), "n_examples": count, "sha256": digest.hexdigest()}


def write_manifest(directory: str | os.PathLike, payload: Dict[str, Any]) -> Path:
    """Merge one split's provenance into the dataset's ``manifest.json``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    existing: Dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    splits = existing.get("splits") or {}
    splits.update(payload.get("splits") or {})
    merged = {**existing, **payload, "splits": splits}
    path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_manifest(data_dir: str | os.PathLike, name: str) -> Optional[Dict[str, Any]]:
    """Provenance for a fetched corpus, or ``None`` if it was never written."""
    path = Path(data_dir) / name / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def available_splits(data_dir: str | os.PathLike, name: str) -> List[str]:
    """Splits present on disk for one benchmark, plus ``sample`` if packaged."""
    found = set()
    base = Path(data_dir) / name
    if base.is_dir():
        for entry in base.iterdir():
            # manifest.json sits beside the splits and shares their suffix.
            if entry.name != MANIFEST_NAME and entry.suffix in CORPUS_SUFFIXES:
                found.add(entry.stem)
    if (FIXTURES_DIR / name).is_dir():
        found.add(SAMPLE_SPLIT)
    return sorted(found)


def missing_corpus_error(
    name: str, split: str, data_dir: str | os.PathLike, fetch_hint: str = ""
) -> DatasetNotAvailable:
    """A refusal that tells the user the exact command that fixes it."""
    present = available_splits(data_dir, name)
    lines = [
        f"no local corpus for dataset {name!r} split {split!r}.",
        f"Looked for: {Path(data_dir) / name / (split + '.jsonl')}",
        f"Splits available locally: {present or 'none'}",
        "",
        "Fetch it (writes the normalised JSONL plus a provenance manifest):",
        f"    python scripts/fetch_datasets.py {name} --split {split}",
    ]
    if fetch_hint:
        lines += ["", fetch_hint]
    lines += [
        "",
        f"For an offline smoke test use the packaged synthetic split: "
        f"dataset.split: {SAMPLE_SPLIT} (invented items, never a real result).",
    ]
    return DatasetNotAvailable("\n".join(lines))


def load_corpus(
    *,
    name: str,
    split: str,
    data_dir: str | os.PathLike,
    fetch_hint: str = "",
) -> List[CorpusRecord]:
    """Read one benchmark split, or raise with the command that would fix it."""
    path = corpus_path(data_dir, name, split)
    if path is None:
        raise missing_corpus_error(name, split, data_dir, fetch_hint)
    return read_records(path)


__all__ = [
    "CorpusRecord",
    "DatasetNotAvailable",
    "FIXTURES_DIR",
    "MANIFEST_NAME",
    "SAMPLE_SPLIT",
    "available_splits",
    "corpus_path",
    "load_corpus",
    "read_manifest",
    "read_records",
    "write_manifest",
    "write_records",
]
