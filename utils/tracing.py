"""JSONL tracer for node-level events.

The tracer is intentionally minimal: it appends one JSON object per call to
the trace file. Each entry is timestamped and gets a monotonically increasing
sequence number so that downstream analysis can recover total event order
even if multiple worker processes are writing (we serialize via fcntl on
POSIX -- best effort; a production setup might prefer SQLite).
"""

from __future__ import annotations

import json
import math
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo


UK_TZ = ZoneInfo("Europe/London")


def _json_default(obj: Any) -> Any:
    """Best-effort JSON fallback for objects we don't normally serialize."""
    # ``set`` / ``frozenset`` / ``tuple`` are very common in metrics traces.
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    if hasattr(obj, "tolist"):  # numpy scalars / arrays
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def json_safe(value: Any) -> Any:
    """Recursively replace NaN / Infinity with ``None``.

    ``json.dumps`` emits bare ``NaN`` and ``Infinity`` tokens. Python reads
    them back, but they are **not valid JSON** -- strict parsers (JavaScript's
    ``JSON.parse``, ``jq``, most dataframe loaders) reject the file outright.
    The diagnostics produce NaN by design (``safe_div`` on a zero denominator),
    so anything carrying them is unreadable outside Python without this.

    Note ``json.dumps``'s ``default=`` hook cannot do this job: floats are
    serialized natively and never routed through it.
    """
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, Mapping):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in value]
    return value


def dumps_safe(obj: Any, **kwargs: Any) -> str:
    """``json.dumps`` that cannot emit invalid JSON.

    Sanitizes non-finite floats to ``None`` first, then serializes with
    ``allow_nan=False`` so any that somehow survive raise instead of being
    written as an invalid token. This is the single serialization entry point
    for every file the harness writes -- trace, results, routing and summary --
    so the three writers cannot drift apart.
    """
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("default", _json_default)
    kwargs["allow_nan"] = False
    return json.dumps(json_safe(obj), **kwargs)


class JsonlTracer:
    """Append-only JSONL writer with timestamps and sequence numbers.

    Thread-safe within a single process. Designed for write-once-read-many
    workloads: the file is opened lazily on the first ``write``.

    Examples
    --------
    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as d:
    ...     t = JsonlTracer(os.path.join(d, "trace.jsonl"))
    ...     t.write({"event": "ping"})
    ...     t.close()
    """

    def __init__(self, path: str | os.PathLike) -> None:
        """Create a tracer that will append events to ``path``.

        The file is created lazily, so constructing a tracer is cheap.
        """
        self._path = Path(path)
        self._lock = threading.Lock()
        self._fp = None  # type: ignore[assignment]
        self._seq = 0

    def _ensure_open(self) -> None:
        """Open the underlying file handle lazily."""
        if self._fp is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Line-buffered append; UTF-8.
            self._fp = open(self._path, "a", buffering=1, encoding="utf-8")

    def write(self, event: Mapping[str, Any]) -> None:
        """Append a single event as one JSON line.

        The original mapping is *not* mutated; we copy and add ``ts`` and
        ``seq`` keys before serialization. Non-JSON-serializable values fall
        back to ``str(obj)``.
        """
        with self._lock:
            self._ensure_open()
            self._seq += 1
            from core.state import RUNTIME_ONLY_KEYS  # local import avoids cycle
            payload = dict(event)
            payload.setdefault("ts", datetime.now(UK_TZ).isoformat())
            payload.setdefault("seq", self._seq)
            # Drop runtime-only channels (RNGs, Task object, ...) and any
            # caller-introduced underscore-prefixed keys.
            payload = {
                k: v for k, v in payload.items()
                if not k.startswith("_") and k not in RUNTIME_ONLY_KEYS
            }
            self._fp.write(dumps_safe(payload))
            self._fp.write("\n")

    def write_many(self, events: list[Mapping[str, Any]]) -> None:
        """Append several events in one critical section."""
        for ev in events:
            self.write(ev)

    def close(self) -> None:
        """Close the underlying file handle if it was ever opened."""
        with self._lock:
            if self._fp is not None:
                self._fp.close()
                self._fp = None

    def __enter__(self) -> "JsonlTracer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
