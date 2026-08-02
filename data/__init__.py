"""Dataset adapters.

``tasks.py`` normalises each benchmark into the shared :class:`~data.tasks.Example`
format and exposes the :class:`~data.tasks.Task` interface that the runner is
written against.

``local.py`` defines the frozen on-disk corpus format (one JSONL per split plus
a provenance manifest) that the real benchmarks are read from;
``normalize.py`` holds the answer parsers and equivalence rules, keyed by
answer *type* rather than by dataset. ``scripts/fetch_datasets.py`` writes the
corpora. See ``data/README.md``.
"""
