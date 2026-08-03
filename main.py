"""Project entry point for PEAR experiments.

This script is the single canonical CLI for the PEAR harness. It parses
command-line flags, optionally overrides a few common config knobs, and
hands off to :func:`runner.experiment.run_experiment` which does the heavy
lifting (timestamped output directory, JSONL trace, summary aggregation).

Usage
-----
Run the default settings with a configured model backend
(no API key required)::

    python main.py --config configs/default.yaml

Override the model registry entry from the CLI without editing the YAML::

    python main.py --config configs/default.yaml \\
                   --model gpt-4o-mini --num-examples 50

The repository root *is* the source directory: every subpackage (``core``,
``models``, ``runner``, ...) sits next to this file, and ``main.py`` is
launched from the repo root, so Python automatically puts the right
directory on ``sys.path``. No path manipulation is required.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import yaml

from runner.experiment import ExperimentConfig, run_experiment


# Argument parsing
def _parse_model_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected KEY=VALUE")
    key, raw_value = text.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("override key cannot be empty")
    value = yaml.safe_load(raw_value)
    return key, value


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser used by :func:`main`.

    Kept as a separate function so unit tests / tooling can introspect the
    CLI surface without invoking the runner.
    """
    parser = argparse.ArgumentParser(
        prog="pear",
        description=(
            "Run an PEAR multi-agent debate experiment from a YAML config. "
            "Outputs are written to a timestamped directory under the path "
            "named in the config."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the experiment YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional short slug appended to the run-directory timestamp.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Override agents.model. Must match a key in configs/models.yaml "
            "(e.g. 'qwen2.5-7b-vllm', 'claude-haiku-4-5')."
        ),
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Override dataset.num_examples (0 = all).",
    )
    parser.add_argument(
        "--base-topology",
        type=str,
        choices=["clique", "star", "ring", "chain", "random_sparse", "k_regular"],
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "cot",
            "cot_sc",
            "fixed",
            "pear_full",
            "random",
            "random_k_regular",
        ],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Override dataset.name. Registered: "
            "gsm8k | math_500 | mmlu_pro | gpqa | truthful_qa."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help=(
            "Override dataset.split. Omit to use the benchmark's own default. "
            "Use 'sample' for the packaged synthetic items (smoke tests only)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override paths.output_dir for this run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override replication.seeds (decoding seeds; space-separated ints).",
    )
    parser.add_argument(
        "--perm-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Override replication.agent_perm_seeds (topology-shuffle seeds).",
    )
    parser.add_argument(
        "--model-override",
        type=_parse_model_override,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override one key from the selected model registry entry. "
            "May be repeated, e.g. --model-override tensor_parallel_size=4."
        ),
    )
    parser.add_argument(
        "--parallel-examples",
        type=int,
        default=None,
        help=(
            "Batch this many independent examples per agent turn. "
            "Use >1 with vLLM for higher throughput."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars in the command line.",
    )
    return parser


# Main entry
def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the desired process exit code.

    Parameters
    ----------
    argv:
        Optional explicit argv list (mainly for unit testing). When ``None``
        the standard ``sys.argv[1:]`` is used.

    Returns
    -------
    int
        ``0`` on success, ``1`` if the runner raised an unhandled exception.
    """
    args = _build_parser().parse_args(argv)

    # Load + resolve YAML (handles ``extends:`` chaining inside).
    cfg = ExperimentConfig.from_file(args.config, run_tag=args.tag)

    # Apply CLI overrides without round-tripping through YAML.
    if args.model is not None:
        cfg.raw.setdefault("agents", {})["model"] = args.model
    if args.num_examples is not None:
        cfg.raw.setdefault("dataset", {})["num_examples"] = args.num_examples
    if args.base_topology is not None:
        cfg.raw.setdefault("debate", {})["base_topology"] = args.base_topology
    if args.mode is not None:
        cfg.raw.setdefault("debate", {})["mode"] = args.mode
    if args.dataset is not None:
        cfg.raw.setdefault("dataset", {})["name"] = args.dataset
    if args.split is not None:
        cfg.raw.setdefault("dataset", {})["split"] = args.split
    if args.output_dir is not None:
        cfg.raw.setdefault("paths", {})["output_dir"] = args.output_dir
    if args.seeds is not None:
        cfg.raw.setdefault("replication", {})["seeds"] = list(args.seeds)
    if args.perm_seeds is not None:
        cfg.raw.setdefault("replication", {})["agent_perm_seeds"] = list(args.perm_seeds)
    if args.model_override:
        overrides = cfg.raw.setdefault("agents", {}).setdefault("model_overrides", {})
        overrides.update(dict(args.model_override))
    if args.parallel_examples is not None:
        cfg.raw.setdefault("runner", {})["parallel_examples"] = args.parallel_examples
    if args.no_progress:
        cfg.raw.setdefault("runner", {})["show_progress"] = False

    try:
        summaries = run_experiment(cfg)
    except Exception as exc:  # pragma: no cover - top-level catch-all
        print(f"experiment failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    # Pretty-print a one-row-per-condition summary so the user can see the
    # outcome without grepping the run-dir's summary.json.
    payload = [
        {
            "condition": s.condition,
            "accuracy": s.accuracy,
            "n_runs": s.n_runs,
            "n_examples": s.n_examples,
            **s.summary,
        }
        for s in summaries
    ]
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
