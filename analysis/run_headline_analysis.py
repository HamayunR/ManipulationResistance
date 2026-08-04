"""One command: validate runs, normalise them, and draw what the data supports.

Usage
-----
    python analysis/run_headline_analysis.py \\
        outputs/experiment_root \\
        --output analysis_artifacts/pilot

    python analysis/run_headline_analysis.py outputs/ --output art \\
        --figures 1 3 4 --strict

It analyses existing outputs and never runs an experiment.

Pipeline:

1. discover run directories (recursively, through the adapter registry);
2. validate them and refuse to pool what should not be pooled;
3. write the normalised tables;
4. write ``readiness.json``;
5. draw only the figures whose required arms and capabilities are present, and
   print the exact reason for each one it skips.

A skipped figure is a result: it names the experiment still to run. ``--strict``
turns "requested but unavailable" into a non-zero exit for CI.

Figures needing a choice that has no defensible default must be given it:
Figure 2 needs ``--w-rho`` and ``--w-adoption``, Figure 5 needs
``--heatmap-metric``. Neither is guessed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.check_figure_readiness import HEATMAP_METRICS, check_readiness, print_readiness  # noqa: E402
from analysis.collect_runs import collect  # noqa: E402
from analysis.common import RunLoadError, SUPPORTED_SCHEMA_VERSIONS, discover_runs  # noqa: E402
from analysis.figure_utils import FigureError  # noqa: E402
from analysis.statistics import DEFAULT_ANALYSIS_SEED, DEFAULT_BOOTSTRAP_REPETITIONS  # noqa: E402

import analysis.figure_accuracy_vs_adversaries as figure1  # noqa: E402
import analysis.figure_accuracy_vs_cost as figure4  # noqa: E402
import analysis.figure_empirical_ic_regret as figure2  # noqa: E402
import analysis.figure_exposure_vs_confidence as figure3  # noqa: E402
import analysis.figure_routing_heatmap as figure5  # noqa: E402

ALL_FIGURES = (1, 2, 3, 4, 5)

FIGURE_NAMES = {
    1: figure1.FIGURE_NAME,
    2: figure2.FIGURE_NAME,
    3: figure3.FIGURE_NAME,
    4: figure4.FIGURE_NAME,
    5: figure5.FIGURE_NAME,
}


def _builders(options: Dict[str, Any]) -> Dict[int, Callable[[Path], Dict[str, Any]]]:
    """Bind each figure module to the shared options.

    Every entry has the same shape, so adding a figure is adding a line here --
    the driver knows nothing about what any of them plots.
    """
    common = {
        "repetitions": options["repetitions"],
        "analysis_seed": options["analysis_seed"],
    }

    def _figure2(analysis_dir: Path) -> Dict[str, Any]:
        if options.get("w_rho") is None or options.get("w_adoption") is None:
            raise FigureError(
                "Figure 2 needs explicit utility weights: pass --w-rho and "
                "--w-adoption. They are not chosen for you, and this command "
                "does not search over them."
            )
        return figure2.build_figure(
            analysis_dir, w_rho=options["w_rho"], w_adoption=options["w_adoption"], **common
        )

    def _figure5(analysis_dir: Path) -> Dict[str, Any]:
        metric = options.get("heatmap_metric")
        if not metric:
            raise FigureError(
                "Figure 5 needs an explicit metric: pass --heatmap-metric "
                f"({', '.join(HEATMAP_METRICS)}). No metric is selected automatically."
            )
        return figure5.build_figure(analysis_dir, metric=metric, **common)

    return {
        1: lambda d: figure1.build_figure(d, **common),
        2: _figure2,
        3: lambda d: figure3.build_figure(d, **common),
        4: lambda d: figure4.build_figure(d, **common),
        5: _figure5,
    }


def run(
    paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    figures: Sequence[int] = ALL_FIGURES,
    strict: bool = False,
    repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    analysis_seed: int = DEFAULT_ANALYSIS_SEED,
    w_rho: Optional[float] = None,
    w_adoption: Optional[float] = None,
    heatmap_metric: Optional[str] = None,
    allowed_schema_versions: Optional[Sequence[int]] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Run the whole analysis; return a summary of what was drawn and skipped."""
    output_dir = Path(output_dir)
    wanted = sorted({int(f) for f in figures})
    unknown = [f for f in wanted if f not in ALL_FIGURES]
    if unknown:
        raise FigureError(f"unknown figure(s) {unknown}; known figures: {list(ALL_FIGURES)}")

    run_dirs = discover_runs(paths)
    if not quiet:
        print(f"Discovered {len(run_dirs)} run director{'y' if len(run_dirs) == 1 else 'ies'}")
        for run_dir in run_dirs:
            print(f"  {run_dir}")
        print()

    collection = collect(
        paths,
        output_dir,
        allowed_schema_versions=allowed_schema_versions or SUPPORTED_SCHEMA_VERSIONS,
        command=" ".join(["python analysis/run_headline_analysis.py", *sys.argv[1:]]),
        quiet=quiet,
    )

    readiness = check_readiness(
        output_dir, heatmap_metric=heatmap_metric or "accuracy", figures=wanted
    )
    (output_dir / "readiness.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    if not quiet:
        print()
        print_readiness(readiness)

    builders = _builders(
        {
            "repetitions": repetitions,
            "analysis_seed": analysis_seed,
            "w_rho": w_rho,
            "w_adoption": w_adoption,
            "heatmap_metric": heatmap_metric,
        }
    )

    generated: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for number in wanted:
        try:
            result = builders[number](output_dir)
        except (FigureError, RunLoadError) as exc:
            skipped.append({"figure": number, "name": FIGURE_NAMES[number], "reason": str(exc)})
            continue
        generated.append(
            {
                "figure": number,
                "name": FIGURE_NAMES[number],
                "n_rows": int(len(result["table"])),
                "outputs": result["paths"].as_dict(),
            }
        )

    summary = {
        "output_dir": str(output_dir),
        "input_paths": [str(p) for p in paths],
        "n_runs": collection.manifest["n_runs"],
        "routing_modes": collection.manifest["routing_modes"],
        "requested_figures": wanted,
        "generated": generated,
        "skipped": skipped,
        "strict": strict,
    }
    (output_dir / "headline_analysis.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if not quiet:
        print("\n" + "=" * 78)
        print("HEADLINE ANALYSIS")
        print("=" * 78)
        for entry in generated:
            print(f"  generated  Figure {entry['figure']}: {entry['name']}")
            print(f"             {entry['outputs']['png']}")
        for entry in skipped:
            print(f"  SKIPPED    Figure {entry['figure']}: {entry['name']}")
            print(f"             reason: {entry['reason']}")
        if not skipped:
            print("  no figures skipped")
        print(f"\nwrote {output_dir / 'headline_analysis.json'}")

    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dirs", nargs="+", help="Run directories, or directories containing them.")
    parser.add_argument("--output", required=True, help="Analysis artifact directory to write.")
    parser.add_argument(
        "--figures",
        type=int,
        nargs="+",
        default=list(ALL_FIGURES),
        choices=list(ALL_FIGURES),
        help="Figures to attempt (default: all).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when a requested figure could not be generated.",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=DEFAULT_BOOTSTRAP_REPETITIONS)
    parser.add_argument("--analysis-seed", type=int, default=DEFAULT_ANALYSIS_SEED)
    parser.add_argument("--w-rho", type=float, help="Figure 2 utility weight on rho. Required for Figure 2.")
    parser.add_argument(
        "--w-adoption", type=float, help="Figure 2 utility weight on answer adoption. Required for Figure 2."
    )
    parser.add_argument(
        "--heatmap-metric",
        choices=list(HEATMAP_METRICS),
        help="Figure 5 metric. Required for Figure 5; nothing is selected automatically.",
    )
    parser.add_argument("--allow-schema-version", type=int, action="append", default=[], metavar="N")
    args = parser.parse_args(argv)

    try:
        summary = run(
            args.run_dirs,
            args.output,
            figures=args.figures,
            strict=args.strict,
            repetitions=args.bootstrap_repetitions,
            analysis_seed=args.analysis_seed,
            w_rho=args.w_rho,
            w_adoption=args.w_adoption,
            heatmap_metric=args.heatmap_metric,
            allowed_schema_versions=set(SUPPORTED_SCHEMA_VERSIONS) | set(args.allow_schema_version),
        )
    except (FigureError, RunLoadError) as exc:
        print(f"\nanalysis failed: {exc}", file=sys.stderr)
        return 2

    if args.strict and summary["skipped"]:
        print(
            f"\n--strict: {len(summary['skipped'])} requested figure(s) unavailable",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
