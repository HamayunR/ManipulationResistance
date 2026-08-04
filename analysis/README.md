# The analysis layer

This directory turns run outputs into figures you can put in a paper, and — just
as importantly — tells you which figures you *cannot* draw yet and what to run
to fix that.

## Start here: the mental model

**The raw output files are the evidence.** Every run writes a directory under
`outputs/` containing `config.yaml`, `summary.json`, `results.jsonl`,
`routing.jsonl`, `trace.jsonl` and `transcripts.jsonl`. Nothing in this
directory ever changes them.

Four steps sit on top of that evidence:

| Step | Script | What it does |
| --- | --- | --- |
| 1. Check | `validate_runs.py` | Are these runs intact, and may they be pooled? |
| 2. Normalise | `collect_runs.py` | Flatten them into three CSVs everything else reads. |
| 3. Plan | `check_figure_readiness.py` | Which figures are possible, and what is missing? |
| 4. Draw | `figure_*.py` | Compute and plot, reproducibly, with provenance. |

**Do not compute a figure by hand from individual JSONL rows.** It is not that
it is tedious; it is that the four traps below are invisible when you do:

* pooling runs that should not be pooled — a v1 log with a v2 log, or two
  runs scored against different corpus bytes;
* treating decoding seeds, permutation seeds and debate rounds as if they were
  independent questions (they are replications of the same question);
* pooling `sampled` routing with `enumerated` routing — different mechanisms;
* reading a missing value as a zero.

The scripts refuse all four. A hand-written aggregation does not.

## Quick start

```bash
# everything, in one command
python analysis/run_headline_analysis.py \
    outputs/exp_my_experiment \
    --output analysis_artifacts/pilot \
    --w-rho 1.0 --w-adoption 1.0 \
    --heatmap-metric accuracy
```

It discovers runs, validates them, writes the tables, writes `readiness.json`,
draws every figure whose data exists, and prints the precise reason for each
one it skips. Add `--strict` to make a missing requested figure a non-zero exit
(useful in CI). Add `--figures 1 3 4` to attempt only some.

## The commands, one at a time

```bash
# 1. validate (exit code 0 only if there are no errors)
python analysis/validate_runs.py outputs/exp_a outputs/exp_b
python analysis/validate_runs.py outputs/ --json validation.json

# 2. normalise
python analysis/collect_runs.py \
    outputs/exp_a outputs/exp_b \
    --output analysis_artifacts/my_analysis

# 3. what can be drawn?
python analysis/check_figure_readiness.py analysis_artifacts/my_analysis

# 4. figures (each writes CSV + PNG + PDF + metadata JSON)
python analysis/figure_accuracy_vs_adversaries.py analysis_artifacts/my_analysis
python analysis/figure_empirical_ic_regret.py    analysis_artifacts/my_analysis \
    --w-rho 1.0 --w-adoption 1.0
python analysis/figure_exposure_vs_confidence.py analysis_artifacts/my_analysis
python analysis/figure_accuracy_vs_cost.py       analysis_artifacts/my_analysis
python analysis/figure_routing_heatmap.py        analysis_artifacts/my_analysis \
    --metric accuracy
```

Shared flags: `--bootstrap-repetitions N`, `--analysis-seed N`.
Two things are never guessed: Figure 2's utility weights (`--w-rho`,
`--w-adoption`) and Figure 5's `--metric`.

## What gets written

```
analysis_artifacts/my_analysis/
├── manifest.json                       what was collected, when, from where
├── readiness.json                      per-figure verdicts and missing arms
├── headline_analysis.json              master command: generated vs skipped
├── tables/
│   ├── runs.csv                        one row per run x condition
│   ├── results.csv                     one row per run x condition x example x seed x perm_seed
│   ├── routing.csv                     one row per routing decision x agent
│   └── figureN_*.csv                   exactly the plotted values
└── figures/
    └── figureN_*.png / .pdf
```

Every figure also writes `figureN_*.meta.json`: the filters, the grouping
columns, the analysis seed, the bootstrap count, and the analysis code's git
commit. That last one is **the analysis commit, not the
commit that generated the experiments** — the runner does not log its own
commit, and the metadata says so rather than implying a provenance chain that
does not exist.

## The architecture, and why it is shaped this way

```
  native run dirs          adapters              normalised tables        figures
  ---------------          --------              -----------------        -------
  outputs/.../          RUN_ADAPTERS = {          runs.csv                figure_*.py
    config.yaml           "pear":    ...    -->   results.csv       -->   + statistics.py
    summary.json          "generic": ...          routing.csv
    results.jsonl       }
    routing.jsonl
```

* **Native adapters** (`common.py`) know what one system wrote to disk.
* **Normalised schemas** (`common.py`) are what everything above the adapters
  speaks: a generic per-example result row plus an explicit capability record.
* **Shared statistics** (`statistics.py`) is the one implementation of the
  cluster bootstrap, so no two figures disagree about what an error bar means.
* **Figure modules** (`figure_*.py`) each own one quantity and nothing else.

Consequences, which are the point of the split:

* **Adding a model requires no analysis changes.** The model name is data.
* **Adding a benchmark requires no figure changes.** The dataset name is a
  grouping key; figures never hard-code one.
* **Adding a competitor requires one adapter** — see below.
* **A competitor without routing still appears** in the accuracy (Figure 1) and
  cost (Figure 4) figures. It is excluded from the routing figures with a
  stated capability reason, not with zeros.
* **Missing capabilities are not zeros.** `has_routing=False` means "this
  method does not route", not "it routes with degree 0". Missing token usage
  stays missing, and the method is excluded from the cost axis by name rather
  than plotted at zero cost.

### Adding a competitor system

Write the competitor's output as a run directory:

```
my_competitor_run/
├── summary.json     {"schema_version": 2, "adapter": "generic",
│                     "method": "some_baseline", "model": "..."}
├── results.jsonl    one object per example x seed: example_id, prediction (or
│                    decision), correct, optional prompt_tokens /
│                    completion_tokens (or a budget block)
└── config.yaml      optional; dataset/model metadata
```

That is the whole contract — `load_generic_run` in `common.py` handles it, and
`detect_run_adapter` picks it up from the `adapter` key. If the competitor's
native format is something else, add one function and one entry:

```python
RUN_ADAPTERS = {
    "pear": load_pear_run,
    "generic": load_generic_run,
    "their_system": load_their_system_run,   # <- the only change needed
}
```

No figure script changes. Declare the capabilities the system actually has and
the readiness report will place it correctly.

## The figures

| # | Script | Quantity |
| --- | --- | --- |
| 1 | `figure_accuracy_vs_adversaries.py` | Accuracy vs adversarial fraction. The clean run is each attack curve's 0.0 point. |
| 2 | `figure_empirical_ic_regret.py` | Empirical best fixed-report utility gain, from paired truthful/fixed-report runs. |
| 3 | `figure_exposure_vs_confidence.py` | Adversary routing-exposure share vs reported confidence. |
| 4 | `figure_accuracy_vs_cost.py` | Clean accuracy vs mean total tokens per example. |
| 5 | `figure_routing_heatmap.py` | routing_temperature x alpha_influence grid, coloured by an explicit metric. |

### Terminology that matters

* **Routing exposure** is `out_degree_share`: the share of a round's edges
  originating at an agent. Figure 3 plots this.
* **Accumulated influence** is PEAR's rho, carried as `influence_before` in
  `routing.csv` and `influence_history` in `results.csv`. It is a different
  quantity and is never relabelled as exposure.
* **Reported / verified / routing confidence** are three different columns.
  With verification off the router scores the reported value; with verification
  on it scores `verified_confidence = min(reported, g_i)`. `routing_confidence`
  is whichever one the router actually used.
* Figure 2's number is the **empirical best fixed-report utility gain**
  (equivalently, empirical end-to-end IC-regret). It is not exact or universal
  strategyproofness regret; the limitations are listed in the script's
  docstring and copied into its metadata.

## The PEAR diagnostic (not part of this layer)

`check_logs.py` is a **PEAR routing diagnostic**: per-round out-degrees,
shares, confidences and targeted-cross eligibility straight from
`routing.jsonl`. It reads the raw logs directly and is deliberately
mechanism-specific — it is not the common analysis API, and the figure scripts
do not use it.

Note that on small datasets the confidence intervals are still narrow-sample
artefacts. They describe variation in the table, not scientific uncertainty
about a model — check `n_examples` before reading anything into them.
