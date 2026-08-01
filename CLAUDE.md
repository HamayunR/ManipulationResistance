# Project notes

Conventions that are easy to get wrong and expensive to get wrong. See
`analysis/README.md` for the analysis layer itself and `README.md` for the
experiment harness.

## Analysis conventions

* **Sampled IC-regret is measured from paired runs.** The empirical
  fixed-report utility gain compares a truthful arm against `fixed_report`
  arms for values 1-5 over identical (example, seed, perm_seed) coverage. It
  is an end-to-end difference between two runs, not a per-round counterfactual.
* **Offline replay of sampled PEAR would require logging the realised
  candidate pool.** The pool is not logged, so it cannot be replayed after the
  fact; any "what would the router have done" analysis needs that logging
  added first.
* **Enumerated routing is a separate analytic diagnostic.** `routing_mode` is
  `sampled` or `enumerated` (or `identity`, `uniform`, `subgroup`,
  `random_k_regular`). These are different mechanisms. Figures must split on
  `routing_mode`; pooling them is a validation error inside a condition and a
  warning across runs.
* **Out-degree share is routing exposure.** It is the share of a round's edges
  originating at an agent.
* **Accumulated influence is rho** (`influence` on the routing event,
  `influence_before` in `routing.csv`, `influence_history` in `results.csv`).
  It is a different quantity from routing exposure and must not be relabelled
  as one.
* **Figure-ready analysis saves four artefacts**: a CSV of exactly the plotted
  values, a PNG, a PDF, and a metadata JSON recording filters, grouping
  columns, analysis seed, bootstrap repetitions, mock status and the analysis
  code's git commit.

## Confidence provenance

Four distinct per-agent quantities, all logged separately in `routing.jsonl`:

* `clean_confidence` — what the model itself reported.
* `reported_confidence` — what the agent reports after any attack.
* `g_i` — the corroboration ceiling under an active verification mode.
* `verified_confidence` — `min(reported, g_i)`, what the router scores when
  verification is on.

With `verified_confidence_mode: none` the router scores `reported_confidence`
and both verification columns are null. Reading the wrong one describes a
mechanism the run never used.

## Mock runs

The offline mock provider tags every artefact `mock: true`. Mock outputs prove
that code paths run; they are never evidence about a model. Analysis marks them
`diagnostic_only`, and mixing mock with real runs is rejected.
