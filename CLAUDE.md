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
  columns, analysis seed, bootstrap repetitions and the analysis code's git
  commit.

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

## Reading model output as JSON

Agents answer in JSON and write mathematics in the string fields, which breaks
JSON escaping. `utils/json_output.py` owns the recovery; `runner/experiment.py`
only calls it.

* **A backslash command is not an escape.** `\sigma` and `\(` fail to decode
  and lose the whole payload -- including `answer` and `confidence`. `\theta`,
  `\times`, `\frac`, `\rangle` and `\begin` are *worse*: they decode as control
  characters, so the payload looks fine and the prose is mangled.
* **A lost payload does not fail loudly.** `_parse_answer_payload` falls back to
  scanning the raw text for an answer and to a **hard-coded confidence of 3**.
  A run whose `parse_failures` is high therefore routed on a constant, not on
  what the model reported. Check `parse_failures` before trusting any
  confidence-dependent result.
* **`json_repairs` counts payloads that only decoded after repair**, and each
  answer event carries `json_repair`. Non-`none` means the decoded text is not
  byte-for-byte what the model emitted.
* **Only named LaTeX commands override a valid escape.** `"line one\nStep two"`
  is a real newline; `\nabla` is not. The list in `utils/json_output.py` is what
  separates them, and anything not on it keeps its JSON meaning.

## Baselines from other papers

`baselines/<paper>/` holds reimplementations of published methods, dispatched
from `run_one` by `debate.mode`. See `baselines/README.md` for the contract.

* **`debate.rounds` counts debate rounds *after* the independent first answer**,
  for every method. A paper that says "3 rounds" meaning one answer plus two
  debate turns is configured as `rounds: 2`; DebUnc runs also record
  `debunc_total_rounds` so the two readings cannot be confused.
* **`method` says which system produced a row; `adapter` says which on-disk
  format it was read from.** A baseline runs through this harness and writes its
  log format, so its adapter is `pear` and its method is not. `method` is
  resolved per condition from `debate.mode`
  (`analysis/common.py::method_for_mode`), because one run can hold both.
  Labelling a baseline `pear` makes a figure compare PEAR against itself.
* **DebUnc confidence is on a 1-10 scale**, derived from token entropy. PEAR's
  `confidence_history` is a self-reported 1-5. They are different quantities on
  different scales; pooling them is a validation error.
* **DebUnc emits no routing rows.** Its graph is a fixed clique with no routing
  decision, so `routing.jsonl` gets nothing from those conditions. That is the
  correct outcome, not a missing log.
* **Baselines never import the top-level `prompts.py`.** Its JSON schema and
  confidence rubric are part of PEAR's mechanism, not a shared utility.
* **A baseline implements the cells of its paper that it says it does.** DebUnc
  here is Confidence-in-Prompt with mean token entropy only; an unknown
  `debate.debunc` key is an error, so a stale config cannot look honoured.
