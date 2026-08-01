# CLAUDE.md

## Figures the logs must support

1. accuracy vs adversarial fraction — needs accuracy per adversary count
2. IC-regret by mechanism — needs enough state to recompute routing offline
   under a counterfactual confidence dict (see core/topology.py: pure functions)
3. adversary influence share vs reported confidence — needs per-agent out-degree per round
4. clean accuracy vs token cost — needs token counts per condition
5. heatmap over routing_temperature × alpha_influence — needs both in the resolved config

If a change would make any of these uncomputable from the logs, say so.

## Routing mode is logged per round

`state_aware_permutation` builds its candidate pool two ways, and every round's
`topology` trace event carries `routing_mode` so results are attributable:

- `sampled` — vanilla PEAR: identity-seeded, `mc_permutations` draws with
  replacement, no dedupe. **This is the baseline and the only harness validation
  against the paper's Table 1. It must stay bit-identical — do not "fix" it.**
- `enumerated` — opt-in exact enumeration of S_n via `mc_permutations: null` or
  `enumerate_permutations: true`. No identity seeding. Refuses when
  `n! > enumeration_max_factorial` (default 5040, so n<=7).

Other branches log `identity`, `uniform`, `subgroup`, `random_k_regular`.
Figures mixing sampled and enumerated rounds are mixing two mechanisms; split on
`routing_mode` before aggregating.

## Log schema is versioned

`runner/experiment.py: SCHEMA_VERSION` is stamped into `summary.json`, every
row of `routing.jsonl`, and the trace's first `run_meta` event. **Bump it when
a logged field changes meaning, is renamed, or is removed** -- purely additive
fields do not need a bump. Analysis code should assert on it rather than
silently mis-reading an older run.

`routing.jsonl` is the flat, analysis-ready routing table: one row per
(condition x example x seed x perm_seed x round), carrying `perm`, `topology`,
`in_degree`, `out_degree`, `influence`, `selected_terms`, `objective` and
`targeted_cross_eligibility`. Field sets differ by `routing_mode` -- non
state-aware modes (`identity`, `uniform`, `subgroup`, `random_k_regular`) carry
no `objective` or `selected_terms`, so read defensively.

## Reported vs clean confidence

Every routing row carries two complete per-agent confidence maps.
`reported_confidence` is what the router actually scored; `clean_confidence` is
what the model itself reported. They are equal for every agent unless a
confidence-reporting attack is active, and differ only for attacked agents.
Both are validated for completeness before the event is written -- a partial
map raises rather than producing an unanalysable row.

Rows also carry `confidence_inflation_agent_ids` / `_mode` / `_value`, so no
analysis has to infer which confidence a decision consumed. **Never read
`answer` / `answer_update` events for reported confidence:** those are emitted
before the robustness components run, so they carry the pre-attack value. The
attack's own deltas are in `robustness_confidence_inflation` events.

Two confidence-writing components exist and are mutually exclusive by
construction (enabling both raises): the pre-existing `confidence_perturbation`
(global, several strategies gold-conditioned) and `confidence_inflation`
(scoped to explicit `agent_ids`, leaves answers/reasoning/critiques untouched).
The latter's `fixed_report` mode *replaces* the report with the configured
value -- despite the name it is not monotone, 5 -> 4 is a legal outcome.

## Known gap: mock acceptance policy (blocks stage 5)

`MockLLM` always answers `REJECT` in the update phase, so:

- `critique_acceptance_rate` is pinned at 0.0 and `critique_precision` at NaN
- **the influence-update path is never exercised.** Influence rho accumulates
  from adoption (`runner/experiment.py`, EMA with `influence_beta`), which
  follows from ACCEPT decisions. The utility function is defined over rho, so
  rho must be exercised before stage 5.

Deferred deliberately, not dropped: scripted rounds were chosen over an
acceptance policy so the expected metric values are derivable before a run.
Add an acceptance-policy fixture once stages 1-4 land.

Also uncovered: `_extract_json_object`'s trailing-comma recovery branch --
the mock never emits trailing commas.

## Open question: is the identity seeding deliberate?

`core/topology.py` seeds the sampled candidate pool with the identity
permutation before drawing the rest. **Unresolved whether this is intentional in
upstream PEAR or an artefact.** Do not present it as a known flaw until checked
against the paper.

Measured consequences at n=5, `mc_permutations: 100`:

- Expected identity multiplicity ~1.83 vs ~0.83 for every other permutation, so
  identity carries **~2.2x** the softmax weight. Measured selection frequency
  over 4000 replications: 0.0195 sampled vs 0.0085 enumerated (**2.29x**).
- The pool covers only **~57%** of S_5 (~68 of 120 distinct), so an offline
  expectation over all 120 is a *different mechanism*, not a tighter estimate of
  the same one. This is why figure 2 must be computed under the same
  `routing_mode` as figure 1.
- Plausible interaction with the paper's **equivariance claim**: a distinguished
  permutation in every pool breaks exchangeability over agent labels.
- **Bears on Theorem 5** — check whether the statement assumes softmax over the
  full symmetric group. If it does, the deployed sampled mechanism does not
  satisfy its premise.
- Confounds agent 1 specifically: identity maps role 1 -> agent 1, and
  `make_base_topology("star", ...)` puts the hub at role 1. Hence
  `adversary_agent_id` now accepts a list so the adversary slot can be
  randomised off agent 1; the chosen id and candidate set are logged in the
  `init` event.

## Open question: config vs paper appendix

No config in the repo matches the appendix's n=5, R=5: `default.yaml` is n=5/R=3,
`main_large.yaml` is n=4/R=2. There is no `--n-agents` or `--rounds` CLI flag and
no script sets them, so this is a genuine mismatch rather than a missed override.
Resolve before claiming reproduction. Related: `main_large.yaml`'s
`mc_permutations: 24` at n=4 equals 4!, which looks like intended enumeration but
under `sampled` covers only ~15 of 24 permutations.
