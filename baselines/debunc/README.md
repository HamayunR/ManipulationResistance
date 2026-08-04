# DebUnc

Reimplementation of **DebUnc: Improving Large Language Model Agent
Communication With Uncertainty Metrics** — Luke Yoffe, Alfonso Amayuelas,
William Yang Wang (UC Santa Barbara), [arXiv:2407.06426][paper]. Reference code:
[github.com/lukeyoffe/debunc][code].

DebUnc takes a plain multi-agent debate and changes one thing: an agent's
confidence is **measured from its own token probabilities** instead of asked
for in words, and that confidence is then conveyed to the other agents.

That makes it the natural comparison point for this repository's verified
confidence: same problem (a confidently wrong agent misleads the others),
different lever (measure confidence rather than corroborate a reported one).

[paper]: https://arxiv.org/abs/2407.06426
[code]: https://github.com/lukeyoffe/debunc

---

## What this port implements

The paper explores a grid: three ways of communicating confidence × three
uncertainty metrics. **This port implements one cell** — Confidence in Prompt
with Mean Token Entropy:

- each agent's uncertainty is the mean full-vocabulary entropy of the tokens it
  generated;
- one round's uncertainties are rescaled onto a 1-10 integer scale;
- the next round's prompt quotes each other agent's response *labelled with its
  confidence level*, and tells the model not to state a confidence of its own;
- the final answer is a majority vote after the last round.

**Not implemented**, and no config accepts them — an unknown `debate.debunc`
key is an error rather than a silently ignored setting:

| left out | why |
|---|---|
| Attention-Others / Attention-All | Rescaling attention onto each agent's token spans requires reaching inside the model's forward pass. It ties the harness to a patched attention kernel and to per-tokenizer span arithmetic, for a mechanism that is not what this repository is comparing against. |
| TokenSAR | One cross-encoder pass per generated token. The paper measures it as slightly *worse* than mean token entropy (AUROC 0.617 vs 0.627) at much higher cost. |
| Oracle | Sets confidence from whether the answer was right, so it needs the gold answer. It is the paper's simulated perfect metric — a ceiling, not a method. |
| 5-shot MMLU | This harness has no few-shot path, and adding one would change what every other method in it is measured on. |

The attention-based methods are the paper's *best* results, so this port does
not reproduce DebUnc's headline number. What it reproduces is the row a
prompt-level confidence mechanism can be compared against.

---

## Running it

```bash
python main.py --config configs/debunc.yaml
```

DebUnc needs a **whitebox backend** — `provider: hf` in `configs/models.yaml`.
It reads full-vocabulary token distributions, which an API backend does not
expose and vLLM does not return. Attempting it with another provider raises
rather than silently running a different mechanism.

```yaml
debate:
  mode: debunc          # or debunc_prompt; the same thing
  n_agents: 3
  rounds: 2
  debunc:
    temperature: 1.0    # decoding only; uncertainty is read before these apply
    top_k: 50
    top_p: 1.0
    repetition_penalty: 1.0
    min_new_tokens: 2
```

---

## What maps to what

| Paper / repo | Here |
|---|---|
| §3.1 Mean Token Entropy | `uncertainty.mean_token_entropy` |
| §3.2 uncertainty → 1-10 confidence | `confidence.confidences_from_uncertainties` |
| `debate/gen_utils.py::unc_to_confidence` | `confidence.confidences_from_uncertainties` |
| `debate/*/common.py` `START_PREFIX_PROMPT`, `END_PREFIX_NO_CONF` | `prompts.py` |
| `debate/*/eval_*.py::parse_answer` | `answers.py` |
| `mmlu_prompt.py` / `gsm_prompt.py` / `truth_prompt.py` loops | `runner.py` |

### The confidence conversion

With `c_i = 1/u_i` over `n` agents:

```
s_i = c_i / sum_j(c_j) * (5n - 1) + 1/n
```

which sums to `5n`, so the mean confidence is 5 whatever scale the metric is
on. Clamped to `[1, 10]` and rounded. `tests/test_debunc.py` checks this against
a transcription of the authors' `unc_to_confidence`, including its hard-coded
`14 = 5n - 1` for three agents.

### Where the uncertainty comes from

Per-token scores are collected by a logits processor rather than by asking
`generate` for `output_logits=True`. Same values — transformers merges
caller-supplied processors into the chain *before* it appends the sampling
warpers, so the scores seen are the model's own distribution rather than one
already narrowed by temperature, top-k or top-p. This mirrors LM-Polygraph's
`_ScoresProcessor`, which is what the reference measures. The reason is memory:
retaining full logits for 1k generated tokens over a 150k vocabulary is ~600 MB
per call. `tests/test_whitebox_hf.py` asserts the distribution really is
unfiltered, and the backend logs a warning if a future transformers release
changes that ordering.

---

## Other deviations from the reference

1. **`debate.rounds` counts rounds *after* the first answer.** The paper's
   "3 agents, 3 rounds" is one independent answer plus two debate turns, so
   `n_agents: 3, rounds: 2`. `rounds` means the same thing for every method in
   this harness, which matters more than matching the paper's phrasing; runs
   record `debunc_total_rounds: 3` so the two readings cannot be confused.

2. **Answer-letter instructions follow the question's own option count.** The
   reference hard-codes `one of ABCD` for MMLU. MMLU-Pro has up to ten options,
   so the instruction and the answer regex are built from the example's choices.
   For a four-option question the prompt is byte-identical to the reference's.

3. **MATH-500 has no reference prompt**, since DebUnc does not evaluate it. It
   reuses the free-form (GSM8k) shape with a mathematical answer instruction,
   and its answers go through the task's LaTeX normaliser. DebUnc's Arithmetic
   benchmark has no counterpart here and is not ported.

4. **The dependency on LM-Polygraph is dropped.** The reference vendors a copy
   of the library; mean token entropy is a handful of lines and is implemented
   directly against the same definition.

### Reference quirks preserved on purpose

Faithfully reproduced even though they look like bugs, because changing them
would change the numbers:

- **Answer extraction takes the first `Answer:` in a response**, not the last,
  even though the prompt asks for it on the last line (`re.search` in
  `eval_*.py`). A model that mentions "Answer: B" while reasoning and concludes
  C is scored as B.
- **An unparseable response votes.** In `compute_accuracy` a failed parse enters
  the majority vote as its own bucket and can win it, making the example wrong.
  Here it enters as `""` with the same effect, and is counted in
  `parse_failures`.
- **The leading space in `"\n\n One agent solution: ..."`** is in the reference
  prompt and is kept.
- **`min_new_tokens=2`** comes from LM-Polygraph's generation defaults.

---

## What is logged

DebUnc rows use the same schema as PEAR rows, so accuracy, cost and transcript
tooling need no special case. Differences worth knowing before analysing them:

- **`method` is `debunc`, not `pear`.** The *adapter* is `pear` — these runs are
  written by this harness in its log format — but the method is not, and
  pooling them would have a figure compare PEAR against itself. See
  `analysis/common.py::method_for_mode`.
- **`confidence_history` is on DebUnc's 1-10 scale**, derived from token
  entropy, not PEAR's self-reported 1-5. Never pool the two scales.
- **No routing rows.** DebUnc's graph is a fixed clique with no routing
  decision, so no `topology` events are emitted and `routing.jsonl` gets nothing
  from these conditions. `validate_runs.py` reports the absence as a warning;
  that is the correct outcome, not a missing log.
- **`influence_history` is empty.** DebUnc has no accumulated influence. Empty
  rather than zeroed: a column of zeros reads as "measured, found to be nothing".
- Per-response trace events (`event: "debunc_response"`) carry the answer, its
  correctness and the measured uncertainty, which is what an
  uncertainty-quality analysis (the paper's figure 4 AUROC axis) needs.

---

## Results to expect

From the paper, Mistral-7B, 100 questions per benchmark, 5 repeats (table 1),
for the two rows this port covers:

| metric | method | MMLU-0 | GSM8k | TruthfulQA | average |
|---|---|---|---|---|---|
| — | Standard (not ported) | 0.52 | 0.51 | 0.60 | 0.53 |
| Entropy | Prompt | 0.52 | 0.54 | 0.60 | 0.54 |

Confidence in Prompt is the *weakest* of the paper's three communication
methods — its accuracy-vs-AUROC slope is 0.17 against 0.59 for Attention-All.
Expect a small effect. Note also that these are MMLU (4 options), not the
MMLU-Pro (10 options) this harness ships, so the numbers are not directly
comparable; expect lower accuracy here.
