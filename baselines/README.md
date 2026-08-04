# Baselines

Reimplementations of published multi-agent methods, so this repository's method
can be compared against them on the same benchmarks, the same models and the
same scoring.

| package | paper |
|---|---|
| [`debunc/`](debunc/) | DebUnc: Improving Large Language Model Agent Communication With Uncertainty Metrics — Yoffe, Amayuelas & Wang, 2024 ([arXiv:2407.06426](https://arxiv.org/abs/2407.06426)) |

Each baseline is one package. They live here rather than as modes inside
`runner/experiment.py` because they are not variants of PEAR: they have their
own debate protocol, their own prompts and their own notion of confidence.
Folding them into PEAR's two-phase loop would quietly make them share PEAR's
mechanism, which is the thing the comparison is supposed to isolate.

## Running one

A baseline is selected by `debate.mode` and dispatched from `run_one`:

```bash
python main.py --config configs/debunc.yaml
```

Its rows land in the same `results.jsonl`, `trace.jsonl` and `transcripts.jsonl`
as any other condition, tagged with `mode`, so a config can mix a baseline and a
PEAR condition and compare them directly.

## Adding another paper

1. **Read the paper and the authors' code.** Where they disagree, the code is
   what produced the numbers; note the disagreement in your README.
2. **Create `baselines/<name>/`** with the pieces separated so each is testable
   on its own: `config.py` (parsed and validated settings), `prompts.py`
   (verbatim templates), `answers.py` (their answer extraction), `runner.py`
   (the debate loop), plus whatever the method is actually about.
3. **Expose `run_<name>_example(task, example, *, debate_cfg, llm, seed,
   perm_seed, judge_llm=None) -> dict`** returning exactly the row
   `runner.experiment.run_one` returns. Fill in what the method has; leave what
   it does not have empty rather than zeroed. A column of zeros reads as
   "measured and found to be nothing".
4. **Dispatch from `run_one`**, one `if` at the top, and add the mode to
   `main.py`'s `--mode` choices.
5. **Score through `data.tasks.Task`.** Parse answers however the paper does —
   extraction is part of the method — but let `Task.score` decide correctness,
   so "accuracy" means one thing across the repository.
6. **Never import from the top-level `prompts.py`.** That module is PEAR's; its
   JSON schema and 1-5 confidence rubric are part of PEAR's mechanism.
7. **Write `README.md`** with the mapping from paper sections and reference
   files to your modules, an explicit list of deviations, and the reference
   quirks you preserved on purpose.
8. **Test against the reference, not against yourself.** Transcribe the
   authors' formulas into the test file and assert equality; check prompt
   strings against their constants. A test that only pins your own behaviour
   will happily pin a misreading.
9. **Add config knobs only where the paper has a choice.** Every invented knob
   is a way for a run to describe a mechanism no paper ever evaluated.

### Backend capabilities

A method may need more from a model than a completion string. DebUnc needs
full-vocabulary token distributions, which live in `models/whitebox.py` as an
optional capability layered on `BaseLLM` and implemented only by the Hugging
Face backend.

Declare a capability the same way if you need one: an interface a backend either
implements or does not, checked up front with a message that names the fix.
The failure mode to avoid is a method that quietly degrades to a weaker
mechanism when the backend cannot do what the paper did.
