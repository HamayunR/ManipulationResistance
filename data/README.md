# Benchmark data

Five benchmarks are wired up. Each is read from a **frozen local corpus** —
one JSONL file per split, plus a manifest recording where it came from.

| `dataset.name` | Benchmark | Answer type | Default split | Items | Source | Licence |
| --- | --- | --- | --- | --- | --- | --- |
| `gsm8k` | GSM8K | number | `test` | 1,319 | `openai/gsm8k` (`main`) | MIT |
| `math_500` | MATH-500 | LaTeX | `test` | 500 | `HuggingFaceH4/MATH-500` | MIT |
| `mmlu_pro` | MMLU-Pro | letter (A–J) | `test` | 12,032 | `TIGER-Lab/MMLU-Pro` | MIT |
| `truthful_qa` | TruthfulQA MC1 | letter | `validation` | 817 | `truthfulqa/truthful_qa` (`multiple_choice`) | Apache-2.0 |
| `gpqa` | GPQA Diamond | letter (A–D) | `diamond` | 198 | `Idavidrein/gpqa` | CC-BY-4.0, **gated** |

## Fetching

```bash
python scripts/fetch_datasets.py --list           # catalogue
python scripts/fetch_datasets.py all              # the four public benchmarks
python scripts/fetch_datasets.py gsm8k --split train
python scripts/fetch_datasets.py mmlu_pro --limit 500      # pilot-sized subset
python scripts/fetch_datasets.py --verify         # checksum every fetched corpus
                                                 # (runs do this automatically too)
```

GPQA is gated. Accept the conditions on its dataset page, then:

```bash
export HF_TOKEN=hf_...
python scripts/fetch_datasets.py gpqa --split diamond
```

The fetcher downloads the parquet conversion over plain HTTP — no `datasets`
library is needed on the machine that runs it, only pandas and pyarrow.

Fetched corpora are **gitignored**: they are reproducible from the command
above, MMLU-Pro is large, and GPQA must not be redistributed. Run the fetch once
per checkout (and on each cluster node, or fetch to shared storage and point
`paths.data_dir` at it).

## The corpus format

`data/<name>/<split>.jsonl`, one object per line:

```json
{"id": "gsm8k-test-00000",
 "question": "Janet's ducks lay 16 eggs per day...",
 "answer": "18",
 "choices": null,
 "metadata": {"rationale": "..."}}
```

* `answer` is the **option letter** for multiple-choice benchmarks, the value
  or expression for free-form ones.
* `choices` non-empty is what makes the runner treat answers as letters
  (`runner/experiment.py` uses it for the random baseline and the adversary's
  wrong answer). Free-form tasks must leave it `null`.
* `metadata` is provenance and is never used for scoring.

`data/<name>/manifest.json` records the source repo, the resolved commit sha,
the config and upstream split, the row count, the sha256 of the written file,
any `--limit` cap, the option-shuffle seed and the fetch timestamp.

### Integrity is checked automatically

Every run verifies the corpus against that checksum **when it loads the data**,
before any tokens are spent. ~13 ms for the largest corpus. If the file changed
after it was fetched, the run refuses to start and prints the re-fetch command;
`PEAR_SKIP_CORPUS_CHECK=1` overrides it deliberately for one run.

Each run then records what it scored in its `summary.json`:

```json
{"dataset": "gsm8k", "dataset_split": "test",
 "dataset_sha256": "0e39e9ffa05a640e...", "dataset_integrity": "verified"}
```

and `analysis/validate_runs.py` refuses to pool runs of the same split that
scored different corpus bytes (`dataset_corpus_mismatch`).

So there is nothing to remember. `python scripts/fetch_datasets.py --verify` is
the standalone version — useful for checking a machine before a long sweep, or
after copying corpora to shared storage — not a step you have to run.

### Why files rather than a dataset library call

* **Reproducibility.** A run is scored against exact bytes with a checksum.
  Upstream datasets get revised; a silent revision between two runs would move
  a headline number with nothing in the logs to explain it.
* **Frozen option order.** GPQA stores the correct answer in its own column and
  TruthfulQA MC1 always lists it first. Shuffling happens **once**, at fetch
  time, seeded per item — so the gold letter is a property of the data, not of
  whatever code last ran. Verified: the fetched TruthfulQA golds spread across
  A–K rather than sitting on A.
* **Offline runs.** vLLM/cluster nodes need no network and no agreement about
  dataset-library versions.

## Running experiments on them

```bash
python main.py --config configs/default.yaml --dataset gsm8k --num-examples 100
python main.py --config configs/main_large.yaml --dataset mmlu_pro
python main.py --config configs/default.yaml --dataset truthful_qa   # validation split
```

Omit `--split` to get the benchmark's own default. `paths.data_dir` (default
`data`) is the corpus root.

If a corpus is missing, the run fails immediately with the exact fetch command
rather than starting and burning tokens.

## Offline smoke tests: the `sample` split

Every benchmark except GPQA ships a three-item **synthetic** split under
`data/fixtures/<name>/sample.jsonl`:

```bash
python main.py --config configs/default.yaml --dataset gsm8k --split sample
```

The items are invented and structurally identical to the real thing. They exist
so the pipeline can be exercised with no network and no downloads. They are
reachable **only** through the explicit `sample` split, so a real evaluation can
never silently fall back to toy data — and a number produced from them is not a
result. GPQA has none, because its authors ask that items are not republished.

## Answer handling

Parsers live in `data/normalize.py`, keyed by answer *type* rather than by
dataset, so two benchmarks of the same type cannot disagree about what counts
as correct.

| Type | Parsing | Scoring |
| --- | --- | --- |
| `letter` | explicit statements first (`answer is (C)`, `\boxed{C}`, `**C**`), then a lone letter | exact letter match |
| `number` | last number, commas/currency/units stripped | numeric equality, 1e-6 tolerance |
| `math` | last `\boxed{...}` (brace-matched), else the final line | normalised LaTeX match, then numeric |

Two deliberate refusals:

* A letter beyond the option count (`"J"` on a four-option question) parses to
  `""`, not to a wrong option. That surfaces a prompting bug instead of hiding
  it in the error rate.
* `math_equal` does **not** claim symbolic equivalence: `\frac{1}{2}` and `0.5`
  stay different. Doing it properly needs a CAS, and faking it would inflate
  accuracy in a way no one could audit from the logs.

Every parser returns `""` when it can extract nothing — the runner counts that
as a parse failure, so no parser may invent a plausible-looking answer.

## Adding a benchmark

1. Add a `Source` entry to `SOURCES` in `scripts/fetch_datasets.py` with a
   `normalize_*` function mapping upstream rows to `CorpusRecord`s.
2. Add a `LocalCorpusTask` (or `MultipleChoiceTask`) subclass in
   `data/tasks.py` and register it in `TASK_REGISTRY`.
3. Add a prompt template to `prompts.py`.

Nothing in `analysis/` changes: the dataset name is a grouping key there, never
a branch.
