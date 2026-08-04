# PEAR: Permutation-Equivariant Adaptive Routing Multi-Agent Debate

PEAR is an experiment harness for **Permutation-Equivariant Adaptive Routing Multi-Agent Debate**. It studies multi-agent reasoning systems where agents first produce independent answers and then revise them through sparse, dynamically routed critiques.

The current method uses a two-phase debate loop:

1. **Answer phase**: each agent proposes or updates an answer with a confidence score.
2. **Critique phase**: each agent receives a fixed number of critiques from other agents. For `pear_full`, the routing graph is selected using state-aware signals rather than a fixed topology.

The full PEAR routing objective combines three components:

- **Targeted cross-answer routing**: prioritize high-confidence agents with different answers as sources for lower-confidence targets.
- **Influence balancing**: avoid repeatedly over-exposing the group to one historically influential source.
- **Low-confidence filtering**: reduce routing exposure from low-confidence sources.

This repository includes fixed-topology baselines, CoT / CoT-SC baselines, random sparse routing controls, ablations, robustness experiments, local vLLM inference, and OpenAI-compatible closed-model inference.

---

## Repository Structure

The repository root is the source directory. `main.py` is the canonical CLI entry point, and each subdirectory is importable as a top-level package. Generated run directories and paper notes are intentionally not expanded here.

```text
PEAR/
|- main.py                         # CLI entry point; loads YAML config and calls runner.run_experiment
|- prompts.py                      # all LLM-facing prompt templates, confidence rubric, robustness prompts
|- requirements.txt                # Python dependencies, including vLLM / transformers / API clients
|- .env.example                    # template for API keys and OpenAI-compatible base URLs
|- configs/                        # agents/debate/dataset/replication/logging settings
|- core/
|  |- state.py                     # shared debate state schema and mode/type definitions
|  |- topology.py                  # topology construction, routing scores, random k-regular logic
|  `- views.py                     # per-agent visibility/view helpers
|- nodes/
|  |- agent_runner.py              # calls LLM agents for initial answers and updates
|  |- aggregator.py                # final answer aggregation, e.g. majority vote / judge
|  |- scorer.py                    # example-level scoring node
|  |- topology_scheduler.py        # topology/routing schedule helpers
|  `- turn_scheduler.py            # turn order / round scheduling helpers
|- graph/
|  `- builder.py                   # LangGraph StateGraph construction for PEAR execution
|- runner/
|  `- experiment.py                # high-level experiment loop, condition sweeps, summaries, transcripts
|- models/
|  |- base.py                      # BaseLLM and Generation abstractions
|  |- model.py                     # OpenAI-compatible, Anthropic, HF, MLX and vLLM backend implementations
|  |- factory.py                   # loads configs/models.yaml and instantiates the selected backend
|  `- whitebox.py                  # optional capability: the model's own token-level scores
|- baselines/
|  `- debunc/                      # DebUnc (Yoffe et al. 2024) reimplemented as a comparison baseline
|- data/
|  |- loaders.py                   # dataset loading utilities
|  |- tasks.py                     # dataset adapters: formatting, answer parsing, scoring
|  |- gsm8k/                       # GSM8K local JSON files
|  |- mmlu_pro/                    # MMLU-Pro local JSON files
|  |- math500/                     # MATH-500 local JSON files
|  |- truthful_qa/                 # TruthfulQA local JSON files
|- metrics/
|  |- scorers.py                   # accuracy and task-level scoring helpers
|  |- diagnostics.py               # W2R/R2W, routing, confidence, influence diagnostics
|  `- stability.py                 # stability / tail-risk / influence-distribution metrics
|- utils/
|  |- budget.py                    # optional call/token budget accounting
|  |- json_output.py               # reading agent JSON, incl. LaTeX-safe string repair
|  |- logging.py                   # run directory creation, logger setup, timestamp handling
|  |- seed.py                      # reproducibility helpers
|  `- tracing.py                   # JSONL trace writing
|- scripts/
`- analysis/
```

---

## Installation

The project can be run with a regular Python environment, but local open-model experiments should use a dedicated vLLM environment. The scripts default to `.venv-vllm/bin/python`.

### Create the vLLM environment

Using `uv`:

```bash
uv venv .venv-vllm --python 3.12
source .venv-vllm/bin/activate
uv pip install -r requirements.txt
```

Or using standard `venv` / `pip`:

```bash
python3.12 -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


For closed-source or OpenAI-compatible APIs, create `.env` from the template:

```bash
cp .env.example .env
# Fill OPENAI_API_KEY / OPENAI_BASE_URL as needed.
```



---

## Quickstart

Run the default config:

```bash
python main.py --config configs/default.yaml
```

Override common fields from the CLI:

```bash
python main.py \
  --config configs/main_large.yaml \
  --model qwen2.5-7b-vllm \
  --dataset mmlu_pro \
  --num-examples 50
```

Run with explicit seeds:

```bash
python main.py \
  --config configs/main_large.yaml \
  --seeds 1 2 3 \
  --perm-seeds 10
```

Run the parameterized vLLM shell entry point:

```bash
scripts/run_vllm.sh qwen7b
scripts/run_vllm.sh llama
scripts/run_vllm.sh gemma
scripts/run_vllm.sh qwen3
scripts/run_vllm.sh open4
```

Useful shell overrides:

```bash
VLLM_DATASETS="gsm8k mmlu_pro" \
VLLM_NUM_EXAMPLES=200 \
PEAR_SEEDS="1 2 3" \
PEAR_PERM_SEEDS="10" \
scripts/run_vllm.sh qwen7b
```

Backward-compatible wrappers are also kept:

```bash
scripts/run_qwen_vllm.sh
scripts/run_llama_vllm.sh
scripts/run_gemma_vllm.sh
scripts/run_qwen3_main_vllm.sh
```

---

## Configuration

`configs/default.yaml` contains the main knobs. Important fields:

```yaml
agents:
  model: qwen2.5-7b-vllm

debate:
  n_agents: 5
  rounds: 3
  base_topology: k_regular
  k_regular_degree: 2
  mode: pear_full
  agg_mode: majority_vote
  max_tokens_per_call: 512
  mc_permutations: 100
  routing_temperature: 0.7
  alpha_targeted_cross: 0.2
  alpha_influence: 0.7
  alpha_low_confidence: 0.7
  normalize_routing_terms: true
  low_confidence_threshold: 3
  targeted_cross_source_confidence_min: 4
  targeted_cross_target_confidence_max: 3
  influence_beta: 0.6

dataset:
  name: mmlu_pro
  num_examples: 100

replication:
  seeds: [0]
  agent_perm_seeds: [10]
```

`configs/main_large.yaml` defines the main comparison conditions:

| condition | meaning |
|---|---|
| `cot` | single-agent chain-of-thought baseline |
| `cot_sc` | self-consistency over multiple independent agents/samples |
| `fixed_clique` | fixed fully connected debate |
| `fixed_star` | fixed hub-and-spoke debate |
| `fixed_chain` | fixed sequential chain debate |
| `fixed_ring` | fixed local-neighbor ring debate |
| `random_k_regular` | random sparse k-regular routing baseline |
| `pear_full` | full adaptive routing method |

`configs/ablation.yaml` isolates PEAR routing components:

| condition | targeted cross | influence balancing | low-confidence filtering |
|---|---:|---:|---:|
| `pear_targeted_cross` | yes | no | no |
| `pear_influence` | no | yes | no |
| `pear_low_confidence` | no | no | yes |
| `pear_targeted_influence` | yes | yes | no |
| `pear_targeted_low_confidence` | yes | no | yes |
| `pear_influence_low_confidence` | no | yes | yes |
| `pear_full` | yes | yes | yes |

---

## Baselines from other papers

`baselines/` holds reimplementations of published multi-agent methods, so PEAR
can be compared against them on the same benchmarks, models and scoring. They
are selected with `debate.mode` like any other condition and write the same
result rows, so a config can mix them with PEAR conditions.

| package | paper | modes |
|---|---|---|
| [`baselines/debunc`](baselines/debunc/) | DebUnc (Yoffe, Amayuelas & Wang, 2024) | `debunc`, `debunc_prompt` |

DebUnc measures each agent's confidence from its own token probabilities rather
than asking for it, then states that confidence next to the agent's response in
the next round's prompt. It needs a whitebox backend (`provider: hf`). Its
README lists the parts of the paper this port does not implement.

```bash
python main.py --config configs/debunc.yaml
```

See [`baselines/README.md`](baselines/README.md) for the contract a baseline
must satisfy, and each package's `README.md` for its mapping to the paper and
its list of deviations.

---

## Datasets

All datasets are loaded from local JSON files under `data/`; the runner does not download data at runtime.

| name | source files | answer format |
|---|---|---|
| `mmlu_pro` | `data/mmlu_pro/{test,validation}.json` | A-J multiple choice |
| `gsm8k` | `data/gsm8k/{test,train}.json` | numeric exact match |
| `truthful_qa` | `data/truthful_qa/mc_task.json` | multiple choice letter |
| `math_500` | `data/math500/test.json` | math answer exact match |

`data/tasks.py` normalizes each dataset into a shared `Example` format and provides benchmark-specific `format_question`, `parse_answer`, and `score` methods.

---

## Models

Models are registered in `configs/models.yaml`. Each entry is selected by `agents.model` or `--model`.

Common local vLLM entries include:

| model key | backend |
|---|---|
| `qwen2.5-7b-vllm` | vLLM |
| `llama-3.1-8b-vllm` | vLLM |
| `gemma-3-12b-vllm` | vLLM |
| `qwen2.5-14b-vllm` | vLLM |
| `qwen3-30b-a3b-local-vllm` | vLLM |

Closed / API models can use the OpenAI-compatible backend through `OPENAI_API_KEY` and `OPENAI_BASE_URL`.

---

## Running Experiments

Main open-model experiment:

```bash
scripts/run_main_large_vllm.sh all
```

Parameterized vLLM runner:

```bash
scripts/run_vllm.sh qwen7b llama gemma qwen3
```

Routing ablation:

```bash
scripts/run_ablation_vllm.sh all
```

Random k-regular baseline:

```bash
scripts/run_random_k_regular_vllm.sh open4
```


Closed-model main experiment:

```bash
scripts/run_closed_main.sh gpt
scripts/run_closed_main.sh claude
```


## Metrics

The primary metric is accuracy. The code also records diagnostics that are useful for understanding debate dynamics:

| metric | meaning |
|---|---|
| `accuracy` | final answer correctness |
| `w2r_rate` | wrong-to-right answer transition rate |
| `r2w_rate` | right-to-wrong answer transition rate |
| `answer_entropy` | diversity of agent answers |
| `influence_entropy` | concentration or balance of influence |
| `targeted_cross_rate` | frequency of targeted high-confidence dissent routing |
| `critique_acceptance_rate` | how often critiques are adopted |
| `parse_failures` | model outputs no JSON object could be read from |
| `json_repairs` | outputs that only decoded after their string literals were repaired |

`parse_failures` is worth checking before anything else. A lost payload does not
abort the round: the runner falls back to scanning the raw text for an answer
and to a **default confidence of 3**, so an example with many parse failures was
routed on a constant rather than on what the agents reported. `json_repairs`
counts the outputs the LaTeX-safe repair in `utils/json_output.py` recovered;
those are intact, but their text is not byte-for-byte what the model emitted.

Analysis scripts live in `analysis/` and generated figures/reports should be written under `analysis/figures/` or dedicated report subfolders.


---



## Citation

```bibtex
@article{pear2026,
  title={PEAR: Permutation-Equivariant Adaptive Routing Multi-Agent Debate},
  author={Feng, Yang and Xu, Ziwei and Hu, Xia and He, Fengxiang},
  year={2026}
}
```

## License

See `LICENSE`.
