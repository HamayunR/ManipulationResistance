"""The DebUnc debate loop.

Structure (paper section 3, figure 2), unchanged from the reference
implementation:

* every agent keeps its own conversation with the model and sees its whole
  history, so round 3 still has round 1 in context;
* round 1 is answered independently; every later round appends one user message
  quoting *all* the other agents' previous responses, so the communication graph
  is a fully connected clique with no routing decision to make;
* after each round each agent's uncertainty is measured from the token
  probabilities of its own response, converted to a 1-10 confidence, and stated
  next to that response in the next round's prompt; and
* the final answer is a majority vote over the last round's parsed answers.

The result row is the one :func:`runner.experiment.run_one` produces, so
tracing, transcripts and the analysis layer treat a DebUnc condition like any
other. Fields that describe machinery DebUnc does not have -- routing,
accumulated influence -- are empty rather than filled with lookalike values.
"""

from __future__ import annotations

import math
import zlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from baselines.debunc.answers import parse_answer
from baselines.debunc.config import DebUncConfig
from baselines.debunc.confidence import confidences_from_uncertainties
from baselines.debunc.prompts import build_debate_message, prompts_for
from baselines.debunc.uncertainty import UNCERTAINTY_METRIC, mean_token_entropy
from data.tasks import Task
from metrics.diagnostics import entropy, trajectory_event_rates
from models.base import BaseLLM
from models.whitebox import WhiteboxLLM
from nodes.aggregator import _majority_vote
from utils.budget import Budget
from utils.logging import get_logger
from utils.seed import seeded_rng

_log = get_logger("baselines.debunc")

#: Upper bound for a derived per-call sampler seed.
_SEED_SPACE = 2**31


def run_debunc_example(
    task: Task,
    example,
    *,
    debate_cfg: Mapping[str, Any],
    llm: Optional[BaseLLM],
    seed: int,
    perm_seed: int,
    judge_llm: Optional[BaseLLM] = None,
) -> Dict[str, Any]:
    """Run one example through DebUnc; return a runner-compatible result row.

    ``debate.rounds`` keeps its harness meaning: the number of debate rounds
    *after* the independent first answer. The paper's "3 agents, 3 rounds"
    setup is therefore ``n_agents: 3, rounds: 2``, which is recorded as
    ``debunc_total_rounds: 3`` so the two conventions cannot be confused.

    ``perm_seed`` is accepted and recorded but unused: DebUnc has no agent
    relabelling, so there is nothing for it to vary.
    """
    config = DebUncConfig.from_debate_cfg(debate_cfg)
    n_agents = int(debate_cfg.get("n_agents", 3))
    debate_rounds = int(debate_cfg.get("rounds", 2))
    max_tokens = int(debate_cfg.get("max_tokens_per_call", 1024))
    if n_agents < 2:
        raise ValueError(f"DebUnc needs at least two agents, got {n_agents}")

    agg_mode = str(debate_cfg.get("agg_mode", "majority_vote"))
    if agg_mode != "majority_vote":
        _log.warning(
            "agg_mode=%s is not part of DebUnc, which decides by majority vote "
            "after the final round; using majority_vote.",
            agg_mode,
        )

    model = _require_whitebox(llm)
    prompts = prompts_for(task, example)
    is_correct = lambda answer: bool(task.score(answer, example))
    # Sampler seeds are drawn from a stream keyed on the run coordinates *and*
    # the example, so two examples under one seed do not share a decode noise
    # sequence. Draw order is fixed (round, then agent), so a rerun of the same
    # coordinates reproduces the debate.
    rng = seeded_rng(
        zlib.crc32(f"debunc:{seed}:{perm_seed}:{example.id}".encode("utf-8"))
    )
    budget = Budget()

    conversations: List[List[Dict[str, str]]] = [
        [{"role": "user", "content": prompts.initial}] for _ in range(n_agents)
    ]
    messages: List[Dict[str, Any]] = []
    trace_events: List[Dict[str, Any]] = [
        {
            "event": "init",
            "n_agents": n_agents,
            "rounds": debate_rounds,
            "mode": str(debate_cfg.get("mode", "debunc")),
            "seed": int(seed),
            "perm_seed": int(perm_seed),
            "debunc_total_rounds": debate_rounds + 1,
            **config.to_metadata(),
        }
    ]

    answer_history: List[Dict[int, str]] = []
    confidence_history: List[Dict[int, int]] = []
    parse_failures = 0
    uncertainty_records: List[Tuple[float, bool]] = []

    previous_responses: List[str] = []
    previous_confidences: Tuple[int, ...] = ()

    for round_idx in range(debate_rounds + 1):
        round_responses: List[str] = []
        round_answers: Dict[int, str] = {}
        round_uncertainties: List[float] = []

        for agent_index in range(n_agents):
            agent_id = agent_index + 1
            if round_idx > 0:
                others = [j for j in range(n_agents) if j != agent_index]
                conversations[agent_index].append(
                    {
                        "role": "user",
                        "content": build_debate_message(
                            prompts,
                            [previous_responses[j] for j in others],
                            [previous_confidences[j] for j in others],
                        ),
                    }
                )

            generation = model.chat_generate(
                conversations[agent_index],
                max_tokens=max_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                min_new_tokens=config.min_new_tokens,
                seed=rng.randrange(_SEED_SPACE),
            )
            budget.charge(
                calls=1,
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
            )
            conversations[agent_index].append(
                {"role": "assistant", "content": generation.text}
            )

            answer = parse_answer(generation.text, task=task, example=example)
            parse_failures += int(not answer)
            uncertainty = mean_token_entropy(generation)

            round_responses.append(generation.text)
            round_answers[agent_id] = answer
            round_uncertainties.append(uncertainty)
            if math.isfinite(uncertainty):
                uncertainty_records.append((uncertainty, is_correct(answer)))

            messages.append(
                {
                    "speaker": agent_id,
                    "round": round_idx,
                    "phase": "initial_answer" if round_idx == 0 else "answer_update",
                    "content": generation.text,
                }
            )
            trace_events.append(
                {
                    "event": "debunc_response",
                    "round": round_idx,
                    "agent_id": agent_id,
                    "seed": int(seed),
                    "perm_seed": int(perm_seed),
                    "answer": answer,
                    "correct": is_correct(answer),
                    "uncertainty": uncertainty,
                    "uncertainty_metric": UNCERTAINTY_METRIC,
                    "prompt_tokens": generation.prompt_tokens,
                    "completion_tokens": generation.completion_tokens,
                    "logits_are_unprocessed": generation.logits_are_unprocessed,
                }
            )

        confidences = confidences_from_uncertainties(_sanitise(round_uncertainties))
        trace_events.append(
            {
                "event": "debunc_confidence",
                "round": round_idx,
                "seed": int(seed),
                "perm_seed": int(perm_seed),
                "uncertainty_metric": UNCERTAINTY_METRIC,
                "uncertainties": {
                    str(i + 1): round_uncertainties[i] for i in range(n_agents)
                },
                # What the next round's prompt states next to each response.
                "prompt_confidence": {
                    str(i + 1): confidences[i] for i in range(n_agents)
                },
            }
        )

        answer_history.append(dict(round_answers))
        confidence_history.append(
            {i + 1: confidences[i] for i in range(n_agents)}
        )
        previous_responses = round_responses
        previous_confidences = confidences

    final_candidates = dict(answer_history[-1])
    decision = _majority_vote(final_candidates)
    correct = bool(task.score(decision or "", example))

    diagnostics = trajectory_event_rates(answer_history[0], final_candidates, is_correct)
    diagnostics.update(
        {
            "answer_entropy_initial": entropy(list(answer_history[0].values())),
            "answer_entropy_final": entropy(list(final_candidates.values())),
            "mean_token_entropy": _mean([u for u, _ in uncertainty_records]),
            "uncertainty_mean_correct": _mean(
                [u for u, ok in uncertainty_records if ok]
            ),
            "uncertainty_mean_incorrect": _mean(
                [u for u, ok in uncertainty_records if not ok]
            ),
            "parse_failures": float(parse_failures),
        }
    )

    trace_events.append(
        {
            "event": "aggregate",
            "agg_mode": "majority_vote",
            "decision": decision,
            "candidates": final_candidates,
        }
    )
    trace_events.append(
        {
            "event": "score",
            "prediction": decision,
            "gold": example.answer,
            "correct": correct,
        }
    )

    return {
        "example_id": example.id,
        "decision": decision,
        "correct": correct,
        "parse_failures": int(parse_failures),
        "n_messages": len(messages),
        "budget": budget.to_dict(),
        "trace_events": trace_events,
        "messages": messages,
        "final_candidates": final_candidates,
        "seed": int(seed),
        "perm_seed": int(perm_seed),
        "answer_history": answer_history,
        "confidence_history": confidence_history,
        # DebUnc has no routing and no accumulated influence. Empty, not zeroed:
        # a column of zeros would read as "measured and found to be nothing".
        "influence_history": [],
        "diagnostics": diagnostics,
    }


# ----------------------------------------------------------------- helpers --
def _sanitise(uncertainties: Sequence[float]) -> List[float]:
    """Replace unmeasurable uncertainties with the round's worst measured one.

    An agent that generated nothing has no distribution to score. Treating it as
    the least confident agent in the round is the conservative reading, and it
    keeps the conversion from having to handle a hole.
    """
    finite = [u for u in uncertainties if math.isfinite(u)]
    fallback = max(finite) if finite else 1.0
    if len(finite) != len(uncertainties):
        _log.warning(
            "%d of %d uncertainties were unmeasurable this round; substituting "
            "the round's highest (%.4f)",
            len(uncertainties) - len(finite),
            len(uncertainties),
            fallback,
        )
    return [u if math.isfinite(u) else fallback for u in uncertainties]


def _require_whitebox(llm: Optional[BaseLLM]) -> WhiteboxLLM:
    if not isinstance(llm, WhiteboxLLM):
        raise TypeError(
            "DebUnc measures uncertainty from token probabilities, so it needs "
            f"a whitebox backend. {type(llm).__name__} is not one; use "
            "provider: hf in configs/models.yaml."
        )
    return llm


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


__all__ = ["run_debunc_example"]
