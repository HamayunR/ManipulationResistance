"""Parsed, validated settings for a DebUnc condition.

Everything the mechanism can be told to do goes through :class:`DebUncConfig`,
which is built once per example *before* any generation happens, so a typo in a
YAML config fails immediately instead of after an hour of decoding.

The paper explores two axes -- how confidence is communicated, and which
uncertainty metric produces it. This port implements one point on each:
**Confidence in Prompt** with **Mean Token Entropy**, the combination that needs
nothing beyond a whitebox backend and a single generation per turn. There is no
knob for the others because there is no implementation behind them; see
``README.md`` for what was left out and why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

#: ``debate.mode`` values that select this baseline. ``debunc_prompt`` names the
#: paper's communication method and is kept as a synonym, because a config that
#: says only ``debunc`` does not say *which* DebUnc.
DEBUNC_MODES: Tuple[str, ...] = ("debunc", "debunc_prompt")


def is_debunc_mode(mode: str) -> bool:
    """Whether ``debate.mode`` selects the DebUnc baseline."""
    return str(mode or "") in DEBUNC_MODES


@dataclass(frozen=True)
class DebUncConfig:
    """Decoding settings for one DebUnc condition.

    The defaults are LM-Polygraph's ``GenerationParameters``, which is what the
    reference implementation generates under, combined with the paper's stated
    temperature of 1.

    Attributes
    ----------
    temperature, top_k, top_p, repetition_penalty:
        Sampling parameters. They do not affect the uncertainty measurement,
        which reads the model's distribution before they are applied.
    min_new_tokens:
        From LM-Polygraph's generation defaults; forbids an immediate
        end-of-sequence, which would leave nothing to measure.
    """

    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    min_new_tokens: int = 2

    @classmethod
    def from_debate_cfg(cls, debate_cfg: Mapping[str, Any]) -> "DebUncConfig":
        """Build from a ``debate:`` block, validating the nested ``debunc`` one."""
        mode = str(debate_cfg.get("mode", ""))
        if not is_debunc_mode(mode):
            raise ValueError(
                f"{mode!r} is not a DebUnc mode; expected one of {list(DEBUNC_MODES)}"
            )

        block = debate_cfg.get("debunc") or {}
        if not isinstance(block, Mapping):
            raise ValueError("debate.debunc must be a mapping")

        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(block) - known)
        if unknown:
            # Silently ignoring a key is how a run ends up describing a
            # mechanism it never ran -- a stale `communication: attention_all`
            # would otherwise look honoured.
            raise ValueError(
                f"unknown debate.debunc key(s) {unknown}; this port implements "
                f"confidence-in-prompt with mean token entropy only. Known "
                f"keys: {sorted(known)}"
            )

        return cls(
            temperature=float(block.get("temperature", 1.0)),
            top_k=int(block.get("top_k", 50)),
            top_p=float(block.get("top_p", 1.0)),
            repetition_penalty=float(block.get("repetition_penalty", 1.0)),
            min_new_tokens=int(block.get("min_new_tokens", 2)),
        )

    def to_metadata(self) -> dict:
        """Flat, loggable description of the condition."""
        return {
            # Fixed for this port, but recorded rather than assumed: a row that
            # does not say which mechanism produced it cannot be pooled safely
            # if another is ever added.
            "debunc_communication": "prompt",
            "debunc_uncertainty": "mean_token_entropy",
            "debunc_temperature": self.temperature,
            "debunc_top_k": self.top_k,
            "debunc_top_p": self.top_p,
            "debunc_repetition_penalty": self.repetition_penalty,
            "debunc_min_new_tokens": self.min_new_tokens,
        }


__all__ = ["DEBUNC_MODES", "DebUncConfig", "is_debunc_mode"]
