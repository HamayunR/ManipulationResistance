"""Whitebox backend capability: the model's own per-token scores.

Most of this harness only needs a completion string, which is why
:class:`models.base.BaseLLM` exposes nothing else. Some methods need more than
a chat API can give: DebUnc (Yoffe et al., 2024) measures an agent's confidence
from the token probabilities of its own response, which requires the
next-token distribution at every generated position.

That is declared here as an *optional* capability layered on top of
``BaseLLM``. A backend that cannot provide it simply does not subclass
:class:`WhiteboxLLM`; a method that needs it then fails with a clear message
instead of quietly running a different mechanism.

One convention matters and is easy to get wrong:
:attr:`WhiteboxGeneration.logits_are_unprocessed` records whether the per-token
scores came from the raw model head or from logits that sampling parameters had
already modified. An uncertainty metric computed on top-k filtered logits is a
different quantity from one computed on the full distribution, and reporting
the two under one name is a silent error.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence

from models.base import BaseLLM, Generation


@dataclass
class WhiteboxGeneration(Generation):
    """A completion plus the per-token scores the model produced with it.

    Attributes
    ----------
    token_ids:
        Generated token ids, including a trailing end-of-sequence token when
        the model emitted one. ``text`` excludes that token.
    token_logprobs:
        ``log p(t_i | t_<i, x)`` for each generated token, from the
        distribution described by :attr:`logits_are_unprocessed`.
    token_entropies:
        Full-vocabulary entropy in nats at each generated position.
    logits_are_unprocessed:
        ``True`` when the scores come from the model head before sampling
        parameters (temperature, top-k, top-p) touched them. Uncertainty
        metrics assume this; see the module docstring.
    """

    token_ids: List[int] = field(default_factory=list)
    token_logprobs: List[float] = field(default_factory=list)
    token_entropies: List[float] = field(default_factory=list)
    logits_are_unprocessed: bool = True

    @property
    def mean_token_entropy(self) -> float:
        """Mean entropy across generated tokens; ``nan`` when nothing was generated."""
        if not self.token_entropies:
            return float("nan")
        return float(sum(self.token_entropies) / len(self.token_entropies))


class WhiteboxLLM(BaseLLM, abc.ABC):
    """A backend that exposes the model's own token-level scores.

    ``chat_generate`` takes a whole conversation rather than one prompt string,
    because the methods that need this capability run multi-turn debates where
    each agent keeps its full history.
    """

    @abc.abstractmethod
    def chat_generate(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        min_new_tokens: int = 0,
        seed: Optional[int] = None,
    ) -> WhiteboxGeneration:
        """Continue ``messages`` and report the model's own per-token scores.

        Parameters
        ----------
        seed:
            Seeds the sampler for this call, so a run is reproducible without
            the caller having to reach into the backend's RNG.
        """
        raise NotImplementedError

    @property
    def tokenizer(self) -> Any:
        """The backend's tokenizer, for callers that need to decode tokens."""
        raise NotImplementedError


__all__ = [
    "WhiteboxGeneration",
    "WhiteboxLLM",
]
