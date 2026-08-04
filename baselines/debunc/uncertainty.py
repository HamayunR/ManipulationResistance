"""Mean Token Entropy, DebUnc's default uncertainty metric (section 3.1).

Reads the model's own next-token distribution, which is why DebUnc needs a
whitebox backend. Higher values mean *more* uncertain.

The paper also evaluates TokenSAR and an Oracle metric. Neither is ported here:
TokenSAR needs a cross-encoder pass per generated token, and the Oracle needs
the gold answer, so it is a ceiling rather than a method. See ``README.md``.
"""

from __future__ import annotations

from models.whitebox import WhiteboxGeneration

#: Stable name for the metric, logged with every measurement.
UNCERTAINTY_METRIC = "mean_token_entropy"


def mean_token_entropy(generation: WhiteboxGeneration) -> float:
    """Mean full-vocabulary entropy across generated tokens (Fomicheva et al., 2020).

    ``H(X) = -sum_{x in V} p(x) log p(x)`` at each generated position, averaged
    over positions. Entropy is in nats, over the whole vocabulary rather than a
    top-k slice, so the value does not depend on the sampling parameters.

    Returns ``nan`` when nothing was generated: there is no distribution to
    score, and the caller decides what an unanswerable turn means rather than
    receiving a fabricated number.
    """
    return generation.mean_token_entropy


__all__ = ["UNCERTAINTY_METRIC", "mean_token_entropy"]
