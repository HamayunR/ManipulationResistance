"""Turning uncertainties into the confidence levels DebUnc states in the prompt.

An uncertainty metric produces a non-negative number per agent whose *scale*
carries no meaning -- mean token entropy is bounded by the vocabulary size --
so only the relative differences between agents are informative (paper section
3.2). DebUnc therefore rescales one round's uncertainties onto a 1-10 integer
scale before writing them into the next round's text.

The scale is fixed at 1-10 rather than configurable: the debate prompt tells
the model "confidence values from 1 to 10", so a different range here would
make the prompt describe a scale the numbers are not on.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

#: Bounds of the confidence scale, quoted verbatim in the debate prompt.
CONFIDENCE_MIN = 1
CONFIDENCE_MAX = 10

#: "We then scale these values such that the average confidence s_i of all
#: agents is 5" (paper section 3.2).
TARGET_MEAN_CONFIDENCE = 5.0

#: Uncertainties are inverted, so an exact zero would give an infinite weight.
#: Zero entropy means the model was completely certain at every position, which
#: is a legitimate outcome for a short deterministic answer, so it is floored
#: rather than rejected.
UNCERTAINTY_FLOOR = 1e-10


def confidences_from_uncertainties(uncertainties: Sequence[float]) -> Tuple[int, ...]:
    """Map one round's uncertainties to integer confidence levels.

    Implements section 3.2's conversion. With ``c_i = 1 / u_i`` and ``n``
    agents::

        s_i = c_i / sum_j(c_j) * (5n - 1) + 1/n

    which sums to ``5n``, so the mean confidence is 5 whatever the metric's
    scale. The result is then clamped to 1-10 and rounded to an integer.
    """
    values = [float(u) for u in uncertainties]
    if not values:
        raise ValueError("no uncertainties to convert")
    for u in values:
        if math.isnan(u) or u < 0:
            raise ValueError(f"uncertainty must be non-negative and finite, got {u}")

    inverse = [1.0 / max(u, UNCERTAINTY_FLOOR) for u in values]
    n = len(inverse)
    total = sum(inverse)
    if total <= 0 or not math.isfinite(total):
        # Every agent was infinitely uncertain: nothing distinguishes them, so
        # report the scale's midpoint rather than inventing an ordering.
        return tuple(int(round(TARGET_MEAN_CONFIDENCE)) for _ in inverse)

    scaled = [c / total * (TARGET_MEAN_CONFIDENCE * n - 1.0) + 1.0 / n for c in inverse]
    return tuple(
        int(round(min(max(s, CONFIDENCE_MIN), CONFIDENCE_MAX))) for s in scaled
    )


__all__ = [
    "CONFIDENCE_MAX",
    "CONFIDENCE_MIN",
    "TARGET_MEAN_CONFIDENCE",
    "UNCERTAINTY_FLOOR",
    "confidences_from_uncertainties",
]
