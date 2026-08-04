"""DebUnc: multi-agent debate with token-probability confidence.

Reimplementation of *DebUnc: Improving Large Language Model Agent Communication
With Uncertainty Metrics* (Yoffe, Amayuelas and Wang, 2024;
https://arxiv.org/abs/2407.06426), from the paper and the authors' code at
https://github.com/lukeyoffe/debunc.

The method takes a plain multi-agent debate and adds one thing: each agent's
confidence is measured from the token probabilities of its own response rather
than asked for in words, and is then stated next to that response in the next
round's prompt.

This port implements one point in the paper's grid -- **Confidence in Prompt**
with **Mean Token Entropy**. See ``README.md`` for what was left out and why.

Enable it with ``debate.mode: debunc`` (or ``debunc_prompt``).
"""

from baselines.debunc.config import DEBUNC_MODES, DebUncConfig, is_debunc_mode
from baselines.debunc.runner import run_debunc_example

__all__ = [
    "DEBUNC_MODES",
    "DebUncConfig",
    "is_debunc_mode",
    "run_debunc_example",
]
