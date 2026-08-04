"""Reimplementations of published multi-agent methods, for comparison.

Each subpackage is one paper. A baseline lives here rather than in
:mod:`runner` because it is *not* a variant of this repository's method: it has
its own debate protocol, its own prompts and its own notion of confidence, and
mixing those into the PEAR loop would blur the comparison it exists to support.

Every baseline exposes ``run_<name>_example(task, example, *, debate_cfg, llm,
seed, perm_seed) -> dict`` returning the row :func:`runner.experiment.run_one`
returns, so logging, scoring and the analysis layer need no special cases.

See ``README.md`` in this directory for the full contract and for how to add
another paper.

Available baselines:

``debunc``
    DebUnc: Improving Large Language Model Agent Communication With
    Uncertainty Metrics (Yoffe, Amayuelas and Wang, 2024).
"""

__all__: list[str] = []
