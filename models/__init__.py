"""LLM model framework.

The contract is intentionally tiny: every backend implements
:class:`models.base.BaseLLM` with a single ``generate`` method. The
factory in :mod:`models.factory` instantiates a backend from an entry
in ``configs/models.yaml`` (or an inline dict) and dispatches to
:mod:`models.model`, which holds the closed-source (OpenAI, Anthropic) and
open-source (Hugging Face Transformers, MLX, vLLM) backends in one module.

Provider names recognised by the factory: ``openai``, ``anthropic``,
``hf`` (transformers), ``mlx``, ``vllm``.

:mod:`models.whitebox` adds an *optional* capability on top of that contract,
for methods that need the model's own token-level scores. Only ``hf``
implements it.
"""

from models.base import BaseLLM, Generation
from models.factory import build_llm, load_model_registry
from models.whitebox import WhiteboxGeneration, WhiteboxLLM

__all__ = [
    "BaseLLM",
    "Generation",
    "WhiteboxGeneration",
    "WhiteboxLLM",
    "build_llm",
    "load_model_registry",
]
