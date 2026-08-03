"""Factory and registry for LLM backends.

The factory takes either a model registry entry name (looked up in
``configs/models.yaml``) or an inline dict, and returns an instance of one of
the :class:`BaseLLM` subclasses.

Closed-source providers require credentials in the process environment; we
auto-load ``.env`` (via ``python-dotenv``) the first time the factory is
called. Loading is idempotent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from models.base import BaseLLM
from utils.logging import get_logger

_log = get_logger("models.factory")
_DOTENV_LOADED = False


def _ensure_dotenv() -> None:
    """Load ``.env`` once per process if ``python-dotenv`` is available.

    We never *require* a .env: missing keys are surfaced lazily by the
    closed-source backends when they try to read them.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:  # pragma: no cover - python-dotenv is in requirements.txt
        _DOTENV_LOADED = True
        return

    # Walk up from cwd looking for a .env (project root or above).
    for parent in [Path.cwd(), *Path.cwd().parents]:
        env_path = parent / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=False)
            _log.debug("Loaded environment from %s", env_path)
            break
    _DOTENV_LOADED = True


def load_model_registry(path: str | os.PathLike) -> Dict[str, Any]:
    """Read ``configs/models.yaml`` (or any compatible file) into a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Model registry at {path} must be a mapping")
    return data


def build_llm(
    spec: str | Mapping[str, Any],
    *,
    registry: Optional[Mapping[str, Any]] = None,
) -> BaseLLM:
    """Instantiate an LLM backend from a registry name or an inline spec.

    Parameters
    ----------
    spec:
        Either the *name* of an entry in ``registry["models"]`` or an inline
        dict with at least a ``provider`` field.
    registry:
        Parsed model registry dict. Required if ``spec`` is a string.

    Returns
    -------
    BaseLLM

    Raises
    ------
    KeyError
        If a string ``spec`` does not match any registry entry.
    ValueError
        If the resolved spec lacks a ``provider`` field.
    """
    _ensure_dotenv()

    if isinstance(spec, str):
        if registry is None:
            raise ValueError("registry must be provided when spec is a string name")
        models = registry.get("models", {})
        if spec not in models:
            raise KeyError(f"Unknown model: {spec!r}. Known: {sorted(models)}")
        config: Dict[str, Any] = dict(models[spec])
        config.setdefault("_name", spec)
    else:
        config = dict(spec)

    provider = config.pop("provider", None)
    if not provider:
        raise ValueError(f"Model spec missing 'provider' field: {config}")

    name = config.pop("_name", provider)

    # Dispatch
    # All backends live in models.model; imports stay lazy so the third-party
    # SDKs only load on demand (so installing vllm is not required to run an
    # OpenAI experiment, and vice versa).
    if provider == "openai":
        from models.model import OpenAILLM

        llm = OpenAILLM(**config)
    elif provider == "anthropic":
        from models.model import AnthropicLLM

        llm = AnthropicLLM(**config)
    elif provider == "hf":
        from models.model import HFLLM

        llm = HFLLM(**config)
    elif provider == "vllm":
        from models.model import VLLMLLM

        llm = VLLMLLM(**config)
    else:
        raise ValueError(f"Unknown provider: {provider!r}")

    llm.name = name
    _log.info("Built LLM %s (provider=%s)", name, provider)
    return llm
