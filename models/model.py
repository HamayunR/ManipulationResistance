"""Concrete LLM backends: closed-source (OpenAI / Anthropic) and open-source (HF).

This module merges the previous ``closed_source.py`` and ``open_source.py``
files into one place so that experiment authors only need to look in one
spot to see how every non-stub backend works.

Backends are *lazily importable*: the third-party SDKs (``openai``,
``anthropic``, ``transformers``, ``torch``) are only imported inside the
constructor of the backend that needs them, so the rest of the harness can
be installed without those optional extras.

Conventions shared by every backend
-----------------------------------
* Subclass :class:`models.base.BaseLLM` and implement ``generate``.
* Read API keys from the process environment, never from constructor args
  (the model factory loads ``.env`` once per process). Missing required
  keys raise :class:`RuntimeError` with a pointer to ``.env.example``.
* Return :class:`models.base.Generation` with best-effort token usage so the
  budget tracker (:mod:`utils.budget`) can enforce ExpPlan.md fairness caps.
* Never log secrets and never echo prompts in error messages.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

from models.base import BaseLLM, Generation
from utils.logging import get_logger

_log = get_logger("models.model")


# Closed-source backends
class OpenAILLM(BaseLLM):
    """OpenAI Chat Completions backend.

    Reads the API key from ``OPENAI_API_KEY`` and (optionally) a custom base
    URL from ``OPENAI_BASE_URL`` (useful for OpenAI-compatible gateways such
    as Azure-OpenAI proxies). The organisation can be set via
    ``OPENAI_ORG_ID``.
    """

    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 256,
        max_tokens_param: str = "max_tokens",
        timeout: float = 60.0,
        **client_kwargs: Any,
    ) -> None:
        """Build the underlying ``openai.OpenAI`` client and stash defaults.

        Parameters
        ----------
        model:
            Model name passed to ``chat.completions.create`` (e.g.
            ``"gpt-4o-mini"``).
        temperature, max_tokens:
            Default decoding parameters; can be overridden per call.
        timeout:
            Per-call timeout in seconds.
        **client_kwargs:
            Forwarded to the ``OpenAI`` constructor for advanced setups.
        """
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "The 'openai' package is required for OpenAILLM. "
                "Install with `pip install openai`."
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )

        base_url = os.getenv("OPENAI_BASE_URL") or None
        org = os.getenv("OPENAI_ORG_ID") or None

        self._client = OpenAI(
            api_key=api_key, base_url=base_url, organization=org, **client_kwargs
        )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_tokens_param = max_tokens_param
        self._timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        agent_id: Optional[int] = None,
    ) -> Generation:
        """Call ``chat.completions.create`` and return a :class:`Generation`."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature if temperature is None else temperature,
            "timeout": self._timeout,
            self._max_tokens_param: self._max_tokens if max_tokens is None else max_tokens,
        }
        resp = self._client.chat.completions.create(**request)
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return Generation(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            raw=resp,
        )


class AnthropicLLM(BaseLLM):
    """Anthropic Messages API backend.

    Reads the API key from ``ANTHROPIC_API_KEY`` and an optional base URL
    from ``ANTHROPIC_BASE_URL``.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 512,
        max_tokens_param: str = "max_tokens",
        timeout: float = 60.0,
        **client_kwargs: Any,
    ) -> None:
        """Build the underlying ``anthropic.Anthropic`` client.

        Parameters
        ----------
        model:
            Model id passed to ``messages.create`` (e.g.
            ``"claude-3-5-sonnet-latest"``).
        temperature, max_tokens, timeout:
            Default decoding parameters; can be overridden per call.
        max_tokens_param:
            Accepted for parity with the OpenAI backend; unused here.
        **client_kwargs:
            Forwarded to the ``Anthropic`` constructor for advanced setups.
        """
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "The 'anthropic' package is required for AnthropicLLM. "
                "Install with `pip install anthropic`."
            ) from exc

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
            )

        base_url = os.getenv("ANTHROPIC_BASE_URL") or None
        self._client = Anthropic(api_key=api_key, base_url=base_url, **client_kwargs)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_tokens_param = max_tokens_param
        self._timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        agent_id: Optional[int] = None,
    ) -> Generation:
        """Call ``messages.create`` and return a :class:`Generation`.

        The Messages API returns a list of content blocks; we concatenate the
        ``"text"`` blocks and ignore tool-use blocks (this harness does not
        use tools yet).
        """
        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=self._max_tokens if max_tokens is None else max_tokens,
            temperature=self._temperature if temperature is None else temperature,
            messages=[{"role": "user", "content": prompt}],
            timeout=self._timeout,
        )
        if system:
            kwargs["system"] = system

        resp = self._client.messages.create(**kwargs)
        parts = [
            block.text
            for block in resp.content
            if getattr(block, "type", None) == "text"
        ]
        text = "".join(parts)
        usage = getattr(resp, "usage", None)
        return Generation(
            text=text,
            prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage, "output_tokens", 0) or 0,
            raw=resp,
        )


# Open-source backends
class HFLLM(BaseLLM):
    """Hugging Face Transformers causal-LM backend.

    Loads the tokenizer and model once at construction; subsequent
    ``generate`` calls reuse the same pipeline. The backend reads ``HF_TOKEN``
    from the environment for gated checkpoints.

    Parameters
    ----------
    model:
        Model name on Hugging Face Hub or local path.
    dtype:
        ``"bfloat16"``, ``"float16"``, ``"float32"`` or ``"auto"``.
    device:
        ``"auto"``, ``"cuda"``, ``"cpu"`` or any string accepted by
        ``transformers.pipeline``'s ``device_map`` argument.
    temperature, max_new_tokens:
        Default decoding parameters; can be overridden per call.
    trust_remote_code:
        Forwarded to ``AutoModelForCausalLM.from_pretrained``; required by
        some community checkpoints.
    **gen_kwargs:
        Extra keyword arguments forwarded to every ``pipeline()`` call (e.g.
        ``top_p``, ``repetition_penalty``).
    """

    name = "hf"

    def __init__(
        self,
        model: str,
        *,
        dtype: str = "auto",
        device: str = "auto",
        temperature: float = 0.2,
        max_new_tokens: int = 256,
        trust_remote_code: bool = False,
        **gen_kwargs: Any,
    ) -> None:
        try:
            import torch  # type: ignore
            import transformers  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForCausalLM,
                AutoTokenizer,
                pipeline,
            )
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "transformers / torch are required for HFLLM. Install via "
                "`pip install transformers torch accelerate`."
            ) from exc

        resolved_dtype = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, "auto")

        token = os.getenv("HF_TOKEN") or None

        self._tokenizer = AutoTokenizer.from_pretrained(
            model, token=token, trust_remote_code=trust_remote_code
        )

        load_kwargs: Dict[str, Any] = {
            "token": token,
            "trust_remote_code": trust_remote_code,
        }
        # transformers 5 renamed ``torch_dtype`` to ``dtype`` and warns on the
        # old name; 4.x only knows the old one. Pick by what is accepted so the
        # backend works across the range in requirements.txt.
        import inspect

        signature = inspect.signature(AutoModelForCausalLM.from_pretrained)
        takes_kwargs = any(
            param.kind is inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        dtype_key = (
            "dtype"
            if ("dtype" in signature.parameters or takes_kwargs)
            and int(getattr(transformers, "__version__", "0").split(".")[0] or 0) >= 5
            else "torch_dtype"
        )
        load_kwargs[dtype_key] = resolved_dtype

        # Apple Silicon: accelerate's device_map placement segfaults while
        # sharding onto "mps" (observed on torch 2.13 / transformers 5.14).
        # Loading on CPU and moving the whole model afterwards is the
        # documented path, and a model small enough to fit unified memory
        # never needed sharding in the first place.
        mps_requested = str(device).lower() in {"mps", "mps:0"}
        if not mps_requested:
            load_kwargs["device_map"] = device

        self._model_obj = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
        if mps_requested:
            if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
                raise RuntimeError(
                    "device: mps was requested but this PyTorch build has no "
                    "working MPS backend. Use device: cpu, or install a macOS "
                    "arm64 build of torch."
                )
            self._model_obj = self._model_obj.to("mps")
        self._model_obj.eval()

        self._pipe = pipeline(
            "text-generation",
            model=self._model_obj,
            tokenizer=self._tokenizer,
        )
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._extra = gen_kwargs

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        agent_id: Optional[int] = None,
    ) -> Generation:
        """Run ``pipeline(prompt)`` and return only the newly-generated tokens.

        Many instruction-tuned models expect a chat template; if the
        tokenizer exposes one, we use it so the special tokens are correct.
        Otherwise we fall back to a simple ``"<system>\\n\\n<prompt>"``
        concatenation.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            full_prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            full_prompt = (system + "\n\n" + prompt) if system else prompt

        out = self._pipe(
            full_prompt,
            do_sample=True,
            temperature=self._temperature if temperature is None else temperature,
            max_new_tokens=self._max_new_tokens if max_tokens is None else max_tokens,
            pad_token_id=self._tokenizer.eos_token_id,
            return_full_text=False,
            **self._extra,
        )
        text = out[0]["generated_text"] if out else ""

        # Token counts are best-effort: re-tokenise the prompt + completion.
        prompt_tokens = len(self._tokenizer.encode(full_prompt))
        completion_tokens = len(self._tokenizer.encode(text)) if text else 0
        return Generation(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw=out,
        )


class VLLMLLM(BaseLLM):
    """vLLM offline-inference backend for local checkpoints.

    Wraps :class:`vllm.LLM` so that experiment runs benefit from vLLM's
    paged KV cache and batched scheduling without leaving the Python
    process. The engine is constructed once (slow: loads weights into GPU
    memory and warms up the CUDA graph) and reused for every subsequent
    :meth:`generate` call.

    Parameters
    ----------
    model:
        Hugging Face Hub repo id or absolute path to a local checkpoint
        directory (the same directory layout that :class:`HFLLM` accepts).
    dtype:
        ``"auto"`` / ``"bfloat16"`` / ``"float16"`` / ``"float32"``.
        Forwarded as a string to vLLM, which does its own dtype mapping.
    tensor_parallel_size:
        Number of GPUs to shard the model across. Default 1 = single GPU.
    gpu_memory_utilization:
        Fraction of GPU memory vLLM is allowed to claim (0.0-1.0).
        Lower this if other processes share the same GPU.
    max_model_len:
        Optional cap on context length. Defaults to whatever the model
        declares in its ``config.json``.
    trust_remote_code:
        Forwarded to vLLM's tokenizer / config loaders. Required for some
        community checkpoints (e.g. older Qwen revisions that ship custom
        modeling code in the repo).
    temperature, max_tokens:
        Default decoding parameters; can be overridden per call.
    **engine_kwargs:
        Extra keyword arguments forwarded to ``vllm.LLM(...)`` (e.g.
        ``swap_space``, ``enforce_eager``, ``quantization``).

    Notes
    -----
    * vLLM is **CUDA-only** at the time of writing; CPU/MPS users should
      stick with :class:`HFLLM`.
    * For best throughput call this in a long-lived process; tearing the
      engine up and down per example is wasteful.
    * Token-usage counters reuse vLLM's reported ``prompt_token_ids`` and
      per-output ``token_ids`` lists so the budget tracker still works.
    """

    name = "vllm"

    def __init__(
        self,
        model: str,
        *,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 256,
        max_new_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[str | list[str]] = None,
        **engine_kwargs: Any,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "The 'vllm' package is required for VLLMLLM. "
                "Install with `pip install vllm` (CUDA-only)."
            ) from exc

        # Build the long-lived engine. vLLM tolerates ``max_model_len=None``
        # via simply not passing the kwarg; we filter None out for cleanness.
        engine_init_kwargs: dict[str, Any] = dict(
            model=model,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            **engine_kwargs,
        )
        if max_model_len is not None:
            engine_init_kwargs["max_model_len"] = int(max_model_len)

        self._engine = LLM(**engine_init_kwargs)
        self._SamplingParams = SamplingParams
        # Reuse vLLM's tokenizer for chat-template formatting + token counts.
        self._tokenizer = self._engine.get_tokenizer()
        self._temperature = temperature
        self._max_tokens = int(max_new_tokens) if max_new_tokens is not None else max_tokens
        self._sampling_defaults: dict[str, Any] = {}
        if top_p is not None:
            self._sampling_defaults["top_p"] = top_p
        if repetition_penalty is not None:
            self._sampling_defaults["repetition_penalty"] = repetition_penalty
        if stop is not None:
            self._sampling_defaults["stop"] = [stop] if isinstance(stop, str) else stop

    def _render_prompt(self, prompt: str, system: Optional[str]) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return (system + "\n\n" + prompt) if system else prompt

    def _sampling_params(
        self,
        *,
        max_tokens: Optional[int],
        temperature: Optional[float],
    ) -> Any:
        sampling_kwargs = dict(self._sampling_defaults)
        sampling_kwargs["temperature"] = self._temperature if temperature is None else temperature
        sampling_kwargs["max_tokens"] = self._max_tokens if max_tokens is None else max_tokens
        return self._SamplingParams(**sampling_kwargs)

    @staticmethod
    def _generation_from_output(out: Any) -> Generation:
        text = out.outputs[0].text if out.outputs else ""
        prompt_token_ids = getattr(out, "prompt_token_ids", None) or []
        completion_token_ids: list[int] = []
        if out.outputs:
            completion_token_ids = list(getattr(out.outputs[0], "token_ids", []) or [])
        return Generation(
            text=text,
            prompt_tokens=len(prompt_token_ids),
            completion_tokens=len(completion_token_ids),
            raw=out,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        agent_id: Optional[int] = None,
    ) -> Generation:
        """Run one synchronous request through the in-process vLLM engine.

        We render the (system, user) message pair through the tokenizer's
        chat template when one exists; otherwise we concatenate ``system``
        and ``prompt`` with a blank line in between (same fallback policy as
        :class:`HFLLM`).
        """
        full_prompt = self._render_prompt(prompt, system)
        sampling_params = self._sampling_params(
            max_tokens=max_tokens,
            temperature=temperature,
        )
        # ``LLM.generate`` is synchronous and accepts a list of prompts; we
        # submit one prompt here to preserve the single-call interface.
        outputs = self._engine.generate([full_prompt], sampling_params)
        if not outputs:
            return Generation(text="", prompt_tokens=0, completion_tokens=0, raw=None)
        return self._generation_from_output(outputs[0])

    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
        system: Optional[str] = None,
        agent_ids: Optional[Sequence[Optional[int]]] = None,
    ) -> list[Generation]:
        """Run a true batched request through the in-process vLLM engine."""
        if not prompts:
            return []
        if agent_ids is not None and len(agent_ids) != len(prompts):
            raise ValueError("agent_ids must have the same length as prompts")

        full_prompts = [self._render_prompt(prompt, system) for prompt in prompts]
        sampling_params = self._sampling_params(
            max_tokens=max_tokens,
            temperature=temperature,
        )
        outputs = self._engine.generate(full_prompts, sampling_params)
        if len(outputs) != len(full_prompts):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(full_prompts)} prompts"
            )
        return [self._generation_from_output(out) for out in outputs]


__all__ = ["AnthropicLLM", "HFLLM", "OpenAILLM", "VLLMLLM"]
