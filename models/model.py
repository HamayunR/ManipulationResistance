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

import json
import os
import re
from typing import Any, Mapping, Optional, Sequence

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


# Offline mock backend
#
# Phase markers. Each is a section header that appears in exactly one template
# in ``prompts.py``, so a prompt can be classified without extra state:
#   "SOLUTIONS TO REVIEW"    -> CRITIQUE_GENERATION_TEMPLATE (and the malicious
#                               variant), i.e. the critique-generation phase.
#   "CRITIQUES YOU RECEIVED" -> ANSWER_UPDATE_TEMPLATE, i.e. the update phase.
# Neither present -> INITIAL_ANSWER_TEMPLATE, the round-0 answer phase.
_MARKER_CRITIQUE_PHASE = "SOLUTIONS TO REVIEW"
_MARKER_UPDATE_PHASE = "CRITIQUES YOU RECEIVED"

#: Target blocks rendered by ``runner.experiment._render_targets``.
_RE_PARTICIPANT = re.compile(r"\[Participant (\d+)\]\s*\nAnswer:\s*(.*)")
#: Critique blocks rendered by ``runner.experiment._render_critiques_for_agent``.
_RE_CRITIQUE_SOURCE = re.compile(r"\[Critique from Agent (\d+)\]")

#: Round counter the mock embeds in its own ``reasoning`` and reads back out of
#: ``previous_reasoning`` on the next update turn. The prompts carry no round
#: index, and ``models.base.BaseLLM`` requires backends to be stateless across
#: calls, so the counter has to ride round-trip through the transcript.
_RE_ROUND_MARKER = re.compile(r"\[mock round (\d+)\]")
#: Own current answer, from the critique prompt's YOUR CURRENT ANSWER block.
_RE_OWN_ANSWER = re.compile(r"YOUR CURRENT ANSWER\s*\nAnswer:\s*(.*)")
#: Single binary arithmetic expression, for solver mode on the dummy dataset.
_RE_ARITHMETIC = re.compile(r"(-?\d+)\s*([+\-*])\s*(-?\d+)")

#: Answer tokens that are resolved against the question instead of emitted
#: literally. Anything else in ``answer:`` is used verbatim.
_ANSWER_CORRECT = "correct"
_ANSWER_WRONG = "wrong"


class MockLLM(BaseLLM):
    """Deterministic offline backend. No network call, no randomness.

    Behaviour comes entirely from the registry entry's ``mock`` block::

        mock:
          agents:
            1: {answer: "42", confidence: 3}
            5: {answer: "7",  confidence: 2}
          default: {answer: "42", confidence: 3}

    Output deliberately mirrors what a real backend produces rather than an
    idealised version of it: the JSON object is wrapped in a markdown fence and
    preceded by a prose line. ``AGENT_SYSTEM`` forbids both, real models emit
    them anyway, and ``runner.experiment._extract_json_object`` is written to
    tolerate them -- so emitting bare JSON would bypass the very scan the
    parser exists to perform.

    Not emitted: trailing commas. The parser has a dedicated recovery path for
    them (``_extract_json_object``'s regex fallback); emitting them
    unconditionally would exercise only that branch and never the fast path,
    so that recovery stays uncovered by this backend.

    Every artefact produced through this backend is tagged ``mock: true`` --
    see ``is_mock`` and its use in :mod:`runner.experiment`.
    """

    name = "mock"

    #: Sentinel read by the runner to tag results, traces and logs.
    is_mock = True

    def __init__(
        self,
        model: str = "mock",
        *,
        mock: Optional[Mapping[str, Any]] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_tokens_param: str = "max_tokens",
        timeout: float = 0.0,
        **_ignored: Any,
    ) -> None:
        """Store the canned per-agent responses.

        Parameters
        ----------
        mock:
            Mapping with ``agents`` (per-agent-id overrides) and ``default``.
            Agent keys may be ints or strings; both are normalised to int.
        temperature, max_tokens, timeout, max_tokens_param:
            Accepted and ignored, so a mock entry is drop-in swappable with a
            real provider entry.
        """
        block = dict(mock or {})
        self._model = model
        self._default = self._normalise_response(block.get("default"))
        self._agents: dict[int, dict[str, Any]] = {}
        for key, value in (block.get("agents") or {}).items():
            try:
                agent_id = int(key)
            except (TypeError, ValueError):
                raise ValueError(
                    f"mock.agents keys must be integer agent ids, got {key!r}"
                ) from None
            self._agents[agent_id] = self._normalise_response(value)

    @classmethod
    def _normalise_response(cls, value: Any) -> dict[str, Any]:
        """Split an agent entry into its base state and its round schedule.

        ``rounds`` maps a round index to an override applied from that round
        onward (a step function, so a scripted change persists rather than
        firing once). Omitted fields inherit from the base entry.
        """
        entry = dict(value or {})
        schedule_raw = entry.pop("rounds", None) or {}
        base = cls._normalise_state(entry)
        schedule: dict[int, dict[str, Any]] = {}
        for key, override in schedule_raw.items():
            try:
                round_idx = int(key)
            except (TypeError, ValueError):
                raise ValueError(
                    f"mock.agents[...].rounds keys must be integer round "
                    f"indices, got {key!r}"
                ) from None
            merged = dict(entry)
            merged.update(dict(override or {}))
            schedule[round_idx] = cls._normalise_state(merged)
        return {"base": base, "rounds": schedule}

    @staticmethod
    def _normalise_state(entry: Mapping[str, Any]) -> dict[str, Any]:
        confidence = entry.get("confidence", 3)
        try:
            confidence = int(confidence)
        except (TypeError, ValueError):
            confidence = 3
        return {
            "answer": str(entry.get("answer", "")),
            # Clamped to the 1-5 rubric in prompts.CONFIDENCE_RUBRIC; the
            # runner clamps again in _clamp_confidence, this just keeps the
            # emitted text self-consistent.
            "confidence": max(1, min(5, confidence)),
        }

    def _state_for(self, agent_id: Optional[int], round_idx: int) -> dict[str, Any]:
        """Resolve an agent's scripted state at ``round_idx``."""
        spec = self._default if agent_id is None else self._agents.get(
            int(agent_id), self._default
        )
        state = dict(spec["base"])
        for scheduled_round in sorted(spec["rounds"]):
            if round_idx >= scheduled_round:
                state = dict(spec["rounds"][scheduled_round])
        return state

    @staticmethod
    def _question_line(prompt: str) -> str:
        """Best-effort extraction of the single question line from a prompt.

        Scoped deliberately narrowly: ``CONFIDENCE_RUBRIC`` contains the string
        ``"integer 1-5"``, which a naive arithmetic scan over the whole prompt
        would read as ``1 - 5``.
        """
        body = prompt.split("PROBLEM\n", 1)[-1]
        if "Problem:\n" in body:
            body = body.split("Problem:\n", 1)[1]
        return body.splitlines()[0] if body.splitlines() else ""

    @classmethod
    def _solve(cls, prompt: str) -> Optional[int]:
        """Evaluate the dummy dataset's single-operator arithmetic question."""
        match = _RE_ARITHMETIC.search(cls._question_line(prompt))
        if not match:
            return None
        left, op, right = int(match.group(1)), match.group(2), int(match.group(3))
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        return left * right

    @classmethod
    def _materialise_answer(cls, token: str, prompt: str) -> str:
        """Resolve ``correct`` / ``wrong`` against the question; else literal.

        ``wrong`` is gold + 1, mirroring the numeric branch of
        ``runner.experiment._default_wrong_answer``, so a wrong answer is
        plausibly wrong for the question rather than an unrelated constant.
        """
        lowered = token.strip().lower()
        if lowered not in (_ANSWER_CORRECT, _ANSWER_WRONG):
            return token
        gold = cls._solve(prompt)
        if gold is None:
            return token
        return str(gold if lowered == _ANSWER_CORRECT else gold + 1)

    @staticmethod
    def _wrap(payload: dict[str, Any], lead_in: str) -> str:
        """Render ``payload`` the way a real model tends to: prose + fence."""
        body = json.dumps(payload, ensure_ascii=False, indent=2)
        return f"{lead_in}\n\n```json\n{body}\n```"

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system: Optional[str] = None,
        agent_id: Optional[int] = None,
    ) -> Generation:
        """Return canned output shaped for whichever phase ``prompt`` is."""
        if _MARKER_CRITIQUE_PHASE in prompt:
            text = self._critique_text(prompt, agent_id)
        elif _MARKER_UPDATE_PHASE in prompt:
            # The prompt's own previous_reasoning carries the last round index.
            marker = _RE_ROUND_MARKER.search(prompt)
            round_idx = int(marker.group(1)) + 1 if marker else 1
            own = self._state_for(agent_id, round_idx)
            own["answer"] = self._materialise_answer(own["answer"], prompt)
            text = self._update_text(prompt, own, agent_id, round_idx)
        else:
            own = self._state_for(agent_id, 0)
            own["answer"] = self._materialise_answer(own["answer"], prompt)
            text = self._answer_text(own, agent_id, 0)

        # Synthetic but deterministic, so token-cost figures stay computable
        # under the mock. Roughly 4 characters per token.
        return Generation(
            text=text,
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(text) // 4,
            raw=None,
        )

    def _answer_text(
        self, own: dict[str, Any], agent_id: Optional[int], round_idx: int
    ) -> str:
        return self._wrap(
            {
                "answer": own["answer"],
                "confidence": own["confidence"],
                "reasoning": (
                    f"Mock agent {agent_id} returns its configured answer "
                    f"{own['answer']!r} with no reasoning performed. "
                    f"[mock round {round_idx}]"
                ),
            },
            "Here is my answer.",
        )

    def _update_text(
        self,
        prompt: str,
        own: dict[str, Any],
        agent_id: Optional[int],
        round_idx: int,
    ) -> str:
        # Always REJECT: the config pins one answer per agent, so accepting a
        # critique that changed the answer would contradict determinism.
        critique_response = {
            source: {
                "decision": "REJECT",
                "reason": "Mock agent holds its configured answer.",
            }
            for source in _RE_CRITIQUE_SOURCE.findall(prompt)
        }
        return self._wrap(
            {
                "answer": own["answer"],
                "confidence": own["confidence"],
                "reasoning": (
                    f"Mock agent {agent_id} holds scripted answer "
                    f"{own['answer']!r} for this round. "
                    f"[mock round {round_idx}]"
                ),
                "critique_response": critique_response,
            },
            "I have considered the critiques.",
        )

    def _critique_text(self, prompt: str, agent_id: Optional[int]) -> str:
        # Read the agent's own current answer straight off the prompt rather
        # than re-deriving it: the critique phase has no round marker, and the
        # runner has already rendered the authoritative value.
        own_match = _RE_OWN_ANSWER.search(prompt)
        own_answer = own_match.group(1).strip() if own_match else ""
        reviews = []
        for target_id, target_answer in _RE_PARTICIPANT.findall(prompt):
            agrees = target_answer.strip() == own_answer
            reviews.append(
                {
                    "target": int(target_id),
                    "step_loc": (
                        "No error identified."
                        if agrees
                        else f"Target answer {target_answer.strip()!r} disagrees "
                        f"with {own_answer!r}."
                    ),
                    "correction": "" if agrees else own_answer,
                    # Derived, not fixed, so disagreement is visible to the
                    # routing objective's targeted-cross and disagreement terms.
                    "assessment": "Strong" if agrees else "Flawed",
                }
            )
        return self._wrap({"reviews": reviews}, "Here are my reviews.")


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

        torch_dtype = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, "auto")

        token = os.getenv("HF_TOKEN") or None

        self._tokenizer = AutoTokenizer.from_pretrained(
            model, token=token, trust_remote_code=trust_remote_code
        )
        self._model_obj = AutoModelForCausalLM.from_pretrained(
            model,
            token=token,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
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
