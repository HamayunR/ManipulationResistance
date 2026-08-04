"""End-to-end checks of the whitebox backend against a real checkpoint.

The thing that cannot be faked: whether the per-token scores are the model's
own distribution, or one already narrowed by the sampling parameters. An
uncertainty metric computed on top-k filtered logits is a different quantity,
and nothing downstream would notice the difference.

These need a locally cached checkpoint and are skipped otherwise, so a machine
without one still gets a green run from the rest of the suite. Fetch it with::

    huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from models.model import HFLLM  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def llm():
    from transformers import AutoTokenizer

    try:
        AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except Exception:  # pragma: no cover - depends on the local cache
        pytest.skip(f"{MODEL} is not in the local Hugging Face cache")
    return HFLLM(model=MODEL, dtype="float32", device="cpu")


@pytest.fixture
def conversation():
    return [
        {"role": "user", "content": "What is 2+2? Answer with a number."},
        {"role": "assistant", "content": "The answer is 4."},
        {"role": "user", "content": "One agent solution: ```the answer is 5```"},
    ]


def test_scores_are_the_models_own_distribution(llm, conversation):
    # Sampling with top_k=50 would leave 50 finite entries if the scores were
    # read after the warpers. Uncertainty metrics assume the full vocabulary.
    generation = llm.chat_generate(
        conversation, max_tokens=8, temperature=1.0, top_k=50, top_p=1.0, seed=0
    )
    assert generation.logits_are_unprocessed
    assert generation.token_ids
    assert len(generation.token_entropies) == len(generation.token_ids)
    assert len(generation.token_logprobs) == len(generation.token_ids)
    assert all(e > 0 for e in generation.token_entropies)
    assert all(lp <= 0 for lp in generation.token_logprobs)
    # Entropy over a 150k vocabulary is bounded by log(150k) ~ 11.9 nats.
    assert all(e < math.log(len(llm.tokenizer)) + 1 for e in generation.token_entropies)
    assert math.isfinite(generation.mean_token_entropy)


def test_generation_is_reproducible_under_a_seed(llm, conversation):
    kwargs = dict(max_tokens=12, temperature=1.0, top_k=50, seed=1234)
    first = llm.chat_generate(conversation, **kwargs)
    second = llm.chat_generate(conversation, **kwargs)
    assert first.text == second.text
    assert first.token_logprobs == pytest.approx(second.token_logprobs)


def test_greedy_decoding_matches_transformers(llm, conversation):
    ours = llm.chat_generate(conversation, max_tokens=10, temperature=0.0)
    templated = llm.tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True
    )
    prompt_ids = templated["input_ids"] if hasattr(templated, "keys") else templated
    ids = torch.tensor(
        [list(prompt_ids)], dtype=torch.long, device=llm._model_obj.device
    )
    with torch.no_grad():
        reference = llm._model_obj.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=llm.tokenizer.eos_token_id,
        )
    expected = reference[0, ids.shape[1] :].tolist()
    assert ours.token_ids == expected[: len(ours.token_ids)]


def test_the_conversation_reaches_the_model(llm):
    # The whole history is sent, not just the last turn: a debate depends on
    # round three still having round one in context.
    kwargs = dict(max_tokens=6, temperature=0.0)
    short = llm.chat_generate([{"role": "user", "content": "Say the word red."}], **kwargs)
    with_history = llm.chat_generate(
        [
            {"role": "user", "content": "Say the word red."},
            {"role": "assistant", "content": "red"},
            {"role": "user", "content": "Now say the word blue instead."},
        ],
        **kwargs,
    )
    assert with_history.prompt_tokens > short.prompt_tokens
    assert with_history.text != short.text


def test_an_empty_conversation_is_refused(llm):
    with pytest.raises(Exception):
        llm.chat_generate([], max_tokens=4, temperature=0.0)
