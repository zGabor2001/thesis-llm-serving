"""Tests for the prompt tokenizer used by the DRR scheduler."""

from pathlib import Path

import pytest

from gateway.tokenizer import PromptTokenizer, fallback_token_count


def test_fallback_token_count():
    assert fallback_token_count("") == 1
    assert fallback_token_count("hello") == 2
    assert fallback_token_count("a" * 8) == 2


def test_prompt_tokenizer_falls_back_when_file_missing(tmp_path):
    tok = PromptTokenizer(tmp_path / "does-not-exist.json")

    assert not tok.is_loaded
    text = "Say hello in one sentence."
    assert tok.count_tokens(text) == fallback_token_count(text)


def test_prompt_tokenizer_real_counts_if_available():
    path = Path(__file__).resolve().parent.parent / "models" / "base" / "tokenizer.json"
    if not path.is_file():
        pytest.skip("models/base/tokenizer.json not present")

    tok = PromptTokenizer(path)
    assert tok.is_loaded

    short = tok.count_tokens("Say hello in one sentence.")
    long = tok.count_tokens(
        "Write a thorough technical explanation of how paged attention improves "
        "KV-cache memory utilization in vLLM, with many concrete examples."
    )

    assert short >= 1
    assert long > short
