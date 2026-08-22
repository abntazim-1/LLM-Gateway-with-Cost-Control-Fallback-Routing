from typing import Any, AsyncGenerator, Dict, List

import pytest

from gateway.adapters.base import (
    DEFAULT_TOKEN_OVERHEAD_PER_MESSAGE,
    REPLY_PRIMING_TOKENS,
    BaseAdapter,
    NormalizedResponse,
)


class _Adapter(BaseAdapter):
    async def complete(self, messages, **kwargs) -> NormalizedResponse: ...

    async def complete_stream(
        self, messages, **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        yield {}

    async def health_check(self) -> bool:
        return True


def _make(**overrides) -> _Adapter:
    cfg = {
        "id": "a",
        "model": "m",
        "cost_per_1k_prompt": 1.0,
        "cost_per_1k_completion": 1.0,
    }
    cfg.update(overrides)
    return _Adapter(cfg)


def test_uses_real_tokenizer_not_char_approximation():
    # The chars/4 fallback this replaced would return 1 for this string;
    # a real tokenizer plus chat overhead must be materially higher.
    adapter = _make()
    text = "Say OK"
    tokens = adapter.count_prompt_tokens([{"role": "user", "content": text}])

    assert tokens > len(text) / 4.0
    assert tokens == (
        REPLY_PRIMING_TOKENS
        + DEFAULT_TOKEN_OVERHEAD_PER_MESSAGE
        + adapter.count_completion_tokens(text)
    )


def test_prompt_count_includes_per_message_chat_overhead():
    """Providers bill role/delimiter scaffolding per message, not just content."""
    adapter = _make(token_overhead_per_message=10)
    one = adapter.count_prompt_tokens([{"role": "user", "content": "hello"}])
    two = adapter.count_prompt_tokens(
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    )

    assert two - one == 10 + adapter.count_completion_tokens("hi")


def test_overhead_is_configurable_per_backend():
    """Different models ship different chat templates, so overhead differs."""
    light = _make(token_overhead_per_message=4)
    heavy = _make(token_overhead_per_message=26)
    msgs = [{"role": "user", "content": "hello"}]

    assert heavy.count_prompt_tokens(msgs) - light.count_prompt_tokens(msgs) == 22


def test_unknown_tokenizer_falls_back_without_raising():
    adapter = _make(tokenizer="definitely-not-a-real-encoding")
    assert adapter.count_prompt_tokens([{"role": "user", "content": "hello"}]) > 0


def test_handles_empty_and_missing_content():
    adapter = _make()
    assert adapter.count_prompt_tokens([]) >= 1
    assert adapter.count_prompt_tokens([{"role": "user"}]) >= 1
    assert adapter.count_completion_tokens("") == 0


def test_non_ascii_is_not_undercounted_as_one_token_per_four_chars():
    """chars/4 badly undercounts non-Latin scripts; a tokenizer must not."""
    adapter = _make()
    bengali = "নমস্কার, আপনি কেমন আছেন?"
    assert adapter.count_completion_tokens(bengali) > len(bengali) / 4.0
