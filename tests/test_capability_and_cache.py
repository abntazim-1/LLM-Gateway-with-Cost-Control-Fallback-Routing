"""Capability-based routing, cache determinism, JSON validation, fallback signal.

Covers F8, F10, F11 and F12 from docs/AI_ML_FLAWS.md.
"""

from typing import Any, AsyncGenerator, Dict, List

import pytest

from gateway.adapters.base import BaseAdapter, NormalizedMessage, NormalizedResponse
from gateway.ledger.store import LedgerStore
from gateway.policy.cache import PromptCache
from gateway.policy.circuit_breaker import CircuitBreakerRegistry
from gateway.policy.guardrails import GuardrailsPipeline
from gateway.policy.router import Router


class _Adapter(BaseAdapter):
    def __init__(self, config, reply="A complete and sufficiently long answer."):
        super().__init__(config)
        self.reply = reply

    async def complete(self, messages, **kwargs) -> NormalizedResponse:
        return NormalizedResponse(
            id=self.id,
            backend_id=self.id,
            model=self.model,
            messages=[NormalizedMessage(role="assistant", content=self.reply)],
            prompt_tokens=1,
            completion_tokens=1,
            cost_usd=0.01,
            latency_ms=1.0,
        )

    async def complete_stream(
        self, messages, **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        yield {}

    async def health_check(self) -> bool:
        return True


def _router(*adapters, strategy):
    return Router(
        adapters=list(adapters),
        circuit_registry=CircuitBreakerRegistry(ledger=LedgerStore(":memory:")),
        strategy=strategy,
    )


# ── F8: capability is not price ──────────────────────────────────────────


def _mismatched_pair():
    """A cheap-but-strong and an expensive-but-weak backend.

    This is the real local configuration: qwen2.5:0.5b is priced ten times
    above phi3 despite being a far smaller model, so ranking "the better
    model" by price picks the weaker one.
    """
    strong_cheap = _Adapter(
        {
            "id": "strong-cheap",
            "model": "strong",
            "cost_per_1k_prompt": 0.0001,
            "cost_per_1k_completion": 0.0002,
            "capability_tier": 2,
        }
    )
    weak_expensive = _Adapter(
        {
            "id": "weak-expensive",
            "model": "weak",
            "cost_per_1k_prompt": 0.001,
            "cost_per_1k_completion": 0.002,
            "capability_tier": 1,
        }
    )
    return strong_cheap, weak_expensive


@pytest.mark.asyncio
async def test_complex_request_routes_to_most_capable_not_most_expensive():
    strong_cheap, weak_expensive = _mismatched_pair()
    router = _router(weak_expensive, strong_cheap, strategy="complexity")

    ranked = await router.get_ranked_adapters(
        messages=[
            {
                "role": "user",
                "content": "Why would a two-tier cache produce stale reads "
                "under concurrent writes, given eventual consistency?",
            }
        ]
    )

    assert ranked[0].id == "strong-cheap"


@pytest.mark.asyncio
async def test_cascade_escalates_up_in_capability_not_price():
    strong_cheap, weak_expensive = _mismatched_pair()
    router = _router(weak_expensive, strong_cheap, strategy="cascade")

    ranked = await router.get_ranked_adapters(
        messages=[{"role": "user", "content": "hi"}]
    )

    # Weakest tried first so escalation moves toward more capability.
    assert [a.id for a in ranked] == ["weak-expensive", "strong-cheap"]


@pytest.mark.asyncio
async def test_falls_back_to_price_when_no_tier_declared():
    """Configs without capability_tier keep their previous behaviour."""
    cheap = _Adapter(
        {
            "id": "cheap",
            "model": "c",
            "cost_per_1k_prompt": 0.001,
            "cost_per_1k_completion": 0.002,
        }
    )
    pricey = _Adapter(
        {
            "id": "pricey",
            "model": "p",
            "cost_per_1k_prompt": 0.01,
            "cost_per_1k_completion": 0.02,
        }
    )
    router = _router(cheap, pricey, strategy="complexity")

    ranked = await router.get_ranked_adapters(
        messages=[
            {
                "role": "user",
                "content": "Prove this theorem and analyse its time complexity "
                "and space complexity.",
            }
        ]
    )

    assert ranked[0].id == "pricey"


# ── F12: fallback must be visible ────────────────────────────────────────


class _FailingAdapter(_Adapter):
    async def complete(self, messages, **kwargs) -> NormalizedResponse:
        raise RuntimeError("backend down")


@pytest.mark.asyncio
async def test_response_is_marked_when_a_fallback_backend_answered():
    primary = _FailingAdapter(
        {
            "id": "primary",
            "model": "p",
            "cost_per_1k_prompt": 0.001,
            "cost_per_1k_completion": 0.002,
        }
    )
    secondary = _Adapter(
        {
            "id": "secondary",
            "model": "s",
            "cost_per_1k_prompt": 0.01,
            "cost_per_1k_completion": 0.02,
        }
    )
    router = _router(primary, secondary, strategy="cost_first")
    router.timeout_sec = 5

    response = await router.execute(messages=[{"role": "user", "content": "hi"}])

    assert response.backend_id == "secondary"
    assert response.is_fallback is True


@pytest.mark.asyncio
async def test_first_choice_response_is_not_marked_as_fallback():
    adapter = _Adapter(
        {
            "id": "only",
            "model": "m",
            "cost_per_1k_prompt": 0.001,
            "cost_per_1k_completion": 0.002,
        }
    )
    response = await _router(adapter, strategy="cost_first").execute(
        messages=[{"role": "user", "content": "hi"}]
    )

    assert response.is_fallback is False


# ── F10: cache and non-determinism ───────────────────────────────────────


def test_seed_is_part_of_the_cache_key():
    """Two requests differing only by seed want different samples."""
    cache = PromptCache()
    messages = [{"role": "user", "content": "write a poem"}]
    cache.set(messages, {"seed": 1}, {"id": "first"})

    assert cache.get(messages, {"seed": 1})["id"] == "first"
    assert cache.get(messages, {"seed": 2}) is None


def test_high_temperature_requests_bypass_the_cache():
    cache = PromptCache(max_cacheable_temperature=0.0)
    messages = [{"role": "user", "content": "write a poem"}]

    cache.set(messages, {"temperature": 1.2}, {"id": "sampled"})
    assert cache.get(messages, {"temperature": 1.2}) is None

    cache.set(messages, {"temperature": 0.0}, {"id": "deterministic"})
    assert cache.get(messages, {"temperature": 0.0})["id"] == "deterministic"


def test_default_threshold_preserves_caching():
    """The default must not silently collapse the hit rate."""
    cache = PromptCache()
    messages = [{"role": "user", "content": "hello"}]
    cache.set(messages, {"temperature": 0.7}, {"id": "cached"})

    assert cache.get(messages, {"temperature": 0.7})["id"] == "cached"


def test_explicit_no_cache_is_honoured():
    cache = PromptCache()
    messages = [{"role": "user", "content": "hello"}]
    cache.set(messages, {"no_cache": True}, {"id": "x"})

    assert cache.get(messages, {"no_cache": True}) is None


# ── F11: json_mode output validation ─────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"response_format": {"type": "json_object"}}, True),
        ({"json_mode": True}, True),
        ({}, False),
        ({"response_format": {"type": "text"}}, False),
    ],
)
def test_json_requested_detection(kwargs, expected):
    assert GuardrailsPipeline.json_requested(kwargs) is expected


def test_fenced_json_is_unwrapped_and_accepted():
    """Small models very often wrap JSON in a markdown fence, which is not
    itself valid JSON — previously passed through as a success."""
    text, valid = GuardrailsPipeline.repair_json('```json\n{"a": 1}\n```')

    assert valid is True
    assert text == '{"a": 1}'


def test_plain_json_passes_unchanged():
    text, valid = GuardrailsPipeline.repair_json('{"a": 1}')
    assert (text, valid) == ('{"a": 1}', True)


def test_prose_is_reported_as_invalid_json():
    text, valid = GuardrailsPipeline.repair_json("Sure! Here is the data you wanted.")
    assert valid is False
    assert text == "Sure! Here is the data you wanted."
