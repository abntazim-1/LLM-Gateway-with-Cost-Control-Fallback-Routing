"""Context-window enforcement.

`context_length` was declared per backend but referenced nowhere in the
gateway, so oversized requests were routed anyway and failed upstream — and
failover would try every other backend that also couldn't fit them. These
tests pin the corrected behaviour: candidates that cannot fit the request are
excluded, and if none fit the request is rejected up front.
"""

from typing import Any, AsyncGenerator, Dict, List

import pytest

from gateway.adapters.base import BaseAdapter, NormalizedResponse
from gateway.ledger.store import LedgerStore
from gateway.policy.circuit_breaker import CircuitBreakerRegistry
from gateway.policy.router import ContextLengthExceededException, Router


class _Adapter(BaseAdapter):
    async def complete(self, messages, **kwargs) -> NormalizedResponse: ...

    async def complete_stream(
        self, messages, **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        yield {}

    async def health_check(self) -> bool:
        return True


def _adapter(aid: str, context_length: int, cost: float = 0.001) -> _Adapter:
    return _Adapter(
        {
            "id": aid,
            "model": aid,
            "cost_per_1k_prompt": cost,
            "cost_per_1k_completion": cost,
            "context_length": context_length,
        }
    )


def _router(*adapters, strategy="cost_first") -> Router:
    return Router(
        adapters=list(adapters),
        circuit_registry=CircuitBreakerRegistry(ledger=LedgerStore(":memory:")),
        strategy=strategy,
    )


def _msgs(approx_tokens: int) -> List[Dict[str, str]]:
    # "word " is one token under cl100k_base.
    return [{"role": "user", "content": "word " * approx_tokens}]


@pytest.mark.asyncio
async def test_backend_too_small_is_excluded():
    small = _adapter("small", context_length=100, cost=0.001)
    large = _adapter("large", context_length=100_000, cost=0.999)

    ranked = await _router(small, large).get_ranked_adapters(messages=_msgs(500))

    # `small` is far cheaper and would win on cost_first, but cannot fit.
    assert [a.id for a in ranked] == ["large"]


@pytest.mark.asyncio
async def test_fitting_backend_still_selected_normally():
    small = _adapter("small", context_length=100_000, cost=0.001)
    large = _adapter("large", context_length=100_000, cost=0.999)

    ranked = await _router(small, large).get_ranked_adapters(messages=_msgs(10))

    assert ranked[0].id == "small"


@pytest.mark.asyncio
async def test_max_tokens_counts_against_the_window():
    """A prompt that fits alone can still fail once the reply is reserved."""
    adapter = _adapter("only", context_length=1000)
    router = _router(adapter)

    assert await router.get_ranked_adapters(messages=_msgs(500))

    with pytest.raises(ContextLengthExceededException):
        await router.get_ranked_adapters(messages=_msgs(500), max_tokens=900)


@pytest.mark.asyncio
async def test_raises_when_no_backend_can_fit():
    router = _router(_adapter("a", context_length=100), _adapter("b", 200))

    with pytest.raises(ContextLengthExceededException) as exc:
        await router.get_ranked_adapters(messages=_msgs(5000))

    # Message should name the largest window and the shortfall, not just fail.
    assert "200" in str(exc.value)
    assert "over by" in str(exc.value)


@pytest.mark.asyncio
async def test_unknown_context_length_is_treated_as_unconstrained():
    """A backend that declares no window must not be silently excluded."""
    router = _router(_adapter("unknown", context_length=0))

    ranked = await router.get_ranked_adapters(messages=_msgs(50_000))

    assert [a.id for a in ranked] == ["unknown"]


@pytest.mark.asyncio
async def test_context_filter_is_hard_not_best_effort():
    """Other filters fall back to all candidates when nothing matches; this
    one must not, since an oversized request cannot succeed anywhere."""
    router = _router(_adapter("a", context_length=50))

    with pytest.raises(ContextLengthExceededException):
        await router.get_ranked_adapters(messages=_msgs(5000))
