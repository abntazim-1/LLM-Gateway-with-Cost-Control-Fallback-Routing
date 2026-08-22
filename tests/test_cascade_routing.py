from typing import Any, AsyncGenerator, Dict, List

import pytest

from gateway.adapters.base import BaseAdapter, NormalizedMessage, NormalizedResponse
from gateway.ledger.store import LedgerStore
from gateway.policy.circuit_breaker import CircuitBreakerRegistry
from gateway.policy.router import Router


class ScriptedAdapter(BaseAdapter):
    """Returns a fixed reply so cascade escalation can be tested deterministically."""

    def __init__(self, config: Dict[str, Any], reply: str, cost: float = 0.01):
        super().__init__(config)
        self.reply = reply
        self.cost = cost
        self.call_count = 0

    async def complete(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> NormalizedResponse:
        self.call_count += 1
        return NormalizedResponse(
            id=self.id,
            backend_id=self.id,
            model=self.model,
            messages=[NormalizedMessage(role="assistant", content=self.reply)],
            prompt_tokens=10,
            completion_tokens=10,
            cost_usd=self.cost,
            latency_ms=100.0,
        )

    async def complete_stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        yield {"id": "stream-id", "choices": [{"delta": {"content": self.reply}}]}

    async def health_check(self) -> bool:
        return True


CHEAP_COST = 0.005
PREMIUM_COST = 0.05


def _make_router(cheap_reply: str, premium_reply: str = "A properly detailed answer."):
    ledger = LedgerStore(":memory:")
    registry = CircuitBreakerRegistry(ledger=ledger)
    cheap = ScriptedAdapter(
        {
            "id": "cheap",
            "model": "m1",
            "cost_per_1k_prompt": 0.001,
            "cost_per_1k_completion": 0.002,
        },
        reply=cheap_reply,
        cost=CHEAP_COST,
    )
    premium = ScriptedAdapter(
        {
            "id": "premium",
            "model": "m2",
            "cost_per_1k_prompt": 0.01,
            "cost_per_1k_completion": 0.02,
        },
        reply=premium_reply,
        cost=PREMIUM_COST,
    )
    router = Router(
        adapters=[premium, cheap], circuit_registry=registry, strategy="cascade"
    )
    return router, cheap, premium


@pytest.mark.asyncio
async def test_cascade_escalates_on_inadequate_response():
    router, cheap, premium = _make_router(cheap_reply="I don't know.")

    response = await router.execute(messages=[{"role": "user", "content": "hi"}])

    assert response.backend_id == "premium"
    assert cheap.call_count == 1
    assert premium.call_count == 1


@pytest.mark.asyncio
async def test_cascade_bills_both_attempts_on_escalation():
    # The discarded cheap attempt still cost real money against a real
    # backend — it must be folded into the final billed cost, not dropped.
    router, cheap, premium = _make_router(cheap_reply="I don't know.")

    response = await router.execute(messages=[{"role": "user", "content": "hi"}])

    assert response.cost_usd == pytest.approx(CHEAP_COST + PREMIUM_COST)


@pytest.mark.asyncio
async def test_cascade_bills_only_one_attempt_when_no_escalation():
    router, cheap, premium = _make_router(
        cheap_reply="Here is a complete, well-formed answer to your question."
    )

    response = await router.execute(messages=[{"role": "user", "content": "hi"}])

    assert response.cost_usd == pytest.approx(CHEAP_COST)


@pytest.mark.asyncio
async def test_cascade_escalates_on_too_short_response():
    router, cheap, premium = _make_router(cheap_reply="ok")

    response = await router.execute(messages=[{"role": "user", "content": "hi"}])

    assert response.backend_id == "premium"


@pytest.mark.asyncio
async def test_cascade_stays_cheap_on_adequate_response():
    router, cheap, premium = _make_router(
        cheap_reply="Here is a complete, well-formed answer to your question."
    )

    response = await router.execute(messages=[{"role": "user", "content": "hi"}])

    assert response.backend_id == "cheap"
    assert premium.call_count == 0


@pytest.mark.asyncio
async def test_cascade_returns_last_candidate_even_if_still_inadequate():
    # No further backend to escalate to past the last one — must still return
    # something rather than raising, even if the heuristic isn't satisfied.
    router, cheap, premium = _make_router(
        cheap_reply="I don't know.", premium_reply="I'm not sure either."
    )

    response = await router.execute(messages=[{"role": "user", "content": "hi"}])

    assert response.backend_id == "premium"
