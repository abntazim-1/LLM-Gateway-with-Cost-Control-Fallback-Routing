import pytest
from gateway.policy.router import Router, NoAvailableBackendException
from gateway.policy.circuit_breaker import CircuitBreakerRegistry
from gateway.adapters.base import BaseAdapter, NormalizedResponse
from gateway.ledger.store import LedgerStore
from typing import List, Dict, Any

class MockRoutingAdapter(BaseAdapter):
    async def complete(self, messages: List[Dict[str, str]], **kwargs) -> NormalizedResponse:
        return NormalizedResponse(
            id=self.id,
            backend_id=self.id,
            model=self.model,
            messages=[],
            prompt_tokens=10,
            completion_tokens=10,
            cost_usd=0.01,
            latency_ms=100.0
        )

    async def complete_stream(self, messages: List[Dict[str, str]], **kwargs):
        raise NotImplementedError()

    async def health_check(self) -> bool:
        return True

@pytest.mark.asyncio
async def test_router_complexity_routing_simple():
    ledger = LedgerStore(":memory:")
    registry = CircuitBreakerRegistry(ledger=ledger)
    
    # Cheap adapter has low prompt cost, expensive has high prompt cost
    cheap = MockRoutingAdapter({"id": "cheap-model", "model": "m1", "cost_per_1k_prompt": 0.001, "cost_per_1k_completion": 0.002})
    premium = MockRoutingAdapter({"id": "premium-model", "model": "m2", "cost_per_1k_prompt": 0.01, "cost_per_1k_completion": 0.02})
    
    router = Router(adapters=[premium, cheap], circuit_registry=registry, strategy="complexity")
    
    # Simple message should route to cheap-model first
    ranked = await router.get_ranked_adapters(messages=[{"role": "user", "content": "hi"}])
    assert len(ranked) == 2
    assert ranked[0].id == "cheap-model"
    assert ranked[1].id == "premium-model"

@pytest.mark.asyncio
async def test_router_complexity_routing_complex():
    ledger = LedgerStore(":memory:")
    registry = CircuitBreakerRegistry(ledger=ledger)
    
    cheap = MockRoutingAdapter({"id": "cheap-model", "model": "m1", "cost_per_1k_prompt": 0.001, "cost_per_1k_completion": 0.002})
    premium = MockRoutingAdapter({"id": "premium-model", "model": "m2", "cost_per_1k_prompt": 0.01, "cost_per_1k_completion": 0.02})
    
    router = Router(adapters=[premium, cheap], circuit_registry=registry, strategy="complexity")
    
    # Message with reasoning keyword and length should route to premium-model first
    ranked = await router.get_ranked_adapters(
        messages=[{"role": "user", "content": "Please write a high performance code to optimize and solve this numerical analysis proof."}]
    )
    assert len(ranked) == 2
    assert ranked[0].id == "premium-model"
    assert ranked[1].id == "cheap-model"
