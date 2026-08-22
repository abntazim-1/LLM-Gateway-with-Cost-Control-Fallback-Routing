from typing import Any, Dict, List

import pytest

from gateway.adapters.base import BaseAdapter, NormalizedResponse
from gateway.ledger.store import LedgerStore
from gateway.policy.circuit_breaker import CircuitBreakerRegistry
from gateway.policy.router import NoAvailableBackendException, Router


class MockRoutingAdapter(BaseAdapter):
    async def complete(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> NormalizedResponse:
        return NormalizedResponse(
            id=self.id,
            backend_id=self.id,
            model=self.model,
            messages=[],
            prompt_tokens=10,
            completion_tokens=10,
            cost_usd=0.01,
            latency_ms=100.0,
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
    cheap = MockRoutingAdapter(
        {
            "id": "cheap-model",
            "model": "m1",
            "cost_per_1k_prompt": 0.001,
            "cost_per_1k_completion": 0.002,
        }
    )
    premium = MockRoutingAdapter(
        {
            "id": "premium-model",
            "model": "m2",
            "cost_per_1k_prompt": 0.01,
            "cost_per_1k_completion": 0.02,
        }
    )

    router = Router(
        adapters=[premium, cheap], circuit_registry=registry, strategy="complexity"
    )

    # Simple message should route to cheap-model first
    ranked = await router.get_ranked_adapters(
        messages=[{"role": "user", "content": "hi"}]
    )
    assert len(ranked) == 2
    assert ranked[0].id == "cheap-model"
    assert ranked[1].id == "premium-model"

    # Conversational 'why' query should NOT trigger premium model
    ranked_why = await router.get_ranked_adapters(
        messages=[{"role": "user", "content": "Why is the sky blue?"}]
    )
    assert ranked_why[0].id == "cheap-model"

    ranked_short = await router.get_ranked_adapters(
        messages=[{"role": "user", "content": "Why?"}]
    )
    assert ranked_short[0].id == "cheap-model"


@pytest.mark.asyncio
async def test_router_complexity_routing_complex():
    ledger = LedgerStore(":memory:")
    registry = CircuitBreakerRegistry(ledger=ledger)

    cheap = MockRoutingAdapter(
        {
            "id": "cheap-model",
            "model": "m1",
            "cost_per_1k_prompt": 0.001,
            "cost_per_1k_completion": 0.002,
        }
    )
    premium = MockRoutingAdapter(
        {
            "id": "premium-model",
            "model": "m2",
            "cost_per_1k_prompt": 0.01,
            "cost_per_1k_completion": 0.02,
        }
    )

    router = Router(
        adapters=[premium, cheap], circuit_registry=registry, strategy="complexity"
    )

    # Message with reasoning keyword and length should route to premium-model first
    ranked = await router.get_ranked_adapters(
        messages=[
            {
                "role": "user",
                "content": "Please write a high performance code to optimize and benchmark this algorithm proof.",
            }
        ]
    )
    assert len(ranked) == 2
    assert ranked[0].id == "premium-model"
    assert ranked[1].id == "cheap-model"

    # Multi-line code block routes to premium
    code_prompt = "Review this code:\n```python\ndef solve(matrix):\n    return np.linalg.eig(matrix)\n```"
    ranked_code = await router.get_ranked_adapters(
        messages=[{"role": "user", "content": code_prompt}]
    )
    assert ranked_code[0].id == "premium-model"

    # Large token volume routes to premium
    large_prompt = "Analyze this text in detail: " + (
        "lorem ipsum dolor sit amet " * 150
    )
    ranked_large = await router.get_ranked_adapters(
        messages=[{"role": "user", "content": large_prompt}]
    )
    assert ranked_large[0].id == "premium-model"


@pytest.mark.asyncio
async def test_router_capability_aware_routing():
    ledger = LedgerStore(":memory:")
    registry = CircuitBreakerRegistry(ledger=ledger)

    standard = MockRoutingAdapter(
        {
            "id": "standard-model",
            "model": "m1",
            "cost_per_1k_prompt": 0.001,
            "cost_per_1k_completion": 0.002,
            "capabilities": [],
        }
    )
    json_capable = MockRoutingAdapter(
        {
            "id": "json-model",
            "model": "m2",
            "cost_per_1k_prompt": 0.005,
            "cost_per_1k_completion": 0.010,
            "capabilities": ["json_mode"],
        }
    )

    router = Router(
        adapters=[standard, json_capable],
        circuit_registry=registry,
        strategy="cost_first",
    )

    # Normal request selects standard (cheapest)
    ranked_normal = await router.get_ranked_adapters(
        messages=[{"role": "user", "content": "hello"}]
    )
    assert ranked_normal[0].id == "standard-model"

    # JSON mode request filters for json_mode capability
    ranked_json = await router.get_ranked_adapters(
        messages=[{"role": "user", "content": "return json"}],
        response_format={"type": "json_object"},
    )
    assert len(ranked_json) == 1
    assert ranked_json[0].id == "json-model"


@pytest.mark.asyncio
async def test_router_required_capabilities_filtering():
    ledger = LedgerStore(":memory:")
    registry = CircuitBreakerRegistry(ledger=ledger)

    basic = MockRoutingAdapter(
        {
            "id": "basic-model",
            "model": "m1",
            "cost_per_1k_prompt": 0.001,
            "cost_per_1k_completion": 0.002,
            "capabilities": ["json_mode"],
        }
    )
    advanced = MockRoutingAdapter(
        {
            "id": "advanced-model",
            "model": "m2",
            "cost_per_1k_prompt": 0.01,
            "cost_per_1k_completion": 0.02,
            "capabilities": ["json_mode", "vision", "function_calling"],
        }
    )

    router = Router(
        adapters=[basic, advanced], circuit_registry=registry, strategy="cost_first"
    )

    # Explicit capability requirement filters to the adapter advertising it
    ranked = await router.get_ranked_adapters(
        messages=[{"role": "user", "content": "describe this image"}],
        required_capabilities=["vision"],
    )
    assert [a.id for a in ranked] == ["advanced-model"]

    # Multiple required capabilities must ALL be advertised
    ranked_multi = await router.get_ranked_adapters(
        messages=[{"role": "user", "content": "analyze"}],
        required_capabilities=["vision", "function_calling"],
    )
    assert [a.id for a in ranked_multi] == ["advanced-model"]

    # No backend matches -> silent fallback to all candidates (availability)
    ranked_fallback = await router.get_ranked_adapters(
        messages=[{"role": "user", "content": "analyze"}],
        required_capabilities=["embeddings"],
    )
    assert {a.id for a in ranked_fallback} == {"basic-model", "advanced-model"}

    # Inferred json_mode combines with explicit requirements
    ranked_combined = await router.get_ranked_adapters(
        messages=[{"role": "user", "content": "return json"}],
        response_format={"type": "json_object"},
        required_capabilities=["vision"],
    )
    assert [a.id for a in ranked_combined] == ["advanced-model"]
