import pytest
from typing import List, Dict, Any, AsyncGenerator
from gateway.policy.router import Router
from gateway.policy.circuit_breaker import CircuitBreakerRegistry
from gateway.adapters.base import BaseAdapter, NormalizedResponse, NormalizedMessage
from gateway.ledger.store import LedgerStore

class ModelCapturingAdapter(BaseAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.captured_model = None

    async def complete(self, messages: List[Dict[str, str]], **kwargs) -> NormalizedResponse:
        self.captured_model = kwargs.get("model")
        return NormalizedResponse(
            id="test-id",
            backend_id=self.id,
            model=self.model,
            messages=[NormalizedMessage(role="assistant", content="OK")],
            prompt_tokens=10,
            completion_tokens=10,
            cost_usd=0.001,
            latency_ms=20.0
        )

    async def complete_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        self.captured_model = kwargs.get("model")
        yield {"id": "chunk-1", "choices": [{"delta": {"content": "OK"}}]}

    async def health_check(self) -> bool:
        return True

@pytest.mark.asyncio
async def test_cross_provider_fallback_model_mapping():
    ledger = LedgerStore(":memory:")
    registry = CircuitBreakerRegistry(ledger=ledger)
    
    # Primary adapter configured for anthropic claude model
    adapter = ModelCapturingAdapter({
        "id": "anthropic-primary",
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20240620",
        "cost_per_1k_prompt": 0.003,
        "cost_per_1k_completion": 0.015
    })
    
    router = Router(adapters=[adapter], circuit_registry=registry, strategy="cost_first")
    
    # Client sends request specifying 'gpt-4o'
    response = await router.execute([{"role": "user", "content": "hello"}], model="gpt-4o")
    
    # Verify the router mapped the model parameter to the adapter's target model 'claude-3-5-sonnet-20240620'
    assert adapter.captured_model == "claude-3-5-sonnet-20240620"
    assert response.backend_id == "anthropic-primary"

@pytest.mark.asyncio
async def test_round_robin_strategy_ranking():
    ledger = LedgerStore(":memory:")
    registry = CircuitBreakerRegistry(ledger=ledger)
    
    a1 = ModelCapturingAdapter({"id": "a1", "model": "m1", "cost_per_1k_prompt": 0.01})
    a2 = ModelCapturingAdapter({"id": "a2", "model": "m2", "cost_per_1k_prompt": 0.01})
    
    router = Router(adapters=[a1, a2], circuit_registry=registry, strategy="round_robin")
    
    ranked_1 = await router.get_ranked_adapters()
    ranked_2 = await router.get_ranked_adapters()
    
    assert ranked_1[0].id == "a1"
    assert ranked_2[0].id == "a2"
