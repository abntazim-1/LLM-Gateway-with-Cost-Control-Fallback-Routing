import pytest
from gateway.policy.router import Router, NoAvailableBackendException
from gateway.policy.circuit_breaker import CircuitBreakerRegistry
from gateway.adapters.base import BaseAdapter, NormalizedResponse
from gateway.ledger.store import LedgerStore
from typing import List, Dict, Any, AsyncGenerator

class MockAdapter(BaseAdapter):
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

    async def complete_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        yield {
            "id": "mock-stream-id",
            "choices": [{"delta": {"content": "Hello from mock stream!"}}]
        }

    async def health_check(self) -> bool:
        return True

class MockFailsAdapter(BaseAdapter):
    async def complete(self, messages: List[Dict[str, str]], **kwargs) -> NormalizedResponse:
        raise Exception("Failure")
        
    async def complete_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        # Empty generator that fails immediately
        if False:
            yield {}
        raise Exception("Stream startup failure")

    async def health_check(self) -> bool:
        return False

@pytest.mark.asyncio
async def test_router_cost_first_ranking():
    # Setup database and circuit breakers
    ledger = LedgerStore(":memory:")
    registry = CircuitBreakerRegistry(ledger=ledger)
    
    # Create adapters with different prompt costs
    adapter_cheap = MockAdapter({"id": "cheap", "model": "m1", "cost_per_1k_prompt": 0.001, "cost_per_1k_completion": 0.002})
    adapter_expensive = MockAdapter({"id": "expensive", "model": "m2", "cost_per_1k_prompt": 0.01, "cost_per_1k_completion": 0.02})
    
    router = Router(adapters=[adapter_expensive, adapter_cheap], circuit_registry=registry, strategy="cost_first")
    
    ranked = await router.get_ranked_adapters()
    
    assert len(ranked) == 2
    assert ranked[0].id == "cheap"
    assert ranked[1].id == "expensive"

@pytest.mark.asyncio
async def test_router_excludes_tripped_circuit_breaker():
    ledger = LedgerStore(":memory:")
    registry = CircuitBreakerRegistry(ledger=ledger)
    
    adapter1 = MockAdapter({"id": "adapter1", "model": "m1", "cost_per_1k_prompt": 0.001, "cost_per_1k_completion": 0.002})
    adapter2 = MockAdapter({"id": "adapter2", "model": "m2", "cost_per_1k_prompt": 0.005, "cost_per_1k_completion": 0.006})
    
    # Trip circuit breaker for adapter1
    breaker1 = registry.get_breaker("adapter1")
    # failure_threshold is 3 by default. Trip it.
    await breaker1.record_failure()
    await breaker1.record_failure()
    await breaker1.record_failure()
    
    router = Router(adapters=[adapter1, adapter2], circuit_registry=registry, strategy="cost_first")
    
    ranked = await router.get_ranked_adapters()
    
    assert len(ranked) == 1
    assert ranked[0].id == "adapter2"

@pytest.mark.asyncio
async def test_router_no_available_backend_raises_exception():
    ledger = LedgerStore(":memory:")
    registry = CircuitBreakerRegistry(ledger=ledger)
    
    adapter1 = MockAdapter({"id": "adapter1", "model": "m1", "cost_per_1k_prompt": 0.001, "cost_per_1k_completion": 0.002})
    
    breaker = registry.get_breaker("adapter1")
    await breaker.record_failure()
    await breaker.record_failure()
    await breaker.record_failure()
    
    router = Router(adapters=[adapter1], circuit_registry=registry)
    
    with pytest.raises(NoAvailableBackendException):
        await router.get_ranked_adapters()

@pytest.mark.asyncio
async def test_router_execute_stream_logs_spend():
    ledger = LedgerStore(":memory:")
    # Initialize budget configuration
    budgets = [{"api_key": "sk-stream-test", "daily_limit_usd": 10.0, "monthly_limit_usd": 100.0}]
    await ledger.load_budgets_from_config(budgets)
    
    registry = CircuitBreakerRegistry(ledger=ledger)
    adapter = MockAdapter({"id": "test-stream-backend", "model": "m1", "cost_per_1k_prompt": 0.002, "cost_per_1k_completion": 0.004})
    
    router = Router(adapters=[adapter], circuit_registry=registry)
    
    chunks = []
    async for chunk in router.execute_stream(
        messages=[{"role": "user", "content": "hello"}],
        api_key="sk-stream-test"
    ):
        chunks.append(chunk)
        
    assert len(chunks) == 1
    assert chunks[0]["id"] == "mock-stream-id"
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello from mock stream!"
    
    # Wait for async thread DB write to be captured
    import asyncio
    await asyncio.sleep(0.1)
    
    # Verify spend was recorded in the database
    requests = await ledger.get_all_requests()
    assert len(requests) == 1
    assert requests[0]["backend"] == "test-stream-backend"
    assert requests[0]["cost_usd"] > 0
