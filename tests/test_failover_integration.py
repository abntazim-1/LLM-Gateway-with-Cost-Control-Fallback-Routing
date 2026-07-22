import pytest
from gateway.policy.router import Router, NoAvailableBackendException
from gateway.policy.circuit_breaker import CircuitBreakerRegistry, CircuitState
from gateway.adapters.base import BaseAdapter, NormalizedResponse, AdapterException
from gateway.ledger.store import LedgerStore
from typing import List, Dict, Any, AsyncGenerator

class FailsAdapter(BaseAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.attempts = 0

    async def complete(self, messages: List[Dict[str, str]], **kwargs) -> NormalizedResponse:
        self.attempts += 1
        raise AdapterException("Simulated backend connection failure")

    async def complete_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        self.attempts += 1
        if False:
            yield {}
        raise AdapterException("Simulated stream startup failure")

    async def health_check(self) -> bool:
        return False

class SucceedsAdapter(BaseAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.attempts = 0

    async def complete(self, messages: List[Dict[str, str]], **kwargs) -> NormalizedResponse:
        self.attempts += 1
        from gateway.adapters.base import NormalizedMessage
        return NormalizedResponse(
            id="success-id",
            backend_id=self.id,
            model=self.model,
            messages=[NormalizedMessage(role="assistant", content="Hello from backup!")],
            prompt_tokens=5,
            completion_tokens=5,
            cost_usd=0.005,
            latency_ms=50.0
        )

    async def complete_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        self.attempts += 1
        yield {
            "id": "backup-stream-id",
            "choices": [{"delta": {"content": "Hello from streaming backup!"}}]
        }

    async def health_check(self) -> bool:
        return True

@pytest.mark.asyncio
async def test_router_failover_integration():
    ledger = LedgerStore(":memory:")
    registry = CircuitBreakerRegistry(ledger=ledger, failure_threshold=2, cooldown_sec=10)
    
    # Primary adapter will fail, secondary will succeed.
    primary = FailsAdapter({"id": "primary", "model": "m1", "cost_per_1k_prompt": 0.001, "cost_per_1k_completion": 0.002})
    backup = SucceedsAdapter({"id": "backup", "model": "m2", "cost_per_1k_prompt": 0.01, "cost_per_1k_completion": 0.02})
    
    router = Router(adapters=[primary, backup], circuit_registry=registry, strategy="cost_first")
    
    # Execute should automatically catch the error on primary, fail over, try backup, and succeed
    response = await router.execute([{"role": "user", "content": "test message"}])
    
    assert response.backend_id == "backup"
    assert response.id == "success-id"
    assert len(response.messages) == 1
    assert response.messages[0].content == "Hello from backup!"
    
    primary_breaker = registry.get_breaker("primary")
    state_info = await primary_breaker._get_state()
    
    # Verify primary has recorded failures
    assert state_info[1] >= 1

@pytest.mark.asyncio
async def test_router_streaming_failover_integration():
    ledger = LedgerStore(":memory:")
    registry = CircuitBreakerRegistry(ledger=ledger, failure_threshold=2, cooldown_sec=10)
    
    primary = FailsAdapter({"id": "primary", "model": "m1", "cost_per_1k_prompt": 0.001, "cost_per_1k_completion": 0.002})
    backup = SucceedsAdapter({"id": "backup", "model": "m2", "cost_per_1k_prompt": 0.01, "cost_per_1k_completion": 0.02})
    
    router = Router(adapters=[primary, backup], circuit_registry=registry, strategy="cost_first")
    
    # execute_stream should catch primary's startup failure, fall back to backup, and stream successfully
    chunks = []
    async for chunk in router.execute_stream([{"role": "user", "content": "test streaming message"}]):
        chunks.append(chunk)
        
    assert len(chunks) == 1
    assert chunks[0]["id"] == "backup-stream-id"
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello from streaming backup!"
    
    primary_breaker = registry.get_breaker("primary")
    state_info = await primary_breaker._get_state()
    
    # Verify primary circuit breaker recorded the failure
    assert state_info[1] >= 1
