import time
import pytest
from gateway.policy.circuit_breaker import CircuitBreaker, CircuitState
from gateway.ledger.store import LedgerStore

@pytest.mark.asyncio
async def test_circuit_breaker_state_machine():
    ledger = LedgerStore(":memory:")
    cb = CircuitBreaker(backend_id="test_backend", ledger=ledger, failure_threshold=2, cooldown_sec=1)
    
    assert await cb.get_state() == CircuitState.CLOSED
    assert await cb.can_request() == True
    
    # First failure
    await cb.record_failure()
    assert await cb.get_state() == CircuitState.CLOSED
    assert await cb.can_request() == True
    
    # Second failure (hits threshold)
    await cb.record_failure()
    assert await cb.get_state() == CircuitState.OPEN
    assert await cb.can_request() == False
    
    # Wait for cooldown
    time.sleep(1.1)
    
    # Now it should allow one request through (HALF_OPEN)
    assert await cb.can_request() == True
    assert await cb.get_state() == CircuitState.HALF_OPEN
    
    # Subsequent requests are blocked while HALF_OPEN
    assert await cb.can_request() == False
    
    # If the test request succeeds, we close
    await cb.record_success()
    assert await cb.get_state() == CircuitState.CLOSED
    assert await cb.can_request() == True
