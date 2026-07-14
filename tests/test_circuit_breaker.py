import time
import pytest
from gateway.policy.circuit_breaker import CircuitBreaker, CircuitState
from gateway.ledger.store import LedgerStore

def test_circuit_breaker_state_machine():
    ledger = LedgerStore(":memory:")
    cb = CircuitBreaker(backend_id="test_backend", ledger=ledger, failure_threshold=2, cooldown_sec=1)
    
    assert cb.state == CircuitState.CLOSED
    assert cb.can_request() == True
    
    # First failure
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    assert cb.can_request() == True
    
    # Second failure (hits threshold)
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.can_request() == False
    
    # Wait for cooldown
    time.sleep(1.1)
    
    # Now it should allow one request through (HALF_OPEN)
    assert cb.can_request() == True
    assert cb.state == CircuitState.HALF_OPEN
    
    # Subsequent requests are blocked while HALF_OPEN
    assert cb.can_request() == False
    
    # If the test request succeeds, we close
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.can_request() == True
