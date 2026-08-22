import time

import pytest

from gateway.ledger.store import LedgerStore
from gateway.policy.circuit_breaker import CircuitBreaker, CircuitState


@pytest.mark.asyncio
async def test_circuit_breaker_state_machine():
    ledger = LedgerStore(":memory:")
    cb = CircuitBreaker(
        backend_id="test_backend", ledger=ledger, failure_threshold=2, cooldown_sec=1
    )

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


@pytest.mark.asyncio
async def test_circuit_breaker_recovery_via_health_check():
    """Health check loop calls record_success() directly (bypassing can_request).
    This test ensures an OPEN breaker recovers when the backend comes back healthy
    regardless of whether the cooldown has elapsed."""
    ledger = LedgerStore(":memory:")
    cb = CircuitBreaker(
        backend_id="hc_backend", ledger=ledger, failure_threshold=2, cooldown_sec=9999
    )

    # Trip the breaker
    await cb.record_failure()
    await cb.record_failure()
    assert await cb.get_state() == CircuitState.OPEN
    assert await cb.can_request() == False

    # Simulate health check loop detecting the backend is healthy again
    # (cooldown has NOT elapsed - 9999 sec - so can_request alone can't recover it)
    await cb.record_success()

    # Breaker must now be CLOSED and accepting traffic
    assert await cb.get_state() == CircuitState.CLOSED
    assert await cb.can_request() == True


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_failure_reopens():
    """A failure during HALF_OPEN must reopen the breaker immediately."""
    ledger = LedgerStore(":memory:")
    cb = CircuitBreaker(
        backend_id="ho_backend", ledger=ledger, failure_threshold=2, cooldown_sec=1
    )

    await cb.record_failure()
    await cb.record_failure()
    assert await cb.get_state() == CircuitState.OPEN

    time.sleep(1.1)
    assert await cb.can_request() == True  # transitions to HALF_OPEN
    assert await cb.get_state() == CircuitState.HALF_OPEN

    # Probe request fails → reopen immediately
    await cb.record_failure()
    assert await cb.get_state() == CircuitState.OPEN
    assert await cb.can_request() == False
