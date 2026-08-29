"""Time budget and error classification during failover.

Two failures the router previously had no opinion about:

* how long a request may take *in total*, as opposed to per attempt
* the difference between a backend that is broken and one that is busy
"""

import asyncio
import time
from typing import Any, AsyncGenerator, Dict, List

import pytest

from gateway.adapters.base import AdapterException, BaseAdapter, NormalizedResponse
from gateway.ledger.store import LedgerStore
from gateway.policy.circuit_breaker import CircuitBreakerRegistry
from gateway.policy.router import (
    DeadlineExceededException,
    Router,
    is_rate_limit_error,
    retry_after_seconds,
)


class _Adapter(BaseAdapter):
    """Backend whose failure mode is scripted."""

    def __init__(self, config, behaviour="ok", delay=0.0):
        super().__init__(config)
        self.behaviour = behaviour
        self.delay = delay
        self.calls = 0

    async def complete(self, messages, **kwargs) -> NormalizedResponse:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.behaviour == "rate_limited":
            raise AdapterException("Groq request failed: 429 Too Many Requests")
        if self.behaviour == "broken":
            raise AdapterException("Internal server error")
        return NormalizedResponse(
            id=self.id,
            backend_id=self.id,
            model=self.model,
            messages=[],
            prompt_tokens=1,
            completion_tokens=1,
            cost_usd=0.01,
            latency_ms=1.0,
        )

    async def complete_stream(
        self, messages, **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        yield {}

    async def health_check(self) -> bool:
        return True


def _cfg(aid, **over):
    base = {
        "id": aid,
        "model": aid,
        "cost_per_1k_prompt": 0.001,
        "cost_per_1k_completion": 0.002,
    }
    base.update(over)
    return base


def _router(*adapters, **kw):
    return Router(
        adapters=list(adapters),
        circuit_registry=CircuitBreakerRegistry(ledger=LedgerStore(":memory:")),
        strategy="cost_first",
        **kw,
    )


MSG = [{"role": "user", "content": "hi"}]


# ── Total time budget ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_request_stops_at_its_deadline_rather_than_per_attempt_timeout():
    """Per-attempt timeouts bound nothing useful on their own.

    Three attempts against two backends plus backoff ran to minutes before the
    caller saw an error; the deadline is what makes the wait predictable.
    """
    slow = _Adapter(_cfg("slow"), delay=5.0)
    router = _router(slow, timeout_sec=5.0, total_deadline_sec=1.5)

    started = time.monotonic()
    with pytest.raises(DeadlineExceededException):
        await router.execute(messages=MSG)
    elapsed = time.monotonic() - started

    # Bounded by the deadline, not by attempts x timeout.
    assert elapsed < 4.0, f"took {elapsed:.1f}s despite a 1.5s budget"


@pytest.mark.asyncio
async def test_deadline_is_not_hit_by_a_request_that_succeeds():
    fast = _Adapter(_cfg("fast"))
    router = _router(fast, timeout_sec=5.0, total_deadline_sec=5.0)

    response = await router.execute(messages=MSG)

    assert response.backend_id == "fast"


@pytest.mark.asyncio
async def test_deadline_defaults_are_derived_from_the_attempt_timeout():
    """A config that never sets a deadline must still get one."""
    router = _router(_Adapter(_cfg("a")), timeout_sec=30.0)

    assert router.total_deadline_sec == 60.0


# ── Rate limiting is not breakage ────────────────────────────────────────


@pytest.mark.parametrize(
    "message",
    [
        "Groq request failed: 429 Too Many Requests",
        "rate limit exceeded for this model",
        "Error: quota exceeded",
    ],
)
def test_throttling_is_recognised(message):
    assert is_rate_limit_error(AdapterException(message))


@pytest.mark.parametrize(
    "message",
    ["Internal server error", "connection refused", "model not found"],
)
def test_ordinary_failures_are_not_mistaken_for_throttling(message):
    assert not is_rate_limit_error(AdapterException(message))


def test_retry_after_is_read_from_the_provider_when_present():
    class _Resp:
        headers = {"Retry-After": "7"}

    exc = AdapterException("429")
    exc.response = _Resp()

    assert retry_after_seconds(exc) == 7.0


def test_retry_after_absent_is_reported_as_unknown():
    assert retry_after_seconds(AdapterException("429")) is None


# The breaker counts failed *requests*, not failed attempts, and its
# threshold is 3 — so a single request never opens it whatever the error.
# Both cases below drive enough requests to cross that threshold, otherwise
# they would pass without discriminating between the two error kinds at all.
_FAILURES_TO_TRIP = 3


@pytest.mark.asyncio
async def test_throttled_backend_keeps_its_breaker_closed():
    """A rate-limited backend is healthy and busy.

    Opening its breaker routes traffic to a pricier backend for the cooldown,
    so hitting a free tier's limit would quietly become a bill.
    """
    throttled = _Adapter(_cfg("cheap"), behaviour="rate_limited")
    healthy = _Adapter(_cfg("pricey", cost_per_1k_prompt=0.05))
    router = _router(throttled, healthy, timeout_sec=2.0, total_deadline_sec=20.0)

    for _ in range(_FAILURES_TO_TRIP):
        response = await router.execute(messages=MSG)
        assert response.backend_id == "pricey"  # served, so availability is kept

    breaker = router.circuit_registry.get_breaker("cheap")
    assert await breaker.can_request(), "throttling must not open the breaker"


@pytest.mark.asyncio
async def test_a_genuinely_broken_backend_does_open_its_breaker():
    """The contrast case — this is what the breaker exists for."""
    broken = _Adapter(_cfg("cheap"), behaviour="broken")
    healthy = _Adapter(_cfg("pricey", cost_per_1k_prompt=0.05))
    router = _router(broken, healthy, timeout_sec=2.0, total_deadline_sec=20.0)

    for _ in range(_FAILURES_TO_TRIP):
        await router.execute(messages=MSG)

    breaker = router.circuit_registry.get_breaker("cheap")
    assert not await breaker.can_request(), "a failing backend must be cut off"


# ── Metrics ──────────────────────────────────────────────────────────────
# Failover and throttling both keep requests succeeding, which is the point —
# and also why they are invisible without a counter. An outage would otherwise
# surface only as next month's bill.


def _counter_value(counter, **labels) -> float:
    child = counter.labels(**labels) if labels else counter
    return child._value.get()


@pytest.mark.asyncio
async def test_failover_increments_the_fallback_counter():
    from gateway.telemetry.metrics import FALLBACK_TOTAL

    broken = _Adapter(_cfg("cheap"), behaviour="broken")
    healthy = _Adapter(_cfg("pricey", cost_per_1k_prompt=0.05))
    router = _router(broken, healthy, timeout_sec=2.0, total_deadline_sec=20.0)

    before = _counter_value(FALLBACK_TOTAL, backend="pricey")
    await router.execute(messages=MSG)

    assert _counter_value(FALLBACK_TOTAL, backend="pricey") == before + 1


@pytest.mark.asyncio
async def test_throttling_increments_its_own_counter():
    from gateway.telemetry.metrics import THROTTLED_TOTAL

    throttled = _Adapter(_cfg("cheap"), behaviour="rate_limited")
    healthy = _Adapter(_cfg("pricey", cost_per_1k_prompt=0.05))
    router = _router(throttled, healthy, timeout_sec=2.0, total_deadline_sec=20.0)

    before = _counter_value(THROTTLED_TOTAL, backend="cheap")
    await router.execute(messages=MSG)

    # Counted per failed attempt, so the signal reflects how hard the limit
    # is being hit rather than merely that it was.
    assert _counter_value(THROTTLED_TOTAL, backend="cheap") > before


@pytest.mark.asyncio
async def test_exhausting_the_budget_increments_the_deadline_counter():
    from gateway.telemetry.metrics import DEADLINE_EXCEEDED_TOTAL

    slow = _Adapter(_cfg("slow"), delay=5.0)
    router = _router(slow, timeout_sec=5.0, total_deadline_sec=1.5)

    before = _counter_value(DEADLINE_EXCEEDED_TOTAL)
    with pytest.raises(DeadlineExceededException):
        await router.execute(messages=MSG)

    assert _counter_value(DEADLINE_EXCEEDED_TOTAL) == before + 1


def test_latency_buckets_cover_real_llm_latencies():
    """Prometheus' defaults top out at 10, assuming seconds. Latency here is
    milliseconds, so every real call previously landed in +Inf and no
    percentile could be computed from the histogram at all."""
    from gateway.telemetry.metrics import LATENCY_BUCKETS_MS

    observed_ms = [394, 587, 4753, 50717]
    for ms in observed_ms:
        assert any(
            ms <= b for b in LATENCY_BUCKETS_MS if b != float("inf")
        ), f"{ms}ms falls outside every finite bucket"
