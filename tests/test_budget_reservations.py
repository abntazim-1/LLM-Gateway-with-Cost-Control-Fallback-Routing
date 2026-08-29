"""Budget holds must survive failover and process death.

Two ways a reserve-then-reconcile scheme leaks:

* the hold is priced on the backend the request is *tried* on first, while
  failover may land it on a pricier one — so the limit is checked against a
  price that is never paid
* the hold is released by the request that took it, so a process killed
  outright leaves budget held against a client forever
"""

import time
import uuid
from typing import Any, AsyncGenerator, Dict, List

import pytest
from fastapi.testclient import TestClient

import gateway.main as main
from gateway.adapters.base import BaseAdapter, NormalizedResponse
from gateway.ledger.store import LedgerStore
from gateway.policy.budget import BudgetPolicy

MAX_TOKENS = 1000

# A spread wide enough to make the gap unmistakable. Real spreads between a
# small model and a frontier one are this large or larger.
CHEAP = {"cost_per_1k_prompt": 0.001, "cost_per_1k_completion": 0.002}
PRICEY = {"cost_per_1k_prompt": 0.05, "cost_per_1k_completion": 0.10}

# Sits between the two reservations: ~$0.002 on the cheap backend, ~$0.10 on
# the pricey one. Which side of the limit the request falls on is therefore
# decided entirely by which backend the hold is priced against.
LIMIT_BETWEEN = 0.05


class _Adapter(BaseAdapter):
    async def complete(self, messages, **kwargs) -> NormalizedResponse:
        return NormalizedResponse(
            id=self.id,
            backend_id=self.id,
            model=self.model,
            messages=[{"role": "assistant", "content": "ok"}],
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.001,
            latency_ms=1.0,
        )

    async def complete_stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        yield {}

    async def health_check(self) -> bool:
        return True


def _adapter(aid, prices):
    return _Adapter({"id": aid, "provider": "ollama", "model": aid, **prices})


@pytest.fixture
def client_and_key(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")
    api_key = f"sk-reserve-{uuid.uuid4().hex[:8]}"

    with TestClient(main.app) as client:
        client.patch(
            f"/admin/budgets/{api_key}",
            headers={"X-Admin-Token": "test-admin-key"},
            json={
                "daily_limit_usd": LIMIT_BETWEEN,
                "monthly_limit_usd": 1000.0,
                "requests_per_minute": 1000,
            },
        )
        state = main.get_state()
        # cost_first ranks the cheap one first, so it is the backend the old
        # code priced the hold against.
        state.router.adapters = [
            _adapter("cheap", CHEAP),
            _adapter("pricey", PRICEY),
        ]
        state.prompt_cache.clear()
        yield client, api_key


def _ask(client, api_key):
    return client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": f"hi {uuid.uuid4().hex}"}],
            "max_tokens": MAX_TOKENS,
        },
    )


# ── The hold must cover the worst price the request could pay ────────────


def test_hold_is_sized_for_the_priciest_reachable_backend(client_and_key):
    """Failover to a pricier backend must not be able to breach the limit.

    Pricing the hold on the first choice let this request through on a ~$0.002
    estimate; had it then failed over, it would have paid ~$0.10 against a
    $0.05 limit that had already said yes.
    """
    client, api_key = client_and_key

    response = _ask(client, api_key)

    assert response.status_code == 429, (
        "admitted on the cheap backend's price while a pricier one was "
        f"reachable (got {response.status_code})"
    )


def test_a_request_that_fits_the_worst_case_is_still_served(client_and_key):
    """The contrast case — reserving high must not reject everything."""
    client, api_key = client_and_key
    client.patch(
        f"/admin/budgets/{api_key}",
        headers={"X-Admin-Token": "test-admin-key"},
        json={"daily_limit_usd": 1000.0, "monthly_limit_usd": 1000.0},
    )

    assert _ask(client, api_key).status_code == 200


# ── A hold must not outlive the process that took it ─────────────────────

KEY = "sk-reclaim-test"


async def _ledger_with_budget(limit=10.0):
    ledger = LedgerStore(":memory:")
    await ledger.load_budgets_from_config(
        [{"api_key": KEY, "daily_limit_usd": limit, "monthly_limit_usd": limit}]
    )
    return ledger


def _age(ledger, request_id, seconds):
    """Backdate a hold so it looks like it was taken `seconds` ago."""
    conn = ledger._get_connection()
    conn.execute(
        "UPDATE reservations SET reserved_at = ? WHERE request_id = ?",
        (time.time() - seconds, request_id),
    )
    conn.commit()


@pytest.mark.asyncio
async def test_a_hold_left_by_a_dead_process_is_given_back():
    """No `finally` runs when a process is killed, so nothing releases it."""
    ledger = await _ledger_with_budget()
    policy = BudgetPolicy(ledger)

    await policy.check_and_reserve(KEY, estimated_cost=3.0, request_id="orphan")
    assert (await ledger.get_budget(KEY))["spend_today"] == pytest.approx(3.0)

    _age(ledger, "orphan", seconds=600)
    freed = await ledger.reclaim_stale_reservations(older_than_sec=300)

    assert freed["count"] == 1
    assert (await ledger.get_budget(KEY))["spend_today"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_a_request_still_in_flight_keeps_its_hold():
    """Reclaiming a live request's budget would let it overspend."""
    ledger = await _ledger_with_budget()
    policy = BudgetPolicy(ledger)

    await policy.check_and_reserve(KEY, estimated_cost=3.0, request_id="live")
    freed = await ledger.reclaim_stale_reservations(older_than_sec=300)

    assert freed["count"] == 0
    assert (await ledger.get_budget(KEY))["spend_today"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_a_reconciled_hold_is_never_reclaimed():
    """Once real usage is recorded the hold is closed, so a later sweep must
    not refund spend that actually happened."""
    ledger = await _ledger_with_budget()
    policy = BudgetPolicy(ledger)

    await policy.check_and_reserve(KEY, estimated_cost=3.0, request_id="req-1")
    await ledger.record_request(
        KEY, "req-1", "b", "m", 10, 10, 0.5, 100.0, reserved_cost=3.0
    )
    assert (await ledger.get_budget(KEY))["spend_today"] == pytest.approx(0.5)

    freed = await ledger.reclaim_stale_reservations(older_than_sec=0)

    assert freed["count"] == 0
    assert (await ledger.get_budget(KEY))["spend_today"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_a_hold_from_a_past_period_does_not_refund_twice():
    """The daily reset already zeroed that counter. Subtracting again would
    hand back budget that this period's real requests are using."""
    ledger = await _ledger_with_budget()
    policy = BudgetPolicy(ledger)

    await policy.check_and_reserve(KEY, estimated_cost=3.0, request_id="yesterday")
    conn = ledger._get_connection()
    # Simulate the period rolling over: the hold's stamp is now stale and the
    # counter it contributed to has been reset.
    conn.execute("UPDATE reservations SET reset_date = '1999-01-01'")
    conn.execute("UPDATE budgets SET spend_today = 2.0")
    conn.commit()

    _age(ledger, "yesterday", seconds=600)
    await ledger.reclaim_stale_reservations(older_than_sec=300)

    budget = await ledger.get_budget(KEY)
    assert budget["spend_today"] == pytest.approx(2.0), "refunded a reset period"
