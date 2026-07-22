import pytest
import asyncio
from gateway.ledger.store import LedgerStore
from gateway.ledger.async_queue import AsyncLedgerQueue
from gateway.ledger.base_store import BaseLedgerStore

@pytest.mark.asyncio
async def test_async_ledger_queue_buffering():
    store = LedgerStore(":memory:")
    # Seed budget for key
    await store.load_budgets_from_config([
        {"api_key": "sk-async-test", "daily_limit_usd": 10.0, "monthly_limit_usd": 100.0}
    ])

    queue = AsyncLedgerQueue(store=store)
    queue.start()

    await queue.record_request(
        api_key="sk-async-test",
        req_id="async-req-1",
        backend="openai",
        model="gpt-4o-mini",
        prompt_tokens=10,
        comp_tokens=20,
        cost=0.001,
        latency=120.0
    )

    # Allow worker to process queue
    await asyncio.sleep(0.2)
    await queue.stop()

    requests = await store.get_all_requests(limit=10)
    assert len(requests) == 1
    assert requests[0]["id"] == "async-req-1"
    assert requests[0]["api_key"] == "sk-async-test"

def test_ledger_store_isinstance_of_base():
    store = LedgerStore(":memory:")
    assert isinstance(store, BaseLedgerStore)
