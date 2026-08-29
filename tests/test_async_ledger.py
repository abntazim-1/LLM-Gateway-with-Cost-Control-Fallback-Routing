import asyncio

import pytest

from gateway.ledger.async_queue import AsyncLedgerQueue
from gateway.ledger.base_store import BaseLedgerStore
from gateway.ledger.store import LedgerStore, hash_api_key


@pytest.mark.asyncio
async def test_async_ledger_queue_buffering():
    store = LedgerStore(":memory:")
    # Seed budget for key
    await store.load_budgets_from_config(
        [
            {
                "api_key": "sk-async-test",
                "daily_limit_usd": 10.0,
                "monthly_limit_usd": 100.0,
            }
        ]
    )

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
        latency=120.0,
    )

    # Allow worker to process queue
    await asyncio.sleep(0.2)
    await queue.stop()

    requests = await store.get_all_requests(limit=10)
    assert len(requests) == 1
    assert requests[0]["id"] == "async-req-1"
    # Stored as a digest — the plaintext key is never persisted.
    assert requests[0]["api_key"] == hash_api_key("sk-async-test")


def test_ledger_store_isinstance_of_base():
    store = LedgerStore(":memory:")
    assert isinstance(store, BaseLedgerStore)
    store.close()


@pytest.mark.asyncio
async def test_concurrent_reads_and_writes(tmp_path):
    db_file = str(tmp_path / "concurrent_test.db")
    store = LedgerStore(db_file)

    await store.load_budgets_from_config(
        [
            {
                "api_key": "sk-concurrent-key",
                "daily_limit_usd": 100.0,
                "monthly_limit_usd": 1000.0,
                "requests_per_minute": 500,
            }
        ]
    )

    async def write_op(i):
        await store.record_request(
            api_key="sk-concurrent-key",
            req_id=f"req-conc-{i}",
            backend="openai",
            model="gpt-4o",
            prompt_tokens=10,
            comp_tokens=10,
            cost=0.01,
            latency=50.0,
        )

    async def read_op(i):
        b = await store.get_budget("sk-concurrent-key")
        assert b is not None
        reqs = await store.get_all_requests(limit=10)
        assert isinstance(reqs, list)

    # Spawn 30 concurrent readers and 20 concurrent writers simultaneously
    tasks = [write_op(i) for i in range(20)] + [read_op(i) for i in range(30)]
    import random

    random.shuffle(tasks)
    await asyncio.gather(*tasks)

    # Verify all 20 writes persisted cleanly
    all_reqs = await store.get_all_requests(limit=100)
    assert len(all_reqs) == 20

    budget = await store.get_budget("sk-concurrent-key")
    assert round(budget["spend_today"], 2) == 0.20

    store.close()
