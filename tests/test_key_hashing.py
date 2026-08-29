"""Client API keys are stored as digests, never in the clear.

A leaked database previously handed an attacker every customer's credential in
usable form. Keys are now hashed on the way in, so the stored value cannot be
presented as a credential — the same reason password hashes exist.
"""

import os
import sqlite3
import tempfile

import pytest

from gateway.auth import VALID_API_KEYS, load_api_keys
from gateway.ledger.store import LedgerStore, hash_api_key, is_hashed

PLAINTEXT = "sk-test-plaintext-key-value"


@pytest.fixture
def db_path():
    return os.path.join(tempfile.mkdtemp(), "keys.db")


@pytest.fixture
def store(db_path):
    return LedgerStore(db_path)


async def _seed(store, key=PLAINTEXT, **over):
    cfg = {
        "api_key": key,
        "daily_limit_usd": 5.0,
        "monthly_limit_usd": 50.0,
        "requests_per_minute": 60,
    }
    cfg.update(over)
    await store.load_budgets_from_config([cfg])


# ── Storage ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_key_is_stored_as_a_digest(store, db_path):
    await _seed(store)

    con = sqlite3.connect(db_path)
    stored = con.execute("SELECT api_key FROM budgets").fetchone()[0]

    assert stored == hash_api_key(PLAINTEXT)
    assert is_hashed(stored)
    assert stored != PLAINTEXT


@pytest.mark.asyncio
async def test_plaintext_never_reaches_the_database_file(store, db_path):
    """The value must be absent from the raw bytes, not merely from a query."""
    await _seed(store)

    blob = b""
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(db_path + suffix):
            blob += open(db_path + suffix, "rb").read()

    assert PLAINTEXT.encode() not in blob


@pytest.mark.asyncio
async def test_display_prefix_is_kept_for_identification(store, db_path):
    """An operator still needs to tell one client from another."""
    await _seed(store)

    con = sqlite3.connect(db_path)
    prefix = con.execute("SELECT key_prefix FROM budgets").fetchone()[0]

    assert prefix and PLAINTEXT.startswith(prefix)
    # Enough to recognise, not enough to use.
    assert len(prefix) < len(PLAINTEXT)


# ── Behaviour is unchanged for callers ───────────────────────────────────


@pytest.mark.asyncio
async def test_the_real_key_still_looks_up_its_budget(store):
    await _seed(store)

    budget = await store.get_budget(PLAINTEXT)

    assert budget is not None
    assert budget["daily_limit_usd"] == 5.0


@pytest.mark.asyncio
async def test_a_different_key_does_not_collide(store):
    await _seed(store)

    assert await store.get_budget("sk-some-other-key") is None


@pytest.mark.asyncio
async def test_spend_and_rate_limits_key_off_the_same_digest(store):
    """Every table keyed on api_key must agree, or spend stops joining to the
    budget that produced it."""
    await _seed(store)
    await store.record_request(PLAINTEXT, "r1", "b", "m", 10, 10, 0.5, 100.0)

    budget = await store.get_budget(PLAINTEXT)
    rows = await store.get_all_requests(limit=10)

    assert budget["spend_today"] == pytest.approx(0.5)
    assert rows[0]["api_key"] == hash_api_key(PLAINTEXT)


@pytest.mark.asyncio
async def test_auth_holds_digests_not_keys(store, monkeypatch):
    await _seed(store)
    load_api_keys(ledger_store=store)

    assert hash_api_key(PLAINTEXT) in VALID_API_KEYS
    assert PLAINTEXT not in VALID_API_KEYS


# ── Upgrading an existing database ───────────────────────────────────────


@pytest.mark.asyncio
async def test_existing_plaintext_database_is_migrated_in_place(db_path):
    """A database written before hashing must upgrade without breaking the
    keys already issued to clients."""
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE budgets (api_key TEXT PRIMARY KEY, daily_limit_usd REAL,"
        " monthly_limit_usd REAL, spend_today REAL DEFAULT 0,"
        " spend_month REAL DEFAULT 0, last_reset_date TEXT,"
        " last_reset_month TEXT, requests_per_minute INTEGER DEFAULT 60)"
    )
    con.execute(
        "CREATE TABLE requests (id TEXT PRIMARY KEY, api_key TEXT, backend TEXT,"
        " model TEXT, prompt_tokens INT, completion_tokens INT, cost_usd REAL,"
        " latency_ms REAL, timestamp DATETIME)"
    )
    con.execute(
        "INSERT INTO budgets (api_key, daily_limit_usd, monthly_limit_usd)"
        " VALUES (?, ?, ?)",
        (PLAINTEXT, 5.0, 50.0),
    )
    con.execute(
        "INSERT INTO requests VALUES ('r1', ?, 'b', 'm', 1, 1, 0.01, 1, '2026-01-01')",
        (PLAINTEXT,),
    )
    con.commit()
    con.close()

    store = LedgerStore(db_path)  # migration runs on open

    # The client's existing key keeps working.
    assert await store.get_budget(PLAINTEXT) is not None

    con = sqlite3.connect(db_path)
    assert con.execute("SELECT api_key FROM budgets").fetchone()[0] == hash_api_key(
        PLAINTEXT
    )
    # Spend history migrates too, or it stops joining to its budget.
    assert con.execute("SELECT api_key FROM requests").fetchone()[0] == hash_api_key(
        PLAINTEXT
    )


@pytest.mark.asyncio
async def test_migration_is_idempotent(db_path):
    """Reopening must not hash the digest a second time."""
    store = LedgerStore(db_path)
    await _seed(store)

    LedgerStore(db_path)
    reopened = LedgerStore(db_path)

    assert await reopened.get_budget(PLAINTEXT) is not None
