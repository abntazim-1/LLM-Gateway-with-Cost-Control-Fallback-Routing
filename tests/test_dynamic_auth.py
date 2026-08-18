import pytest
import time
from fastapi import HTTPException
from fastapi.testclient import TestClient
from gateway.ledger.store import LedgerStore
from gateway.auth import load_api_keys, verify_api_key, VALID_API_KEYS, RATE_LIMIT_RULES
from gateway.main import app

@pytest.mark.asyncio
async def test_dynamic_auth_and_rate_limits():
    ledger = LedgerStore(":memory:")
    
    # Load 2 budget keys with custom rate limits
    budgets = [
        {"api_key": "sk-key-low", "daily_limit_usd": 10.0, "monthly_limit_usd": 100.0, "requests_per_minute": 2},
        {"api_key": "sk-key-high", "daily_limit_usd": 20.0, "monthly_limit_usd": 200.0, "requests_per_minute": 5}
    ]
    await ledger.load_budgets_from_config(budgets)
    
    # Load keys into memory using our dynamic loader
    load_api_keys(ledger_store=ledger)
    
    assert "sk-key-low" in VALID_API_KEYS
    assert "sk-key-high" in VALID_API_KEYS
    assert RATE_LIMIT_RULES["sk-key-low"] == 2
    assert RATE_LIMIT_RULES["sk-key-high"] == 5
    
    # Test low rate key
    # Request 1: Success
    tok = await verify_api_key("sk-key-low")
    assert tok == "sk-key-low"
    
    # Request 2: Success
    tok = await verify_api_key("sk-key-low")
    assert tok == "sk-key-low"
    
    # Request 3: Should fail (limit is 2 requests per minute)
    with pytest.raises(HTTPException) as excinfo:
        await verify_api_key("sk-key-low")
    assert excinfo.value.status_code == 429
    assert "Rate limit exceeded" in excinfo.value.detail
    
    # Test high rate key (should easily allow 3 requests)
    tok = await verify_api_key("sk-key-high")
    assert tok == "sk-key-high"
    tok = await verify_api_key("sk-key-high")
    assert tok == "sk-key-high"
    tok = await verify_api_key("sk-key-high")
    assert tok == "sk-key-high"

def test_admin_patch_budget_limits_and_rate_limits(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "test-secure-admin-key")
    # Use context manager to trigger ASGI lifespan
    with TestClient(app) as client:
        # Patch/Update limits for a new key via Admin API
        payload = {
            "daily_limit_usd": 15.0,
            "monthly_limit_usd": 150.0,
            "requests_per_minute": 12
        }
        resp = client.patch(
            "/admin/budgets/sk-key-admin-test",
            headers={"X-Admin-Token": "test-secure-admin-key"},
            json=payload
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        
        # Verify changes propagated in-place to memory auth collections
        assert "sk-key-admin-test" in VALID_API_KEYS
        assert RATE_LIMIT_RULES["sk-key-admin-test"] == 12

def test_admin_auth_rejection_and_unconfigured(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "test-secure-admin-key")
    with TestClient(app) as client:
        # Request with wrong admin key should be rejected with 401
        resp = client.get("/admin/budgets", headers={"X-Admin-Token": "wrong-key"})
        assert resp.status_code == 401
        assert "Unauthorized Admin access" in resp.json()["detail"]

        # Request with missing header should be rejected with 401
        resp = client.get("/admin/budgets")
        assert resp.status_code == 401
        assert "Unauthorized Admin access" in resp.json()["detail"]

    # When ADMIN_API_KEY is not set in environment, fail-closed with 500
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    with TestClient(app) as client:
        resp = client.get("/admin/budgets", headers={"X-Admin-Token": "test-secure-admin-key"})
        assert resp.status_code == 500
        assert "ADMIN_API_KEY environment variable is not configured" in resp.json()["detail"]

