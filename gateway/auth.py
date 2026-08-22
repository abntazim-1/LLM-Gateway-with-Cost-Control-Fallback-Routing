import os

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from gateway import load_config

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

VALID_API_KEYS: set = set()
# RATE_LIMIT_RULES is kept for backwards-compatibility with existing tests and
# callers that import it; enforcement has moved to the DB-backed store.
RATE_LIMIT_RULES: dict = {}

# Module-level reference to the ledger store, injected by main.py at startup
# via set_ledger(). Before injection, rate limiting falls back to the DB-less
# path (no-op) so the app can still boot without a configured store.
_ledger = None


def set_ledger(ledger_store) -> None:
    """Inject the shared LedgerStore so verify_api_key can delegate rate limiting."""
    global _ledger
    _ledger = ledger_store


def load_api_keys(ledger_store=None) -> None:
    """Load valid API keys and per-key RPM rules from the DB or fallback config."""
    global VALID_API_KEYS, RATE_LIMIT_RULES
    if ledger_store:
        set_ledger(ledger_store)
        try:
            records = ledger_store.get_all_api_keys_and_limits_sync()
            VALID_API_KEYS.clear()
            VALID_API_KEYS.update(r["api_key"] for r in records)

            RATE_LIMIT_RULES.clear()
            RATE_LIMIT_RULES.update(
                {r["api_key"]: r["requests_per_minute"] for r in records}
            )
            return
        except Exception:
            pass  # fallback to config file

    raw_env_budgets = os.environ.get("GATEWAY_BUDGETS_JSON") or os.environ.get(
        "GATEWAY_BUDGETS"
    )
    if raw_env_budgets:
        try:
            import json

            parsed = json.loads(raw_env_budgets)
            budgets = (
                parsed.get("budgets", parsed) if isinstance(parsed, dict) else parsed
            )
            VALID_API_KEYS.clear()
            VALID_API_KEYS.update(b["api_key"] for b in budgets)
            RATE_LIMIT_RULES.clear()
            RATE_LIMIT_RULES.update(
                {b["api_key"]: b.get("requests_per_minute", 60) for b in budgets}
            )
            return
        except Exception:
            pass

    config_path = os.environ.get(
        "BUDGETS_CONFIG_PATH",
        os.path.join(os.path.dirname(__file__), "..", "configs", "budgets.yaml"),
    )
    try:
        budgets = load_config(config_path).get("budgets", [])
        VALID_API_KEYS.clear()
        VALID_API_KEYS.update(b["api_key"] for b in budgets)

        RATE_LIMIT_RULES.clear()
        RATE_LIMIT_RULES.update(
            {b["api_key"]: b.get("requests_per_minute", 60) for b in budgets}
        )
    except Exception:
        VALID_API_KEYS.clear()
        RATE_LIMIT_RULES.clear()


async def verify_api_key(api_key_header: str = Security(api_key_header)) -> str:
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key",
        )

    # Strip optional Bearer prefix
    token = (
        api_key_header.removeprefix("Bearer ").strip()
        if api_key_header.startswith("Bearer ")
        else api_key_header
    )

    if token not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )

    # ── Rate limiting ────────────────────────────────────────────────────────
    # Delegate to the DB-backed store so limits are:
    #   • Atomic  – no TOCTOU between count and insert
    #   • Durable – survive process restarts
    #   • Shared  – consistent across all threads/workers on the same DB file
    if _ledger is not None:
        await _ledger.check_rate_limit(token)
    # If the ledger hasn't been injected yet (e.g. during early startup tests)
    # we allow the request through rather than hard-failing.

    return token
