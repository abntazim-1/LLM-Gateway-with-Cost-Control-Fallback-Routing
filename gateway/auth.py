from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from gateway import load_config
import os
import time

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

VALID_API_KEYS = set()
RATE_LIMIT_RULES = {}
RATE_LIMITS = {}
MAX_REQUESTS_PER_MINUTE = 60

def load_api_keys(ledger_store=None):
    """Load valid API keys and custom rate limits from the DB or fallback config in-place."""
    global VALID_API_KEYS, RATE_LIMIT_RULES
    if ledger_store:
        try:
            records = ledger_store.get_all_api_keys_and_limits_sync()
            VALID_API_KEYS.clear()
            VALID_API_KEYS.update(r["api_key"] for r in records)
            
            RATE_LIMIT_RULES.clear()
            RATE_LIMIT_RULES.update({r["api_key"]: r["requests_per_minute"] for r in records})
            return
        except Exception:
            pass # fallback to config file
            
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "budgets.yaml")
    try:
        budgets = load_config(config_path).get("budgets", [])
        VALID_API_KEYS.clear()
        VALID_API_KEYS.update(b["api_key"] for b in budgets)
        
        RATE_LIMIT_RULES.clear()
        RATE_LIMIT_RULES.update({b["api_key"]: b.get("requests_per_minute", 60) for b in budgets})
    except Exception:
        VALID_API_KEYS.clear()
        RATE_LIMIT_RULES.clear()

async def verify_api_key(api_key_header: str = Security(api_key_header)) -> str:
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key",
        )
    
    # Extract token if Bearer prefix present
    token = api_key_header.replace("Bearer ", "") if api_key_header.startswith("Bearer ") else api_key_header
    
    if token not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
        
    # Rate Limiting check
    now = time.time()
    if token not in RATE_LIMITS:
        RATE_LIMITS[token] = []
        
    # Filter out requests older than 60 seconds
    RATE_LIMITS[token] = [ts for ts in RATE_LIMITS[token] if now - ts < 60]
    
    # Retrieve granular per-key rate limit
    limit = RATE_LIMIT_RULES.get(token, MAX_REQUESTS_PER_MINUTE)
    
    if len(RATE_LIMITS[token]) >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {limit} requests per minute."
        )
        
    RATE_LIMITS[token].append(now)
    
    return token
