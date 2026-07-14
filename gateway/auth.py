from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from gateway import load_config
import os
import time

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

VALID_API_KEYS = set()
RATE_LIMITS = {}
MAX_REQUESTS_PER_MINUTE = 60

def load_api_keys():
    """Load valid API keys into memory once at startup."""
    global VALID_API_KEYS
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "budgets.yaml")
    try:
        budgets = load_config(config_path).get("budgets", [])
        VALID_API_KEYS = {b["api_key"] for b in budgets}
    except Exception:
        VALID_API_KEYS = set()

async def verify_api_key(api_key_header: str = Security(api_key_header)) -> str:
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key",
        )
    
    # Extract token if Bearer
    token = api_key_header.replace("Bearer ", "") if api_key_header.startswith("Bearer ") else api_key_header
    
    if token not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
        
    # Rate Limiting
    now = time.time()
    if token not in RATE_LIMITS:
        RATE_LIMITS[token] = []
        
    # Filter out requests older than 60 seconds
    RATE_LIMITS[token] = [ts for ts in RATE_LIMITS[token] if now - ts < 60]
    
    if len(RATE_LIMITS[token]) >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {MAX_REQUESTS_PER_MINUTE} requests per minute."
        )
        
    RATE_LIMITS[token].append(now)
    
    return token
