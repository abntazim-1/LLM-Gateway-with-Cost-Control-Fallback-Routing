from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from gateway import load_config
import os

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def get_valid_api_keys() -> set:
    # Read from budgets config to get known keys
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "budgets.yaml")
    try:
        budgets = load_config(config_path).get("budgets", [])
        return {b["api_key"] for b in budgets}
    except Exception:
        return set()

async def verify_api_key(api_key_header: str = Security(api_key_header)) -> str:
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key",
        )
    
    # Extract token if Bearer
    token = api_key_header.replace("Bearer ", "") if api_key_header.startswith("Bearer ") else api_key_header
    
    valid_keys = get_valid_api_keys()
    if token not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    
    return token
