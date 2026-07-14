from fastapi import FastAPI, Depends, Request, HTTPException
from typing import Dict, Any, List
from gateway.auth import verify_api_key

app = FastAPI(
    title="Enterprise LLM Gateway",
    description="Cost-controlling, routing LLM Gateway",
    version="0.1.0"
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, api_key: str = Depends(verify_api_key)):
    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="Messages list is required")
        
    # TODO: Implement Router & Policy Engine (Day 2)
    # For now, just return a mock response to test the endpoint
    
    return {
        "id": "mock-id-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": body.get("model", "unknown"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock response from the LLM Gateway. Routing engine pending."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }
