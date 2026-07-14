from fastapi import FastAPI, Depends, Request, HTTPException
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import os

from gateway.auth import verify_api_key
from gateway import load_config
from gateway.ledger.store import LedgerStore
from gateway.policy.budget import BudgetPolicy, BudgetExceededException
from gateway.policy.circuit_breaker import CircuitBreakerRegistry
from gateway.policy.router import Router, NoAvailableBackendException

from gateway.adapters.openai_adapter import OpenAIAdapter
from gateway.adapters.anthropic_adapter import AnthropicAdapter
from gateway.adapters.local_vllm_adapter import LocalVLLMAdapter

import logging
logging.basicConfig(level=logging.INFO)

# Globals
ledger = None
budget_policy = None
circuit_registry = None
router = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ledger, budget_policy, circuit_registry, router
    
    # Init ledger
    ledger = LedgerStore("ledger.db")
    budgets_config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "budgets.yaml")
    budgets = load_config(budgets_config_path).get("budgets", [])
    ledger.load_budgets_from_config(budgets)
    
    budget_policy = BudgetPolicy(ledger)
    
    # Init circuit breaker
    cb_config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "circuit_breaker.yaml")
    cb_config = load_config(cb_config_path).get("circuit_breaker", {})
    circuit_registry = CircuitBreakerRegistry(
        failure_threshold=cb_config.get("failure_threshold", 3),
        cooldown_sec=cb_config.get("cooldown_period_sec", 30)
    )
    
    # Init adapters
    backends_config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "backends.yaml")
    backends = load_config(backends_config_path).get("backends", [])
    adapters = []
    for cfg in backends:
        if cfg["provider"] == "openai":
            adapters.append(OpenAIAdapter(cfg))
        elif cfg["provider"] == "anthropic":
            adapters.append(AnthropicAdapter(cfg))
        elif cfg["provider"] == "local_vllm":
            adapters.append(LocalVLLMAdapter(cfg))
            
    # Init router
    routing_config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "routing_policy.yaml")
    strategy = load_config(routing_config_path).get("routing", {}).get("strategy", "cost_first")
    router = Router(adapters, circuit_registry, strategy=strategy)
    
    yield

app = FastAPI(
    title="Enterprise LLM Gateway",
    description="Cost-controlling, routing LLM Gateway",
    version="0.1.0",
    lifespan=lifespan
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
        
    # Estimated cost could be pre-calculated based on token estimate
    try:
        budget_policy.check_preflight(api_key, estimated_cost=0.01)
    except BudgetExceededException as e:
        raise HTTPException(status_code=429, detail=str(e))
        
    try:
        response = await router.execute(messages, **body)
    except NoAvailableBackendException as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gateway internal error: {str(e)}")
        
    # Record spend
    ledger.record_request(
        api_key=api_key,
        req_id=response.id,
        backend=response.backend_id,
        model=response.model,
        prompt_tokens=response.prompt_tokens,
        comp_tokens=response.completion_tokens,
        cost=response.cost_usd,
        latency=response.latency_ms
    )
    
    return {
        "id": response.id,
        "object": "chat.completion",
        "created": int(response.latency_ms), # mockup timestamp
        "model": response.model,
        "choices": [
            {
                "index": i,
                "message": {
                    "role": msg.role,
                    "content": msg.content
                },
                "finish_reason": "stop"
            }
            for i, msg in enumerate(response.messages)
        ],
        "usage": {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.prompt_tokens + response.completion_tokens
        }
    }
