from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import os
import uuid
import asyncio
import logging
from prometheus_client import make_asgi_app

from gateway.auth import verify_api_key, load_api_keys
from gateway import load_config
import time
from gateway.ledger.store import LedgerStore
from gateway.policy.budget import BudgetPolicy, BudgetExceededException
from gateway.policy.circuit_breaker import CircuitBreakerRegistry
from gateway.policy.router import Router, NoAvailableBackendException
from gateway.telemetry.metrics import observe_request

from gateway.adapters.openai_adapter import OpenAIAdapter
from gateway.adapters.anthropic_adapter import AnthropicAdapter
from gateway.adapters.local_vllm_adapter import LocalVLLMAdapter

import json
import logging

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name
        }
        return json.dumps(log_record)

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)
# Remove default handlers if any
for h in logger.handlers[:-1]:
    logger.removeHandler(h)

# Globals
ledger = None
budget_policy = None
circuit_registry = None
router = None

async def health_check_loop():
    while True:
        try:
            if router and circuit_registry:
                for adapter in router.adapters:
                    is_healthy = await adapter.health_check()
                    breaker = circuit_registry.get_breaker(adapter.id)
                    if not is_healthy:
                        breaker.record_failure()
        except Exception as e:
            logging.error(f"Health check loop error: {e}")
        await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ledger, budget_policy, circuit_registry, router
    
    # Init ledger
    ledger = LedgerStore("ledger.db")
    budgets_config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "budgets.yaml")
    budgets = load_config(budgets_config_path).get("budgets", [])
    ledger.load_budgets_from_config(budgets)
    load_api_keys()
    
    budget_policy = BudgetPolicy(ledger)
    
    # Init circuit breaker
    cb_config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "circuit_breaker.yaml")
    cb_config = load_config(cb_config_path).get("circuit_breaker", {})
    circuit_registry = CircuitBreakerRegistry(
        ledger=ledger,
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
    
    bg_task = asyncio.create_task(health_check_loop())
    
    yield
    
    bg_task.cancel()

app = FastAPI(
    title="Enterprise LLM Gateway",
    description="Cost-controlling, routing LLM Gateway",
    version="0.1.0",
    lifespan=lifespan
)

app.mount("/metrics", make_asgi_app())

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, response: Response, api_key: str = Depends(verify_api_key)):
    request_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id
    
    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="Messages list is required")
        
    # Heuristic cost estimation
    try:
        approx_tokens = sum(len(m.get("content", "")) for m in messages) / 4
        estimated_cost = (approx_tokens / 1000.0) * 0.001
        budget_policy.check_preflight(api_key, estimated_cost=estimated_cost)
    except BudgetExceededException as e:
        raise HTTPException(status_code=429, detail=str(e))
        
    try:
        kwargs = {k: v for k, v in body.items() if k not in ["messages", "model"]}
        backend_response = await router.execute(messages, **kwargs)
        observe_request(backend_response.backend_id, "success", backend_response.latency_ms, backend_response.cost_usd)
    except NoAvailableBackendException as e:
        observe_request("unknown", "error", 0.0, 0.0)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        observe_request("unknown", "error", 0.0, 0.0)
        raise HTTPException(status_code=500, detail=f"Gateway internal error: {str(e)}")
        
    # Record spend
    ledger.record_request(
        api_key=api_key,
        req_id=request_id,
        backend=backend_response.backend_id,
        model=backend_response.model,
        prompt_tokens=backend_response.prompt_tokens,
        comp_tokens=backend_response.completion_tokens,
        cost=backend_response.cost_usd,
        latency=backend_response.latency_ms
    )
    
    return {
        "id": backend_response.id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": backend_response.model,
        "choices": [
            {
                "index": i,
                "message": {
                    "role": msg.role,
                    "content": msg.content
                },
                "finish_reason": "stop"
            }
            for i, msg in enumerate(backend_response.messages)
        ],
        "usage": {
            "prompt_tokens": backend_response.prompt_tokens,
            "completion_tokens": backend_response.completion_tokens,
            "total_tokens": backend_response.prompt_tokens + backend_response.completion_tokens
        }
    }
