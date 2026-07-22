from fastapi import FastAPI, Depends, Request, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import Response, StreamingResponse
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
from gateway.ledger.async_queue import AsyncLedgerQueue
from gateway.policy.budget import BudgetPolicy, BudgetExceededException
from gateway.policy.circuit_breaker import CircuitBreakerRegistry
from gateway.policy.router import Router, NoAvailableBackendException
from gateway.policy.guardrails import GuardrailsPipeline, GuardrailViolationException
from gateway.telemetry.metrics import observe_request, observe_cache
from gateway.telemetry.tracer import GatewayTracer
from gateway.config_manager import ConfigManager

from gateway.adapters.openai_adapter import OpenAIAdapter
from gateway.adapters.anthropic_adapter import AnthropicAdapter
from gateway.adapters.local_vllm_adapter import LocalVLLMAdapter
from gateway.policy.pii import PiiSanitizer
from gateway.policy.cache import PromptCache

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
ledger_queue = None
budget_policy = None
circuit_registry = None
router = None
prompt_cache = None
pii_sanitizer = None
guardrails_pipeline = None

async def health_check_loop():
    while True:
        try:
            if router and circuit_registry:
                for adapter in router.adapters:
                    is_healthy = await adapter.health_check()
                    breaker = circuit_registry.get_breaker(adapter.id)
                    if not is_healthy:
                        await breaker.record_failure()
        except Exception as e:
            logging.error(f"Health check loop error: {e}")
        await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ledger, ledger_queue, budget_policy, circuit_registry, router, prompt_cache, pii_sanitizer, guardrails_pipeline
    
    # Init ledger & async queue
    ledger = LedgerStore("ledger.db")
    ledger_queue = AsyncLedgerQueue(ledger)
    ledger_queue.start()

    budgets_config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "budgets.yaml")
    budgets = load_config(budgets_config_path).get("budgets", [])
    await ledger.load_budgets_from_config(budgets)
    load_api_keys(ledger_store=ledger)
    
    prompt_cache = PromptCache(ttl_seconds=300)
    pii_sanitizer = PiiSanitizer()
    guardrails_pipeline = GuardrailsPipeline()
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
    await ledger_queue.stop()

app = FastAPI(
    title="Enterprise LLM Gateway",
    description="Cost-controlling, routing LLM Gateway",
    version="0.1.0",
    lifespan=lifespan
)

app.mount("/metrics", make_asgi_app())

admin_key_header = APIKeyHeader(name="X-Admin-Token", auto_error=False)

def verify_admin_key(admin_key_header: str = Security(admin_key_header)):
    expected_token = os.environ.get("ADMIN_API_KEY", "admin-default-secret")
    if not admin_key_header or admin_key_header != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized Admin access"
        )

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/admin/budgets")
async def get_admin_budgets(admin_key: str = Depends(verify_admin_key)):
    try:
        return await ledger.get_all_budgets()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/requests")
async def get_admin_requests(limit: int = 50, admin_key: str = Depends(verify_admin_key)):
    try:
        return await ledger.get_all_requests(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/circuit-breakers")
async def get_admin_circuit_breakers(admin_key: str = Depends(verify_admin_key)):
    try:
        return await ledger.get_all_circuit_breakers()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/admin/budgets/{api_key}")
async def update_budget(api_key: str, body: Dict[str, Any], admin_key: str = Depends(verify_admin_key)):
    daily = body.get("daily_limit_usd")
    monthly = body.get("monthly_limit_usd")
    requests_per_minute = body.get("requests_per_minute")
    if daily is None or monthly is None:
        raise HTTPException(status_code=400, detail="daily_limit_usd and monthly_limit_usd are required")
    try:
        is_update = await ledger.update_budget_limits(api_key, daily, monthly, requests_per_minute)
        load_api_keys(ledger_store=ledger)
        return {"status": "success", "updated": is_update}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/circuit-breakers/{backend_id}/reset")
async def reset_circuit_breaker(backend_id: str, admin_key: str = Depends(verify_admin_key)):
    try:
        breaker = circuit_registry.get_breaker(backend_id)
        await breaker.record_success()
        return {"status": "success", "message": f"Circuit breaker for {backend_id} has been reset to CLOSED"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/reload-config")
async def reload_config_endpoint(admin_key: str = Depends(verify_admin_key)):
    global router
    try:
        config_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
        cm = ConfigManager(config_dir)
        new_adapters = cm.load_backends()
        new_strategy = cm.load_routing_strategy()
        new_budgets = cm.load_budgets()
        
        if new_adapters:
            router.adapters = new_adapters
            router.strategy = new_strategy
        if new_budgets:
            await ledger.load_budgets_from_config(new_budgets)
            load_api_keys(ledger_store=ledger)
            
        return {"status": "success", "message": f"Reloaded configuration: {len(new_adapters)} adapters, strategy '{new_strategy}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload config: {str(e)}")

@app.get("/health/backends")
async def health_backends():
    try:
        results = {}
        for adapter in router.adapters:
            is_healthy = await adapter.health_check()
            breaker = circuit_registry.get_breaker(adapter.id)
            cb_state = await breaker.get_state()
            results[adapter.id] = {
                "provider": adapter.config["provider"],
                "model": adapter.model,
                "healthy": is_healthy,
                "circuit_breaker_state": cb_state.value
            }
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, response: Response, api_key: str = Depends(verify_api_key)):
    request_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id
    
    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="Messages list is required")

    # Guardrails Pipeline Validation
    if guardrails_pipeline:
        try:
            guardrails_pipeline.validate_messages(messages)
        except GuardrailViolationException as e:
            raise HTTPException(status_code=400, detail=str(e))
        
    # PII Sanitization
    sanitized_messages = pii_sanitizer.sanitize_messages(messages)
    
    is_stream = body.get("stream", False)
    kwargs = {k: v for k, v in body.items() if k not in ["messages", "model"]}
    
    # Cache Check
    cached_response = prompt_cache.get(sanitized_messages, kwargs)
    if cached_response is not None:
        observe_cache(hit=True)
        # Record 0 cost request in the ledger
        store = ledger_queue if ledger_queue else ledger
        await store.record_request(
            api_key=api_key,
            req_id=cached_response.get("id", f"cache-{uuid.uuid4()}"),
            backend="cache",
            model=cached_response.get("model", "cached-model"),
            prompt_tokens=cached_response.get("usage", {}).get("prompt_tokens", 0),
            comp_tokens=cached_response.get("usage", {}).get("completion_tokens", 0),
            cost=0.0,
            latency=0.0
        )
        if is_stream:
            async def cached_stream_generator():
                content = cached_response["choices"][0]["message"]["content"]
                words = content.split(" ")
                for i, word in enumerate(words):
                    space = " " if i > 0 else ""
                    chunk = {
                        "id": cached_response["id"],
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": cached_response["model"],
                        "choices": [{
                            "index": 0,
                            "delta": {"content": space + word},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.01)
                final_chunk = {
                    "id": cached_response["id"],
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": cached_response["model"],
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(cached_stream_generator(), media_type="text/event-stream")
        return cached_response

    observe_cache(hit=False)

    # Dynamic preflight cost estimation
    try:
        try:
            ranked_adapters = await router.get_ranked_adapters(messages=sanitized_messages, **kwargs)
            target_adapter = ranked_adapters[0] if ranked_adapters else None
        except NoAvailableBackendException:
            target_adapter = None

        if target_adapter:
            try:
                import tiktoken
                encoding = tiktoken.get_encoding("cl100k_base")
                prompt_tokens = sum(len(encoding.encode(m.get("content", ""))) for m in sanitized_messages)
            except ImportError:
                approx_tokens = sum(len(m.get("content", "")) for m in sanitized_messages) / 4.0
                prompt_tokens = max(1, int(approx_tokens))
                
            # Estimate completion tokens using request max_tokens or default to 256
            completion_tokens = body.get("max_tokens", 256)
            
            prompt_cost = (prompt_tokens / 1000.0) * target_adapter.cost_per_1k_prompt
            completion_cost = (completion_tokens / 1000.0) * target_adapter.cost_per_1k_completion
            estimated_cost = prompt_cost + completion_cost
        else:
            estimated_cost = 0.0
            
        await budget_policy.check_preflight(api_key, estimated_cost=estimated_cost)
    except BudgetExceededException as e:
        raise HTTPException(status_code=429, detail=str(e))
        
    if is_stream:
        async def stream_generator():
            accumulated_text = ""
            last_chunk = None
            try:
                kwargs["api_key"] = api_key
                async for chunk in router.execute_stream(sanitized_messages, **kwargs):
                    last_chunk = chunk
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            accumulated_text += content
                    yield f"data: {json.dumps(chunk)}\n\n"
                
                # Cache the finished streaming response
                if last_chunk:
                    cached_response_dict = {
                        "id": last_chunk.get("id", f"cache-{uuid.uuid4()}"),
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": last_chunk.get("model", "model-stream"),
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": accumulated_text
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0
                        }
                    }
                    prompt_cache.set(sanitized_messages, kwargs, cached_response_dict)
                yield "data: [DONE]\n\n"
            except Exception as e:
                logging.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'gateway_error'}})}\n\n"
                yield "data: [DONE]\n\n"
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
        
    try:
        backend_response = await router.execute(sanitized_messages, **kwargs)
        observe_request(backend_response.backend_id, "success", backend_response.latency_ms, backend_response.cost_usd)
    except NoAvailableBackendException as e:
        observe_request("unknown", "error", 0.0, 0.0)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        observe_request("unknown", "error", 0.0, 0.0)
        raise HTTPException(status_code=500, detail=f"Gateway internal error: {str(e)}")
        
    # Record spend
    store = ledger_queue if ledger_queue else ledger
    await store.record_request(
        api_key=api_key,
        req_id=request_id,
        backend=backend_response.backend_id,
        model=backend_response.model,
        prompt_tokens=backend_response.prompt_tokens,
        comp_tokens=backend_response.completion_tokens,
        cost=backend_response.cost_usd,
        latency=backend_response.latency_ms
    )
    
    response_dict = {
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
    
    # Save to cache
    prompt_cache.set(sanitized_messages, kwargs, response_dict)
    
    return response_dict
