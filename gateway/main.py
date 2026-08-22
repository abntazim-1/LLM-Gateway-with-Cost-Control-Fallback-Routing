import asyncio
import logging
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import Response, StreamingResponse
from fastapi.security import APIKeyHeader
from prometheus_client import make_asgi_app

# Load .env from the project root (works regardless of CWD)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"), override=False)

import hashlib
import json
import logging
import time

from gateway import load_config
from gateway.adapters.anthropic_adapter import AnthropicAdapter
from gateway.adapters.base import BaseAdapter
from gateway.adapters.local_vllm_adapter import LocalVLLMAdapter
from gateway.adapters.openai_adapter import OpenAIAdapter
from gateway.auth import load_api_keys, set_ledger, verify_api_key
from gateway.config_manager import ConfigManager
from gateway.ledger.async_queue import AsyncLedgerQueue
from gateway.ledger.store import LedgerStore
from gateway.policy.budget import BudgetExceededException, BudgetPolicy
from gateway.policy.cache import PromptCache
from gateway.policy.circuit_breaker import CircuitBreakerRegistry
from gateway.policy.guardrails import GuardrailsPipeline, GuardrailViolationException
from gateway.policy.pii import PiiSanitizer, PiiVault
from gateway.policy.router import (
    ContextLengthExceededException,
    NoAvailableBackendException,
    Router,
)
from gateway.telemetry.metrics import observe_cache, observe_request
from gateway.telemetry.tracer import GatewayTracer


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
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


@dataclass
class GatewayState:
    """Services constructed once at startup by `lifespan`. Holding them here
    (rather than as bare `Optional` module globals) means every field is
    non-Optional once the state exists, so `get_state()` is the single place
    that guards against the pre-startup/shutdown window — callers never need
    their own `if router is None` checks, and mypy can verify the rest."""

    ledger: LedgerStore
    ledger_queue: AsyncLedgerQueue
    budget_policy: BudgetPolicy
    circuit_registry: CircuitBreakerRegistry
    router: Router
    prompt_cache: PromptCache
    pii_sanitizer: PiiSanitizer
    pii_vault: PiiVault
    guardrails_pipeline: GuardrailsPipeline


_state: Optional[GatewayState] = None


def get_state() -> GatewayState:
    if _state is None:
        raise HTTPException(
            status_code=503, detail="Gateway is starting up, please retry shortly"
        )
    return _state


def _key_hash(api_key: str) -> str:
    """Stable non-reversible identifier for telemetry (never the raw key)."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def _find_adapter(router: Router, backend_id: str) -> Optional[BaseAdapter]:
    """Look up the adapter that served a request, if it's still registered."""
    return next((a for a in router.adapters if a.id == backend_id), None)


def _count_tokens(
    messages: List[Dict[str, Any]],
    completion_text: str = "",
    adapter: Optional[BaseAdapter] = None,
) -> tuple:
    """Count tokens using the serving backend's own tokenizer.

    Tokenizers differ per model — the same text can differ by 2x or more
    between backends — so counts are delegated to the adapter whenever one is
    known. Without an adapter (backend never selected, e.g. a stream that died
    before any candidate was chosen) there is no correct tokenizer to use, so
    this returns a rough char-based figure purely so error paths can still
    release their budget reservation."""
    if adapter is not None:
        return adapter.count_tokens(messages, completion_text)

    prompt_tokens = max(1, int(sum(len(m.get("content", "")) for m in messages) / 4.0))
    completion_tokens = max(0, int(len(completion_text) / 4.0))
    return prompt_tokens, completion_tokens


def _adapter_cost(
    router: Router, backend_id: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Compute cost against the adapter that served the request, if known."""
    adapter = _find_adapter(router, backend_id)
    if adapter is None:
        return 0.0
    return (prompt_tokens / 1000.0) * adapter.cost_per_1k_prompt + (
        completion_tokens / 1000.0
    ) * adapter.cost_per_1k_completion


def _finalize_stream_chunk(
    chunk: Dict[str, Any],
    vault_mapping: Dict[str, str],
    pii_vault: PiiVault,
    guardrails_pipeline: GuardrailsPipeline,
) -> tuple:
    """Apply output-guardrail secret redaction (on masked text) and PII vault
    restore to a streaming chunk's delta content.

    Returns (masked_content_for_accumulation, outgoing_chunk). Masked tokens
    contain no spaces, so word-level deltas keep them intact."""
    choices = chunk.get("choices") or []
    if not choices:
        return None, chunk
    delta = choices[0].get("delta") or {}
    content = delta.get("content", "")
    if not content:
        return "", chunk

    content = guardrails_pipeline.sanitize_completion(content)

    restored = (
        pii_vault.restore_text(content, vault_mapping) if vault_mapping else content
    )
    if restored == delta.get("content", ""):
        return content, chunk  # nothing redacted or restored; skip the copy

    delta_copy = {**delta, "content": restored}
    chunk_copy = dict(chunk)
    chunk_copy["choices"] = [{**choices[0], "delta": delta_copy}] + list(choices[1:])
    return content, chunk_copy


async def health_check_loop():
    while True:
        try:
            state = _state
            if state is not None:
                for adapter in state.router.adapters:
                    is_healthy = await adapter.health_check()
                    breaker = state.circuit_registry.get_breaker(adapter.id)
                    if is_healthy:
                        await breaker.record_success()
                    else:
                        await breaker.record_failure()
        except Exception as e:
            logging.error(f"Health check loop error: {e}")
        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _state

    # Init ledger & async queue
    ledger = LedgerStore("ledger.db")
    ledger_queue = AsyncLedgerQueue(ledger)
    ledger_queue.start()

    raw_env_budgets = os.environ.get("GATEWAY_BUDGETS_JSON") or os.environ.get(
        "GATEWAY_BUDGETS"
    )
    if raw_env_budgets:
        try:
            parsed = json.loads(raw_env_budgets)
            budgets = (
                parsed.get("budgets", parsed) if isinstance(parsed, dict) else parsed
            )
        except Exception as e:
            logger.error(f"Failed to parse GATEWAY_BUDGETS environment variable: {e}")
            budgets = []
    else:
        budgets_config_path = os.environ.get(
            "BUDGETS_CONFIG_PATH",
            os.path.join(os.path.dirname(__file__), "..", "configs", "budgets.yaml"),
        )
        budgets = load_config(budgets_config_path).get("budgets", [])

    await ledger.load_budgets_from_config(budgets)
    load_api_keys(ledger_store=ledger)
    set_ledger(ledger)  # inject ledger into auth module for DB-backed rate limiting

    prompt_cache = PromptCache(ttl_seconds=300)
    pii_sanitizer = PiiSanitizer()
    pii_vault = PiiVault()
    guardrails_pipeline = GuardrailsPipeline()
    budget_policy = BudgetPolicy(ledger)

    # Init circuit breaker
    cb_config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "circuit_breaker.yaml"
    )
    cb_config = load_config(cb_config_path).get("circuit_breaker", {})
    circuit_registry = CircuitBreakerRegistry(
        ledger=ledger,
        failure_threshold=cb_config.get("failure_threshold", 3),
        cooldown_sec=cb_config.get("cooldown_period_sec", 30),
    )

    # Init adapters
    backends_config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "backends.yaml"
    )
    backends = load_config(backends_config_path).get("backends", [])
    adapters: List[BaseAdapter] = []
    for cfg in backends:
        if cfg["provider"] == "openai":
            adapters.append(OpenAIAdapter(cfg))
        elif cfg["provider"] == "anthropic":
            adapters.append(AnthropicAdapter(cfg))
        elif cfg["provider"] in ("local_vllm", "ollama"):
            adapters.append(LocalVLLMAdapter(cfg))

    # Init router
    routing_config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "routing_policy.yaml"
    )
    routing_cfg = load_config(routing_config_path).get("routing", {})
    strategy = routing_cfg.get("strategy", "cost_first")
    min_tokens_complex = routing_cfg.get("min_tokens_for_complex", 500)
    router = Router(
        adapters,
        circuit_registry,
        strategy=strategy,
        timeout_sec=cb_config.get("request_timeout_sec", 30.0),
        min_tokens_for_complex=min_tokens_complex,
    )

    if not os.environ.get("ADMIN_API_KEY"):
        logger.warning(
            "ADMIN_API_KEY environment variable is not configured. All admin endpoints will be disabled."
        )

    _state = GatewayState(
        ledger=ledger,
        ledger_queue=ledger_queue,
        budget_policy=budget_policy,
        circuit_registry=circuit_registry,
        router=router,
        prompt_cache=prompt_cache,
        pii_sanitizer=pii_sanitizer,
        pii_vault=pii_vault,
        guardrails_pipeline=guardrails_pipeline,
    )

    bg_task = asyncio.create_task(health_check_loop())

    yield

    bg_task.cancel()
    _state = None
    await ledger_queue.stop()
    await router.close()


app = FastAPI(
    title="Enterprise LLM Gateway",
    description="Cost-controlling, routing LLM Gateway",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/metrics", make_asgi_app())

admin_key_header = APIKeyHeader(name="X-Admin-Token", auto_error=False)


def verify_admin_key(admin_key_header: str = Security(admin_key_header)):
    expected_token = os.environ.get("ADMIN_API_KEY")
    if not expected_token:
        logger.error(
            "Admin endpoint accessed but ADMIN_API_KEY environment variable is not set"
        )
        raise HTTPException(
            status_code=500,
            detail="ADMIN_API_KEY environment variable is not configured on the server",
        )
    if not admin_key_header or not secrets.compare_digest(
        admin_key_header, expected_token
    ):
        raise HTTPException(status_code=401, detail="Unauthorized Admin access")


@app.get("/")
async def root():
    return {
        "service": "Enterprise LLM Gateway",
        "status": "online",
        "documentation": "/docs",
        "health": "/health",
        "backends_health": "/health/backends",
        "metrics": "/metrics",
        "dashboard_ui": "http://localhost:8501",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/admin/budgets")
async def get_admin_budgets(admin_key: str = Depends(verify_admin_key)):
    try:
        return await get_state().ledger.get_all_budgets()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/requests")
async def get_admin_requests(
    limit: int = 50, admin_key: str = Depends(verify_admin_key)
):
    try:
        return await get_state().ledger.get_all_requests(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/circuit-breakers")
async def get_admin_circuit_breakers(admin_key: str = Depends(verify_admin_key)):
    try:
        return await get_state().ledger.get_all_circuit_breakers()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/admin/budgets/{api_key}")
async def update_budget(
    api_key: str, body: Dict[str, Any], admin_key: str = Depends(verify_admin_key)
):
    daily = body.get("daily_limit_usd")
    monthly = body.get("monthly_limit_usd")
    requests_per_minute = body.get("requests_per_minute")
    if daily is None or monthly is None:
        raise HTTPException(
            status_code=400, detail="daily_limit_usd and monthly_limit_usd are required"
        )
    try:
        ledger = get_state().ledger
        is_update = await ledger.update_budget_limits(
            api_key, daily, monthly, requests_per_minute
        )
        load_api_keys(ledger_store=ledger)
        return {"status": "success", "updated": is_update}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/circuit-breakers/{backend_id}/reset")
async def reset_circuit_breaker(
    backend_id: str, admin_key: str = Depends(verify_admin_key)
):
    try:
        breaker = get_state().circuit_registry.get_breaker(backend_id)
        await breaker.record_success()
        return {
            "status": "success",
            "message": f"Circuit breaker for {backend_id} has been reset to CLOSED",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/reload-config")
async def reload_config_endpoint(admin_key: str = Depends(verify_admin_key)):
    state = get_state()
    try:
        config_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
        cm = ConfigManager(config_dir)
        new_adapters = cm.load_backends()
        new_strategy = cm.load_routing_strategy()
        new_budgets = cm.load_budgets()

        if new_adapters:
            old_adapters = state.router.adapters
            state.router.adapters = new_adapters
            state.router.strategy = new_strategy
            for old_adapter in old_adapters:
                try:
                    await old_adapter.close()
                except Exception as e:
                    logger.error(
                        f"Error closing old adapter {getattr(old_adapter, 'id', 'unknown')}: {e}"
                    )
        if new_budgets:
            await state.ledger.load_budgets_from_config(new_budgets)
            load_api_keys(ledger_store=state.ledger)

        return {
            "status": "success",
            "message": f"Reloaded configuration: {len(new_adapters)} adapters, strategy '{new_strategy}'",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to reload config: {str(e)}"
        )


@app.get("/health/backends")
async def health_backends():
    try:
        state = get_state()
        results = {}
        for adapter in state.router.adapters:
            is_healthy = await adapter.health_check()
            breaker = state.circuit_registry.get_breaker(adapter.id)
            cb_state = await breaker.get_state()
            if is_healthy:
                await breaker.record_success()
            else:
                await breaker.record_failure()
            results[adapter.id] = {
                "provider": adapter.config["provider"],
                "model": adapter.model,
                "healthy": is_healthy,
                "circuit_breaker_state": cb_state.value,
                "cost_per_1k_prompt": adapter.cost_per_1k_prompt,
                "cost_per_1k_completion": adapter.cost_per_1k_completion,
            }
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request, response: Response, api_key: str = Depends(verify_api_key)
):
    state = get_state()
    request_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id

    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="Messages list is required")

    # Guardrails Pipeline Validation
    try:
        state.guardrails_pipeline.validate_messages(messages)
    except GuardrailViolationException as e:
        raise HTTPException(status_code=400, detail=str(e))

    # PII masking (reversible): the LLM only ever sees placeholders; the
    # vault mapping restores the caller's original values on the response.
    sanitized_messages, vault_mapping = state.pii_vault.mask_messages(messages)

    is_stream = body.get("stream", False)
    # "model" stays in kwargs (unlike "messages") so the router can honor it
    # as a preference in get_ranked_adapters; whichever adapter actually
    # serves the request overwrites it with its own configured model name
    # before the upstream call, so this never reaches a backend verbatim.
    kwargs = {k: v for k, v in body.items() if k != "messages"}

    # Cache Check
    cached_response = state.prompt_cache.get(sanitized_messages, kwargs)
    if cached_response is not None:
        observe_cache(hit=True)
        with GatewayTracer.trace_span(
            "chat.completion.cache",
            attributes={
                "api_key_hash": _key_hash(api_key),
                "request_id": request_id,
                "cache_hit": True,
                "model": cached_response.get("model", "cached-model"),
            },
        ):
            # Record 0 cost request in the ledger
            await state.ledger_queue.record_request(
                api_key=api_key,
                req_id=cached_response.get("id", f"cache-{uuid.uuid4()}"),
                backend="cache",
                model=cached_response.get("model", "cached-model"),
                prompt_tokens=cached_response.get("usage", {}).get("prompt_tokens", 0),
                comp_tokens=cached_response.get("usage", {}).get(
                    "completion_tokens", 0
                ),
                cost=0.0,
                latency=0.0,
            )
        if is_stream:

            async def cached_stream_generator():
                content = cached_response["choices"][0]["message"]["content"]
                words = content.split(" ")
                for i, word in enumerate(words):
                    space = " " if i > 0 else ""
                    if vault_mapping:
                        word = state.pii_vault.restore_text(word, vault_mapping)
                    chunk = {
                        "id": cached_response["id"],
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": cached_response["model"],
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": space + word},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.01)
                final_chunk = {
                    "id": cached_response["id"],
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": cached_response["model"],
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                cached_stream_generator(), media_type="text/event-stream"
            )
        if vault_mapping:
            # Restore on a copy — the cache entry must stay in masked form so
            # future hits restore against their own request's mapping.
            cached_response = json.loads(json.dumps(cached_response))
            for choice in cached_response.get("choices", []):
                message = choice.get("message", {})
                if message.get("content"):
                    message["content"] = state.pii_vault.restore_text(
                        message["content"], vault_mapping
                    )
        return cached_response

    observe_cache(hit=False)

    # Dynamic preflight cost estimation
    try:
        try:
            ranked_adapters = await state.router.get_ranked_adapters(
                messages=sanitized_messages, **kwargs
            )
            target_adapter = ranked_adapters[0] if ranked_adapters else None
        except NoAvailableBackendException:
            target_adapter = None
        except ContextLengthExceededException as e:
            # Fail fast: no backend can fit this request, so retrying or
            # failing over cannot help. 400 matches the OpenAI convention for
            # context_length_exceeded. This runs before the streaming/
            # non-streaming split, so both get a clean error response rather
            # than a 200 stream that immediately errors.
            observe_request("unknown", "error", 0.0, 0.0)
            raise HTTPException(status_code=400, detail=str(e))

        if target_adapter:
            prompt_tokens = target_adapter.count_prompt_tokens(sanitized_messages)

            # Reserve against max_tokens (the ceiling this request could
            # reach) rather than a guess at actual length — under-reserving
            # would let a request slip past a budget it then exceeds. The
            # reservation is reconciled down to real usage once the response
            # completes, so a high ceiling costs headroom, not money.
            completion_tokens = body.get("max_tokens", 256)

            prompt_cost = (prompt_tokens / 1000.0) * target_adapter.cost_per_1k_prompt
            completion_cost = (
                completion_tokens / 1000.0
            ) * target_adapter.cost_per_1k_completion
            estimated_cost = prompt_cost + completion_cost
        else:
            estimated_cost = 0.0

        await state.budget_policy.check_and_reserve(
            api_key, estimated_cost=estimated_cost
        )
    except BudgetExceededException as e:
        raise HTTPException(status_code=429, detail=str(e))

    if is_stream:

        async def stream_generator():
            accumulated_text = ""
            last_chunk = None
            stream_backend_id = "unknown"
            stream_model = "unknown"
            guardrail_flagged = False
            stream_start = time.time()
            backend_info: Dict[str, str] = {}
            stream_usage: Dict[str, int] = {}
            reconciled = False

            with GatewayTracer.trace_span(
                "chat.completion.stream",
                attributes={
                    "api_key_hash": _key_hash(api_key),
                    "request_id": request_id,
                    "stream": True,
                    "cache_hit": False,
                },
            ) as span:
                try:
                    kwargs["api_key"] = api_key
                    async for chunk in state.router.execute_stream(
                        sanitized_messages, backend_info=backend_info, **kwargs
                    ):
                        last_chunk = chunk
                        masked_delta, outgoing = _finalize_stream_chunk(
                            chunk,
                            vault_mapping,
                            state.pii_vault,
                            state.guardrails_pipeline,
                        )
                        if masked_delta:
                            accumulated_text += masked_delta
                        # Capture the model from the first chunk that has it
                        if chunk.get("model") and stream_model == "unknown":
                            stream_model = chunk["model"]
                        stream_backend_id = backend_info.get(
                            "backend_id", stream_backend_id
                        )
                        # Providers that support it append a final usage-bearing
                        # chunk; that is ground truth and beats any local
                        # estimate, so keep it if it arrives.
                        if chunk.get("usage"):
                            stream_usage = chunk["usage"]
                        yield f"data: {json.dumps(outgoing)}\n\n"

                    # ── Output guardrail validation on the full completion ────
                    try:
                        state.guardrails_pipeline.validate_completion(accumulated_text)
                    except GuardrailViolationException as e:
                        # Chunks were already streamed to the client; flag so
                        # the completion is not cached and the event is auditable.
                        guardrail_flagged = True
                        logging.warning(f"{e} (request {request_id})")

                    # ── Spend recording ────────────────────────────────────────
                    latency_ms = (time.time() - stream_start) * 1000.0

                    # Prefer the provider's own usage when the backend reports
                    # it (OpenAI-compatible `stream_options.include_usage`,
                    # or Anthropic's message_delta, normalized by the adapter).
                    # Fall back to counting the accumulated text locally for
                    # backends that report nothing. Either way, bill for what
                    # was generated — billing the reservation instead (sized to
                    # max_tokens) charged every stream its ceiling regardless
                    # of how little it produced.
                    if stream_usage:
                        prompt_tokens_count = stream_usage.get("prompt_tokens", 0)
                        completion_tokens_count = stream_usage.get(
                            "completion_tokens", 0
                        )
                    else:
                        serving_adapter = _find_adapter(state.router, stream_backend_id)
                        prompt_tokens_count, completion_tokens_count = _count_tokens(
                            sanitized_messages,
                            accumulated_text,
                            adapter=serving_adapter,
                        )
                    completion_tokens_count = max(1, completion_tokens_count)

                    actual_cost = _adapter_cost(
                        state.router,
                        stream_backend_id,
                        prompt_tokens_count,
                        completion_tokens_count,
                    )

                    observe_request(
                        stream_backend_id, "success", latency_ms, actual_cost
                    )

                    await state.ledger_queue.record_request(
                        api_key=api_key,
                        req_id=request_id,  # gateway UUID, not a backend chunk id
                        backend=stream_backend_id,
                        model=stream_model,
                        prompt_tokens=prompt_tokens_count,
                        comp_tokens=completion_tokens_count,
                        cost=actual_cost,
                        latency=latency_ms,
                        reserved_cost=estimated_cost,  # reconcile pre-reservation
                    )
                    reconciled = True

                    if span:
                        span.set_attribute("backend_id", stream_backend_id)
                        span.set_attribute("model", stream_model)
                        span.set_attribute("cost_usd", actual_cost)
                        span.set_attribute("guardrail_flagged", guardrail_flagged)

                    # Cache the finished streaming response (masked form, never
                    # cache guardrail-flagged completions)
                    if last_chunk and not guardrail_flagged:
                        cached_response_dict = {
                            "id": last_chunk.get("id", f"cache-{uuid.uuid4()}"),
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": stream_model,
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": accumulated_text,
                                    },
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": prompt_tokens_count,
                                "completion_tokens": completion_tokens_count,
                                "total_tokens": prompt_tokens_count
                                + completion_tokens_count,
                            },
                        }
                        state.prompt_cache.set(
                            sanitized_messages, kwargs, cached_response_dict
                        )

                    yield "data: [DONE]\n\n"

                except asyncio.TimeoutError as e:
                    latency_ms = (time.time() - stream_start) * 1000.0
                    observe_request(stream_backend_id, "error", latency_ms, 0.0)
                    logging.error(f"Streaming timeout: {e}")
                    try:
                        p_tok, c_tok = _count_tokens(
                            sanitized_messages,
                            accumulated_text,
                            adapter=_find_adapter(state.router, stream_backend_id),
                        )
                        # Record the partial spend so record_request reconciles
                        # (releases) the amount reserved in the preflight check.
                        await state.ledger_queue.record_request(
                            api_key=api_key,
                            req_id=request_id,
                            backend=stream_backend_id,
                            model=stream_model,
                            prompt_tokens=p_tok,
                            comp_tokens=c_tok,
                            cost=_adapter_cost(
                                state.router, stream_backend_id, p_tok, c_tok
                            ),
                            latency=latency_ms,
                            reserved_cost=estimated_cost,
                        )
                        reconciled = True
                    except Exception as ledger_err:
                        logging.error(
                            f"Failed to reconcile streaming budget reservation: {ledger_err}"
                        )
                    yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'gateway_timeout'}})}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    latency_ms = (time.time() - stream_start) * 1000.0
                    observe_request(stream_backend_id, "error", latency_ms, 0.0)
                    logging.error(f"Streaming error: {e}")
                    try:
                        p_tok, c_tok = _count_tokens(
                            sanitized_messages,
                            accumulated_text,
                            adapter=_find_adapter(state.router, stream_backend_id),
                        )
                        await state.ledger_queue.record_request(
                            api_key=api_key,
                            req_id=request_id,
                            backend=stream_backend_id,
                            model=stream_model,
                            prompt_tokens=p_tok,
                            comp_tokens=c_tok,
                            cost=_adapter_cost(
                                state.router, stream_backend_id, p_tok, c_tok
                            ),
                            latency=latency_ms,
                            reserved_cost=estimated_cost,
                        )
                        reconciled = True
                    except Exception as ledger_err:
                        logging.error(
                            f"Failed to reconcile streaming budget reservation: {ledger_err}"
                        )
                    yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'gateway_error'}})}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    # Safety net for paths the except clauses above can't catch:
                    # a client disconnect (e.g. dashboard timeout) raises
                    # GeneratorExit here, which `except Exception` does not
                    # catch, so without this the budget reservation taken
                    # upfront would never be released.
                    if not reconciled:
                        try:
                            p_tok, c_tok = _count_tokens(
                                sanitized_messages,
                                accumulated_text,
                                adapter=_find_adapter(state.router, stream_backend_id),
                            )
                            await state.ledger_queue.record_request(
                                api_key=api_key,
                                req_id=request_id,
                                backend=stream_backend_id,
                                model=stream_model,
                                prompt_tokens=p_tok,
                                comp_tokens=c_tok,
                                cost=_adapter_cost(
                                    state.router, stream_backend_id, p_tok, c_tok
                                ),
                                latency=(time.time() - stream_start) * 1000.0,
                                reserved_cost=estimated_cost,
                            )
                        except Exception as ledger_err:
                            logging.error(
                                f"Failed to reconcile abandoned stream budget "
                                f"reservation: {ledger_err}"
                            )

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    with GatewayTracer.trace_span(
        "chat.completion",
        attributes={
            "api_key_hash": _key_hash(api_key),
            "request_id": request_id,
            "stream": False,
            "cache_hit": False,
        },
    ) as span:
        try:
            backend_response = await state.router.execute(sanitized_messages, **kwargs)
            observe_request(
                backend_response.backend_id,
                "success",
                backend_response.latency_ms,
                backend_response.cost_usd,
            )
            # Which backend actually answered. The body's `model` field says
            # this too, but a caller that asked for one model and silently got
            # another has no way to notice without an explicit signal.
            response.headers["X-Backend-Id"] = backend_response.backend_id
            response.headers["X-Backend-Model"] = backend_response.model
            if backend_response.is_fallback:
                response.headers["X-Backend-Fallback"] = "true"
        except ContextLengthExceededException as e:
            observe_request("unknown", "error", 0.0, 0.0)
            raise HTTPException(status_code=400, detail=str(e))
        except NoAvailableBackendException as e:
            observe_request("unknown", "error", 0.0, 0.0)
            raise HTTPException(status_code=503, detail=str(e))
        except asyncio.TimeoutError as e:
            observe_request("unknown", "error", 0.0, 0.0)
            raise HTTPException(status_code=504, detail=f"Gateway timeout: {str(e)}")
        except Exception as e:
            observe_request("unknown", "error", 0.0, 0.0)
            raise HTTPException(
                status_code=500, detail=f"Gateway internal error: {str(e)}"
            )

        # Record spend
        await state.ledger_queue.record_request(
            api_key=api_key,
            req_id=request_id,
            backend=backend_response.backend_id,
            model=backend_response.model,
            prompt_tokens=backend_response.prompt_tokens,
            comp_tokens=backend_response.completion_tokens,
            cost=backend_response.cost_usd,
            latency=backend_response.latency_ms,
            reserved_cost=estimated_cost,
        )

        # Output guardrails: redact leaked secrets on the masked text, then
        # reject the response if the model regurgitated its instructions.
        sanitized_contents = [
            state.guardrails_pipeline.sanitize_completion(m.content)
            for m in backend_response.messages
        ]
        try:
            state.guardrails_pipeline.validate_completion("\n".join(sanitized_contents))
        except GuardrailViolationException as e:
            if span:
                span.set_attribute("guardrail_violation", True)
            raise HTTPException(status_code=400, detail=str(e))

        # JSON mode was used to *route* to a capable backend but the output
        # was never checked, so prose — or a ```json fenced block, which is
        # not itself valid JSON — passed through as success. Unwrap fences
        # where possible and flag what remains unparseable.
        if state.guardrails_pipeline.json_requested(kwargs):
            repaired, all_valid = [], True
            for content in sanitized_contents:
                text, is_valid = state.guardrails_pipeline.repair_json(content)
                repaired.append(text)
                all_valid = all_valid and is_valid
            sanitized_contents = repaired
            if span:
                span.set_attribute("json_valid", all_valid)
            if not all_valid:
                logger.warning(
                    f"Backend {backend_response.backend_id} returned invalid "
                    f"JSON for a json_mode request (request {request_id})"
                )
                response.headers["X-JSON-Valid"] = "false"

        if span:
            span.set_attribute("backend_id", backend_response.backend_id)
            span.set_attribute("model", backend_response.model)
            span.set_attribute("cost_usd", backend_response.cost_usd)

        response_dict = {
            "id": backend_response.id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": backend_response.model,
            "choices": [
                {
                    "index": i,
                    "message": {"role": msg.role, "content": content},
                    "finish_reason": "stop",
                }
                for i, (msg, content) in enumerate(
                    zip(backend_response.messages, sanitized_contents)
                )
            ],
            "usage": {
                "prompt_tokens": backend_response.prompt_tokens,
                "completion_tokens": backend_response.completion_tokens,
                "total_tokens": backend_response.prompt_tokens
                + backend_response.completion_tokens,
            },
        }

        # Save to cache in masked form — future cache hits restore against
        # their own request's vault mapping. Deep copy: PromptCache stores by
        # reference and the dict is restored in place below.
        state.prompt_cache.set(
            sanitized_messages, kwargs, json.loads(json.dumps(response_dict))
        )

        # Restore the caller's PII on the returned response
        if vault_mapping:
            response_dict["choices"] = [
                {
                    **choice,
                    "message": {
                        **choice["message"],
                        "content": state.pii_vault.restore_text(
                            choice["message"]["content"], vault_mapping
                        ),
                    },
                }
                for choice in response_dict["choices"]
            ]

        return response_dict
