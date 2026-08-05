# LLM Gateway — Enterprise-Grade Improvement Roadmap

> Full codebase audit completed. Improvements are grouped by **priority** and **category**.
> Each item links to the exact file/line that needs changing.

---

## 🔴 Phase 1 — Critical Bugs & Security Gaps (Fix Now)

These are production-blocking issues. They must be resolved before this project can be called production-ready.

### 1.1 — Budget is Never Reset (Broken Daily/Monthly Windows)
**File:** `gateway/ledger/store.py`

`spend_today` and `spend_month` are stored as raw running totals in SQLite and are **never reset**. There is no daily reset cron job, no timestamp-based window, and no `date` column on the `budgets` table. This means:
- `spend_today` accumulates forever — a user will get permanently blocked after the first day.
- Budget enforcement is mathematically wrong.

**Fix:** Add a `last_reset_date` column. On every call to `get_budget()`, compare the current date. If a new day has started, zero out `spend_today`. Same for month.

---

### 1.2 — Budget Preflight Uses a Fake Estimated Cost
**File:** `gateway/main.py` → Line 85

```python
budget_policy.check_preflight(api_key, estimated_cost=0.01)
```

The estimated cost is hardcoded to `$0.01` for **every single request**. This means the preflight check is completely meaningless — it won't block a user until they have spent roughly `daily_limit / 0.01` requests. A request with 100k tokens costs the same as a 10-token one at preflight time.

**Fix:** Implement a token-count estimator using `tiktoken` or a character-based heuristic. Calculate an estimated cost based on the actual message content length before routing the request.

---

### 1.3 — Auth Reads the YAML File on Every Single Request
**File:** `gateway/auth.py` → `get_valid_api_keys()`

```python
def get_valid_api_keys() -> set:
    config_path = ...
    budgets = load_config(config_path).get("budgets", [])
```

This function opens, reads, and parses `budgets.yaml` from disk on **every authenticated request**. Under load, this is a serious I/O bottleneck and a latency killer.

**Fix:** Load valid API keys once at startup (in `lifespan`), store them in a module-level set, and pass them as a dependency. Optionally add a TTL-based refresh.

---

### 1.4 — Circuit Breaker State is Ephemeral (In-Memory Only)
**File:** `gateway/policy/circuit_breaker.py`

All circuit breaker state (`OPEN/CLOSED/HALF_OPEN`, `consecutive_failures`) lives in a Python dict in memory. If the gateway crashes or restarts, all circuit state is wiped. This means a flapping backend that just caused 50 failures will appear "healthy" again immediately after a deploy.

**Fix:** Persist circuit breaker state to Redis or SQLite. The registry should load state on startup and write back on every `record_failure()` / `record_success()`.

---

### 1.5 — `LedgerStore` Singleton Breaks Test Isolation
**File:** `gateway/ledger/store.py` → Lines 8-16

The `LedgerStore` is a global singleton. This means unit tests that instantiate `LedgerStore(":memory:")` will all share the **same instance** after the first test runs. `test_budget_enforcement.py` tries to workaround this by calling `ledger._init_db(":memory:")`, which is a hack that doesn't actually create a new isolated DB.

**Fix:** Remove the Singleton pattern. Use proper dependency injection (FastAPI `Depends`) to pass the `LedgerStore` instance through the app. Tests can then create fresh instances.

---

### 1.6 — `kwargs` Passed Directly to LLM API (Injection Risk)
**File:** `gateway/adapters/local_vllm_adapter.py`, `openai_adapter.py`

```python
json={
    "model": self.model,
    "messages": messages,
    **kwargs  # ← Any key the client sends goes straight to the backend
}
```

The raw HTTP request body is unpacked and forwarded directly to the backend LLM APIs. A client can send arbitrary keys like `api_key`, `user`, `n: 100` (to get 100 completions), or `stream: true` (to break the response parser). This is an API injection vulnerability.

**Fix:** Use an explicit allowlist of permitted kwargs (`temperature`, `max_tokens`, `top_p`, `stream`, `stop`). Strip all other keys before forwarding.

---

## 🟠 Phase 2 — Architectural Weaknesses (Fix This Week)

These are design-level issues that limit scalability, correctness, and maintainability.

### 2.1 — Prometheus Metrics are Defined but Never Emitted
**File:** `gateway/telemetry/metrics.py` and `gateway/main.py`

The `observe_request()` function and all the Prometheus counters/histograms are defined but **never called anywhere** in `main.py` or the router. There is also no `/metrics` endpoint exposed. The entire telemetry module is dead code.

**Fix:** 
1. Call `observe_request()` in `router.py` after each success/failure.
2. Update `CIRCUIT_BREAKER_STATE` gauge in `record_failure()` and `record_success()`.
3. Expose a `/metrics` endpoint using `prometheus_client.make_asgi_app()`.

---

### 2.2 — No Request ID Tracing Across the Stack
**File:** `gateway/main.py`

There is no `X-Request-ID` / `X-Trace-ID` header generated or propagated. When something goes wrong, you cannot correlate a client's failed request with a specific log line in Uvicorn, a row in the ledger, or a backend response.

**Fix:** Generate a UUID `request_id` at the top of the `chat_completions` handler. Pass it to the router, store it in the ledger, and return it to the client in the response headers as `X-Request-ID`.

---

### 2.3 — Router Has No Latency-First Strategy Implementation
**File:** `gateway/policy/router.py` → Lines 43-45

```python
elif self.strategy == "latency_first":
    # Real implementation would use historical latency
    pass
```

The `latency_first` strategy is a stub. The config supports it, but it silently does nothing — backends are returned in their original unsorted order.

**Fix:** Track a rolling average latency per backend (using an `EMA` — exponential moving average), stored either in Redis or an in-memory dict on the router. Sort by this metric when `latency_first` is selected.

---

### 2.4 — `created` Field in Response is Wrong
**File:** `gateway/main.py` → Line 112

```python
"created": int(response.latency_ms),  # mockup timestamp
```

The `created` field in the OpenAI-compatible response is supposed to be a Unix timestamp (seconds since epoch). Instead, it's being set to the request latency in milliseconds, which is a completely wrong value (e.g., `1052` instead of `1752413400`). Any client parsing this as a timestamp will get garbage.

**Fix:** Replace with `int(time.time())`.

---

### 2.5 — Dashboard Reads SQLite Directly (Bypasses Gateway)
**File:** `dashboard/app.py` → Lines 60-77

The Streamlit dashboard connects directly to `ledger.db` via a raw SQLite connection. This tight coupling means:
- The dashboard cannot work when the gateway runs in a separate Docker container (different filesystem).
- There's no access control — any Streamlit user can read all API keys and budgets.
- Adding a new backend storage (e.g., PostgreSQL) requires changing the dashboard too.

**Fix:** Expose dedicated read-only API endpoints on the gateway (`GET /admin/budgets`, `GET /admin/requests`, `GET /admin/circuit-breakers`). The dashboard should call these endpoints. Protect them with an admin API key.

---

### 2.6 — Singleton `httpx.AsyncClient` Creates a New Connection Per Request
**File:** `gateway/adapters/local_vllm_adapter.py`, `openai_adapter.py`, `anthropic_adapter.py`

```python
async with httpx.AsyncClient() as client:
```

A new `httpx.AsyncClient()` is created and destroyed for every single request. This means there is no HTTP connection pooling to the backend LLM servers. Each request incurs TCP handshake overhead.

**Fix:** Create a shared `httpx.AsyncClient` per adapter in `__init__()` (or as a lifespan-managed app-level client). Use `limits=httpx.Limits(max_keepalive_connections=20)` for connection pooling.

---

## 🟡 Phase 3 — Missing Enterprise Features (Implement Next)

These are the features that separate a hobby project from a real enterprise tool.

### 3.1 — No Rate Limiting (Requests Per Minute)
Currently only USD budget limits are enforced. A client could send 10,000 requests in one second (all very cheap ones) and there is nothing to stop them. Real enterprise gateways enforce RPM/TPM limits.

**Add:** A per-API-key rate limiter using a sliding window counter (token bucket algorithm). Can be implemented with an in-memory dict for single-node or Redis for multi-node. Expose limit headers (`X-RateLimit-Remaining`, `X-RateLimit-Reset`) in responses.

---

### 3.2 — No Streaming Support
**File:** `gateway/main.py`

The gateway only supports non-streaming (`stream: false`) completions. All modern LLM clients default to `stream: true` for a better user experience. Without streaming, users see a blank screen until the full response is ready.

**Add:** Detect `stream: true` in the request body. Use `httpx` streaming responses and return a `StreamingResponse` from FastAPI that proxies the SSE (Server-Sent Events) chunks directly to the client.

---

### 3.3 — No Health Check Endpoint for Backends
**File:** `gateway/main.py`

The `health_check()` method is defined on every adapter but is **never called**. The gateway discovers that a backend is down only when a live user request fails (and then records a circuit breaker failure). This is reactive, not proactive.

**Add:** A background `asyncio` task (started in `lifespan`) that pings all backend health endpoints every N seconds. Update the circuit breaker state proactively. Expose the results at `GET /health/backends`.

---

### 3.4 — No Admin API for Dynamic Configuration
Currently, changing a budget, adding a new backend, or adjusting the circuit breaker threshold requires editing YAML files and restarting the server.

**Add:** A protected `/admin` API set:
- `POST /admin/backends` — add a backend at runtime
- `PATCH /admin/budgets/{api_key}` — update spending limits
- `GET /admin/circuit-breakers` — view current state of all breakers
- `POST /admin/circuit-breakers/{id}/reset` — manually reset a tripped breaker

---

### 3.5 — No Retry Logic with Exponential Backoff
**File:** `gateway/policy/router.py`

The router tries each backend exactly once. If a backend returns a transient error (e.g., HTTP 429 rate limit or a 500 from Ollama), it immediately moves on and records a circuit breaker failure. There is no retry with backoff on the same backend before giving up.

**Add:** Per-backend retry logic with exponential backoff + jitter. Only record a circuit breaker failure after all retries are exhausted. Use `tenacity` library for clean retry decorators.

---

### 3.6 — No Structured Logging (JSON)
**File:** `gateway/main.py` and all modules

`logging.basicConfig(level=logging.INFO)` produces unstructured plaintext logs. In production, logs need to be machine-parseable for ingestion into Splunk, Datadog, or CloudWatch.

**Add:** Replace `logging.basicConfig` with `python-json-logger` (or `structlog`). Every log line should be JSON with fields: `timestamp`, `level`, `request_id`, `backend`, `model`, `latency_ms`, `cost_usd`.

---

## 🔵 Phase 4 — Observability & DevOps (Polish)

### 4.1 — Add a `GET /metrics` Prometheus Endpoint
Wire the existing `gateway/telemetry/metrics.py` to a real endpoint and add a `docker-compose.yml` entry for Grafana + Prometheus so the dashboard shows live metrics.

### 4.2 — Fix `docker-compose.yml`
- There is no `Dockerfile` in the repo (referenced but missing).
- There is no `dashboard.Dockerfile` (referenced but missing).
- The `local_vllm` service references `dummy.gguf` which doesn't exist.
- Docker containers depend on `local_vllm` but the current setup uses Ollama.
- The `version: '3.8'` key is deprecated in Compose V2.

### 4.3 — Add GitHub Actions CI Pipeline
**File:** `.github/` (exists but workflows are likely empty)

Add a CI workflow that: runs `pytest` on push, runs `black --check` and `isort --check`, builds the Docker image, and posts test coverage to the PR.

### 4.4 — Fill In Empty Test Files
`tests/test_failover_integration.py` and `tests/test_routing.py` are **empty files** (0 bytes). These are the most important integration tests — failover routing and the routing strategy — and they don't exist.

---

## 🟢 Phase 5 — Advanced Features (Portfolio Differentiation)

These elevate the project from "solid implementation" to "impressive portfolio piece."

### 5.1 — Multi-Tenant with Quota Tiers
Define quota tiers (`free`, `pro`, `enterprise`) in config. API keys inherit a tier's limits. `POST /admin/keys` creates a new key and assigns it to a tier.

### 5.2 — Smart Cost Estimation Before Routing
Use `tiktoken` to count tokens in the request *before* sending it. Use real token counts (not the hardcoded `$0.01`) for the preflight budget check. Return `X-Estimated-Cost` in the response header.

### 5.3 — Request/Response Caching (Semantic Cache)
Cache recent LLM responses keyed by a hash of the messages. Serve cache hits instantly with zero cost. Even a simple exact-match cache (Redis `SET` with TTL) can save 20-40% of costs on repeated similar queries.

### 5.4 — A/B Testing / Shadow Mode Routing
Route a configurable percentage of traffic to a "shadow" backend. Record its responses and latencies without returning them to the user. Enables safe evaluation of new models in production.

### 5.5 — OpenAI-Compatible `/v1/models` Endpoint
Return a list of all configured backends as if they were OpenAI model IDs. Any client that calls `GET /v1/models` can discover what's available. This makes the gateway a drop-in replacement for the OpenAI SDK.

---

## Summary Table

| Priority | Count | Status |
|----------|-------|--------|
| 🔴 Critical (bugs/security) | 6 | Must fix now |
| 🟠 Architectural | 6 | Fix this week |
| 🟡 Missing enterprise features | 6 | Implement next sprint |
| 🔵 Observability/DevOps | 4 | Polish phase |
| 🟢 Advanced/Portfolio | 5 | Differentiators |
| **Total** | **27** | |
