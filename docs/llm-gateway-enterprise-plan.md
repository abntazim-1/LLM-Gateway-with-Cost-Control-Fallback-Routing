# LLM Gateway with Cost Control & Fallback Routing
## Enterprise-Grade Implementation Plan

---

## 1. Objective & Success Criteria

**Objective:** Build a single ingress point that sits in front of multiple LLM providers/models (OpenAI, Anthropic, local vLLM/Ollama endpoints), enforcing budget limits, routing by cost/latency/capability, and failing over automatically — the piece of infrastructure that turns "I built 4 separate AI projects" into "I built a platform." This is the connective tissue between your RAG system, drift detection, A/B testing framework, and quantization benchmark: the gateway is what would actually sit in front of all of them in production.

**Success criteria:**
- [ ] Single OpenAI-compatible endpoint that transparently routes to 3+ backends (e.g., OpenAI, Anthropic, local vLLM)
- [ ] Hard budget enforcement (per-key, per-day) that actually blocks requests when exceeded, not just logs a warning
- [ ] Automatic fallback on provider error/timeout, demoed live (kill a backend mid-demo, traffic reroutes)
- [ ] Routing policy is configurable, not hardcoded (cheapest-first, fastest-first, capability-based)
- [ ] Cost dashboard showing spend by model/key/day
- [ ] Load-testable, with a clear "before/after" story: what happens without the gateway (one provider down = outage) vs. with it (graceful degradation)

---

## 2. Architecture Overview

```
                         ┌───────────────────────────────┐
                         │        Client Request          │
                         │  (OpenAI-compatible schema)     │
                         └───────────────┬─────────────────┘
                                         ▼
                         ┌───────────────────────────────┐
                         │         API Gateway Layer       │
                         │   FastAPI, auth, rate limit      │
                         └───────────────┬─────────────────┘
                                         ▼
                 ┌───────────────────────────────────────────┐
                 │              Policy Engine                  │
                 │  - Budget check (per-key daily/monthly)     │
                 │  - Routing strategy (cost/latency/capability)│
                 │  - Circuit breaker state per backend         │
                 └───────────────┬───────────────────────────┘
                                 ▼
             ┌───────────────────┼───────────────────┐
             ▼                   ▼                   ▼
    ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
    │  OpenAI Adapter │  │ Anthropic      │  │  Local vLLM/    │
    │                 │  │ Adapter        │  │  Ollama Adapter │
    └────────┬────────┘  └────────┬───────┘  └────────┬────────┘
             └────────────────────┼───────────────────┘
                                  ▼
                   ┌───────────────────────────────┐
                   │      Response Normalizer         │
                   │  unify schema, attach cost/       │
                   │  latency metadata                 │
                   └───────────────┬───────────────────┘
                                  ▼
                   ┌───────────────────────────────┐
                   │      Telemetry & Ledger          │
                   │  cost ledger (SQLite/Postgres),  │
                   │  request logs, Prometheus metrics │
                   └───────────────┬───────────────────┘
                                  ▼
                   ┌───────────────────────────────┐
                   │      Cost Dashboard              │
                   │  (simple web UI or Grafana)      │
                   └───────────────────────────────┘
```

**Design principle:** the policy engine and the adapters are decoupled — adding a new backend means writing one adapter class implementing a fixed interface (`complete(prompt, **kwargs) -> NormalizedResponse`), not touching routing logic. This is the detail that makes it read as "gateway" rather than "if/else script."

---

## 3. Repository Structure

```
llm-gateway/
├── README.md
├── Makefile                       # make run, make test, make loadtest, make demo-failover
├── pyproject.toml
├── configs/
│   ├── backends.yaml              # provider configs, pricing per 1K tokens, endpoints
│   ├── routing_policy.yaml        # strategy: cost_first | latency_first | capability
│   ├── budgets.yaml                # per-key daily/monthly limits
│   └── circuit_breaker.yaml       # failure threshold, cooldown period
├── gateway/
│   ├── main.py                    # FastAPI app entrypoint
│   ├── auth.py                    # API key validation
│   ├── policy/
│   │   ├── router.py              # routing strategy implementations
│   │   ├── budget.py              # budget enforcement, ledger writes
│   │   └── circuit_breaker.py     # per-backend health state machine
│   ├── adapters/
│   │   ├── base.py                # abstract adapter interface
│   │   ├── openai_adapter.py
│   │   ├── anthropic_adapter.py
│   │   └── local_vllm_adapter.py
│   ├── ledger/
│   │   ├── models.py              # SQLAlchemy models: requests, spend, backend_health
│   │   └── store.py               # read/write cost ledger
│   └── telemetry/
│       └── metrics.py             # Prometheus counters/histograms
├── dashboard/
│   └── app.py                     # simple Streamlit/FastAPI+HTMX cost dashboard
├── tests/
│   ├── test_routing.py
│   ├── test_budget_enforcement.py
│   ├── test_circuit_breaker.py
│   └── test_failover_integration.py
├── loadtest/
│   └── gateway_loadtest.py         # async client hammering the gateway
├── docker-compose.yml              # gateway + local vLLM + Postgres + Grafana
├── .github/workflows/
│   └── ci.yml                      # unit tests + one integration smoke test
└── docs/
    ├── ARCHITECTURE.md
    ├── ROUTING_POLICIES.md
    └── DEMO_SCRIPT.md               # the "kill a backend live" demo, scripted step by step
```

---

## 4. Phased Delivery Plan (3 days)

### Day 1 — Core Gateway + Adapters (fits ~6–7 hrs)
- FastAPI app exposing `/v1/chat/completions` (OpenAI-compatible schema so any existing client library works unmodified against it).
- Adapter interface (`base.py`): every backend implements `complete()`, `stream()`, `estimate_cost()`, `health_check()`.
- Three adapters: OpenAI, Anthropic, and a local vLLM/Ollama endpoint (reuse the serving stack from the quantization benchmark project here — genuine platform reuse, not just narrative reuse).
- Response normalizer: every adapter's output gets mapped to one internal schema carrying token counts, latency, and computed cost, regardless of provider-specific response shape.

### Day 2 — Policy Engine: Budget, Routing, Circuit Breaker (~6–7 hrs)
- **Budget enforcement:** ledger table (SQLite for the demo, swappable to Postgres) tracking spend per API key per day. Enforcement is pre-flight — a request that would exceed budget is rejected with a clear 429-style error *before* it hits a paid backend, not logged after the fact.
- **Routing strategies** (config-selectable, not hardcoded):
  - `cost_first` — cheapest backend that meets a minimum capability tier
  - `latency_first` — backend with best rolling p50 latency
  - `capability_based` — route by requested context length / function-calling support / etc.
- **Circuit breaker per backend:** standard closed → open → half-open state machine. N consecutive failures within a window trips it open; requests reroute to the next-best backend; a cooldown timer moves it to half-open for a trial request before fully re-closing.
- This is the day that produces the "why should I trust this over an if/else" answer — the circuit breaker state machine and pre-flight budget check are the two pieces of genuine engineering here.

### Day 3 — Telemetry, Dashboard, Load Test, Demo Polish (~6–7 hrs)
- Prometheus metrics: request count, cost, latency, and circuit-breaker state per backend, exposed on `/metrics`.
- Cost dashboard (Streamlit is fastest to ship in a day; a Grafana panel via `docker-compose` is the more "enterprise" look if time allows) — spend by key, spend by model, requests routed per backend over time.
- Async load test script hammering the gateway at increasing concurrency, with **a scripted mid-run failure injection**: kill the primary backend's health check partway through the run and show traffic reroute with zero dropped requests, only a latency blip — this is the money demo.
- `docs/DEMO_SCRIPT.md`: exact commands to reproduce the "kill a backend live" demo for interviews — this is worth writing down precisely, since the deliverable is a demo as much as a repo.
- CI: unit tests for budget enforcement and circuit breaker logic (these are the two components a reviewer will actually poke at) plus one integration smoke test that boots the whole `docker-compose` stack and confirms failover works headlessly.

---

## 5. Data Model (cost ledger)

| Table | Key columns |
|---|---|
| `requests` | `id`, `api_key`, `backend`, `model`, `prompt_tokens`, `completion_tokens`, `cost_usd`, `latency_ms`, `status`, `timestamp` |
| `budgets` | `api_key`, `daily_limit_usd`, `monthly_limit_usd`, `spend_today`, `spend_month` |
| `backend_health` | `backend`, `state` (closed/open/half_open), `consecutive_failures`, `last_failure_at`, `cooldown_until` |

---

## 6. Risk Register

| Risk | Mitigation |
|---|---|
| Real API keys cost real money during dev/demo | Use a mock adapter for local dev (fixed latency + fake token counts); only hit real OpenAI/Anthropic keys for the final recorded demo, with a tiny budget cap set intentionally low |
| Circuit breaker false-trips on transient blips | Require N consecutive failures (not 1) before opening; document the threshold as a tunable in `circuit_breaker.yaml` |
| Demo failover doesn't look convincing live | Script it precisely in `DEMO_SCRIPT.md` and rehearse; log line at the moment of failover ("backend=openai state=open, rerouting to backend=local_vllm") makes the moment legible to an interviewer watching a terminal |
| Scope creep — this can balloon past 3 days if you add per-user rate limiting, streaming, semantic caching, etc. | Freeze scope to budget + routing + circuit breaker + dashboard for the 3-day build; explicitly list caching/streaming/per-user quotas as "future work" in the README rather than starting them |
| OpenAI-compatible schema drift across providers (function calling, system prompts differ) | Normalize to the OpenAI chat schema as the lowest common denominator for v1; note provider-specific feature gaps in `ROUTING_POLICIES.md` rather than silently dropping them |

---

## 7. Deliverables Checklist

- [ ] Repo with `docker-compose up` bringing up gateway + local vLLM + Postgres + dashboard in one command
- [ ] Passing CI (budget + circuit breaker unit tests, one integration smoke test)
- [ ] Recorded or live-reproducible failover demo (`docs/DEMO_SCRIPT.md`)
- [ ] Cost dashboard screenshot/GIF in README
- [ ] Resume bullet:
  > *"Built an LLM gateway routing requests across OpenAI, Anthropic, and self-hosted vLLM backends with pre-flight budget enforcement, cost/latency-aware routing, and automatic circuit-breaker failover — demoed zero-downtime rerouting under live backend failure."*

---

## 8. How This Ties Your Portfolio Together

- **RAG system** and **A/B testing framework** become gateway *clients* in the narrative — "the gateway is what would sit in front of these in production."
- **Quantization benchmark** supplies the local vLLM backend directly — you're not re-explaining self-hosted serving, you're reusing it.
- **Drift detection** is the natural next extension (worth naming in the README as future work, not building now): gateway-level telemetry is exactly the data a drift detector would consume.
- Interview framing shifts from "here are four projects" to "here's a platform, and here are the four pieces of it I can go deep on individually."

---

*Next possible steps: scaffold `gateway/policy/circuit_breaker.py` (the state machine) or `gateway/policy/budget.py` (pre-flight enforcement) as working code — these two are the components most likely to get probed in a technical interview.*
