<div align="center">

# LLM Gateway

**Cost control, model routing, and failover for LLM traffic.**

One OpenAI-compatible endpoint that decides *which model* serves a request,
*whether the caller can afford it*, and *what the model is allowed to see*.

[![CI](https://github.com/abntazim-1/LLM-Gateway-with-Cost-Control-Fallback-Routing/actions/workflows/ci.yml/badge.svg)](https://github.com/abntazim-1/LLM-Gateway-with-Cost-Control-Fallback-Routing/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-134%20passing-brightgreen.svg)](#quality)
[![Evals](https://img.shields.io/badge/policy%20evals-61%2F61-brightgreen.svg)](#quality)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

</div>

---

## The problem

Every application that calls an LLM re-implements the same concerns: budget
limits, retries when a provider fails, deciding which model is worth paying for,
and keeping sensitive data out of a third party's logs. Implemented per app,
these drift apart, and the failures are quiet — a runaway bill, an outage that
surfaces as a broken feature, PII in someone else's telemetry.

This gateway moves those decisions into one layer that every application shares.

```
                       ┌──────────────────────────┐
                       │   Application / Client   │
                       └────────────┬─────────────┘
                                    │  POST /v1/chat/completions
                                    ▼
   ┌────────────────────────────────────────────────────────────────┐
   │                      FastAPI Data Plane                        │
   │                                                                │
   │   Auth + rate limit  →  Guardrails  →  PII vault (mask)        │
   │        →  Prompt cache  →  Budget reservation                  │
   └────────────────────────────────┬───────────────────────────────┘
                                    ▼
                       ┌──────────────────────────┐
                       │         Router           │
                       │  filter: health, context │
                       │          capability      │
                       │  rank:   strategy        │
                       └────────────┬─────────────┘
                                    │  failover · retry · cascade
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
  ┌────────────┐            ┌──────────────┐            ┌──────────────┐
  │   OpenAI   │            │  Anthropic   │            │ vLLM/Ollama  │
  └────────────┘            └──────────────┘            └──────────────┘
        └───────────────────────────┼───────────────────────────┘
                                    ▼
              PII restore  →  output guardrails  →  cache write
                     →  spend reconciliation (async ledger)
```

---

## Routing in action

Same endpoint, no model specified, `complexity` strategy — the router picks
based on what the request actually needs:

```bash
curl -sD- localhost:8080/v1/chat/completions -H "Authorization: Bearer $KEY" -d '{"messages":[{"role":"user","content":"hi there"}]}' | grep -i x-backend-model
```
```
x-backend-model: qwen2.5:0.5b        ← $0.00005 / 1k
```

```bash
curl -sD- localhost:8080/v1/chat/completions -H "Authorization: Bearer $KEY" -d '{"messages":[{"role":"user","content":"Why would a two-tier cache produce stale reads under concurrent writes?"}]}' | grep -i x-backend-model
```
```
x-backend-model: phi3:latest         ← $0.00040 / 1k
```

A trivial prompt costs **8× less** without the caller doing anything. Every
response carries `X-Backend-Id` and `X-Backend-Model`, so which model answered
is never a mystery — and `X-Backend-Fallback` appears when failover substituted
a different one.

---

## Capabilities

### Cost control
Requests are priced before they run and reconciled against real provider usage
afterwards. Daily and monthly budgets are checked and reserved **atomically**,
so concurrent requests cannot collectively overshoot a limit. Every request is
recorded per client and per backend in a SQLite ledger.

### Model routing
| Strategy | Selects |
|---|---|
| `cost_first` | Cheapest healthy backend |
| `latency_first` | Fastest by rolling average; unmeasured backends are probed first rather than starved |
| `complexity` | Scores the prompt and escalates only when it warrants it |
| `cascade` | Answers cheaply, then escalates **only if the answer was inadequate** |
| `round_robin` · `weighted_round_robin` | Even distribution |

Candidates are filtered before ranking: unhealthy backends, backends whose
context window cannot fit the request, and backends lacking a required
capability are excluded. Callers may pin a specific model, overriding routing.

### Reliability
A per-backend circuit breaker (`closed → open → half-open`) stops sending
traffic to failing providers, and requests fail over to the next ranked backend
automatically. Verified under test: with the primary backend deliberately
broken, **every client request still returned 200**, and once the breaker
opened, subsequent requests skipped the dead backend in under 0.5 s.

### Data handling
PII is replaced with indexed placeholders (`[EMAIL_1]`) before the prompt
reaches a provider and restored on the way back — the model never sees real
values, the caller never sees placeholders. Prompts are screened for injection
patterns including obfuscated forms; completions are screened for leaked
credentials.

### Observability
Prometheus metrics, OpenTelemetry spans, a queryable request ledger, and a
Streamlit operator dashboard covering spend, latency, cache hit rate, backend
health, and answer-quality ratings.

---

## Quickstart

Requires **Python 3.10+**. The default configuration runs entirely on local
models via [Ollama](https://ollama.com) — no provider API keys needed to try it.

```bash
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
```
```bash
pip install -e ".[dev]"
```
```bash
ollama pull qwen2.5:0.5b && ollama pull phi3
```
```bash
cp .env.example .env
```

Set `ADMIN_API_KEY` and at least one client key in `.env`. Provider keys are
only needed if you enable those backends in `configs/backends.yaml`.

**Run the gateway:**
```bash
uvicorn gateway.main:app --port 8080
```

**Run the dashboard** (optional, separate terminal):
```bash
streamlit run dashboard/app.py
```

| | |
|---|---|
| Interactive API docs | http://localhost:8080/docs |
| Operator dashboard | http://localhost:8501 |

**Send a request:**
```bash
curl -X POST http://localhost:8080/v1/chat/completions -H "Authorization: Bearer $CLIENT_API_KEY_TIER1" -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

> Omit `model` to let the router choose. A `Makefile` provides
> `make run` / `make dashboard` / `make test` where `make` is available.

---

## Deploying

Swapping local models for cloud providers is a **config change, not a code
change**. The router, every strategy, budgets, guardrails, PII masking and the
circuit breakers operate on adapters, not on any particular vendor.

```bash
BACKENDS_CONFIG_PATH=configs/backends.cloud.yaml
```

That config ships ready to use: Groq's free tier as the cheap default, GPT-4o
mini as the premium tier, and Claude Haiku as a **different-vendor** failover —
so one provider's outage doesn't take the gateway down with it. Note that
`provider: openai` means "speaks the OpenAI wire protocol", not "is OpenAI":
Groq, Together, Fireworks, OpenRouter and vLLM all use that adapter with a
different `endpoint`.

Two values must be re-derived per model rather than copied:

- **`token_overhead_per_message`** — measured, not guessed. Run
  `python docs/probes/probe_tokens.py`; a wrong value skews every budget
  estimate.
- **`capability_tier`** — higher is more capable. Escalation ranks on this,
  *not* on price. Set it backwards and "use the better model" routes to the
  weaker one.

**For a public demo**, run the free tier and issue one capped key rather than
asking visitors for their own — nobody pastes a live API key into a stranger's
gateway. A `$0.25/day, 10 req/min` key is safe to publish, because the budget
and rate limiting that bound the blast radius are the features being
demonstrated.

Keep `/admin/*` unreachable on a public instance — those endpoints hold
operator authority.

---

## API

**Inference**

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible completion, streaming or buffered |
| `POST /v1/feedback` | Rate a completed request (`+1` / `-1`) |

**Operations** — require the `X-Admin-Token` header

| Endpoint | Purpose |
|---|---|
| `GET /health` · `/health/backends` | Liveness; per-backend health, circuit state, pricing |
| `GET /metrics` | Prometheus metrics |
| `GET /admin/requests` · `/admin/budgets` · `/admin/feedback` | Ledger, budgets, quality ratings |
| `PATCH /admin/budgets/{api_key}` | Create or update a client's limits |
| `POST /admin/circuit-breakers/{id}/reset` | Force a breaker closed |
| `POST /admin/reload-config` | Hot-reload YAML config without restarting |

---

## Configuration

| File | Controls |
|---|---|
| `configs/backends.yaml` | Endpoint, pricing, `capability_tier`, `context_length`, tokenizer |
| `configs/routing_policy.yaml` | Active strategy and thresholds |
| `configs/budgets.yaml` | Per-client spend and rate limits |
| `configs/circuit_breaker.yaml` | Failure threshold, cooldown, request timeout |

`capability_tier` is declared **separately from price** deliberately. Routing
that means "use the better model" ranks on capability, not cost. The two usually
correlate — but when they diverge, ranking by price selects the *weaker* model,
which is the opposite of the intent.

---

## Quality

```bash
pytest                      # 134 tests
python evals/run_evals.py   # 61 policy eval cases
```

Tests cover routing, budget enforcement, rate limiting, circuit breaking,
failover, PII masking, guardrails, token accounting, streaming cost
reconciliation, and context-window enforcement.

The **eval harness** is separate from the test suite and answers a different
question. Unit tests pin individual behaviours; the evals score each policy
against labelled datasets so it is possible to tell whether routing, screening
and escalation are getting *better or worse overall*. They are deterministic,
need no model calls, and gate CI.

Both run on every push via GitHub Actions.

---

## Scope and limitations

This is a portfolio project, not a production deployment. Its behaviour has been
**measured rather than assumed** — including what does not work:

| Area | Measured behaviour |
|---|---|
| **Token accounting** | Per-backend tokenizers, within ~3% of real usage (was 67–97% off using a character heuristic). Streaming bills provider-reported usage where available. |
| **Injection screening** | Blocks common **English** patterns including obfuscated forms — zero-width characters, homoglyphs, leetspeak, separator padding. ⚠️ **Non-English prompts are not screened.** |
| **PII masking** | Detects structured identifiers: emails, SSNs, cards (Luhn-validated), IBANs, IPs, dates of birth, passports, API credentials. ⚠️ **Does not detect names, addresses, or medical conditions** — these need NER, not pattern matching. |
| **Answer adequacy** (`cascade`) | Detects refusals, truncation and non-answers. ⚠️ **Does not detect fluent but incorrect answers.** |

These gaps are not disclaimers in prose — each is encoded as a `known_gap` case
in `evals/datasets/`, counted on every CI run, with a test that **fails if one
is silently closed** without being recorded. Currently 8 are tracked.

---

## Project layout

```
gateway/
  main.py          FastAPI app and request pipeline
  adapters/        Provider clients, tokenization, parameter translation
  policy/          Router, budgets, circuit breaker, cache, guardrails, PII
  ledger/          SQLite spend store and async write queue
  telemetry/       Prometheus metrics, OpenTelemetry tracing
dashboard/         Streamlit operator UI
evals/             Policy evaluation harness and labelled datasets
tests/             Test suite
```

---

<div align="center">

MIT licensed — see [LICENSE](LICENSE)

</div>
