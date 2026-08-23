# LLM Gateway — Cost Control & Fallback Routing

A control layer between your applications and LLM providers. One
OpenAI-compatible endpoint that decides **which model** serves a request,
**whether the caller can afford it**, and **what the model is allowed to see** —
so cost, reliability, and data handling are enforced once, centrally, instead of
being re-implemented in every application that calls an LLM.

```
                       ┌──────────────────────────┐
                       │   Application / Client   │
                       └────────────┬─────────────┘
                                    │  POST /v1/chat/completions
                                    ▼
   ┌────────────────────────────────────────────────────────────────┐
   │                      FastAPI Data Plane                        │
   │                                                                │
   │   Auth + rate limit  →  Guardrails  →  PII vault (mask)         │
   │        →  Prompt cache  →  Budget reservation                   │
   └────────────────────────────────┬───────────────────────────────┘
                                    ▼
                       ┌──────────────────────────┐
                       │        Router            │
                       │  filter: health, context │
                       │  window, capability      │
                       │  rank:   strategy        │
                       └────────────┬─────────────┘
                                    │  failover · retry · cascade
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
  ┌────────────┐            ┌──────────────┐            ┌──────────────┐
  │  OpenAI    │            │  Anthropic   │            │ vLLM/Ollama  │
  └────────────┘            └──────────────┘            └──────────────┘
        └───────────────────────────┼───────────────────────────┘
                                    ▼
              PII restore  →  output guardrails  →  cache write
                     →  spend reconciliation (async ledger)
```

---

## What it does

**Cost control.** Every request is priced before it runs and reconciled against
real provider usage afterwards. Per-client daily and monthly budgets are checked
and reserved atomically, so concurrent requests cannot collectively overshoot a
limit. Spend is recorded per request, per backend, in a SQLite ledger.

**Model routing.** Six strategies decide which backend serves a request:

| Strategy | Selects |
|---|---|
| `cost_first` | Cheapest healthy backend |
| `latency_first` | Fastest by rolling average, probing unmeasured backends first |
| `complexity` | Scores the prompt; escalates only when it looks like it needs it |
| `cascade` | Answers cheaply, then escalates **only if the answer was inadequate** |
| `round_robin` / `weighted_round_robin` | Even distribution |

Callers may pin a specific model, which overrides routing entirely.

**Reliability.** A per-backend circuit breaker (closed → open → half-open) stops
sending traffic to failing providers. Requests fail over to the next ranked
backend automatically, and responses carry `X-Backend-Fallback` so a silent
model substitution is observable rather than invisible.

**Data handling.** PII is replaced with indexed placeholders (`[EMAIL_1]`)
before the prompt reaches a provider, and restored on the way back — so the
model never sees the real values but the caller still gets them. Prompts are
screened for common injection patterns; completions are screened for leaked
credentials.

**Observability.** Prometheus metrics, OpenTelemetry spans, a request ledger,
and a Streamlit dashboard for spend, latency, cache hit rate, backend health,
and answer-quality ratings.

---

## Scope and limitations

This is a portfolio project, not a production deployment. Its behaviour has been
measured rather than assumed — including what does **not** work:

| Area | Measured behaviour |
|---|---|
| Token accounting | Per-backend tokenizers; estimates within ~3% of real usage. Streaming bills provider-reported usage where available. |
| Injection screening | Blocks common **English** patterns, including obfuscated forms (zero-width characters, homoglyphs, leetspeak). **Non-English prompts are not screened.** |
| PII masking | Detects structured identifiers — emails, SSNs, cards (Luhn-validated), IBANs, IPs, dates of birth, passports, API credentials. **Does not detect names, street addresses, or medical conditions**, which need NER rather than pattern matching. |
| Answer adequacy (`cascade`) | Detects refusals, truncation, and non-answers. **Does not detect fluent but incorrect answers.** |

Those gaps are not footnotes — they are encoded as `known_gap` cases in
`evals/datasets/`, counted on every CI run, with a test that fails if one is
silently closed without being recorded.

---

## Quickstart

Requires Python 3.10+. The example configuration runs entirely on local models
via [Ollama](https://ollama.com), so no provider API keys are needed to try it.

**1. Install**

```bash
python -m venv venv && source venv/Scripts/activate  # Windows
```

```bash
pip install -e ".[dev]"
```

**2. Pull the local models used by the example config**

```bash
ollama pull qwen2.5:0.5b && ollama pull phi3
```

**3. Configure**

```bash
cp .env.example .env
```

Set `ADMIN_API_KEY` and at least one client key in `.env`. Provider keys
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) are only needed if you enable those
backends in `configs/backends.yaml`.

**4. Run the gateway**

```bash
uvicorn gateway.main:app --port 8080
```

**5. Run the dashboard** (optional, separate terminal)

```bash
streamlit run dashboard/app.py
```

Interactive API docs are at `http://localhost:8080/docs`, the dashboard at
`http://localhost:8501`.

> A `Makefile` provides `make run` / `make dashboard` / `make test` shortcuts on
> platforms where `make` is available.

**6. Send a request**

```bash
curl -X POST http://localhost:8080/v1/chat/completions -H "Authorization: Bearer $CLIENT_API_KEY_TIER1" -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

Omitting `model` lets the router choose. The response carries `X-Backend-Id`
and `X-Backend-Model` showing which backend actually served it.

---

## API

**Inference**

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible completion, streaming or not |
| `POST /v1/feedback` | Rate a completed request (`+1` / `-1`) |

**Operations**

| Endpoint | Purpose |
|---|---|
| `GET /health` · `GET /health/backends` | Liveness; per-backend health, circuit state, pricing |
| `GET /metrics` | Prometheus metrics |
| `GET /admin/requests` · `/admin/budgets` · `/admin/feedback` | Ledger, budgets, quality ratings |
| `PATCH /admin/budgets/{api_key}` | Create or update a client's limits |
| `POST /admin/circuit-breakers/{id}/reset` | Force a breaker closed |
| `POST /admin/reload-config` | Hot-reload YAML config without restarting |

Admin endpoints require the `X-Admin-Token` header.

---

## Configuration

| File | Controls |
|---|---|
| `configs/backends.yaml` | Backends: endpoint, pricing, `capability_tier`, `context_length`, tokenizer |
| `configs/routing_policy.yaml` | Active routing strategy and thresholds |
| `configs/budgets.yaml` | Per-client spend limits and rate limits |
| `configs/circuit_breaker.yaml` | Failure threshold, cooldown, request timeout |

`capability_tier` is declared separately from price on purpose. Routing that
wants "the better model" ranks on capability, not cost — the two usually
correlate, but when they diverge, ranking by price selects the *weaker* model.

---

## Testing

```bash
pytest
```

134 tests covering routing, budget enforcement, rate limiting, circuit breaking,
failover, PII masking, guardrails, token accounting, streaming cost
reconciliation, and context-window enforcement.

```bash
python evals/run_evals.py
```

Scores routing, screening, PII, and answer-adequacy policy against labelled
datasets in `evals/datasets/`. Deterministic, no model calls, runs in CI.
Reports pass rate and tracked known gaps; exits non-zero on regression.

Both run on every push via GitHub Actions.

---

## Project layout

```
gateway/
  main.py            FastAPI app, request pipeline
  adapters/          Per-provider clients, tokenization, parameter translation
  policy/            Router, budgets, circuit breaker, cache, guardrails, PII
  ledger/            SQLite spend store and async write queue
  telemetry/         Prometheus metrics, OpenTelemetry tracing
dashboard/           Streamlit operator UI
evals/               Policy evaluation harness and labelled datasets
```

---

## License

MIT — see [LICENSE](LICENSE).
