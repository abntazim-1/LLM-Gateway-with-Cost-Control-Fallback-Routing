# Enterprise LLM Gateway with Cost Control, Guardrails & Fallback Routing

An industry-grade, production-ready AI Gateway that sits between your applications and LLM providers. It provides a single OpenAI-compatible API endpoint (`/v1/chat/completions`) that intelligently routes traffic across multiple backends (OpenAI, Anthropic, local vLLM), enforces hard pre-flight budgets, redacts PII and sensitive credentials, screens for prompt injections, and automatically fails over with zero dropped client requests.

---

## Key Enterprise Features

- ⚡ **OpenAI-Compatible API:** Drop-in replacement for OpenAI SDKs—just change the `base_url` and API key.
- 🔀 **Multi-Provider Routing & Automatic Translation:** Dynamically routes requests based on `cost_first`, `latency_first`, `complexity`, or `weighted_round_robin`. Automatically translates parameters across backends (OpenAI <-> Anthropic Messages API).
- 🛡️ **Reversible PII Vault & Secret Leak Protection:** Masks emails, credit cards, SSNs, and phone numbers with indexed tokens (`[EMAIL_1]`) and restores responses. Detects and blocks AWS access keys, JWTs, and API credentials from leaking to upstream LLMs.
- 🚫 **Guardrails Pipeline Engine:** Screens input prompts for prompt injection, system override attempts, and jailbreaks (`ignore previous instructions`, `act as DAN`). Violations immediately return HTTP 400 Bad Request.
- 🔑 **Virtual Provider Key Pools & Rotation:** Rotates through pools of master provider API keys per request to bypass single-account RPM/TPM rate limits, with automatic cooldown tracking for rate-limited keys.
- ⚡ **Asynchronous Ledger Queue:** Non-blocking background telemetry worker offloads spend logging away from HTTP inference loops using an `asyncio.Queue`, ensuring ultra-low P99 response latencies.
- 💾 **Bounded Thread-Safe LRU Cache:** TTL-based, capacity-bounded LRU prompt cache with Prometheus hit/miss ratio tracking.
- 🔄 **Circuit Breaker Failover:** Closed -> Open -> Half-Open state machine per backend. If OpenAI goes down, traffic seamlessly shifts to Anthropic or self-hosted vLLM.
- 📊 **Prometheus & OpenTelemetry Tracing:** Native `/metrics` endpoint and OpenTelemetry distributed span tracing.
- ⚙️ **Dynamic Config Hot-Reloading:** Reload YAML configs live via `/admin/reload-config` without restarting the server.

---

## Architecture Overview

See [docs/ENTERPRISE_ARCHITECTURE.md](docs/ENTERPRISE_ARCHITECTURE.md) for full architectural blueprints, component diagrams, and deployment guides.

```
                          ┌───────────────────────────┐
                          │   Application / Clients   │
                          └─────────────┬─────────────┘
                                        │ HTTP / REST (/v1/chat/completions)
                                        ▼
                          ┌───────────────────────────┐
                          │    FastAPI Data Plane     │
                          └─────────────┬─────────────┘
                                        │
     ┌──────────────────┬───────────────┼───────────────┬──────────────────┐
     ▼                  ▼               ▼               ▼                  ▼
┌──────────────┐ ┌──────────────┐ ┌───────────┐ ┌───────────────┐ ┌─────────────────┐
│ Guardrails   │ │ Reversible   │ │ Bounded   │ │ Multi-Key     │ │ OpenTelemetry   │
│ Pipeline     │ │ PII Vault    │ │ LRU Cache │ │ Key Pool      │ │ Distributed     │
│ (Injection)  │ │ (Anonymizer) │ │ (300s TTL)│ │ (Round-Robin) │ │ Tracer          │
└──────────────┘ └──────────────┘ └───────────┘ └───────────────┘ └─────────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │ Dynamic Model Router      │
                          │ (Cost / Latency / RR)     │
                          └─────────────┬─────────────┘
                                        │ Fallback & Retry
     ┌──────────────────────────────────┼──────────────────────────────────┐
     ▼                                  ▼                                  ▼
┌──────────────────┐          ┌──────────────────┐               ┌──────────────────┐
│ OpenAI Adapter   │          │ Anthropic Adapter│               │ vLLM Adapter     │
│ (gpt-4o-mini)    │          │ (claude-3-5)     │               │ (local-llama3)   │
└──────────────────┘          └──────────────────┘               └──────────────────┘
```

---

## Quickstart

### 1. Install Dependencies
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 2. Configure Environment & API Keys
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-..."
export ADMIN_API_KEY="admin-secret-key"
```

### 3. Run the Gateway
```bash
make run
# Or directly via uvicorn:
# uvicorn gateway.main:app --host 0.0.0.0 --port 8080 --workers 4
```

### 4. Run the Streamlit Admin & Analytics Portal
```bash
make dashboard
```

### 5. Send Test Inference Request
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-test-tier-1" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Running the Automated Test Suite

Run the full 33-test suite covering parameter translation, async queues, PII vaulting, guardrails, key pools, circuit breakers, and failover integration:

```bash
pytest -v
```

---

## License

MIT License. See `LICENSE` for details.
