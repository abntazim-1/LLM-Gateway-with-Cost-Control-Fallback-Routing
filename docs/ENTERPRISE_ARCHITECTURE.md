# Enterprise LLM Gateway Production Architecture & Operational Manual

This document provides a comprehensive technical blueprint of the production architecture, policy guardrails, distributed telemetry, and key management features implemented in the **Enterprise LLM Gateway**.

---

## Architecture Overview

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
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │ Async Ledger Queue Worker │
                          └─────────────┬─────────────┘
                                        │ Non-blocking Batch Persist
                                        ▼
                          ┌───────────────────────────┐
                          │ BaseLedgerStore (SQLite)  │
                          └───────────────────────────┘
```

---

## Core Enterprise Modules

### 1. Reversible PII Vault & Secret Leak Detection (`gateway/policy/pii.py`)
- **Indexed Token Vaulting:** Replaces sensitive user data (Emails, Phone numbers, SSNs, Credit Cards) with indexed tokens (`[EMAIL_1]`, `[PHONE_1]`) and supports `restore_text()` to restore response contents.
- **Secret Leak Protection:** Rejects or redacts AWS secret keys (`AKIA...`), JWT tokens, and Bearer tokens to prevent credentials from leaking to third-party LLMs.

### 2. Guardrails Pipeline Engine (`gateway/policy/guardrails.py`)
- **Prompt Injection Screening:** Inspects incoming prompt messages for jailbreak attempts (`ignore previous instructions`, `system prompt override`, `act as DAN`). Violations immediately return HTTP 400 Bad Request.

### 3. Virtual Provider Key Rotation Pools (`gateway/policy/key_pool.py`)
- **Multi-Key Round-Robin:** Rotates through arrays of provider master keys per request to bypass single-account RPM/TPM throughput limits.
- **Rate-Limit Cooldowns:** Temporarily marks rate-limited keys in cooldown to maintain high availability.

### 4. Asynchronous Batch Queue Logging (`gateway/ledger/async_queue.py`)
- **Non-blocking Telemetry:** Offloads database IO away from HTTP inference loops using an `asyncio.Queue` worker task, optimizing P99 response latencies.

### 5. Cross-Provider Parameter Translation (`gateway/adapters/transformer.py`)
- Translates OpenAI payload parameters to Anthropic Messages API format (`system` message extraction, strictly alternating user/assistant roles, `max_tokens` mapping, `stop_sequences`).

### 6. Dynamic Configuration Hot-Reloading (`gateway/config_manager.py`)
- Trigger `/admin/reload-config` to reload backend configurations, budgets, and routing strategies live without server downtime.

---

## Deployment & Monitoring

- **Prometheus Metrics:** Exported at `/metrics` (includes `gateway_requests_total`, `gateway_latency_ms`, `gateway_cost_usd_total`, `gateway_cache_hits_total`, `gateway_cache_misses_total`).
- **OpenTelemetry Tracing:** Managed via `GatewayTracer` with graceful fallback when OpenTelemetry packages are omitted.
