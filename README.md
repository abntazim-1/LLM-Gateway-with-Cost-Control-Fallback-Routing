# Enterprise LLM Gateway with Cost Control & Fallback Routing

A production-ready AI Gateway that sits between your applications and LLM providers. It provides a single OpenAI-compatible API endpoint that intelligently routes traffic across multiple backends (OpenAI, Anthropic, local vLLM) based on cost and capability, enforces hard budgets per API key, and automatically fails over when a provider goes down.

## Features

- **Multi-Provider Routing:** Uses a `cost_first` strategy to dynamically route requests to the cheapest capable model, or fallback to alternatives.
- **Pre-flight Budget Enforcement:** Blocks requests *before* they are sent to a paid API if the user's daily or monthly budget limit is exceeded.
- **Circuit Breaker Failover:** Implements a closed -> open -> half-open state machine per backend. If OpenAI goes down, traffic seamlessly shifts to Anthropic or a self-hosted local model with zero dropped client requests.
- **Cost Ledger & Dashboard:** Tracks spend per API key and model using SQLite/Postgres. Includes a Streamlit dashboard for real-time cost visibility.
- **OpenAI-Compatible API:** Your applications don't need to change. Just swap the base URL and API key to point to the Gateway.

## Architecture

![Architecture](docs/ARCHITECTURE.md) (See ARCHITECTURE.md for full details)

The gateway is built with FastAPI and uses a decoupled Adapter pattern. Adding a new provider requires implementing a single `BaseAdapter` interface, without touching the core routing or budget logic.

## Quickstart

1. Install dependencies:
   ```bash
   pip install -r requirements.txt # or use standard python packaging if using hatch
   pip install -e .
   ```

2. Set your API keys:
   ```bash
   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-..."
   ```

3. Run the Gateway:
   ```bash
   make run
   ```

4. Run the Dashboard:
   ```bash
   make dashboard
   ```

5. Test the endpoint:
   ```bash
   curl -X POST http://localhost:8080/v1/chat/completions \
     -H "Authorization: Bearer sk-test-tier-1" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "gpt-4o-mini",
       "messages": [{"role": "user", "content": "Hello!"}]
     }'
   ```

## Demo

See `docs/DEMO_SCRIPT.md` for a guided walk-through of the live circuit-breaker failover.
