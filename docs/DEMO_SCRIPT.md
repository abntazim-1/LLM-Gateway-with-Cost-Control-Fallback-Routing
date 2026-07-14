# Enterprise LLM Gateway Failover Demo

This script outlines how to demonstrate the fallback routing and circuit breaker live in an interview.

## Setup
1. Ensure the gateway is running: `make run`
2. Open the dashboard in another terminal: `make dashboard` (http://localhost:8501)
3. Simulate load using Locust: `make loadtest` (http://localhost:8089)

## The Demo

1. **Start normal load**
   - In the Locust UI, start a load test with 5 users and 1 user/sec spawn rate.
   - Show the terminal where `uvicorn` is running. You will see traffic routing to `openai-gpt-4o-mini` (or whichever is cheapest based on your `cost_first` strategy).
   - Show the Streamlit dashboard updating the spend and request counts.

2. **Trigger the Circuit Breaker (Failover)**
   - In a new terminal, simulate the primary backend failing. (If using a mock backend, kill the mock process. If using OpenAI, temporarily corrupt the `OPENAI_API_KEY` in the environment or block the domain in your hosts file).
   - Watch the gateway terminal.
   - You should see logs like:
     ```
     Backend openai-gpt-4o-mini failed: OpenAI request failed...
     Circuit breaker opened! Failures: 3
     Routing request to anthropic-claude-3-haiku
     ```
   
3. **Observe Zero Downtime**
   - Switch back to the Locust UI.
   - The failure rate should be 0% (or very low). The requests didn't fail to the client; the Gateway caught the 500, recorded a circuit breaker failure, and immediately re-routed to the next cheapest provider (Anthropic).

4. **Show Dashboard Impact**
   - Refresh the Streamlit dashboard.
   - You will see the spend graph shift from OpenAI to Anthropic immediately.

5. **Recovery (Half-Open state)**
   - Restore the primary backend (fix the API key or unblock the domain).
   - After the cooldown period (default 30s), the circuit breaker enters `HALF_OPEN`.
   - You will see one request routed back to the primary backend.
   - Upon success: `Circuit breaker closed (healthy).`
   - All subsequent traffic reroutes back to the primary, cheaper backend.
