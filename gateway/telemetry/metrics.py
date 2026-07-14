from prometheus_client import Counter, Histogram, Gauge

# Request Counters
REQUESTS_TOTAL = Counter(
    "gateway_requests_total",
    "Total number of requests routed through the gateway",
    ["backend", "status"]
)

# Cost Counter
COST_TOTAL = Counter(
    "gateway_cost_usd_total",
    "Total cost in USD incurred by the gateway",
    ["backend"]
)

# Latency Histogram
LATENCY_MS = Histogram(
    "gateway_latency_ms",
    "Request latency in milliseconds",
    ["backend"]
)

# Circuit Breaker State Gauge (0=CLOSED, 1=HALF_OPEN, 2=OPEN)
CIRCUIT_BREAKER_STATE = Gauge(
    "gateway_circuit_breaker_state",
    "Circuit breaker state per backend (0=CLOSED, 1=HALF_OPEN, 2=OPEN)",
    ["backend"]
)

def observe_request(backend: str, status: str, latency: float, cost: float):
    REQUESTS_TOTAL.labels(backend=backend, status=status).inc()
    LATENCY_MS.labels(backend=backend).observe(latency)
    COST_TOTAL.labels(backend=backend).inc(cost)
