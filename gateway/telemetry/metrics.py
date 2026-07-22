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

# Cache Metrics
CACHE_HITS_TOTAL = Counter(
    "gateway_cache_hits_total",
    "Total number of prompt cache hits"
)

CACHE_MISSES_TOTAL = Counter(
    "gateway_cache_misses_total",
    "Total number of prompt cache misses"
)

def observe_request(backend: str, status: str, latency: float, cost: float):
    REQUESTS_TOTAL.labels(backend=backend, status=status).inc()
    LATENCY_MS.labels(backend=backend).observe(latency)
    COST_TOTAL.labels(backend=backend).inc(cost)

def observe_cache(hit: bool):
    if hit:
        CACHE_HITS_TOTAL.inc()
    else:
        CACHE_MISSES_TOTAL.inc()

