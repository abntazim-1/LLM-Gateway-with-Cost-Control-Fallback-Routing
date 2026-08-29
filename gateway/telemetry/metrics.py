from prometheus_client import Counter, Gauge, Histogram

# Request Counters
REQUESTS_TOTAL = Counter(
    "gateway_requests_total",
    "Total number of requests routed through the gateway",
    ["backend", "status"],
)

# Cost Counter
COST_TOTAL = Counter(
    "gateway_cost_usd_total", "Total cost in USD incurred by the gateway", ["backend"]
)

# Latency is recorded in milliseconds, but Prometheus' default buckets top out
# at 10 — they assume seconds. Every real LLM call (hundreds to tens of
# thousands of ms) therefore landed in +Inf, leaving the histogram unable to
# answer the only question anyone asks of it: what is p95?
#
# These boundaries span a fast cached reply through to a slow reasoning model
# against the 90s request deadline.
LATENCY_BUCKETS_MS = (
    50.0,
    100.0,
    250.0,
    500.0,
    1_000.0,
    2_500.0,
    5_000.0,
    10_000.0,
    20_000.0,
    30_000.0,
    60_000.0,
    90_000.0,
    float("inf"),
)

LATENCY_MS = Histogram(
    "gateway_latency_ms",
    "Request latency in milliseconds",
    ["backend"],
    buckets=LATENCY_BUCKETS_MS,
)

# Circuit Breaker State Gauge (0=CLOSED, 1=HALF_OPEN, 2=OPEN)
CIRCUIT_BREAKER_STATE = Gauge(
    "gateway_circuit_breaker_state",
    "Circuit breaker state per backend (0=CLOSED, 1=HALF_OPEN, 2=OPEN)",
    ["backend"],
)

# Cache Metrics
CACHE_HITS_TOTAL = Counter(
    "gateway_cache_hits_total", "Total number of prompt cache hits"
)

CACHE_MISSES_TOTAL = Counter(
    "gateway_cache_misses_total", "Total number of prompt cache misses"
)

# ── Routing health ───────────────────────────────────────────────────────
# Failover keeps requests succeeding, which is the point — but it means an
# outage shows up as a silently larger bill rather than as errors. Without a
# signal here, the only symptom of the cheap backend being down is the
# expensive one's spend going up, noticed at the end of the month.
FALLBACK_TOTAL = Counter(
    "gateway_fallback_total",
    "Requests served by a backend other than the router's first choice",
    ["backend"],
)

# A throttled backend is healthy and busy. It no longer trips the circuit
# breaker, which is correct — but that also makes it invisible, so count it:
# a rising rate means the cheap tier's limit is being reached and traffic is
# quietly moving to a pricier one.
THROTTLED_TOTAL = Counter(
    "gateway_throttled_total",
    "Rate-limit (429) responses received from a backend",
    ["backend"],
)

# Requests killed by the total time budget rather than by any backend error.
# Distinct from a backend failure: nothing was broken, the work just did not
# fit in the time allowed.
DEADLINE_EXCEEDED_TOTAL = Counter(
    "gateway_deadline_exceeded_total",
    "Requests abandoned because the total time budget ran out",
)


# Guardrails were the only policy with no counter, so there was no way to see
# either failure mode. Blocking is invisible by design — the caller gets an
# error and goes away — which makes over-blocking the more dangerous of the
# two: a pattern that starts catching ordinary traffic produces silence, not
# a bug report.
GUARDRAIL_BLOCKED_TOTAL = Counter(
    "gateway_guardrail_blocked_total",
    "Requests or responses stopped by a guardrail",
    ["stage"],
)

# Redaction is not a block — the response is still delivered, minus the
# secret. Counted separately so it is not read as traffic being refused.
GUARDRAIL_REDACTED_TOTAL = Counter(
    "gateway_guardrail_redacted_total",
    "Responses that had a secret redacted before delivery",
)


def observe_guardrail_block(stage: str):
    """A guardrail stopped something. `stage` is where: input_injection,
    output_leak."""
    GUARDRAIL_BLOCKED_TOTAL.labels(stage=stage).inc()


def observe_guardrail_redaction():
    """A secret was stripped from a response that was still delivered."""
    GUARDRAIL_REDACTED_TOTAL.inc()


def observe_request(backend: str, status: str, latency: float, cost: float):
    REQUESTS_TOTAL.labels(backend=backend, status=status).inc()
    LATENCY_MS.labels(backend=backend).observe(latency)
    COST_TOTAL.labels(backend=backend).inc(cost)


def observe_cache(hit: bool):
    if hit:
        CACHE_HITS_TOTAL.inc()
    else:
        CACHE_MISSES_TOTAL.inc()


def observe_fallback(backend: str):
    """A non-first-choice backend served the request."""
    FALLBACK_TOTAL.labels(backend=backend).inc()


def observe_throttled(backend: str):
    """A backend replied 429."""
    THROTTLED_TOTAL.labels(backend=backend).inc()


def observe_deadline_exceeded():
    """The request's total time budget ran out."""
    DEADLINE_EXCEEDED_TOTAL.inc()
