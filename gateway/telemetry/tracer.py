import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    HAS_OPENTELEMETRY = True
    tracer = trace.get_tracer("gateway.tracer", "0.1.0")
except ImportError:
    HAS_OPENTELEMETRY = False
    tracer = None

class GatewayTracer:
    """OpenTelemetry instrumentation wrapper with graceful fallback if OpenTelemetry packages are omitted."""

    @staticmethod
    @contextmanager
    def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None):
        if not HAS_OPENTELEMETRY or tracer is None:
            yield None
            return

        with tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    if v is not None:
                        span.set_attribute(k, str(v) if not isinstance(v, (int, float, bool, str)) else v)
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
