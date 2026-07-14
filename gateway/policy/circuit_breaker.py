import time
from enum import Enum
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "CLOSED"       # Healthy, requests flow normally
    OPEN = "OPEN"           # Unhealthy, requests fail immediately
    HALF_OPEN = "HALF_OPEN" # Testing recovery, 1 request allowed through

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, cooldown_sec: int = 30):
        self.failure_threshold = failure_threshold
        self.cooldown_sec = cooldown_sec
        
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.last_failure_time = 0.0

    def record_success(self):
        """Called when a request succeeds."""
        if self.state in [CircuitState.OPEN, CircuitState.HALF_OPEN]:
            logger.info("Circuit breaker closed (healthy).")
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.last_failure_time = 0.0

    def record_failure(self):
        """Called when a request fails."""
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN or self.consecutive_failures >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                logger.warning(f"Circuit breaker opened! Failures: {self.consecutive_failures}")
            self.state = CircuitState.OPEN

    def can_request(self) -> bool:
        """Determines if a request should be allowed through the circuit breaker."""
        if self.state == CircuitState.CLOSED:
            return True
            
        if self.state == CircuitState.OPEN:
            # Check if cooldown has elapsed
            if time.time() - self.last_failure_time >= self.cooldown_sec:
                logger.info("Circuit breaker half-open. Testing recovery.")
                self.state = CircuitState.HALF_OPEN
                return True
            return False
            
        if self.state == CircuitState.HALF_OPEN:
            # Already let one through to test, deny others until that one resolves
            return False
            
        return False

class CircuitBreakerRegistry:
    def __init__(self, failure_threshold: int = 3, cooldown_sec: int = 30):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self.failure_threshold = failure_threshold
        self.cooldown_sec = cooldown_sec

    def get_breaker(self, backend_id: str) -> CircuitBreaker:
        if backend_id not in self._breakers:
            self._breakers[backend_id] = CircuitBreaker(
                failure_threshold=self.failure_threshold,
                cooldown_sec=self.cooldown_sec
            )
        return self._breakers[backend_id]
