import time
from enum import Enum
from typing import Dict, Optional
import logging
from gateway.telemetry.metrics import CIRCUIT_BREAKER_STATE

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "CLOSED"       # Healthy, requests flow normally
    OPEN = "OPEN"           # Unhealthy, requests fail immediately
    HALF_OPEN = "HALF_OPEN" # Testing recovery, 1 request allowed through

class CircuitBreaker:
    def __init__(self, backend_id: str, ledger, failure_threshold: int = 3, cooldown_sec: int = 30):
        self.backend_id = backend_id
        self.ledger = ledger
        self.failure_threshold = failure_threshold
        self.cooldown_sec = cooldown_sec

    def _get_state(self):
        if self.ledger:
            data = self.ledger.get_circuit_breaker_state(self.backend_id)
            if data:
                return CircuitState(data["state"]), data["consecutive_failures"], data["last_failure_time"]
        return CircuitState.CLOSED, 0, 0.0

    def _save_state(self, state: CircuitState, consecutive_failures: int, last_failure_time: float):
        if self.ledger:
            self.ledger.update_circuit_breaker_state(
                self.backend_id, state.value, consecutive_failures, last_failure_time
            )
        val_map = {CircuitState.CLOSED: 0, CircuitState.HALF_OPEN: 1, CircuitState.OPEN: 2}
        CIRCUIT_BREAKER_STATE.labels(backend=self.backend_id).set(val_map[state])

    @property
    def state(self) -> CircuitState:
        return self._get_state()[0]

    def record_success(self):
        """Called when a request succeeds."""
        state, _, _ = self._get_state()
        if state in [CircuitState.OPEN, CircuitState.HALF_OPEN]:
            logger.info("Circuit breaker closed (healthy).")
        self._save_state(CircuitState.CLOSED, 0, 0.0)

    def record_failure(self):
        """Called when a request fails."""
        state, consecutive_failures, _ = self._get_state()
        consecutive_failures += 1
        last_failure_time = time.time()
        
        if state == CircuitState.HALF_OPEN or consecutive_failures >= self.failure_threshold:
            if state != CircuitState.OPEN:
                logger.warning(f"Circuit breaker opened! Failures: {consecutive_failures}")
            state = CircuitState.OPEN
            
        self._save_state(state, consecutive_failures, last_failure_time)

    def can_request(self) -> bool:
        """Determines if a request should be allowed through the circuit breaker."""
        state, _, last_failure_time = self._get_state()
        
        if state == CircuitState.CLOSED:
            return True
            
        if state == CircuitState.OPEN:
            # Check if cooldown has elapsed
            if time.time() - last_failure_time >= self.cooldown_sec:
                logger.info("Circuit breaker half-open. Testing recovery.")
                self._save_state(CircuitState.HALF_OPEN, _, last_failure_time)
                return True
            return False
            
        if state == CircuitState.HALF_OPEN:
            # Already let one through to test, deny others until that one resolves
            return False
            
        return False

class CircuitBreakerRegistry:
    def __init__(self, ledger, failure_threshold: int = 3, cooldown_sec: int = 30):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self.ledger = ledger
        self.failure_threshold = failure_threshold
        self.cooldown_sec = cooldown_sec

    def get_breaker(self, backend_id: str) -> CircuitBreaker:
        if backend_id not in self._breakers:
            self._breakers[backend_id] = CircuitBreaker(
                backend_id=backend_id,
                ledger=self.ledger,
                failure_threshold=self.failure_threshold,
                cooldown_sec=self.cooldown_sec
            )
        return self._breakers[backend_id]
