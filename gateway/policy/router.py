from typing import List, Dict, Any, Optional
from gateway.adapters.base import BaseAdapter
from gateway.policy.circuit_breaker import CircuitBreakerRegistry
import logging

logger = logging.getLogger(__name__)

class NoAvailableBackendException(Exception):
    pass

class Router:
    def __init__(
        self, 
        adapters: List[BaseAdapter], 
        circuit_registry: CircuitBreakerRegistry,
        strategy: str = "cost_first"
    ):
        self.adapters = adapters
        self.circuit_registry = circuit_registry
        self.strategy = strategy

    def get_ranked_adapters(self, **kwargs) -> List[BaseAdapter]:
        """Rank adapters based on the routing strategy and capabilities."""
        # Filter out adapters that don't meet capability requirements (simplified here)
        required_caps = set(kwargs.get("capabilities_required", []))
        
        candidates = []
        for adapter in self.adapters:
            # Check circuit breaker
            breaker = self.circuit_registry.get_breaker(adapter.id)
            if not breaker.can_request():
                continue
                
            candidates.append(adapter)

        if not candidates:
            raise NoAvailableBackendException("No healthy backends available to serve the request.")

        # Rank candidates
        if self.strategy == "cost_first":
            # For cost first, prioritize lowest cost per 1k prompt
            candidates.sort(key=lambda a: a.cost_per_1k_prompt)
        elif self.strategy == "latency_first":
            # Real implementation would use historical latency
            pass
            
        return candidates

    async def execute(self, messages: List[Dict[str, str]], **kwargs):
        ranked = self.get_ranked_adapters(**kwargs)
        
        last_error = None
        for adapter in ranked:
            breaker = self.circuit_registry.get_breaker(adapter.id)
            try:
                logger.info(f"Routing request to {adapter.id}")
                response = await adapter.complete(messages, **kwargs)
                breaker.record_success()
                return response
            except Exception as e:
                logger.warning(f"Backend {adapter.id} failed: {str(e)}")
                breaker.record_failure()
                last_error = e

        raise NoAvailableBackendException(f"All capable backends failed. Last error: {str(last_error)}")
