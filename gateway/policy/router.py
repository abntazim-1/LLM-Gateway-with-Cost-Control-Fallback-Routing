from typing import List, Dict, Any, Optional
import asyncio
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
        self.latency_ema: Dict[str, float] = {}

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
            # Rank by EMA latency, defaults to a high value if unknown
            candidates.sort(key=lambda a: self.latency_ema.get(a.id, 9999.0))
            
        return candidates

    async def execute(self, messages: List[Dict[str, str]], **kwargs):
        ranked = self.get_ranked_adapters(**kwargs)
        
        last_error = None
        for adapter in ranked:
            breaker = self.circuit_registry.get_breaker(adapter.id)
            
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    logger.info(f"Routing request to {adapter.id} (attempt {attempt + 1}/{max_retries + 1})")
                    response = await adapter.complete(messages, **kwargs)
                    
                    # Update Exponential Moving Average (EMA) for latency
                    alpha = 0.2
                    current_ema = self.latency_ema.get(adapter.id, response.latency_ms)
                    self.latency_ema[adapter.id] = (alpha * response.latency_ms) + ((1 - alpha) * current_ema)
                    
                    breaker.record_success()
                    return response
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        backoff = 2 ** attempt
                        logger.warning(f"Transient failure on {adapter.id} ({str(e)}). Retrying in {backoff}s...")
                        await asyncio.sleep(backoff)
                    else:
                        logger.warning(f"Backend {adapter.id} failed after {max_retries + 1} attempts: {str(e)}")
                        breaker.record_failure()
                        break # Move to next adapter

        raise NoAvailableBackendException(f"All capable backends failed. Last error: {str(last_error)}")
