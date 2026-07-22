from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio
import time
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
        self._rr_counter = 0

    async def get_ranked_adapters(self, messages: Optional[List[Dict[str, str]]] = None, **kwargs) -> List[BaseAdapter]:
        """Rank adapters based on the routing strategy and capabilities."""
        candidates = []
        for adapter in self.adapters:
            # Check circuit breaker
            breaker = self.circuit_registry.get_breaker(adapter.id)
            if not await breaker.can_request():
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
        elif self.strategy == "round_robin" or self.strategy == "weighted_round_robin":
            if len(candidates) > 1:
                idx = self._rr_counter % len(candidates)
                self._rr_counter += 1
                candidates = candidates[idx:] + candidates[:idx]
        elif self.strategy == "complexity":
            is_complex = False
            if messages:
                full_prompt = " ".join([m.get("content", "") for m in messages]).lower()
                reasoning_keywords = {"solve", "code", "optimize", "analyze", "debug", "proof", "explain", "why", "how to", "write a", "implement"}
                has_keywords = any(kw in full_prompt for kw in reasoning_keywords)
                if len(full_prompt) > 120 or has_keywords:
                    is_complex = True
            
            if is_complex:
                # Rank premium models first (higher cost first)
                candidates.sort(key=lambda a: a.cost_per_1k_prompt, reverse=True)
            else:
                # Rank budget models first (cheaper cost first)
                candidates.sort(key=lambda a: a.cost_per_1k_prompt)
            
        return candidates

    async def execute(self, messages: List[Dict[str, str]], **kwargs):
        ranked = await self.get_ranked_adapters(messages=messages, **kwargs)
        
        last_error = None
        for adapter in ranked:
            breaker = self.circuit_registry.get_breaker(adapter.id)
            # Map request kwargs model to the adapter's native model during failover
            call_kwargs = dict(kwargs)
            call_kwargs["model"] = adapter.model
            
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    logger.info(f"Routing request to {adapter.id} using model {adapter.model} (attempt {attempt + 1}/{max_retries + 1})")
                    response = await adapter.complete(messages, **call_kwargs)
                    
                    # Update Exponential Moving Average (EMA) for latency
                    alpha = 0.2
                    current_ema = self.latency_ema.get(adapter.id, response.latency_ms)
                    self.latency_ema[adapter.id] = (alpha * response.latency_ms) + ((1 - alpha) * current_ema)
                    
                    await breaker.record_success()
                    return response
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        backoff = 2 ** attempt
                        logger.warning(f"Transient failure on {adapter.id} ({str(e)}). Retrying in {backoff}s...")
                        await asyncio.sleep(backoff)
                    else:
                        logger.warning(f"Backend {adapter.id} failed after {max_retries + 1} attempts: {str(e)}")
                        await breaker.record_failure()
                        break # Move to next adapter

        raise NoAvailableBackendException(f"All capable backends failed. Last error: {str(last_error)}")

    async def execute_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        ranked = await self.get_ranked_adapters(messages=messages, **kwargs)
        
        last_error = None
        target_adapter = None
        stream_generator = None
        
        for adapter in ranked:
            breaker = self.circuit_registry.get_breaker(adapter.id)
            call_kwargs = dict(kwargs)
            call_kwargs["model"] = adapter.model
            
            try:
                logger.info(f"Opening stream routing to {adapter.id} using model {adapter.model}")
                gen = adapter.complete_stream(messages, **call_kwargs)
                first_chunk = await gen.__anext__()
                
                # Stream successfully initialized
                target_adapter = adapter
                stream_generator = gen
                yield first_chunk
                break
            except (StopAsyncIteration, Exception) as e:
                last_error = e
                logger.warning(f"Backend stream startup on {adapter.id} failed: {str(e)}")
                await breaker.record_failure()
                continue
                
        if not target_adapter or not stream_generator:
            raise NoAvailableBackendException(f"All backends failed to initialize stream. Last error: {str(last_error)}")
            
        start_time = time.time()
        accumulated_text = ""
        last_chunk_id = "stream-chunk"
        last_model = target_adapter.model
        
        try:
            async for chunk in stream_generator:
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        accumulated_text += content
                if chunk.get("id"):
                    last_chunk_id = chunk["id"]
                yield chunk
                
            breaker = self.circuit_registry.get_breaker(target_adapter.id)
            await breaker.record_success()
        except Exception as e:
            logger.error(f"Stream interrupted on {target_adapter.id} after initial chunks: {str(e)}")
            raise e
        finally:
            latency_ms = (time.time() - start_time) * 1000.0
            
            try:
                import tiktoken
                encoding = tiktoken.get_encoding("cl100k_base")
                prompt_tokens = sum(len(encoding.encode(m.get("content", ""))) for m in messages)
                completion_tokens = len(encoding.encode(accumulated_text))
            except Exception:
                approx_p = sum(len(m.get("content", "")) for m in messages) / 4.0
                prompt_tokens = max(1, int(approx_p))
                completion_tokens = max(1, int(len(accumulated_text) / 4.0))
                
            cost_usd = (prompt_tokens / 1000.0) * target_adapter.cost_per_1k_prompt + \
                       (completion_tokens / 1000.0) * target_adapter.cost_per_1k_completion
                       
            if self.circuit_registry.ledger:
                await self.circuit_registry.ledger.record_request(
                    api_key=kwargs.get("api_key", "sk-unknown"),
                    req_id=last_chunk_id,
                    backend=target_adapter.id,
                    model=last_model,
                    prompt_tokens=prompt_tokens,
                    comp_tokens=completion_tokens,
                    cost=cost_usd,
                    latency=latency_ms
                )
