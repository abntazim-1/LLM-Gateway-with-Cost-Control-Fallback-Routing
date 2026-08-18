from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio
import time
from gateway.adapters.base import BaseAdapter
from gateway.policy.circuit_breaker import CircuitBreakerRegistry
import logging

import re

logger = logging.getLogger(__name__)

# Word-boundary patterns for complex reasoning and engineering domains
COMPLEX_PATTERNS = [
    re.compile(r'\b(optimize|optimization|benchmark|benchmarking|refactor|refactoring)\b', re.IGNORECASE),
    re.compile(r'\b(proof|prove|theorem|derivation|differential|calculus|eigenvalue)\b', re.IGNORECASE),
    re.compile(r'\b(debug|stacktrace|traceback|segmentation fault|memory leak|deadlock|race condition)\b', re.IGNORECASE),
    re.compile(r'\b(algorithm|data structure|dynamic programming|complexity analysis)\b', re.IGNORECASE),
    re.compile(r'```[a-zA-Z0-9_-]*\n[\s\S]+?```'),  # Embedded multi-line code block
]

SIMPLE_PREFIXES = re.compile(
    r'^(why is|why are|why do|why does|what is|what are|how are|who is|who are|tell me about|explain simply)\b',
    re.IGNORECASE
)

class NoAvailableBackendException(Exception):
    pass

class Router:
    def __init__(
        self,
        adapters: List[BaseAdapter],
        circuit_registry: CircuitBreakerRegistry,
        strategy: str = "cost_first",
        timeout_sec: float = 30.0,
        min_tokens_for_complex: int = 500,
    ):
        self.adapters = adapters
        self.circuit_registry = circuit_registry
        self.strategy = strategy
        self.timeout_sec = timeout_sec
        self.min_tokens_for_complex = min_tokens_for_complex
        self.latency_ema: Dict[str, float] = {}
        self._rr_counter = 0

    async def close(self) -> None:
        """Close all managed backend adapters to release connection pools and sockets."""
        for adapter in self.adapters:
            try:
                await adapter.close()
            except Exception as e:
                logger.error(f"Error closing adapter {getattr(adapter, 'id', 'unknown')}: {e}")

    def _is_complex_request(self, messages: Optional[List[Dict[str, str]]]) -> bool:
        """Evaluate request complexity using token count thresholds and structured reasoning signals."""
        if not messages:
            return False
        
        contents = [m.get("content", "") for m in messages if isinstance(m.get("content"), str)]
        full_prompt = " ".join(contents).strip()
        if not full_prompt:
            return False

        approx_tokens = len(full_prompt) / 4.0
        
        # High token volume (long prompt / document / code review)
        if approx_tokens >= self.min_tokens_for_complex:
            return True

        # Disqualify simple conversational queries without code
        first_user_msg = next((m.get("content", "").strip() for m in messages if m.get("role") == "user"), "")
        if approx_tokens < 60 and SIMPLE_PREFIXES.search(first_user_msg) and not re.search(r'```', full_prompt):
            return False

        # Embedded multi-line code block indicates programming task
        if re.search(r'```[a-zA-Z0-9_-]*\n[\s\S]+?```', full_prompt):
            return True

        # Count reasoning keyword matches with word boundaries
        matches = sum(1 for pattern in COMPLEX_PATTERNS[:-1] if pattern.search(full_prompt))
        return matches >= 2 or (approx_tokens > 150 and matches >= 1)

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

        # Capability-aware filtering. Callers may pass an explicit
        # `required_capabilities` list; json_mode is also inferred from
        # response_format / json_mode kwargs. Falls back to all candidates
        # when no backend advertises every required capability.
        required = {str(c) for c in (kwargs.get("required_capabilities") or [])}
        response_format = kwargs.get("response_format") or {}
        if (
            (isinstance(response_format, dict) and response_format.get("type") == "json_object")
            or kwargs.get("json_mode", False)
        ):
            required.add("json_mode")
        if required:
            capable_candidates = [
                a for a in candidates
                if required.issubset(set(a.config.get("capabilities") or []))
            ]
            if capable_candidates:
                candidates = capable_candidates

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
            is_complex = self._is_complex_request(messages)
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
                    logger.info(
                        f"Routing request to {adapter.id} using model {adapter.model} "
                        f"(attempt {attempt + 1}/{max_retries + 1}, timeout {self.timeout_sec}s)"
                    )
                    response = await asyncio.wait_for(
                        adapter.complete(messages, **call_kwargs),
                        timeout=self.timeout_sec,
                    )

                    # Update Exponential Moving Average (EMA) for latency
                    alpha = 0.2
                    current_ema = self.latency_ema.get(adapter.id, response.latency_ms)
                    self.latency_ema[adapter.id] = (alpha * response.latency_ms) + ((1 - alpha) * current_ema)

                    await breaker.record_success()
                    return response
                except asyncio.TimeoutError:
                    last_error = asyncio.TimeoutError(
                        f"Backend {adapter.id} timed out after {self.timeout_sec}s"
                    )
                    logger.warning(str(last_error))
                    if attempt < max_retries:
                        backoff = 2 ** attempt
                        logger.warning(f"Retrying {adapter.id} in {backoff}s...")
                        await asyncio.sleep(backoff)
                    else:
                        await breaker.record_failure()
                        break  # move to next adapter
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        backoff = 2 ** attempt
                        logger.warning(f"Transient failure on {adapter.id} ({str(e)}). Retrying in {backoff}s...")
                        await asyncio.sleep(backoff)
                    else:
                        logger.warning(f"Backend {adapter.id} failed after {max_retries + 1} attempts: {str(e)}")
                        await breaker.record_failure()
                        break  # move to next adapter

        raise NoAvailableBackendException(f"All capable backends failed. Last error: {str(last_error)}")

    async def execute_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        ranked = await self.get_ranked_adapters(messages=messages, **kwargs)

        last_error = None
        target_adapter = None
        stream_generator = None
        stream_start = time.time()

        for adapter in ranked:
            breaker = self.circuit_registry.get_breaker(adapter.id)
            call_kwargs = dict(kwargs)
            call_kwargs["model"] = adapter.model

            try:
                logger.info(
                    f"Opening stream routing to {adapter.id} using model {adapter.model} "
                    f"(timeout {self.timeout_sec}s)"
                )
                gen = adapter.complete_stream(messages, **call_kwargs)
                # Enforce timeout on the first chunk so a hung backend doesn't
                # block indefinitely before we can fail over.
                first_chunk = await asyncio.wait_for(
                    gen.__anext__(),
                    timeout=self.timeout_sec,
                )

                # Stream successfully initialized
                target_adapter = adapter
                stream_generator = gen
                yield first_chunk
                break
            except (asyncio.TimeoutError, StopAsyncIteration, Exception) as e:
                last_error = e
                logger.warning(f"Backend stream startup on {adapter.id} failed: {str(e)}")
                await breaker.record_failure()
                continue

        if not target_adapter or not stream_generator:
            raise NoAvailableBackendException(f"All backends failed to initialize stream. Last error: {str(last_error)}")

        accumulated_text = ""
        last_chunk_id = "stream-chunk"
        last_model = target_adapter.model

        try:
            async for chunk in stream_generator:
                # Enforce the remaining deadline on each chunk read.
                elapsed = time.time() - stream_start
                remaining = self.timeout_sec - elapsed
                if remaining <= 0:
                    raise asyncio.TimeoutError(
                        f"Stream timed out after {self.timeout_sec}s on {target_adapter.id}"
                    )

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
        except asyncio.TimeoutError as e:
            logger.error(f"Stream timed out on {target_adapter.id}: {str(e)}")
            await self.circuit_registry.get_breaker(target_adapter.id).record_failure()
            raise
        except Exception as e:
            logger.error(f"Stream interrupted on {target_adapter.id} after initial chunks: {str(e)}")
            raise
        finally:
            latency_ms = (time.time() - stream_start) * 1000.0

            try:
                import tiktoken
                encoding = tiktoken.get_encoding("cl100k_base")
                prompt_tokens = sum(len(encoding.encode(m.get("content", ""))) for m in messages)
                completion_tokens = len(encoding.encode(accumulated_text))
            except Exception:
                approx_p = sum(len(m.get("content", "")) for m in messages) / 4.0
                prompt_tokens = max(1, int(approx_p))
                completion_tokens = max(1, int(len(accumulated_text) / 4.0))

            cost_usd = (
                (prompt_tokens / 1000.0) * target_adapter.cost_per_1k_prompt
                + (completion_tokens / 1000.0) * target_adapter.cost_per_1k_completion
            )

            # ── Spend recording intentionally omitted here ───────────────────
            # Recording is handled by the caller (main.py stream_generator)
            # which has access to the correct gateway request_id, the
            # AsyncLedgerQueue, and the pre-reservation amount. Doing it
            # here would use a backend chunk-id as the req_id, bypass the
            # queue, and double-count the reserved budget.

