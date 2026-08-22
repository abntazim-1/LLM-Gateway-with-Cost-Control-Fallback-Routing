import logging
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import tiktoken
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Loaded encodings are process-wide and immutable, so cache them rather than
# paying the (network-backed, first-time) load cost on every request.
_ENCODING_CACHE: Dict[str, Any] = {}

# Chat models don't bill raw content tokens — the provider wraps each message
# in role/delimiter scaffolding and primes the reply, and bills that too.
# These are OpenAI's documented per-message and per-reply constants; models
# with heavier templates (Qwen ships a default system block, for example)
# should override `token_overhead_per_message` in their backend config.
DEFAULT_TOKEN_OVERHEAD_PER_MESSAGE = 4
REPLY_PRIMING_TOKENS = 3


class NormalizedMessage(BaseModel):
    role: str
    content: str


class NormalizedResponse(BaseModel):
    id: str
    backend_id: str
    model: str
    messages: List[NormalizedMessage]
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    # True when the backend that answered was not the router's first choice —
    # i.e. the caller received a different (possibly much weaker) model than
    # routing selected. Set by the router, surfaced to clients as a response
    # header so a silent substitution is at least observable.
    is_fallback: bool = False


class AdapterException(Exception):
    """Base exception for all adapter errors."""

    pass


class BaseAdapter(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.id = config["id"]
        self.model = config["model"]
        self.endpoint = config.get("endpoint", "")
        self.cost_per_1k_prompt = config.get("cost_per_1k_prompt", 0.0)
        self.cost_per_1k_completion = config.get("cost_per_1k_completion", 0.0)
        # 0 / absent means "unknown", which is treated as unconstrained rather
        # than as a zero-size window.
        self.context_length = config.get("context_length", 0) or 0
        # Declared capability, independent of price. Higher is more capable.
        # Price is a poor proxy — a small fast model can cost more than a
        # large self-hosted one — so routing that wants "the better model"
        # must rank on this, falling back to cost only when it is unset.
        self.capability_tier = config.get("capability_tier", 0) or 0
        self.tokenizer_name = config.get("tokenizer", "cl100k_base")
        self.token_overhead_per_message = config.get(
            "token_overhead_per_message", DEFAULT_TOKEN_OVERHEAD_PER_MESSAGE
        )
        import httpx

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=20)
        )

    async def close(self) -> None:
        """Close underlying httpx.AsyncClient session and release connection pool / file descriptors."""
        if hasattr(self, "client") and self.client is not None:
            await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _get_encoding(self):
        """Resolve this backend's tokenizer, caching it process-wide.

        Falls back to cl100k_base for an unrecognised tokenizer name rather
        than failing the request — but logs it, because a silently wrong
        tokenizer means silently wrong cost estimates."""
        name = self.tokenizer_name
        if name not in _ENCODING_CACHE:
            try:
                _ENCODING_CACHE[name] = tiktoken.get_encoding(name)
            except Exception as e:
                logger.warning(
                    f"Unknown tokenizer '{name}' for backend {self.id} "
                    f"({e}); falling back to cl100k_base. Token counts and "
                    f"cost estimates for this backend may be inaccurate."
                )
                _ENCODING_CACHE[name] = tiktoken.get_encoding("cl100k_base")
        return _ENCODING_CACHE[name]

    def count_prompt_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate the billable prompt tokens for `messages` on this backend.

        This is an estimate used for the pre-flight budget reservation; the
        reservation is later reconciled against the provider's own reported
        usage where available. It is deliberately per-adapter: tokenizers are
        a property of the model, not of the gateway."""
        encoding = self._get_encoding()
        total = REPLY_PRIMING_TOKENS
        for m in messages:
            content = m.get("content") or ""
            total += self.token_overhead_per_message
            total += len(encoding.encode(str(content)))
        return max(1, total)

    def count_completion_tokens(self, text: str) -> int:
        """Estimate billable completion tokens for generated `text`."""
        if not text:
            return 0
        return len(self._get_encoding().encode(text))

    def count_tokens(
        self, messages: List[Dict[str, Any]], completion_text: str = ""
    ) -> Tuple[int, int]:
        """Convenience pair of (prompt_tokens, completion_tokens)."""
        return (
            self.count_prompt_tokens(messages),
            self.count_completion_tokens(completion_text),
        )

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        prompt_cost = (prompt_tokens / 1000.0) * self.cost_per_1k_prompt
        completion_cost = (completion_tokens / 1000.0) * self.cost_per_1k_completion
        return prompt_cost + completion_cost

    def _filter_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out dangerous or unsupported kwargs to prevent injection."""
        allowed_keys = {
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "presence_penalty",
            "frequency_penalty",
        }
        return {k: v for k, v in kwargs.items() if k in allowed_keys}

    @abstractmethod
    async def complete(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> NormalizedResponse:
        """Execute a completion request."""
        pass

    @abstractmethod
    async def complete_stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a streaming completion request."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is healthy."""
        pass
