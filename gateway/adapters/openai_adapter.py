import os
import time
from typing import Any, AsyncGenerator, Dict, List

import httpx

from gateway.adapters.base import (
    AdapterException,
    BaseAdapter,
    NormalizedMessage,
    NormalizedResponse,
)
from gateway.adapters.transformer import ParameterTransformer
from gateway.policy.key_pool import ProviderKeyPool


class OpenAIAdapter(BaseAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.key_pool = ProviderKeyPool()

        # A backend's own `api_keys` are authoritative. OPENAI_API_KEY is a
        # fallback for backends that declare none — never an addition.
        #
        # Previously both were pooled together, so a backend configured with
        # its own credential (an OpenAI-compatible provider such as Groq)
        # round-robined between that key and the global OpenAI one, sending
        # roughly half its traffic upstream with a credential for a different
        # vendor. Requests failed intermittently and the backend looked
        # flaky rather than misconfigured.
        cfg_keys = [k for k in (config.get("api_keys") or []) if k]
        if cfg_keys:
            for k in cfg_keys:
                self.key_pool.add_key(k)
        elif self.api_key:
            for k in self.api_key.split(","):
                if k.strip():
                    self.key_pool.add_key(k.strip())

    def _get_active_key(self) -> str:
        key = self.key_pool.get_next_key()
        if not key:
            raise AdapterException("OPENAI_API_KEY not set")
        return key

    async def complete(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> NormalizedResponse:
        active_key = self._get_active_key()
        start_time = time.time()

        try:
            response = await self.client.post(
                f"{self.endpoint}/chat/completions",
                headers={
                    "Authorization": f"Bearer {active_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    **ParameterTransformer.openai_clean_kwargs(kwargs),
                },
                timeout=kwargs.get("timeout", 10.0),
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise AdapterException(f"OpenAI request failed: {str(e)}")

        latency_ms = (time.time() - start_time) * 1000

        prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        completion_tokens = data.get("usage", {}).get("completion_tokens", 0)

        cost_usd = self._calculate_cost(prompt_tokens, completion_tokens)

        normalized_messages = [
            NormalizedMessage(
                role=choice["message"]["role"],
                content=choice["message"]["content"] or "",
            )
            for choice in data.get("choices", [])
        ]

        return NormalizedResponse(
            id=data.get("id", ""),
            backend_id=self.id,
            model=self.model,
            messages=normalized_messages,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )

    async def complete_stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        active_key = self._get_active_key()

        try:
            request_data = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                **ParameterTransformer.openai_clean_kwargs(kwargs),
                # Ask the provider to append a final usage-bearing chunk.
                # Streams otherwise carry no usage, forcing the gateway to
                # bill from a local token estimate; this gives it ground truth.
                "stream_options": {"include_usage": True},
            }
            async with self.client.stream(
                "POST",
                f"{self.endpoint}/chat/completions",
                headers={
                    "Authorization": f"Bearer {active_key}",
                    "Content-Type": "application/json",
                },
                json=request_data,
                timeout=kwargs.get("timeout", 30.0),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        import json

                        try:
                            chunk_data = json.loads(data_str)
                            yield chunk_data
                        except Exception:
                            continue
        except Exception as e:
            raise AdapterException(f"OpenAI stream request failed: {str(e)}")

    async def health_check(self) -> bool:
        """Probe the backend with the same credential a real request uses.

        This previously read `self.api_key`, which is populated only from the
        OPENAI_API_KEY environment variable, while completions authenticate
        from the key pool. Any backend whose key comes from its own
        `api_keys` config — an OpenAI-compatible provider such as Groq, for
        instance — therefore probed with the wrong credential (or none) and
        reported unhealthy forever, even while serving requests perfectly.
        The health loop would then trip its circuit breaker and route traffic
        away from a working backend.
        """
        try:
            key = self._get_active_key()
        except AdapterException:
            return False
        try:
            response = await self.client.get(
                f"{self.endpoint}/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            return False
