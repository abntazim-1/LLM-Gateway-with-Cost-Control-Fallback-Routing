import time
from typing import Any, Dict, List
import httpx
from gateway.adapters.base import BaseAdapter, NormalizedResponse, NormalizedMessage, AdapterException

class LocalVLLMAdapter(BaseAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    async def complete(self, messages: List[Dict[str, str]], **kwargs) -> NormalizedResponse:
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.endpoint}/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": self.model,
                        "messages": messages,
                        **kwargs
                    },
                    timeout=kwargs.get("timeout", 10.0)
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                raise AdapterException(f"Local vLLM request failed: {str(e)}")

        latency_ms = (time.time() - start_time) * 1000
        
        prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
        
        cost_usd = self._calculate_cost(prompt_tokens, completion_tokens)
        
        normalized_messages = [
            NormalizedMessage(
                role=choice["message"]["role"],
                content=choice["message"]["content"] or ""
            ) for choice in data.get("choices", [])
        ]

        return NormalizedResponse(
            id=data.get("id", ""),
            backend_id=self.id,
            model=self.model,
            messages=normalized_messages,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms
        )

    async def health_check(self) -> bool:
        async with httpx.AsyncClient() as client:
            try:
                # vLLM provides a /health endpoint usually
                # If using standard OpenAI compat server, /v1/models is safe
                response = await client.get(
                    f"{self.endpoint}/models",
                    timeout=2.0
                )
                return response.status_code == 200
            except Exception:
                return False
