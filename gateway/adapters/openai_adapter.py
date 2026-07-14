import time
import os
from typing import Any, Dict, List
import httpx
from gateway.adapters.base import BaseAdapter, NormalizedResponse, NormalizedMessage, AdapterException

class OpenAIAdapter(BaseAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = os.environ.get("OPENAI_API_KEY")

    async def complete(self, messages: List[Dict[str, str]], **kwargs) -> NormalizedResponse:
        if not self.api_key:
            raise AdapterException("OPENAI_API_KEY not set")
            
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.endpoint}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
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
                raise AdapterException(f"OpenAI request failed: {str(e)}")

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
        if not self.api_key:
            return False
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.endpoint}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5.0
                )
                return response.status_code == 200
            except Exception:
                return False
