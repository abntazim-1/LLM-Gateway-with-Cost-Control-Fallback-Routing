import time
import os
from typing import Any, Dict, List
import httpx
from gateway.adapters.base import BaseAdapter, NormalizedResponse, NormalizedMessage, AdapterException

class AnthropicAdapter(BaseAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")

    def _convert_messages(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Convert OpenAI format to Anthropic format"""
        system = ""
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
        return {"system": system, "messages": anthropic_messages}

    async def complete(self, messages: List[Dict[str, str]], **kwargs) -> NormalizedResponse:
        if not self.api_key:
            raise AdapterException("ANTHROPIC_API_KEY not set")
            
        start_time = time.time()
        
        converted = self._convert_messages(messages)
        
        # Strip unsupported kwargs
        supported_kwargs = {k: v for k, v in kwargs.items() if k in ["temperature", "max_tokens", "top_p"]}
        if "max_tokens" not in supported_kwargs:
            supported_kwargs["max_tokens"] = 1024
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.endpoint}/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "system": converted["system"],
                        "messages": converted["messages"],
                        **supported_kwargs
                    },
                    timeout=kwargs.get("timeout", 10.0)
                )
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                raise AdapterException(f"Anthropic request failed: {str(e)}")

        latency_ms = (time.time() - start_time) * 1000
        
        prompt_tokens = data.get("usage", {}).get("input_tokens", 0)
        completion_tokens = data.get("usage", {}).get("output_tokens", 0)
        
        cost_usd = self._calculate_cost(prompt_tokens, completion_tokens)
        
        normalized_messages = [
            NormalizedMessage(
                role=data.get("role", "assistant"),
                content="".join([block["text"] for block in data.get("content", []) if block["type"] == "text"])
            )
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
        # Anthropic doesn't have a simple health endpoint without a payload, 
        # so we can just check if the endpoint is reachable
        if not self.api_key:
            return False
        return True # Simplified for this demo
