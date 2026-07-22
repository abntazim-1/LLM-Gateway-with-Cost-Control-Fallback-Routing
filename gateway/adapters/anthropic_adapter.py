import time
import os
from typing import Any, Dict, List, AsyncGenerator
import httpx
from gateway.adapters.base import BaseAdapter, NormalizedResponse, NormalizedMessage, AdapterException
from gateway.adapters.transformer import ParameterTransformer
from gateway.policy.key_pool import ProviderKeyPool

class AnthropicAdapter(BaseAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.key_pool = ProviderKeyPool()
        
        cfg_keys = config.get("api_keys", [])
        if isinstance(cfg_keys, list):
            for k in cfg_keys:
                self.key_pool.add_key(k)
        if self.api_key:
            for k in self.api_key.split(","):
                self.key_pool.add_key(k)

    def _get_active_key(self) -> str:
        key = self.key_pool.get_next_key()
        if not key:
            raise AdapterException("ANTHROPIC_API_KEY not set")
        return key

    async def complete(self, messages: List[Dict[str, str]], **kwargs) -> NormalizedResponse:
        active_key = self._get_active_key()
        start_time = time.time()
        
        payload = ParameterTransformer.openai_to_anthropic(messages, kwargs)
        payload["model"] = self.model
        
        try:
            response = await self.client.post(
                f"{self.endpoint}/messages",
                headers={
                    "x-api-key": active_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json=payload,
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

    async def complete_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        active_key = self._get_active_key()
            
        payload = ParameterTransformer.openai_to_anthropic(messages, kwargs)
        payload["model"] = self.model
        payload["stream"] = True
            
        try:
            msg_id = "anthropic-" + os.urandom(8).hex()
            async with self.client.stream(
                "POST",
                f"{self.endpoint}/messages",
                headers={
                    "x-api-key": active_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json=payload,
                timeout=kwargs.get("timeout", 30.0)
            ) as response:
                response.raise_for_status()
                
                import json
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        try:
                            event_data = json.loads(data_str)
                            event_type = event_data.get("type")
                            
                            if event_type == "message_start":
                                if "message" in event_data:
                                    msg_id = event_data["message"].get("id", msg_id)
                                    
                            elif event_type == "content_block_delta":
                                delta = event_data.get("delta", {})
                                if delta.get("type") == "text_delta" and "text" in delta:
                                    yield {
                                        "id": msg_id,
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": self.model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": delta["text"]},
                                            "finish_reason": None
                                        }]
                                    }
                                    
                            elif event_type == "message_stop":
                                yield {
                                    "id": msg_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": self.model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "stop"
                                    }]
                                }
                                break
                        except Exception:
                            continue
        except Exception as e:
            raise AdapterException(f"Anthropic stream request failed: {str(e)}")

    async def health_check(self) -> bool:
        if not self.api_key:
            return False
        try:
            # Send a minimal request to test endpoint connectivity.
            # Any HTTP response status (even 4xx auth errors) indicates the endpoint is reachable.
            response = await self.client.post(
                f"{self.endpoint}/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1
                },
                timeout=3.0
            )
            return response.status_code in (200, 400, 401, 403)
        except Exception:
            return False
