import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

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
        
    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        prompt_cost = (prompt_tokens / 1000.0) * self.cost_per_1k_prompt
        completion_cost = (completion_tokens / 1000.0) * self.cost_per_1k_completion
        return prompt_cost + completion_cost

    @abstractmethod
    async def complete(self, messages: List[Dict[str, str]], **kwargs) -> NormalizedResponse:
        """Execute a completion request."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the backend is healthy."""
        pass
