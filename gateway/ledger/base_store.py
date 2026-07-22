from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseLedgerStore(ABC):
    """Abstract base class establishing the interface contract for gateway ledger storage engines."""

    @abstractmethod
    async def load_budgets_from_config(self, budgets_config: list) -> None:
        """Seed initial budgets into the store."""
        pass

    @abstractmethod
    async def get_budget(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve budget limits and spend metrics for a given API key."""
        pass

    @abstractmethod
    async def record_request(
        self, 
        api_key: str, 
        req_id: str, 
        backend: str, 
        model: str, 
        prompt_tokens: int, 
        comp_tokens: int, 
        cost: float, 
        latency: float
    ) -> None:
        """Record a completed inference request and update spend totals."""
        pass

    @abstractmethod
    async def get_circuit_breaker_state(self, backend_id: str) -> Optional[Dict[str, Any]]:
        """Fetch stored circuit breaker state for a backend."""
        pass

    @abstractmethod
    async def update_circuit_breaker_state(
        self, 
        backend_id: str, 
        state: str, 
        consecutive_failures: int, 
        last_failure_time: float
    ) -> None:
        """Update circuit breaker state for a backend."""
        pass

    @abstractmethod
    async def get_all_budgets(self) -> list:
        """Retrieve all client budgets."""
        pass

    @abstractmethod
    async def get_all_requests(self, limit: int = 50) -> list:
        """Retrieve recent request logs up to limit."""
        pass

    @abstractmethod
    async def get_all_circuit_breakers(self) -> list:
        """Retrieve states of all registered circuit breakers."""
        pass

    @abstractmethod
    async def update_budget_limits(
        self, 
        api_key: str, 
        daily_limit_usd: float, 
        monthly_limit_usd: float, 
        requests_per_minute: Optional[int] = None
    ) -> bool:
        """Update or insert budget limits for an API key."""
        pass
