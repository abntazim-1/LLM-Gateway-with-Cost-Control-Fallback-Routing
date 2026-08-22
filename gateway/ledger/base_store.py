from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


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
        latency: float,
    ) -> None:
        """Record a completed inference request and update spend totals."""
        pass

    @abstractmethod
    async def get_circuit_breaker_state(
        self, backend_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch stored circuit breaker state for a backend."""
        pass

    @abstractmethod
    async def update_circuit_breaker_state(
        self,
        backend_id: str,
        state: str,
        consecutive_failures: int,
        last_failure_time: float,
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
        requests_per_minute: Optional[int] = None,
    ) -> bool:
        """Update or insert budget limits for an API key."""
        pass

    @abstractmethod
    async def check_and_reserve_budget(
        self,
        api_key: str,
        estimated_cost: float,
    ) -> Dict[str, Any]:
        """
        Atomically verify that adding *estimated_cost* to the current spend
        would not breach either limit, then tentatively reserve it by
        incrementing the spend counters.

        Returns the budget row on success.
        Raises BudgetExceededException (imported by callers from
        gateway.policy.budget) if a limit would be breached.
        Raises ValueError if the api_key is not found.
        """
        pass

    @abstractmethod
    async def check_rate_limit(self, api_key: str) -> None:
        """Enforce the per-minute sliding-window rate limit for *api_key*.

        Must be atomic: the count check and the admission record must happen
        in a single transaction so concurrent calls cannot both pass a limit
        that only one should be allowed through.

        Raises HTTPException(429) when the limit is exceeded.
        Must survive process restarts (state persisted externally, not in-memory).
        """
        pass
