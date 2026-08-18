from gateway.ledger.store import LedgerStore


class BudgetExceededException(Exception):
    pass


class BudgetPolicy:
    def __init__(self, ledger: LedgerStore):
        self.ledger = ledger

    async def check_and_reserve(self, api_key: str, estimated_cost: float = 0.0) -> bool:
        """
        Atomically verify budget limits and reserve *estimated_cost* against
        both daily and monthly spend counters.

        The check AND the increment happen inside a single DB lock/transaction,
        so concurrent requests cannot both read the same stale spend value and
        both pass a budget that only one of them should be allowed to use.

        Raises BudgetExceededException if a limit would be breached.
        The reservation is later corrected by record_request(reserved_cost=...).
        """
        await self.ledger.check_and_reserve_budget(api_key, estimated_cost)
        return True

    # ── Backwards-compat shim ────────────────────────────────────────────────
    async def check_preflight(self, api_key: str, estimated_cost: float = 0.0) -> bool:
        """Deprecated: use check_and_reserve instead (same semantics, atomic)."""
        return await self.check_and_reserve(api_key, estimated_cost)

