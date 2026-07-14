from gateway.ledger.store import LedgerStore

class BudgetExceededException(Exception):
    pass

class BudgetPolicy:
    def __init__(self, ledger: LedgerStore):
        self.ledger = ledger

    def check_preflight(self, api_key: str, estimated_cost: float = 0.0) -> bool:
        """
        Check if the request is allowed given the current spend and estimated cost.
        Raises BudgetExceededException if budget is blown.
        """
        budget = self.ledger.get_budget(api_key)
        if not budget:
            # If no budget is found, we might default to deny
            raise BudgetExceededException("No budget found for API key")

        # Check daily
        projected_daily = budget["spend_today"] + estimated_cost
        if budget["daily_limit_usd"] and projected_daily > budget["daily_limit_usd"]:
            raise BudgetExceededException(f"Daily budget exceeded. Limit: ${budget['daily_limit_usd']}, Projected: ${projected_daily}")

        # Check monthly
        projected_monthly = budget["spend_month"] + estimated_cost
        if budget["monthly_limit_usd"] and projected_monthly > budget["monthly_limit_usd"]:
            raise BudgetExceededException(f"Monthly budget exceeded. Limit: ${budget['monthly_limit_usd']}, Projected: ${projected_monthly}")

        return True
