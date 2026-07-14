import pytest
from gateway.policy.budget import BudgetPolicy, BudgetExceededException
from gateway.ledger.store import LedgerStore

def test_budget_enforcement():
    ledger = LedgerStore(":memory:")
    
    budgets = [
        {"api_key": "sk-test", "daily_limit_usd": 1.0, "monthly_limit_usd": 10.0}
    ]
    ledger.load_budgets_from_config(budgets)
    
    policy = BudgetPolicy(ledger)
    
    # Should allow
    assert policy.check_preflight("sk-test", estimated_cost=0.5) == True
    
    # Manually add spend
    ledger.record_request("sk-test", "req-1", "backend-1", "model-1", 100, 100, 0.6, 100)
    
    # Should block next request because 0.6 + 0.5 = 1.1 > 1.0
    with pytest.raises(BudgetExceededException):
        policy.check_preflight("sk-test", estimated_cost=0.5)
