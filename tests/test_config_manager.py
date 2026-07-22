import os
import pytest
from gateway.config_manager import ConfigManager

def test_config_manager_load_from_configs_dir():
    config_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
    cm = ConfigManager(config_dir)
    
    adapters = cm.load_backends()
    strategy = cm.load_routing_strategy()
    budgets = cm.load_budgets()
    
    assert isinstance(adapters, list)
    assert len(adapters) > 0
    assert isinstance(strategy, str)
    assert isinstance(budgets, list)

def test_config_manager_missing_directory():
    cm = ConfigManager("non_existent_dir")
    adapters = cm.load_backends()
    strategy = cm.load_routing_strategy()
    
    assert adapters == []
    assert strategy == "cost_first"
