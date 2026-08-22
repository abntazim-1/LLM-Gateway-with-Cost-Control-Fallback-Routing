import json
import os

import pytest

from gateway import expand_env_vars, load_config
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


def test_env_var_expansion(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_CLIENT_KEY", "sk-custom-injected-key")

    # Test inline expansion helper
    raw = "api_key: ${TEST_CLIENT_KEY}\nfallback: ${MISSING_VAR:-default-val}"
    expanded = expand_env_vars(raw)
    assert "sk-custom-injected-key" in expanded
    assert "default-val" in expanded

    # Test load_config with file
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(raw, encoding="utf-8")

    parsed = load_config(str(cfg_file))
    assert parsed["api_key"] == "sk-custom-injected-key"
    assert parsed["fallback"] == "default-val"


def test_config_manager_loads_budgets_from_env(monkeypatch, tmp_path):
    env_budgets = [
        {
            "api_key": "sk-env-key-1",
            "daily_limit_usd": 15.0,
            "monthly_limit_usd": 150.0,
        },
        {
            "api_key": "sk-env-key-2",
            "daily_limit_usd": 25.0,
            "monthly_limit_usd": 250.0,
        },
    ]
    monkeypatch.setenv("GATEWAY_BUDGETS_JSON", json.dumps(env_budgets))

    cm = ConfigManager(str(tmp_path))
    budgets = cm.load_budgets()

    assert len(budgets) == 2
    assert budgets[0]["api_key"] == "sk-env-key-1"
    assert budgets[1]["api_key"] == "sk-env-key-2"
