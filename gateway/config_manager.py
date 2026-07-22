import os
import logging
from typing import Dict, Any, List, Tuple
from gateway import load_config
from gateway.adapters.openai_adapter import OpenAIAdapter
from gateway.adapters.anthropic_adapter import AnthropicAdapter
from gateway.adapters.local_vllm_adapter import LocalVLLMAdapter
from gateway.adapters.base import BaseAdapter

logger = logging.getLogger(__name__)

class ConfigManager:
    """Dynamic configuration loader supporting live hot-reloading of backends and routing policies."""

    def __init__(self, config_dir: str):
        self.config_dir = config_dir

    def load_backends(self) -> List[BaseAdapter]:
        backends_path = os.path.join(self.config_dir, "backends.yaml")
        if not os.path.exists(backends_path):
            logger.warning(f"Backends config file missing at {backends_path}")
            return []

        backends_cfg = load_config(backends_path).get("backends", [])
        adapters = []
        for cfg in backends_cfg:
            provider = cfg.get("provider")
            if provider == "openai":
                adapters.append(OpenAIAdapter(cfg))
            elif provider == "anthropic":
                adapters.append(AnthropicAdapter(cfg))
            elif provider == "local_vllm":
                adapters.append(LocalVLLMAdapter(cfg))
            else:
                logger.warning(f"Unknown backend provider '{provider}' in configuration.")
        return adapters

    def load_routing_strategy(self) -> str:
        routing_path = os.path.join(self.config_dir, "routing_policy.yaml")
        if not os.path.exists(routing_path):
            return "cost_first"

        routing_cfg = load_config(routing_path).get("routing", {})
        return routing_cfg.get("strategy", "cost_first")

    def load_budgets(self) -> list:
        budgets_path = os.path.join(self.config_dir, "budgets.yaml")
        if not os.path.exists(budgets_path):
            return []
        return load_config(budgets_path).get("budgets", [])
