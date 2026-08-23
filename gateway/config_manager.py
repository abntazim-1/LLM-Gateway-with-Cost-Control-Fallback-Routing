import logging
import os
from typing import Any, Dict, List, Tuple

from gateway import load_config
from gateway.adapters.anthropic_adapter import AnthropicAdapter
from gateway.adapters.base import BaseAdapter
from gateway.adapters.local_vllm_adapter import LocalVLLMAdapter
from gateway.adapters.openai_adapter import OpenAIAdapter

logger = logging.getLogger(__name__)


class ConfigManager:
    """Dynamic configuration loader supporting live hot-reloading of backends and routing policies."""

    def __init__(self, config_dir: str):
        self.config_dir = config_dir

    def load_backends(self) -> List[BaseAdapter]:
        # Honours the same override as startup, so a hot reload cannot swap a
        # deployment back onto the repository's default backends.
        backends_path = os.environ.get(
            "BACKENDS_CONFIG_PATH", os.path.join(self.config_dir, "backends.yaml")
        )
        if not os.path.exists(backends_path):
            logger.warning(f"Backends config file missing at {backends_path}")
            return []

        backends_cfg = load_config(backends_path).get("backends", [])
        adapters: List[BaseAdapter] = []
        for cfg in backends_cfg:
            provider = cfg.get("provider")
            if provider == "openai":
                adapters.append(OpenAIAdapter(cfg))
            elif provider == "anthropic":
                adapters.append(AnthropicAdapter(cfg))
            elif provider in ("local_vllm", "ollama"):
                adapters.append(LocalVLLMAdapter(cfg))
            else:
                logger.warning(
                    f"Unknown backend provider '{provider}' in configuration."
                )
        return adapters

    def load_routing_strategy(self) -> str:
        routing_path = os.path.join(self.config_dir, "routing_policy.yaml")
        if not os.path.exists(routing_path):
            return "cost_first"

        routing_cfg = load_config(routing_path).get("routing", {})
        return routing_cfg.get("strategy", "cost_first")

    def load_budgets(self) -> list:
        raw_env = os.environ.get("GATEWAY_BUDGETS_JSON") or os.environ.get(
            "GATEWAY_BUDGETS"
        )
        if raw_env:
            try:
                import json

                parsed = json.loads(raw_env)
                return (
                    parsed.get("budgets", parsed)
                    if isinstance(parsed, dict)
                    else parsed
                )
            except Exception as e:
                logger.error(
                    f"Failed to parse GATEWAY_BUDGETS environment variable: {e}"
                )

        budgets_path = os.path.join(self.config_dir, "budgets.yaml")
        if not os.path.exists(budgets_path):
            return []
        return load_config(budgets_path).get("budgets", [])
