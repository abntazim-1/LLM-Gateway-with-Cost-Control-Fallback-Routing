import os
import re
import yaml
from typing import Any

ENV_VAR_PATTERN = re.compile(r'\$\{([^}:]+)(?::-([^}]*))?\}')

def expand_env_vars(content: str) -> str:
    """Expand ${VAR} and ${VAR:-default} patterns using os.environ."""
    def replace(match: re.Match) -> str:
        var_name = match.group(1)
        default_val = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default_val)
    return ENV_VAR_PATTERN.sub(replace, content)

def load_config(path: str) -> Any:
    """Load a YAML configuration file with environment variable expansion."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    expanded = expand_env_vars(content)
    return yaml.safe_load(expanded) or {}

__all__ = ["load_config", "expand_env_vars"]

