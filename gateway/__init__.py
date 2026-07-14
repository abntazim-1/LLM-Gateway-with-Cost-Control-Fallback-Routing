import yaml
from typing import Dict, Any, List

def load_config(path: str) -> Any:
    with open(path, "r") as f:
        return yaml.safe_load(f)

__all__ = ["load_config"]
