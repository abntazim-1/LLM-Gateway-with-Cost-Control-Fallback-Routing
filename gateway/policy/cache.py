import time
import hashlib
import json
import threading
from collections import OrderedDict
from typing import Dict, Any, List, Optional

class PromptCache:
    """Thread-safe, TTL-based, LRU bounded cache for LLM responses."""

    def __init__(self, ttl_seconds: int = 300, max_entries: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.lock = threading.Lock()

    def _generate_key(self, messages: List[Dict[str, str]], kwargs: Dict[str, Any]) -> str:
        # Normalize messages
        normalized_messages = [
            {"role": str(m.get("role", "")).strip(), "content": str(m.get("content", "")).strip()}
            for m in messages
        ]
        # Normalize kwargs (filter out non-comparable values)
        normalized_kwargs = {
            k: v for k, v in kwargs.items()
            if k in {"temperature", "max_tokens", "top_p", "presence_penalty", "frequency_penalty"}
        }
        serialized = json.dumps(
            {"messages": normalized_messages, "kwargs": normalized_kwargs},
            sort_keys=True
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, messages: List[Dict[str, str]], kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        key = self._generate_key(messages, kwargs)
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            # Check TTL expiration
            if time.time() - entry["timestamp"] > self.ttl_seconds:
                del self.cache[key]
                return None
                
            # Move to end to mark as recently used
            self.cache.move_to_end(key)
            return entry["response"]

    def set(self, messages: List[Dict[str, str]], kwargs: Dict[str, Any], response: Dict[str, Any]):
        key = self._generate_key(messages, kwargs)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = {
                "timestamp": time.time(),
                "response": response
            }
            # Enforce max capacity eviction (FIFO/LRU eviction of oldest item)
            while len(self.cache) > self.max_entries:
                self.cache.popitem(last=False)

    def clear(self):
        """Clear all cache contents."""
        with self.lock:
            self.cache.clear()

