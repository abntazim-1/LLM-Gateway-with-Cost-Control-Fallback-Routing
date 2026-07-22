import time
import threading
from typing import List, Dict, Optional

class ProviderKeyPool:
    """
    Round-robin key pool manager supporting multi-key rotation and rate-limit cooldowns
    to bypass provider RPM/TPM throughput limits across multiple API accounts.
    """

    def __init__(self, keys: Optional[List[str]] = None, default_cooldown_sec: float = 60.0):
        self.keys: List[str] = [k.strip() for k in (keys or []) if k.strip()]
        self.default_cooldown_sec = default_cooldown_sec
        self.cooldowns: Dict[str, float] = {} # key -> expire_timestamp
        self._index = 0
        self.lock = threading.Lock()

    def add_key(self, key: str):
        with self.lock:
            key_clean = key.strip()
            if key_clean and key_clean not in self.keys:
                self.keys.append(key_clean)

    def get_next_key(self) -> Optional[str]:
        with self.lock:
            if not self.keys:
                return None

            now = time.time()
            # Try finding a key not currently in cooldown
            for _ in range(len(self.keys)):
                key = self.keys[self._index % len(self.keys)]
                self._index += 1

                # Check if cooldown has expired
                cooldown_until = self.cooldowns.get(key, 0.0)
                if now >= cooldown_until:
                    return key

            # If all keys are in cooldown, fallback to the one expiring soonest
            soonest_key = min(self.keys, key=lambda k: self.cooldowns.get(k, 0.0))
            return soonest_key

    def mark_rate_limited(self, key: str, cooldown_sec: Optional[float] = None):
        with self.lock:
            duration = cooldown_sec if cooldown_sec is not None else self.default_cooldown_sec
            self.cooldowns[key] = time.time() + duration
