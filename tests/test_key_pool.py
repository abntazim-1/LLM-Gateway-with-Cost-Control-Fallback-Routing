import time
import pytest
from gateway.policy.key_pool import ProviderKeyPool

def test_key_pool_round_robin_rotation():
    pool = ProviderKeyPool(keys=["key1", "key2", "key3"])
    
    assert pool.get_next_key() == "key1"
    assert pool.get_next_key() == "key2"
    assert pool.get_next_key() == "key3"
    assert pool.get_next_key() == "key1"

def test_key_pool_cooldown_skipping():
    pool = ProviderKeyPool(keys=["key1", "key2"])
    
    # Mark key1 in cooldown for 10 seconds
    pool.mark_rate_limited("key1", cooldown_sec=10.0)
    
    # Next key should automatically skip key1 and return key2
    assert pool.get_next_key() == "key2"
    assert pool.get_next_key() == "key2"

def test_key_pool_add_key():
    pool = ProviderKeyPool()
    pool.add_key("key-new")
    assert pool.get_next_key() == "key-new"
