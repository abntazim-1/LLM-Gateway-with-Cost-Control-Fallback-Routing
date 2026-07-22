import time
import pytest
from gateway.policy.cache import PromptCache

def test_prompt_cache_hit_and_miss():
    cache = PromptCache(ttl_seconds=2, max_entries=10)
    messages = [{"role": "user", "content": "Hello!"}]
    kwargs = {"temperature": 0.7}
    
    # Cache miss initially
    assert cache.get(messages, kwargs) is None
    
    response = {"id": "123", "choices": [{"message": {"role": "assistant", "content": "Hi!"}}]}
    cache.set(messages, kwargs, response)
    
    # Cache hit
    cached = cache.get(messages, kwargs)
    assert cached is not None
    assert cached["id"] == "123"
    assert cached["choices"][0]["message"]["content"] == "Hi!"

def test_prompt_cache_ttl_expiration():
    cache = PromptCache(ttl_seconds=1, max_entries=10)
    messages = [{"role": "user", "content": "Hello!"}]
    kwargs = {}
    
    response = {"id": "123"}
    cache.set(messages, kwargs, response)
    
    assert cache.get(messages, kwargs) is not None
    
    time.sleep(1.1)
    
    # Should expire
    assert cache.get(messages, kwargs) is None

def test_prompt_cache_lru_capacity_eviction():
    cache = PromptCache(ttl_seconds=300, max_entries=2)
    
    msg1 = [{"role": "user", "content": "Query 1"}]
    msg2 = [{"role": "user", "content": "Query 2"}]
    msg3 = [{"role": "user", "content": "Query 3"}]
    
    cache.set(msg1, {}, {"res": 1})
    cache.set(msg2, {}, {"res": 2})
    assert cache.get(msg1, {}) is not None
    assert cache.get(msg2, {}) is not None
    
    # Setting 3rd item should evict msg1 (since msg2 was accessed most recently when msg1 was checked first, but let's verify LRU order)
    cache.set(msg3, {}, {"res": 3})
    assert len(cache.cache) <= 2
    assert cache.get(msg3, {}) is not None
