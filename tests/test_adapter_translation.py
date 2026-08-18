import pytest
from gateway.adapters.transformer import ParameterTransformer

def test_openai_to_anthropic_translation():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I am fine, thanks!"}
    ]
    kwargs = {
        "max_tokens": 512,
        "temperature": 0.7,
        "stop": ["END", "STOP"],
        "unsupported_param": "ignored"
    }

    payload = ParameterTransformer.openai_to_anthropic(messages, kwargs)

    assert payload["system"] == "You are a helpful assistant."
    # Back-to-back user messages should be merged
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"] == "Hello!\n\nHow are you?"
    assert payload["messages"][1]["role"] == "assistant"
    assert payload["messages"][1]["content"] == "I am fine, thanks!"
    assert payload["max_tokens"] == 512
    assert payload["temperature"] == 0.7
    assert payload["stop_sequences"] == ["END", "STOP"]
    assert "unsupported_param" not in payload

def test_openai_clean_kwargs():
    kwargs = {
        "temperature": 0.5,
        "max_tokens": 100,
        "custom_internal_flag": "test",
        "api_key": "sk-xxx"
    }
    cleaned = ParameterTransformer.openai_clean_kwargs(kwargs)
    assert cleaned == {"temperature": 0.5, "max_tokens": 100}

@pytest.mark.asyncio
async def test_adapter_client_close():
    from gateway.adapters.local_vllm_adapter import LocalVLLMAdapter
    cfg = {"id": "test-vllm", "provider": "local_vllm", "model": "qwen", "endpoint": "http://localhost:11434/v1"}
    adapter = LocalVLLMAdapter(cfg)
    assert not adapter.client.is_closed

    await adapter.close()
    assert adapter.client.is_closed

@pytest.mark.asyncio
async def test_adapter_async_context_manager():
    from gateway.adapters.local_vllm_adapter import LocalVLLMAdapter
    cfg = {"id": "test-vllm", "provider": "local_vllm", "model": "qwen", "endpoint": "http://localhost:11434/v1"}
    async with LocalVLLMAdapter(cfg) as adapter:
        assert not adapter.client.is_closed
    assert adapter.client.is_closed

@pytest.mark.asyncio
async def test_router_close_all_adapters():
    from gateway.adapters.local_vllm_adapter import LocalVLLMAdapter
    from gateway.policy.router import Router
    from gateway.policy.circuit_breaker import CircuitBreakerRegistry
    from gateway.ledger.store import LedgerStore

    ledger = LedgerStore(":memory:")
    registry = CircuitBreakerRegistry(ledger=ledger)
    a1 = LocalVLLMAdapter({"id": "vllm-1", "provider": "local_vllm", "model": "m1", "endpoint": "http://localhost:11434/v1"})
    a2 = LocalVLLMAdapter({"id": "vllm-2", "provider": "local_vllm", "model": "m2", "endpoint": "http://localhost:11434/v1"})

    router = Router(adapters=[a1, a2], circuit_registry=registry)
    assert not a1.client.is_closed
    assert not a2.client.is_closed

    await router.close()
    assert a1.client.is_closed
    assert a2.client.is_closed

