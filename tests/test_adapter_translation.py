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
