import pytest

from gateway.policy.pii import PiiSanitizer, PiiVault


def test_pii_vault_masking_and_unmasking():
    vault = PiiVault()
    text = "Please contact john.doe@example.com or call 555-123-4567 regarding SSN 000-12-3456."

    masked_text, mapping = vault.mask_text(text)

    # Verify PII values are replaced by token placeholders
    assert "john.doe@example.com" not in masked_text
    assert "555-123-4567" not in masked_text
    assert "000-12-3456" not in masked_text
    assert "[EMAIL_1]" in masked_text or "[EMAIL" in masked_text

    # Unmask text using the mapping dictionary
    restored_text = vault.restore_text(masked_text, mapping)
    assert restored_text == text


def test_secret_token_detection():
    vault = PiiVault()
    text = "My AWS key is AKIAIOSFODNN7EXAMPLE and token is Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-o-k-e-n"

    masked_text, mapping = vault.mask_text(text)

    assert "AKIAIOSFODNN7EXAMPLE" not in masked_text
    assert "AWS_KEY" in masked_text or "BEARER" in masked_text


def test_mask_messages_unique_tokens_across_conversation():
    vault = PiiVault()
    messages = [
        {"role": "system", "content": "Reply to the user's email address."},
        {"role": "user", "content": "My email is alice@example.com"},
        {"role": "user", "content": "Also cc bob@example.com about my SSN 123-45-6789"},
    ]

    masked_messages, mapping = vault.mask_messages(messages)

    # Placeholder counters are shared across messages -> unique tokens
    assert masked_messages[1]["content"] == "My email is [EMAIL_1]"
    assert "[EMAIL_2]" in masked_messages[2]["content"]
    assert "[SSN_1]" in masked_messages[2]["content"]
    assert mapping["[EMAIL_1]"] == "alice@example.com"
    assert mapping["[EMAIL_2]"] == "bob@example.com"
    assert mapping["[SSN_1]"] == "123-45-6789"

    # No PII leaks into the messages sent to the LLM
    for msg in masked_messages:
        assert "alice@example.com" not in msg["content"]
        assert "bob@example.com" not in msg["content"]
        assert "123-45-6789" not in msg["content"]

    # Restoring a model response that echoes placeholders returns the originals
    response = "Emailed [EMAIL_1] and cc'd [EMAIL_2] regarding [SSN_1]."
    restored = vault.restore_text(response, mapping)
    assert "alice@example.com" in restored
    assert "bob@example.com" in restored
    assert "123-45-6789" in restored


def test_mask_messages_without_pii_is_passthrough():
    vault = PiiVault()
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    masked_messages, mapping = vault.mask_messages(messages)
    assert masked_messages == messages
    assert mapping == {}
