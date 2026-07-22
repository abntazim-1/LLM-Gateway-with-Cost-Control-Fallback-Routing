import pytest
from gateway.policy.pii import PiiVault, PiiSanitizer

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
