import pytest
from gateway.policy.pii import PiiSanitizer

def test_pii_sanitizer_text():
    sanitizer = PiiSanitizer()
    
    # Test Email
    text_with_email = "Please contact me at john.doe@example.com."
    assert sanitizer.sanitize_text(text_with_email) == "Please contact me at [EMAIL_REDACTED]."
    
    # Test Phone Number
    text_with_phone = "Call me at +1 (555) 555-5555 or 555-555-5555."
    sanitized_phone = sanitizer.sanitize_text(text_with_phone)
    assert "[PHONE_REDACTED]" in sanitized_phone
    
    # Test Credit Card
    text_with_cc = "My card number is 4111 1111 1111 1111."
    assert sanitizer.sanitize_text(text_with_cc) == "My card number is [CREDIT_CARD_REDACTED]."
    
    # Test SSN
    text_with_ssn = "My SSN is 000-12-3456."
    assert sanitizer.sanitize_text(text_with_ssn) == "My SSN is [SSN_REDACTED]."

def test_pii_sanitizer_messages():
    sanitizer = PiiSanitizer()
    messages = [
        {"role": "system", "content": "Keep it safe."},
        {"role": "user", "content": "Email is test@example.com and phone is 555-555-5555."}
    ]
    
    sanitized = sanitizer.sanitize_messages(messages)
    assert len(sanitized) == 2
    assert sanitized[0]["content"] == "Keep it safe."
    assert "[EMAIL_REDACTED]" in sanitized[1]["content"]
    assert "[PHONE_REDACTED]" in sanitized[1]["content"]
