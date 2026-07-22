import re
from typing import List, Dict, Tuple

class PiiVault:
    """Reversible PII Anonymizer Vault for masking sensitive tokens before sending to LLMs."""

    def __init__(self):
        self.patterns = {
            "EMAIL": re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]*[a-zA-Z0-9-]\b"),
            "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
            "PHONE": re.compile(r"\b(?:\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b"),
            "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "AWS_KEY": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
            "JWT": re.compile(r"\beyJ[a-zA-Z0-9_\-]*\.[a-zA-Z0-9_\-]*\.[a-zA-Z0-9_\-]*\b"),
            "BEARER": re.compile(r"\bBearer\s+[a-zA-Z0-9_\-\.]{20,}\b")
        }

    def mask_text(self, text: str) -> Tuple[str, Dict[str, str]]:
        if not text:
            return text, {}

        vault_mapping = {}
        counters = {}
        sanitized = text

        for label, pattern in self.patterns.items():
            matches = list(pattern.finditer(sanitized))
            # Process matches in reverse order so character offsets remain valid
            for match in reversed(matches):
                val = match.group(0)
                counters[label] = counters.get(label, 0) + 1
                token = f"[{label}_{counters[label]}]"
                vault_mapping[token] = val
                sanitized = sanitized[:match.start()] + token + sanitized[match.end():]

        return sanitized, vault_mapping

    def restore_text(self, text: str, vault_mapping: Dict[str, str]) -> str:
        if not text or not vault_mapping:
            return text

        restored = text
        for token, original_val in vault_mapping.items():
            restored = restored.replace(token, original_val)
        return restored

class PiiSanitizer:
    """Enterprise-grade PII Sanitizer middleware for static regex redaction."""

    def __init__(self):
        self.vault = PiiVault()

    def sanitize_text(self, text: str) -> str:
        if not text:
            return text
        sanitized = text
        for label, pattern in self.vault.patterns.items():
            sanitized = pattern.sub(f"[{label}_REDACTED]", sanitized)
        return sanitized

    def sanitize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        sanitized_messages = []
        for msg in messages:
            sanitized_messages.append({
                "role": msg.get("role", ""),
                "content": self.sanitize_text(msg.get("content", ""))
            })
        return sanitized_messages
