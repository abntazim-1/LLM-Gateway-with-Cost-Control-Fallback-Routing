import re
from typing import Dict, List, Optional, Tuple


def _luhn_valid(digits: str) -> bool:
    """Luhn checksum, the validity rule every real card number satisfies.

    Without this, any 13-16 digit run — order IDs, transaction refs,
    timestamps — is masked as a card number, corrupting the prompt before the
    model ever sees it. A random number passes Luhn only ~10% of the time, so
    this removes the large majority of false positives at no cost to recall.
    """
    if not 13 <= len(digits) <= 19 or not digits.isdigit():
        return False
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


class PiiVault:
    """Reversible PII Anonymizer Vault for masking sensitive tokens before sending to LLMs.

    Detection is regex-based, which bounds what it can find: structured
    identifiers with a distinctive shape are reliable, while free-text
    identifiers — personal names, street addresses, medical conditions — have
    no such shape and are NOT detected. Those require NER; see the note in
    `docs/AI_ML_FLAWS.md` (F6).
    """

    # Matches that must additionally satisfy a validator to count. Keeps
    # high-recall patterns from masking ordinary numbers.
    VALIDATORS = {"CREDIT_CARD": _luhn_valid}

    def __init__(self):
        self.patterns = {
            "EMAIL": re.compile(
                r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]*[a-zA-Z0-9-]\b"
            ),
            # Obfuscated forms people use to dodge scrapers; still trivially
            # readable to a model, so still PII. BOTH the at- and dot-markers
            # must be obfuscated, otherwise ordinary prose ("me at jane . doe")
            # reads as an address.
            "EMAIL_OBFUSCATED": re.compile(
                r"\b[a-zA-Z0-9_.+-]+\s*(?:\[at\]|\(at\)|\sat\s)\s*"
                r"[a-zA-Z0-9-]+\s*(?:\[dot\]|\(dot\)|\sdot\s)\s*[a-zA-Z]{2,}\b",
                re.IGNORECASE,
            ),
            # Same address with whitespace padding but a literal @.
            "EMAIL_SPACED": re.compile(
                r"\b[a-zA-Z0-9_+-]+(?:\s*\.\s*[a-zA-Z0-9_+-]+)*\s*@\s*"
                r"[a-zA-Z0-9-]+\s*\.\s*[a-zA-Z]{2,}\b"
            ),
            "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
            "IBAN": re.compile(
                r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b"
            ),
            "SSN": re.compile(r"(?<!\d)\d{3}[- ]\d{2}[- ]\d{4}(?!\d)"),
            # Either separated groups, or exactly ten bare digits. A longer
            # unbroken digit run is an identifier of some other kind, not a
            # phone number — matching those masked order IDs and timestamps.
            "PHONE": re.compile(
                r"(?<!\d)(?:\+\d{1,3}[-. ]?)?(?:\(\d{3}\)|\d{3})[-. ]\d{3}[-. ]\d{4}(?!\d)"
                r"|(?<!\d)\+\d{1,3}[-. ]?\d{6,12}(?!\d)"
                r"|(?<!\d)\d{10}(?!\d)"
            ),
            "IP_ADDRESS": re.compile(
                r"\b(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}"
                r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\b"
            ),
            "DOB": re.compile(
                r"\b(?:(?:0?[1-9]|[12]\d|3[01])\s+(?:jan|feb|mar|apr|may|jun|jul|"
                r"aug|sep|oct|nov|dec)[a-z]*\s+\d{4}"
                r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+"
                r"(?:0?[1-9]|[12]\d|3[01]),?\s+\d{4}"
                r"|\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))\b",
                re.IGNORECASE,
            ),
            # Anchored on the word "passport" because the identifier shape
            # alone (a letter or two plus digits) matches far too much.
            "PASSPORT": re.compile(
                r"\bpassport\s*(?:number|no\.?|#)?\s*(?:is|:)?\s*"
                r"\b[A-Z]{1,2}\d{6,9}\b",
                re.IGNORECASE,
            ),
            "AWS_KEY": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
            "JWT": re.compile(
                r"\beyJ[a-zA-Z0-9_\-]*\.[a-zA-Z0-9_\-]*\.[a-zA-Z0-9_\-]*\b"
            ),
            "BEARER": re.compile(r"\bBearer\s+[a-zA-Z0-9_\-\.]{20,}\b"),
        }

    def _accepts(self, label: str, value: str) -> bool:
        """Apply a label's validator, if it has one."""
        validator = self.VALIDATORS.get(label)
        if validator is None:
            return True
        return validator(re.sub(r"\D", "", value))

    def mask_text(self, text: str) -> Tuple[str, Dict[str, str]]:
        if not text:
            return text, {}

        vault_mapping: Dict[str, str] = {}
        counters: Dict[str, int] = {}
        sanitized = self._mask_one(text, vault_mapping, counters)
        return sanitized, vault_mapping

    def _mask_one(
        self,
        text: str,
        vault_mapping: Dict[str, str],
        counters: Dict[str, int],
    ) -> str:
        """Mask every pattern in `text`, recording placeholders into
        `vault_mapping` and advancing the shared `counters`."""
        sanitized = text
        for label, pattern in self.patterns.items():
            matches = list(pattern.finditer(sanitized))
            # Process matches in reverse order so character offsets stay valid
            for match in reversed(matches):
                val = match.group(0)
                if not self._accepts(label, val):
                    continue
                counters[label] = counters.get(label, 0) + 1
                token = f"[{label}_{counters[label]}]"
                vault_mapping[token] = val
                sanitized = (
                    sanitized[: match.start()] + token + sanitized[match.end() :]
                )
        return sanitized

    def mask_messages(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        """Mask PII across a full conversation. Placeholder counters are shared
        across messages so tokens stay unique and the merged mapping restores
        every occurrence correctly."""
        vault_mapping: Dict[str, str] = {}
        counters: Dict[str, int] = {}
        masked_messages: List[Dict[str, str]] = []

        for msg in messages:
            content = msg.get("content", "")
            if not content:
                masked_messages.append(
                    {"role": msg.get("role", ""), "content": content}
                )
                continue

            sanitized = self._mask_one(content, vault_mapping, counters)
            masked_messages.append({"role": msg.get("role", ""), "content": sanitized})

        return masked_messages, vault_mapping

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
            sanitized_messages.append(
                {
                    "role": msg.get("role", ""),
                    "content": self.sanitize_text(msg.get("content", "")),
                }
            )
        return sanitized_messages
