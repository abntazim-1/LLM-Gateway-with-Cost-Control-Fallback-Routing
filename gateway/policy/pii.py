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
        seen: Dict[Tuple[str, str], str] = {}
        sanitized = self._mask_one(text, vault_mapping, counters, seen)
        return sanitized, vault_mapping

    def _mask_one(
        self,
        text: str,
        vault_mapping: Dict[str, str],
        counters: Dict[str, int],
        seen: Dict[Tuple[str, str], str],
    ) -> str:
        """Mask every pattern in `text`, recording placeholders into
        `vault_mapping` and advancing the shared `counters`.

        `seen` maps a value to the placeholder already issued for it, so one
        value keeps one placeholder wherever it appears."""
        sanitized = text
        for label, pattern in self.patterns.items():
            matches = [
                m
                for m in pattern.finditer(sanitized)
                if self._accepts(label, m.group(0))
            ]

            # Issue tokens in reading order, so [EMAIL_1] is the first address
            # in the text rather than the last. Substitution then runs in
            # reverse, which is what keeps earlier offsets valid.
            tokens = []
            for match in matches:
                value = match.group(0)
                token = seen.get((label, value))
                if token is None:
                    # A value that has been masked before keeps its original
                    # placeholder. Numbering each occurrence separately told
                    # the model that one address repeated twice was two
                    # different addresses, and no answer that depends on them
                    # being the same could then be right.
                    counters[label] = counters.get(label, 0) + 1
                    token = f"[{label}_{counters[label]}]"
                    seen[(label, value)] = token
                    vault_mapping[token] = value
                tokens.append(token)

            for match, token in zip(reversed(matches), reversed(tokens)):
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
        # Shared across messages too: an address in the system prompt and the
        # same address in the user turn are one value, not two.
        seen: Dict[Tuple[str, str], str] = {}
        masked_messages: List[Dict[str, str]] = []

        for msg in messages:
            content = msg.get("content", "")
            if not content:
                masked_messages.append(
                    {"role": msg.get("role", ""), "content": content}
                )
                continue

            sanitized = self._mask_one(content, vault_mapping, counters, seen)
            masked_messages.append({"role": msg.get("role", ""), "content": sanitized})

        return masked_messages, vault_mapping

    def restore_text(self, text: str, vault_mapping: Dict[str, str]) -> str:
        """Swap placeholders back for the values they stand for.

        An exact string match was too strict: models reformat, and
        `[email_1]` in prose or a bracket-less `EMAIL_1` in a list left the
        caller looking at the gateway's internal plumbing. Case and the
        separator are therefore allowed to vary.

        The bracket-less form still requires the underscore — accepting a
        bare `EMAIL 1` would start rewriting ordinary prose.
        """
        if not text or not vault_mapping:
            return text

        # One pass over the text rather than a replace per token, so a value
        # that has just been restored is never rescanned as if it were
        # another placeholder.
        tokens = sorted(vault_mapping, key=len, reverse=True)
        alternatives = []
        for i, token in enumerate(tokens):
            label, _, number = token.strip("[]").rpartition("_")
            label_re = re.escape(label)
            number_re = re.escape(number)
            alternatives.append(
                rf"(?P<t{i}>\[\s*{label_re}[\s_-]*{number_re}\s*\]"
                rf"|\b{label_re}_{number_re}\b)"
            )

        combined = re.compile("|".join(alternatives), re.IGNORECASE)

        def resolve(match: "re.Match[str]") -> str:
            name = match.lastgroup
            if name is None:
                # Unreachable: every alternative above is a named group. If it
                # ever happens, leave the text as found rather than raising —
                # a visible placeholder beats a failed response.
                return match.group(0)
            return vault_mapping[tokens[int(name[1:])]]

        return combined.sub(resolve, text)


class VaultRestorer:
    """Restores vault placeholders in a response that arrives in pieces.

    Restoring per chunk cannot work: a placeholder split as `[EMA` + `IL_1`
    + `] to` appears in no chunk whole, so the swap never happens and the
    caller is shown `[EMAIL_1]` — the gateway's internal plumbing — where
    their own address should be. The non-streaming path, handed the whole
    string at once, got this right, which is what kept it hidden.

    The fix is to release text only up to the last point a placeholder could
    still be forming, and hold the rest until it completes.

    Bracket-less placeholders (`EMAIL_1`, which `restore_text` also accepts
    because models reformat) are not tracked here: recognising one before it
    is complete would mean holding back on any capitalised word. A model that
    both drops the brackets and is cut mid-token leaves that one placeholder
    unrestored.
    """

    # Longest real placeholder is around 22 characters
    # ("[EMAIL_OBFUSCATED_12]"). Past this a bracket is ordinary text and
    # holding for it would stall the stream.
    MAX_TOKEN_LEN = 64

    def __init__(self, vault: "PiiVault", mapping: Dict[str, str]):
        self._vault = vault
        self._mapping = mapping
        self._pending = ""

    def feed(self, text: str) -> str:
        """Take the next piece; return what can be restored and sent now."""
        if not self._mapping:
            return text  # nothing was masked, so nothing can be split
        self._pending += text
        boundary = self._boundary(self._pending)
        ready, self._pending = self._pending[:boundary], self._pending[boundary:]
        return self._vault.restore_text(ready, self._mapping)

    def flush(self) -> str:
        """Release the held tail once no more text can arrive."""
        if not self._pending:
            return ""
        ready, self._pending = self._pending, ""
        return self._vault.restore_text(ready, self._mapping)

    def _boundary(self, text: str) -> int:
        """How much of *text* is past any placeholder still being written."""
        opened = text.rfind("[")
        if opened == -1 or "]" in text[opened:]:
            return len(text)  # no open bracket, so nothing is half-written
        if len(text) - opened > self.MAX_TOKEN_LEN:
            return len(text)  # too long to be a placeholder; a stray bracket
        return opened


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
