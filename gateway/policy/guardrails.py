import json
import re
from typing import Any, Dict, List, Optional, Tuple

from gateway.policy.normalize import matching_variants


class GuardrailViolationException(Exception):
    """Exception raised when a request violates prompt safety or content guardrails."""

    pass


# Verbs used to tell a model to abandon its instructions, and the things it
# gets told to abandon. Kept as separate alternation groups and combined so
# that adding a synonym to either side covers every phrasing on the other —
# enumerating whole sentences instead is what let "disregard all previous
# instructions" through while "ignore" was blocked.
_DISCARD_VERBS = (
    r"ignor\w*|disregard\w*|forget|overrid\w*|bypass\w*|discard\w*|"
    r"skip|drop|delete|erase|abandon|set\s+aside|pay\s+no\s+attention\s+to"
)
_INSTRUCTION_NOUNS = (
    r"instruction|directive|prompt|rule|guideline|command|order|constraint|"
    r"restriction|guardrail|polic\w+|training|programming"
)
_PRIOR_QUALIFIERS = (
    r"previous|prior|above|earlier|preceding|foregoing|initial|original|"
    r"system|all|any|your|the"
)


# ── Streaming output screening ───────────────────────────────────────────
# The first characters of every secret in `output_secret_patterns`. A secret
# cannot exist without one, so these are the only places a match can begin —
# which is what makes it safe to release everything else immediately instead
# of buffering the whole response.
_SECRET_STARTERS = re.compile(
    r"sk-|AKIA|ASIA|gh[pousr]_|xox[baprs]-|eyJ|Bearer\s", re.IGNORECASE
)
# The longest starter ("Bearer ") is 7 characters. Holding back 6 makes it
# impossible for a starter to be split across an emit boundary and missed,
# which would defeat the scheme entirely.
_MIN_HOLDBACK = 6
# How far back a secret that is still arriving may have begun. JWTs are the
# long case; beyond this a starter is treated as ordinary text.
_MAX_SECRET_SPAN = 512


class GuardrailsPipeline:
    """
    Content safety and security guardrails pipeline.
    Inspects prompts for prompt injection / jailbreak patterns before sending requests to LLMs.

    Input screening normalizes text first (see `normalize_for_matching`) so
    zero-width characters, homoglyphs, and leetspeak cannot slip a known
    pattern through. Pattern matching still only covers English phrasings it
    has seen — a paraphrase in another language reads as different text, which
    the eval set records as a known gap rather than papering over.

    Output screening must be driven through `StreamingOutputFilter` when the
    response is streamed; calling the per-chunk methods directly cannot work,
    for the reasons documented on that class.
    """

    def __init__(self):
        self.prompt_injection_patterns = [
            # "ignore/disregard/forget [all] [previous] instructions"
            re.compile(
                rf"\b({_DISCARD_VERBS})\b[^.!?\n]{{0,40}}?"
                rf"\b({_PRIOR_QUALIFIERS})\b[^.!?\n]{{0,20}}?"
                rf"\b({_INSTRUCTION_NOUNS})s?\b",
                re.IGNORECASE,
            ),
            # Same idea without a qualifier: "ignore your programming",
            # "forget everything you were told".
            re.compile(
                rf"\b({_DISCARD_VERBS})\b[^.!?\n]{{0,30}}?"
                rf"\b(everything|anything|all)\b[^.!?\n]{{0,30}}?"
                r"\b(you\s+were\s+(told|instructed|given)|before|prior|earlier)\b",
                re.IGNORECASE,
            ),
            # Role/mode hijacking.
            re.compile(
                r"\byou\s+are\s+now\s+(in\s+)?(developer|debug|god|admin|"
                r"unrestricted|unfiltered|dan)\s*(mode)?\b",
                re.IGNORECASE,
            ),
            re.compile(r"\bjailbreak\s*(mode)?\b", re.IGNORECASE),
            re.compile(r"\bact\s+as\s+(dan|an?\s+unrestricted)\b", re.IGNORECASE),
            re.compile(
                r"\b(enable|activate|enter|switch\s+to)\s+"
                r"(developer|debug|god|dan|unrestricted|unfiltered)\s*mode\b",
                re.IGNORECASE,
            ),
            # System-prompt extraction: asking the model to disclose or repeat
            # its instructions.
            re.compile(
                rf"\b(reveal|show|print|output|repeat|display|tell\s+me|"
                rf"what\s+(is|are|were))\b[^.!?\n]{{0,40}}?"
                rf"\b(your|the)\b[^.!?\n]{{0,20}}?"
                rf"\b(system\s+prompt|{_INSTRUCTION_NOUNS}s?|hidden\s+\w+)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\brepeat\s+(the\s+)?(text|words|everything)\s+above\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bwhat\s+(were|was)\s+you\s+(told|instructed|programmed)\b"
                r"[^.!?\n]{0,30}\bnot\s+to\b",
                re.IGNORECASE,
            ),
        ]
        # Secrets a model must never echo back to the client. These are
        # redacted from the completion rather than blocking the response.
        self.output_secret_patterns = {
            "OPENAI_KEY": re.compile(r"\bsk-[a-zA-Z0-9_\-]{20,}\b"),
            "AWS_KEY": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
            "GITHUB_TOKEN": re.compile(r"\bgh[pousr]_[a-zA-Z0-9]{30,}\b"),
            "SLACK_TOKEN": re.compile(r"\bxox[baprs]-[a-zA-Z0-9\-]{10,}\b"),
            "JWT": re.compile(
                r"\beyJ[a-zA-Z0-9_\-]*\.[a-zA-Z0-9_\-]*\.[a-zA-Z0-9_\-]*\b"
            ),
            "BEARER": re.compile(r"\bBearer\s+[a-zA-Z0-9_\-\.]{20,}\b"),
        }
        # System-prompt / hidden-instruction regurgitation. These indicate the
        # model leaked its instructions and the response is rejected outright.
        self.output_leak_patterns = [
            re.compile(
                r"\bmy\s+system\s+prompt\s+(is|was|says|contains)\b", re.IGNORECASE
            ),
            re.compile(
                r"\b(?:my|the)\s+(?:internal|hidden|original|secret)\s+instructions\s+(?:are|were)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\bas\s+part\s+of\s+my\s+(?:system\s+)?instructions\b", re.IGNORECASE
            ),
            re.compile(
                r"\bi\s+was\s+(?:instructed|told)\s+to\s+(?:never|always)\b",
                re.IGNORECASE,
            ),
        ]

    def validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """Validate input prompt messages against injection/jailbreak rules.

        Matching runs against a normalized copy so obfuscated spellings cannot
        evade a pattern; the caller's original text is untouched."""
        for msg in messages:
            content = msg.get("content", "")
            if not content:
                continue

            candidates = matching_variants(content)
            for pattern in self.prompt_injection_patterns:
                if any(pattern.search(c) for c in candidates):
                    raise GuardrailViolationException(
                        f"Prompt Guardrail Violation: Potential prompt injection / jailbreak pattern detected: '{pattern.pattern}'"
                    )

    def sanitize_completion(self, completion_text: str) -> str:
        """Redact leaked secrets (API keys, tokens, JWTs) from completion text.

        Redaction happens on the masked text (before PII vault restore) so a
        user's own PII echoed back via vault placeholders is preserved while
        newly generated secrets are stripped."""
        if not completion_text:
            return completion_text
        for label, pattern in self.output_secret_patterns.items():
            completion_text = pattern.sub(f"[{label}_REDACTED]", completion_text)
        return completion_text

    @staticmethod
    def json_requested(kwargs: Dict[str, Any]) -> bool:
        """Whether the caller asked for JSON output."""
        response_format = kwargs.get("response_format") or {}
        return bool(
            (
                isinstance(response_format, dict)
                and response_format.get("type") == "json_object"
            )
            or kwargs.get("json_mode", False)
        )

    @staticmethod
    def repair_json(completion_text: str) -> Tuple[str, bool]:
        """Return (text, is_valid_json), stripping markdown fences first.

        Backends advertising `json_mode` were routed to on that basis but the
        output was never checked, so a reply of prose — or the very common
        ```json fenced block, which is not itself valid JSON — was passed
        through as success. Unwrapping a fence is a safe, deterministic
        repair; anything still unparseable is reported rather than fixed.
        """
        if not completion_text:
            return completion_text, False

        text = completion_text.strip()
        try:
            json.loads(text)
            return text, True
        except ValueError:
            pass

        fenced = re.match(
            r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", text, re.DOTALL | re.IGNORECASE
        )
        if fenced:
            inner = fenced.group(1).strip()
            try:
                json.loads(inner)
                return inner, True
            except ValueError:
                return completion_text, False

        return completion_text, False

    def emit_boundary(self, text: str) -> int:
        """How much of *text* can be released without splitting a secret.

        Redaction is a whole-string operation: `sk-abc...` arriving as
        `sk-a` + `bc...` matches neither piece. Streaming therefore has to
        hold back any tail that could still grow into a secret, and release
        everything before it.

        Returns an index into *text*; the caller emits up to it and keeps the
        rest for the next chunk.
        """
        n = len(text)
        # Never release the last few characters: a starter split across the
        # boundary would never be recognised on either side of it.
        boundary = max(0, n - _MIN_HOLDBACK)

        window = max(0, n - _MAX_SECRET_SPAN)
        for starter in _SECRET_STARTERS.finditer(text, window):
            start = starter.start()
            if start >= boundary:
                break
            tail = text[start:]
            # A secret that has already matched in full, with text after it,
            # has stopped growing.
            finished = None
            for pattern in self.output_secret_patterns.values():
                match = pattern.match(tail)
                if match and match.end() < len(tail):
                    finished = match
                    break

            # Release it only if the whole match sits inside the prefix being
            # emitted. Otherwise the cut would land inside the secret, and the
            # half left behind would no longer look like one — which is how a
            # trailing character used to escape redaction.
            if finished and start + finished.end() <= boundary:
                continue

            boundary = start

        return boundary

    def validate_completion(self, completion_text: str) -> None:
        """Validate output completion text against output guardrail rules.

        Raises GuardrailViolationException when the model regurgitates its
        system prompt or hidden instructions."""
        if not completion_text:
            return
        for pattern in self.output_leak_patterns:
            if pattern.search(completion_text):
                raise GuardrailViolationException(
                    f"Output Guardrail Violation: completion leaked system instructions (pattern: '{pattern.pattern}')"
                )


class StreamingOutputFilter:
    """Applies the output guardrails to a response that arrives in pieces.

    Both output guardrails are whole-string operations, and streaming gives
    them neither the whole string nor a second chance:

    * secret redaction was run per chunk, so a key arriving as `sk-pro` +
      `j-AbCd` + `EfGh` matched nothing and reassembled intact in the client
    * leak validation ran after the last chunk, by which point every chunk
      had already been sent — it could only decline to cache what the client
      had already read

    This class holds back the smallest tail that could still grow into a
    secret (see `emit_boundary`) and checks the leak patterns against the
    unsent text as well as the sent text, so a leak recognised while it is
    still pending never reaches the client at all.

    Detection cannot be made perfect on a stream: a leak phrase spanning the
    boundary may be partly released before it is recognisable. What stopping
    at that point protects is the payload after the phrase, which is the part
    that matters.
    """

    def __init__(self, pipeline: GuardrailsPipeline):
        self._pipeline = pipeline
        self._pending = ""
        self._released = ""
        self.leaked = False
        self.redacted = False

    def feed(self, delta: str) -> str:
        """Take the next piece; return only what is now safe to send."""
        if self.leaked:
            return ""
        self._pending += delta

        if self._is_leak(self._released + self._pending):
            # Caught before release: drop the tail and stop the stream.
            self.leaked = True
            self._pending = ""
            return ""

        boundary = self._pipeline.emit_boundary(self._pending)
        return self._release(self._pending[:boundary], boundary)

    def flush(self) -> str:
        """Release the held tail once no more text can arrive."""
        if self.leaked:
            return ""
        if self._is_leak(self._released + self._pending):
            self.leaked = True
            self._pending = ""
            return ""
        return self._release(self._pending, len(self._pending))

    def _release(self, raw: str, consumed: int) -> str:
        safe = self._pipeline.sanitize_completion(raw)
        if safe != raw:
            self.redacted = True
        self._pending = self._pending[consumed:]
        self._released += safe
        return safe

    def _is_leak(self, text: str) -> bool:
        try:
            self._pipeline.validate_completion(text)
            return False
        except GuardrailViolationException:
            return True

    @property
    def text(self) -> str:
        """Everything seen so far, redacted — for billing and cache decisions."""
        return self._released + self._pending
