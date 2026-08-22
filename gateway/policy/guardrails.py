import re
from typing import Dict, List, Optional


class GuardrailViolationException(Exception):
    """Exception raised when a request violates prompt safety or content guardrails."""

    pass


class GuardrailsPipeline:
    """
    Content safety and security guardrails pipeline.
    Inspects prompts for prompt injection / jailbreak patterns before sending requests to LLMs.
    """

    def __init__(self):
        self.prompt_injection_patterns = [
            re.compile(
                r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE
            ),
            re.compile(r"override\s+(the\s+)?system\s+prompt", re.IGNORECASE),
            re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.IGNORECASE),
            re.compile(r"jailbreak\s+mode", re.IGNORECASE),
            re.compile(r"act\s+as\s+DAN", re.IGNORECASE),
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
        """Validate input prompt messages against injection/jailbreak rules."""
        for msg in messages:
            content = msg.get("content", "")
            if not content:
                continue

            for pattern in self.prompt_injection_patterns:
                if pattern.search(content):
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
