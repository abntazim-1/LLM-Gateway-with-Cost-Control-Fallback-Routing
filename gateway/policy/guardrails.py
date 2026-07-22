import re
from typing import List, Dict, Optional

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
            re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
            re.compile(r"override\s+(the\s+)?system\s+prompt", re.IGNORECASE),
            re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.IGNORECASE),
            re.compile(r"jailbreak\s+mode", re.IGNORECASE),
            re.compile(r"act\s+as\s+DAN", re.IGNORECASE)
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

    def validate_completion(self, completion_text: str) -> None:
        """Validate output completion text against output guardrail rules."""
        if not completion_text:
            return
        # Add output guardrail checks if needed
        pass
