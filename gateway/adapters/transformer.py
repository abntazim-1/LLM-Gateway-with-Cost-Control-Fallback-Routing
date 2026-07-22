from typing import List, Dict, Any

class ParameterTransformer:
    """Utility class to normalize OpenAI-compatible payloads across provider APIs."""

    @staticmethod
    def openai_to_anthropic(messages: List[Dict[str, Any]], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translates OpenAI chat messages and completion kwargs into Anthropic Messages API format.
        - Extracts system prompt(s).
        - Ensures strictly alternating user/assistant roles.
        - Translates parameters like max_tokens and stop sequences.
        """
        system_prompts = []
        anthropic_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                if content:
                    system_prompts.append(content)
            elif role in ("user", "assistant"):
                # Anthropic does not allow back-to-back messages of the same role. Merge if necessary.
                if anthropic_messages and anthropic_messages[-1]["role"] == role:
                    anthropic_messages[-1]["content"] += f"\n\n{content}"
                else:
                    anthropic_messages.append({"role": role, "content": content})

        system_str = "\n\n".join(system_prompts) if system_prompts else ""
        
        # Anthropic requires at least one user message
        if not anthropic_messages:
            anthropic_messages.append({"role": "user", "content": "."})

        # Map kwargs
        supported = {}
        if "max_tokens" in kwargs and isinstance(kwargs["max_tokens"], int):
            supported["max_tokens"] = kwargs["max_tokens"]
        else:
            supported["max_tokens"] = 1024
            
        if "temperature" in kwargs and isinstance(kwargs["temperature"], (int, float)):
            # Anthropic temperature range is 0.0 to 1.0
            supported["temperature"] = max(0.0, min(1.0, float(kwargs["temperature"])))
            
        if "top_p" in kwargs and isinstance(kwargs["top_p"], (int, float)):
            supported["top_p"] = max(0.0, min(1.0, float(kwargs["top_p"])))
            
        if "stop" in kwargs:
            stop_val = kwargs["stop"]
            if isinstance(stop_val, str):
                supported["stop_sequences"] = [stop_val]
            elif isinstance(stop_val, list):
                supported["stop_sequences"] = [str(s) for s in stop_val]

        payload = {
            "messages": anthropic_messages,
            **supported
        }
        if system_str:
            payload["system"] = system_str
            
        return payload

    @staticmethod
    def openai_clean_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filters out non-standard or internal gateway parameters from OpenAI-compatible payloads."""
        allowed_keys = {
            "temperature", "max_tokens", "top_p", "presence_penalty", 
            "frequency_penalty", "stop", "user", "seed", "response_format"
        }
        return {k: v for k, v in kwargs.items() if k in allowed_keys and v is not None}
