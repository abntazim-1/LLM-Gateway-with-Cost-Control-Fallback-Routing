"""Measure gateway token-estimate error against real per-model tokenizers.

Requires Ollama running on :11434. Compares the adapter's own
count_prompt_tokens() against the usage figures the backend actually reports,
and reports the per-backend overhead that would minimise the error (useful for
calibrating `token_overhead_per_message` in backends.yaml).
"""

import json
import statistics
import urllib.request

from gateway.adapters.local_vllm_adapter import LocalVLLMAdapter

PROMPTS = [
    ("short english", "Say OK"),
    ("normal english", "Explain what a circuit breaker does in one sentence."),
    ("code", "def f(x):\n    return [i**2 for i in range(x) if i % 3 == 0]"),
    ("json payload", '{"user_id": 88213, "status": "active", "tags": ["a","b"]}'),
    ("non-latin", "নমস্কার, আপনি কেমন আছেন? আজ আবহাওয়া খুব সুন্দর।"),
    ("repeated punct", "!!!???...---===+++"),
]

BACKENDS = [
    {
        "id": "local-ollama-phi3",
        "provider": "ollama",
        "model": "phi3:latest",
        "endpoint": "http://localhost:11434/v1",
        "cost_per_1k_prompt": 0.0001,
        "cost_per_1k_completion": 0.0002,
        "token_overhead_per_message": 7,
    },
    {
        "id": "local-ollama-qwen2.5-0.5b",
        "provider": "ollama",
        "model": "qwen2.5:0.5b",
        "endpoint": "http://localhost:11434/v1",
        "cost_per_1k_prompt": 0.001,
        "cost_per_1k_completion": 0.002,
        "token_overhead_per_message": 26,
    },
]


def real_prompt_tokens(model: str, text: str) -> int:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        "http://localhost:11434/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["usage"]["prompt_tokens"]


for cfg in BACKENDS:
    adapter = LocalVLLMAdapter(cfg)
    print("=" * 74)
    print(
        f"{cfg['model']}  (tokenizer={adapter.tokenizer_name}, "
        f"overhead/msg={adapter.token_overhead_per_message})"
    )
    print("=" * 74)
    print(f"{'case':16} {'estimated':>10} {'actual':>8} {'error':>9}   residual")
    print("-" * 74)
    residuals, errors = [], []
    for label, text in PROMPTS:
        msgs = [{"role": "user", "content": text}]
        est = adapter.count_prompt_tokens(msgs)
        try:
            actual = real_prompt_tokens(cfg["model"], text)
        except Exception as e:
            print(f"{label:16}  error: {e}")
            continue
        err = (est - actual) / actual * 100
        residuals.append(actual - est)
        errors.append(abs(err))
        print(f"{label:16} {est:>10} {actual:>8} {err:>+8.1f}%   {actual - est:+d}")
    if errors:
        need = adapter.token_overhead_per_message + statistics.median(residuals)
        print("-" * 74)
        print(f"  mean |error|: {statistics.mean(errors):.1f}%")
        print(f"  suggested token_overhead_per_message: {need:.0f}")
    print()
