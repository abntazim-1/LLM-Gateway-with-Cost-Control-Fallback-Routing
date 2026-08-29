"""A cache hit has to be the fast path, and has to say the same thing.

Replaying a cached stream paced itself at 10ms a word to look generated,
which made a 400-word hit take ~5s against ~0.17s for the backend call it
was avoiding — thirty times slower than not caching at all. The same loop
restored PII word by word, so any placeholder containing a space was never
swapped back.

Both were measured through the running endpoint before being fixed.
"""

import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict

import pytest
from fastapi.testclient import TestClient

import gateway.main as main
from gateway.adapters.base import BaseAdapter, NormalizedResponse
from gateway.policy.cache import PromptCache

EMAIL = "jane.doe@corp.com"
WORDS = 300
# What the old per-word pacing would have cost for this response. The test
# bound sits well under it so a reintroduced sleep fails loudly, without
# making the test sensitive to ordinary machine noise.
OLD_PACING_SEC = WORDS * 0.01


# ── The cache must not hand out its own state ────────────────────────────


def test_get_returns_a_copy_the_caller_cannot_corrupt():
    """Callers restore PII into the response they are given. Doing that to
    the stored entry would bake one caller's private values into what every
    later caller receives."""
    cache = PromptCache()
    messages = [{"role": "user", "content": "hello"}]
    cache.set(messages, {}, {"choices": [{"message": {"content": "[EMAIL_1]"}}]})

    first = cache.get(messages, {})
    first["choices"][0]["message"]["content"] = "alice@corp.com"
    second = cache.get(messages, {})

    assert second["choices"][0]["message"]["content"] == "[EMAIL_1]"


def test_set_copies_so_later_edits_do_not_reach_the_entry():
    cache = PromptCache()
    messages = [{"role": "user", "content": "hello"}]
    response = {"choices": [{"message": {"content": "original"}}]}
    cache.set(messages, {}, response)

    response["choices"][0]["message"]["content"] = "mutated afterwards"

    assert cache.get(messages, {})["choices"][0]["message"]["content"] == "original"


# ── Through the real endpoint ────────────────────────────────────────────


class _Fake(BaseAdapter):
    answer = ""

    async def complete(self, messages, **kw) -> NormalizedResponse:
        return NormalizedResponse(
            id=self.id,
            backend_id=self.id,
            model=self.model,
            messages=[{"role": "assistant", "content": self.answer}],
            prompt_tokens=5,
            completion_tokens=5,
            cost_usd=0.0,
            latency_ms=1.0,
        )

    async def complete_stream(
        self, messages, **kw
    ) -> AsyncGenerator[Dict[str, Any], None]:
        for word in self.answer.split(" "):
            yield {
                "id": "c",
                "model": self.model,
                "choices": [{"index": 0, "delta": {"content": word + " "}}],
            }
        yield {
            "id": "c",
            "model": self.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }

    async def health_check(self) -> bool:
        return True


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")
    api_key = f"sk-cache-{uuid.uuid4().hex[:8]}"

    with TestClient(main.app) as client:
        client.patch(
            f"/admin/budgets/{api_key}",
            headers={"X-Admin-Token": "test-admin-key"},
            json={
                "daily_limit_usd": 1000.0,
                "monthly_limit_usd": 1000.0,
                "requests_per_minute": 1000,
            },
        )

        def serve(text: str):
            _Fake.answer = text
            main.get_state().router.adapters = [
                _Fake(
                    {
                        "id": "fake",
                        "provider": "ollama",
                        "model": "fake",
                        "cost_per_1k_prompt": 0.0,
                        "cost_per_1k_completion": 0.0,
                    }
                )
            ]
            main.get_state().prompt_cache.clear()

        yield client, api_key, serve


def _stream(client, api_key, prompt):
    started = time.perf_counter()
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "auto",
            "stream": True,
            "max_tokens": 600,
            "messages": [{"role": "user", "content": prompt}],
        },
    ) as response:
        raw = "".join(line for line in response.iter_lines())
    elapsed = time.perf_counter() - started

    text, usage = "", None
    for part in raw.split("data: "):
        part = part.strip()
        if not part or part == "[DONE]":
            continue
        try:
            obj = json.loads(part)
        except ValueError:
            continue
        if obj.get("usage"):
            usage = obj["usage"]
        for choice in obj.get("choices") or []:
            text += (choice.get("delta") or {}).get("content") or ""
    return text, usage, elapsed


def test_a_cache_hit_is_not_paced_like_a_generated_answer(gateway):
    """The point of a cache is speed. Pacing the replay made the hit slower
    than the backend call it replaced."""
    client, api_key, serve = gateway
    serve(" ".join(["word"] * WORDS))
    prompt = "a long answer please"

    _stream(client, api_key, prompt)  # populate
    _, _, hit = _stream(client, api_key, prompt)  # replay

    assert (
        hit < OLD_PACING_SEC / 3
    ), f"cache hit took {hit:.2f}s; pacing would cost {OLD_PACING_SEC:.2f}s"


def test_a_cached_replay_says_the_same_thing_as_a_live_stream(gateway):
    client, api_key, serve = gateway
    answer = "Paris is the capital of France and it is quite pleasant."
    serve(answer)
    prompt = "capital of france"

    live, _, _ = _stream(client, api_key, prompt)
    replayed, _, _ = _stream(client, api_key, prompt)

    assert replayed == live == answer + " "


def test_a_cached_replay_restores_a_placeholder_containing_a_space(gateway):
    """Restoring word by word could not swap back `[EMAIL 1]`, which the
    tolerant restore otherwise accepts."""
    client, api_key, serve = gateway
    serve("Contact [EMAIL_1] and also [EMAIL 1] today")
    prompt = f"please contact {EMAIL}"

    _stream(client, api_key, prompt)  # populate
    replayed, _, _ = _stream(client, api_key, prompt)  # from cache

    assert replayed.count(EMAIL) == 2
    assert "[EMAIL" not in replayed


def test_a_cached_replay_reports_usage(gateway):
    """A client reading token counts should get the same shape from a hit as
    from a miss."""
    client, api_key, serve = gateway
    serve("a short answer")
    prompt = "something short"

    _stream(client, api_key, prompt)
    _, usage, _ = _stream(client, api_key, prompt)

    assert usage and usage.get("prompt_tokens")


def test_the_stored_entry_stays_masked_for_the_next_caller(gateway):
    """Two callers share one entry and each restore their own values, so the
    entry itself must never be restored in place."""
    client, api_key, serve = gateway
    serve("Reply to [EMAIL_1] please")

    first, _, _ = _stream(client, api_key, f"mail {EMAIL}")
    second, _, _ = _stream(client, api_key, "mail someone.else@other.org")

    assert EMAIL in first
    assert "someone.else@other.org" in second
    assert EMAIL not in second, "one caller's address reached another"
