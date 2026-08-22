"""Regression guard for streaming cost accounting.

Streams carry no `usage` object, and the gateway used to bill them by
reusing the pre-flight reservation — which is sized to `max_tokens`. That
charged every stream its ceiling no matter how little it generated (measured
at ~200x on a one-word answer). These tests pin the corrected behaviour:
streams bill what they actually produced.
"""

import time
import uuid
from typing import Any, AsyncGenerator, Dict, List

import pytest
from fastapi.testclient import TestClient

import gateway.main as main
from gateway.adapters.base import BaseAdapter, NormalizedResponse

MAX_TOKENS_CEILING = 4000
STREAMED_WORDS = ["yes"]


class ShortStreamAdapter(BaseAdapter):
    """Streams a deliberately tiny response regardless of max_tokens."""

    async def complete(self, messages, **kwargs) -> NormalizedResponse: ...

    async def complete_stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        for word in STREAMED_WORDS:
            yield {
                "id": "chatcmpl-test",
                "model": self.model,
                "choices": [{"index": 0, "delta": {"content": word}}],
            }

    async def health_check(self) -> bool:
        return True


@pytest.fixture
def client_and_key(monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key")
    api_key = f"sk-stream-cost-{uuid.uuid4().hex[:8]}"

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
        state = main.get_state()
        state.router.adapters = [
            ShortStreamAdapter(
                {
                    "id": "test-stream-backend",
                    "provider": "ollama",
                    "model": "test-model",
                    "cost_per_1k_prompt": 1.0,
                    "cost_per_1k_completion": 1.0,
                }
            )
        ]
        state.prompt_cache.clear()
        yield client, api_key


def _billed_cost(client, api_key, timeout_s: float = 5.0) -> float:
    """Read this key's ledger row, polling because writes go through the
    async ledger queue and are not visible synchronously."""
    deadline = time.time() + timeout_s
    while True:
        rows = client.get(
            "/admin/requests?limit=50", headers={"X-Admin-Token": "test-admin-key"}
        ).json()
        mine = [r for r in rows if r["api_key"] == api_key]
        if mine:
            return mine[0]["cost_usd"]
        if time.time() > deadline:
            raise AssertionError("no ledger row recorded for this request")
        time.sleep(0.05)


def _stream_once(client, api_key, content="Reply with one word"):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "messages": [{"role": "user", "content": content}],
            "stream": True,
            "max_tokens": MAX_TOKENS_CEILING,
        },
    ) as r:
        assert r.status_code == 200
        for _ in r.iter_lines():
            pass


def test_stream_is_not_billed_at_the_max_tokens_ceiling(client_and_key):
    client, api_key = client_and_key
    _stream_once(client, api_key)

    billed = _billed_cost(client, api_key)
    ceiling_cost = MAX_TOKENS_CEILING / 1000.0 * 1.0  # completion rate is 1.0/1k

    # The old bug billed exactly the ceiling. One word must cost far less.
    assert billed < ceiling_cost / 100, (
        f"stream billed {billed} against a ceiling cost of {ceiling_cost} — "
        "this is the max_tokens over-billing regression"
    )


def test_stream_cost_reflects_generated_tokens(client_and_key):
    client, api_key = client_and_key
    _stream_once(client, api_key)

    billed = _billed_cost(client, api_key)
    adapter = main.get_state().router.adapters[0]
    prompt_tokens = adapter.count_prompt_tokens(
        [{"role": "user", "content": "Reply with one word"}]
    )
    completion_tokens = max(1, adapter.count_completion_tokens("".join(STREAMED_WORDS)))
    expected = (prompt_tokens + completion_tokens) / 1000.0  # both rates are 1.0/1k

    assert billed == pytest.approx(expected, rel=0.01)
