"""The vault has to survive being read back in pieces, and being read back
by a model that reformats.

Three failures, all invisible unless you look at what the caller actually
receives:

* one value appearing twice was given two placeholders, so the model was
  told two different people were involved
* `restore_text` matched exactly, so `[email_1]` or a bracket-less `EMAIL_1`
  left the placeholder in the answer
* restore ran per chunk, so a placeholder split as `[EMA` + `IL_1` + `] to`
  was never swapped back and the caller saw the gateway's plumbing

The last one was verified against the running endpoint: the same model
output came back correct without `stream: true` and wrong with it.
"""

import json
import uuid
from typing import Any, AsyncGenerator, Dict

import pytest
from fastapi.testclient import TestClient

import gateway.main as main
from gateway.adapters.base import BaseAdapter, NormalizedResponse
from gateway.policy.pii import PiiVault, VaultRestorer

EMAIL = "jane.doe@corp.com"
OTHER = "bob@other.org"


@pytest.fixture
def vault():
    return PiiVault()


# ── One value, one placeholder ───────────────────────────────────────────


def test_a_repeated_value_keeps_one_placeholder(vault):
    """Two placeholders for one address told the model there were two
    addresses, and nothing that depends on them being the same could be
    answered correctly after that."""
    masked, mapping = vault.mask_text(f"Email {EMAIL} then cc {EMAIL}.")

    assert len(mapping) == 1
    assert masked.count("[EMAIL_1]") == 2


def test_distinct_values_still_get_distinct_placeholders(vault):
    masked, mapping = vault.mask_text(f"Email {EMAIL} and {OTHER}.")

    assert len(mapping) == 2
    assert set(mapping.values()) == {EMAIL, OTHER}


def test_placeholders_are_numbered_in_reading_order(vault):
    """Substitution runs back-to-front to keep offsets valid, which used to
    number the tokens back-to-front with it."""
    masked, _ = vault.mask_text(f"First {EMAIL}, second {OTHER}.")

    assert masked.index("[EMAIL_1]") < masked.index("[EMAIL_2]")


def test_a_value_is_shared_across_messages(vault):
    """The same address in a system prompt and a user turn is one value."""
    messages = [
        {"role": "system", "content": f"The account owner is {EMAIL}."},
        {"role": "user", "content": f"Does {EMAIL} have access?"},
    ]

    masked, mapping = vault.mask_messages(messages)

    assert len(mapping) == 1
    assert all("[EMAIL_1]" in m["content"] for m in masked)


# ── Restore tolerates how a model actually writes ────────────────────────


@pytest.mark.parametrize(
    "written",
    ["[EMAIL_1]", "[email_1]", "[Email_1]", "EMAIL_1", "[EMAIL 1]", "[EMAIL-1]"],
)
def test_restore_accepts_reformatted_placeholders(vault, written):
    _, mapping = vault.mask_text(f"my address is {EMAIL}")

    assert EMAIL in vault.restore_text(f"Sent to {written}.", mapping)


def test_restore_leaves_prose_alone(vault):
    """Accepting a bare `EMAIL 1` would start rewriting ordinary text, so the
    bracket-less form requires the underscore."""
    _, mapping = vault.mask_text(f"my address is {EMAIL}")

    assert vault.restore_text("Send EMAIL 1 to the printer.", mapping) == (
        "Send EMAIL 1 to the printer."
    )


def test_an_unrelated_token_is_not_substituted(vault):
    """A label that was never masked must not be rewritten."""
    _, mapping = vault.mask_text(f"my address is {EMAIL}")

    assert vault.restore_text("See PHONE_1 below.", mapping) == "See PHONE_1 below."


def test_a_longer_number_is_not_matched_by_a_shorter_one(vault):
    """[EMAIL_1] must not match inside [EMAIL_11]."""
    addresses = " ".join(f"user{i}@corp.com" for i in range(12))
    _, mapping = vault.mask_text(addresses)
    assert "[EMAIL_11]" in mapping

    restored = vault.restore_text("[EMAIL_11]", mapping)

    assert restored == mapping["[EMAIL_11]"]


# ── Restoring a response that arrives in pieces ──────────────────────────


def _restore_streamed(vault, mapping, text: str, size: int) -> str:
    r = VaultRestorer(vault, mapping)
    out = "".join(r.feed(text[i : i + size]) for i in range(0, len(text), size))
    return out + r.flush()


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 8, 9, 17, 500])
def test_a_split_placeholder_is_still_restored(vault, size):
    masked, mapping = vault.mask_text(f"Contact {EMAIL} now")

    streamed = _restore_streamed(vault, mapping, masked, size)

    assert streamed == vault.restore_text(masked, mapping)
    assert EMAIL in streamed


@pytest.mark.parametrize("size", [1, 3, 7])
def test_several_split_placeholders_are_restored(vault, size):
    masked, mapping = vault.mask_text(f"Mail {EMAIL}, cc {OTHER}, then {EMAIL} again")

    streamed = _restore_streamed(vault, mapping, masked, size)

    assert streamed.count(EMAIL) == 2
    assert OTHER in streamed
    assert "[EMAIL_" not in streamed


def test_a_stray_bracket_does_not_stall_the_stream(vault):
    """An unclosed bracket in ordinary prose must not hold text forever."""
    _, mapping = vault.mask_text(f"address {EMAIL}")
    r = VaultRestorer(vault, mapping)

    released = r.feed("A list item [ that never closes " + "x" * 80)

    assert released, "text was held behind a bracket that is not a placeholder"


def test_nothing_is_held_when_nothing_was_masked(vault):
    """No mapping means no placeholder can exist, so no reason to buffer."""
    r = VaultRestorer(vault, {})

    assert r.feed("plain text [with brackets]") == "plain text [with brackets]"


# ── Through the real endpoint ────────────────────────────────────────────


class _Echo(BaseAdapter):
    """Repeats the masked prompt back, as a model asked to confirm would."""

    @staticmethod
    def _answer(messages) -> str:
        return f"Sure — I will contact {messages[-1]['content'].split('contact ')[-1]}"

    async def complete(self, messages, **kw) -> NormalizedResponse:
        return NormalizedResponse(
            id=self.id,
            backend_id=self.id,
            model=self.model,
            messages=[{"role": "assistant", "content": self._answer(messages)}],
            prompt_tokens=5,
            completion_tokens=5,
            cost_usd=0.0,
            latency_ms=1.0,
        )

    async def complete_stream(
        self, messages, **kw
    ) -> AsyncGenerator[Dict[str, Any], None]:
        text = self._answer(messages)
        for i in range(0, len(text), 4):  # 4-char pieces, as a tokenizer emits
            yield {
                "id": "c",
                "model": self.model,
                "choices": [{"index": 0, "delta": {"content": text[i : i + 4]}}],
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
    api_key = f"sk-pii-{uuid.uuid4().hex[:8]}"

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
        main.get_state().router.adapters = [
            _Echo(
                {
                    "id": "echo",
                    "provider": "ollama",
                    "model": "echo",
                    "cost_per_1k_prompt": 0.0,
                    "cost_per_1k_completion": 0.0,
                }
            )
        ]
        yield client, api_key


def _ask(client, api_key, stream: bool) -> str:
    main.get_state().prompt_cache.clear()
    body = {
        "model": "auto",
        "max_tokens": 100,
        "stream": stream,
        "messages": [{"role": "user", "content": f"Please contact {EMAIL} today"}],
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    if not stream:
        response = client.post("/v1/chat/completions", headers=headers, json=body)
        return response.json()["choices"][0]["message"]["content"]

    with client.stream(
        "POST", "/v1/chat/completions", headers=headers, json=body
    ) as response:
        raw = "".join(line for line in response.iter_lines())

    text = ""
    for part in raw.split("data: "):
        part = part.strip()
        if not part or part == "[DONE]":
            continue
        try:
            obj = json.loads(part)
        except ValueError:
            continue
        for choice in obj.get("choices") or []:
            text += (choice.get("delta") or {}).get("content") or ""
    return text


def test_a_streamed_placeholder_reaches_the_client_restored(gateway):
    """The original failure: the caller was shown `[EMAIL_1]`."""
    client, api_key = gateway

    text = _ask(client, api_key, stream=True)

    assert EMAIL in text
    assert "[EMAIL_" not in text


def test_streaming_and_non_streaming_agree(gateway):
    client, api_key = gateway

    assert _ask(client, api_key, stream=True) == _ask(client, api_key, stream=False)
