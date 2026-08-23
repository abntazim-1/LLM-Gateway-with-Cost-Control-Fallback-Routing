"""Backend set selection via BACKENDS_CONFIG_PATH.

A deployment runs different backends than the repository ships — cloud
providers instead of local models. That switch has to be an environment
change, not an edit to a tracked file, or deploying means carrying a local
diff forever. These tests pin the override and validate the cloud config the
repo ships, so a broken one is caught here rather than on a deployed instance.
"""

import os

import pytest
import yaml

from gateway.config_manager import ConfigManager

CLOUD_CONFIG = "configs/backends.cloud.yaml"


@pytest.fixture(autouse=True)
def _provider_keys(monkeypatch):
    # Adapters read credentials at construction; values are irrelevant here.
    for var in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(var, f"test-{var.lower()}")


def test_defaults_to_the_repository_backends(monkeypatch):
    monkeypatch.delenv("BACKENDS_CONFIG_PATH", raising=False)

    ids = [a.id for a in ConfigManager("configs").load_backends()]

    assert ids, "default backend config failed to load"
    assert all("ollama" in i for i in ids), f"expected local backends, got {ids}"


def test_override_selects_a_different_backend_set(monkeypatch):
    monkeypatch.setenv("BACKENDS_CONFIG_PATH", CLOUD_CONFIG)

    ids = [a.id for a in ConfigManager("configs").load_backends()]

    assert ids
    assert not any("ollama" in i for i in ids), f"override ignored, got {ids}"


def test_hot_reload_honours_the_same_override(monkeypatch):
    """Startup and /admin/reload-config must resolve the same file.

    If reload fell back to the default path, reloading a deployed gateway
    would silently swap it onto the repository's local backends.
    """
    monkeypatch.setenv("BACKENDS_CONFIG_PATH", CLOUD_CONFIG)

    first = [a.id for a in ConfigManager("configs").load_backends()]
    second = [a.id for a in ConfigManager("configs").load_backends()]

    assert first == second
    assert not any("ollama" in i for i in first)


# ── The shipped cloud config must be usable ──────────────────────────────


def _cloud_backends():
    with open(CLOUD_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)["backends"]


def test_cloud_config_declares_everything_routing_needs():
    for backend in _cloud_backends():
        for field in (
            "id",
            "provider",
            "model",
            "endpoint",
            "cost_per_1k_prompt",
            "cost_per_1k_completion",
            "capability_tier",
            "context_length",
        ):
            assert field in backend, f"{backend.get('id')} is missing {field}"


def test_cloud_config_never_inlines_a_credential():
    """Keys must be environment references, not literals."""
    for backend in _cloud_backends():
        for key in backend.get("api_keys", []):
            assert key.startswith("${") and key.endswith(
                "}"
            ), f"{backend['id']} inlines a credential: {key!r}"


def test_cloud_config_prices_capability_above_the_cheap_tier():
    """Tier 1 must genuinely be the cheapest.

    Escalation ranks on capability_tier, so a tier ordering that disagrees
    with cost means 'escalate to the better model' spends more without the
    cheap tier ever being the default — or, worse, escalates downward.
    """
    backends = _cloud_backends()
    cheapest = min(backends, key=lambda b: b["cost_per_1k_prompt"])
    lowest_tier = min(b["capability_tier"] for b in backends)

    assert (
        cheapest["capability_tier"] == lowest_tier
    ), f"{cheapest['id']} is cheapest but is not the lowest tier"


def test_cloud_config_has_something_to_route_between():
    """At least two distinct capability tiers must be active.

    A single-tier config makes every routing strategy degenerate — there is
    nothing to escalate to, so cost_first, complexity and cascade all collapse
    to the same backend. Cross-vendor failover is stronger still but needs a
    second provider's credential, so it ships commented out rather than
    breaking a setup that only has one key.
    """
    tiers = {b["capability_tier"] for b in _cloud_backends()}

    assert len(tiers) > 1, f"only one capability tier active: {tiers}"


# ── Credential isolation ─────────────────────────────────────────────────


def _openai_adapter(**overrides):
    from gateway.adapters.openai_adapter import OpenAIAdapter

    cfg = {
        "id": "b",
        "model": "m",
        "endpoint": "https://api.example.com/v1",
        "cost_per_1k_prompt": 0.0,
        "cost_per_1k_completion": 0.0,
    }
    cfg.update(overrides)
    return OpenAIAdapter(cfg)


def test_backend_keys_are_not_mixed_with_the_global_openai_key(monkeypatch):
    """A backend's own credential must be the only one it uses.

    Both were previously pooled together, so a backend configured for an
    OpenAI-compatible provider round-robined between its own key and the
    global OpenAI one — sending roughly half its traffic upstream with a
    credential for a different vendor. Requests failed intermittently and the
    backend looked flaky rather than misconfigured.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-global-should-not-be-used")
    adapter = _openai_adapter(api_keys=["gsk-backend-own-key"])

    used = {adapter._get_active_key() for _ in range(8)}

    assert used == {"gsk-backend-own-key"}, f"leaked global key: {used}"


def test_falls_back_to_the_env_key_when_backend_declares_none(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-environment")

    assert _openai_adapter()._get_active_key() == "sk-from-environment"


def test_health_check_probes_with_the_key_requests_will_use(monkeypatch):
    """Health checks previously read the env var directly while completions
    authenticated from the pool, so a backend using its own credential
    reported unhealthy forever while serving traffic perfectly — and the
    health loop then tripped its breaker and routed around a working
    backend."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-global")
    adapter = _openai_adapter(api_keys=["gsk-backend-own-key"])

    seen = {}

    class _Resp:
        status_code = 200

    async def _fake_get(url, headers=None, timeout=None):
        seen["auth"] = headers["Authorization"]
        return _Resp()

    monkeypatch.setattr(adapter.client, "get", _fake_get)

    import asyncio

    assert asyncio.run(adapter.health_check()) is True
    assert seen["auth"] == "Bearer gsk-backend-own-key"
