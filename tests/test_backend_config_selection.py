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


def test_cloud_config_spans_more_than_one_vendor():
    """Cross-vendor failover is the capability an in-house router cannot
    provide; a cloud config pinned to one vendor gives that up."""
    endpoints = {b["endpoint"].split("/")[2] for b in _cloud_backends()}

    assert len(endpoints) > 1, f"only one vendor configured: {endpoints}"
