"""Tests for third-party provider profiles (claude_swap.providers)."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from claude_swap.providers import PRESETS, ProviderError, ProviderStore, launch_env


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    config = tmp_path / "claude"
    config.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    return ProviderStore(tmp_path / "backup"), config / "settings.json"


def _add(store, name="mm", **extra):
    store.add(name, {**PRESETS["minimax"], "ANTHROPIC_AUTH_TOKEN": "sk-mm-secret", **extra})


def test_add_requires_key_and_url(env):
    store, _ = env
    with pytest.raises(ProviderError):
        store.add("x", {"ANTHROPIC_AUTH_TOKEN": "k"})
    with pytest.raises(ProviderError):
        store.add("x", {"ANTHROPIC_BASE_URL": "https://e"})
    _add(store)
    with pytest.raises(ProviderError):
        _add(store)  # exists without force


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_store_is_private(env):
    store, _ = env
    _add(store)
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_use_then_off_restores_settings_exactly(env):
    store, settings = env
    original = {"theme": "dark", "env": {"ANTHROPIC_MODEL": "mine", "FOO": "1"}}
    settings.write_text(json.dumps(original))
    _add(store)

    store.use("mm")
    applied = json.loads(settings.read_text())
    assert applied["theme"] == "dark"
    assert applied["env"]["ANTHROPIC_BASE_URL"] == "https://api.minimax.io/anthropic"
    assert applied["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-mm-secret"
    assert applied["env"]["FOO"] == "1"
    assert store.active_name() == "mm"

    assert store.off() == "mm"
    assert json.loads(settings.read_text()) == original
    assert store.off() is None


def test_provider_to_provider_swap_keeps_original_previous(env):
    store, settings = env
    settings.write_text(json.dumps({"env": {"ANTHROPIC_MODEL": "mine"}}))
    _add(store, "a", ONLY_A="1")
    store.add("b", {"ANTHROPIC_BASE_URL": "https://b", "ANTHROPIC_AUTH_TOKEN": "kb",
                    "ANTHROPIC_MODEL": "bm"})

    store.use("a")
    store.use("b")
    applied = json.loads(settings.read_text())["env"]
    assert "ONLY_A" not in applied
    assert applied["ANTHROPIC_MODEL"] == "bm"

    store.off()
    assert json.loads(settings.read_text()) == {"env": {"ANTHROPIC_MODEL": "mine"}}


def test_off_drops_env_it_created(env):
    store, settings = env
    settings.write_text(json.dumps({"theme": "dark"}))
    _add(store)
    store.use("mm")
    store.off()
    assert json.loads(settings.read_text()) == {"theme": "dark"}


def test_refuses_invalid_settings_json(env):
    store, settings = env
    settings.write_text("{not json")
    _add(store)
    with pytest.raises(ProviderError):
        store.use("mm")
    assert settings.read_text() == "{not json"
    assert store.active_name() is None


def test_remove_active_refused(env):
    store, _ = env
    _add(store)
    store.use("mm")
    with pytest.raises(ProviderError):
        store.remove("mm")
    store.off()
    store.remove("mm")
    assert store.load()["providers"] == {}


def test_launch_env_scrubs_other_auth(env):
    out = launch_env(
        {"ANTHROPIC_BASE_URL": "https://e", "ANTHROPIC_AUTH_TOKEN": "k"},
        {"ANTHROPIC_API_KEY": "sk-ant-api03-x", "CLAUDE_CODE_OAUTH_TOKEN": "t", "PATH": "/bin"},
    )
    assert out == {"ANTHROPIC_BASE_URL": "https://e", "ANTHROPIC_AUTH_TOKEN": "k", "PATH": "/bin"}
