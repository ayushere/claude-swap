"""Third-party provider profiles (MiniMax, OpenRouter, ...).

A provider is not an account slot: it is a set of env vars that points Claude
Code at an Anthropic-compatible endpoint (``ANTHROPIC_BASE_URL`` +
``ANTHROPIC_AUTH_TOKEN``), which bypasses the OAuth login entirely. So it
lives beside the account store, never inside it, and never touches OAuth
credentials.

Two ways to use one:
- ``run``: exec claude with the env set, this terminal only.
- ``use``/``off``: write the env into ``<config_home>/settings.json`` ``env``
  (global), remembering what each key held before so ``off`` restores it.

Persisted to ``<backup_dir>/providers.json`` (0600 — it holds API keys).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from claude_swap.exceptions import ClaudeSwitchError
from claude_swap.fsutil import replace_with_retry
from claude_swap.paths import get_claude_config_home

SCHEMA_VERSION = 1
AUTH_TOKEN_VAR = "ANTHROPIC_AUTH_TOKEN"

_MINIMAX_MODEL = "MiniMax-M3[1m]"


def _minimax(base_url: str) -> dict[str, str]:
    return {
        "ANTHROPIC_BASE_URL": base_url,
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "1000000",
        "ANTHROPIC_MODEL": _MINIMAX_MODEL,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": _MINIMAX_MODEL,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": _MINIMAX_MODEL,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": _MINIMAX_MODEL,
    }


# Per https://platform.minimax.io/docs/token-plan/claude-code
PRESETS: dict[str, dict[str, str]] = {
    "minimax": _minimax("https://api.minimax.io/anthropic"),
    "minimax-cn": _minimax("https://api.minimax.cn/anthropic"),
}

MODEL_VARS = (
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
)


class ProviderError(ClaudeSwitchError):
    """Provider profile problem."""


def mask(token: str) -> str:
    return f"{token[:4]}…{token[-4:]}" if len(token) > 12 else "****"


def claude_settings_path() -> Path:
    return get_claude_config_home() / "settings.json"


def _atomic_write_json(path: Path, data: dict, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}-", suffix=".tmp")
    try:
        if sys.platform != "win32":
            os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=2) + "\n")
        replace_with_retry(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class ProviderStore:
    """Reads and writes ``<backup_dir>/providers.json``.

    Shape: ``{"schemaVersion", "providers": {name: {"env": {...}}},
    "active": {"name", "previous": {var: old value | None}} | None}``.
    """

    def __init__(self, backup_dir: Path):
        self.path = Path(backup_dir) / "providers.json"

    def load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            data = {}
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            # Refuse rather than silently drop stored keys on the next write.
            raise ProviderError(f"{self.path} is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ProviderError(f"{self.path} is not a JSON object")
        data.setdefault("providers", {})
        data.setdefault("active", None)
        return data

    def _save(self, data: dict) -> None:
        data["schemaVersion"] = SCHEMA_VERSION
        if sys.platform != "win32":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            os.chmod(self.path.parent, 0o700)
        _atomic_write_json(self.path, data)

    def get(self, name: str) -> dict:
        provider = self.load()["providers"].get(name)
        if provider is None:
            raise ProviderError(f"No provider named '{name}' (see `cswap provider list`)")
        return provider

    def add(self, name: str, env: dict[str, str], force: bool = False) -> None:
        if not name or name.startswith("-") or not name.replace("-", "").replace("_", "").isalnum():
            raise ProviderError("Provider name must be letters, digits, '-' or '_'")
        if not env.get("ANTHROPIC_BASE_URL"):
            raise ProviderError("A provider needs a base URL (--base-url or --preset)")
        if not env.get(AUTH_TOKEN_VAR):
            raise ProviderError("A provider needs an API key")
        data = self.load()
        if name in data["providers"] and not force:
            raise ProviderError(f"Provider '{name}' exists; pass --force to replace it")
        data["providers"][name] = {"env": env}
        self._save(data)

    def remove(self, name: str) -> None:
        data = self.load()
        if name not in data["providers"]:
            raise ProviderError(f"No provider named '{name}'")
        if (data["active"] or {}).get("name") == name:
            raise ProviderError(f"'{name}' is active; run `cswap provider off` first")
        del data["providers"][name]
        self._save(data)

    def active_name(self) -> str | None:
        return (self.load()["active"] or {}).get("name")

    def use(self, name: str) -> None:
        """Write ``name``'s env into Claude Code's settings.json (global)."""
        data = self.load()
        env = self.get(name)["env"]
        active = data["active"]
        settings_path = claude_settings_path()
        settings = _read_settings(settings_path)
        settings_env = settings.setdefault("env", {})
        if not isinstance(settings_env, dict):
            raise ProviderError(f"'env' in {settings_path} is not an object")

        # Keep the pre-provider values from the first `use`; a provider-to-
        # provider swap must not record the old provider's vars as "previous".
        previous = dict(active["previous"]) if active else {}
        if active:
            old_env = data["providers"].get(active["name"], {}).get("env", {})
            for var in old_env:
                if var not in env:
                    _restore(settings_env, var, previous.pop(var, None))
        for var, value in env.items():
            if var not in previous:
                previous[var] = settings_env.get(var)
            settings_env[var] = value

        _write_settings(settings_path, settings)
        data["active"] = {"name": name, "previous": previous}
        self._save(data)

    def off(self) -> str | None:
        """Restore settings.json to its pre-provider state. Returns the name cleared."""
        data = self.load()
        active = data["active"]
        if not active:
            return None
        settings_path = claude_settings_path()
        settings = _read_settings(settings_path)
        settings_env = settings.get("env")
        if isinstance(settings_env, dict):
            for var, old in active["previous"].items():
                _restore(settings_env, var, old)
            if not settings_env:
                settings.pop("env")
            _write_settings(settings_path, settings)
        data["active"] = None
        self._save(data)
        return active["name"]


def _restore(env: dict, var: str, old: str | None) -> None:
    if old is None:
        env.pop(var, None)
    else:
        env[var] = old


def _read_settings(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ProviderError(f"Refusing to edit {path}: not valid JSON ({e})") from e
    if not isinstance(data, dict):
        raise ProviderError(f"Refusing to edit {path}: not a JSON object")
    return data


def _write_settings(path: Path, settings: dict) -> None:
    # settings.json now carries an API key: keep the user's mode but drop
    # group/other access.
    try:
        mode = path.stat().st_mode & 0o700
    except FileNotFoundError:
        mode = 0o600
    _atomic_write_json(path, settings, mode=mode or 0o600)


def launch_env(provider_env: dict[str, str], base: dict[str, str]) -> dict[str, str]:
    """Env for a one-terminal provider launch: scrub other auth overrides."""
    from claude_swap.session import AUTH_OVERRIDE_ENV_VARS

    env = {k: v for k, v in base.items() if k not in AUTH_OVERRIDE_ENV_VARS}
    env.update(provider_env)
    return env
