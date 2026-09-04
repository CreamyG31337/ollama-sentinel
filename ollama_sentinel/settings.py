"""User-toggleable settings, persisted outside `.env`.

`.env` is deployment config: it is edited by hand, lives in the repo folder, and
is how the machine is provisioned. Toggles someone flips in the GUI need
somewhere else to live, or the app would have to rewrite a file the user owns
and lose their comments and ordering doing it.

So this store is **sparse**: it holds only the keys a user has explicitly
changed. Anything absent falls through to `.env` and then to the code default.
That keeps precedence explainable — "the GUI wins because you set it there,
and only for the things you actually set" — and means deleting settings.json
restores whatever `.env` said rather than a set of hardcoded defaults.

Settings are declared once in `SETTINGS` so the GUI renders itself from the
registry and a new feature flag does not need UI code written for it.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ollama_sentinel.paths import app_data_dir

_LOCK = threading.Lock()


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    help: str
    default: Any
    section: str = "General"
    #: Attribute on AppConfig this maps to, when it maps to one.
    config_attr: str | None = None
    kind: str = "bool"  # bool | number
    minimum: float | None = None
    maximum: float | None = None


SETTINGS: tuple[Setting, ...] = (
    Setting(
        key="advisor",
        label="Model advisories",
        help="Heuristic findings about fit, quantization and config drift.",
        default=True,
        section="Analysis",
        config_attr="advisor",
    ),
    Setting(
        key="ctx_pressure",
        label="Context-window pressure",
        help="Detect prompts filling the served window and replies being truncated.",
        default=True,
        section="Analysis",
    ),
    Setting(
        key="update_check",
        label="Report pending Ollama updates",
        help="Notice when an update has been downloaded but not installed.",
        default=True,
        section="Updates",
    ),
    Setting(
        key="update_auto_apply",
        label="Install updates automatically when idle",
        help=(
            "Runs the downloaded installer. Ollama is DOWN for about a minute and "
            "remote clients lose the API too, so this is off unless you turn it on."
        ),
        default=False,
        section="Updates",
    ),
    Setting(
        key="update_idle_seconds",
        label="Quiet period before auto-install (seconds)",
        help="No loaded model and no request newer than this before an update is applied.",
        default=900,
        section="Updates",
        kind="number",
        minimum=60,
        maximum=86_400,
    ),
    Setting(
        key="notifications",
        label="Desktop notifications",
        help="Toast when an alarm fires or resolves.",
        default=True,
        section="Alerts",
    ),
    Setting(
        key="proc_vram",
        label="Per-process VRAM",
        help="Poll which processes hold GPU memory. Costs a little CPU.",
        default=True,
        section="Monitoring",
        config_attr="proc_vram",
    ),
    Setting(
        key="metrics",
        label="Keep metric history",
        help="Retain samples for the Charts page.",
        default=True,
        section="Monitoring",
        config_attr="metrics",
    ),
    Setting(
        key="gaming_yield",
        label="Yield the GPU to games",
        help="Unload models when a game needs the VRAM.",
        default=False,
        section="Monitoring",
        config_attr="gaming_yield",
    ),
)

BY_KEY: dict[str, Setting] = {s.key: s for s in SETTINGS}


def settings_path() -> Path:
    return app_data_dir() / "settings.json"


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Explicitly-set values only. A broken file reads as empty, never as defaults."""
    p = path or settings_path()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if k in BY_KEY}


def save_settings(values: dict[str, Any], path: Path | None = None) -> None:
    """Write atomically; a torn settings file would silently reset every toggle."""
    p = path or settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with _LOCK:
        tmp.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(p)


def set_setting(key: str, value: Any, path: Path | None = None) -> dict[str, Any]:
    """Persist one toggle and return the whole stored set."""
    if key not in BY_KEY:
        raise KeyError(key)
    values = load_settings(path)
    values[key] = _coerce(BY_KEY[key], value)
    save_settings(values, path)
    return values


def _coerce(setting: Setting, value: Any) -> Any:
    if setting.kind == "bool":
        return bool(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return setting.default
    if setting.minimum is not None:
        number = max(setting.minimum, number)
    if setting.maximum is not None:
        number = min(setting.maximum, number)
    return int(number) if float(number).is_integer() else number


def effective(key: str, cfg: Any = None, settings: dict[str, Any] | None = None) -> Any:
    """Resolve one setting: stored value, else `.env`/AppConfig, else the default.

    `cfg` is consulted only for settings that map to an AppConfig attribute, so
    a GUI-only toggle does not silently pick up an unrelated attribute of the
    same name.
    """
    setting = BY_KEY[key]
    values = load_settings() if settings is None else settings
    if key in values:
        return _coerce(setting, values[key])
    if cfg is not None and setting.config_attr:
        current = getattr(cfg, setting.config_attr, None)
        if current is not None:
            return current
    return setting.default


def apply_to_config(cfg: Any, settings: dict[str, Any] | None = None) -> Any:
    """Overlay stored settings onto an AppConfig, in place.

    Only keys the user actually set are applied, so `.env` still owns anything
    untouched in the GUI.
    """
    values = load_settings() if settings is None else settings
    for setting in SETTINGS:
        if setting.config_attr and setting.key in values:
            setattr(cfg, setting.config_attr, _coerce(setting, values[setting.key]))
    return cfg


def sections() -> list[tuple[str, list[Setting]]]:
    """Settings grouped for display, in declaration order."""
    order: list[str] = []
    grouped: dict[str, list[Setting]] = {}
    for setting in SETTINGS:
        if setting.section not in grouped:
            grouped[setting.section] = []
            order.append(setting.section)
        grouped[setting.section].append(setting)
    return [(name, grouped[name]) for name in order]
