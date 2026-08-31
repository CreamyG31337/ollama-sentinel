"""Platform-specific paths for state and cache."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def app_data_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "ollama-sentinel"
    xdg = os.environ.get("XDG_STATE_HOME")
    if xdg:
        return Path(xdg) / "ollama-sentinel"
    return Path.home() / ".local" / "state" / "ollama-sentinel"


def cache_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "ollama-sentinel"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "ollama-sentinel"
    return Path.home() / ".cache" / "ollama-sentinel"


def default_state_path() -> Path:
    return app_data_dir() / "state.json"


def default_hub_cache_path() -> Path:
    return cache_dir() / "hub-cache.json"
