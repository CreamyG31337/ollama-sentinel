"""Read client configs and cross-check model names against server inventory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_client_config(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    clients = data.get("clients") if isinstance(data, dict) else data
    if not isinstance(clients, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in clients:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or "client"
        models = entry.get("models") or []
        if isinstance(models, str):
            models = [models]
        out.append({"name": str(name), "models": [str(m) for m in models if m]})
    return out


def installed_model_names(snapshots: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for snap in snapshots:
        if not snap.get("reachable"):
            continue
        for tag in snap.get("tags") or []:
            n = tag.get("name") or tag.get("model")
            if n:
                names.add(n)
    return names


def missing_client_models(
    clients: list[dict[str, Any]],
    installed: set[str],
) -> list[tuple[str, str]]:
    """Return (client_name, model_name) pairs not in installed set."""
    missing: list[tuple[str, str]] = []
    for client in clients:
        cname = client.get("name") or "client"
        for model in client.get("models") or []:
            if model not in installed:
                missing.append((cname, model))
    return missing
