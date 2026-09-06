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
        record: dict[str, Any] = {
            "name": str(name),
            "models": [str(m) for m in models if m],
        }
        ctx = entry.get("context_length")
        if ctx is not None:
            try:
                record["context_length"] = int(ctx)
            except (TypeError, ValueError):
                pass
        for key in ("context_length_file", "context_length_key", "context_length_match"):
            if entry.get(key):
                record[key] = str(entry[key])
        addrs = entry.get("addrs") or []
        if isinstance(addrs, str):
            addrs = [addrs]
        if isinstance(addrs, list):
            cleaned = [str(a).strip() for a in addrs if a]
            if cleaned:
                record["addrs"] = cleaned
        out.append(record)
    return out


def normalize_tag(name: str) -> str:
    """Ollama's implicit `:latest`, made explicit.

    `bge-m3` and `bge-m3:latest` are the same model, but a client config will
    often write the short form while /api/tags always reports the long one.
    """
    name = name.strip()
    return name if ":" in name else f"{name}:latest"


def installed_model_names(snapshots: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for snap in snapshots:
        if not snap.get("reachable"):
            continue
        for tag in snap.get("tags") or []:
            n = tag.get("name") or tag.get("model")
            if n:
                names.add(normalize_tag(str(n)))
    return names


def inventory_is_complete(snapshots: list[dict[str, Any]]) -> bool:
    """True only when every configured server answered.

    With a host down we cannot know what it holds, so "this model is installed
    nowhere" is unprovable. Optional servers (a gaming rig that is deliberately
    off) do not count against completeness.
    """
    for snap in snapshots:
        if not snap.get("reachable") and not snap.get("optional"):
            return False
    return True


def missing_client_models(
    clients: list[dict[str, Any]],
    installed: set[str],
    *,
    inventory_complete: bool = True,
) -> list[tuple[str, str]]:
    """Return (client_name, model_name) pairs not in the installed set.

    Returns nothing when the inventory is incomplete: an unreachable server
    would otherwise make every model it holds look missing, which is the kind
    of false alarm that trains people to ignore the tool.
    """
    if not inventory_complete:
        return []
    missing: list[tuple[str, str]] = []
    for client in clients:
        cname = client.get("name") or "client"
        for model in client.get("models") or []:
            if normalize_tag(str(model)) not in installed:
                missing.append((cname, model))
    return missing


def overcommitted_clients(
    clients: list[dict[str, Any]],
    served_context: int | None,
) -> list[tuple[str, int, str]]:
    """Clients whose context window exceeds the one Ollama actually serves.

    Ollama advertises a model's *architectural* context (e.g. 262144) but serves
    whatever ``OLLAMA_CONTEXT_LENGTH`` says (e.g. 65536). A client that
    auto-detects the former sizes its history — and its compaction threshold — to
    a window that does not exist, so it fills the real one and every reply is
    truncated.

    Returns ``(client, window, source)`` where source is ``file`` when read from
    the client's own config and ``declared`` when taken from a static value.
    """
    if not served_context or served_context <= 0:
        return []
    from ollama_sentinel.client_probe import resolve_client_context

    out: list[tuple[str, int, str]] = []
    for client in clients:
        window, source = resolve_client_context(client)
        if window is not None and window > served_context:
            out.append((str(client.get("name") or "client"), window, source))
    return out
