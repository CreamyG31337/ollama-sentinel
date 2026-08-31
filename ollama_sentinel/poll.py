"""HTTP polling for Ollama servers."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from ollama_sentinel.telemetry import polled_at_iso


DEFAULT_TIMEOUT = 10


def _get_json(url: str, path: str, timeout: float = DEFAULT_TIMEOUT) -> tuple[Any | None, str | None]:
    full = url.rstrip("/") + path
    req = urllib.request.Request(full, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode()), None
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return None, str(exc.reason)
    except (TimeoutError, json.JSONDecodeError, OSError) as exc:
        return None, str(exc)


def poll_server(
    url: str,
    server_name: str,
    *,
    attach_gpus: bool = False,
    gpu_filter: int | None = None,
    query_gpus_fn=None,
    polled_at: float | None = None,
) -> dict[str, Any]:
    """Poll one Ollama server. GPU metrics only if attach_gpus is True."""
    snapshot: dict[str, Any] = {
        "server": server_name,
        "url": url,
        "reachable": False,
        "version": None,
        "models": [],
        "tags": [],
        "gpus": None,
        "error": None,
        "stale": False,
    }

    version, verr = _get_json(url, "/api/version")
    ps, perr = _get_json(url, "/api/ps")
    tags, terr = _get_json(url, "/api/tags")

    ts = polled_at if polled_at is not None else time.time()
    snapshot["polled_at"] = polled_at_iso(ts)
    snapshot["polled_at_ts"] = ts

    if ps is None and tags is None and version is None:
        snapshot["error"] = verr or perr or terr or "unreachable"
        if attach_gpus and query_gpus_fn:
            snapshot["gpus"] = query_gpus_fn(gpu_filter)
        return snapshot

    snapshot["reachable"] = True
    if version:
        snapshot["version"] = version.get("version")
    if ps:
        snapshot["models"] = ps.get("models") or []
    if tags:
        snapshot["tags"] = tags.get("models") or []

    if attach_gpus and query_gpus_fn:
        snapshot["gpus"] = query_gpus_fn(gpu_filter)

    return snapshot


def poll_all(
    servers: list[dict[str, Any]],
    *,
    gpu_filter: int | None = None,
    query_gpus_fn=None,
) -> list[dict[str, Any]]:
    """Poll every configured server."""
    local_gpu_data = query_gpus_fn(gpu_filter) if query_gpus_fn else None
    polled_at = time.time()

    results = []
    for srv in servers:
        attach = bool(srv.get("local_gpu")) and local_gpu_data is not None
        snap = poll_server(
            srv["url"],
            srv["name"],
            attach_gpus=attach,
            gpu_filter=gpu_filter,
            query_gpus_fn=lambda gf: local_gpu_data,
            polled_at=polled_at,
        )
        results.append(snap)
    return results
