"""HTTP polling for Ollama servers."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlparse

from ollama_sentinel.net_errors import format_network_error
from ollama_sentinel.telemetry import polled_at_iso


DEFAULT_TIMEOUT = 10
# Windows retries SYN several times before WSAECONNREFUSED (~2s on this host).
# Cap the TCP probe well below that so a dead Ollama fails the switch quickly
# instead of waiting out the refuse dance (or a full HTTP timeout).
CONNECT_TIMEOUT = 0.5


def _tcp_connect_error(url: str, *, timeout: float = CONNECT_TIMEOUT) -> str | None:
    """Return a short error if TCP cannot connect; None if the port accepted us."""
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return "invalid URL"
    port = parsed.port or (443 if (parsed.scheme or "").lower() == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return None
    except OSError as exc:
        return format_network_error(exc)


def _get_json(url: str, path: str, timeout: float = DEFAULT_TIMEOUT) -> tuple[Any | None, str | None]:
    full = url.rstrip("/") + path
    req = urllib.request.Request(full, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode()), None
    except urllib.error.HTTPError as exc:
        return None, format_network_error(exc)
    except urllib.error.URLError as exc:
        return None, format_network_error(exc)
    except (TimeoutError, json.JSONDecodeError, OSError) as exc:
        return None, format_network_error(exc)


def poll_server(
    url: str,
    server_name: str,
    *,
    attach_gpus: bool = False,
    gpu_filter: int | None = None,
    query_gpus_fn=None,
    polled_at: float | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    connect_timeout: float = CONNECT_TIMEOUT,
) -> dict[str, Any]:
    """Poll one Ollama server. GPU metrics only if attach_gpus is True.

    A short TCP probe runs first so connection-refused / black-holed hosts fail
    in well under a second. Only then do ``/api/version``, ``/api/ps`` and
    ``/api/tags`` run concurrently.
    """
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

    conn_err = _tcp_connect_error(url, timeout=connect_timeout)
    if conn_err:
        ts = polled_at if polled_at is not None else time.time()
        snapshot["polled_at"] = polled_at_iso(ts)
        snapshot["polled_at_ts"] = ts
        snapshot["error"] = conn_err
        if attach_gpus and query_gpus_fn:
            snapshot["gpus"] = query_gpus_fn(gpu_filter)
        return snapshot

    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_version = pool.submit(_get_json, url, "/api/version", timeout)
        fut_ps = pool.submit(_get_json, url, "/api/ps", timeout)
        fut_tags = pool.submit(_get_json, url, "/api/tags", timeout)
        version, verr = fut_version.result()
        ps, perr = fut_ps.result()
        tags, terr = fut_tags.result()

    # Stamp when *this* server's poll finished, not when it started — a slow
    # peer must not backdate a fast one (and vice versa).
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
    polled_at: float | None = None,
) -> list[dict[str, Any]]:
    """Poll every configured server.

    Each snapshot is stamped with the time *its own* poll finished, not a single
    timestamp taken before the loop. A shared start-stamp made every server look
    as old as the whole cycle took, so one slow or unreachable host (which pays a
    full connect timeout) dragged every healthy host past the staleness threshold.
    """
    local_gpu_data = query_gpus_fn(gpu_filter) if query_gpus_fn else None

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
        snap["local_gpu"] = bool(srv.get("local_gpu"))
        snap["optional"] = bool(srv.get("optional"))
        snap["gpu_data_available"] = snap.get("gpus") is not None
        results.append(snap)
    return results
