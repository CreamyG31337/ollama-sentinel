"""Cached GET /api/show client."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from ollama_sentinel.show_parse import parse_show_bundle

DEFAULT_TIMEOUT = 30
DEFAULT_TTL = 900.0


def fetch_show(url: str, model: str, *, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    full = url.rstrip("/") + "/api/show"
    body = json.dumps({"name": model}).encode()
    req = urllib.request.Request(
        full,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}"}
    except urllib.error.URLError as exc:
        return {"error": str(exc.reason)}
    except (TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"error": str(exc)}
    if not isinstance(raw, dict):
        return {"error": "invalid response"}
    return parse_show_bundle(raw)


class ShowCache:
    def __init__(self, *, ttl: float = DEFAULT_TTL) -> None:
        self.ttl = ttl
        self._entries: dict[str, tuple[float, dict[str, Any]]] = {}

    def get(self, url: str, model: str, *, force: bool = False) -> dict[str, Any]:
        key = f"{url}|{model}"
        now = time.time()
        if not force:
            hit = self._entries.get(key)
            if hit and (now - hit[0]) < self.ttl:
                return hit[1]
        bundle = fetch_show(url, model)
        self._entries[key] = (now, bundle)
        return bundle

    def fetch_all(self, url: str, model_names: list[str]) -> dict[str, dict[str, Any]]:
        return {name: self.get(url, name) for name in model_names}
