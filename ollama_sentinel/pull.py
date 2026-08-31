"""Streaming POST /api/pull to an Ollama server."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any


def pull_model(url: str, model: str) -> Iterator[dict[str, Any]]:
    full = url.rstrip("/") + "/api/pull"
    body = json.dumps({"name": model, "stream": True}).encode()
    req = urllib.request.Request(
        full,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line.decode())
                except json.JSONDecodeError:
                    yield {"status": line.decode()}
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode(errors="replace")
        yield {"error": f"HTTP {exc.code}: {err_body}"}
    except urllib.error.URLError as exc:
        yield {"error": str(exc.reason)}
