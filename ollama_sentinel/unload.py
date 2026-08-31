"""Unload loaded models via POST /api/generate keep_alive=0."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

DEFAULT_TIMEOUT = 120


def unload_model(url: str, model: str, *, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Evict one loaded model from memory. Returns Ollama JSON or {"error": ...}."""
    full = url.rstrip("/") + "/api/generate"
    body = json.dumps({"model": model, "prompt": "", "keep_alive": 0, "stream": False}).encode()
    req = urllib.request.Request(
        full,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode(errors="replace")
        return {"error": f"HTTP {exc.code}: {err_body}"}
    except urllib.error.URLError as exc:
        return {"error": str(exc.reason)}
    except (TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"error": str(exc)}

    if data.get("error"):
        return {"error": data["error"]}
    if not data.get("done"):
        return {"error": "unload did not complete", "response": data}
    return data


def unload_models(url: str, models: list[str]) -> list[dict[str, Any]]:
    """Unload each model in order. Each entry is success payload or {"error", "model"}."""
    results: list[dict[str, Any]] = []
    for model in models:
        result = unload_model(url, model)
        if "error" in result:
            result = dict(result)
            result["model"] = model
        else:
            result = dict(result)
            result.setdefault("model", model)
        results.append(result)
    return results
