"""Hugging Face Hub catalog — stdlib HTTP only."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

HUB_BASE = "https://huggingface.co"
CACHE_TTL = 900  # 15 minutes


def _hub_get(path: str, params: dict[str, str] | None = None, token: str | None = None) -> Any:
    url = HUB_BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {"Accept": "application/json", "User-Agent": "ollama-sentinel/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def search_models(
    query: str = "",
    *,
    sort: str = "trendingScore",
    limit: int = 20,
    token: str | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, str] = {
        "apps": "ollama",
        "limit": str(min(limit, 100)),
        "sort": sort,
        "direction": "-1",
    }
    if query:
        params["search"] = query
    raw = _hub_get("/api/models", params, token)
    results: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        model_id = item.get("id") or item.get("modelId") or ""
        results.append(
            {
                "id": model_id,
                "pull_name": f"hf.co/{model_id}",
                "downloads": item.get("downloads"),
                "last_modified": item.get("lastModified"),
                "tags": item.get("tags") or [],
            }
        )
    return results


def typeahead(query: str, *, limit: int = 8, token: str | None = None) -> list[dict[str, Any]]:
    if len(query.strip()) < 2:
        return []
    return search_models(query.strip(), sort="downloads", limit=limit, token=token)


def load_cache(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if time.time() - data.get("ts", 0) > CACHE_TTL:
        return None
    return data


def save_cache(path: Path, key: str, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    existing[key] = payload
    existing["ts"] = time.time()
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
