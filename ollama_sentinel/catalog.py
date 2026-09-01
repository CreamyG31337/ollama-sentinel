"""Hugging Face Hub catalog — stdlib HTTP only."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

HUB_BASE = "https://huggingface.co"
CACHE_TTL = 900  # 15 minutes
README_MAX_CHARS = 12_000

# Hugging Face /api/models sort fields (all work with apps=ollama).
SEARCH_SORTS: tuple[tuple[str, str], ...] = (
    ("trendingScore", "Trending"),
    ("downloads", "Downloads"),
    ("likes", "Likes"),
    ("lastModified", "Recently updated"),
    ("createdAt", "Newest"),
)


from ollama_sentinel.net_errors import HubRequestError, format_network_error


def _hub_get(path: str, params: dict[str, str] | None = None, token: str | None = None) -> Any:
    url = HUB_BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {"Accept": "application/json", "User-Agent": "ollama-sentinel/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise HubRequestError(format_network_error(exc, context="Hugging Face"), cause=exc) from exc


def tag_value(tags: list[str], prefix: str) -> str | None:
    for tag in tags:
        if tag.startswith(prefix):
            return tag[len(prefix) :]
    return None


def format_count(value: int | None) -> str:
    if value is None:
        return "—"
    if value >= 1_000_000:
        return f"{value / 1e6:.1f}M"
    if value >= 1_000:
        return f"{value / 1e3:.1f}k"
    return str(value)


def pull_name_for_file(model_id: str, filename: str) -> str:
    return f"hf.co/{model_id}:{filename}"


def hf_model_url(model_id: str) -> str:
    return f"{HUB_BASE}/{model_id}"


def summarize_list_item(item: dict[str, Any]) -> str:
    parts: list[str] = []
    pipeline = item.get("pipeline_tag")
    if pipeline:
        parts.append(pipeline.replace("-", " "))
    license_name = item.get("license")
    if license_name:
        parts.append(license_name)
    parts.append(f"{format_count(item.get('downloads'))} dl")
    likes = item.get("likes")
    if likes:
        parts.append(f"{format_count(likes)} likes")
    return " · ".join(parts)


def parse_list_item(item: dict[str, Any]) -> dict[str, Any]:
    model_id = item.get("id") or item.get("modelId") or ""
    tags = item.get("tags") or []
    license_name = tag_value(tags, "license:")
    parsed = {
        "id": model_id,
        "pull_name": f"hf.co/{model_id}",
        "downloads": item.get("downloads"),
        "likes": item.get("likes"),
        "trending_score": item.get("trendingScore"),
        "pipeline_tag": item.get("pipeline_tag"),
        "last_modified": item.get("lastModified"),
        "created_at": item.get("createdAt"),
        "tags": tags,
        "license": license_name,
        "private": bool(item.get("private")),
        "hf_url": hf_model_url(model_id),
    }
    parsed["summary"] = summarize_list_item(parsed)
    return parsed


def parse_model_detail(raw: dict[str, Any]) -> dict[str, Any]:
    model_id = raw.get("id") or raw.get("modelId") or ""
    tags = raw.get("tags") or []
    card = raw.get("cardData") or {}
    gguf = raw.get("gguf") or {}

    variants: list[dict[str, str]] = []
    for sibling in raw.get("siblings") or []:
        filename = sibling.get("rfilename") or sibling.get("filename") or ""
        if not filename.lower().endswith(".gguf"):
            continue
        variants.append(
            {
                "filename": filename,
                "pull_name": pull_name_for_file(model_id, filename),
            }
        )
    variants.sort(key=lambda row: row["filename"])

    gated = raw.get("gated")
    return {
        "id": model_id,
        "pull_name": f"hf.co/{model_id}",
        "architecture": gguf.get("architecture"),
        "context_length": gguf.get("context_length"),
        "gguf_total_bytes": gguf.get("total"),
        "base_model": card.get("base_model"),
        "license": card.get("license") or tag_value(tags, "license:"),
        "pipeline_tag": raw.get("pipeline_tag") or card.get("pipeline_tag"),
        "likes": raw.get("likes"),
        "downloads": raw.get("downloads"),
        "last_modified": raw.get("lastModified"),
        "gated": gated not in (False, None, "", "false"),
        "variants": variants,
        "tags": tags,
        "hf_url": hf_model_url(model_id),
    }


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
        results.append(parse_list_item(item))
    return results


def fetch_model_detail(model_id: str, *, token: str | None = None) -> dict[str, Any]:
    encoded = urllib.parse.quote(model_id, safe="/")
    raw = _hub_get(f"/api/models/{encoded}", None, token)
    if not isinstance(raw, dict):
        raise ValueError("unexpected Hugging Face detail response")
    return parse_model_detail(raw)


def strip_yaml_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text.strip()
    end = text.find("\n---", 3)
    if end == -1:
        return text.strip()
    return text[end + 4 :].lstrip("\n")


def truncate_readme(text: str, *, max_chars: int = README_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n\n… *(truncated)*"


def fetch_model_readme(model_id: str, *, token: str | None = None) -> str | None:
    """Fetch README.md from the model repo (public repos need no token)."""
    encoded = urllib.parse.quote(model_id, safe="/")
    url = f"{HUB_BASE}/{encoded}/raw/main/README.md"
    headers = {"User-Agent": "ollama-sentinel/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403, 404):
            return None
        raise
    text = strip_yaml_frontmatter(text)
    if not text.strip():
        return None
    return truncate_readme(text)


def fetch_model_bundle(model_id: str, *, token: str | None = None) -> dict[str, Any]:
    """Detail metadata plus README for the Discover expand panel."""
    detail = fetch_model_detail(model_id, token=token)
    try:
        detail["readme"] = fetch_model_readme(model_id, token=token)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        detail["readme"] = None
    return detail


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
