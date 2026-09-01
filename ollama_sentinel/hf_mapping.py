"""Map Ollama tag names to Hugging Face search hints (heuristic, not guaranteed)."""

from __future__ import annotations

import re
from typing import Any

_HF_PULL = re.compile(r"^hf\.co/([^:]+)(?::(.+))?$", re.IGNORECASE)


def parse_hf_pull_name(name: str) -> tuple[str, str | None] | None:
    """Return (repo_id, filename) for hf.co/org/repo:file.gguf pulls."""
    m = _HF_PULL.match(name.strip())
    if not m:
        return None
    return m.group(1), m.group(2)


def hf_search_query_from_tag(name: str, *, family: str | None = None) -> str:
    """Best-effort Discover search string for an installed tag."""
    hf = parse_hf_pull_name(name)
    if hf:
        repo, _file = hf
        return repo.split("/")[-1].replace("-GGUF", "").replace("_GGUF", "")
    base = name.split(":")[0]
    base = re.sub(r"-mtp\b", "", base, flags=re.IGNORECASE)
    if family:
        return family
    return base


def suggest_hf_pull_variants(
    detail: dict[str, Any],
    *,
    want_mtp: bool = False,
    prefer_lighter: bool = False,
) -> list[str]:
    """Return pull_name strings from HF detail variants (no network)."""
    variants = detail.get("variants") or []
    names = [v.get("pull_name") or "" for v in variants if v.get("pull_name")]
    if want_mtp:
        mtp = [n for n in names if "mtp" in n.lower()]
        if mtp:
            names = mtp
    if prefer_lighter:
        ranked = sorted(names, key=lambda n: (_quant_rank(n), len(n)))
        return ranked[:3]
    return names[:5]


def _quant_rank(filename: str) -> int:
    lower = filename.lower()
    if "q2" in lower or "iq2" in lower:
        return 0
    if "q3" in lower or "iq3" in lower:
        return 1
    if "q4" in lower:
        return 2
    if "q5" in lower:
        return 3
    if "q6" in lower:
        return 4
    if "q8" in lower:
        return 5
    if "f16" in lower:
        return 6
    return 4
