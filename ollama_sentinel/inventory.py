"""Installed vs loaded library and fit preview."""

from __future__ import annotations

from typing import Any

from ollama_sentinel.alarms import gpu_pct


def free_vram_bytes(gpus: list[dict[str, Any]] | None) -> int | None:
    if not gpus:
        return None
    total = sum(g.get("memory_total") or 0 for g in gpus)
    used = sum(g.get("memory_used") or 0 for g in gpus)
    if total <= 0:
        return None
    return max(0, int(total - used))


def build_inventory(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Join /api/tags with /api/ps and optional free VRAM."""
    loaded_by_name: dict[str, dict[str, Any]] = {}
    for m in snapshot.get("models") or []:
        key = m.get("name") or m.get("model")
        if key:
            loaded_by_name[key] = m

    free = free_vram_bytes(snapshot.get("gpus"))
    rows: list[dict[str, Any]] = []

    for tag in snapshot.get("tags") or []:
        name = tag.get("name") or tag.get("model") or "unknown"
        size = tag.get("size") or 0
        loaded = name in loaded_by_name
        row: dict[str, Any] = {
            "name": name,
            "size": size,
            "size_gb": size / 1e9 if size else 0,
            "loaded": loaded,
            "quantization": (tag.get("details") or {}).get("quantization_level"),
            "family": (tag.get("details") or {}).get("family"),
            "modified_at": tag.get("modified_at"),
            "digest": tag.get("digest"),
        }
        if loaded:
            lm = loaded_by_name[name]
            sv = lm.get("size_vram") or 0
            ls = lm.get("size") or size
            row["size_vram"] = sv
            row["gpu_pct"] = gpu_pct(ls, sv)
            row["would_spill"] = False
        elif free is not None and size > free:
            row["would_spill"] = True
        else:
            row["would_spill"] = False if free is not None else None
        rows.append(row)

    return rows


def inventory_summary(
    rows: list[dict[str, Any]],
    *,
    free_vram_gb: float | None = None,
    free_vram_pct: float | None = None,
) -> str:
    installed = len(rows)
    loaded = sum(1 for r in rows if r.get("loaded"))
    would = sum(1 for r in rows if r.get("would_spill"))
    parts = [f"{installed} installed", f"{loaded} loaded"]
    if any(r.get("would_spill") is not None for r in rows):
        parts.append(f"{would} would spill")
    if free_vram_gb is not None and free_vram_pct is not None:
        parts.append(f"free VRAM: {free_vram_gb:.1f} GB ({free_vram_pct:.0f}%)")
    return " | ".join(parts)
