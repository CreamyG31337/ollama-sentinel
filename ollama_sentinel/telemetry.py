"""Pure telemetry formatting and GPU enrichment."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

MIB = 1024 * 1024

_MISSING = frozenset({"", "[N/A]", "[Not Supported]", "N/A"})


def _parse_numeric(value: str) -> float | None:
    value = value.strip()
    if value in _MISSING:
        return None
    for suffix in (" %", "%", " MHz", "MHz", " MiB", "MiB", " W", "W"):
        if value.endswith(suffix):
            value = value[: -len(suffix)].strip()
            break
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def mib_to_bytes(mib: float | None) -> int | None:
    if mib is None:
        return None
    return int(mib * MIB)


def enrich_gpu(gpu: dict[str, Any]) -> dict[str, Any]:
    """Add derived free/reserved fields in-place and return gpu."""
    used = gpu.get("memory_used")
    total = gpu.get("memory_total")
    if used is not None and total is not None and total > 0:
        free = max(0, int(total - used))
        gpu["memory_free"] = free
        gpu["memory_free_pct"] = round(100 * free / total, 1)
    else:
        gpu["memory_free"] = None
        gpu["memory_free_pct"] = None
    return gpu


def format_field(value: Any, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value == int(value):
            return f"{int(value)}{suffix}"
        return f"{value:.1f}{suffix}"
    return f"{value}{suffix}"


def format_bytes_gb(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{value / 1e9:.1f} GB"


def format_poll_age(polled_at: float, now: float) -> str:
    dt = datetime.fromtimestamp(polled_at).astimezone()
    age = max(0, int(now - polled_at))
    return f"{dt.strftime('%H:%M:%S')} ({age}s ago)"


def polled_at_iso(polled_at: float) -> str:
    return datetime.fromtimestamp(polled_at, tz=timezone.utc).isoformat()


def is_stale(polled_at: float | None, interval: float, now: float) -> bool:
    if polled_at is None:
        return True
    return (now - polled_at) > (3 * interval)


def format_throttle(gpu: dict[str, Any]) -> str | None:
    parts: list[str] = []
    if gpu.get("throttle_hw_thermal") == "Active":
        parts.append("thermal slowdown")
    if gpu.get("throttle_sw_power_cap") == "Active":
        parts.append("power cap")
    if not parts:
        return None
    return "Throttling: " + ", ".join(parts)


def format_gpu_line(gpu: dict[str, Any]) -> str:
    used = gpu.get("memory_used")
    total = gpu.get("memory_total")
    free = gpu.get("memory_free")
    free_pct = gpu.get("memory_free_pct")
    reserved = gpu.get("memory_reserved")
    parts = [
        f"#{gpu.get('index', 0)}",
        f"used {format_bytes_gb(used)}",
        f"free {format_bytes_gb(free)} ({format_field(free_pct, '%')})",
        f"total {format_bytes_gb(total)}",
        f"reserved {format_bytes_gb(reserved)}",
        f"temp {format_field(gpu.get('temperature'), '°C')}",
        f"fan {format_field(gpu.get('fan_speed'), '%')}",
        f"util {format_field(gpu.get('utilization'), '%')}",
        f"mem util {format_field(gpu.get('memory_utilization'), '%')}",
        f"clocks {format_field(gpu.get('clock_sm'), ' MHz')}/{format_field(gpu.get('clock_mem'), ' MHz')}",
        f"{format_field(gpu.get('pstate'))}",
        f"power {format_field(gpu.get('power_draw'), ' W')}/{format_field(gpu.get('power_limit'), ' W')}",
    ]
    line = " · ".join(parts)
    throttle = format_throttle(gpu)
    if throttle:
        line += f" · {throttle}"
    return line
