"""Pure telemetry formatting and GPU enrichment."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

MIB = 1024 * 1024

_MISSING = frozenset({"", "[N/A]", "[Not Supported]", "N/A"})
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


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


def is_local_server_url(url: str | None) -> bool:
    if not url:
        return True
    host = (urlparse(url).hostname or "").lower()
    return host in LOCAL_HOSTS


def parse_rfc3339(value: str) -> datetime | None:
    if not value:
        return None
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dot = s.find(".")
    if dot != -1:
        plus = s.find("+", dot)
        minus = s.find("-", dot + 1)
        tz_at = plus if plus != -1 else minus
        if tz_at != -1:
            frac = s[dot + 1 : tz_at]
            # Windows/Ollama may emit 7+ fractional digits; Python accepts 6 (microseconds).
            frac = (frac + "000000")[:6]
            s = s[: dot + 1] + frac + s[tz_at:]
        else:
            s = s[:dot] + s[dot : dot + 7]
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _format_duration_seconds(sec: int) -> str:
    if sec < 60:
        return f"{sec}s"
    minutes = sec // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    rem = minutes % 60
    if rem:
        return f"{hours}h {rem}m"
    return f"{hours}h"


def format_relative_delta(target: datetime, now: datetime) -> str:
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    delta_sec = int((target - now).total_seconds())
    if delta_sec >= 0:
        return f"in {_format_duration_seconds(delta_sec)}"
    return f"expired {_format_duration_seconds(-delta_sec)} ago"


def format_clock_hms(dt: datetime, *, viewer_local: bool = True) -> str:
    if viewer_local:
        dt = dt.astimezone()
    return dt.strftime("%H:%M:%S")


def format_expires_display(
    expires_at: str | None,
    *,
    server_url: str | None = None,
    now: datetime | None = None,
) -> str:
    if not expires_at:
        return "—"
    if expires_at[:4].isdigit() and int(expires_at[:4]) >= 2100:
        return "Forever"
    dt = parse_rfc3339(expires_at)
    if dt is None:
        return "—"
    now_dt = now if now is not None else datetime.now().astimezone()
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    rel = format_relative_delta(dt, now_dt)
    if is_local_server_url(server_url):
        clock = format_clock_hms(dt, viewer_local=False)
    else:
        clock = format_clock_hms(dt, viewer_local=True)
    return f"{rel} ({clock})"


def format_ts_local(ts: float) -> str:
    """Viewer-local HH:MM:SS for JSONL human fields."""
    return datetime.fromtimestamp(ts).astimezone().strftime("%H:%M:%S")


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
