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


def poll_age_seconds(polled_at: float | None, now: float) -> float | None:
    if polled_at is None:
        return None
    return max(0.0, now - polled_at)


def freshness_level(
    polled_at: float | None,
    interval: float,
    now: float,
) -> str:
    """ok | aging | stale | unknown.

    * ok — within one poll interval
    * aging — past one interval but not yet the 3× STALE threshold (refresh late)
    * stale — older than 3× interval (or never polled)
    """
    if polled_at is None:
        return "unknown"
    age = now - polled_at
    if age > 3 * interval:
        return "stale"
    if age > interval:
        return "aging"
    return "ok"


def format_freshness_line(
    polled_at: float | None,
    interval: float,
    now: float,
    *,
    reachable: bool = True,
    live_at: float | None = None,
) -> tuple[str, str]:
    """Return (level, human label) for the status freshness banner."""
    if not reachable:
        if polled_at is None:
            return "stale", "Unreachable · no poll yet"
        return "stale", f"Unreachable · last {format_poll_age(polled_at, now)}"

    level = freshness_level(polled_at, interval, now)
    if polled_at is None:
        return "unknown", "Waiting for first poll…"

    age_text = format_poll_age(polled_at, now)
    if level == "stale":
        label = f"STALE · last poll {age_text}"
    elif level == "aging":
        label = f"Late · last poll {age_text}"
    else:
        label = f"Live · polled {age_text}"

    if live_at is not None:
        live_age = max(0, int(now - live_at))
        label += f" · activity {live_age}s ago"
    return level, label


def polled_at_iso(polled_at: float) -> str:
    return datetime.fromtimestamp(polled_at, tz=timezone.utc).isoformat()


def is_stale(polled_at: float | None, interval: float, now: float) -> bool:
    if polled_at is None:
        return True
    return (now - polled_at) > (3 * interval)


def metric_severity(kind: str, value: float | None, *, ref: float | None = None) -> str:
    """ok | warn | alarm | muted | busy for a GPU telemetry field.

    ``ref`` is the comparison ceiling when needed (power limit, VRAM total).
    Thresholds are tuned for a gaming/datacenter NVIDIA card under LLM load,
    not a silent desktop idle.
    """
    if value is None:
        return "muted"
    if kind == "temperature":
        if value >= 83:
            return "alarm"
        if value >= 75:
            return "warn"
        if value >= 60:
            return "busy"
        return "ok"
    if kind == "utilization":
        if value >= 90:
            return "busy"
        if value >= 5:
            return "ok"
        return "muted"
    if kind == "fan":
        if value >= 80:
            return "warn"
        if value >= 40:
            return "ok"
        return "muted"
    if kind == "power":
        if ref and ref > 0:
            pct = 100.0 * value / ref
            if pct >= 95:
                return "alarm"
            if pct >= 80:
                return "warn"
            if pct >= 20:
                return "ok"
            return "muted"
        return "ok" if value > 0 else "muted"
    if kind == "vram_used_pct":
        if value >= 95:
            return "alarm"
        if value >= 85:
            return "warn"
        if value >= 50:
            return "ok"
        return "muted"
    if kind == "vram_free_pct":
        if value <= 5:
            return "alarm"
        if value <= 15:
            return "warn"
        return "ok"
    return "ok"


def gpu_metric_rows(gpu: dict[str, Any]) -> list[dict[str, Any]]:
    """Structured GPU rows for the status table (icon key, label, value, severity)."""
    used = gpu.get("memory_used")
    total = gpu.get("memory_total")
    free = gpu.get("memory_free")
    free_pct = gpu.get("memory_free_pct")
    used_pct = None
    if used is not None and total:
        used_pct = 100.0 * float(used) / float(total)

    power_draw = gpu.get("power_draw")
    power_limit = gpu.get("power_limit") or gpu.get("power_limit_enforced")
    temp = gpu.get("temperature")
    fan = gpu.get("fan_speed")
    util = gpu.get("utilization")
    mem_util = gpu.get("memory_utilization")

    rows: list[dict[str, Any]] = [
        {
            "key": "vram_used",
            "icon": "memory",
            "label": "VRAM used",
            "value": format_bytes_gb(used),
            "severity": metric_severity("vram_used_pct", used_pct),
        },
        {
            "key": "vram_free",
            "icon": "memory",
            "label": "VRAM free",
            "value": f"{format_bytes_gb(free)} ({format_field(free_pct, '%')})",
            "severity": metric_severity("vram_free_pct", free_pct),
        },
        {
            "key": "vram_total",
            "icon": "memory",
            "label": "VRAM total",
            "value": format_bytes_gb(total),
            "severity": "muted",
        },
        {
            "key": "vram_reserved",
            "icon": "memory",
            "label": "Reserved",
            "value": format_bytes_gb(gpu.get("memory_reserved")),
            "severity": "muted",
        },
        {
            "key": "temperature",
            "icon": "thermostat",
            "label": "Temp",
            "value": format_field(temp, "°C"),
            "severity": metric_severity("temperature", temp),
        },
        {
            "key": "fan",
            "icon": "mode_fan",
            "label": "Fan",
            "value": format_field(fan, "%"),
            "severity": metric_severity("fan", fan),
        },
        {
            "key": "gpu_util",
            "icon": "speed",
            "label": "GPU util",
            "value": format_field(util, "%"),
            "severity": metric_severity("utilization", util),
        },
        {
            "key": "mem_util",
            "icon": "swap_vert",
            "label": "Mem util",
            "value": format_field(mem_util, "%"),
            "severity": metric_severity("utilization", mem_util),
        },
        {
            "key": "power",
            "icon": "bolt",
            "label": "Power",
            "value": f"{format_field(power_draw, ' W')} / {format_field(power_limit, ' W')}",
            "severity": metric_severity("power", power_draw, ref=power_limit),
        },
        {
            "key": "pstate",
            "icon": "tune",
            "label": "Pstate",
            "value": format_field(gpu.get("pstate")),
            "severity": "muted",
        },
        {
            "key": "clocks",
            "icon": "schedule",
            "label": "Clocks",
            "value": (
                f"{format_field(gpu.get('clock_sm'), ' MHz')} / "
                f"{format_field(gpu.get('clock_mem'), ' MHz')}"
            ),
            "severity": "muted",
        },
    ]
    return rows


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
