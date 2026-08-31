"""Pure alarm evaluation — no I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ollama_sentinel.telemetry import format_expires_display


@dataclass
class Thresholds:
    paging_util_pct: float = 85.0
    paging_power_frac: float = 0.60
    paging_power_w: float | None = None
    paging_polls: int = 3
    vram_pressure: float = 0.95


@dataclass
class AlarmTransition:
    kind: str  # FIRE | RESOLVED
    alarm_id: str
    message: str


@dataclass
class AlarmState:
    paging_streak: dict[str, int] = field(default_factory=dict)
    active_ids: set[str] = field(default_factory=set)


def format_expires(
    expires_at: str | None,
    *,
    server_url: str | None = None,
    now: datetime | None = None,
) -> str:
    return format_expires_display(expires_at, server_url=server_url, now=now)


def gpu_pct(size: int, size_vram: int) -> int:
    if size <= 0:
        return 100
    return round(100 * size_vram / size)


def _paging_power_threshold(gpu: dict[str, Any], thresholds: Thresholds) -> float | None:
    if thresholds.paging_power_w is not None:
        return thresholds.paging_power_w
    limit = gpu.get("power_limit")
    if limit is None or limit <= 0:
        return None
    return thresholds.paging_power_frac * limit


def evaluate_alarms(
    snapshot: dict[str, Any],
    prev_state: AlarmState | None,
    thresholds: Thresholds,
) -> tuple[list[dict[str, Any]], AlarmState, list[AlarmTransition]]:
    """Evaluate alarms from a parsed snapshot. Pure function."""
    prev = prev_state or AlarmState()
    new_state = AlarmState(
        paging_streak=dict(prev.paging_streak),
        active_ids=set(),
    )
    active: list[dict[str, Any]] = []
    transitions: list[AlarmTransition] = []

    server = snapshot.get("server", "default")

    if not snapshot.get("reachable", False):
        return active, new_state, transitions

    # A — SPILL
    for model in snapshot.get("models", []):
        size = model.get("size") or 0
        size_vram = model.get("size_vram") or 0
        name = model.get("name") or model.get("model") or "unknown"
        if size > 0 and size_vram < size:
            cpu_gb = (size - size_vram) / 1e9
            pct = gpu_pct(size, size_vram)
            alarm_id = f"spill:{server}:{name}"
            msg = (
                f"SPILL [{server}] {name}: {cpu_gb:.1f} GB on CPU "
                f"({100 - pct}% CPU / {pct}% GPU)"
            )
            active.append({"id": alarm_id, "type": "spill", "message": msg})
            new_state.active_ids.add(alarm_id)

    # B / C — GPU (only if gpu data present)
    for gpu in snapshot.get("gpus") or []:
        idx = gpu.get("index", 0)
        util = gpu.get("utilization")
        power = gpu.get("power_draw")
        used = gpu.get("memory_used")
        total = gpu.get("memory_total")

        # B — PCIe paging
        if util is not None and power is not None:
            power_thresh = _paging_power_threshold(gpu, thresholds)
            streak_key = f"{server}:gpu{idx}"
            if (
                power_thresh is not None
                and util > thresholds.paging_util_pct
                and power < power_thresh
            ):
                streak = new_state.paging_streak.get(streak_key, 0) + 1
                new_state.paging_streak[streak_key] = streak
                if streak >= thresholds.paging_polls:
                    alarm_id = f"paging:{server}:{idx}"
                    msg = (
                        f"PCIe PAGING [{server}] GPU {idx}: "
                        f"{util:.0f}% util at {power:.0f} W "
                        f"(threshold {power_thresh:.0f} W)"
                    )
                    active.append({"id": alarm_id, "type": "paging", "message": msg})
                    new_state.active_ids.add(alarm_id)
            else:
                new_state.paging_streak[streak_key] = 0

        # C — VRAM pressure
        if used is not None and total and total > 0:
            ratio = used / total
            if ratio > thresholds.vram_pressure:
                alarm_id = f"vram:{server}:{idx}"
                msg = (
                    f"VRAM PRESSURE [{server}] GPU {idx}: "
                    f"{used / 1e9:.1f}/{total / 1e9:.1f} GB "
                    f"({ratio * 100:.0f}%)"
                )
                active.append({"id": alarm_id, "type": "vram", "message": msg})
                new_state.active_ids.add(alarm_id)

    # Hysteresis transitions
    prev_active = prev.active_ids
    curr_active = new_state.active_ids
    for aid in curr_active - prev_active:
        msg = next((a["message"] for a in active if a["id"] == aid), aid)
        transitions.append(AlarmTransition("FIRE", aid, msg))
    for aid in prev_active - curr_active:
        transitions.append(AlarmTransition("RESOLVED", aid, f"RESOLVED {aid}"))

    new_state.active_ids = curr_active
    return active, new_state, transitions
