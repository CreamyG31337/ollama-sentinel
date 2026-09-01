"""Parse nvidia-smi and rocm-smi GPU memory output."""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Any

from ollama_sentinel.telemetry import _parse_numeric, enrich_gpu, mib_to_bytes

GPU_QUERY = (
    "name,temperature.gpu,fan.speed,utilization.gpu,utilization.memory,"
    "clocks.sm,clocks.mem,pstate,memory.used,memory.total,memory.reserved,"
    "power.draw,power.limit,enforced.power.limit,"
    "clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_power_cap"
)


def _no_window() -> dict:
    """Keep subprocesses from flashing a console window under pythonw."""
    if sys.platform != "win32":
        return {}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = subprocess.SW_HIDE
    return {"creationflags": flags, "startupinfo": si}


def _parse_throttle(value: str) -> str | None:
    value = value.strip()
    if value in ("", "[N/A]", "[Not Supported]"):
        return None
    return value


def parse_nvidia_smi_csv(text: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for idx, line in enumerate(text.strip().splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 16:
            continue
        gpu = {
            "index": idx,
            "name": parts[0],
            "temperature": _parse_numeric(parts[1]),
            "fan_speed": _parse_numeric(parts[2]),
            "utilization": _parse_numeric(parts[3]),
            "memory_utilization": _parse_numeric(parts[4]),
            "clock_sm": _parse_numeric(parts[5]),
            "clock_mem": _parse_numeric(parts[6]),
            "pstate": parts[7] if parts[7] not in ("[N/A]", "[Not Supported]") else None,
            "memory_used": mib_to_bytes(_parse_numeric(parts[8])),
            "memory_total": mib_to_bytes(_parse_numeric(parts[9])),
            "memory_reserved": mib_to_bytes(_parse_numeric(parts[10])),
            "power_draw": _parse_numeric(parts[11]),
            "power_limit": _parse_numeric(parts[12]),
            "power_limit_enforced": _parse_numeric(parts[13]),
            "throttle_hw_thermal": _parse_throttle(parts[14]),
            "throttle_sw_power_cap": _parse_throttle(parts[15]),
        }
        gpus.append(enrich_gpu(gpu))
    return gpus


_ROCM_TOTAL = re.compile(r"Total Memory \(B\):\s*(\d+)", re.I)
_ROCM_USED = re.compile(r"Total Used Memory \(B\):\s*(\d+)", re.I)


def parse_rocm_smi_vram(text: str) -> list[dict[str, Any]]:
    """Parse `rocm-smi --showmeminfo vram` text output."""
    gpu_line = re.compile(r"GPU\[(\d+)\]", re.I)
    partial: dict[int, dict[str, Any]] = {}
    for line in text.splitlines():
        header = gpu_line.search(line)
        if not header:
            continue
        idx = int(header.group(1))
        entry = partial.setdefault(
            idx,
            {
                "index": idx,
                "name": "AMD GPU",
                "temperature": None,
                "fan_speed": None,
                "utilization": None,
                "memory_utilization": None,
                "clock_sm": None,
                "clock_mem": None,
                "pstate": None,
                "memory_used": 0,
                "memory_total": 0,
                "memory_reserved": None,
                "power_draw": None,
                "power_limit": None,
                "power_limit_enforced": None,
                "throttle_hw_thermal": None,
                "throttle_sw_power_cap": None,
            },
        )
        total_m = _ROCM_TOTAL.search(line)
        if total_m:
            entry["memory_total"] = int(total_m.group(1))
        used_m = _ROCM_USED.search(line)
        if used_m:
            entry["memory_used"] = int(used_m.group(1))
    gpus = [partial[i] for i in sorted(partial)]
    return [enrich_gpu(g) for g in gpus if g.get("memory_total")]


def query_rocm_gpus(gpu_filter: int | None = None) -> list[dict[str, Any]] | None:
    cmd = ["rocm-smi", "--showmeminfo", "vram"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15, **_no_window()
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    gpus = parse_rocm_smi_vram(result.stdout)
    if gpu_filter is not None:
        gpus = [g for g in gpus if g["index"] == gpu_filter]
    return gpus if gpus else None


def query_gpus(gpu_filter: int | None = None) -> list[dict[str, Any]] | None:
    gpus = query_nvidia_gpus(gpu_filter)
    if gpus:
        return gpus
    return query_rocm_gpus(gpu_filter)


def query_nvidia_gpus(gpu_filter: int | None = None) -> list[dict[str, Any]] | None:
    cmd = [
        "nvidia-smi",
        f"--query-gpu={GPU_QUERY}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15, **_no_window()
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    gpus = parse_nvidia_smi_csv(result.stdout)
    if gpu_filter is not None:
        gpus = [g for g in gpus if g["index"] == gpu_filter]
    return gpus
