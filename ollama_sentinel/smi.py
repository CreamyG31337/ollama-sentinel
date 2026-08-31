"""Parse nvidia-smi CSV output."""

from __future__ import annotations

import subprocess
from typing import Any


def _parse_float(value: str) -> float | None:
    value = value.strip()
    if not value or value == "[N/A]":
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def parse_nvidia_smi_csv(text: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for idx, line in enumerate(text.strip().splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        used = _parse_float(parts[1])
        total = _parse_float(parts[2])
        # nvidia-smi reports MiB with nounits
        mib = 1024 * 1024
        gpus.append(
            {
                "index": idx,
                "name": parts[0],
                "memory_used": used * mib if used is not None else None,
                "memory_total": total * mib if total is not None else None,
                "utilization": _parse_float(parts[3]),
                "power_draw": _parse_float(parts[4]),
                "power_limit": _parse_float(parts[5]),
                "temperature": _parse_float(parts[6]),
            }
        )
    return gpus


def query_gpus(gpu_filter: int | None = None) -> list[dict[str, Any]] | None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu,"
        "power.draw,power.limit,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    gpus = parse_nvidia_smi_csv(result.stdout)
    if gpu_filter is not None:
        gpus = [g for g in gpus if g["index"] == gpu_filter]
    return gpus
