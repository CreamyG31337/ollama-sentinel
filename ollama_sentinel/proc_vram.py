"""Per-process GPU VRAM (slow path — background thread only)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from typing import Any

from ollama_sentinel.smi import _no_window
from ollama_sentinel.telemetry import _parse_numeric, mib_to_bytes, polled_at_iso

PID_RE = re.compile(r"pid_(\d+)")


def _resolve_process_name(pid: int) -> str:
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
                **_no_window(),
            )
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().splitlines()[0]
                if "No tasks" in line:
                    return f"pid {pid} (exited)"
                parts = line.split(",")
                if parts:
                    name = parts[0].strip('"')
                    if name:
                        return name
        except (OSError, subprocess.TimeoutExpired):
            pass
        return f"pid {pid} (exited)"
    try:
        with open(f"/proc/{pid}/comm", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return f"pid {pid} (exited)"


def _query_linux(min_bytes: int) -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, **_no_window())
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi compute-apps failed")

    by_pid: dict[int, dict[str, Any]] = {}
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        mib = _parse_numeric(parts[2])
        if mib is None:
            continue
        nbytes = mib_to_bytes(mib) or 0
        if nbytes < min_bytes:
            continue
        by_pid[pid] = {
            "pid": pid,
            "name": parts[1],
            "bytes": nbytes,
            "non_local_bytes": None,
            "engine_3d_pct": None,
        }

    rows = sorted(by_pid.values(), key=lambda r: r["bytes"], reverse=True)
    return rows


def _parse_counter_json(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if not text:
        return []
    data = json.loads(text)
    if isinstance(data, dict):
        return [data]
    return data


ENGTYPE_3D_RE = re.compile(r"engtype_3[dD]", re.IGNORECASE)


def _query_windows(min_bytes: int) -> list[dict[str, Any]]:
    ps_script = (
        "$local = (Get-Counter '\\GPU Process Memory(*)\\Local Usage').CounterSamples; "
        "$nonlocal = (Get-Counter '\\GPU Process Memory(*)\\Non Local Usage').CounterSamples; "
        "$engine = (Get-Counter '\\GPU Engine(*)\\Utilization Percentage').CounterSamples; "
        "$rows = @(); "
        "foreach ($s in $local) { "
        "$rows += [PSCustomObject]@{ Instance=$s.InstanceName; Kind='local'; Value=$s.CookedValue } }; "
        "foreach ($s in $nonlocal) { "
        "$rows += [PSCustomObject]@{ Instance=$s.InstanceName; Kind='nonlocal'; Value=$s.CookedValue } }; "
        "foreach ($s in $engine) { "
        "$rows += [PSCustomObject]@{ Instance=$s.InstanceName; Kind='engine'; Value=$s.CookedValue } }; "
        "$rows | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script],
        capture_output=True,
        text=True,
        timeout=45,
        **_no_window(),
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Get-Counter failed")

    local_by_pid: dict[int, int] = {}
    nonlocal_by_pid: dict[int, int] = {}
    engine_3d_by_pid: dict[int, float] = {}
    for sample in _parse_counter_json(result.stdout):
        instance = sample.get("Instance") or ""
        match = PID_RE.search(instance)
        if not match:
            continue
        pid = int(match.group(1))
        value = sample.get("Value") or 0
        kind = sample.get("Kind")
        if kind == "local":
            local_by_pid[pid] = local_by_pid.get(pid, 0) + int(value)
        elif kind == "nonlocal":
            nonlocal_by_pid[pid] = nonlocal_by_pid.get(pid, 0) + int(value)
        elif kind == "engine" and ENGTYPE_3D_RE.search(instance):
            engine_3d_by_pid[pid] = engine_3d_by_pid.get(pid, 0.0) + float(value)

    # Include PIDs that only show up on the engine counter (util without much VRAM yet).
    all_pids = set(local_by_pid) | set(engine_3d_by_pid)
    rows: list[dict[str, Any]] = []
    for pid in all_pids:
        nbytes = local_by_pid.get(pid, 0)
        util = engine_3d_by_pid.get(pid, 0.0)
        if nbytes < min_bytes and util < 1.0:
            continue
        rows.append(
            {
                "pid": pid,
                "name": _resolve_process_name(pid),
                "bytes": nbytes,
                "non_local_bytes": nonlocal_by_pid.get(pid) or 0,
                "engine_3d_pct": round(util, 1),
            }
        )
    rows.sort(key=lambda r: (r.get("bytes") or 0, r.get("engine_3d_pct") or 0), reverse=True)
    return rows


def query_process_vram(min_bytes: int = 64 * 1024 * 1024) -> list[dict[str, Any]]:
    """Return per-process VRAM rows sorted by local bytes descending."""
    if sys.platform == "win32":
        return _query_windows(min_bytes)
    return _query_linux(min_bytes)


class ProcessVramCollector:
    """Background cache for slow per-process VRAM queries."""

    def __init__(
        self,
        *,
        interval: float = 30,
        enabled: bool = True,
        min_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.interval = interval
        self.enabled = enabled
        self.min_bytes = min_bytes
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cache: dict[str, Any] = {
            "rows": [],
            "polled_at": None,
            "polled_at_ts": None,
            "error": None,
            "stale": False,
        }

    def _poll_once(self) -> None:
        try:
            rows = query_process_vram(self.min_bytes)
            now = time.time()
            with self._lock:
                self._cache = {
                    "rows": rows,
                    "polled_at": polled_at_iso(now),
                    "polled_at_ts": now,
                    "error": None,
                    "stale": False,
                }
        except Exception as exc:
            with self._lock:
                if self._cache.get("rows"):
                    self._cache["stale"] = True
                    self._cache["error"] = str(exc)
                else:
                    self._cache = {
                        "rows": [],
                        "polled_at": None,
                        "polled_at_ts": None,
                        "error": str(exc),
                        "stale": True,
                    }

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(self.interval)

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="proc-vram")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._cache)
