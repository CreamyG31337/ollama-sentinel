"""In-memory metrics history — piggybacks on existing polls (no extra subprocess I/O)."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

MetricField = Literal[
    "mem_used_pct",
    "mem_free_gb",
    "util",
    "power_draw",
    "temp",
    "loaded_vram_gb",
    "llama_util",
]


@dataclass(frozen=True)
class GpuPoint:
    ts: float
    server: str
    index: int
    mem_used: int | None
    mem_total: int | None
    mem_free: int | None
    util: float | None
    power_draw: float | None
    temp: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def value(self, field_name: MetricField) -> float | None:
        if field_name == "mem_used_pct":
            used, total = self.mem_used, self.mem_total
            if used is None or total is None or total <= 0:
                return None
            return 100.0 * used / total
        if field_name == "mem_free_gb":
            if self.mem_free is None:
                return None
            return self.mem_free / 1e9
        if field_name == "util":
            return self.util
        if field_name == "power_draw":
            return self.power_draw
        if field_name == "temp":
            return self.temp
        return None


@dataclass(frozen=True)
class LoadPoint:
    ts: float
    server: str
    loaded_vram_gb: float
    model_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LlamaUtilPoint:
    ts: float
    max_util: float
    busy_runners: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MetricsStore:
    """Ring buffers fed from poll snapshots and proc_vram cache — zero extra GPU queries."""

    def __init__(
        self,
        *,
        max_samples: int = 720,
        retention_s: float = 3600.0,
        log_path: Path | None = None,
    ) -> None:
        self.max_samples = max_samples
        self.retention_s = retention_s
        self.log_path = log_path
        self._gpu: deque[GpuPoint] = deque(maxlen=max_samples)
        self._load: deque[LoadPoint] = deque(maxlen=max_samples)
        self._llama_util: deque[LlamaUtilPoint] = deque(maxlen=max_samples)
        self._lock = threading.Lock()

    def ingest_snapshot(self, snap: dict[str, Any]) -> None:
        """Extract metrics from an existing poll snapshot (main poll path)."""
        if not snap.get("reachable"):
            return
        ts = float(snap.get("polled_at_ts") or time.time())
        server = str(snap.get("server") or "local")
        points: list[GpuPoint] = []
        for gpu in snap.get("gpus") or []:
            points.append(
                GpuPoint(
                    ts=ts,
                    server=server,
                    index=int(gpu.get("index") or 0),
                    mem_used=gpu.get("memory_used"),
                    mem_total=gpu.get("memory_total"),
                    mem_free=gpu.get("memory_free"),
                    util=gpu.get("utilization"),
                    power_draw=gpu.get("power_draw"),
                    temp=gpu.get("temperature"),
                )
            )
        models = snap.get("models") or []
        loaded_vram = sum(int(m.get("size_vram") or 0) for m in models)
        load_pt = LoadPoint(
            ts=ts,
            server=server,
            loaded_vram_gb=loaded_vram / 1e9,
            model_count=len(models),
        )
        with self._lock:
            self._prune_locked(ts)
            for pt in points:
                self._gpu.append(pt)
            self._load.append(load_pt)
        self._maybe_log(ts, server, points, load_pt)

    def ingest_proc_vram(self, rows: list[dict[str, Any]], *, ts: float | None = None) -> None:
        """Piggyback on ProcessVramCollector output (30 s cadence on Windows)."""
        if not rows:
            return
        now = float(ts or time.time())
        max_util = 0.0
        busy = 0
        for row in rows:
            name = (row.get("name") or "").lower()
            if "llama-server" not in name:
                continue
            util = float(row.get("engine_3d_pct") or 0.0)
            max_util = max(max_util, util)
            if util >= 5.0:
                busy += 1
        pt = LlamaUtilPoint(ts=now, max_util=max_util, busy_runners=busy)
        with self._lock:
            self._prune_locked(now)
            self._llama_util.append(pt)

    def series(
        self,
        field: MetricField,
        *,
        window_s: float = 300.0,
        server: str | None = None,
        gpu_index: int = 0,
    ) -> list[tuple[float, float]]:
        """Return (timestamp, value) pairs for charting. Newest last."""
        cutoff = time.time() - window_s
        out: list[tuple[float, float]] = []

        with self._lock:
            if field == "loaded_vram_gb":
                for pt in self._load:
                    if pt.ts < cutoff:
                        continue
                    if server and pt.server != server:
                        continue
                    out.append((pt.ts, pt.loaded_vram_gb))
            elif field == "llama_util":
                for pt in self._llama_util:
                    if pt.ts < cutoff:
                        continue
                    out.append((pt.ts, pt.max_util))
            else:
                for pt in self._gpu:
                    if pt.ts < cutoff:
                        continue
                    if server and pt.server != server:
                        continue
                    if pt.index != gpu_index:
                        continue
                    val = pt.value(field)
                    if val is not None:
                        out.append((pt.ts, val))

        return out

    def snapshot(self, *, window_s: float = 300.0) -> dict[str, Any]:
        """Export recent history for --json or debugging."""
        cutoff = time.time() - window_s
        with self._lock:
            gpu = [p.to_dict() for p in self._gpu if p.ts >= cutoff]
            load = [p.to_dict() for p in self._load if p.ts >= cutoff]
            llama = [p.to_dict() for p in self._llama_util if p.ts >= cutoff]
            counts = {
                "gpu": len(self._gpu),
                "load": len(self._load),
                "llama_util": len(self._llama_util),
            }
        return {
            "window_s": window_s,
            "gpu": gpu,
            "load": load,
            "llama_util": llama,
            "counts": counts,
        }

    def _prune_locked(self, now: float) -> None:
        cutoff = now - self.retention_s
        while self._gpu and self._gpu[0].ts < cutoff:
            self._gpu.popleft()
        while self._load and self._load[0].ts < cutoff:
            self._load.popleft()
        while self._llama_util and self._llama_util[0].ts < cutoff:
            self._llama_util.popleft()

    def _maybe_log(
        self,
        ts: float,
        server: str,
        gpu_pts: list[GpuPoint],
        load_pt: LoadPoint,
    ) -> None:
        if self.log_path is None:
            return
        entry = {
            "ts": ts,
            "server": server,
            "gpu": [p.to_dict() for p in gpu_pts],
            "load": load_pt.to_dict(),
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
        except OSError:
            pass


def make_metrics_store(cfg) -> MetricsStore | None:
    if not getattr(cfg, "metrics", True):
        return None
    log_path = getattr(cfg, "metrics_log", None)
    return MetricsStore(
        max_samples=int(getattr(cfg, "metrics_max_samples", 720)),
        retention_s=float(getattr(cfg, "metrics_history_sec", 3600)),
        log_path=log_path,
    )
