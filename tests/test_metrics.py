"""Tests for metrics ring buffer."""

from __future__ import annotations

import time

from ollama_sentinel.metrics import MetricsStore


def _gpu_snap(ts: float, *, util: float = 50.0, used: int = 20_000_000_000) -> dict:
    return {
        "server": "local",
        "reachable": True,
        "polled_at_ts": ts,
        "gpus": [
            {
                "index": 0,
                "memory_used": used,
                "memory_total": 25_000_000_000,
                "memory_free": 5_000_000_000,
                "utilization": util,
                "power_draw": 200.0,
                "temperature": 65.0,
            }
        ],
        "models": [{"name": "m", "size_vram": used, "size": used}],
    }


def test_ingest_and_series():
    store = MetricsStore(max_samples=100, retention_s=600)
    base = time.time()
    store.ingest_snapshot(_gpu_snap(base, util=10))
    store.ingest_snapshot(_gpu_snap(base + 5, util=90))
    series = store.series("util", window_s=60)
    assert len(series) == 2
    assert series[0][1] == 10
    assert series[1][1] == 90


def test_loaded_vram_series():
    store = MetricsStore()
    ts = time.time()
    store.ingest_snapshot(_gpu_snap(ts, used=17_000_000_000))
    series = store.series("loaded_vram_gb", window_s=300)
    assert len(series) == 1
    assert abs(series[0][1] - 17.0) < 0.1


def test_llama_util_from_proc_rows():
    store = MetricsStore()
    ts = time.time()
    store.ingest_proc_vram(
        [
            {"name": "llama-server.exe", "bytes": 1e9, "engine_3d_pct": 72.0},
            {"name": "dwm.exe", "bytes": 1e8, "engine_3d_pct": 2.0},
        ],
        ts=ts,
    )
    series = store.series("llama_util", window_s=60)
    assert series[0][1] == 72.0


def test_snapshot_export():
    store = MetricsStore()
    store.ingest_snapshot(_gpu_snap(time.time()))
    snap = store.snapshot(window_s=300)
    assert snap["counts"]["gpu"] == 1
    assert snap["load"][0]["model_count"] == 1
