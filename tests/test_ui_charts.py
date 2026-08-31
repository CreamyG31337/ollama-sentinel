"""Tests for metrics chart layout helpers."""

from __future__ import annotations

import time

from ollama_sentinel.metrics import MetricsStore
from ollama_sentinel.ui_charts import (
    CHART_WIDTH,
    charts_subtitle,
    format_metric_value,
    series_plot_points,
)


def test_series_plot_points_maps_time_and_value():
    base = 1000.0
    series = [(base, 0.0), (base + 50, 100.0)]
    pts, y_lo, y_hi = series_plot_points(series, ymin=0, ymax=100)
    assert y_lo == 0
    assert y_hi == 100
    assert len(pts) == 2
    assert pts[0][0] < pts[1][0]
    assert pts[0][1] > pts[1][1]  # higher value = lower y


def test_series_plot_points_empty():
    pts, y_lo, y_hi = series_plot_points([])
    assert pts == []
    assert y_hi > y_lo


def test_format_metric_value():
    assert format_metric_value(87.4, "%") == "87%"
    assert format_metric_value(212.7, "W") == "213 W"
    assert format_metric_value(17.25, "GB") == "17.2 GB"


def test_plot_points_span_chart_width():
    now = time.time()
    series = [(now - 60, 10.0), (now, 90.0)]
    pts, _, _ = series_plot_points(series, ymin=0, ymax=100)
    assert pts[0][0] >= 36
    assert pts[-1][0] <= CHART_WIDTH - 12


def test_charts_subtitle_warmup():
    store = MetricsStore(max_samples=100, retention_s=3600)
    text = charts_subtitle(store, window_s=300, server="local", poll_interval=5)
    assert "local" in text
    assert "waiting for data" in text


def test_charts_subtitle_since_start():
    store = MetricsStore(max_samples=100, retention_s=3600)
    now = time.time()
    store.ingest_snapshot(
        {
            "reachable": True,
            "server": "local",
            "polled_at_ts": now - 30,
            "models": [],
            "gpus": [
                {
                    "index": 0,
                    "memory_used": 12_000_000_000,
                    "memory_total": 24_000_000_000,
                    "utilization": 40,
                    "power_draw": 200,
                }
            ],
        }
    )
    store.ingest_snapshot(
        {
            "reachable": True,
            "server": "local",
            "polled_at_ts": now,
            "models": [],
            "gpus": [
                {
                    "index": 0,
                    "memory_used": 13_000_000_000,
                    "memory_total": 24_000_000_000,
                    "utilization": 55,
                    "power_draw": 220,
                }
            ],
        }
    )
    text = charts_subtitle(store, window_s=300, server="local", poll_interval=5)
    assert "since" in text
    assert "last 5 minutes" not in text


def test_charts_subtitle_full_window():
    store = MetricsStore(max_samples=100, retention_s=3600)
    now = time.time()
    for i in range(20):
        store.ingest_snapshot(
            {
                "reachable": True,
                "server": "local",
                "polled_at_ts": now - 280 + i * 15,
                "models": [],
                "gpus": [
                    {
                        "index": 0,
                        "memory_used": 12_000_000_000,
                        "memory_total": 24_000_000_000,
                        "utilization": 40 + i,
                        "power_draw": 200,
                    }
                ],
            }
        )
    text = charts_subtitle(store, window_s=300, server="local", poll_interval=5)
    assert "last 5 minutes" in text
