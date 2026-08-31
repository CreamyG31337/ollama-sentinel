"""Tests for metrics chart layout helpers."""

from __future__ import annotations

import time

from ollama_sentinel.ui_charts import (
    CHART_WIDTH,
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
