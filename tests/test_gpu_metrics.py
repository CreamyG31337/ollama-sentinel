"""Tests for GPU metric severity / freshness labeling."""

from __future__ import annotations

import unittest

from ollama_sentinel.telemetry import (
    format_freshness_line,
    freshness_level,
    gpu_metric_rows,
    metric_severity,
)


class TestFreshness(unittest.TestCase):
    def test_levels(self):
        self.assertEqual(freshness_level(100.0, 5.0, 103.0), "ok")
        self.assertEqual(freshness_level(100.0, 5.0, 106.0), "aging")
        self.assertEqual(freshness_level(100.0, 5.0, 116.0), "stale")
        self.assertEqual(freshness_level(None, 5.0, 100.0), "unknown")

    def test_format_line_live_and_activity(self):
        t0 = 1_700_000_000.0
        level, label = format_freshness_line(
            t0, 5.0, t0 + 2.0, reachable=True, live_at=t0 + 1.5
        )
        self.assertEqual(level, "ok")
        self.assertIn("Live", label)
        self.assertIn("activity 0s ago", label)

    def test_format_line_unreachable(self):
        t0 = 1_700_000_000.0
        level, label = format_freshness_line(t0, 5.0, t0 + 2.0, reachable=False)
        self.assertEqual(level, "stale")
        self.assertIn("Unreachable", label)


class TestMetricSeverity(unittest.TestCase):
    def test_temperature(self):
        self.assertEqual(metric_severity("temperature", 45), "ok")
        self.assertEqual(metric_severity("temperature", 65), "busy")
        self.assertEqual(metric_severity("temperature", 78), "warn")
        self.assertEqual(metric_severity("temperature", 90), "alarm")
        self.assertEqual(metric_severity("temperature", None), "muted")

    def test_power_vs_limit(self):
        self.assertEqual(metric_severity("power", 50, ref=350), "muted")
        self.assertEqual(metric_severity("power", 200, ref=350), "ok")
        self.assertEqual(metric_severity("power", 300, ref=350), "warn")
        self.assertEqual(metric_severity("power", 340, ref=350), "alarm")

    def test_vram_free(self):
        self.assertEqual(metric_severity("vram_free_pct", 40), "ok")
        self.assertEqual(metric_severity("vram_free_pct", 10), "warn")
        self.assertEqual(metric_severity("vram_free_pct", 2), "alarm")

    def test_gpu_metric_rows_include_icons_and_severity(self):
        gpu = {
            "memory_used": 20 * 1024**3,
            "memory_total": 24 * 1024**3,
            "memory_free": 4 * 1024**3,
            "memory_free_pct": 100 * 4 / 24,
            "memory_reserved": 254 * 1024**2,
            "temperature": 78,
            "fan_speed": 55,
            "utilization": 92,
            "memory_utilization": 40,
            "power_draw": 320,
            "power_limit": 350,
            "pstate": "P0",
            "clock_sm": 1695,
            "clock_mem": 9751,
        }
        rows = {r["key"]: r for r in gpu_metric_rows(gpu)}
        self.assertEqual(rows["temperature"]["icon"], "thermostat")
        self.assertEqual(rows["temperature"]["severity"], "warn")
        self.assertEqual(rows["gpu_util"]["severity"], "busy")
        self.assertEqual(rows["power"]["severity"], "warn")
        self.assertIn("°C", rows["temperature"]["value"])


if __name__ == "__main__":
    unittest.main()
