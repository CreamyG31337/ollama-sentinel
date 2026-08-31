"""Telemetry helper tests."""

import unittest

from ollama_sentinel.telemetry import (
    enrich_gpu,
    format_field,
    format_poll_age,
    format_throttle,
    is_stale,
)


class TestTelemetry(unittest.TestCase):
    def test_enrich_gpu_free(self):
        gpu = enrich_gpu({"memory_used": 20 * 1024**3, "memory_total": 24 * 1024**3})
        self.assertEqual(gpu["memory_free"], 4 * 1024**3)
        self.assertAlmostEqual(gpu["memory_free_pct"], 100 * 4 / 24, places=1)

    def test_enrich_gpu_missing(self):
        gpu = enrich_gpu({"memory_used": None, "memory_total": 24 * 1024**3})
        self.assertIsNone(gpu["memory_free"])
        self.assertIsNone(gpu["memory_free_pct"])

    def test_is_stale(self):
        self.assertFalse(is_stale(100.0, 5.0, 110.0))
        self.assertFalse(is_stale(100.0, 5.0, 115.0))
        self.assertTrue(is_stale(100.0, 5.0, 116.0))
        self.assertTrue(is_stale(None, 5.0, 100.0))

    def test_format_poll_age(self):
        text = format_poll_age(1_700_000_000.0, 1_700_000_003.0)
        self.assertIn("3s ago", text)

    def test_format_field(self):
        self.assertEqual(format_field(None), "—")
        self.assertEqual(format_field(50, "%"), "50%")
        self.assertEqual(format_field(38.4, " W"), "38.4 W")

    def test_format_throttle(self):
        self.assertIsNone(format_throttle({}))
        self.assertIn("thermal", format_throttle({"throttle_hw_thermal": "Active"}) or "")
        self.assertIn("power cap", format_throttle({"throttle_sw_power_cap": "Active"}) or "")


if __name__ == "__main__":
    unittest.main()
