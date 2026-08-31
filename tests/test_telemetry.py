"""Telemetry helper tests."""

import unittest
from datetime import datetime, timezone, timedelta

from ollama_sentinel.telemetry import (
    enrich_gpu,
    format_expires_display,
    format_field,
    format_poll_age,
    format_relative_delta,
    format_throttle,
    format_ts_local,
    is_local_server_url,
    is_stale,
    parse_rfc3339,
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
        self.assertNotIn(".", text.split("(")[0])

    def test_format_field(self):
        self.assertEqual(format_field(None), "—")
        self.assertEqual(format_field(50, "%"), "50%")
        self.assertEqual(format_field(38.4, " W"), "38.4 W")

    def test_format_throttle(self):
        self.assertIsNone(format_throttle({}))
        self.assertIn("thermal", format_throttle({"throttle_hw_thermal": "Active"}) or "")
        self.assertIn("power cap", format_throttle({"throttle_sw_power_cap": "Active"}) or "")

    def test_is_local_server_url(self):
        self.assertTrue(is_local_server_url("http://127.0.0.1:11434"))
        self.assertTrue(is_local_server_url("http://localhost:11434"))
        self.assertFalse(is_local_server_url("http://100.75.27.13:11434"))

    def test_parse_rfc3339_nanoseconds(self):
        dt = parse_rfc3339("2026-08-30T19:17:21.227279507-08:00")
        self.assertIsNotNone(dt)
        assert dt is not None
        self.assertEqual(dt.second, 21)
        self.assertEqual(dt.microsecond, 227279)

    def test_parse_rfc3339_windows_seven_digit_fraction(self):
        dt = parse_rfc3339("2026-08-30T20:29:29.3914533-07:00")
        self.assertIsNotNone(dt)
        assert dt is not None
        self.assertEqual(dt.strftime("%H:%M:%S"), "20:29:29")
        self.assertEqual(dt.microsecond, 391453)

    def test_format_expires_never_returns_raw_iso(self):
        text = format_expires_display("2026-08-30T20:29:29.3914533-07:00")
        self.assertNotIn("T", text)
        self.assertNotIn(".391", text)
        self.assertIn("(", text)

    def test_format_expires_forever(self):
        self.assertEqual(
            format_expires_display("2318-01-01T00:00:00Z", server_url="http://127.0.0.1:11434"),
            "Forever",
        )

    def test_format_expires_local_relative_and_clock(self):
        tz = timezone(timedelta(hours=-8))
        now = datetime(2026, 8, 30, 18, 50, 0, tzinfo=tz)
        text = format_expires_display(
            "2026-08-30T19:17:21.227279507-08:00",
            server_url="http://127.0.0.1:11434",
            now=now,
        )
        self.assertTrue(text.startswith("in "))
        self.assertIn("(19:17:21)", text)
        self.assertNotIn(".", text)
        self.assertNotIn("-08:00", text)

    def test_format_expires_remote_converts_to_viewer_local(self):
        pacific = timezone(timedelta(hours=-8))
        mountain = timezone(timedelta(hours=-7))
        now = datetime(2026, 8, 30, 18, 50, 0, tzinfo=mountain)
        text = format_expires_display(
            "2026-08-30T20:00:00-08:00",
            server_url="http://100.75.27.13:11434",
            now=now,
        )
        self.assertIn("(21:00:00)", text)
        self.assertNotIn("(20:00:00)", text)

    def test_format_expires_malformed(self):
        self.assertEqual(format_expires_display(None), "—")
        self.assertEqual(format_expires_display("not-a-date"), "—")

    def test_format_relative_delta(self):
        tz = timezone.utc
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=tz)
        future = now + timedelta(minutes=27)
        self.assertEqual(format_relative_delta(future, now), "in 27m")
        past = now - timedelta(minutes=3)
        self.assertEqual(format_relative_delta(past, now), "expired 3m ago")

    def test_format_ts_local_no_fraction(self):
        ts = datetime(2026, 8, 30, 19, 17, 21, 500000, tzinfo=timezone.utc).timestamp()
        text = format_ts_local(ts)
        self.assertRegex(text, r"^\d{2}:\d{2}:\d{2}$")
        self.assertNotIn(".", text)


if __name__ == "__main__":
    unittest.main()
