"""Synthetic alarm tests."""

import unittest

from ollama_sentinel.alarms import AlarmState, Thresholds, evaluate_alarms, format_expires, gpu_pct


def _snap(models=None, gpus=None, reachable=True, server="local"):
    return {
        "server": server,
        "reachable": reachable,
        "models": models or [],
        "gpus": gpus,
    }


class TestAlarms(unittest.TestCase):
    def test_fully_resident_no_spill(self):
        snap = _snap(models=[{"name": "m", "size": 1000, "size_vram": 1000}])
        active, _, _ = evaluate_alarms(snap, AlarmState(), Thresholds())
        self.assertEqual(active, [])

    def test_spilled_model(self):
        snap = _snap(models=[{"name": "m", "size": 1000, "size_vram": 600}])
        active, _, _ = evaluate_alarms(snap, AlarmState(), Thresholds())
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["type"], "spill")
        self.assertIn("40% CPU / 60% GPU", active[0]["message"])

    def test_paging_fires_on_third_poll(self):
        th = Thresholds(paging_polls=3, paging_power_w=200)
        gpu = [{"index": 0, "utilization": 90, "power_draw": 150, "power_limit": 350, "memory_used": 1e9, "memory_total": 2e10}]
        state = AlarmState()
        for i in range(2):
            active, state, trans = evaluate_alarms(_snap(gpus=gpu), state, th)
            self.assertEqual(active, [])
            self.assertEqual(trans, [])
        active, _, trans = evaluate_alarms(_snap(gpus=gpu), state, th)
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["type"], "paging")
        self.assertTrue(trans)

    def test_paging_resets_on_miss(self):
        th = Thresholds(paging_polls=3, paging_power_w=200)
        hot = [{"index": 0, "utilization": 90, "power_draw": 150, "power_limit": 350}]
        cool = [{"index": 0, "utilization": 90, "power_draw": 300, "power_limit": 350}]
        state = AlarmState()
        _, state, _ = evaluate_alarms(_snap(gpus=hot), state, th)
        _, state, _ = evaluate_alarms(_snap(gpus=hot), state, th)
        _, state, _ = evaluate_alarms(_snap(gpus=cool), state, th)
        active, _, _ = evaluate_alarms(_snap(gpus=hot), state, th)
        self.assertEqual(active, [])

    def test_vram_pressure(self):
        gpu = [{"index": 0, "memory_used": 24e9, "memory_total": 25e9}]
        active, _, _ = evaluate_alarms(_snap(gpus=gpu), AlarmState(), Thresholds())
        self.assertEqual(active[0]["type"], "vram")

    def test_unreachable_empty(self):
        active, _, _ = evaluate_alarms(_snap(reachable=False), AlarmState(), Thresholds())
        self.assertEqual(active, [])

    def test_hysteresis_fire_once(self):
        snap = _snap(models=[{"name": "m", "size": 100, "size_vram": 50}])
        state = AlarmState()
        _, state, t1 = evaluate_alarms(snap, state, Thresholds())
        self.assertEqual(len(t1), 1)
        _, state, t2 = evaluate_alarms(snap, state, Thresholds())
        self.assertEqual(t2, [])

    def test_forever_expires(self):
        self.assertEqual(format_expires("2318-01-01T00:00:00Z"), "Forever")

    def test_expires_human_readable(self):
        from datetime import datetime, timedelta, timezone

        tz = timezone(timedelta(hours=-8))
        now = datetime(2026, 8, 30, 18, 50, 0, tzinfo=tz)
        text = format_expires(
            "2026-08-30T19:17:21-08:00",
            server_url="http://127.0.0.1:11434",
            now=now,
        )
        self.assertIn("in ", text)
        self.assertIn("(19:17:21)", text)

    def test_gpu_pct(self):
        self.assertEqual(gpu_pct(100, 60), 60)


if __name__ == "__main__":
    unittest.main()
