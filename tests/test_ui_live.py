"""Helpers for the live (in-place, fingerprint-gated) UI paints."""

from __future__ import annotations

import unittest

from ollama_sentinel.ui_widgets import (
    LiveFreshnessBanner,
    activity_fingerprint,
    gpu_fingerprint,
)


def _activity(**overrides) -> dict:
    base = {
        "phase": "generating",
        "summary": "qwen3.8:27b-heretic generating",
        "stale": False,
        "model": "qwen3.8:27b-heretic",
        "n_gen": 1200,
        "gen_tps": 41.23,
        "gen_tps_3s": 40.11,
        "prompt_tokens": 8_000,
        "n_ctx_slot": 65_536,
        "ctx_fill": 0.122,
        "last_request": {
            "method": "POST",
            "path": "/v1/chat/completions",
            "client": "127.0.0.1",
            "client_name": "hermes",
            "duration_s": 0.42,
        },
        "recent_requests": [
            {
                "method": "POST",
                "path": "/api/chat",
                "client": "100.75.27.13",
                "client_name": "open-webui",
                "duration_s": 2.10,
            }
        ],
        "peers": [{"addr": "100.75.27.13:52100", "name": "open-webui"}],
        "runners": [
            {"pid": 4242, "busy": True, "engine_3d_pct": 91.2, "vram_bytes": 20_500_000_000}
        ],
    }
    base.update(overrides)
    return base


class ActivityFingerprintTests(unittest.TestCase):
    def test_identical_activity_gives_identical_fingerprint(self):
        self.assertEqual(activity_fingerprint(_activity()), activity_fingerprint(_activity()))

    def test_none_activity_gives_none(self):
        self.assertIsNone(activity_fingerprint(None))

    def test_n_gen_change_differs(self):
        self.assertNotEqual(
            activity_fingerprint(_activity(n_gen=1201)),
            activity_fingerprint(_activity(n_gen=1200)),
        )

    def test_phase_change_differs(self):
        self.assertNotEqual(
            activity_fingerprint(_activity(phase="idle", summary="idle")),
            activity_fingerprint(_activity()),
        )

    def test_peers_change_differs(self):
        changed = _activity(peers=[{"addr": "100.75.27.13:52101", "name": None}])
        self.assertNotEqual(activity_fingerprint(changed), activity_fingerprint(_activity()))

    def test_sub_display_tps_jitter_is_ignored(self):
        """tok/s noise below one decimal must not rebuild the card."""
        self.assertEqual(
            activity_fingerprint(_activity(gen_tps=41.24)),
            activity_fingerprint(_activity(gen_tps=41.19)),
        )

    def test_dataclass_and_its_dict_agree(self):
        from ollama_sentinel.activity import ServerActivity

        act = ServerActivity(phase="idle", summary="idle")
        self.assertEqual(activity_fingerprint(act), activity_fingerprint(act.to_dict()))


class LiveFreshnessBannerTests(unittest.TestCase):
    def test_set_mutates_in_place(self):
        banner = LiveFreshnessBanner(interval_s=5)
        control = banner.control
        banner.set("ok", "polled 2s ago")
        banner.set("stale", "polled 90s ago")
        self.assertIs(banner.control, control)
        self.assertEqual(banner.badge_text.value, "STALE")
        self.assertEqual(banner.label.value, "polled 90s ago")

    def test_repeat_identical_set_returns_false(self):
        banner = LiveFreshnessBanner()
        self.assertTrue(banner.set("ok", "polled 2s ago"))
        self.assertFalse(banner.set("ok", "polled 2s ago"))

    def test_level_change_returns_true_and_recolors(self):
        banner = LiveFreshnessBanner()
        banner.set("ok", "polled 2s ago")
        before_bg = banner.control.bgcolor
        self.assertTrue(banner.set("aging", "polled 2s ago"))
        self.assertNotEqual(banner.control.bgcolor, before_bg)


class GpuFingerprintTests(unittest.TestCase):
    def test_identical_gpus_match(self):
        gpus = [_gpu()]
        self.assertEqual(gpu_fingerprint(gpus), gpu_fingerprint([_gpu()]))

    def test_sub_display_jitter_is_ignored(self):
        jittered = _gpu()
        jittered["memory_used"] += 2 * 1024 * 1024  # 2 MB -> same 0.1 GB bucket
        jittered["utilization"] = 37.04  # rounds to 37.0 either way
        jittered["power_draw"] = 201.04
        self.assertEqual(gpu_fingerprint([_gpu()]), gpu_fingerprint([jittered]))

    def test_display_precision_change_differs(self):
        jittered = _gpu()
        jittered["utilization"] = 38.0
        self.assertNotEqual(gpu_fingerprint([_gpu()]), gpu_fingerprint([jittered]))

    def test_empty_and_none(self):
        self.assertEqual(gpu_fingerprint(None), ())
        self.assertEqual(gpu_fingerprint([]), ())


def _gpu() -> dict:
    return {
        "index": 0,
        "name": "NVIDIA GeForce RTX 3090",
        "temperature": 61.0,
        "fan_speed": 43,
        "utilization": 37.0,
        "memory_utilization": 88.0,
        "memory_used": 20_500_000_000,
        "memory_free": 3_500_000_000,
        "memory_total": 24_000_000_000,
        "memory_reserved": 21_000_000_000,
        "power_draw": 201.0,
        "power_limit": 350.0,
        "pstate": "P0",
        "clock_sm": 1950,
        "clock_mem": 9501,
        "throttle_hw_thermal": "N/A",
        "throttle_sw_power_cap": "Not Active",
    }


if __name__ == "__main__":
    unittest.main()
