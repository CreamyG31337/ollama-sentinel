"""Spill guard: which models count as too big, debounce, gating and notification throttle."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ollama_sentinel import settings as S
from ollama_sentinel.config import config_from_env
from ollama_sentinel.spill_guard import NOTIFY_COOLDOWN_SEC, SpillGuard, oversize_models

GB = 1_000_000_000

FITS = {"name": "small:4b", "size": 5 * GB, "size_vram": 5 * GB}
SLIGHT = {"name": "mid:9b", "size": 10 * GB, "size_vram": 9 * GB}  # 90% GPU
HUGE = {"name": "big:27b", "size": 17 * GB, "size_vram": 7 * GB}  # 41% GPU


class OversizeModelsTest(unittest.TestCase):
    def test_only_models_below_threshold(self):
        out = oversize_models([FITS, SLIGHT, HUGE], 75)
        self.assertEqual([m["name"] for m in out], ["big:27b"])
        self.assertEqual(out[0]["gpu_pct"], 41)

    def test_small_spill_is_tolerated_but_threshold_is_adjustable(self):
        self.assertEqual(oversize_models([SLIGHT], 75), [])
        self.assertEqual([m["name"] for m in oversize_models([SLIGHT], 95)], ["mid:9b"])

    def test_zero_size_and_nameless_rows_are_ignored(self):
        rows = [{"name": "x", "size": 0, "size_vram": 0}, {"size": 10 * GB, "size_vram": 0}]
        self.assertEqual(oversize_models(rows, 75), [])


class GuardHarness:
    def __init__(self, models, *, enabled=True, threshold=75, confirm=2):
        self.models = models
        self.enabled = enabled
        self.unloaded: list[str] = []
        self.notified: list[str] = []
        self.now = 0.0
        self._tmp = tempfile.TemporaryDirectory()
        self.guard = SpillGuard(
            server="local",
            url="http://127.0.0.1:11434",
            confirm_polls=confirm,
            enabled_fn=lambda: self.enabled,
            min_gpu_pct_fn=lambda: threshold,
            list_models=lambda: self.models,
            unload_fn=self._unload,
            notify_fn=lambda t: self.notified.append(t.message),
            log_path=Path(self._tmp.name) / "spill_guard.jsonl",
            clock=lambda: self.now,
        )

    def _unload(self, url, model):
        self.unloaded.append(model)
        return {"done": True, "model": model}

    def close(self):
        self._tmp.cleanup()


class SpillGuardTest(unittest.TestCase):
    def harness(self, models, **kw):
        h = GuardHarness(models, **kw)
        self.addCleanup(h.close)
        return h

    def test_needs_consecutive_polls_before_unloading(self):
        # A runner caught mid-load must not be evicted on a single sighting.
        h = self.harness([FITS, HUGE])
        self.assertEqual(h.guard.poll_once(), [])
        self.assertEqual(h.unloaded, [])
        h.guard.poll_once()
        self.assertEqual(h.unloaded, ["big:27b"])

    def test_streak_resets_when_model_recovers(self):
        h = self.harness([HUGE])
        h.guard.poll_once()
        h.models = [FITS]
        h.guard.poll_once()
        h.models = [HUGE]
        h.guard.poll_once()
        self.assertEqual(h.unloaded, [])

    def test_fitting_models_are_never_touched(self):
        h = self.harness([FITS, SLIGHT])
        for _ in range(5):
            h.guard.poll_once()
        self.assertEqual(h.unloaded, [])

    def test_disabled_guard_does_nothing(self):
        h = self.harness([HUGE], enabled=False)
        for _ in range(5):
            h.guard.poll_once()
        self.assertEqual(h.unloaded, [])

    def test_unreachable_server_resets_streaks(self):
        h = self.harness([HUGE])
        h.guard.poll_once()
        h.models = None
        h.guard.poll_once()
        h.models = [HUGE]
        h.guard.poll_once()
        self.assertEqual(h.unloaded, [])

    def test_notifications_are_throttled_per_model(self):
        # A client that retries in a loop reloads the model every time.
        h = self.harness([HUGE], confirm=1)
        h.guard.poll_once()
        h.guard.poll_once()
        self.assertEqual(h.unloaded, ["big:27b", "big:27b"])
        self.assertEqual(len(h.notified), 1)
        self.assertIn("big:27b", h.notified[0])
        h.now += NOTIFY_COOLDOWN_SEC + 1
        h.guard.poll_once()
        self.assertEqual(len(h.notified), 2)


class SpillGuardSettingsTest(unittest.TestCase):
    def test_off_by_default(self):
        # It evicts models other clients may be using; opting in must be deliberate.
        self.assertFalse(S.BY_KEY["spill_guard"].default)
        self.assertFalse(config_from_env({}).spill_guard)

    def test_env_enables_and_sets_threshold(self):
        cfg = config_from_env({"SPILL_GUARD": "1", "SPILL_GUARD_MIN_GPU_PCT": "60"})
        self.assertTrue(cfg.spill_guard)
        self.assertEqual(cfg.spill_guard_min_gpu_pct, 60.0)
        self.assertEqual(S.effective("spill_guard_min_gpu_pct", cfg, {}), 60.0)

    def test_gui_value_wins_over_env(self):
        cfg = config_from_env({"SPILL_GUARD": "1"})
        self.assertFalse(S.effective("spill_guard", cfg, {"spill_guard": False}))


if __name__ == "__main__":
    unittest.main()
