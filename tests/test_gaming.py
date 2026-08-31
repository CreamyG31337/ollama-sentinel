"""Gaming yield decision and watcher tests (synthetic)."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from ollama_sentinel.gaming import (
    GamingHysteresis,
    GamingSignals,
    is_fullscreen_bounds,
    is_gaming,
    is_gaming_candidate,
    is_ollama_busy,
    parse_exclude_list,
    passes_solitaire_gate,
    should_unload,
)
from ollama_sentinel.gaming_yield import GamingYieldWatcher, build_signals


class TestDecisionRule(unittest.TestCase):
    def test_c_alone_never_fires(self) -> None:
        s = GamingSignals(in_game_list=True, solitaire_gate=True)
        self.assertFalse(is_gaming_candidate(s))
        self.assertFalse(is_gaming(s))

    def test_a_plus_e_fires(self) -> None:
        s = GamingSignals(exclusive_fullscreen=True, solitaire_gate=True, name="game.exe", pid=42)
        self.assertTrue(is_gaming_candidate(s))
        self.assertTrue(is_gaming(s))

    def test_solitaire_a_without_e(self) -> None:
        s = GamingSignals(exclusive_fullscreen=True, solitaire_gate=False, name="SolitaireCollection.exe")
        self.assertTrue(is_gaming_candidate(s))
        self.assertFalse(is_gaming(s))

    def test_exclude_list(self) -> None:
        s = GamingSignals(
            exclusive_fullscreen=True,
            solitaire_gate=True,
            name="SolitaireCollection.exe",
            pid=9,
        )
        self.assertFalse(is_gaming(s, exclude={"SolitaireCollection"}))

    def test_borderless_with_c_and_e(self) -> None:
        s = GamingSignals(
            borderless_fullscreen=True,
            in_game_list=True,
            solitaire_gate=True,
            name="PRAGMATA.exe",
            pid=1,
        )
        self.assertTrue(is_gaming(s))

    def test_borderless_needs_c_or_d(self) -> None:
        s = GamingSignals(borderless_fullscreen=True, solitaire_gate=True, name="app.exe")
        self.assertFalse(is_gaming_candidate(s))


class TestSolitaireGate(unittest.TestCase):
    def test_vram_gate(self) -> None:
        self.assertTrue(passes_solitaire_gate(2 * 1024**3, 0.0))
        self.assertFalse(passes_solitaire_gate(100_000_000, 0.0))

    def test_util_gate(self) -> None:
        self.assertTrue(passes_solitaire_gate(0, 55.0))
        self.assertFalse(passes_solitaire_gate(0, 10.0))


class TestHysteresis(unittest.TestCase):
    def test_fires_on_nth(self) -> None:
        h = GamingHysteresis(fire_n=2, clear_m=4)
        self.assertIsNone(h.update(True))
        self.assertEqual(h.update(True), "detected")
        self.assertTrue(h.active)

    def test_clears_after_m(self) -> None:
        h = GamingHysteresis(fire_n=2, clear_m=4)
        h.update(True)
        h.update(True)
        for _ in range(3):
            self.assertIsNone(h.update(False))
        self.assertEqual(h.update(False), "cleared")
        self.assertFalse(h.active)


class TestFullscreenBounds(unittest.TestCase):
    def test_maximized_not_fullscreen(self) -> None:
        # Measured: maximized terminal -7,-7 → 2568x1400 on 2560x1440
        self.assertFalse(
            is_fullscreen_bounds(-7, -7, 2561, 1393, 0, 0, 2560, 1440)
        )

    def test_exact_bounds_is_fullscreen(self) -> None:
        self.assertTrue(is_fullscreen_bounds(0, 0, 2560, 1440, 0, 0, 2560, 1440))


class TestUnloadGate(unittest.TestCase):
    def test_busy_blocks(self) -> None:
        self.assertTrue(is_ollama_busy(93.0, busy_util=20.0))
        self.assertFalse(should_unload(gaming=True, busy=True, yield_enabled=True))

    def test_yield_off_never_unloads(self) -> None:
        self.assertFalse(should_unload(gaming=True, busy=False, yield_enabled=False))

    def test_yield_on_when_idle(self) -> None:
        self.assertTrue(should_unload(gaming=True, busy=False, yield_enabled=True))


class TestBuildSignals(unittest.TestCase):
    def test_d_from_util(self) -> None:
        win = {
            "exclusive_fullscreen": False,
            "borderless_fullscreen": True,
            "in_game_list": False,
            "pid": 100,
            "name": "game.exe",
            "exe_path": r"C:\Games\game.exe",
        }
        rows = [{"pid": 100, "name": "game.exe", "bytes": 2 * 1024**3, "engine_3d_pct": 80.0}]
        s = build_signals(win, rows, min_vram=1536 * 1024**2, min_util=50.0)
        self.assertTrue(s.non_ollama_3d_util)
        self.assertTrue(s.solitaire_gate)


class TestWatcher(unittest.TestCase):
    def test_observe_no_unload(self) -> None:
        unload = MagicMock(return_value=[])
        win_calls = {"n": 0}

        def collect():
            win_calls["n"] += 1
            return {
                "exclusive_fullscreen": True,
                "borderless_fullscreen": False,
                "in_game_list": False,
                "pid": 55,
                "name": "Game.exe",
                "exe_path": r"C:\Game.exe",
            }

        with TemporaryDirectory() as tmp:
            watcher = GamingYieldWatcher(
                enabled=True,
                yield_enabled=False,
                interval=1,
                exclude=set(),
                min_vram_bytes=100,
                min_util=1.0,
                busy_util=20.0,
                ollama_url="http://127.0.0.1:11434",
                log_path=Path(tmp) / "gaming.jsonl",
                get_proc_snapshot=lambda: {
                    "rows": [
                        {"pid": 55, "name": "Game.exe", "bytes": 2 * 1024**3, "engine_3d_pct": 80.0}
                    ]
                },
                list_loaded_models=lambda: ["m1"],
                collect_win=collect,
                unload_fn=unload,
            )
            # Force win32 path for enabled flag in test by setting enabled after init... 
            # Constructor ANDs with win32. Override:
            watcher.enabled = True
            self.assertIsNone(watcher.poll_once())
            self.assertEqual(watcher.poll_once(), "detected")
            unload.assert_not_called()

    def test_yield_calls_unload_when_idle(self) -> None:
        unload = MagicMock(return_value=[{"model": "m1", "done_reason": "unload"}])

        def collect():
            return {
                "exclusive_fullscreen": True,
                "borderless_fullscreen": False,
                "in_game_list": False,
                "pid": 55,
                "name": "Game.exe",
                "exe_path": r"C:\Game.exe",
            }

        with TemporaryDirectory() as tmp:
            watcher = GamingYieldWatcher(
                enabled=True,
                yield_enabled=True,
                interval=1,
                exclude=set(),
                min_vram_bytes=100,
                min_util=1.0,
                busy_util=20.0,
                ollama_url="http://127.0.0.1:11434",
                log_path=Path(tmp) / "gaming.jsonl",
                get_proc_snapshot=lambda: {
                    "rows": [
                        {"pid": 55, "name": "Game.exe", "bytes": 2 * 1024**3, "engine_3d_pct": 80.0},
                        {"pid": 1, "name": "llama-server.exe", "bytes": 20 * 1024**3, "engine_3d_pct": 3.0},
                    ]
                },
                list_loaded_models=lambda: ["m1"],
                collect_win=collect,
                unload_fn=unload,
            )
            watcher.enabled = True
            watcher.poll_once()
            watcher.poll_once()  # detected
            # busy idle streak needs 2 polls while active
            watcher.poll_once()
            unload.assert_called()
            args = unload.call_args[0]
            self.assertEqual(args[1], ["m1"])


class TestParseExclude(unittest.TestCase):
    def test_parse(self) -> None:
        self.assertEqual(parse_exclude_list("SolitaireCollection, foo"), {"SolitaireCollection", "foo"})


if __name__ == "__main__":
    unittest.main()
