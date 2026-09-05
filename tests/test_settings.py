"""User settings: sparse storage, precedence over .env, and the GUI registry."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from ollama_sentinel import settings as S


@dataclass
class FakeConfig:
    advisor: bool = True
    proc_vram: bool = True
    metrics: bool = True
    gaming_yield: bool = False


class StoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "settings.json"

    def test_missing_file_is_empty_not_defaults(self):
        # "Nothing stored" must stay distinguishable from "stored as default",
        # or .env could never be the source for an untouched setting.
        self.assertEqual(S.load_settings(self.path), {})

    def test_round_trip(self):
        S.set_setting("advisor", False, self.path)
        self.assertEqual(S.load_settings(self.path), {"advisor": False})

    def test_only_changed_keys_are_stored(self):
        S.set_setting("notifications", False, self.path)
        stored = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(list(stored), ["notifications"])

    def test_corrupt_file_reads_as_empty(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(S.load_settings(self.path), {})

    def test_unknown_keys_are_dropped_on_read(self):
        self.path.write_text('{"advisor": false, "removed_flag": true}', encoding="utf-8")
        self.assertEqual(S.load_settings(self.path), {"advisor": False})

    def test_unknown_key_cannot_be_set(self):
        with self.assertRaises(KeyError):
            S.set_setting("no_such_setting", True, self.path)

    def test_write_is_atomic(self):
        # A torn write would silently reset every toggle on next read.
        S.set_setting("advisor", False, self.path)
        self.assertFalse(self.path.with_suffix(".json.tmp").exists())
        self.assertEqual(S.load_settings(self.path), {"advisor": False})


class CoercionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "settings.json"

    def test_bools_are_coerced(self):
        S.set_setting("advisor", "yes", self.path)
        self.assertIs(S.load_settings(self.path)["advisor"], True)

    def test_numbers_are_clamped_to_range(self):
        S.set_setting("update_idle_seconds", 1, self.path)
        self.assertEqual(S.load_settings(self.path)["update_idle_seconds"], 60)
        S.set_setting("update_idle_seconds", 10**9, self.path)
        self.assertEqual(S.load_settings(self.path)["update_idle_seconds"], 86_400)

    def test_garbage_number_falls_back_to_default(self):
        # A GUI text field can hand us "" or "abc"; that must not persist as 0
        # and silently make every server look idle.
        S.set_setting("update_idle_seconds", "abc", self.path)
        self.assertEqual(S.load_settings(self.path)["update_idle_seconds"], 900)


class PrecedenceTest(unittest.TestCase):
    def test_stored_value_wins(self):
        cfg = FakeConfig(advisor=True)
        self.assertFalse(S.effective("advisor", cfg, {"advisor": False}))

    def test_env_config_used_when_unset(self):
        cfg = FakeConfig(advisor=False)
        self.assertFalse(S.effective("advisor", cfg, {}))

    def test_declared_default_when_nothing_else(self):
        self.assertTrue(S.effective("advisor", None, {}))

    def test_gui_only_setting_ignores_same_named_config_attr(self):
        # ctx_pressure has no config_attr; it must not read an unrelated
        # attribute that happens to share the name.
        cfg = FakeConfig()
        setattr(cfg, "ctx_pressure", False)
        self.assertTrue(S.effective("ctx_pressure", cfg, {}))

    def test_apply_to_config_only_touches_stored_keys(self):
        cfg = FakeConfig(advisor=True, metrics=True)
        S.apply_to_config(cfg, {"advisor": False})
        self.assertFalse(cfg.advisor)
        self.assertTrue(cfg.metrics)  # untouched, so .env still owns it

    def test_apply_ignores_settings_without_a_config_attr(self):
        cfg = FakeConfig()
        S.apply_to_config(cfg, {"update_auto_apply": True})
        self.assertFalse(hasattr(cfg, "update_auto_apply"))


class RegistryTest(unittest.TestCase):
    def test_keys_are_unique(self):
        keys = [s.key for s in S.SETTINGS]
        self.assertEqual(len(keys), len(set(keys)))

    def test_every_setting_has_help_text(self):
        for s in S.SETTINGS:
            self.assertTrue(s.label.strip(), s.key)
            self.assertTrue(s.help.strip(), s.key)

    def test_number_settings_declare_a_range(self):
        for s in S.SETTINGS:
            if s.kind == "number":
                self.assertIsNotNone(s.minimum, s.key)
                self.assertIsNotNone(s.maximum, s.key)

    def test_config_attrs_exist_on_appconfig(self):
        from ollama_sentinel.config import AppConfig

        cfg = AppConfig()
        for s in S.SETTINGS:
            if s.config_attr:
                self.assertTrue(hasattr(cfg, s.config_attr), s.key)

    def test_auto_apply_is_off_by_default(self):
        # It takes the API down for remote clients; opting in must be deliberate.
        self.assertFalse(S.BY_KEY["update_auto_apply"].default)

    def test_sections_cover_every_setting_once(self):
        seen = [s.key for _, group in S.sections() for s in group]
        self.assertEqual(sorted(seen), sorted(s.key for s in S.SETTINGS))


class AutoApplyTest(unittest.TestCase):
    def setUp(self):
        from ollama_sentinel.ollama_update import reset_auto_apply_guard

        reset_auto_apply_guard()
        self.addCleanup(reset_auto_apply_guard)

    def snapshot(self, **kw):
        base = {"reachable": True, "models": [], "version": "0.33.2"}
        base.update(kw)
        return base

    def test_disabled_does_nothing(self):
        from ollama_sentinel.ollama_update import maybe_auto_apply

        started, reason = maybe_auto_apply(self.snapshot(), enabled=False, idle_seconds=900)
        self.assertFalse(started)
        self.assertIn("disabled", reason)

    def test_busy_server_is_not_interrupted(self):
        from pathlib import Path
        from unittest.mock import patch

        from ollama_sentinel.ollama_update import UpdateStatus, maybe_auto_apply

        staged = UpdateStatus(
            running_version="0.33.2",
            staged_version="0.33.3",
            installer=Path("OllamaSetup.exe"),
        )
        with patch("ollama_sentinel.ollama_update.update_status", return_value=staged):
            started, reason = maybe_auto_apply(
                self.snapshot(models=[{"name": "m"}]), enabled=True, idle_seconds=900
            )
        self.assertFalse(started)
        self.assertIn("mid-conversation", reason)

    def test_latch_prevents_a_second_installer(self):
        import ollama_sentinel.ollama_update as U

        U._APPLIED_THIS_PROCESS = True
        started, reason = U.maybe_auto_apply(self.snapshot(), enabled=True, idle_seconds=900)
        self.assertFalse(started)
        self.assertIn("already applied", reason)


if __name__ == "__main__":
    unittest.main()
