"""Pending-update detection and the idle gate that guards applying one.

The fixture shapes are the real ones from this host on 2026-09-04: Ollama 0.33.2
running, 0.33.3 staged in updates_v2/<sha>/OllamaSetup.exe.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from ollama_sentinel.advisor import evaluate_advisories
from ollama_sentinel.ollama_update import (
    INSTALL_ARGS,
    UpdateStatus,
    apply_update,
    find_staged_installer,
    format_update_status_line,
    idle_verdict,
    staged_version_from_app_log,
    update_status,
)

APP_LOG_LINE = (
    'time=2026-09-04T14:22:05.520-07:00 level=INFO source=updater.go:133 '
    'msg="New update available at '
    'https://github.com/ollama/ollama/releases/download/v{v}/OllamaSetup.exe"\n'
)


def gin_time(seconds_ago: float) -> str:
    ts = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return ts.strftime("%Y/%m/%d - %H:%M:%S")


class HomeFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = Path(self._tmp.name)

    def stage(self, sha: str = "a" * 8, name: str = "OllamaSetup.exe") -> Path:
        d = self.home / "updates_v2" / sha
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        p.write_bytes(b"installer")
        return p

    def app_log(self, *versions: str):
        (self.home / "app.log").write_text(
            "".join(APP_LOG_LINE.format(v=v) for v in versions), encoding="utf-8"
        )


class DetectionTest(HomeFixture):
    def test_finds_staged_installer_and_version(self):
        self.stage()
        self.app_log("0.33.3")
        st = update_status(running_version="0.33.2", home=self.home)
        self.assertTrue(st.pending)
        self.assertEqual(st.staged_version, "0.33.3")
        self.assertIn("0.33.3", st.summary)

    def test_nothing_staged_is_not_pending(self):
        st = update_status(running_version="0.33.2", home=self.home)
        self.assertFalse(st.pending)
        self.assertIn("current", st.summary)

    def test_installer_for_the_running_version_is_not_pending(self):
        # Ollama does not always clean up after applying an update.
        self.stage()
        self.app_log("0.33.2")
        self.assertFalse(update_status(running_version="0.33.2", home=self.home).pending)

    def test_older_staged_build_is_not_an_update(self):
        self.stage()
        self.app_log("0.33.1")
        self.assertFalse(update_status(running_version="0.33.2", home=self.home).pending)

    def test_unparseable_version_still_counts_as_pending(self):
        # The file only exists because Ollama downloaded it; failing to read the
        # version must not hide the thing we were asked to watch for.
        self.stage()
        st = update_status(running_version="0.33.2", home=self.home)
        self.assertIsNone(st.staged_version)
        self.assertTrue(st.pending)

    def test_newest_bundle_wins(self):
        import os
        import time

        old = self.stage(sha="old")
        time.sleep(0.01)
        new = self.stage(sha="new")
        os.utime(old, (time.time() - 500, time.time() - 500))
        self.assertEqual(find_staged_installer(self.home), new)

    def test_app_log_takes_the_latest_announcement(self):
        self.app_log("0.33.1", "0.33.2", "0.33.3")
        self.assertEqual(staged_version_from_app_log(self.home), "0.33.3")

    def test_missing_app_log_is_none_not_an_error(self):
        self.assertIsNone(staged_version_from_app_log(self.home))

    def test_multi_digit_versions_compare_numerically(self):
        # String compare would rank "0.9.0" above "0.10.0".
        self.assertTrue(
            UpdateStatus("0.9.0", "0.10.0", Path("x")).pending
        )
        self.assertFalse(
            UpdateStatus("0.10.0", "0.9.0", Path("x")).pending
        )


class IdleGateTest(unittest.TestCase):
    """This server has remote consumers, so the gate is deliberately strict."""

    def test_unreachable_server_is_never_touched(self):
        v = idle_verdict({"reachable": False})
        self.assertFalse(v.idle)
        self.assertIn("unreachable", v.reason)

    def test_loaded_model_blocks(self):
        v = idle_verdict({"reachable": True, "models": [{"name": "qwen3.8:27b-heretic"}]})
        self.assertFalse(v.idle)
        self.assertIn("qwen3.8:27b-heretic", v.reason)

    def test_generating_blocks(self):
        activity = SimpleNamespace(phase="generating", recent_requests=[])
        v = idle_verdict({"reachable": True, "models": []}, activity)
        self.assertFalse(v.idle)
        self.assertIn("generating", v.reason)

    def test_recent_request_blocks(self):
        activity = SimpleNamespace(
            phase="idle", recent_requests=[SimpleNamespace(at=gin_time(120))]
        )
        v = idle_verdict({"reachable": True, "models": []}, activity, idle_seconds=900)
        self.assertFalse(v.idle)
        self.assertIn("120s ago", v.reason)

    def test_old_request_allows(self):
        activity = SimpleNamespace(
            phase="idle", recent_requests=[SimpleNamespace(at=gin_time(3600))]
        )
        self.assertTrue(idle_verdict({"reachable": True, "models": []}, activity).idle)

    def test_quiet_server_allows(self):
        activity = SimpleNamespace(phase="idle", recent_requests=[])
        self.assertTrue(idle_verdict({"reachable": True, "models": []}, activity).idle)

    def test_unparseable_timestamp_does_not_fake_idleness(self):
        # A garbled line must not be read as "no recent traffic"; it is simply
        # not evidence either way, and the other gates still apply.
        activity = SimpleNamespace(phase="idle", recent_requests=[SimpleNamespace(at="???")])
        self.assertTrue(idle_verdict({"reachable": True, "models": []}, activity).idle)


class ApplyTest(HomeFixture):
    def test_dry_run_reports_the_real_command_without_running(self):
        installer = self.stage()
        started, msg = apply_update(installer, dry_run=True)
        self.assertFalse(started)
        for flag in INSTALL_ARGS:
            self.assertIn(flag, msg)

    def test_missing_installer_is_reported(self):
        started, msg = apply_update(self.home / "nope.exe")
        self.assertFalse(started)
        self.assertIn("missing", msg)

    def test_flags_match_ollamas_own_upgrade(self):
        # Silent + close-running-apps, so it behaves like a tray-click upgrade.
        self.assertIn("/VERYSILENT", INSTALL_ARGS)
        self.assertIn("/FORCECLOSEAPPLICATIONS", INSTALL_ARGS)
        self.assertIn("/SUPPRESSMSGBOXES", INSTALL_ARGS)


class AdvisorTest(unittest.TestCase):
    def snapshot(self):
        return {"reachable": True, "server": "local", "models": [], "tags": [], "gpus": []}

    def test_pending_update_is_info_not_an_alarm(self):
        from ollama_sentinel.advisor import evaluate_advisor_alarms

        st = UpdateStatus("0.33.2", "0.33.3", Path("x"))
        findings = evaluate_advisories(
            self.snapshot(), update_status=st, gpu_data_available=False
        )
        hit = [f for f in findings if f.id == "config:update_pending:local"]
        self.assertTrue(hit)
        self.assertEqual(hit[0].severity, "info")
        # An alarm that cannot clear until someone restarts a service gets muted.
        self.assertNotIn(
            "config:update_pending:local", {a["id"] for a in evaluate_advisor_alarms(findings)}
        )

    def test_remedy_corrects_the_restart_misconception(self):
        st = UpdateStatus("0.33.2", "0.33.3", Path("x"))
        findings = evaluate_advisories(
            self.snapshot(), update_status=st, gpu_data_available=False
        )
        remedy = next(f.remedy for f in findings if f.id == "config:update_pending:local")
        self.assertIn("does NOT apply", remedy)

    def test_no_finding_when_current(self):
        st = UpdateStatus("0.33.2", None, None)
        ids = {
            f.id
            for f in evaluate_advisories(
                self.snapshot(), update_status=st, gpu_data_available=False
            )
        }
        self.assertNotIn("config:update_pending:local", ids)


class FormatStatusLineTest(unittest.TestCase):
    def test_hidden_when_nothing_pending(self):
        text, key = format_update_status_line(
            pending=False,
            summary="Ollama 0.33.2 is current",
            auto_apply=True,
            started=False,
            reason="no update staged",
        )
        self.assertEqual(text, "")
        self.assertEqual(key, "muted")

    def test_auto_apply_shows_idle_refusal(self):
        text, key = format_update_status_line(
            pending=True,
            summary="Ollama 0.33.3 downloaded and waiting (running 0.33.2)",
            auto_apply=True,
            started=False,
            reason="model loaded (qwen) — someone may be mid-conversation",
        )
        self.assertIn("model loaded", text)
        self.assertIn("Update pending", text)
        self.assertEqual(key, "warn")

    def test_manual_mode_shows_summary_only(self):
        text, key = format_update_status_line(
            pending=True,
            summary="Ollama 0.33.3 downloaded and waiting (running 0.33.2)",
            auto_apply=False,
            started=False,
            reason="",
        )
        self.assertIn("0.33.3", text)
        self.assertEqual(key, "muted")

    def test_started_installer(self):
        text, key = format_update_status_line(
            pending=True,
            summary="…",
            auto_apply=True,
            started=True,
            reason="started OllamaSetup.exe (~1 min; the API drops while it runs)",
        )
        self.assertIn("started", text)
        self.assertEqual(key, "warn")


class NestedYamlTest(unittest.TestCase):
    """The probe must read a real nested config, not a flattened approximation."""

    def parse(self, text: str):
        from ollama_sentinel.client_probe import _parse_simple_yaml_ints

        return _parse_simple_yaml_ints(text)

    def test_nesting_is_preserved(self):
        got = self.parse(
            "providers:\n"
            "  local-ollama:\n"
            "    models:\n"
            '      "qwen3.8:27b-heretic":\n'
            "        context_length: 64000\n"
        )
        self.assertEqual(
            got, {"providers": {"local-ollama": {"models": {"qwen3.8:27b-heretic": {"context_length": 64000}}}}}
        )

    def test_sibling_providers_do_not_merge(self):
        got = self.parse(
            "providers:\n"
            "  a:\n"
            "    ctx: 1000\n"
            "  b:\n"
            "    ctx: 2000\n"
        )
        self.assertEqual(got["providers"]["a"]["ctx"], 1000)
        self.assertEqual(got["providers"]["b"]["ctx"], 2000)

    def test_dedent_back_to_top_level(self):
        got = self.parse("a:\n  x: 1\nb: 2\n")
        self.assertEqual(got, {"a": {"x": 1}, "b": 2})

    def test_dotted_lookup_selects_one_provider(self):
        from ollama_sentinel.client_probe import _descend, _max_int_leaf

        got = self.parse(
            "providers:\n"
            "  local-ollama:\n"
            "    models:\n"
            '      "qwen3.8:27b-heretic":\n'
            "        context_length: 64000\n"
            "  other:\n"
            "    models:\n"
            "      big:\n"
            "        context_length: 1000000\n"
        )
        node = _descend(got, "providers.local-ollama.models")
        self.assertEqual(_max_int_leaf(node), 64000)


if __name__ == "__main__":
    unittest.main()
