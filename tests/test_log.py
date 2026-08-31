"""Alarm log rotation tests."""

import json
import tempfile
import unittest
from pathlib import Path

from ollama_sentinel.alarms import AlarmTransition
from ollama_sentinel.log import append_alarm_log


class TestLog(unittest.TestCase):
    def test_skips_when_no_alarms_once_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alarms.jsonl"
            self.assertFalse(
                append_alarm_log(path, alarms=[], on_transition_only=False)
            )
            self.assertFalse(path.exists())

    def test_logs_active_alarms_once_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alarms.jsonl"
            alarms = [{"id": "spill:local:m", "type": "spill", "message": "SPILL m"}]
            self.assertTrue(
                append_alarm_log(path, alarms=alarms, on_transition_only=False)
            )
            line = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(line["alarms"], alarms)
            self.assertNotIn("inventory", line)
            self.assertNotIn("snapshots", line)

    def test_live_mode_only_on_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alarms.jsonl"
            alarms = [{"id": "spill:local:m", "type": "spill", "message": "SPILL m"}]
            self.assertFalse(
                append_alarm_log(path, alarms=alarms, on_transition_only=True)
            )
            trans = [AlarmTransition("FIRE", "spill:local:m", "SPILL m")]
            self.assertTrue(
                append_alarm_log(
                    path, alarms=alarms, transitions=trans, on_transition_only=True
                )
            )
            line = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(line["transitions"][0]["kind"], "FIRE")

    def test_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alarms.jsonl"
            alarms = [{"id": "a", "type": "spill", "message": "x"}]
            for _ in range(5):
                append_alarm_log(
                    path,
                    alarms=alarms,
                    on_transition_only=False,
                    max_bytes=80,
                    backup_count=2,
                )
            self.assertTrue(path.exists())
            self.assertTrue(Path(f"{path}.1").exists())
            self.assertTrue(Path(f"{path}.2").exists())


if __name__ == "__main__":
    unittest.main()
