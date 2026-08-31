"""Per-process VRAM tests (synthetic)."""

import json
import unittest
from unittest.mock import MagicMock, patch

from ollama_sentinel.proc_vram import (
    PID_RE,
    ProcessVramCollector,
    _parse_counter_json,
    _query_linux,
    _query_windows,
    query_process_vram,
)


class TestProcVramParsing(unittest.TestCase):
    def test_pid_regex(self):
        self.assertEqual(PID_RE.search("pid_45188_luid_0x00000000_0x0000ABCD_phys_0").group(1), "45188")

    def test_parse_counter_json_single(self):
        data = _parse_counter_json('{"Instance":"pid_1_x","Kind":"local","Value":100}')
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["Value"], 100)

    def test_parse_counter_json_array(self):
        data = _parse_counter_json('[{"Instance":"a","Value":1},{"Instance":"b","Value":2}]')
        self.assertEqual(len(data), 2)


class TestProcVramWindows(unittest.TestCase):
    @patch("ollama_sentinel.proc_vram._resolve_process_name", return_value="llama-server")
    @patch("ollama_sentinel.proc_vram.subprocess.run")
    def test_sums_instances_per_pid(self, mock_run, mock_name):
        counter_out = json.dumps(
            [
                {"Instance": "pid_100_luid_0_phys_0", "Kind": "local", "Value": 1_000_000_000},
                {"Instance": "pid_100_luid_1_phys_0", "Kind": "local", "Value": 500_000_000},
                {"Instance": "pid_100_luid_0_phys_0", "Kind": "nonlocal", "Value": 100_000_000},
                {"Instance": "pid_200_luid_0_phys_0", "Kind": "local", "Value": 10_000_000},
            ]
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=counter_out, stderr="")
        rows = _query_windows(min_bytes=64 * 1024 * 1024)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pid"], 100)
        self.assertEqual(rows[0]["bytes"], 1_500_000_000)
        self.assertEqual(rows[0]["non_local_bytes"], 100_000_000)
        self.assertEqual(rows[0]["name"], "llama-server")

    @patch("ollama_sentinel.proc_vram._resolve_process_name", return_value="pid 999 (exited)")
    @patch("ollama_sentinel.proc_vram.subprocess.run")
    def test_exited_pid_name(self, mock_run, mock_name):
        counter_out = json.dumps(
            [{"Instance": "pid_999_luid_0_phys_0", "Kind": "local", "Value": 200_000_000}]
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=counter_out, stderr="")
        rows = _query_windows(min_bytes=64 * 1024 * 1024)
        self.assertEqual(rows[0]["name"], "pid 999 (exited)")


class TestProcVramLinux(unittest.TestCase):
    @patch("ollama_sentinel.proc_vram.subprocess.run")
    def test_linux_compute_apps(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="45188, llama-server, 20480\n2888, dwm, 512\n",
            stderr="",
        )
        rows = _query_linux(min_bytes=64 * 1024 * 1024)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["pid"], 45188)
        self.assertIsNone(rows[0]["non_local_bytes"])


class TestProcVramDispatch(unittest.TestCase):
    @patch("ollama_sentinel.proc_vram._query_windows", return_value=[])
    @patch("sys.platform", "win32")
    def test_windows_dispatch(self, mock_win):
        query_process_vram()
        mock_win.assert_called_once()

    @patch("ollama_sentinel.proc_vram._query_linux", return_value=[])
    @patch("sys.platform", "linux")
    def test_linux_dispatch(self, mock_lin):
        query_process_vram()
        mock_lin.assert_called_once()


class TestProcessVramCollector(unittest.TestCase):
    @patch("ollama_sentinel.proc_vram.query_process_vram", side_effect=RuntimeError("boom"))
    def test_keeps_last_good_on_error(self, mock_query):
        collector = ProcessVramCollector(interval=30, enabled=True)
        collector._cache = {
            "rows": [{"pid": 1, "name": "a", "bytes": 100, "non_local_bytes": 0}],
            "polled_at": "x",
            "polled_at_ts": 1.0,
            "error": None,
            "stale": False,
        }
        collector._poll_once()
        snap = collector.get_snapshot()
        self.assertTrue(snap["stale"])
        self.assertEqual(len(snap["rows"]), 1)
        self.assertIn("boom", snap["error"])


if __name__ == "__main__":
    unittest.main()
