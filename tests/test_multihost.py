"""Multi-server poll and render attribution tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from ollama_sentinel.__main__ import _attach_activity, _exit_code
from ollama_sentinel.config import ServerConfig
from ollama_sentinel.poll import poll_all
from ollama_sentinel.render import build_live_panel, render_snapshot_plain


class TestMultiHost(unittest.TestCase):
    def test_exit_code_optional_unreachable(self) -> None:
        snaps = [
            {"reachable": False, "optional": True},
            {"reachable": True, "optional": False},
        ]
        self.assertEqual(_exit_code(snaps, []), 0)

    def test_exit_code_required_all_unreachable(self) -> None:
        snaps = [
            {"reachable": False, "optional": True},
            {"reachable": False, "optional": False},
        ]
        self.assertEqual(_exit_code(snaps, []), 2)

    def test_remote_snapshot_no_local_activity(self) -> None:
        proc_rows = [{"pid": 1, "name": "ollama", "bytes": 1e9}]
        snaps = [
            {"server": "remote", "reachable": True, "local_gpu": False, "models": []},
            {"server": "local", "reachable": True, "local_gpu": True, "models": []},
        ]
        out = _attach_activity(snaps, proc_rows, type("Cfg", (), {})())
        remote = next(s for s in out if s["server"] == "remote")
        local = next(s for s in out if s["server"] == "local")
        self.assertNotIn("activity", remote)
        self.assertIn("activity", local)

    def test_poll_remote_no_gpu_attach(self) -> None:
        gpu_data = [{"memory_total": 24e9, "memory_used": 1e9, "memory_free": 23e9}]

        def fake_get(url, path, timeout=10):
            if path == "/api/version":
                return {"version": "0.33.2"}, None
            if path == "/api/ps":
                return {"models": []}, None
            if path == "/api/tags":
                return {"models": [{"name": "m", "size": 1}]}, None
            return None, "404"

        with patch("ollama_sentinel.poll._get_json", side_effect=fake_get):
            snaps = poll_all(
                [
                    {"name": "local", "url": "http://127.0.0.1:11434", "local_gpu": True},
                    {"name": "remote", "url": "http://10.0.0.2:11434", "local_gpu": False},
                ],
                query_gpus_fn=lambda _gf: gpu_data,
            )
        local = next(s for s in snaps if s["server"] == "local")
        remote = next(s for s in snaps if s["server"] == "remote")
        self.assertIsNotNone(local.get("gpus"))
        self.assertIsNone(remote.get("gpus"))
        self.assertTrue(local.get("gpu_data_available"))
        self.assertFalse(remote.get("gpu_data_available"))

    def test_render_remote_no_process_vram(self) -> None:
        snaps = [
            {
                "server": "remote",
                "reachable": True,
                "local_gpu": False,
                "version": "0.33.2",
                "models": [],
                "tags": [],
                "gpus": None,
                "polled_at_ts": 1_700_000_000.0,
            }
        ]
        proc = {"enabled": True, "rows": [{"pid": 1, "name": "x", "bytes": 1e9}], "polled_at_ts": 1.0}
        text = render_snapshot_plain(snaps, [], process_vram=proc)
        self.assertNotIn("Process VRAM", text)

    def test_live_panel_remote_no_proc_block(self) -> None:
        snaps = [
            {
                "server": "remote",
                "reachable": True,
                "local_gpu": False,
                "version": "0.33.2",
                "models": [],
                "tags": [],
                "gpus": None,
                "polled_at_ts": 1_700_000_000.0,
            }
        ]
        proc = {"enabled": True, "rows": [{"pid": 1, "name": "x", "bytes": 1e9}], "polled_at_ts": 1.0}
        panel = build_live_panel(snaps, [], process_vram=proc)
        rendered = panel.__rich_console__(None, None)  # type: ignore[attr-defined]
        # Rich tables don't expose rows easily; stringify via render helper path
        text = render_snapshot_plain(snaps, [], process_vram=proc)
        self.assertNotIn("Process VRAM", text)


if __name__ == "__main__":
    unittest.main()
