"""Host switch must never leave another device's stats on screen."""

from __future__ import annotations

import unittest

from ollama_sentinel.ui import (
    clear_switch_state,
    host_context_line,
    host_dropdown_label,
    host_dropdown_option,
    host_status_color,
    loading_caption,
    show_local_process_panels,
)
from ollama_sentinel.ui_widgets import PALETTE


class HostSwitchBlankTests(unittest.TestCase):
    def test_loading_caption_names_the_selected_server(self):
        self.assertEqual(loading_caption("ubuntu-rx6800"), "Loading ubuntu-rx6800...")

    def test_loading_caption_fallback_when_unset(self):
        self.assertEqual(loading_caption(None), "Loading server...")
        self.assertEqual(loading_caption(""), "Loading server...")

    def test_host_context_line_shows_name_and_url(self):
        self.assertEqual(
            host_context_line("ubuntu-rx6800", "http://100.64.188.1:11434"),
            "ubuntu-rx6800  ·  http://100.64.188.1:11434",
        )

    def test_host_context_line_loading(self):
        self.assertTrue(
            host_context_line("local", "http://127.0.0.1:11434", loading=True).startswith(
                "Loading local"
            )
        )

    def test_host_dropdown_label_online_offline(self):
        self.assertEqual(
            host_dropdown_label("ubuntu-rx6800", True),
            "● online  ·  ubuntu-rx6800",
        )
        self.assertEqual(
            host_dropdown_label("ts-desktop-3070", False),
            "○ offline  ·  ts-desktop-3070",
        )
        self.assertIn("…", host_dropdown_label("local", None))

    def test_host_status_color_online_offline(self):
        self.assertEqual(host_status_color(True), PALETTE["ok"])
        self.assertEqual(host_status_color(False), PALETTE["alarm"])
        self.assertEqual(host_status_color(None), PALETTE["muted"])

    def test_host_dropdown_option_colors_status(self):
        online = host_dropdown_option("ubuntu-rx6800", True)
        offline = host_dropdown_option("ts-desktop-3070", False)
        self.assertEqual(online.key, "ubuntu-rx6800")
        self.assertEqual(offline.key, "ts-desktop-3070")
        self.assertEqual(online.content.controls[1].value, "online")
        self.assertEqual(online.content.controls[1].color, PALETTE["ok"])
        self.assertEqual(offline.content.controls[1].value, "offline")
        self.assertEqual(offline.content.controls[1].color, PALETTE["alarm"])

    def test_clear_switch_state_drops_snap_and_poll_age(self):
        last_snap = {"server": "cr-desktop-3090", "models": [{"name": "qwen"}]}
        poll_state = {"polled_ts": 1_700_000_000.0, "stale": True, "reachable": True}
        clear_switch_state(last_snap, poll_state)
        self.assertEqual(last_snap, {})
        self.assertIsNone(poll_state["polled_ts"])
        self.assertFalse(poll_state["stale"])
        # reachable is left alone; footer_tick skips while polled_ts is None
        self.assertTrue(poll_state["reachable"])


class LocalProcessPanelGateTests(unittest.TestCase):
    def test_local_gpu_shows_process_panels(self):
        self.assertTrue(show_local_process_panels(True))

    def test_remote_hides_process_panels(self):
        self.assertFalse(show_local_process_panels(False))


if __name__ == "__main__":
    unittest.main()
