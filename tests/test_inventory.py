"""Inventory tests."""

import unittest

from ollama_sentinel.inventory import build_inventory, inventory_summary


class TestInventory(unittest.TestCase):
    def test_would_spill(self):
        snap = {
            "tags": [{"name": "big", "size": 20e9}],
            "models": [],
            "gpus": [{"memory_used": 22e9, "memory_total": 24e9}],
        }
        rows = build_inventory(snap)
        self.assertTrue(rows[0]["would_spill"])

    def test_fits(self):
        snap = {
            "tags": [{"name": "small", "size": 4e9}],
            "models": [],
            "gpus": [{"memory_used": 2e9, "memory_total": 24e9}],
        }
        rows = build_inventory(snap)
        self.assertFalse(rows[0]["would_spill"])

    def test_loaded_not_would_spill(self):
        snap = {
            "tags": [{"name": "m", "size": 20e9}],
            "models": [{"name": "m", "size": 20e9, "size_vram": 20e9}],
            "gpus": [{"memory_used": 22e9, "memory_total": 24e9}],
        }
        rows = build_inventory(snap)
        self.assertFalse(rows[0]["would_spill"])
        self.assertTrue(rows[0]["loaded"])

    def test_summary(self):
        rows = [{"loaded": True, "would_spill": False}, {"loaded": False, "would_spill": True}]
        s = inventory_summary(rows)
        self.assertIn("2 installed", s)

    def test_summary_free_vram(self):
        rows = [{"loaded": False, "would_spill": False}]
        s = inventory_summary(rows, free_vram_gb=4.2, free_vram_pct=17.0)
        self.assertIn("free VRAM: 4.2 GB (17%)", s)


if __name__ == "__main__":
    unittest.main()
