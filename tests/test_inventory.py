"""Inventory tests."""

import unittest

from ollama_sentinel.inventory import build_inventory, inventory_detail_line, inventory_summary


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

    def test_metadata_from_tags(self):
        snap = {
            "tags": [
                {
                    "name": "qwen3:27b",
                    "size": 16e9,
                    "details": {
                        "quantization_level": "Q4_K_M",
                        "family": "qwen35",
                        "parameter_size": "27.3B",
                    },
                }
            ],
            "models": [],
            "gpus": [],
        }
        rows = build_inventory(snap)
        self.assertEqual(rows[0]["quantization"], "Q4_K_M")
        self.assertEqual(rows[0]["family"], "qwen35")
        self.assertEqual(rows[0]["parameter_size"], "27.3B")
        self.assertEqual(
            inventory_detail_line(rows[0]),
            "Q4_K_M · qwen35 · 27.3B",
        )

    def test_loaded_context_in_detail_line(self):
        snap = {
            "tags": [
                {
                    "name": "m",
                    "size": 8e9,
                    "details": {"quantization_level": "Q8_0", "family": "llama"},
                }
            ],
            "models": [
                {
                    "name": "m",
                    "size": 8e9,
                    "size_vram": 8e9,
                    "context_length": 65536,
                }
            ],
            "gpus": [],
        }
        rows = build_inventory(snap)
        self.assertEqual(rows[0]["context_length"], 65536)
        self.assertIn("ctx 65,536", inventory_detail_line(rows[0]))
        self.assertIn("Q8_0", inventory_detail_line(rows[0]))

    def test_detail_line_missing_metadata(self):
        self.assertEqual(inventory_detail_line({"name": "bare"}), "—")

    def test_fit_unknown_without_gpu(self):
        snap = {
            "tags": [{"name": "m", "size": 8e9}],
            "models": [],
            "gpus": None,
            "gpu_data_available": False,
        }
        rows = build_inventory(snap)
        self.assertIsNone(rows[0]["would_spill"])
        s = inventory_summary(rows)
        self.assertIn("fit unknown", s)


if __name__ == "__main__":
    unittest.main()
