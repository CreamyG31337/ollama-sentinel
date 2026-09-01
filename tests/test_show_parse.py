"""Tests for /api/show parsers."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from ollama_sentinel.show_parse import (
    parse_draft_num_predict,
    parse_parameters_block,
    parse_show_bundle,
    quant_from_show,
    tensor_weight_bytes,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestShowParse(unittest.TestCase):
    def test_parameters_block(self) -> None:
        params = parse_parameters_block("draft_num_predict              4\ntop_k 20")
        self.assertEqual(params["draft_num_predict"], "4")
        self.assertEqual(params["top_k"], "20")

    def test_draft_num_predict(self) -> None:
        self.assertEqual(parse_draft_num_predict("draft_num_predict 0"), 0)
        self.assertIsNone(parse_draft_num_predict(None))

    def test_qwen_mtp_fixture(self) -> None:
        raw = json.loads((FIXTURES / "api_show_qwen38_mtp.json").read_text(encoding="utf-8"))
        bundle = parse_show_bundle(raw)
        self.assertIsNone(bundle.get("error"))
        self.assertEqual(bundle["draft_num_predict"], 4)
        self.assertEqual(bundle["mtp_layers"], 1)
        self.assertEqual(bundle["quantization"], "Q4_K_M")
        self.assertIsNotNone(bundle["weight_bytes"])

    def test_granite_quant_from_show(self) -> None:
        raw = json.loads((FIXTURES / "api_show_granite41.json").read_text(encoding="utf-8"))
        self.assertEqual(quant_from_show(raw), "Q8_0")
        bundle = parse_show_bundle(raw)
        self.assertIsNone(bundle.get("mtp_layers"))
        self.assertIsNone(bundle.get("draft_num_predict"))

    def test_tensor_weight_bytes_empty(self) -> None:
        self.assertIsNone(tensor_weight_bytes([]))
        self.assertIsNone(tensor_weight_bytes(None))


if __name__ == "__main__":
    unittest.main()
