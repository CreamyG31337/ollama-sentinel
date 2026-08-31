"""Catalog parse tests (no network)."""

import unittest

from ollama_sentinel.catalog import (
    parse_list_item,
    parse_model_detail,
    pull_name_for_file,
    strip_yaml_frontmatter,
    summarize_list_item,
    truncate_readme,
    typeahead,
)


class TestCatalog(unittest.TestCase):
    def test_typeahead_min_length(self):
        self.assertEqual(typeahead("q"), [])

    def test_parse_list_item(self):
        item = parse_list_item(
            {
                "id": "org/model-GGUF",
                "downloads": 1200,
                "likes": 42,
                "pipeline_tag": "text-generation",
                "tags": ["gguf", "license:apache-2.0"],
            }
        )
        self.assertEqual(item["pull_name"], "hf.co/org/model-GGUF")
        self.assertEqual(item["license"], "apache-2.0")
        self.assertIn("1.2k dl", item["summary"])

    def test_parse_model_detail(self):
        detail = parse_model_detail(
            {
                "id": "org/model-GGUF",
                "tags": ["gguf", "license:mit"],
                "cardData": {"license": "mit", "base_model": "org/base"},
                "gguf": {
                    "architecture": "llama",
                    "context_length": 8192,
                    "total": 4_000_000_000,
                },
                "siblings": [
                    {"rfilename": "README.md"},
                    {"rfilename": "model-Q4_K_M.gguf"},
                    {"rfilename": "model-Q8_0.gguf"},
                ],
                "downloads": 900,
                "likes": 10,
                "gated": False,
            }
        )
        self.assertEqual(len(detail["variants"]), 2)
        self.assertEqual(
            detail["variants"][0]["pull_name"],
            pull_name_for_file("org/model-GGUF", "model-Q4_K_M.gguf"),
        )
        self.assertEqual(detail["architecture"], "llama")
        self.assertEqual(detail["context_length"], 8192)

    def test_summarize_list_item(self):
        text = summarize_list_item(
            {
                "pipeline_tag": "text-generation",
                "license": "apache-2.0",
                "downloads": 2_500_000,
                "likes": 1500,
            }
        )
        self.assertIn("text generation", text)
        self.assertIn("2.5M dl", text)

    def test_strip_yaml_frontmatter(self):
        raw = "---\nlicense: apache-2.0\n---\n\n# Title\n\nBody"
        self.assertEqual(strip_yaml_frontmatter(raw), "# Title\n\nBody")

    def test_truncate_readme(self):
        long = "x" * 100
        out = truncate_readme(long, max_chars=50)
        self.assertIn("truncated", out)
        self.assertLess(len(out), 100)


if __name__ == "__main__":
    unittest.main()
