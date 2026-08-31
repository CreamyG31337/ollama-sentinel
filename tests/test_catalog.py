"""Catalog parse tests (no network)."""

import unittest

from ollama_sentinel.catalog import typeahead


class TestCatalog(unittest.TestCase):
    def test_typeahead_min_length(self):
        self.assertEqual(typeahead("q"), [])


if __name__ == "__main__":
    unittest.main()
