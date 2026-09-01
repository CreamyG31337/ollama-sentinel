"""Network error formatting tests."""

from __future__ import annotations

import unittest
import urllib.error

from ollama_sentinel.net_errors import HubRequestError, format_network_error


class TestNetErrors(unittest.TestCase):
    def test_timeout(self) -> None:
        self.assertIn("Timed out", format_network_error(TimeoutError()))

    def test_http_429(self) -> None:
        err = urllib.error.HTTPError("https://hf.co", 429, "Too Many", {}, None)
        self.assertIn("Rate limited", format_network_error(err))

    def test_connection_refused_windows(self) -> None:
        err = OSError(10061, "No connection could be made")
        self.assertIn("Connection refused", format_network_error(err))

    def test_hub_request_error_passthrough(self) -> None:
        inner = HubRequestError("Hugging Face: Timed out")
        self.assertEqual(
            format_network_error(inner, context="Discover"),
            "Discover: Hugging Face: Timed out",
        )


if __name__ == "__main__":
    unittest.main()
