"""Unload API tests."""

import json
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

from ollama_sentinel.unload import unload_model, unload_models


class TestUnload(unittest.TestCase):
    @patch("ollama_sentinel.unload.urllib.request.urlopen")
    def test_unload_success(self, mock_urlopen) -> None:
        payload = {
            "model": "llama3.2",
            "done": True,
            "done_reason": "unload",
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(payload).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = unload_model("http://127.0.0.1:11434", "llama3.2")
        self.assertEqual(result["done_reason"], "unload")
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode())
        self.assertEqual(body["keep_alive"], 0)
        self.assertEqual(body["model"], "llama3.2")

    @patch("ollama_sentinel.unload.unload_model")
    def test_unload_models_collects_errors(self, mock_unload) -> None:
        mock_unload.side_effect = [
            {"done": True, "model": "a"},
            {"error": "failed", "model": "b"},
        ]
        results = unload_models("http://127.0.0.1:11434", ["a", "b"])
        self.assertEqual(len(results), 2)
        self.assertIn("error", results[1])


if __name__ == "__main__":
    unittest.main()
