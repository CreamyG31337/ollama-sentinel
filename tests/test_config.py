"""Config tests."""

import unittest
from pathlib import Path

from ollama_sentinel.config import DEFAULT_URL, Thresholds, config_from_env, load_servers, parse_dotenv


class TestConfig(unittest.TestCase):
    def test_default_url_not_from_ollama_host(self):
        import os

        os.environ["OLLAMA_HOST"] = "0.0.0.0:11434"
        cfg = config_from_env({})
        self.assertEqual(cfg.ollama_url, DEFAULT_URL)

    def test_paging_power_w_overrides(self):
        th = Thresholds(paging_power_w=200, paging_power_frac=0.6)
        from ollama_sentinel.alarms import _paging_power_threshold

        self.assertEqual(_paging_power_threshold({"power_limit": 350}, th), 200)

    def test_servers_fallback(self):
        servers = load_servers(Path("/nonexistent"), DEFAULT_URL)
        self.assertEqual(len(servers), 1)
        self.assertTrue(servers[0].local_gpu)

    def test_gitignore_has_env(self):
        gi = Path(__file__).resolve().parents[1] / ".gitignore"
        text = gi.read_text(encoding="utf-8")
        self.assertIn(".env", text)
        self.assertIn("servers.json", text)


if __name__ == "__main__":
    unittest.main()
