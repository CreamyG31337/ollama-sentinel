"""ShowCache fetch behaviour."""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from ollama_sentinel.show import ShowCache, fetch_show


class ShowCacheFetchAllTests(unittest.TestCase):
    def test_fetch_all_parallel_returns_every_model(self):
        cache = ShowCache(ttl=900)

        def fake_get(url, model, *, force=False):
            time.sleep(0.05)
            return {"model": model}

        with patch.object(cache, "get", side_effect=fake_get):
            t0 = time.perf_counter()
            out = cache.fetch_all("http://x", [f"m{i}" for i in range(8)])
            elapsed = time.perf_counter() - t0
        self.assertEqual(len(out), 8)
        # Serial would be ~0.4s; parallel with 8 workers should be near one sleep.
        self.assertLess(elapsed, 0.25)

    def test_fetch_show_timeout_default_is_short(self):
        from ollama_sentinel import show as show_mod

        self.assertLessEqual(show_mod.DEFAULT_TIMEOUT, 10)


if __name__ == "__main__":
    unittest.main()
