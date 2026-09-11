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


class ShowCacheBoundTests(unittest.TestCase):
    def test_cap_evicts_oldest_insertion(self):
        cache = ShowCache(ttl=900, max_entries=3)

        def fake_fetch(url, model, *, timeout=5):
            return {"model": model}

        with patch("ollama_sentinel.show.fetch_show", side_effect=fake_fetch):
            for i in range(5):
                cache.get("http://x", f"m{i}")
        self.assertEqual(len(cache._entries), 3)
        self.assertIn("http://x|m4", cache._entries)
        self.assertNotIn("http://x|m0", cache._entries)

    def test_expired_entries_dropped_on_write(self):
        cache = ShowCache(ttl=0.05)

        def fake_fetch(url, model, *, timeout=5):
            return {"model": model}

        with patch("ollama_sentinel.show.fetch_show", side_effect=fake_fetch):
            cache.get("http://x", "old")
            time.sleep(0.1)
            cache.get("http://x", "new")
        self.assertEqual(list(cache._entries), ["http://x|new"])

    def test_get_is_thread_safe_under_fetch_all(self):
        cache = ShowCache(ttl=900, max_entries=8)

        def fake_fetch(url, model, *, timeout=5):
            time.sleep(0.02)
            return {"model": model}

        with patch("ollama_sentinel.show.fetch_show", side_effect=fake_fetch):
            out = cache.fetch_all("http://x", [f"m{i}" for i in range(8)])
        self.assertEqual(len(out), 8)
        self.assertEqual(len(cache._entries), 8)


if __name__ == "__main__":
    unittest.main()
