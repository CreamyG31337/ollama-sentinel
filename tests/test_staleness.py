"""Staleness must measure data age, not the app's own polling latency.

Regression tests for the false STALE seen on a fresh `--once` across four
servers (2026-08-31): two unreachable hosts each paid a full connect timeout,
and because every snapshot shared one timestamp taken before the poll loop,
the healthy hosts were reported as 55-68s old and marked STALE.
"""

from __future__ import annotations

import time
import unittest
from unittest import mock

from ollama_sentinel.poll import poll_all
from ollama_sentinel.render import render_snapshot_plain


def _ok(_url, path, timeout=None):
    if path == "/api/version":
        return {"version": "0.33.2"}, None
    return {"models": []}, None


class Clock:
    """Deterministic stand-in for time.time()."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value


class PolledAtPerServer(unittest.TestCase):
    def test_each_snapshot_stamped_at_its_own_completion(self):
        """A slow host must not share (and so backdate) the others' timestamp.

        Under the old shared pre-loop stamp both servers reported T0, hiding that
        the second finished 30s later. Now the fast server keeps its own early
        stamp and the slow one carries its own.
        """
        clock = Clock()
        t0 = clock.value
        slowed = {"done": False}

        def slow_second(url, path, timeout=None):
            # One connect-timeout worth of delay for the slow host (HTTP is
            # parallel now, so don't count 3× path calls).
            if "second" in url and not slowed["done"]:
                slowed["done"] = True
                clock.value += 30.0
            return _ok(url, path, timeout)

        with mock.patch("ollama_sentinel.poll._tcp_connect_error", return_value=None):
            with mock.patch("ollama_sentinel.poll._get_json", side_effect=slow_second):
                with mock.patch("ollama_sentinel.poll.time.time", clock):
                    snaps = poll_all(
                        [
                            {"name": "first", "url": "http://first:11434", "local_gpu": False},
                            {"name": "second", "url": "http://second:11434", "local_gpu": False},
                        ]
                    )

        first, second = snaps
        self.assertEqual(first["polled_at_ts"], t0, "fast host keeps its own early stamp")
        self.assertEqual(second["polled_at_ts"], t0 + 30.0, "slow host carries its own stamp")
        self.assertGreater(second["polled_at_ts"], first["polled_at_ts"])

    def test_explicit_polled_at_still_shared_for_determinism(self):
        """Callers and tests may still pin one timestamp when they want to."""
        with mock.patch("ollama_sentinel.poll._tcp_connect_error", return_value=None):
            with mock.patch("ollama_sentinel.poll._get_json", side_effect=_ok):
                snaps = poll_all(
                    [
                        {"name": "a", "url": "http://a:11434", "local_gpu": False},
                        {"name": "b", "url": "http://b:11434", "local_gpu": False},
                    ],
                    polled_at=1_700_000_000.0,
                )
        self.assertEqual(snaps[0]["polled_at_ts"], 1_700_000_000.0)
        self.assertEqual(snaps[1]["polled_at_ts"], 1_700_000_000.0)

    def test_fresh_multi_server_poll_is_not_stale(self):
        """The end-to-end symptom: a just-completed poll must not read STALE."""
        with mock.patch("ollama_sentinel.poll._tcp_connect_error", return_value=None):
            with mock.patch("ollama_sentinel.poll._get_json", side_effect=_ok):
                snaps = poll_all(
                    [
                        {"name": "a", "url": "http://a:11434", "local_gpu": False},
                        {"name": "b", "url": "http://b:11434", "local_gpu": False},
                    ]
                )
        now = time.time()
        for snap in snaps:
            self.assertLess(now - snap["polled_at_ts"], 5.0)


class OnceSuppressesStaleness(unittest.TestCase):
    """A single snapshot has no refresh cadence, so it cannot be 'stale'."""

    def _snap(self, age_seconds: float) -> dict:
        return {
            "server": "solo",
            "reachable": True,
            "version": "0.33.2",
            "models": [],
            "tags": [],
            "gpus": None,
            "local_gpu": False,
            "polled_at_ts": time.time() - age_seconds,
        }

    def test_once_does_not_mark_stale(self):
        out = render_snapshot_plain([self._snap(600)], [], poll_interval=5.0, once=True)
        self.assertNotIn("STALE", out)

    def test_live_mode_still_marks_genuinely_old_data(self):
        out = render_snapshot_plain([self._snap(600)], [], poll_interval=5.0, once=False)
        self.assertIn("STALE", out)


if __name__ == "__main__":
    unittest.main()
