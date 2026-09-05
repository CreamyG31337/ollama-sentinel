"""The server switcher must never show one server's data under another's name."""

from __future__ import annotations

import threading
import unittest

from ollama_sentinel.refresh_guard import RefreshGuard


class RefreshGuardTests(unittest.TestCase):
    def test_result_for_the_selected_server_is_applied(self):
        g = RefreshGuard()
        seq = g.issue()
        self.assertTrue(g.accept(seq, target="alpha", current="alpha"))

    def test_result_for_a_server_the_user_left_is_dropped(self):
        """The reported bug: a slow poll lands after the user switched away."""
        g = RefreshGuard()
        slow = g.issue()          # started against alpha
        fast = g.issue()          # user switches to beta; that poll finishes first
        self.assertTrue(g.accept(fast, target="beta", current="beta"))
        # alpha's 30s timeout finally returns -- it must not overwrite beta.
        self.assertFalse(g.accept(slow, target="alpha", current="beta"))

    def test_out_of_order_results_for_one_server_are_dropped(self):
        g = RefreshGuard()
        first = g.issue()
        second = g.issue()
        self.assertTrue(g.accept(second, target="alpha", current="alpha"))
        self.assertFalse(g.accept(first, target="alpha", current="alpha"))

    def test_latest_result_still_wins_after_a_drop(self):
        g = RefreshGuard()
        old = g.issue()
        new = g.issue()
        g.accept(new, target="alpha", current="alpha")
        g.accept(old, target="alpha", current="alpha")
        newest = g.issue()
        self.assertTrue(g.accept(newest, target="alpha", current="alpha"))

    def test_unknown_current_selection_does_not_block(self):
        """Before the dropdown has a value, ordering alone decides."""
        g = RefreshGuard()
        seq = g.issue()
        self.assertTrue(g.accept(seq, target="alpha", current=None))

    def test_concurrent_issue_gives_unique_tickets(self):
        g = RefreshGuard()
        seen: list[int] = []
        lock = threading.Lock()

        def worker():
            s = g.issue()
            with lock:
                seen.append(s)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(seen), 50)
        self.assertEqual(len(set(seen)), 50, "tickets must be unique under concurrency")

    def test_invalidate_drops_in_flight_poll(self):
        """Blank-on-switch must kill a poll that started for the previous host."""
        g = RefreshGuard()
        slow = g.issue()
        g.invalidate()
        fresh = g.issue()
        self.assertTrue(g.accept(fresh, target="beta", current="beta"))
        self.assertFalse(g.accept(slow, target="alpha", current="beta"))
        self.assertFalse(g.accept(slow, target="beta", current="beta"))

    def test_still_current_after_accept(self):
        g = RefreshGuard()
        seq = g.issue()
        g.accept(seq, target="alpha", current="alpha")
        self.assertTrue(g.still_current(seq, "alpha", "alpha"))
        self.assertFalse(g.still_current(seq, "alpha", "beta"))

    def test_still_current_false_after_invalidate(self):
        g = RefreshGuard()
        seq = g.issue()
        g.accept(seq, target="alpha", current="alpha")
        g.invalidate()
        self.assertFalse(g.still_current(seq, "alpha", "alpha"))


if __name__ == "__main__":
    unittest.main()
