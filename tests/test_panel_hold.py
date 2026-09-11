"""Tests for UnreachableHold grace window logic."""

from __future__ import annotations

import unittest

from ollama_sentinel.panel_hold import UNREACHABLE_GRACE_S, UnreachableHold


class UnreachableHoldTests(unittest.TestCase):
    def test_reachable_is_fresh_and_first_failure_is_hold(self):
        hold = UnreachableHold()
        # Initial good poll records last_good_ts and returns 'fresh'
        self.assertEqual(hold.record("local", True, 1000.0), "fresh")
        self.assertEqual(hold.last_good_ts("local"), 1000.0)

        # First failure right after a good poll returns 'hold'
        self.assertEqual(hold.record("local", False, 1005.0), "hold")
        # last_good_ts is unchanged
        self.assertEqual(hold.last_good_ts("local"), 1000.0)

    def test_failure_with_no_good_poll_blanks(self):
        hold = UnreachableHold()
        self.assertEqual(hold.record("remote", False, 1000.0), "blank")
        self.assertIsNone(hold.last_good_ts("remote"))

    def test_failures_past_grace_window_blank_and_stay_blank(self):
        hold = UnreachableHold()
        hold.record("local", True, 1000.0)

        # Still within grace at exactly the boundary
        self.assertEqual(hold.record("local", False, 1000.0 + UNREACHABLE_GRACE_S), "hold")

        # Past grace returns 'blank'
        self.assertEqual(hold.record("local", False, 1000.0 + UNREACHABLE_GRACE_S + 0.1), "blank")
        # Subsequent failures stay 'blank'
        self.assertEqual(hold.record("local", False, 1000.0 + UNREACHABLE_GRACE_S + 30.0), "blank")
        self.assertEqual(hold.record("local", False, 1000.0 + UNREACHABLE_GRACE_S + 60.0), "blank")
        # last_good_ts is still the good poll ts
        self.assertEqual(hold.last_good_ts("local"), 1000.0)

    def test_recovery_is_fresh_and_later_failure_gets_new_grace(self):
        hold = UnreachableHold()
        hold.record("local", True, 1000.0)
        self.assertEqual(hold.record("local", False, 1100.0), "blank")  # grace ran out

        # Recovery poll
        self.assertEqual(hold.record("local", True, 1200.0), "fresh")
        self.assertEqual(hold.last_good_ts("local"), 1200.0)

        # Subsequent failure is within the new grace window
        self.assertEqual(hold.record("local", False, 1210.0), "hold")
        self.assertEqual(hold.last_good_ts("local"), 1200.0)

    def test_grace_is_per_server(self):
        hold = UnreachableHold()
        hold.record("srv-a", True, 1000.0)

        # srv-b has had no good poll
        self.assertEqual(hold.record("srv-b", False, 1010.0), "blank")
        # srv-a is within grace
        self.assertEqual(hold.record("srv-a", False, 1010.0), "hold")

    def test_reset_forgets_everything(self):
        hold = UnreachableHold()
        hold.record("srv-a", True, 1000.0)
        hold.record("srv-b", True, 1005.0)

        hold.reset()
        self.assertIsNone(hold.last_good_ts("srv-a"))
        self.assertIsNone(hold.last_good_ts("srv-b"))

        # Both now blank on failure because good polls were forgotten
        self.assertEqual(hold.record("srv-a", False, 1010.0), "blank")
        self.assertEqual(hold.record("srv-b", False, 1010.0), "blank")


if __name__ == "__main__":
    unittest.main()
