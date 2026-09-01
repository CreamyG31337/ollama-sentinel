"""Ordering guard for concurrent GUI refreshes.

The GUI polls one server on a timer *and* on demand when the user picks a
different one from the dropdown. Both run on background threads, and a poll
against an unreachable host can take ~30s (three HTTP calls at a 10s timeout)
plus an /api/show per installed model.

Without a guard, two things went wrong:

* a slow poll for server A would finish after the user had switched to B and
  overwrite B's fresh data, leaving the panels showing A's models under B's
  name -- the UI appeared "stuck on the wrong server";
* two polls of the same server could land out of order, so an older result
  could replace a newer one.

`RefreshGuard` issues a monotonic ticket per refresh and accepts a result only
if it is still wanted.
"""

from __future__ import annotations

import threading


class RefreshGuard:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._issued = 0
        self._applied = 0

    def issue(self) -> int:
        """Take a ticket for a refresh that is about to start."""
        with self._lock:
            self._issued += 1
            return self._issued

    def accept(self, seq: int, target: str | None, current: str | None) -> bool:
        """True if this result should be applied to the UI.

        `target` is the server the refresh was started for; `current` is the
        server selected right now. A result for a server the user has navigated
        away from is dropped, as is one older than a result already applied.
        """
        with self._lock:
            if current is not None and target is not None and target != current:
                return False
            if seq < self._applied:
                return False
            self._applied = seq
            return True

    @property
    def applied(self) -> int:
        with self._lock:
            return self._applied
