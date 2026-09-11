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

    def invalidate(self) -> int:
        """Drop every in-flight refresh — used the instant the user switches hosts.

        Any poll that already started has a lower ticket and will be refused by
        ``accept``, so it cannot repaint the previous device's numbers over the
        blank loading state.
        """
        with self._lock:
            self._issued += 1
            self._applied = self._issued
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

    def still_current(self, seq: int, target: str | None, current: str | None) -> bool:
        """True if this ticket is still the latest applied view for ``target``.

        Used after a slow follow-up (advisor /api/show) so enrichment cannot
        paint over a newer switch.
        """
        with self._lock:
            if current is not None and target is not None and target != current:
                return False
            return seq == self._applied

    @property
    def applied(self) -> int:
        with self._lock:
            return self._applied


class SingleFlight:
    """At most one run at a time; requests during a run cause exactly one rerun.

    A manual Refresh click, a host switch and the 5s poll timer all want the
    same refresh work. Starting a thread per request let them stack (each
    doing HTTP + nvidia-smi); dropping the request would lose a host switch
    that arrived mid-refresh. Coalescing instead: while a run is in flight,
    ``request`` only records that one more pass is wanted, and the running
    worker sees ``finish() -> True`` and loops exactly once more.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._pending = False

    def request(self) -> bool:
        """True -> caller should start (or continue) the work; False -> coalesced."""
        with self._lock:
            if self._running:
                self._pending = True
                return False
            self._running = True
            return True

    def finish(self) -> bool:
        """True -> a request arrived during the run, so run once more."""
        with self._lock:
            if self._pending:
                self._pending = False
                return True
            self._running = False
            return False
