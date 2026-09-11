"""Hold last good panels on screen across brief connection hiccups."""

from __future__ import annotations

UNREACHABLE_GRACE_S = 60.0


class UnreachableHold:
    """Decides whether a failed poll may keep the last good panels on screen.

    Grace is measured from the last good poll, not from the first failure.
    Within grace of a successful poll, panels remain on screen while the
    freshness strip reports the true age of the data.
    """

    def __init__(self, grace_s: float = UNREACHABLE_GRACE_S) -> None:
        self._grace_s = float(grace_s)
        self._last_good: dict[str, float] = {}

    def record(self, server: str, reachable: bool, now: float) -> str:
        """'fresh' (reachable), 'hold' (unreachable within grace of the last good
        poll for this server), or 'blank' (no good poll for this server yet, or
        the grace has run out).
        """
        if reachable:
            self._last_good[server] = now
            return "fresh"
        good_ts = self._last_good.get(server)
        if good_ts is None:
            return "blank"
        if (now - good_ts) <= self._grace_s:
            return "hold"
        return "blank"

    def last_good_ts(self, server: str) -> float | None:
        return self._last_good.get(server)

    def reset(self) -> None:
        self._last_good.clear()
