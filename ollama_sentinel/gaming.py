"""Pure gaming-yield decision logic (no I/O)."""

from __future__ import annotations

from dataclasses import dataclass


HARD_EXCLUDE_NAMES = frozenset(
    {
        "dwm.exe",
        "dwm",
        "explorer.exe",
        "explorer",
        "ollama.exe",
        "ollama",
        "llama-server.exe",
        "llama-server",
        "ollama_llama_server.exe",
    }
)


@dataclass
class GamingSignals:
    exclusive_fullscreen: bool = False  # A
    borderless_fullscreen: bool = False  # B
    in_game_list: bool = False  # C
    non_ollama_3d_util: bool = False  # D (foreground PID high 3D, not ollama)
    solitaire_gate: bool = False  # E
    pid: int | None = None
    name: str | None = None
    exe_path: str | None = None
    vram_bytes: int = 0
    engine_3d_pct: float = 0.0


def is_gaming_candidate(signals: GamingSignals) -> bool:
    """candidate = A or (B and (C or D))."""
    a = signals.exclusive_fullscreen
    b = signals.borderless_fullscreen
    c = signals.in_game_list
    d = signals.non_ollama_3d_util
    return a or (b and (c or d))


def passes_solitaire_gate(
    vram_bytes: int,
    util_pct: float,
    *,
    min_vram: int = 1536 * 1024 * 1024,
    min_util: float = 50.0,
) -> bool:
    """Signal E: process must actually demand the GPU."""
    return vram_bytes > min_vram or util_pct > min_util


def is_hard_excluded(name: str | None, *, self_pid: int | None = None, pid: int | None = None) -> bool:
    if self_pid is not None and pid is not None and pid == self_pid:
        return True
    if not name:
        return False
    base = name.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    return base.lower() in {n.lower() for n in HARD_EXCLUDE_NAMES}


def is_excluded_by_list(name: str | None, exclude: set[str] | frozenset[str]) -> bool:
    if not name or not exclude:
        return False
    base = name.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    lower = {e.lower() for e in exclude}
    return base.lower() in lower or base.lower().removesuffix(".exe") in lower


def is_gaming(
    signals: GamingSignals,
    *,
    exclude: set[str] | frozenset[str] = frozenset(),
    self_pid: int | None = None,
) -> bool:
    """gaming = candidate and E and not excluded."""
    if is_hard_excluded(signals.name, self_pid=self_pid, pid=signals.pid):
        return False
    if is_excluded_by_list(signals.name, exclude):
        return False
    if not is_gaming_candidate(signals):
        return False
    return bool(signals.solitaire_gate)


def should_unload(*, gaming: bool, busy: bool, yield_enabled: bool) -> bool:
    return bool(gaming and yield_enabled and not busy)


def is_ollama_busy(util_pct: float | None, *, busy_util: float = 20.0) -> bool:
    if util_pct is None:
        return False
    return util_pct >= busy_util


@dataclass
class GamingHysteresis:
    """Fire after fire_n consecutive true polls; clear after clear_m consecutive false."""

    fire_n: int = 2
    clear_m: int = 4
    true_streak: int = 0
    false_streak: int = 0
    active: bool = False

    def update(self, detected: bool) -> str | None:
        """Return 'detected', 'cleared', or None (no transition)."""
        if detected:
            self.true_streak += 1
            self.false_streak = 0
            if not self.active and self.true_streak >= self.fire_n:
                self.active = True
                return "detected"
        else:
            self.false_streak += 1
            self.true_streak = 0
            if self.active and self.false_streak >= self.clear_m:
                self.active = False
                return "cleared"
        return None


def is_fullscreen_bounds(
    win_left: int,
    win_top: int,
    win_right: int,
    win_bottom: int,
    mon_left: int,
    mon_top: int,
    mon_right: int,
    mon_bottom: int,
    *,
    tolerance: int = 4,
) -> bool:
    """True when window covers the monitor (borderless fullscreen), not a maximized window."""
    return (
        abs(win_left - mon_left) <= tolerance
        and abs(win_top - mon_top) <= tolerance
        and abs(win_right - mon_right) <= tolerance
        and abs(win_bottom - mon_bottom) <= tolerance
    )


def parse_exclude_list(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}
