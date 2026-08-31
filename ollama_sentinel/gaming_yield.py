"""Background watcher: detect gaming and optionally unload Ollama models."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ollama_sentinel.gaming import (
    GamingHysteresis,
    GamingSignals,
    is_gaming,
    is_ollama_busy,
    parse_exclude_list,
    passes_solitaire_gate,
    should_unload,
)
from ollama_sentinel.log import append_event_log
from ollama_sentinel.paths import app_data_dir
from ollama_sentinel.unload import unload_models

OLLAMA_NAME_MARKERS = ("llama-server", "ollama")


def _is_ollama_process(name: str | None) -> bool:
    if not name:
        return False
    lower = name.lower()
    return any(m in lower for m in OLLAMA_NAME_MARKERS)


def _lookup_pid(rows: list[dict[str, Any]], pid: int | None) -> dict[str, Any] | None:
    if pid is None:
        return None
    for row in rows:
        if row.get("pid") == pid:
            return row
    return None


def _ollama_util(rows: list[dict[str, Any]]) -> float | None:
    best: float | None = None
    for row in rows:
        if not _is_ollama_process(row.get("name")):
            continue
        util = row.get("engine_3d_pct")
        if util is None:
            continue
        best = util if best is None else max(best, float(util))
    return best


def build_signals(
    win: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    min_vram: int,
    min_util: float,
) -> GamingSignals:
    pid = win.get("pid")
    name = win.get("name")
    row = _lookup_pid(rows, pid)
    vram = int(row.get("bytes") or 0) if row else 0
    util = float(row.get("engine_3d_pct") or 0.0) if row else 0.0
    gate = passes_solitaire_gate(vram, util, min_vram=min_vram, min_util=min_util)
    # Signal D: foreground PID has meaningful 3D util and is not Ollama.
    non_ollama_3d = (not _is_ollama_process(name)) and util > min_util
    return GamingSignals(
        exclusive_fullscreen=bool(win.get("exclusive_fullscreen")),
        borderless_fullscreen=bool(win.get("borderless_fullscreen")),
        in_game_list=bool(win.get("in_game_list")),
        non_ollama_3d_util=non_ollama_3d,
        solitaire_gate=gate,
        pid=pid,
        name=name,
        exe_path=win.get("exe_path"),
        vram_bytes=vram,
        engine_3d_pct=util,
    )


class GamingYieldWatcher:
    """Poll gaming signals; log transitions; optionally unload when enabled."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        yield_enabled: bool = False,
        interval: float = 12.0,
        exclude: set[str] | None = None,
        min_vram_bytes: int = 1536 * 1024 * 1024,
        min_util: float = 50.0,
        busy_util: float = 20.0,
        ollama_url: str = "http://127.0.0.1:11434",
        log_path: Path | None = None,
        get_proc_snapshot: Callable[[], dict[str, Any]] | None = None,
        list_loaded_models: Callable[[], list[str]] | None = None,
        collect_win: Callable[[], dict[str, Any]] | None = None,
        unload_fn: Callable[[str, list[str]], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.enabled = enabled and sys.platform == "win32"
        self.yield_enabled = yield_enabled
        self.interval = interval
        self.exclude = exclude or set()
        self.min_vram_bytes = min_vram_bytes
        self.min_util = min_util
        self.busy_util = busy_util
        self.ollama_url = ollama_url
        self.log_path = log_path or (app_data_dir() / "gaming.jsonl")
        self.get_proc_snapshot = get_proc_snapshot
        self.list_loaded_models = list_loaded_models
        self.collect_win = collect_win
        self.unload_fn = unload_fn or unload_models
        self._hyst = GamingHysteresis(fire_n=2, clear_m=4)
        self._busy_idle_streak = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = "idle"
        self._last_signals: dict[str, Any] = {}
        self._self_pid = os.getpid()

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "active": self._hyst.active,
                "yield_enabled": self.yield_enabled,
                "signals": dict(self._last_signals),
            }

    def _collect(self) -> GamingSignals:
        if self.collect_win is not None:
            win = self.collect_win()
        else:
            from ollama_sentinel.gaming_win import collect_windows_signals

            win = collect_windows_signals()
        rows: list[dict[str, Any]] = []
        if self.get_proc_snapshot:
            snap = self.get_proc_snapshot() or {}
            rows = list(snap.get("rows") or [])
        return build_signals(
            win,
            rows,
            min_vram=self.min_vram_bytes,
            min_util=self.min_util,
        )

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        try:
            append_event_log(self.log_path, event, payload=payload)
        except OSError:
            pass

    def _maybe_unload(self, signals: GamingSignals) -> None:
        rows: list[dict[str, Any]] = []
        if self.get_proc_snapshot:
            rows = list((self.get_proc_snapshot() or {}).get("rows") or [])
        util = _ollama_util(rows)
        busy = is_ollama_busy(util, busy_util=self.busy_util)
        if busy:
            self._busy_idle_streak = 0
            with self._lock:
                self._status = "detected"
            return
        self._busy_idle_streak += 1
        if self._busy_idle_streak < 2:
            return
        if not should_unload(
            gaming=True,
            busy=False,
            yield_enabled=self.yield_enabled,
        ):
            return
        models: list[str] = []
        if self.list_loaded_models:
            try:
                models = list(self.list_loaded_models() or [])
            except Exception as exc:
                self._log("gaming_yield_error", {"error": str(exc)})
                return
        if not models:
            with self._lock:
                self._status = "detected"
            return
        results = self.unload_fn(self.ollama_url, models)
        self._log(
            "gaming_yield",
            {
                "models": models,
                "pid": signals.pid,
                "name": signals.name,
                "signals": {
                    "A": signals.exclusive_fullscreen,
                    "B": signals.borderless_fullscreen,
                    "C": signals.in_game_list,
                    "D": signals.non_ollama_3d_util,
                    "E": signals.solitaire_gate,
                },
                "results": [
                    {"model": r.get("model"), "error": r.get("error"), "done_reason": r.get("done_reason")}
                    for r in results
                ],
            },
        )
        with self._lock:
            self._status = "yielded"

    def poll_once(self) -> str | None:
        """Run one detection cycle. Returns transition kind or None."""
        if not self.enabled:
            return None
        signals = self._collect()
        detected = is_gaming(signals, exclude=self.exclude, self_pid=self._self_pid)
        with self._lock:
            self._last_signals = {
                "A": signals.exclusive_fullscreen,
                "B": signals.borderless_fullscreen,
                "C": signals.in_game_list,
                "D": signals.non_ollama_3d_util,
                "E": signals.solitaire_gate,
                "pid": signals.pid,
                "name": signals.name,
                "vram_bytes": signals.vram_bytes,
                "engine_3d_pct": signals.engine_3d_pct,
                "raw_detected": detected,
            }
        transition = self._hyst.update(detected)
        if transition == "detected":
            self._log(
                "gaming_detected",
                {
                    "pid": signals.pid,
                    "name": signals.name,
                    "exe_path": signals.exe_path,
                    "signals": {
                        "A": signals.exclusive_fullscreen,
                        "B": signals.borderless_fullscreen,
                        "C": signals.in_game_list,
                        "D": signals.non_ollama_3d_util,
                        "E": signals.solitaire_gate,
                    },
                },
            )
            with self._lock:
                self._status = "detected"
            if self.yield_enabled:
                self._maybe_unload(signals)
        elif transition == "cleared":
            self._busy_idle_streak = 0
            self._log("gaming_cleared", {"pid": signals.pid, "name": signals.name})
            with self._lock:
                self._status = "idle"
        elif self._hyst.active:
            if self.yield_enabled and self._status != "yielded":
                self._maybe_unload(signals)
            with self._lock:
                if self._status not in ("yielded", "detected"):
                    self._status = "detected"
        return transition

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:
                self._log("gaming_error", {"error": str(exc)})
            self._stop.wait(self.interval)

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gaming-yield")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
