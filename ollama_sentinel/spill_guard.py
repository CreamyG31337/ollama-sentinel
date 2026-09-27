"""Spill guard: unload local models that loaded mostly onto the CPU.

Ollama has no "refuse if it does not fit" option -- an oversized model loads anyway,
splits across VRAM and system RAM, and every request to it crawls. Sentinel is not
in the request path, so it cannot refuse the load; the guard catches it on the next
poll and evicts it (``keep_alive: 0``) instead.

Only ``local_gpu`` servers are guarded: unloading on a remote host would pull a model
out from under someone else. A model must be seen below the threshold on
``confirm_polls`` consecutive polls, so a runner caught mid-load is not evicted.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

from ollama_sentinel.alarms import AlarmTransition, gpu_pct
from ollama_sentinel.log import append_event_log
from ollama_sentinel.paths import app_data_dir
from ollama_sentinel.poll import _get_json
from ollama_sentinel.unload import unload_model

DEFAULT_MIN_GPU_PCT = 75
DEFAULT_CONFIRM_POLLS = 2
#: One toast per model per window, so a client retrying in a loop cannot flood.
NOTIFY_COOLDOWN_SEC = 600.0


def oversize_models(models: list[dict[str, Any]], min_gpu_pct: float) -> list[dict[str, Any]]:
    """Loaded models with less than ``min_gpu_pct`` of their weights on the GPU. Pure."""
    out: list[dict[str, Any]] = []
    for model in models or []:
        size = model.get("size") or 0
        size_vram = model.get("size_vram") or 0
        name = model.get("name") or model.get("model")
        if not name or size <= 0:
            continue
        pct = gpu_pct(size, size_vram)
        if pct < min_gpu_pct:
            out.append({"name": name, "gpu_pct": pct, "size": size, "size_vram": size_vram})
    return out


class SpillGuard:
    """Poll one local server's ``/api/ps`` and unload models that spill too far."""

    def __init__(
        self,
        *,
        server: str,
        url: str,
        interval: float = 5.0,
        confirm_polls: int = DEFAULT_CONFIRM_POLLS,
        enabled_fn: Callable[[], bool] = lambda: False,
        min_gpu_pct_fn: Callable[[], float] = lambda: DEFAULT_MIN_GPU_PCT,
        list_models: Callable[[], list[dict[str, Any]] | None] | None = None,
        unload_fn: Callable[[str, str], dict[str, Any]] | None = None,
        notify_fn: Callable[[AlarmTransition], None] | None = None,
        log_path: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.server = server
        self.url = url
        self.interval = interval
        self.confirm_polls = max(1, int(confirm_polls))
        self.enabled_fn = enabled_fn
        self.min_gpu_pct_fn = min_gpu_pct_fn
        self.list_models = list_models or self._fetch_models
        self.unload_fn = unload_fn or unload_model
        self.notify_fn = notify_fn
        self.log_path = log_path or (app_data_dir() / "spill_guard.jsonl")
        self.clock = clock
        self._streak: dict[str, int] = {}
        self._last_notified: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _fetch_models(self) -> list[dict[str, Any]] | None:
        data, err = _get_json(self.url, "/api/ps")
        if err or data is None:
            return None
        return data.get("models") or []

    def _log(self, event: str, payload: dict[str, Any]) -> None:
        try:
            append_event_log(self.log_path, event, payload=payload)
        except OSError:
            pass

    def poll_once(self) -> list[dict[str, Any]]:
        """Run one check. Returns the unload results for models evicted this cycle."""
        if not self.enabled_fn():
            self._streak.clear()
            return []
        models = self.list_models()
        if models is None:  # server down: forget streaks, a reload starts fresh
            self._streak.clear()
            return []

        threshold = float(self.min_gpu_pct_fn())
        offenders = {m["name"]: m for m in oversize_models(models, threshold)}
        self._streak = {name: self._streak.get(name, 0) + 1 for name in offenders}

        results: list[dict[str, Any]] = []
        for name, info in offenders.items():
            if self._streak[name] < self.confirm_polls:
                continue
            result = dict(self.unload_fn(self.url, name))
            result.setdefault("model", name)
            results.append(result)
            self._streak.pop(name, None)
            payload = {
                "server": self.server,
                "model": name,
                "gpu_pct": info["gpu_pct"],
                "size": info["size"],
                "size_vram": info["size_vram"],
                "min_gpu_pct": threshold,
                "error": result.get("error"),
            }
            self._log("spill_guard_unload", payload)
            self._maybe_notify(name, info["gpu_pct"], threshold, result.get("error"))
        return results

    def _maybe_notify(self, name: str, pct: int, threshold: float, error: Any) -> None:
        if self.notify_fn is None:
            return
        now = self.clock()
        last = self._last_notified.get(name)
        if last is not None and now - last < NOTIFY_COOLDOWN_SEC:
            return
        self._last_notified[name] = now
        if error:
            msg = f"SPILL GUARD [{self.server}] could not unload {name} ({pct}% GPU): {error}"
        else:
            msg = (
                f"SPILL GUARD [{self.server}] unloaded {name}: only {pct}% fit on the GPU "
                f"(minimum {threshold:.0f}%)"
            )
        try:
            self.notify_fn(AlarmTransition("FIRE", f"spill_guard:{self.server}:{name}", msg))
        except Exception:
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:
                self._log("spill_guard_error", {"error": str(exc)})
            self._stop.wait(self.interval)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="spill-guard")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


def make_spill_guard(cfg: Any, servers: list[Any], *, notify_fn=None) -> SpillGuard | None:
    """Build the guard for the first ``local_gpu`` server, or None when there is none.

    The toggle and threshold are re-read every poll, so flipping them in the GUI
    applies without a restart.
    """
    local = next((s for s in servers if getattr(s, "local_gpu", False)), None)
    if local is None:
        return None
    from ollama_sentinel.settings import effective

    return SpillGuard(
        server=local.name,
        url=local.url,
        interval=float(getattr(cfg, "poll_interval", 5.0) or 5.0),
        enabled_fn=lambda: bool(effective("spill_guard", cfg)),
        min_gpu_pct_fn=lambda: float(effective("spill_guard_min_gpu_pct", cfg)),
        notify_fn=notify_fn,
    )
