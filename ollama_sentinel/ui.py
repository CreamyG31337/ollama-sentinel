"""Flet Material 3 window."""

from __future__ import annotations

import asyncio
import atexit
import logging
import logging.handlers
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import flet as ft

from ollama_sentinel.metrics import make_metrics_store
from ollama_sentinel.activity import (
    build_peer_name_map,
    build_server_activity,
    listen_port_from_url,
)
from ollama_sentinel.alarms import evaluate_alarms
from ollama_sentinel.catalog import SEARCH_SORTS, fetch_model_bundle, search_models
from ollama_sentinel.client_config import load_client_config
from ollama_sentinel.config import AppConfig, selected_servers
from ollama_sentinel.doctor import (
    collect_doctor_inputs,
    evaluate_doctor_alarms,
    run_doctor,
)
from ollama_sentinel.instance import InstanceLock
from ollama_sentinel.advisor import (
    advisories_for_model,
    advisor_log_context,
    evaluate_advisories,
    evaluate_advisor_alarms,
)
from ollama_sentinel.inventory import build_inventory, enrich_inventory_rows, inventory_summary
from ollama_sentinel.poll import poll_all
from ollama_sentinel.panel_hold import UnreachableHold
from ollama_sentinel.refresh_guard import RefreshGuard, SingleFlight
from ollama_sentinel.net_errors import format_network_error
from ollama_sentinel.pull import pull_model
from ollama_sentinel.proc_vram import ProcessVramCollector
from ollama_sentinel.restart import spawn_restart
from ollama_sentinel.unload import unload_models
from ollama_sentinel.show import ShowCache
from ollama_sentinel.smi import query_gpus
from ollama_sentinel.state import load_state, save_state
from ollama_sentinel.telemetry import (
    format_freshness_line,
    format_poll_age,
    is_stale,
    polled_at_iso,
)
from ollama_sentinel.gaming import parse_exclude_list
from ollama_sentinel.gaming_yield import GamingYieldWatcher
from ollama_sentinel.tray import find_icon_asset
from ollama_sentinel.ui_charts import charts_subtitle, build_live_charts_panel
from ollama_sentinel.settings import apply_to_config, effective, load_settings, set_setting
from ollama_sentinel.ui_widgets import (
    PALETTE,
    LiveFreshnessBanner,
    activity_card,
    activity_fingerprint,
    advisor_panel,
    alarm_banner,
    discover_result_tile,
    gpu_fingerprint,
    gpu_table,
    library_table,
    loaded_models_table,
    process_vram_table,
    section_card,
    settings_panel,
)

log = logging.getLogger(__name__)

# Warn-once state for paint/loop failure sites: the first failure at a site
# logs a WARNING with traceback; repeats are suppressed for 10 minutes so a
# per-tick failure cannot flood the log (or be silently swallowed either).
_WARN_ONCE_INTERVAL_S = 600.0
_warn_once_state: dict[str, float] = {}
_warn_once_lock = threading.Lock()


def _warn_once(site: str, exc: BaseException) -> None:
    now = time.monotonic()
    with _warn_once_lock:
        last = _warn_once_state.get(site)
        if last is not None and (now - last) < _WARN_ONCE_INTERVAL_S:
            return
        _warn_once_state[site] = now
    log.warning("ui: %s failed: %r", site, exc, exc_info=exc)


# Discover detail cache cap (READMEs are large; 20 expanded tiles is plenty).
_DETAIL_CACHE_MAX = 20


def _gui_log_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "ollama-sentinel"
    return Path.home() / ".ollama-sentinel"


def setup_gui_logging(log_dir: Path | None = None) -> Path | None:
    """Attach a rotating file handler to the ``ollama_sentinel`` logger.

    Under ``pythonw`` ``sys.stderr`` is ``None``, so ``logging.lastResort``
    silently drops every record: the warn-once failure sites and the
    visibility INFO lines produced nowhere. Idempotent — calling again for
    the same path adds no second handler. Returns the log file path, or
    ``None`` when the directory cannot be created (logging stays as-is
    rather than taking the GUI down).
    """
    directory = log_dir if log_dir is not None else _gui_log_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "gui.log"
    except OSError:
        return None
    logger = logging.getLogger("ollama_sentinel")
    for handler in logger.handlers:
        if (
            isinstance(handler, logging.handlers.RotatingFileHandler)
            and handler.baseFilename == str(path)
        ):
            return path
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
    except OSError:
        return None
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    return path


def _free_vram_summary(gpus: list[dict[str, Any]] | None) -> tuple[float | None, float | None]:
    if not gpus:
        return None, None
    total = sum(g.get("memory_total") or 0 for g in gpus)
    free = sum(g.get("memory_free") or 0 for g in gpus)
    if total <= 0:
        return None, None
    return free / 1e9, 100 * free / total


def loading_caption(server_name: str | None) -> str:
    """Placeholder shown on every host-scoped panel while a switch poll runs."""
    return f"Loading {server_name or 'server'}..."


def host_context_line(name: str | None, url: str | None, *, loading: bool = False) -> str:
    """Always-visible attribution: which host the panels below belong to."""
    label = name or "server"
    endpoint = url or ""
    if loading:
        return f"Loading {label}  ·  {endpoint}".rstrip(" ·")
    return f"{label}  ·  {endpoint}".rstrip(" ·")


def host_dropdown_label(name: str, online: bool | None) -> str:
    """Dropdown row text: status marker + server name (key stays the bare name)."""
    if online is True:
        return f"● online  ·  {name}"
    if online is False:
        return f"○ offline  ·  {name}"
    return f"· …  ·  {name}"


def host_status_color(online: bool | None) -> str:
    if online is True:
        return PALETTE["ok"]
    if online is False:
        return PALETTE["alarm"]
    return PALETTE["muted"]


def host_dropdown_option(name: str, online: bool | None) -> ft.dropdown.Option:
    """Server menu row with a green/red online/offline marker."""
    if online is True:
        marker, status = "●", "online"
    elif online is False:
        marker, status = "○", "offline"
    else:
        marker, status = "·", "…"
    color = host_status_color(online)
    return ft.dropdown.Option(
        key=name,
        text=host_dropdown_label(name, online),
        content=ft.Row(
            [
                ft.Text(marker, size=12, color=color),
                ft.Text(status, size=12, color=color, weight=ft.FontWeight.W_600),
                ft.Text(f"·  {name}", size=12),
            ],
            spacing=6,
            tight=True,
        ),
    )


def compose_active_alarms(
    base: list[dict[str, Any]] | None,
    doctor: list[dict[str, Any]] | None = None,
    advisor: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Concatenate base, doctor, and advisor alarms without duplicating IDs.

    Preserves first-seen order: base first, then doctor, then advisor.
    """
    seen: set[str] = set()
    composed: list[dict[str, Any]] = []
    for group in (base, doctor, advisor):
        if not group:
            continue
        for alarm in group:
            if not isinstance(alarm, dict):
                composed.append(alarm)
                continue
            aid = alarm.get("id")
            if aid is not None:
                if aid in seen:
                    continue
                seen.add(aid)
            composed.append(alarm)
    return composed


def clear_switch_state(
    last_snap: dict[str, Any],
    poll_state: dict[str, Any],
    last_advisories: list[Any] | None = None,
    last_show_by_model: dict[str, Any] | None = None,
    last_doctor_alarms: list[Any] | None = None,
    last_advisor_alarms: list[Any] | None = None,
    hold: UnreachableHold | None = None,
) -> None:
    """Drop cached snap / poll age and held data so the footer cannot claim the previous host."""
    last_snap.clear()
    poll_state["polled_ts"] = None
    poll_state["live_ts"] = None
    poll_state["stale"] = False
    poll_state["alarms"] = []
    poll_state["reachable"] = True
    poll_state["optional"] = False
    poll_state["live_gpus"] = None
    poll_state["live_gpus_server"] = None
    if last_advisories is not None:
        last_advisories.clear()
    if last_show_by_model is not None:
        last_show_by_model.clear()
    if last_doctor_alarms is not None:
        last_doctor_alarms.clear()
    if last_advisor_alarms is not None:
        last_advisor_alarms.clear()
    if hold is not None:
        hold.reset()


def show_local_process_panels(local_gpu: bool) -> bool:
    """Process VRAM and gaming are this machine's data, not a remote Ollama URL."""
    return bool(local_gpu)


def run_gui(
    cfg: AppConfig,
    *,
    tray: bool = False,
    start_hidden: bool = False,
    instance_lock: InstanceLock | None = None,
) -> None:
    tray_icon: dict = {"icon": None}
    released = {"done": False}

    def release_lock() -> None:
        if released["done"]:
            return
        if instance_lock is not None:
            instance_lock.release()
        released["done"] = True

    if instance_lock is not None:
        atexit.register(release_lock)

    def app(page: ft.Page) -> None:
        page.title = "ollama-sentinel"
        page.theme_mode = ft.ThemeMode.DARK
        page.window.width = 960
        page.window.height = 640
        page.padding = 12
        # Windows taskbar / title-bar icon (Flet's default is the fishy logo).
        icon_path = find_icon_asset()
        if icon_path is not None:
            page.window.icon = str(icon_path.resolve())

        # ---- Cross-thread paint marshaling -------------------------------------
        # Flet 0.86 Page.update() diffs the control tree and mutates the
        # session index on the *calling* thread with no lock. Two worker
        # loops patching overlapping subtrees can corrupt that state and the
        # Flutter client stops rendering — the "blank white window after
        # days" failure. Every mutation of a mounted control (and its
        # update) must therefore run on the event loop; this helper is the
        # only sanctioned path from a worker thread.
        ui_loop = page.loop  # == page.session.connection.loop

        def _on_loop() -> bool:
            try:
                return asyncio.get_running_loop() is ui_loop
            except RuntimeError:
                return False

        def ui_call(fn, *, wait: bool = False, timeout: float = 10.0) -> None:
            """Run fn (control mutations + update) on the Flet event loop."""

            def run() -> None:
                try:
                    fn()
                except Exception as exc:  # surface instead of vanishing
                    _warn_once(f"ui_call:{getattr(fn, '__name__', 'paint')}", exc)

            if _on_loop():
                run()  # event handlers and on-loop loops need no marshaling
                return

            async def _run() -> None:
                run()

            fut = page.run_task(_run)
            if wait:
                # Only ever called from a worker thread — awaiting on-loop
                # would deadlock the loop the task needs.
                fut.result(timeout=timeout)

        # ---- Visibility gating -------------------------------------------------
        # Painting into a hidden/minimized window only floods the client with
        # a backlog it must chew through on resume (Flet's own
        # wait_until_visible docstring). Data paths keep running; only the
        # paints are skipped, and repaint_all() rebuilds the tree on return.
        ui_state = {"visible": not start_hidden, "missed": False}
        repaint_gate = {"last": 0.0}

        def can_paint() -> bool:
            # app_visible covers minimize via the app lifecycle state.
            return ui_state["visible"] and getattr(page, "app_visible", True)

        def paint_blocked() -> bool:
            """Gate for timer-driven paints: records that one was skipped.

            ``missed`` is what tells a FOCUS/RESTORE event whether a repaint
            is actually owed, so an ordinary alt-tab (nothing skipped, window
            never hidden) costs nothing.
            """
            if can_paint():
                return False
            ui_state["missed"] = True
            return True

        async def show_window_async() -> None:
            if not ui_state["visible"]:
                log.info("ui visible=True (show_window)")
            ui_state["visible"] = True
            page.window.visible = True
            page.window.skip_task_bar = False
            page.window.minimized = False
            page.window.to_front()
            page.update()
            await repaint_all()

        async def hide_window_async() -> None:
            if ui_state["visible"]:
                log.info("ui visible=False (hide_window)")
            ui_state["visible"] = False
            page.window.visible = False
            page.window.skip_task_bar = True
            page.update()

        def _is_close_event(e) -> bool:
            if getattr(e, "type", None) == ft.WindowEventType.CLOSE:
                return True
            return getattr(e, "data", None) == "close"

        # A burst of SHOW/RESTORE/FOCUS events must trigger one repaint, not
        # one per event — request_repaint debounces to one pass per 2 s.
        _VISIBLE_WINDOW_EVENTS = (
            ft.WindowEventType.SHOW,
            ft.WindowEventType.RESTORE,
            ft.WindowEventType.FOCUS,
        )

        def _is_show_event(e) -> bool:
            etype = getattr(e, "type", None)
            data = getattr(e, "data", None)
            return any(etype == ev or data == ev.value for ev in _VISIBLE_WINDOW_EVENTS)

        async def close_window_async() -> None:
            if gaming_watcher:
                gaming_watcher.stop()
            if proc_collector:
                proc_collector.stop()
            release_lock()
            page.window.prevent_close = False
            page.window.destroy()

        async def quit_app_async() -> None:
            icon = tray_icon.get("icon")
            if icon is not None:
                icon.stop()
            await close_window_async()

        def request_show_window() -> None:
            page.run_task(show_window_async)

        def request_hide_window() -> None:
            page.run_task(hide_window_async)

        def request_quit_app() -> None:
            page.run_task(quit_app_async)

        def request_close_window() -> None:
            page.run_task(close_window_async)

        async def restart_app_async() -> None:
            if gaming_watcher:
                gaming_watcher.stop()
            if proc_collector:
                proc_collector.stop()
            icon = tray_icon.get("icon")
            if icon is not None:
                icon.stop()
            page.window.visible = False
            page.window.skip_task_bar = True
            page.update()
            release_lock()
            spawn_restart()

        def request_restart_app() -> None:
            page.run_task(restart_app_async)

        if tray:
            from ollama_sentinel.tray import start_tray

            page.window.prevent_close = True

            def on_window_event(e) -> None:
                if _is_close_event(e):
                    request_hide_window()
                elif _is_show_event(e) and ui_state["missed"]:
                    request_repaint()

            page.window.on_event = on_window_event
            tray_icon["icon"] = start_tray(
                on_open=request_show_window,
                on_restart=request_restart_app,
                on_quit=request_quit_app,
            )

            if start_hidden:
                page.window.visible = False
                page.window.skip_task_bar = True
        else:

            def on_window_event(e) -> None:
                if _is_close_event(e):
                    request_close_window()
                elif _is_show_event(e) and ui_state["missed"]:
                    request_repaint()

            page.window.on_event = on_window_event
            if start_hidden:
                page.window.visible = False
                page.window.skip_task_bar = True

        # Coming back from minimized/hidden: app_visible is driven by Flet's
        # lifecycle events (HIDE/PAUSE -> not visible). If it ever sticks at
        # False on Windows the UI would freeze, so transitions are logged.
        def _on_app_lifecycle_change(e) -> None:
            state = getattr(e, "state", None)
            if state is None:
                return
            visible = state not in (ft.AppLifecycleState.HIDE, ft.AppLifecycleState.PAUSE)
            log.info("app lifecycle %s -> visible=%s", getattr(state, "value", state), visible)
            if visible and ui_state["missed"]:
                request_repaint()

        page.on_app_lifecycle_state_change = _on_app_lifecycle_change

        servers = selected_servers(cfg)
        proc_collector: ProcessVramCollector | None = None
        gaming_watcher: GamingYieldWatcher | None = None
        metrics_store = make_metrics_store(cfg)
        last_snap: dict[str, Any] = {}
        if cfg.proc_vram and any(s.local_gpu for s in servers):
            proc_collector = ProcessVramCollector(
                interval=cfg.proc_vram_interval,
                enabled=True,
                min_bytes=cfg.proc_vram_min_mb * 1024 * 1024,
            )
            proc_collector.start()

        if any(s.local_gpu for s in servers) and (cfg.gaming_yield_observe or cfg.gaming_yield):
            local_srv = next((s for s in servers if s.local_gpu), servers[0])

            def _list_loaded() -> list[str]:
                snap = last_snap if last_snap.get("reachable") else {}
                names = [
                    m.get("name") or m.get("model")
                    for m in snap.get("models") or []
                    if m.get("name") or m.get("model")
                ]
                if names:
                    return names
                # Fallback fresh poll if UI cache empty
                polled = poll_all(
                    [{"name": local_srv.name, "url": local_srv.url, "local_gpu": True}],
                    gpu_filter=cfg.gpu_filter,
                    query_gpus_fn=query_gpus,
                )[0]
                return [
                    m.get("name") or m.get("model")
                    for m in polled.get("models") or []
                    if m.get("name") or m.get("model")
                ]

            gaming_watcher = GamingYieldWatcher(
                enabled=True,
                yield_enabled=cfg.gaming_yield,
                interval=cfg.gaming_yield_interval,
                exclude=parse_exclude_list(cfg.gaming_yield_exclude),
                min_vram_bytes=cfg.gaming_yield_min_vram_mb * 1024 * 1024,
                min_util=cfg.gaming_yield_min_util,
                busy_util=cfg.gaming_yield_busy_util,
                ollama_url=local_srv.url,
                get_proc_snapshot=(proc_collector.get_snapshot if proc_collector else None),
                list_loaded_models=_list_loaded,
            )
            gaming_watcher.start()

        server_names = [s.name for s in servers]
        # None = not probed yet; updated by fleet TCP probe + selected-host poll.
        host_online: dict[str, bool | None] = {s.name: None for s in servers}

        def rebuild_server_options() -> None:
            selected = current_server.value
            current_server.options = [
                host_dropdown_option(s.name, host_online.get(s.name)) for s in servers
            ]
            if selected in host_online:
                current_server.value = selected

        current_server = ft.Dropdown(
            label="Server",
            value=server_names[0] if server_names else None,
            options=[
                host_dropdown_option(n, host_online.get(n)) for n in server_names
            ],
            width=340,
        )
        _first = servers[0] if servers else None
        host_banner = ft.Text(
            host_context_line(
                _first.name if _first else None,
                _first.url if _first else None,
            ),
            size=15,
            weight=ft.FontWeight.BOLD,
            color=PALETTE["muted"],
        )
        alarm_host = ft.Container()
        gpu_host = ft.Column(spacing=8)
        models_host = ft.Column(spacing=8)
        activity_host = ft.Container()
        proc_vram_host = ft.Column(spacing=8)
        gaming_status = ft.Text("", size=12, color=PALETTE["muted"])
        doctor_status = ft.Text("", size=12, color=PALETTE["muted"])
        advisor_status = ft.Container()
        update_status_line = ft.Text("", size=12, color=PALETTE["muted"])
        show_cache = ShowCache(ttl=cfg.show_cache_ttl) if cfg.advisor else None
        last_advisories: list = []
        last_show_by_model: dict[str, dict[str, Any]] = {}
        last_doctor_alarms: list[dict[str, Any]] = []
        last_advisor_alarms: list[dict[str, Any]] = []
        unreachable_hold = UnreachableHold()
        # One persistent freshness strip, mutated in place by the 1 Hz tick —
        # not a rebuilt banner per second.
        live_freshness = LiveFreshnessBanner(interval_s=cfg.poll_interval)
        freshness_host = ft.Container(content=live_freshness.control)
        poll_footer = ft.Text("", size=12, color=PALETTE["muted"])
        poll_state: dict[str, Any] = {
            "polled_ts": None,
            "live_ts": None,
            "stale": False,
            "reachable": True,
            "optional": False,
        }
        # Guards against a slow poll for the previous server landing after the
        # user has already switched away, which used to leave the panels showing
        # one server's data under another server's name.
        refresh_guard = RefreshGuard()
        # Last-painted fingerprints: skip rebuilding GPU / activity cards when
        # nothing rendered has changed. Reset on host switch and repaint_all.
        gpu_fp: dict[str, Any] = {"value": None}
        activity_fp: dict[str, Any] = {"value": None}
        library_host = ft.Column(
            [ft.Text("Waiting for first poll…", size=12, color=PALETTE["muted"])],
            spacing=8,
            expand=True,
            scroll=ft.ScrollMode.AUTO,
        )
        charts_subtitle_text = ft.Text("", size=12, color=PALETTE["muted"])
        chart_window_s = {"value": 300.0}
        nav_state = {"index": 0}
        # nvidia-smi is ~70ms here — safe to sample between full Ollama polls.
        LIVE_GPU_INTERVAL_S = 2
        live_charts = build_live_charts_panel(subtitle=charts_subtitle_text)
        charts_host = live_charts.column
        discover_col = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True, spacing=8)
        discover_state: dict[str, Any] = {
            "sort": "trendingScore",
            "search_gen": 0,
            "search_error": None,
            "expanded_id": None,
            "last_results": [],
            "detail_cache": {},
            "detail_errors": {},
            "detail_loading": set(),
        }
        discover_sort = ft.Dropdown(
            label="Sort by",
            width=180,
            value="trendingScore",
            options=[ft.dropdown.Option(key, label) for key, label in SEARCH_SORTS],
        )
        search_field = ft.TextField(label="Search Hugging Face", expand=True, autofocus=True)
        pull_status = ft.Text("")
        unload_status = ft.Text("")

        def get_server_cfg():
            name = current_server.value
            for s in servers:
                if s.name == name:
                    return s
            return servers[0]

        def update_charts(*, push: bool = True) -> None:
            """Refresh persistent chart Canvases from the metrics store."""
            if metrics_store is None:
                charts_subtitle_text.value = "Metrics disabled (set METRICS=1 in .env)"
                if push:
                    try:
                        charts_subtitle_text.update()
                    except Exception as exc:
                        _warn_once("charts:subtitle", exc)
                return
            srv = get_server_cfg()
            live_charts.refresh(
                metrics_store,
                window_s=chart_window_s["value"],
                server=srv.name,
                poll_interval=LIVE_GPU_INTERVAL_S,
                push=push,
            )

        def ingest_live_gpu_metrics() -> bool:
            """Sample nvidia-smi into the metrics store between full Ollama polls."""
            if metrics_store is None:
                return False
            srv = get_server_cfg()
            target = srv.name
            if not srv.local_gpu:
                return False
            if not poll_state.get("reachable", True):
                return False
            try:
                gpus = query_gpus(cfg.gpu_filter)
            except Exception:
                return False
            if not gpus:
                return False
            # Drop the sample if the user switched hosts while nvidia-smi ran.
            if get_server_cfg().name != target:
                return False
            now = time.time()
            metrics_store.ingest_snapshot(
                {
                    "reachable": True,
                    "server": target,
                    "polled_at_ts": now,
                    "models": last_snap.get("models") or [],
                    "gpus": gpus,
                }
            )
            poll_state["live_gpus"] = gpus
            poll_state["live_gpus_server"] = target
            return True

        def paint_gpu_cards(gpus: list[dict[str, Any]] | None) -> None:
            """Rebuild GPU cards only when the displayed values changed."""
            fp = gpu_fingerprint(gpus)
            if fp == gpu_fp["value"]:
                return
            gpu_fp["value"] = fp
            gpu_host.controls = [gpu_table(gpu) for gpu in gpus or []]

        def paint_live_gpu_cards() -> None:
            gpus = poll_state.get("live_gpus")
            if gpus is None or nav_state["index"] != 0:
                return
            if poll_state.get("live_gpus_server") != get_server_cfg().name:
                return
            paint_gpu_cards(gpus)
            try:
                gpu_host.update()
            except Exception as exc:
                _warn_once("paint_live_gpu_cards", exc)

        def on_chart_window(e) -> None:
            sel = e.control.selected
            if sel:
                chart_window_s["value"] = float(sel[0])
            update_charts(push=True)

        async def charts_live_loop() -> None:
            """UI-loop task: sample GPU off-thread, paint Canvases on-thread.

            Replacing chart controls from a worker thread never redraws Flet
            Canvas; mutating shapes here (same path as range-button clicks) does.
            """
            import asyncio

            while True:
                await asyncio.sleep(LIVE_GPU_INTERVAL_S)
                try:
                    # Sampling keeps running while hidden — only the paints
                    # are gated, so the metrics store has no gap on resume.
                    srv = get_server_cfg()
                    if srv.local_gpu and poll_state.get("reachable", True):
                        ingested = await asyncio.to_thread(ingest_live_gpu_metrics)
                        if ingested and not paint_blocked():
                            paint_live_gpu_cards()
                    # Always refresh chart shapes from the store (covers full-poll
                    # ingest too) so the Charts tab moves without a range click.
                    # The freshness footer is owned by the 1 Hz footer tick.
                    if not paint_blocked():
                        update_charts(push=True)
                except Exception as exc:
                    _warn_once("charts_live_loop", exc)

        chart_window_pick = ft.SegmentedButton(
            selected=["300"],
            segments=[
                ft.Segment(value="300", label="5m"),
                ft.Segment(value="900", label="15m"),
                ft.Segment(value="3600", label="1h"),
            ],
            on_change=on_chart_window,
        )

        def update_poll_footer(now: float | None = None) -> bool:
            tick = now if now is not None else time.time()
            level, label = format_freshness_line(
                poll_state.get("polled_ts"),
                cfg.poll_interval,
                tick,
                reachable=bool(poll_state.get("reachable", True)),
                live_at=poll_state.get("live_ts"),
            )
            if poll_state.get("optional") and not poll_state.get("reachable", True):
                label = label.replace("Unreachable", "Offline (optional)", 1)
                if level == "stale":
                    level = "aging"
            # Mutate the persistent banner in place — no remount per tick.
            changed = live_freshness.set(level, label)
            # Keep a quiet footer line for scroll-to-bottom context.
            if level == "stale":
                value = label
                color = PALETTE["stale"] if poll_state.get("reachable", True) else (
                    PALETTE["warn"] if poll_state.get("optional") else PALETTE["alarm"]
                )
            elif level == "aging":
                value = label
                color = PALETTE["warn"]
            else:
                value = label
                color = PALETTE["muted"]
            if poll_footer.value != value or poll_footer.color != color:
                poll_footer.value = value
                poll_footer.color = color
                changed = True
            return changed

        def paint_activity(act) -> bool:
            """Rebuild the activity card only when what it renders changed.

            Returns whether the content changed, so callers can skip the
            update entirely on an idle server.
            """
            if act is None:
                changed = activity_host.content is not None
                activity_host.content = None
                activity_fp["value"] = None
                return changed
            fp = activity_fingerprint(act)
            if fp == activity_fp["value"]:
                return False
            activity_fp["value"] = fp
            activity_host.content = activity_card(act)
            return True

        def rebuild_live_activity() -> bool:
            """Cheap status refresh: re-read server.log / peers without a full poll."""
            srv = get_server_cfg()
            if not srv.local_gpu:
                return False
            if not poll_state.get("reachable", True):
                return False
            proc_rows = None
            if proc_collector:
                proc_rows = (proc_collector.get_snapshot() or {}).get("rows")
            act = build_server_activity(
                proc_rows=proc_rows,
                models=last_snap.get("models"),
                peer_names=build_peer_name_map(load_client_config(cfg.client_config)),
                listen_port=listen_port_from_url(srv.url),
            )
            if act is not None:
                poll_state["live_ts"] = time.time()

            def paint() -> None:
                if paint_blocked():
                    return
                # Scoped diff of the one host this tick can change; skipped
                # entirely while the fingerprint says nothing rendered moved.
                if paint_activity(act):
                    page.update(activity_host)

            ui_call(paint)
            # Tray is pystray, not Flet — stays on this worker thread.
            icon = tray_icon.get("icon")
            if icon is not None:
                from ollama_sentinel.tray import update_tray

                try:
                    update_tray(
                        icon,
                        reachable=True,
                        alarms=poll_state.get("alarms") or [],
                        phase=act.phase,
                        summary=act.summary,
                        server=srv.name,
                    )
                except Exception as exc:
                    _warn_once("tray:update", exc)
            return True

        def _close_dialog(dlg: ft.AlertDialog) -> None:
            # pop_dialog removes it from the dialog stack; the old
            # overlay.append + open=False path leaked one overlay entry per
            # confirmation.
            try:
                page.pop_dialog()
            except Exception as exc:
                _warn_once("dialog:close", exc)

        def _confirm_dialog(title: str, message: str, on_confirm) -> None:
            def cancel(e) -> None:
                _close_dialog(dlg)

            def confirm(e) -> None:
                _close_dialog(dlg)
                on_confirm()

            dlg = ft.AlertDialog(
                modal=True,
                title=ft.Text(title),
                content=ft.Text(message),
                actions=[
                    ft.TextButton("Cancel", on_click=cancel),
                    ft.TextButton("Unload", on_click=confirm),
                ],
                actions_alignment=ft.MainAxisAlignment.END,
            )
            page.show_dialog(dlg)

        def _run_unload(model_names: list[str]) -> None:
            srv = get_server_cfg()
            label = model_names[0] if len(model_names) == 1 else f"{len(model_names)} models"

            def paint_start() -> None:
                unload_status.value = f"Unloading {label}…"
                page.update()

            ui_call(paint_start)

            def worker() -> None:
                results = unload_models(srv.url, model_names)
                errors = [r for r in results if r.get("error")]

                def paint_done() -> None:
                    if errors:
                        unload_status.value = errors[0]["error"]
                    else:
                        unload_status.value = f"Unloaded {label}"
                    page.update()

                ui_call(paint_done)
                kick_refresh()

            threading.Thread(target=worker, daemon=True).start()

        def request_unload(model_name: str) -> None:
            _confirm_dialog(
                f"Unload {model_name}?",
                "Removes this model from VRAM for every client on this Ollama server "
                "(Open WebUI, CLI, etc.). Installed files are not deleted.",
                lambda: _run_unload([model_name]),
            )

        def request_unload_all() -> None:
            names = [
                m.get("name") or m.get("model")
                for m in last_snap.get("models") or []
                if m.get("name") or m.get("model")
            ]
            if not names:
                unload_status.value = "No loaded models"
                page.update()
                return
            joined = ", ".join(names[:3])
            if len(names) > 3:
                joined += f", +{len(names) - 3} more"
            _confirm_dialog(
                "Unload all loaded models?",
                f"This will evict: {joined}. "
                "Use in an emergency to free VRAM; active chats will break.",
                lambda: _run_unload(names),
            )

        def refresh(_=None) -> None:
            """Full poll of the selected server.

            Runs on a worker thread (HTTP, nvidia-smi, registry, /api/show).
            Everything that touches a mounted control is done inside paint
            closures handed to ui_call(..., wait=True), which marshal them
            onto the Flet event loop — worker-thread patching of the control
            tree is what blanked the window after days of uptime.
            """
            srv = get_server_cfg()
            my_seq = refresh_guard.issue()
            target = [
                {
                    "name": srv.name,
                    "url": srv.url,
                    "local_gpu": srv.local_gpu,
                    "optional": srv.optional,
                }
            ]
            try:
                snap = poll_all(
                    target,
                    gpu_filter=cfg.gpu_filter,
                    query_gpus_fn=query_gpus,
                )[0]
            except Exception as exc:
                now = time.time()
                snap = {
                    "server": srv.name,
                    "url": srv.url,
                    "reachable": False,
                    "optional": srv.optional,
                    "local_gpu": srv.local_gpu,
                    "gpu_data_available": False,
                    "version": None,
                    "models": [],
                    "tags": [],
                    "gpus": None,
                    "error": format_network_error(exc, context=srv.name),
                    "polled_at_ts": now,
                    "polled_at": polled_at_iso(now),
                    "stale": False,
                }
            # Poll finished. Apply it only if it is still what the user is
            # looking at and nothing newer has already landed; otherwise a
            # 30s timeout on an unreachable host would overwrite fresh data.
            if not refresh_guard.accept(my_seq, srv.name, current_server.value):
                return

            # ---- compute on this worker thread; no mounted control is touched ----
            now = time.time()
            reachable = bool(snap.get("reachable"))
            disposition = unreachable_hold.record(srv.name, reachable, now)

            if disposition == "hold":
                host_online[srv.name] = False
                if metrics_store is not None:
                    metrics_store.ingest_snapshot(snap)

                state = load_state(cfg.state_file)
                active, new_state, _ = evaluate_alarms(snap, state, cfg.thresholds)
                save_state(cfg.state_file, new_state)

                last_good = unreachable_hold.last_good_ts(srv.name)
                poll_state["polled_ts"] = last_good
                poll_state["stale"] = bool(snap.get("stale"))
                poll_state["reachable"] = False
                poll_state["optional"] = bool(snap.get("optional"))
                poll_state["alarms"] = list(active)
                poll_state["live_ts"] = None

                icon = tray_icon.get("icon")
                if icon is not None:
                    from ollama_sentinel.tray import update_tray

                    try:
                        update_tray(
                            icon,
                            reachable=False,
                            alarms=active,
                            phase=None,
                            summary=snap.get("error"),
                            server=srv.name,
                        )
                    except Exception as exc:
                        _warn_once("tray:update", exc)

                def paint_hold() -> None:
                    if not refresh_guard.still_current(my_seq, srv.name, current_server.value):
                        return
                    if paint_blocked():
                        return
                    rebuild_server_options()
                    host_banner.value = host_context_line(srv.name, srv.url, loading=False)
                    host_banner.color = PALETTE["warn"]
                    update_poll_footer(now)
                    unload_all_btn.disabled = True
                    page.update()

                ui_call(paint_hold, wait=True)
                return

            last_snap.clear()
            last_snap.update(snap)
            host_online[srv.name] = reachable

            if not reachable:
                last_advisories.clear()
                last_show_by_model.clear()
                last_doctor_alarms.clear()
                last_advisor_alarms.clear()

            if metrics_store is not None:
                metrics_store.ingest_snapshot(snap)

            state = load_state(cfg.state_file)
            base_active, new_state, _ = evaluate_alarms(snap, state, cfg.thresholds)
            active_all = compose_active_alarms(
                base_active, last_doctor_alarms, last_advisor_alarms
            )
            new_state.active_ids = {
                a["id"] for a in active_all if isinstance(a, dict) and "id" in a
            }
            save_state(cfg.state_file, new_state)

            update_text = ""
            update_key = "muted"
            if srv.local_gpu:
                try:
                    from ollama_sentinel.ollama_update import (
                        format_update_status_line,
                        maybe_auto_apply,
                        update_status as read_update_status,
                    )
                    from ollama_sentinel.settings import effective as setting_value

                    if setting_value("update_check"):
                        st = read_update_status(running_version=snap.get("version"))
                        auto = bool(setting_value("update_auto_apply"))
                        idle_seconds = float(setting_value("update_idle_seconds"))
                        started, reason = False, ""
                        if st.pending and auto:
                            activity = build_server_activity(
                                fresh_seconds=max(idle_seconds, 45.0),
                                models=snap.get("models"),
                                include_peers=False,
                            )
                            started, reason = maybe_auto_apply(
                                snap,
                                activity,
                                enabled=True,
                                idle_seconds=idle_seconds,
                            )
                        update_text, update_key = format_update_status_line(
                            pending=st.pending,
                            summary=st.summary,
                            auto_apply=auto,
                            started=started,
                            reason=reason,
                        )
                except Exception as exc:
                    _warn_once("refresh:update_status", exc)

            if reachable:
                poll_state["polled_ts"] = snap.get("polled_at_ts")
            else:
                last_good = unreachable_hold.last_good_ts(srv.name)
                poll_state["polled_ts"] = (
                    last_good if last_good is not None else snap.get("polled_at_ts")
                )
            poll_state["stale"] = bool(snap.get("stale"))
            poll_state["reachable"] = reachable
            poll_state["optional"] = bool(snap.get("optional"))
            poll_state["alarms"] = list(active_all)

            act = None
            if reachable and srv.local_gpu:
                proc_rows = None
                if proc_collector:
                    proc_rows = (proc_collector.get_snapshot() or {}).get("rows")
                act = build_server_activity(
                    proc_rows=proc_rows,
                    models=snap.get("models"),
                    peer_names=build_peer_name_map(load_client_config(cfg.client_config)),
                    listen_port=listen_port_from_url(srv.url),
                )
                poll_state["live_ts"] = now
            else:
                poll_state["live_ts"] = None

            # Tray is pystray, not Flet — stays on this worker thread.
            icon = tray_icon.get("icon")
            if icon is not None:
                from ollama_sentinel.tray import update_tray

                try:
                    update_tray(
                        icon,
                        reachable=reachable,
                        alarms=active_all,
                        phase=getattr(act, "phase", None) if act is not None else None,
                        summary=getattr(act, "summary", None) if act is not None else (
                            snap.get("error") if not reachable else None
                        ),
                        server=srv.name,
                    )
                except Exception as exc:
                    _warn_once("tray:update", exc)

            # Controls are built detached (plain construction, safe off-loop)
            # and mounted inside the paint closure.
            alarm_ctrl = alarm_banner(
                reachable,
                active_all,
                error=snap.get("error"),
                optional=bool(snap.get("optional")),
            )
            models_card = (
                loaded_models_table(
                    snap.get("models") or [],
                    server_url=srv.url,
                    on_unload=request_unload,
                )
                if reachable
                else None
            )
            proc_ctrl = None
            if proc_collector:
                pv = proc_collector.get_snapshot()
                # Always ingest local runner util for charts; only build the
                # table when the selected server is this machine.
                if metrics_store is not None and pv.get("rows"):
                    metrics_store.ingest_proc_vram(
                        pv.get("rows") or [],
                        ts=pv.get("polled_at_ts"),
                        server=srv.name if srv.local_gpu else None,
                    )
                if show_local_process_panels(srv.local_gpu):
                    pv_ts = pv.get("polled_at_ts")
                    pv_age = format_poll_age(pv_ts, now) if pv_ts is not None else None
                    pv_stale = bool(
                        pv.get("stale")
                        or (pv_ts is not None and is_stale(pv_ts, cfg.proc_vram_interval, now))
                    )
                    proc_ctrl = process_vram_table(
                        pv.get("rows") or [],
                        stale=pv_stale,
                        age_text=pv_age,
                        error=pv.get("error"),
                    )

            if reachable:
                inv = build_inventory(snap)
                if last_show_by_model:
                    inv = enrich_inventory_rows(inv, last_show_by_model)
                advisories_by_model = {
                    r["name"]: advisories_for_model(last_advisories, r["name"]) for r in inv
                }
                free_gb, free_pct = _free_vram_summary(snap.get("gpus"))
                summary = inventory_summary(inv, free_vram_gb=free_gb, free_vram_pct=free_pct)
                library_ctrl = section_card(
                    "Library",
                    library_table(
                        inv,
                        on_unload=request_unload,
                        advisories_by_model=advisories_by_model if cfg.advisor else None,
                    ),
                    subtitle=summary,
                )
            else:
                err = snap.get("error") or "Ollama is not reachable"
                subtitle = "Optional host — offline is normal" if srv.optional else None
                library_ctrl = section_card(
                    "Library",
                    ft.Column(
                        [
                            ft.Text(err, size=12, color=PALETTE["alarm"] if not srv.optional else PALETTE["warn"]),
                            ft.Text(
                                "Switch servers above or press Refresh when Ollama is back.",
                                size=11,
                                color=PALETTE["muted"],
                            ),
                        ],
                        spacing=4,
                    ),
                    subtitle=subtitle,
                )

            unload_disabled = not reachable or not bool(snap.get("models"))

            gaming_text = ""
            gaming_color = PALETTE["muted"]
            if gaming_watcher and show_local_process_panels(srv.local_gpu):
                gst = gaming_watcher.get_status()
                mode = "yield on" if cfg.gaming_yield else "observe"
                gaming_text = f"Gaming: {gst.get('status', 'idle')} ({mode})"
                if gst.get("status") == "yielded":
                    gaming_color = PALETTE["warn"]
                elif gst.get("status") == "detected":
                    gaming_color = PALETTE["ok"]

            gpus = snap.get("gpus")

            # ---- paint 1: main status (on the event loop) ----
            def paint_main() -> None:
                if not refresh_guard.still_current(my_seq, srv.name, current_server.value):
                    return
                if paint_blocked():
                    return
                rebuild_server_options()
                host_banner.value = host_context_line(srv.name, srv.url, loading=False)
                host_banner.color = PALETTE["ok"] if reachable else (
                    PALETTE["warn"] if snap.get("optional") else PALETTE["alarm"]
                )
                update_poll_footer(now)
                alarm_host.content = alarm_ctrl
                paint_gpu_cards(gpus)
                models_host.controls = [models_card] if models_card is not None else []
                paint_activity(act)
                proc_vram_host.controls = [proc_ctrl] if proc_ctrl is not None else []
                library_host.controls = [library_ctrl]
                unload_all_btn.disabled = unload_disabled
                gaming_status.value = gaming_text
                gaming_status.color = gaming_color
                if not reachable:
                    doctor_status.value = ""
                    doctor_status.color = PALETTE["muted"]
                    advisor_status.content = None
                update_status_line.value = update_text
                update_status_line.color = PALETTE[update_key]
                # Canvas must be painted on the Flet event loop (worker-thread
                # canvas.update is a no-op / silently fails).
                update_charts(push=True)
                page.update()

            ui_call(paint_main, wait=True)

            # ---- doctor after first paint (local host only, worker I/O) ----
            if (
                reachable
                and srv.local_gpu
                and sys.platform == "win32"
                and refresh_guard.still_current(my_seq, srv.name, current_server.value)
            ):
                try:
                    inputs = collect_doctor_inputs()
                    if not refresh_guard.still_current(
                        my_seq, srv.name, current_server.value
                    ):
                        return
                    proc_rows = None
                    if proc_collector:
                        proc_rows = (proc_collector.get_snapshot() or {}).get("rows")
                    findings = run_doctor(
                        snap,
                        registry=inputs["registry"],
                        log_cfg=inputs["log_cfg"],
                        log_path=inputs["log_path"],
                        runners=inputs["runners"],
                        proc_rows=proc_rows,
                        ollama_url=cfg.ollama_url,
                        registry_mtime=inputs["registry_mtime"],
                        restart_remedy=inputs["restart_remedy"],
                        driver_version=inputs.get("driver_version"),
                        driver_cuda=inputs.get("driver_cuda"),
                    )
                    doctor_alarms = evaluate_doctor_alarms(findings)
                    last_doctor_alarms.clear()
                    last_doctor_alarms.extend(doctor_alarms)
                    active_all = compose_active_alarms(
                        base_active, last_doctor_alarms, last_advisor_alarms
                    )
                    new_state.active_ids = {
                        a["id"] for a in active_all if isinstance(a, dict) and "id" in a
                    }
                    save_state(cfg.state_file, new_state)
                    poll_state["alarms"] = list(active_all)

                    doctor_ctrl = alarm_banner(
                        reachable,
                        active_all,
                        error=snap.get("error"),
                        optional=bool(snap.get("optional")),
                    )
                    n = len(doctor_alarms)
                    doctor_text = (
                        f"Doctor: {n} warning"
                        f"{'s' if n != 1 else ''} — run ollama-sentinel doctor"
                        if doctor_alarms
                        else ""
                    )
                    doctor_color = PALETTE["warn"] if doctor_alarms else PALETTE["muted"]

                    def paint_doctor() -> None:
                        if not refresh_guard.still_current(
                            my_seq, srv.name, current_server.value
                        ):
                            return
                        if paint_blocked():
                            return
                        alarm_host.content = doctor_ctrl
                        doctor_status.value = doctor_text
                        doctor_status.color = doctor_color
                        page.update()

                    ui_call(paint_doctor, wait=True)
                except Exception as exc:
                    _warn_once("refresh:doctor", exc)

            # ---- second pass: enrich Library / advisor (worker I/O) ----
            if (
                cfg.advisor
                and show_cache is not None
                and reachable
                and refresh_guard.still_current(my_seq, srv.name, current_server.value)
            ):
                try:
                    from ollama_sentinel.client_config import (
                        installed_model_names,
                        load_client_config,
                        missing_client_models,
                    )

                    names = [
                        t.get("name") or t.get("model")
                        for t in snap.get("tags") or []
                        if t.get("name") or t.get("model")
                    ]
                    show_by_model = show_cache.fetch_all(snap["url"], names)
                    if not refresh_guard.still_current(
                        my_seq, srv.name, current_server.value
                    ):
                        return
                    log_cfg, keep_alive = advisor_log_context()
                    clients = load_client_config(cfg.client_config)
                    # Single-server GUI cannot prove a model is missing fleet-wide.
                    client_missing = missing_client_models(
                        clients,
                        installed_model_names([snap]),
                        inventory_complete=False,
                    )
                    advisor_findings = evaluate_advisories(
                        snap,
                        show_by_model=show_by_model,
                        log_cfg=log_cfg,
                        keep_alive=keep_alive,
                        client_missing=client_missing or None,
                        gpu_data_available=bool(snap.get("gpu_data_available")),
                    )
                    last_show_by_model.clear()
                    last_show_by_model.update(show_by_model)
                    last_advisories.clear()
                    last_advisories.extend(advisor_findings)
                    advisor_alarms = evaluate_advisor_alarms(advisor_findings)
                    last_advisor_alarms.clear()
                    last_advisor_alarms.extend(advisor_alarms)

                    active_all = compose_active_alarms(
                        base_active, last_doctor_alarms, last_advisor_alarms
                    )
                    new_state.active_ids = {
                        a["id"] for a in active_all if isinstance(a, dict) and "id" in a
                    }
                    save_state(cfg.state_file, new_state)
                    poll_state["alarms"] = list(active_all)

                    advisor_alarm_ctrl = alarm_banner(
                        reachable,
                        active_all,
                        error=snap.get("error"),
                        optional=bool(snap.get("optional")),
                    )

                    advisor_ctrl = advisor_panel(last_advisories)

                    inv = build_inventory(snap)
                    if last_show_by_model:
                        inv = enrich_inventory_rows(inv, last_show_by_model)
                    advisories_by_model = {
                        r["name"]: advisories_for_model(last_advisories, r["name"])
                        for r in inv
                    }
                    free_gb, free_pct = _free_vram_summary(snap.get("gpus"))
                    summary = inventory_summary(
                        inv, free_vram_gb=free_gb, free_vram_pct=free_pct
                    )
                    library_ctrl = section_card(
                        "Library",
                        library_table(
                            inv,
                            on_unload=request_unload,
                            advisories_by_model=advisories_by_model,
                        ),
                        subtitle=summary,
                    )

                    def paint_advisor() -> None:
                        if not refresh_guard.still_current(
                            my_seq, srv.name, current_server.value
                        ):
                            return
                        if paint_blocked():
                            return
                        alarm_host.content = advisor_alarm_ctrl
                        advisor_status.content = advisor_ctrl
                        library_host.controls = [library_ctrl]
                        page.update()

                    ui_call(paint_advisor, wait=True)
                except Exception as exc:
                    _warn_once("refresh:advisor", exc)

        def request_pull(model_name: str) -> None:
            srv = get_server_cfg()

            def paint_start() -> None:
                pull_status.value = f"Pulling {model_name}…"
                page.update()

            ui_call(paint_start)

            def worker() -> None:
                for ev in pull_model(srv.url, model_name):
                    if "error" in ev:
                        text = format_network_error(RuntimeError(ev["error"]), context="Pull")

                        def paint_error(text=text) -> None:
                            pull_status.value = text
                            page.update()

                        ui_call(paint_error)
                        return

                    def paint_progress(text=str(ev.get("status") or ev)) -> None:
                        pull_status.value = text
                        page.update()

                    ui_call(paint_progress)

                def paint_done() -> None:
                    pull_status.value = f"Done: {model_name}"
                    page.update()

                ui_call(paint_done)
                kick_refresh()

            threading.Thread(target=worker, daemon=True).start()

        def render_discover_results() -> None:
            discover_col.controls.clear()
            search_error = discover_state.get("search_error")
            if search_error:
                discover_col.controls.append(
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Text(
                                    "Search failed",
                                    size=14,
                                    weight=ft.FontWeight.BOLD,
                                    color=PALETTE["alarm"],
                                ),
                                ft.Text(search_error, size=12, color=ft.Colors.WHITE70),
                                ft.Text(
                                    "Check your connection and press Search to retry.",
                                    size=11,
                                    color=PALETTE["muted"],
                                ),
                            ],
                            spacing=4,
                        ),
                        padding=12,
                        border_radius=8,
                        bgcolor=ft.Colors.RED_900,
                    )
                )
                return
            results = discover_state.get("last_results") or []
            if not results:
                discover_col.controls.append(
                    ft.Text("No models found.", size=12, color=PALETTE["muted"])
                )
                return
            for item in results:
                model_id = item.get("id") or ""

                def make_expand_handler(mid: str = model_id):
                    def handler(e) -> None:
                        if e.control.expanded:
                            discover_state["expanded_id"] = mid
                            load_detail_if_needed(mid)
                        elif discover_state.get("expanded_id") == mid:
                            discover_state["expanded_id"] = None

                    return handler

                discover_col.controls.append(
                    discover_result_tile(
                        item,
                        detail=discover_state["detail_cache"].get(model_id),
                        detail_error=discover_state["detail_errors"].get(model_id),
                        detail_loading=model_id in discover_state["detail_loading"],
                        expanded=discover_state.get("expanded_id") == model_id,
                        on_pull=request_pull,
                        on_open_hf=lambda url=item.get("hf_url", ""): page.launch_url(url),
                        on_expand_change=make_expand_handler(),
                    )
                )

        def load_detail_if_needed(model_id: str) -> None:
            if model_id in discover_state["detail_cache"]:
                render_discover_results()
                page.update()
                return
            if model_id in discover_state["detail_loading"]:
                return
            discover_state["detail_loading"].add(model_id)
            render_discover_results()
            page.update()

            def worker() -> None:
                try:
                    bundle = fetch_model_bundle(model_id, token=cfg.hf_token)
                    cache = discover_state["detail_cache"]
                    cache[model_id] = bundle
                    discover_state["detail_errors"].pop(model_id, None)
                    # Bounded: a long discover session must not accumulate
                    # every README ever expanded.
                    while len(cache) > _DETAIL_CACHE_MAX:
                        oldest = next(iter(cache))
                        cache.pop(oldest, None)
                        discover_state["detail_errors"].pop(oldest, None)
                except Exception as exc:
                    discover_state["detail_errors"][model_id] = format_network_error(
                        exc, context="Model details"
                    )
                finally:
                    discover_state["detail_loading"].discard(model_id)

                def paint() -> None:
                    render_discover_results()
                    page.update()

                ui_call(paint)

            threading.Thread(target=worker, daemon=True).start()

        def do_search(_=None, *, sort_override: str | None = None) -> None:
            query = search_field.value or ""
            sort = sort_override or discover_sort.value or discover_state["sort"]
            discover_state["sort"] = sort
            discover_state["search_gen"] = discover_state.get("search_gen", 0) + 1
            gen = discover_state["search_gen"]
            discover_state["search_error"] = None
            pull_status.value = "Searching…"
            page.update()

            def worker() -> None:
                status_text = ""
                try:
                    if len(query.strip()) >= 2:
                        results = search_models(
                            query.strip(),
                            sort=sort,
                            limit=20,
                            token=cfg.hf_token,
                        )
                    else:
                        results = search_models("", sort=sort, limit=20, token=cfg.hf_token)
                    if discover_state.get("search_gen") != gen:
                        return
                    discover_state["last_results"] = results
                    discover_state["search_error"] = None
                    status_text = f"{len(results)} result{'s' if len(results) != 1 else ''}"
                except Exception as exc:
                    if discover_state.get("search_gen") != gen:
                        return
                    discover_state["last_results"] = []
                    discover_state["search_error"] = format_network_error(exc, context="Discover")
                    status_text = "Search failed"

                def paint(status_text=status_text) -> None:
                    pull_status.value = status_text
                    render_discover_results()
                    page.update()

                ui_call(paint)

            threading.Thread(target=worker, daemon=True).start()

        def on_discover_sort_change(e) -> None:
            do_search(sort_override=e.control.value)

        discover_sort.on_select = on_discover_sort_change
        search_field.on_submit = do_search

        unload_all_btn = ft.OutlinedButton(
            "Unload all",
            on_click=lambda _: request_unload_all(),
            disabled=True,
        )

        refresh_flight = SingleFlight()

        def refresh_worker() -> None:
            """Coalescing single-flight: manual clicks, host switches and the
            poll timer share one worker; requests arriving mid-run cause
            exactly one rerun rather than stacking threads or being dropped.
            """
            while True:
                if not refresh_flight.request():
                    return  # a run is in flight and will pick up our request
                try:
                    refresh()
                except Exception as exc:
                    _warn_once("refresh", exc)
                if not refresh_flight.finish():
                    return

        def kick_refresh(_=None) -> None:
            """Never run refresh on the Flet event thread.

            refresh() does HTTP, nvidia-smi, registry reads and /api/show;
            inline on_click froze the window. Same worker path as host switch
            and the poll timer.
            """
            threading.Thread(target=refresh_worker, daemon=True).start()

        def probe_fleet_reachability() -> None:
            """Lightweight TCP check for every configured host (dropdown dots)."""
            from concurrent.futures import ThreadPoolExecutor

            from ollama_sentinel.poll import _tcp_connect_error

            def one(srv) -> tuple[str, bool]:
                return srv.name, _tcp_connect_error(srv.url, timeout=0.5) is None

            try:
                with ThreadPoolExecutor(max_workers=min(8, max(1, len(servers)))) as pool:
                    for name, ok in pool.map(one, servers):
                        host_online[name] = ok
            except Exception as exc:
                _warn_once("fleet_probe", exc)
                return

            def paint() -> None:
                if paint_blocked():
                    return
                rebuild_server_options()
                try:
                    current_server.update()
                except Exception as exc:
                    _warn_once("fleet_probe:paint", exc)

            ui_call(paint)

        def kick_fleet_probe() -> None:
            threading.Thread(target=probe_fleet_reachability, daemon=True).start()

        action_row = ft.Row(
            [
                ft.ElevatedButton("Refresh", on_click=kick_refresh),
                ft.OutlinedButton("Restart", on_click=lambda _: request_restart_app()),
                unload_all_btn,
            ],
            spacing=8,
        )

        status_page = ft.Column(
            [
                freshness_host,
                alarm_host,
                gpu_host,
                models_host,
                activity_host,
                proc_vram_host,
                gaming_status,
                doctor_status,
                advisor_status,
                update_status_line,
                unload_status,
                poll_footer,
            ],
            expand=True,
            scroll=ft.ScrollMode.AUTO,
            spacing=10,
        )
        charts_page = ft.Column(
            [
                ft.Text("Trends", size=18, weight=ft.FontWeight.BOLD),
                charts_subtitle_text,
                ft.Row(
                    [
                        ft.Text("Range", size=12, color=PALETTE["muted"]),
                        chart_window_pick,
                    ],
                    spacing=8,
                ),
                charts_host,
            ],
            expand=True,
            spacing=10,
        )
        # action_row lives in the shared chrome (_page_column), not in each page —
        # Flet controls can only have one parent, so sharing it across page Columns
        # left Library/Charts blank until a host switch rebuilt the tree.
        library_page = ft.Column([library_host], expand=True, spacing=10)
        discover_search_row = ft.Row(
            [
                search_field,
                ft.ElevatedButton("Search", on_click=do_search),
                discover_sort,
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )
        discover_page = ft.Column(
            [
                discover_search_row,
                pull_status,
                discover_col,
            ],
            expand=True,
            spacing=8,
        )
        settings_status = ft.Text("", size=12, color=PALETTE["muted"])
        stored_settings = load_settings()

        def _current(setting):
            return effective(setting.key, cfg, stored_settings)

        def _on_setting(key: str, value) -> None:
            """Persist a toggle and apply what can take effect without a restart."""
            try:
                stored_settings.clear()
                stored_settings.update(set_setting(key, value))
            except Exception as exc:
                settings_status.value = f"Could not save: {exc}"
                page.update()
                return
            apply_to_config(cfg, stored_settings)
            settings_status.value = f"Saved {key} = {stored_settings[key]}"
            page.update()

        settings_page = ft.Column(
            [
                ft.Text("Settings", size=18, weight=ft.FontWeight.BOLD),
                ft.Text(
                    "Stored per user; anything left untouched still comes from .env.",
                    size=11,
                    color=PALETTE["muted"],
                ),
                settings_panel(_on_setting, current=_current),
                settings_status,
            ],
            expand=True,
            spacing=10,
        )
        pages = [status_page, charts_page, library_page, discover_page, settings_page]

        nav = ft.NavigationRail(
            selected_index=0,
            destinations=[
                ft.NavigationRailDestination(icon=ft.Icons.MONITOR_HEART, label="Status"),
                ft.NavigationRailDestination(icon=ft.Icons.SHOW_CHART, label="Charts"),
                ft.NavigationRailDestination(icon=ft.Icons.LIBRARY_BOOKS, label="Library"),
                ft.NavigationRailDestination(icon=ft.Icons.SEARCH, label="Discover"),
                ft.NavigationRailDestination(icon=ft.Icons.SETTINGS, label="Settings"),
            ],
        )
        def _page_column(page_body: ft.Control) -> ft.Column:
            return ft.Column(
                [current_server, host_banner, page_body, action_row],
                expand=True,
                spacing=10,
            )

        content_area = ft.Container(
            content=_page_column(status_page),
            expand=True,
            padding=ft.Padding(left=8),
        )

        def blank_for_server(name: str | None) -> None:
            """Wipe every host-scoped panel so the previous device cannot linger.

            refresh() can take tens of seconds on an unreachable host; anything
            left painted would be attributed to the newly selected server.
            Must paint immediately (before the worker poll) so the user sees the
            switch within a second — not stale numbers under a new dropdown value.
            """
            # Kill in-flight polls so a late result cannot undo this blank.
            refresh_guard.invalidate()
            caption = loading_caption(name)
            clear_switch_state(
                last_snap,
                poll_state,
                last_advisories,
                last_show_by_model,
                last_doctor_alarms,
                last_advisor_alarms,
                unreachable_hold,
            )

            srv = next((s for s in servers if s.name == name), None)
            url = srv.url if srv else None
            host_banner.value = host_context_line(name, url, loading=True)
            host_banner.color = PALETTE["warn"]

            alarm_host.content = ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            caption,
                            size=16,
                            weight=ft.FontWeight.BOLD,
                            color=PALETTE["warn"],
                        ),
                        ft.Text(
                            "Waiting for this host — previous server cleared",
                            size=12,
                            color=ft.Colors.WHITE70,
                        ),
                    ],
                    spacing=4,
                ),
                padding=12,
                border_radius=8,
                bgcolor=ft.Colors.ORANGE_900,
            )
            models_host.controls = [
                ft.Text(caption, size=12, color=PALETTE["muted"])
            ]
            gpu_host.controls = []
            gpu_fp["value"] = None
            proc_vram_host.controls = []
            activity_host.content = None
            activity_fp["value"] = None
            live_freshness.set("unknown", caption)
            poll_state["live_ts"] = None
            poll_state["alarms"] = []
            poll_state["live_gpus"] = None
            poll_state["live_gpus_server"] = None

            library_host.controls = [
                ft.Text(caption, size=12, color=PALETTE["muted"])
            ]
            charts_subtitle_text.value = caption
            # Keep persistent chart Canvases; just clear their series.
            for chart in live_charts.charts:
                chart.set_series([], push=False)

            poll_footer.value = caption
            poll_footer.color = PALETTE["muted"]
            advisor_status.content = None
            doctor_status.value = ""
            doctor_status.color = PALETTE["muted"]
            gaming_status.value = ""
            gaming_status.color = PALETTE["muted"]
            update_status_line.value = ""
            update_status_line.color = PALETTE["muted"]
            unload_status.value = ""
            unload_all_btn.disabled = True

        def on_server_change(_e) -> None:
            """Switch servers without freezing the window.

            refresh() does three HTTP calls (10s timeout each) plus an
            /api/show per installed model, so running it on the event thread
            locked the UI for up to half a minute on an unreachable host.
            Every other I/O path in this file already uses a worker thread.
            """
            blank_for_server(current_server.value)
            # Push the blank before starting the poll so attribution cannot lag.
            try:
                host_banner.update()
                alarm_host.update()
                models_host.update()
                gpu_host.update()
                proc_vram_host.update()
                activity_host.update()
                freshness_host.update()
                library_host.update()
                charts_host.update()
                charts_subtitle_text.update()
                poll_footer.update()
            except Exception as exc:
                _warn_once("on_server_change:blank", exc)
            page.update()
            kick_refresh()

        # Flet 0.80+: Dropdown selection is on_select, not on_change (text typing).
        current_server.on_select = on_server_change

        def on_nav(e):
            idx = int(e.control.selected_index)
            nav_state["index"] = idx
            page_body = pages[idx]
            content_area.content = _page_column(page_body)
            # Charts were often mutated while unmounted; refresh shapes so the
            # tab never opens on stale geometry from the last visit / size.
            if idx == 1:
                update_charts(push=True)
            page.update()

        nav.on_change = on_nav

        async def repaint_all() -> None:
            """Rebuild the render tree from scratch — recovery path for a
            window that comes back from hidden/minimized (or blanked).

            Debounced to one pass per 2 s so a burst of SHOW/RESTORE/FOCUS
            events cannot queue a repaint each. Height is only restored when
            clearly collapsed (< 300) so a user-resized window is not stomped.
            """
            now = time.monotonic()
            if now - repaint_gate["last"] < 2.0:
                return
            repaint_gate["last"] = now
            ui_state["missed"] = False
            height = page.window.height
            if height is not None and height < 300:
                page.window.width = 960
                page.window.height = 640
            # Force card rebuilds: fingerprints from before the repaint are
            # meaningless for a freshly mounted tree.
            gpu_fp["value"] = None
            activity_fp["value"] = None
            content_area.content = _page_column(pages[nav_state["index"]])
            page.update()
            # Every panel repaints from a fresh poll.
            kick_refresh()

        def request_repaint() -> None:
            page.run_task(repaint_all)

        body = ft.Row([nav, ft.VerticalDivider(width=1), content_area], expand=True)
        page.add(body)
        kick_fleet_probe()
        kick_refresh()
        do_search()
        page.run_task(charts_live_loop)

        def poll_loop():
            while True:
                time.sleep(cfg.poll_interval)
                try:
                    kick_fleet_probe()
                    # Through the single-flight path so the timer and a manual
                    # Refresh / host switch never overlap.
                    kick_refresh()
                except Exception as exc:
                    _warn_once("poll_loop", exc)

        def footer_tick_loop():
            while True:
                time.sleep(1)
                try:
                    # Activity from logs is cheap; refresh it every second so
                    # n_gen / phase track generation without waiting for the
                    # full 5s Ollama poll. GPU/charts run on charts_live_loop
                    # (Flet event loop) so Canvas actually redraws.
                    try:
                        rebuild_live_activity()
                    except Exception as exc:
                        _warn_once("footer_tick:activity", exc)

                    def paint_footer() -> None:
                        if paint_blocked():
                            return
                        # Scoped diff: the 1 Hz tick owns only the banner and
                        # the footer line, and touches them only on change.
                        if update_poll_footer():
                            page.update(freshness_host, poll_footer)

                    ui_call(paint_footer)
                except Exception as exc:
                    _warn_once("footer_tick", exc)

        def show_request_loop():
            while True:
                time.sleep(1)
                try:
                    if instance_lock and instance_lock.consume_show_request():
                        request_show_window()
                except Exception as exc:
                    _warn_once("show_request_loop", exc)

        threading.Thread(target=poll_loop, daemon=True).start()
        threading.Thread(target=footer_tick_loop, daemon=True).start()
        if instance_lock is not None:
            threading.Thread(target=show_request_loop, daemon=True).start()

    icon_path = find_icon_asset()
    assets_dir = str(icon_path.parent.resolve()) if icon_path is not None else None
    setup_gui_logging()
    ft.app(target=app, assets_dir=assets_dir)
