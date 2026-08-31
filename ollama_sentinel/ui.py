"""Flet Material 3 window."""

from __future__ import annotations

import atexit
import sys
import threading
import time
from typing import Any

import flet as ft

from ollama_sentinel.metrics import make_metrics_store
from ollama_sentinel.activity import build_server_activity
from ollama_sentinel.alarms import evaluate_alarms
from ollama_sentinel.catalog import search_models
from ollama_sentinel.config import AppConfig, selected_servers
from ollama_sentinel.doctor import (
    collect_doctor_inputs,
    evaluate_doctor_alarms,
    run_doctor,
)
from ollama_sentinel.instance import InstanceLock
from ollama_sentinel.inventory import build_inventory, inventory_summary
from ollama_sentinel.poll import poll_all
from ollama_sentinel.proc_vram import ProcessVramCollector
from ollama_sentinel.restart import spawn_restart
from ollama_sentinel.unload import unload_models
from ollama_sentinel.smi import query_gpus
from ollama_sentinel.state import load_state, save_state
from ollama_sentinel.telemetry import format_poll_age, is_stale
from ollama_sentinel.gaming import parse_exclude_list
from ollama_sentinel.gaming_yield import GamingYieldWatcher
from ollama_sentinel.ui_charts import metrics_charts_panel
from ollama_sentinel.ui_widgets import (
    PALETTE,
    activity_card,
    alarm_banner,
    gpu_table,
    library_table,
    loaded_models_table,
    process_vram_table,
    section_card,
)


def _free_vram_summary(gpus: list[dict[str, Any]] | None) -> tuple[float | None, float | None]:
    if not gpus:
        return None, None
    total = sum(g.get("memory_total") or 0 for g in gpus)
    free = sum(g.get("memory_free") or 0 for g in gpus)
    if total <= 0:
        return None, None
    return free / 1e9, 100 * free / total


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

        async def show_window_async() -> None:
            page.window.visible = True
            page.window.skip_task_bar = False
            page.window.minimized = False
            page.window.to_front()
            page.update()

        async def hide_window_async() -> None:
            page.window.visible = False
            page.window.skip_task_bar = True
            page.update()

        def _is_close_event(e) -> bool:
            if getattr(e, "type", None) == ft.WindowEventType.CLOSE:
                return True
            return getattr(e, "data", None) == "close"

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
            release_lock()
            spawn_restart()
            icon = tray_icon.get("icon")
            if icon is not None:
                icon.stop()
            page.window.prevent_close = False
            page.window.destroy()

        def request_restart_app() -> None:
            page.run_task(restart_app_async)

        if tray:
            from ollama_sentinel.tray import start_tray

            page.window.prevent_close = True

            def on_window_event(e) -> None:
                if _is_close_event(e):
                    request_hide_window()

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

            page.window.on_event = on_window_event
            if start_hidden:
                page.window.visible = False
                page.window.skip_task_bar = True

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
        current_server = ft.Dropdown(
            label="Server",
            value=server_names[0] if server_names else None,
            options=[ft.dropdown.Option(n) for n in server_names],
            width=220,
        )
        alarm_host = ft.Container()
        gpu_host = ft.Column(spacing=8)
        models_host = ft.Column(spacing=8)
        activity_host = ft.Container()
        proc_vram_host = ft.Column(spacing=8)
        gaming_status = ft.Text("", size=12, color=PALETTE["muted"])
        doctor_status = ft.Text("", size=12, color=PALETTE["muted"])
        poll_footer = ft.Text("", size=12, color=PALETTE["muted"])
        poll_state: dict[str, Any] = {"polled_ts": None, "stale": False, "reachable": True}
        library_host = ft.Column(spacing=8, expand=True)
        charts_host = ft.Column(spacing=8, expand=True, scroll=ft.ScrollMode.AUTO)
        chart_window_s = {"value": 300.0}
        discover_col = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)
        search_field = ft.TextField(label="Search Hugging Face", expand=True)
        pull_status = ft.Text("")
        unload_status = ft.Text("")

        def get_server_cfg():
            name = current_server.value
            for s in servers:
                if s.name == name:
                    return s
            return servers[0]

        def update_charts() -> None:
            if metrics_store is None:
                charts_host.controls = [
                    ft.Text("Metrics disabled (METRICS=0)", size=12, color=PALETTE["muted"]),
                ]
                return
            srv = get_server_cfg()
            charts_host.controls = [
                metrics_charts_panel(
                    metrics_store,
                    window_s=chart_window_s["value"],
                    server=srv.name,
                ),
            ]

        def on_chart_window(e) -> None:
            sel = e.control.selected
            if sel:
                chart_window_s["value"] = float(sel[0])
            update_charts()
            page.update()

        chart_window_pick = ft.SegmentedButton(
            selected=["300"],
            segments=[
                ft.Segment(value="300", label="5m"),
                ft.Segment(value="900", label="15m"),
                ft.Segment(value="3600", label="1h"),
            ],
            on_change=on_chart_window,
        )

        def update_poll_footer(now: float | None = None) -> None:
            polled_ts = poll_state.get("polled_ts")
            if polled_ts is None:
                return
            tick = now if now is not None else time.time()
            stale_poll = bool(
                poll_state.get("stale")
                or is_stale(polled_ts, cfg.poll_interval, tick)
            )
            footer = format_poll_age(polled_ts, tick)
            if stale_poll:
                poll_footer.value = f"STALE · {footer}"
                poll_footer.color = PALETTE["stale"]
            else:
                poll_footer.value = f"Updated {footer}"
                poll_footer.color = PALETTE["muted"]

        def _close_dialog(dlg: ft.AlertDialog) -> None:
            dlg.open = False
            page.update()

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
            page.overlay.append(dlg)
            dlg.open = True
            page.update()

        def _run_unload(model_names: list[str]) -> None:
            srv = get_server_cfg()
            label = model_names[0] if len(model_names) == 1 else f"{len(model_names)} models"
            unload_status.value = f"Unloading {label}…"
            page.update()

            def worker() -> None:
                results = unload_models(srv.url, model_names)
                errors = [r for r in results if r.get("error")]
                if errors:
                    unload_status.value = errors[0]["error"]
                else:
                    unload_status.value = f"Unloaded {label}"
                page.update()
                refresh()

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
            srv = get_server_cfg()
            target = [{"name": srv.name, "url": srv.url, "local_gpu": srv.local_gpu}]
            snap = poll_all(
                target,
                gpu_filter=cfg.gpu_filter,
                query_gpus_fn=query_gpus,
            )[0]
            last_snap.clear()
            last_snap.update(snap)
            reachable = bool(snap.get("reachable"))

            if metrics_store is not None:
                metrics_store.ingest_snapshot(snap)

            state = load_state(cfg.state_file)
            active, new_state, _ = evaluate_alarms(snap, state, cfg.thresholds)

            doctor_alarms: list[dict[str, Any]] = []
            if reachable and srv.local_gpu and sys.platform == "win32":
                try:
                    inputs = collect_doctor_inputs()
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
                    )
                    doctor_alarms = evaluate_doctor_alarms(findings)
                    active = list(active) + doctor_alarms
                    new_state.active_ids = {a["id"] for a in active}
                except Exception:
                    doctor_alarms = []

            save_state(cfg.state_file, new_state)

            if doctor_alarms:
                n = len(doctor_alarms)
                doctor_status.value = (
                    f"Doctor: {n} warning{'s' if n != 1 else ''} — run ollama-sentinel doctor"
                )
                doctor_status.color = PALETTE["warn"]
            else:
                doctor_status.value = ""
                doctor_status.color = PALETTE["muted"]

            icon = tray_icon.get("icon")
            if icon is not None:
                from ollama_sentinel.tray import set_tray_color

                try:
                    set_tray_color(icon, active)
                except Exception:
                    pass

            now = time.time()
            polled_ts = snap.get("polled_at_ts")
            poll_state["polled_ts"] = polled_ts
            poll_state["stale"] = bool(snap.get("stale"))
            poll_state["reachable"] = reachable
            if not reachable:
                poll_footer.value = (
                    f"Unreachable · {format_poll_age(polled_ts, now)}"
                    if polled_ts is not None
                    else "Unreachable"
                )
                poll_footer.color = PALETTE["alarm"]
            else:
                update_poll_footer(now)

            alarm_host.content = alarm_banner(
                reachable,
                active,
                error=snap.get("error"),
            )

            gpu_host.controls.clear()
            for gpu in snap.get("gpus") or []:
                gpu_host.controls.append(gpu_table(gpu))

            models_host.controls.clear()
            if reachable:
                models_card = loaded_models_table(
                    snap.get("models") or [],
                    server_url=srv.url,
                    on_unload=request_unload,
                )
                if models_card is not None:
                    models_host.controls.append(models_card)

            activity_host.content = None
            if reachable and srv.local_gpu and sys.platform == "win32":
                proc_rows = None
                if proc_collector:
                    proc_rows = (proc_collector.get_snapshot() or {}).get("rows")
                act = build_server_activity(proc_rows=proc_rows)
                card = activity_card(act)
                if card is not None:
                    activity_host.content = card

            proc_vram_host.controls.clear()
            if proc_collector:
                pv = proc_collector.get_snapshot()
                if metrics_store is not None and pv.get("rows"):
                    metrics_store.ingest_proc_vram(
                        pv.get("rows") or [],
                        ts=pv.get("polled_at_ts"),
                    )
                pv_ts = pv.get("polled_at_ts")
                pv_age = format_poll_age(pv_ts, now) if pv_ts is not None else None
                pv_stale = bool(
                    pv.get("stale")
                    or (pv_ts is not None and is_stale(pv_ts, cfg.proc_vram_interval, now))
                )
                proc_vram_host.controls.append(
                    process_vram_table(
                        pv.get("rows") or [],
                        stale=pv_stale,
                        age_text=pv_age,
                        error=pv.get("error"),
                    )
                )

            library_host.controls.clear()
            if reachable:
                inv = build_inventory(snap)
                free_gb, free_pct = _free_vram_summary(snap.get("gpus"))
                summary = inventory_summary(inv, free_vram_gb=free_gb, free_vram_pct=free_pct)
                library_host.controls.append(
                    section_card(
                        "Library",
                        library_table(inv, on_unload=request_unload),
                        subtitle=summary,
                    )
                )
            else:
                library_host.controls.append(
                    section_card(
                        "Library",
                        ft.Text("Ollama is not running.", size=12, color=PALETTE["muted"]),
                    )
                )

            unload_all_btn.disabled = not reachable or not bool(snap.get("models"))

            if gaming_watcher:
                gst = gaming_watcher.get_status()
                mode = "yield on" if cfg.gaming_yield else "observe"
                gaming_status.value = f"Gaming: {gst.get('status', 'idle')} ({mode})"
                if gst.get("status") == "yielded":
                    gaming_status.color = PALETTE["warn"]
                elif gst.get("status") == "detected":
                    gaming_status.color = PALETTE["ok"]
                else:
                    gaming_status.color = PALETTE["muted"]
            else:
                gaming_status.value = ""

            update_charts()
            page.update()

        def do_search(_=None) -> None:
            q = search_field.value or ""
            results = search_models(q, token=cfg.hf_token) if len(q) >= 2 else search_models("", token=cfg.hf_token)
            discover_col.controls.clear()
            for r in results[:20]:
                model = r["pull_name"]

                def make_pull(m=model):
                    def handler(_e):
                        srv = get_server_cfg()
                        pull_status.value = f"Pulling {m}…"
                        page.update()

                        def worker():
                            for ev in pull_model(srv.url, m):
                                if "error" in ev:
                                    pull_status.value = ev["error"]
                                    page.update()
                                    return
                                pull_status.value = str(ev.get("status") or ev)
                                page.update()
                            pull_status.value = f"Done: {m}"
                            page.update()
                            refresh()

                        threading.Thread(target=worker, daemon=True).start()

                    return handler

                discover_col.controls.append(
                    ft.Card(
                        content=ft.Container(
                            content=ft.Row(
                                [
                                    ft.Column(
                                        [
                                            ft.Text(r["id"], weight=ft.FontWeight.BOLD),
                                            ft.Text(f"downloads: {r.get('downloads') or '—'}", size=12),
                                        ],
                                        expand=True,
                                    ),
                                    ft.ElevatedButton("Install", on_click=make_pull()),
                                ]
                            ),
                            padding=12,
                        )
                    )
                )
            page.update()

        unload_all_btn = ft.OutlinedButton(
            "Unload all",
            on_click=lambda _: request_unload_all(),
            disabled=True,
        )
        action_row = ft.Row(
            [
                ft.ElevatedButton("Refresh", on_click=refresh),
                ft.OutlinedButton("Restart", on_click=lambda _: request_restart_app()),
                unload_all_btn,
            ],
            spacing=8,
        )

        status_page = ft.Column(
            [
                alarm_host,
                gpu_host,
                models_host,
                activity_host,
                proc_vram_host,
                gaming_status,
                doctor_status,
                unload_status,
                poll_footer,
                action_row,
            ],
            expand=True,
            scroll=ft.ScrollMode.AUTO,
            spacing=10,
        )
        charts_page = ft.Column(
            [
                ft.Row(
                    [
                        ft.Text("Window", size=12, color=PALETTE["muted"]),
                        chart_window_pick,
                    ],
                    spacing=8,
                ),
                charts_host,
                action_row,
            ],
            expand=True,
            spacing=10,
        )
        library_page = ft.Column([library_host, action_row], expand=True, spacing=10)
        discover_page = ft.Column(
            [search_field, ft.ElevatedButton("Search", on_click=do_search), pull_status, discover_col],
            expand=True,
            spacing=10,
        )
        pages = [status_page, charts_page, library_page, discover_page]

        nav = ft.NavigationRail(
            selected_index=0,
            destinations=[
                ft.NavigationRailDestination(icon=ft.Icons.MONITOR_HEART, label="Status"),
                ft.NavigationRailDestination(icon=ft.Icons.SHOW_CHART, label="Charts"),
                ft.NavigationRailDestination(icon=ft.Icons.LIBRARY_BOOKS, label="Library"),
                ft.NavigationRailDestination(icon=ft.Icons.SEARCH, label="Discover"),
            ],
        )
        content_area = ft.Container(
            content=ft.Column(
                [current_server, status_page],
                expand=True,
                spacing=10,
            ),
            expand=True,
            padding=ft.Padding(left=8),
        )

        def on_nav(e):
            idx = int(e.control.selected_index)
            page_body = pages[idx]
            content_area.content = ft.Column(
                [current_server, page_body],
                expand=True,
                spacing=10,
            )
            page.update()

        nav.on_change = on_nav
        body = ft.Row([nav, ft.VerticalDivider(width=1), content_area], expand=True)
        page.add(body)
        refresh()

        def poll_loop():
            while True:
                time.sleep(cfg.poll_interval)
                try:
                    refresh()
                except Exception:
                    pass

        def footer_tick_loop():
            while True:
                time.sleep(1)
                try:
                    if poll_state.get("polled_ts") is None:
                        continue
                    if not poll_state.get("reachable", True):
                        polled_ts = poll_state["polled_ts"]
                        poll_footer.value = f"Unreachable · {format_poll_age(polled_ts, time.time())}"
                        poll_footer.color = PALETTE["alarm"]
                    else:
                        update_poll_footer()
                    poll_footer.update()
                except Exception:
                    pass

        def show_request_loop():
            while True:
                time.sleep(1)
                try:
                    if instance_lock and instance_lock.consume_show_request():
                        request_show_window()
                except Exception:
                    pass

        threading.Thread(target=poll_loop, daemon=True).start()
        threading.Thread(target=footer_tick_loop, daemon=True).start()
        if instance_lock is not None:
            threading.Thread(target=show_request_loop, daemon=True).start()

    ft.app(target=app)
