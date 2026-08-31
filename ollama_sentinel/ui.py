"""Flet Material 3 window."""

from __future__ import annotations

import atexit
import threading
import time
from typing import Any

import flet as ft

from ollama_sentinel.alarms import evaluate_alarms, format_expires, gpu_pct
from ollama_sentinel.catalog import search_models, typeahead
from ollama_sentinel.config import AppConfig, selected_servers
from ollama_sentinel.instance import InstanceLock
from ollama_sentinel.inventory import build_inventory, inventory_summary
from ollama_sentinel.poll import poll_all
from ollama_sentinel.proc_vram import ProcessVramCollector
from ollama_sentinel.pull import pull_model
from ollama_sentinel.smi import query_gpus
from ollama_sentinel.state import load_state, save_state
from ollama_sentinel.telemetry import format_gpu_line, format_poll_age, is_stale


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

        async def quit_app_async() -> None:
            icon = tray_icon.get("icon")
            if icon is not None:
                icon.stop()
            if proc_collector:
                proc_collector.stop()
            release_lock()
            page.window.prevent_close = False
            page.window.destroy()

        def request_show_window() -> None:
            page.run_task(show_window_async)

        def request_hide_window() -> None:
            page.run_task(hide_window_async)

        def request_quit_app() -> None:
            page.run_task(quit_app_async)

        if tray:
            from ollama_sentinel.tray import start_tray

            page.window.prevent_close = True

            def on_window_event(e) -> None:
                if getattr(e, "data", None) == "close":
                    request_hide_window()

            page.window.on_event = on_window_event
            tray_icon["icon"] = start_tray(on_open=request_show_window, on_quit=request_quit_app)

            if start_hidden:
                page.window.visible = False
                page.window.skip_task_bar = True
        else:

            def on_window_event(e) -> None:
                if getattr(e, "data", None) == "close":
                    release_lock()

            page.window.on_event = on_window_event
            if start_hidden:
                page.window.visible = False
                page.window.skip_task_bar = True

        servers = selected_servers(cfg)
        proc_collector: ProcessVramCollector | None = None
        if cfg.proc_vram and any(s.local_gpu for s in servers):
            proc_collector = ProcessVramCollector(
                interval=cfg.proc_vram_interval,
                enabled=True,
                min_bytes=cfg.proc_vram_min_mb * 1024 * 1024,
            )
            proc_collector.start()

        server_names = [s.name for s in servers]
        current_server = ft.Dropdown(
            label="Server",
            value=server_names[0] if server_names else None,
            options=[ft.dropdown.Option(n) for n in server_names],
            width=200,
        )
        status_text = ft.Text("Loading…", size=14)
        poll_footer = ft.Text("", size=12, color=ft.Colors.GREY)
        gpu_col = ft.Column(spacing=4)
        proc_vram_col = ft.Column(spacing=2, scroll=ft.ScrollMode.AUTO, height=160)
        library_col = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)
        discover_col = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)
        search_field = ft.TextField(label="Search Hugging Face", expand=True)
        pull_status = ft.Text("")
        last_good: dict[str, dict[str, Any]] = {}

        def get_server_cfg():
            name = current_server.value
            for s in servers:
                if s.name == name:
                    return s
            return servers[0]

        def refresh(_=None) -> None:
            srv = get_server_cfg()
            target = [{"name": srv.name, "url": srv.url, "local_gpu": srv.local_gpu}]
            snap = poll_all(
                target,
                gpu_filter=cfg.gpu_filter,
                query_gpus_fn=query_gpus,
                last_snapshots=last_good or None,
            )[0]
            if snap.get("reachable"):
                last_good[srv.name] = snap

            state = load_state(cfg.state_file)
            active, new_state, _ = evaluate_alarms(snap, state, cfg.thresholds)
            save_state(cfg.state_file, new_state)

            icon = tray_icon.get("icon")
            if icon is not None:
                from ollama_sentinel.tray import set_tray_color

                try:
                    set_tray_color(icon, active)
                except Exception:
                    pass

            now = time.time()
            polled_ts = snap.get("polled_at_ts")
            stale_poll = bool(
                snap.get("stale")
                or (polled_ts is not None and is_stale(polled_ts, cfg.poll_interval, now))
            )
            if polled_ts is not None:
                footer = format_poll_age(polled_ts, now)
                if stale_poll:
                    poll_footer.value = f"STALE · {footer}"
                    poll_footer.color = ft.Colors.ORANGE
                else:
                    poll_footer.value = f"Updated {footer}"
                    poll_footer.color = ft.Colors.GREY

            if not snap.get("reachable"):
                status_text.value = f"Unreachable: {snap.get('error')}"
                status_text.color = ft.Colors.RED
            elif active:
                status_text.value = "\n".join(a["message"] for a in active)
                status_text.color = ft.Colors.RED
            else:
                status_text.value = "OK — no alarms"
                status_text.color = ft.Colors.GREEN

            gpu_col.controls.clear()
            for gpu in snap.get("gpus") or []:
                gpu_col.controls.append(ft.Text(format_gpu_line(gpu), size=12))

            proc_vram_col.controls.clear()
            if proc_collector:
                pv = proc_collector.get_snapshot()
                pv_ts = pv.get("polled_at_ts")
                if pv_ts is not None:
                    pv_age = format_poll_age(pv_ts, now)
                    pv_stale = pv.get("stale") or is_stale(pv_ts, cfg.proc_vram_interval, now)
                    label = f"Process VRAM · {'STALE · ' if pv_stale else ''}{pv_age}"
                    proc_vram_col.controls.append(
                        ft.Text(label, size=12, color=ft.Colors.ORANGE if pv_stale else ft.Colors.GREY)
                    )
                elif pv.get("error"):
                    proc_vram_col.controls.append(ft.Text(f"Process VRAM error: {pv['error']}", size=12))
                rows = pv.get("rows") or []
                if not rows and not pv.get("error"):
                    proc_vram_col.controls.append(ft.Text("(no processes above threshold)", size=12))
                for row in rows:
                    local_gb = row.get("bytes", 0) / 1e9
                    non_local = row.get("non_local_bytes")
                    if non_local is not None:
                        line = (
                            f"{row.get('pid')} {row.get('name')}: "
                            f"{local_gb:.2f} GB local, {non_local / 1e9:.2f} GB non-local"
                        )
                    else:
                        line = f"{row.get('pid')} {row.get('name')}: {local_gb:.2f} GB"
                    proc_vram_col.controls.append(ft.Text(line, size=12))

            library_col.controls.clear()
            inv = build_inventory(snap)
            free_gb, free_pct = _free_vram_summary(snap.get("gpus"))
            library_col.controls.append(
                ft.Text(
                    inventory_summary(inv, free_vram_gb=free_gb, free_vram_pct=free_pct),
                    weight=ft.FontWeight.BOLD,
                )
            )
            for row in inv:
                fit = ""
                if row.get("loaded"):
                    fit = f"{row.get('gpu_pct')}% GPU"
                elif row.get("would_spill"):
                    fit = "would spill"
                elif row.get("would_spill") is False:
                    fit = "fits"
                library_col.controls.append(
                    ft.ListTile(
                        title=ft.Text(row["name"]),
                        subtitle=ft.Text(
                            f"{row['size_gb']:.1f} GB · {'loaded' if row['loaded'] else 'idle'} · {fit}"
                        ),
                    )
                )
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

        status_page = ft.Column(
            [
                current_server,
                status_text,
                ft.Text("GPU", weight=ft.FontWeight.BOLD),
                gpu_col,
                ft.Text("Process VRAM", weight=ft.FontWeight.BOLD),
                proc_vram_col,
                poll_footer,
                ft.ElevatedButton("Refresh", on_click=refresh),
            ],
            expand=True,
            scroll=ft.ScrollMode.AUTO,
        )
        library_page = ft.Column([library_col, ft.ElevatedButton("Refresh", on_click=refresh)], expand=True)
        discover_page = ft.Column(
            [search_field, ft.ElevatedButton("Search", on_click=do_search), pull_status, discover_col],
            expand=True,
        )
        pages = [status_page, library_page, discover_page]

        nav = ft.NavigationRail(
            selected_index=0,
            destinations=[
                ft.NavigationRailDestination(icon=ft.Icons.MONITOR_HEART, label="Status"),
                ft.NavigationRailDestination(icon=ft.Icons.LIBRARY_BOOKS, label="Library"),
                ft.NavigationRailDestination(icon=ft.Icons.SEARCH, label="Discover"),
            ],
        )
        content_area = ft.Container(content=status_page, expand=True)

        def on_nav(e):
            content_area.content = pages[int(e.control.selected_index)]
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

        def show_request_loop():
            while True:
                time.sleep(1)
                try:
                    if instance_lock and instance_lock.consume_show_request():
                        request_show_window()
                except Exception:
                    pass

        threading.Thread(target=poll_loop, daemon=True).start()
        if instance_lock is not None:
            threading.Thread(target=show_request_loop, daemon=True).start()

    ft.app(target=app)
