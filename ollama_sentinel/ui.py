"""Flet Material 3 window."""

from __future__ import annotations

import threading
import time
from typing import Any

import flet as ft

from ollama_sentinel.alarms import evaluate_alarms, format_expires, gpu_pct
from ollama_sentinel.catalog import search_models, typeahead
from ollama_sentinel.config import AppConfig, selected_servers
from ollama_sentinel.inventory import build_inventory, inventory_summary
from ollama_sentinel.poll import poll_all
from ollama_sentinel.pull import pull_model
from ollama_sentinel.smi import query_gpus
from ollama_sentinel.state import load_state, save_state


def run_gui(cfg: AppConfig, *, tray: bool = False) -> None:
    if tray:
        from ollama_sentinel.tray import start_tray

        start_tray(on_open=lambda: None)

    def app(page: ft.Page) -> None:
        page.title = "ollama-sentinel"
        page.theme_mode = ft.ThemeMode.DARK
        page.window.width = 960
        page.window.height = 640

        servers = selected_servers(cfg)
        server_names = [s.name for s in servers]
        current_server = ft.Dropdown(
            label="Server",
            value=server_names[0] if server_names else None,
            options=[ft.dropdown.Option(n) for n in server_names],
            width=200,
        )
        status_text = ft.Text("Loading…", size=14)
        library_col = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)
        discover_col = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True)
        search_field = ft.TextField(label="Search Hugging Face", expand=True)
        pull_status = ft.Text("")

        def get_server_cfg():
            name = current_server.value
            for s in servers:
                if s.name == name:
                    return s
            return servers[0]

        def refresh(_=None) -> None:
            srv = get_server_cfg()
            target = [{"name": srv.name, "url": srv.url, "local_gpu": srv.local_gpu}]
            snap = poll_all(target, gpu_filter=cfg.gpu_filter, query_gpus_fn=query_gpus)[0]
            state = load_state(cfg.state_file)
            active, new_state, _ = evaluate_alarms(snap, state, cfg.thresholds)
            save_state(cfg.state_file, new_state)

            if not snap.get("reachable"):
                status_text.value = f"Unreachable: {snap.get('error')}"
                status_text.color = ft.Colors.RED
            elif active:
                status_text.value = "\n".join(a["message"] for a in active)
                status_text.color = ft.Colors.RED
            else:
                status_text.value = "OK — no alarms"
                status_text.color = ft.Colors.GREEN

            library_col.controls.clear()
            inv = build_inventory(snap)
            library_col.controls.append(ft.Text(inventory_summary(inv), weight=ft.FontWeight.BOLD))
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

        status_page = ft.Column([current_server, status_text, ft.ElevatedButton("Refresh", on_click=refresh)], expand=True)
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

        threading.Thread(target=poll_loop, daemon=True).start()

    ft.app(target=app)
