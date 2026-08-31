"""Flet widget builders for the GUI."""

from __future__ import annotations

from typing import Any, Callable

import flet as ft

from ollama_sentinel.alarms import format_expires, gpu_pct
from ollama_sentinel.telemetry import format_bytes_gb, format_field, format_throttle

PALETTE = {
    "ok": ft.Colors.GREEN_400,
    "warn": ft.Colors.ORANGE_400,
    "alarm": ft.Colors.RED_400,
    "muted": ft.Colors.GREY_500,
    "stale": ft.Colors.ORANGE_300,
    "surface": ft.Colors.SURFACE_CONTAINER_HIGHEST,
}


def alarm_state(reachable: bool, active: list[dict[str, Any]]) -> tuple[str, str, str | None]:
    """Return (title, body, palette_key)."""
    if not reachable:
        return "Unreachable", "", "alarm"
    if not active:
        return "OK", "No alarms", "ok"
    paging = any(a.get("type") == "paging" for a in active)
    key = "alarm" if paging else "warn"
    body = "\n".join(a["message"] for a in active)
    return "Alarms", body, key


def fit_label(row: dict[str, Any]) -> tuple[str, str | None]:
    if row.get("loaded"):
        return f"{row.get('gpu_pct', '?')}% GPU", PALETTE["ok"]
    if row.get("would_spill"):
        return "would spill", PALETTE["warn"]
    if row.get("would_spill") is False:
        return "fits", PALETTE["muted"]
    return "—", None


def _metric_table(rows: list[tuple[str, str]], *, footer: str | None = None) -> ft.DataTable:
    data_rows = [
        ft.DataRow(
            cells=[
                ft.DataCell(ft.Text(label, size=12, color=PALETTE["muted"])),
                ft.DataCell(ft.Text(value, size=12)),
            ]
        )
        for label, value in rows
    ]
    if footer:
        data_rows.append(
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text("Note", size=12, color=PALETTE["stale"])),
                    ft.DataCell(ft.Text(footer, size=12, color=PALETTE["stale"])),
                ]
            )
        )
    return ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Metric", size=12, weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("Value", size=12, weight=ft.FontWeight.BOLD)),
        ],
        rows=data_rows,
        heading_row_height=32,
        data_row_min_height=28,
        column_spacing=16,
        horizontal_margin=8,
    )


def _cell(text: str, *, color: str | None = None, weight: ft.FontWeight | None = None) -> ft.DataCell:
    return ft.DataCell(ft.Text(text, size=12, color=color, weight=weight))


def section_card(
    title: str,
    content: ft.Control,
    *,
    subtitle: str | None = None,
) -> ft.Card:
    header: list[ft.Control] = [ft.Text(title, weight=ft.FontWeight.BOLD, size=14)]
    if subtitle:
        header.append(ft.Text(subtitle, size=11, color=PALETTE["muted"]))
    return ft.Card(
        content=ft.Container(
            content=ft.Column(
                [
                    ft.Column(header, spacing=2),
                    content,
                ],
                spacing=8,
            ),
            padding=12,
        ),
        elevation=1,
    )


def alarm_banner(reachable: bool, active: list[dict[str, Any]], *, error: str | None = None) -> ft.Container:
    if not reachable:
        title, body, key = "Unreachable", error or "Ollama not reachable", "alarm"
    else:
        title, body, key = alarm_state(reachable, active)
    bg = {
        "ok": ft.Colors.GREEN_900,
        "warn": ft.Colors.ORANGE_900,
        "alarm": ft.Colors.RED_900,
    }[key]
    fg = PALETTE[key]
    lines = [ft.Text(title, size=16, weight=ft.FontWeight.BOLD, color=fg)]
    if body:
        lines.append(ft.Text(body, size=12, color=ft.Colors.WHITE70))
    return ft.Container(
        content=ft.Column(lines, spacing=4),
        padding=12,
        border_radius=8,
        bgcolor=bg,
    )


def gpu_table(gpu: dict[str, Any]) -> ft.Control:
    name = gpu.get("name") or f"GPU {gpu.get('index', 0)}"
    metrics: list[tuple[str, str]] = [
        ("Used", format_bytes_gb(gpu.get("memory_used"))),
        ("Free", f"{format_bytes_gb(gpu.get('memory_free'))} ({format_field(gpu.get('memory_free_pct'), '%')})"),
        ("Total", format_bytes_gb(gpu.get("memory_total"))),
        ("Reserved", format_bytes_gb(gpu.get("memory_reserved"))),
        ("Temp", format_field(gpu.get("temperature"), "°C")),
        ("Fan", format_field(gpu.get("fan_speed"), "%")),
        ("GPU util", format_field(gpu.get("utilization"), "%")),
        ("Mem util", format_field(gpu.get("memory_utilization"), "%")),
        ("Power", f"{format_field(gpu.get('power_draw'), ' W')} / {format_field(gpu.get('power_limit'), ' W')}"),
        ("Pstate", format_field(gpu.get("pstate"))),
        (
            "Clocks",
            f"{format_field(gpu.get('clock_sm'), ' MHz')} / {format_field(gpu.get('clock_mem'), ' MHz')}",
        ),
    ]
    throttle = format_throttle(gpu)
    return section_card(name, _metric_table(metrics, footer=throttle))


def process_vram_table(
    rows: list[dict[str, Any]],
    *,
    stale: bool = False,
    age_text: str | None = None,
    error: str | None = None,
) -> ft.Control:
    subtitle = age_text or ""
    if stale and subtitle:
        subtitle = f"STALE · {subtitle}"
    if error:
        subtitle = error

    if not rows and not error:
        body: ft.Control = ft.Text("(no processes above threshold)", size=12, color=PALETTE["muted"])
    else:
        has_non_local = any(r.get("non_local_bytes") is not None for r in rows)
        columns = [
            ft.DataColumn(ft.Text("PID", size=12, weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("Process", size=12, weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("Local", size=12, weight=ft.FontWeight.BOLD)),
        ]
        if has_non_local:
            columns.append(ft.DataColumn(ft.Text("Non-local", size=12, weight=ft.FontWeight.BOLD)))

        data_rows: list[ft.DataRow] = []
        for row in rows:
            local_gb = row.get("bytes", 0) / 1e9
            cells = [
                _cell(str(row.get("pid", "?"))),
                _cell(str(row.get("name", "?"))),
                _cell(f"{local_gb:.2f} GB"),
            ]
            if has_non_local:
                non_local = row.get("non_local_bytes") or 0
                color = PALETTE["warn"] if non_local > 0 else None
                cells.append(_cell(f"{non_local / 1e9:.2f} GB", color=color))
            data_rows.append(ft.DataRow(cells=cells))

        body = ft.DataTable(
            columns=columns,
            rows=data_rows,
            heading_row_height=32,
            data_row_min_height=28,
            column_spacing=12,
            horizontal_margin=8,
        )

    return section_card("Process VRAM", body, subtitle=subtitle or None)


def _action_cell(control: ft.Control) -> ft.DataCell:
    return ft.DataCell(control)


def loaded_models_table(
    models: list[dict[str, Any]],
    *,
    server_url: str | None = None,
    on_unload: Callable[[str], None] | None = None,
) -> ft.Control | None:
    if not models:
        return None
    data_rows: list[ft.DataRow] = []
    for model in models:
        name = model.get("name") or "?"
        size = model.get("size") or 0
        sv = model.get("size_vram") or 0
        pct = gpu_pct(size, sv)
        spill = sv < size
        cells = [
            _cell(name, weight=ft.FontWeight.W_500),
            _cell(f"{sv/1e9:.1f} GB"),
            _cell(
                f"{100-pct}% CPU / {pct}% GPU",
                color=PALETTE["warn"] if spill else PALETTE["ok"],
            ),
            _cell(format_expires(model.get("expires_at"), server_url=server_url)),
        ]
        if on_unload:
            cells.append(
                _action_cell(
                    ft.OutlinedButton(
                        "Unload",
                        on_click=lambda _e, m=name: on_unload(m),
                        style=ft.ButtonStyle(padding=ft.Padding(left=12, right=12, top=4, bottom=4)),
                    )
                )
            )
        data_rows.append(ft.DataRow(cells=cells))
    columns = [
        ft.DataColumn(ft.Text("Model", size=12, weight=ft.FontWeight.BOLD)),
        ft.DataColumn(ft.Text("VRAM", size=12, weight=ft.FontWeight.BOLD)),
        ft.DataColumn(ft.Text("Split", size=12, weight=ft.FontWeight.BOLD)),
        ft.DataColumn(ft.Text("Expires", size=12, weight=ft.FontWeight.BOLD)),
    ]
    if on_unload:
        columns.append(ft.DataColumn(ft.Text("", size=12)))
    table = ft.DataTable(
        columns=columns,
        rows=data_rows,
        heading_row_height=32,
        data_row_min_height=40,
        column_spacing=12,
        horizontal_margin=8,
    )
    return section_card("Loaded models", table)


def library_table(
    rows: list[dict[str, Any]],
    *,
    on_unload: Callable[[str], None] | None = None,
) -> ft.Control:
    sorted_rows = sorted(rows, key=lambda r: (not r.get("loaded"), r.get("name") or ""))
    data_rows: list[ft.DataRow] = []
    for row in sorted_rows:
        fit_text, fit_color = fit_label(row)
        state = "loaded" if row.get("loaded") else "idle"
        state_color = PALETTE["ok"] if row.get("loaded") else PALETTE["muted"]
        cells = [
            _cell(row["name"]),
            _cell(f"{row['size_gb']:.1f} GB"),
            _cell(state, color=state_color),
            _cell(fit_text, color=fit_color),
        ]
        if on_unload and row.get("loaded"):
            cells.append(
                _action_cell(
                    ft.OutlinedButton(
                        "Unload",
                        on_click=lambda _e, m=row["name"]: on_unload(m),
                        style=ft.ButtonStyle(padding=ft.Padding(left=12, right=12, top=4, bottom=4)),
                    )
                )
            )
        elif on_unload:
            cells.append(_cell(""))
        data_rows.append(ft.DataRow(cells=cells))
    columns = [
        ft.DataColumn(ft.Text("Name", size=12, weight=ft.FontWeight.BOLD)),
        ft.DataColumn(ft.Text("Size", size=12, weight=ft.FontWeight.BOLD)),
        ft.DataColumn(ft.Text("State", size=12, weight=ft.FontWeight.BOLD)),
        ft.DataColumn(ft.Text("Fit", size=12, weight=ft.FontWeight.BOLD)),
    ]
    if on_unload:
        columns.append(ft.DataColumn(ft.Text("", size=12)))
    table = ft.DataTable(
        columns=columns,
        rows=data_rows,
        heading_row_height=32,
        data_row_min_height=40,
        column_spacing=12,
        horizontal_margin=8,
    )
    return table
