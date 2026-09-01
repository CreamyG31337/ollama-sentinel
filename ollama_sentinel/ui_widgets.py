"""Flet widget builders for the GUI."""

from __future__ import annotations

from typing import Any, Callable

import flet as ft

from ollama_sentinel.alarms import format_expires, gpu_pct
from ollama_sentinel.activity import ServerActivity, model_detail_line
from ollama_sentinel.catalog import format_count, summarize_list_item
from ollama_sentinel.inventory import inventory_detail_line
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


def activity_card(activity: ServerActivity | dict[str, Any] | None) -> ft.Control | None:
    if activity is None:
        return None
    if isinstance(activity, dict):
        phase = activity.get("phase", "idle")
        summary = activity.get("summary", "—")
        stale = bool(activity.get("stale"))
        runners = activity.get("runners") or []
        recent = activity.get("recent_requests") or []
        last_req = activity.get("last_request")
    else:
        phase = activity.phase
        summary = activity.summary
        stale = activity.stale
        runners = [r.to_dict() for r in activity.runners]
        recent = [r.to_dict() for r in activity.recent_requests]
        last_req = activity.last_request.to_dict() if activity.last_request else None

    color = PALETTE["ok"] if phase == "idle" else PALETTE["warn"]
    if phase in ("prompt", "generating", "embed"):
        color = PALETTE["ok"] if phase == "embed" else ft.Colors.CYAN_300

    lines: list[ft.Control] = [
        ft.Text(summary, size=13, weight=ft.FontWeight.W_500, color=color),
    ]
    if stale:
        lines.append(ft.Text("Log signal stale; using GPU util", size=11, color=PALETTE["stale"]))

    if runners:
        runner_bits = []
        for r in runners[:4]:
            util = r.get("engine_3d_pct") or 0
            tag = "busy" if r.get("busy") else "idle"
            runner_bits.append(
                f"pid {r.get('pid')} {r.get('vram_bytes', 0) / 1e9:.1f} GB · {util:.0f}% util · {tag}"
            )
        lines.append(ft.Text(" · ".join(runner_bits), size=11, color=PALETTE["muted"]))

    if last_req:
        dur = last_req.get("duration_s")
        dur_s = f" {dur * 1000:.0f}ms" if dur is not None and dur < 1 else (f" {dur:.2f}s" if dur else "")
        lines.append(
            ft.Text(
                f"Last {last_req.get('method')} {last_req.get('path')} "
                f"({last_req.get('client')}{dur_s})",
                size=11,
                color=PALETTE["muted"],
            )
        )
    elif recent:
        req = recent[-1]
        lines.append(
            ft.Text(
                f"Recent {req.get('method')} {req.get('path')} ({req.get('client')})",
                size=11,
                color=PALETTE["muted"],
            )
        )

    return section_card("Activity", ft.Column(lines, spacing=4))


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
            ft.DataColumn(ft.Text("GPU util", size=12, weight=ft.FontWeight.BOLD)),
        ]
        if has_non_local:
            columns.append(ft.DataColumn(ft.Text("Non-local", size=12, weight=ft.FontWeight.BOLD)))

        data_rows: list[ft.DataRow] = []
        for row in rows:
            local_gb = row.get("bytes", 0) / 1e9
            util = row.get("engine_3d_pct")
            util_text = f"{util:.0f}%" if util is not None else "—"
            util_color = PALETTE["ok"] if (util or 0) >= 5 else PALETTE["muted"]
            cells = [
                _cell(str(row.get("pid", "?"))),
                _cell(str(row.get("name", "?"))),
                _cell(f"{local_gb:.2f} GB"),
                _cell(util_text, color=util_color),
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
            _cell(model_detail_line(model)),
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
        ft.DataColumn(ft.Text("Details", size=12, weight=ft.FontWeight.BOLD)),
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
            _cell(inventory_detail_line(row)),
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
        ft.DataColumn(ft.Text("Details", size=12, weight=ft.FontWeight.BOLD)),
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


def discover_result_tile(
    item: dict[str, Any],
    *,
    detail: dict[str, Any] | None = None,
    detail_error: str | None = None,
    detail_loading: bool = False,
    expanded: bool = False,
    on_pull: Callable[[str], None],
    on_open_hf: Callable[[], None],
    on_expand_change: Callable[[Any], None] | None = None,
) -> ft.Control:
    subtitle = item.get("summary") or summarize_list_item(item)
    detail_controls = _discover_detail_controls(
        detail,
        detail_error=detail_error,
        detail_loading=detail_loading,
        on_pull=on_pull,
        on_open_hf=on_open_hf,
    )

    return ft.Card(
        content=ft.Container(
            content=ft.ExpansionTile(
                title=ft.Text(item.get("id") or "?", weight=ft.FontWeight.W_500, size=13),
                subtitle=ft.Text(subtitle, size=11, color=PALETTE["muted"]),
                expanded=expanded,
                maintain_state=True,
                on_change=on_expand_change,
                controls=detail_controls,
                controls_padding=ft.Padding(left=12, right=12, bottom=12),
                trailing=ft.OutlinedButton(
                    "Install",
                    on_click=lambda e, name=item.get("pull_name", ""): on_pull(name),
                    style=ft.ButtonStyle(padding=ft.Padding(left=10, right=10, top=4, bottom=4)),
                ),
            ),
            padding=ft.Padding(left=4, right=4, top=4, bottom=4),
        ),
        elevation=1,
    )


def _discover_detail_controls(
    detail: dict[str, Any] | None,
    *,
    detail_error: str | None,
    detail_loading: bool,
    on_pull: Callable[[str], None],
    on_open_hf: Callable[[], None],
) -> list[ft.Control]:
    if detail_loading:
        return [
            ft.Row(
                [
                    ft.ProgressRing(width=16, height=16, stroke_width=2),
                    ft.Text("Loading model details and README…", size=12, color=PALETTE["muted"]),
                ],
                spacing=8,
            )
        ]
    if detail_error:
        return [ft.Text(detail_error, size=12, color=PALETTE["alarm"])]
    if detail is None:
        return [ft.Text("Expand for README, GGUF quants, context, and license.", size=12, color=PALETTE["muted"])]

    controls: list[ft.Control] = []
    facts: list[str] = []
    if detail.get("architecture"):
        facts.append(f"arch {detail['architecture']}")
    ctx = detail.get("context_length")
    if ctx:
        facts.append(f"context {int(ctx):,}")
    if detail.get("license"):
        facts.append(str(detail["license"]))
    if detail.get("base_model"):
        facts.append(f"base {detail['base_model']}")
    if facts:
        controls.append(ft.Text(" · ".join(facts), size=12))

    meta: list[str] = []
    if detail.get("downloads") is not None:
        meta.append(f"{format_count(detail.get('downloads'))} downloads")
    if detail.get("likes"):
        meta.append(f"{format_count(detail.get('likes'))} likes")
    if detail.get("last_modified"):
        meta.append(f"updated {str(detail['last_modified'])[:10]}")
    if meta:
        controls.append(ft.Text(" · ".join(meta), size=11, color=PALETTE["muted"]))

    readme = detail.get("readme")
    if readme:
        controls.append(ft.Text("README", size=12, weight=ft.FontWeight.W_500))
        controls.append(
            ft.Container(
                content=ft.Column(
                    [
                        ft.Markdown(
                            readme,
                            selectable=True,
                            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                        )
                    ],
                    scroll=ft.ScrollMode.AUTO,
                ),
                height=260,
                border=ft.border.all(1, ft.Colors.with_opacity(0.25, ft.Colors.GREY_500)),
                border_radius=6,
                padding=8,
            )
        )
    elif "readme" in detail:
        controls.append(
            ft.Text(
                "No README.md in repo (or not accessible without HF login).",
                size=11,
                color=PALETTE["muted"],
            )
        )

    if detail.get("gated"):
        controls.append(
            ft.Text("Gated on Hugging Face — pull may require HF_TOKEN.", size=11, color=PALETTE["warn"])
        )

    sample_bytes = detail.get("gguf_total_bytes")
    if sample_bytes:
        controls.append(
            ft.Text(f"Reference size ~{format_bytes_gb(int(sample_bytes))}", size=11, color=PALETTE["muted"])
        )

    variants = detail.get("variants") or []
    if variants:
        controls.append(ft.Text("Quant files", size=12, weight=ft.FontWeight.W_500))
        for variant in variants[:14]:
            pull_name = variant.get("pull_name") or ""
            controls.append(
                ft.Row(
                    [
                        ft.Text(variant.get("filename") or "?", size=11, expand=True),
                        ft.OutlinedButton(
                            "Install",
                            on_click=lambda e, name=pull_name: on_pull(name),
                            style=ft.ButtonStyle(
                                padding=ft.Padding(left=10, right=10, top=2, bottom=2),
                            ),
                        ),
                    ],
                    spacing=8,
                )
            )
        if len(variants) > 14:
            controls.append(
                ft.Text(
                    f"+ {len(variants) - 14} more files on Hugging Face",
                    size=11,
                    color=PALETTE["muted"],
                )
            )
    else:
        controls.append(
            ft.Text(
                "No .gguf filenames listed — use default Install or open on Hugging Face.",
                size=11,
                color=PALETTE["muted"],
            )
        )

    default_pull = detail.get("pull_name")
    if default_pull:
        controls.append(
            ft.Row(
                [
                    ft.Text(f"Default: {default_pull}", size=11, color=PALETTE["muted"], expand=True),
                    ft.TextButton("Install default", on_click=lambda e, name=default_pull: on_pull(name)),
                ],
                spacing=8,
            )
        )

    controls.append(ft.TextButton("Open on Hugging Face", icon=ft.Icons.OPEN_IN_NEW, on_click=lambda _: on_open_hf()))
    return controls
