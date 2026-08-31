"""Time-series charts from MetricsStore (Flet Canvas — no extra deps)."""

from __future__ import annotations

from dataclasses import dataclass

import flet as ft
import flet.canvas as cv
from flet.canvas import Path

from ollama_sentinel.metrics import MetricField, MetricsStore
from ollama_sentinel.ui_widgets import PALETTE, section_card

CHART_WIDTH = 640
CHART_HEIGHT = 88
PAD_LEFT = 36
PAD_RIGHT = 12
PAD_TOP = 6
PAD_BOTTOM = 16


@dataclass(frozen=True)
class ChartSpec:
    field: MetricField
    title: str
    unit: str
    color: str
    ymin: float | None = None
    ymax: float | None = None
    note: str | None = None


CHART_SPECS: tuple[ChartSpec, ...] = (
    ChartSpec("mem_used_pct", "VRAM used", "%", ft.Colors.CYAN_300, 0, 100),
    ChartSpec("util", "GPU utilization", "%", ft.Colors.GREEN_400, 0, 100),
    ChartSpec("power_draw", "Power draw", "W", ft.Colors.ORANGE_400),
    ChartSpec("loaded_vram_gb", "Loaded models", "GB", ft.Colors.PINK_300, 0, 24),
    ChartSpec(
        "llama_util",
        "llama-server util",
        "%",
        ft.Colors.AMBER_400,
        0,
        100,
        note="30s cadence",
    ),
)


def _format_axis(val: float, unit: str) -> str:
    if unit == "%":
        return f"{val:.0f}"
    if unit == "W":
        return f"{val:.0f}"
    if unit == "GB":
        return f"{val:.1f}"
    return f"{val:.1f}"


def format_metric_value(val: float, unit: str) -> str:
    if unit == "%":
        return f"{val:.0f}%"
    if unit == "W":
        return f"{val:.0f} W"
    if unit == "GB":
        return f"{val:.1f} GB"
    return f"{val:.1f}"


def series_plot_points(
    series: list[tuple[float, float]],
    *,
    width: float = CHART_WIDTH,
    height: float = CHART_HEIGHT,
    pad_left: float = PAD_LEFT,
    pad_right: float = PAD_RIGHT,
    pad_top: float = PAD_TOP,
    pad_bottom: float = PAD_BOTTOM,
    ymin: float | None = None,
    ymax: float | None = None,
) -> tuple[list[tuple[float, float]], float, float]:
    """Map (timestamp, value) series to canvas coordinates. Newest last."""
    if not series:
        return [], 0.0, 100.0

    values = [v for _, v in series]
    t_min = series[0][0]
    t_max = series[-1][0]
    if t_max <= t_min:
        t_max = t_min + 1.0

    y_lo = ymin if ymin is not None else min(values)
    y_hi = ymax if ymax is not None else max(values)
    if y_hi <= y_lo:
        y_hi = y_lo + 1.0

    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    pts: list[tuple[float, float]] = []
    for ts, val in series:
        x = pad_left + (ts - t_min) / (t_max - t_min) * plot_w
        y = pad_top + plot_h - (val - y_lo) / (y_hi - y_lo) * plot_h
        pts.append((x, y))
    return pts, y_lo, y_hi


def _build_canvas_shapes(
    series: list[tuple[float, float]],
    *,
    spec: ChartSpec,
) -> list:
    pts, y_lo, y_hi = series_plot_points(series, ymin=spec.ymin, ymax=spec.ymax)
    plot_left = PAD_LEFT
    plot_top = PAD_TOP
    plot_w = CHART_WIDTH - PAD_LEFT - PAD_RIGHT
    plot_h = CHART_HEIGHT - PAD_TOP - PAD_BOTTOM

    shapes: list = [
        cv.Rect(
            plot_left,
            plot_top,
            plot_w,
            plot_h,
            border_radius=4,
            paint=ft.Paint(
                color=ft.Colors.with_opacity(0.12, ft.Colors.GREY_800),
                style=ft.PaintingStyle.FILL,
            ),
        ),
    ]

    grid_paint = ft.Paint(
        color=ft.Colors.with_opacity(0.25, ft.Colors.GREY_500),
        stroke_width=1,
        style=ft.PaintingStyle.STROKE,
    )
    for frac in (0.25, 0.5, 0.75):
        y = plot_top + plot_h * (1 - frac)
        shapes.append(cv.Line(plot_left, y, plot_left + plot_w, y, paint=grid_paint))

    shapes.append(
        cv.Text(
            x=4,
            y=plot_top,
            value=_format_axis(y_hi, spec.unit),
            style=ft.TextStyle(size=10, color=PALETTE["muted"]),
        )
    )
    shapes.append(
        cv.Text(
            x=4,
            y=plot_top + plot_h - 10,
            value=_format_axis(y_lo, spec.unit),
            style=ft.TextStyle(size=10, color=PALETTE["muted"]),
        )
    )

    if len(pts) < 2:
        return shapes

    area_elements: list = [Path.MoveTo(x=pts[0][0], y=pts[0][1])]
    for x, y in pts[1:]:
        area_elements.append(Path.LineTo(x=x, y=y))
    bottom = plot_top + plot_h
    area_elements.append(Path.LineTo(x=pts[-1][0], y=bottom))
    area_elements.append(Path.LineTo(x=pts[0][0], y=bottom))
    area_elements.append(Path.Close())
    shapes.append(
        Path(
            elements=area_elements,
            paint=ft.Paint(
                color=ft.Colors.with_opacity(0.22, spec.color),
                style=ft.PaintingStyle.FILL,
            ),
        )
    )

    line_elements: list = [Path.MoveTo(x=pts[0][0], y=pts[0][1])]
    for x, y in pts[1:]:
        line_elements.append(Path.LineTo(x=x, y=y))
    shapes.append(
        Path(
            elements=line_elements,
            paint=ft.Paint(
                color=spec.color,
                stroke_width=2,
                style=ft.PaintingStyle.STROKE,
                stroke_cap=ft.StrokeCap.ROUND,
                stroke_join=ft.StrokeJoin.ROUND,
            ),
        )
    )

    last_x, last_y = pts[-1]
    shapes.append(
        cv.Circle(
            last_x,
            last_y,
            3,
            paint=ft.Paint(color=spec.color, style=ft.PaintingStyle.FILL),
        )
    )
    return shapes


def metric_chart_card(spec: ChartSpec, series: list[tuple[float, float]]) -> ft.Control:
    current = series[-1][1] if series else None
    sample_n = len(series)

    header_bits: list[ft.Control] = [
        ft.Text(spec.title, size=13, weight=ft.FontWeight.W_500),
    ]
    if current is not None:
        header_bits.append(
            ft.Text(
                format_metric_value(current, spec.unit),
                size=18,
                weight=ft.FontWeight.BOLD,
                color=spec.color,
            )
        )
    else:
        header_bits.append(ft.Text("—", size=18, color=PALETTE["muted"]))

    meta = f"{sample_n} sample{'s' if sample_n != 1 else ''}"
    if spec.note:
        meta = f"{meta} · {spec.note}"
    header_bits.append(ft.Text(meta, size=10, color=PALETTE["muted"]))

    if sample_n < 2:
        body: ft.Control = ft.Container(
            content=ft.Text(
                "Collecting… (need a few polls)",
                size=12,
                color=PALETTE["muted"],
            ),
            height=CHART_HEIGHT,
            alignment=ft.Alignment.CENTER_LEFT,
        )
    else:
        body = cv.Canvas(
            width=CHART_WIDTH,
            height=CHART_HEIGHT,
            shapes=_build_canvas_shapes(series, spec=spec),
        )

    return ft.Card(
        content=ft.Container(
            content=ft.Row(
                [
                    ft.Column(header_bits, spacing=2, width=130),
                    ft.Container(content=body, expand=True),
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=8,
            ),
            padding=12,
        ),
        elevation=1,
    )


def metrics_charts_panel(
    store: MetricsStore | None,
    *,
    window_s: float,
    server: str | None = None,
) -> ft.Control:
    if store is None:
        return section_card(
            "Metrics",
            ft.Text("Metrics disabled (set METRICS=1 in .env)", size=12, color=PALETTE["muted"]),
        )

    window_label = {300: "5 min", 900: "15 min", 3600: "1 hour"}.get(int(window_s), f"{int(window_s)}s")
    cards: list[ft.Control] = [
        ft.Text(
            f"History from existing polls — no extra GPU queries · {window_label} window",
            size=11,
            color=PALETTE["muted"],
        ),
    ]
    for spec in CHART_SPECS:
        series = store.series(spec.field, window_s=window_s, server=server)
        cards.append(metric_chart_card(spec, series))

    snap = store.snapshot(window_s=window_s)
    total = sum(snap.get("counts", {}).values())
    cards.append(
        ft.Text(
            f"Buffer: {total} points retained (up to {int(store.retention_s // 60)} min)",
            size=10,
            color=PALETTE["muted"],
        )
    )
    return ft.Column(cards, spacing=8)
