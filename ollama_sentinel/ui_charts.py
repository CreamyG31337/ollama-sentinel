"""Time-series charts from MetricsStore (Flet Canvas — no extra deps).

Canvases are persistent: live updates mutate ``shapes`` in place and call
``canvas.update()``. Replacing the whole control tree from a worker thread
left Charts frozen until a UI-thread click (range buttons).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import flet as ft
import flet.canvas as cv
from flet.canvas import Path

from ollama_sentinel.metrics import MetricField, MetricsStore
from ollama_sentinel.ui_widgets import PALETTE, section_card

CHART_HEIGHT = 88
PAD_LEFT = 36
PAD_RIGHT = 12
PAD_TOP = 6
PAD_BOTTOM = 16
# Fallback until the first on_resize reports the laid-out width.
DEFAULT_CHART_WIDTH = 640


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


WINDOW_LABELS: dict[int, str] = {
    300: "5 minutes",
    900: "15 minutes",
    3600: "1 hour",
}


def charts_subtitle(
    store: MetricsStore,
    *,
    window_s: float,
    server: str | None = None,
    poll_interval: float = 5.0,
) -> str:
    """User-facing context for the Charts tab — not implementation detail."""
    window = WINDOW_LABELS.get(int(window_s), f"{int(window_s)} seconds")
    target = server or "this server"
    refresh = f"updates every {int(poll_interval)}s" if poll_interval >= 1 else "live updates"

    series = store.series("mem_used_pct", window_s=window_s, server=server)
    if len(series) < 2:
        return f"VRAM, GPU load, and power for {target} · {refresh} · waiting for data"

    t0 = series[0][0]
    span = series[-1][0] - t0
    since = datetime.fromtimestamp(t0).astimezone().strftime("%H:%M:%S")
    if span < window_s * 0.85:
        return f"VRAM, GPU load, and power for {target} · {refresh} · since {since}"

    return f"VRAM, GPU load, and power for {target} · last {window} · {refresh}"


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
    width: float = DEFAULT_CHART_WIDTH,
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

    plot_w = max(1.0, width - pad_left - pad_right)
    plot_h = max(1.0, height - pad_top - pad_bottom)
    pts: list[tuple[float, float]] = []
    for ts, val in series:
        x = pad_left + (ts - t_min) / (t_max - t_min) * plot_w
        y = pad_top + plot_h - (val - y_lo) / (y_hi - y_lo) * plot_h
        pts.append((x, y))
    return pts, y_lo, y_hi


def build_canvas_shapes(
    series: list[tuple[float, float]],
    *,
    spec: ChartSpec,
    width: float = DEFAULT_CHART_WIDTH,
    height: float = CHART_HEIGHT,
) -> list:
    pts, y_lo, y_hi = series_plot_points(
        series, width=width, height=height, ymin=spec.ymin, ymax=spec.ymax
    )
    plot_left = PAD_LEFT
    plot_top = PAD_TOP
    plot_w = max(1.0, width - PAD_LEFT - PAD_RIGHT)
    plot_h = max(1.0, height - PAD_TOP - PAD_BOTTOM)

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
        shapes.append(
            cv.Text(
                x=plot_left + 8,
                y=plot_top + plot_h / 2 - 6,
                value="Not enough data yet",
                style=ft.TextStyle(size=12, color=PALETTE["muted"]),
            )
        )
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


# Back-compat alias used by older tests / callers.
_build_canvas_shapes = build_canvas_shapes
CHART_WIDTH = DEFAULT_CHART_WIDTH


@dataclass
class LiveChart:
    """One persistent metric card: mutate canvas.shapes instead of rebuilding cards."""

    spec: ChartSpec
    title: ft.Text
    value_text: ft.Text
    meta_text: ft.Text
    canvas: cv.Canvas
    series: list[tuple[float, float]] = field(default_factory=list)
    width: float = DEFAULT_CHART_WIDTH
    height: float = CHART_HEIGHT
    card: ft.Control | None = None

    def redraw(self, *, push: bool = True) -> None:
        # Mutate the existing shapes list in place — assigning a new list is
        # what left Canvas frozen until a UI-thread click in Flet 0.86.
        new_shapes = build_canvas_shapes(
            self.series, spec=self.spec, width=self.width, height=self.height
        )
        shapes = self.canvas.shapes
        shapes.clear()
        shapes.extend(new_shapes)
        if push:
            try:
                self.canvas.update()
            except Exception:
                pass

    def set_series(self, series: list[tuple[float, float]], *, push: bool = True) -> None:
        self.series = list(series)
        current = self.series[-1][1] if self.series else None
        if current is not None:
            self.value_text.value = format_metric_value(current, self.spec.unit)
            self.value_text.color = self.spec.color
        else:
            self.value_text.value = "—"
            self.value_text.color = PALETTE["muted"]
        meta = f"{len(self.series)} sample{'s' if len(self.series) != 1 else ''}"
        if self.spec.note:
            meta = f"{meta} · {self.spec.note}"
        self.meta_text.value = meta
        self.redraw(push=push)
        if push:
            try:
                self.value_text.update()
                self.meta_text.update()
            except Exception:
                pass

    def set_width(self, width: float, *, push: bool = True) -> None:
        if width <= 1:
            return
        if abs(width - self.width) < 1:
            return
        self.width = width
        self.redraw(push=push)


def make_live_chart(spec: ChartSpec) -> LiveChart:
    title = ft.Text(spec.title, size=13, weight=ft.FontWeight.W_500)
    value_text = ft.Text("—", size=18, weight=ft.FontWeight.BOLD, color=PALETTE["muted"])
    meta_text = ft.Text("0 samples", size=10, color=PALETTE["muted"])
    chart = LiveChart(
        spec=spec,
        title=title,
        value_text=value_text,
        meta_text=meta_text,
        canvas=cv.Canvas(
            expand=True,
            width=float("inf"),
            height=CHART_HEIGHT,
            shapes=build_canvas_shapes([], spec=spec),
            resize_interval=50,
        ),
    )

    def on_resize(e) -> None:
        w = getattr(e, "width", None)
        if w is not None:
            chart.set_width(float(w), push=True)

    chart.canvas.on_resize = on_resize
    chart.card = ft.Card(
        content=ft.Container(
            content=ft.Row(
                [
                    ft.Column([title, value_text, meta_text], spacing=2, width=130),
                    ft.Container(content=chart.canvas, expand=True, height=CHART_HEIGHT),
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=8,
            ),
            padding=12,
        ),
        elevation=1,
    )
    return chart


@dataclass
class LiveChartsPanel:
    """Persistent Charts tab body — refresh data without tearing down Canvases."""

    charts: list[LiveChart]
    column: ft.Column
    subtitle: ft.Text | None = None

    def refresh(
        self,
        store: MetricsStore | None,
        *,
        window_s: float,
        server: str | None = None,
        poll_interval: float = 2.0,
        push: bool = True,
    ) -> None:
        if store is None:
            if self.subtitle is not None:
                self.subtitle.value = "Metrics disabled (set METRICS=1 in .env)"
            return
        if self.subtitle is not None:
            self.subtitle.value = charts_subtitle(
                store,
                window_s=window_s,
                server=server,
                poll_interval=poll_interval,
            )
            if push:
                try:
                    self.subtitle.update()
                except Exception:
                    pass
        for chart in self.charts:
            series = store.series(chart.spec.field, window_s=window_s, server=server)
            chart.set_series(series, push=push)


def build_live_charts_panel(*, subtitle: ft.Text | None = None) -> LiveChartsPanel:
    charts = [make_live_chart(spec) for spec in CHART_SPECS]
    column = ft.Column(
        [c.card for c in charts if c.card is not None],
        spacing=8,
        expand=True,
        scroll=ft.ScrollMode.AUTO,
    )
    return LiveChartsPanel(charts=charts, column=column, subtitle=subtitle)


def metric_chart_card(spec: ChartSpec, series: list[tuple[float, float]]) -> ft.Control:
    """One-shot card (tests / non-live callers). Prefer LiveChartsPanel in the GUI."""
    chart = make_live_chart(spec)
    chart.set_series(series, push=False)
    assert chart.card is not None
    return chart.card


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
    panel = build_live_charts_panel()
    panel.refresh(store, window_s=window_s, server=server, push=False)
    return panel.column
