"""System tray icon (Windows-first).

Uses the branded ``assets/ollama-sentinel.ico`` and recolors its green accents
for a small set of statuses that stay readable at 16–32 px:

* ok    — idle, clear
* busy  — generating / prompt / embed (no alarms)
* warn  — spill / VRAM / other non-paging alarms
* alarm — paging
* down  — unreachable
"""

from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw
import pystray

# Brand green in the ICO (~#40C878). Recolor pixels near this hue.
_BRAND_RGB = (64, 200, 120)

STATUS_COLORS: dict[str, tuple[int, int, int]] = {
    "ok": _BRAND_RGB,
    "busy": (34, 211, 238),
    "warn": (245, 158, 11),
    "alarm": (239, 68, 68),
    "down": (107, 114, 128),
}

STATUS_LABELS = {
    "ok": "OK",
    "busy": "Busy",
    "warn": "Warn",
    "alarm": "Paging",
    "down": "Down",
}


def icon_asset_candidates() -> list[Path]:
    here = Path(__file__).resolve()
    return [
        here.parent.parent / "assets" / "ollama-sentinel.ico",
        here.parent / "assets" / "ollama-sentinel.ico",
        Path.cwd() / "assets" / "ollama-sentinel.ico",
    ]


def find_icon_asset() -> Path | None:
    for path in icon_asset_candidates():
        if path.is_file():
            return path
    return None


def _fallback_base(size: int = 64) -> Image.Image:
    """Simple O-ring if the ICO is missing (tests / odd installs)."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    pad = size // 10
    draw.ellipse((pad, pad, size - pad, size - pad), outline=_BRAND_RGB + (255,), width=max(3, size // 10))
    inner = size // 3
    draw.ellipse(
        (inner, inner, size - inner, size - inner),
        outline=_BRAND_RGB + (255,),
        width=max(2, size // 14),
    )
    return img


@lru_cache(maxsize=1)
def _load_base_icon(size: int = 64) -> Image.Image:
    path = find_icon_asset()
    if path is None:
        return _fallback_base(size)
    try:
        img = Image.open(path).convert("RGBA")
        if img.size != (size, size):
            img = img.resize((size, size), Image.Resampling.LANCZOS)
        return img
    except OSError:
        return _fallback_base(size)


def _is_brand_green(r: int, g: int, b: int, a: int) -> bool:
    if a < 40:
        return False
    # Green-dominant and reasonably bright (matches the ICO accents).
    return g >= 90 and g > r + 25 and g > b + 25


def recolor_brand(img: Image.Image, rgb: tuple[int, int, int]) -> Image.Image:
    """Replace brand-green accent pixels with ``rgb``, preserving alpha."""
    out = img.copy()
    pixels = out.load()
    tr, tg, tb = rgb
    w, h = out.size
    for y in range(h):
        for x in range(w):
            r, g, b, a = pixels[x, y]
            if _is_brand_green(r, g, b, a):
                pixels[x, y] = (tr, tg, tb, a)
    return out


def make_status_icon(status: str, *, size: int = 64) -> Image.Image:
    color = STATUS_COLORS.get(status, STATUS_COLORS["ok"])
    base = _load_base_icon(size)
    if status == "ok" and color == _BRAND_RGB:
        return base.copy()
    return recolor_brand(base, color)


def resolve_tray_status(
    *,
    reachable: bool,
    alarms: list[dict[str, Any]] | None = None,
    phase: str | None = None,
) -> str:
    """Pick one of ok / busy / warn / alarm / down."""
    if not reachable:
        return "down"
    alarms = alarms or []
    if any(a.get("type") == "paging" for a in alarms):
        return "alarm"
    if alarms:
        return "warn"
    if phase in ("generating", "prompt", "embed", "request"):
        return "busy"
    return "ok"


def format_tray_tooltip(
    *,
    status: str,
    server: str | None = None,
    summary: str | None = None,
    alarms: list[dict[str, Any]] | None = None,
) -> str:
    """Compact hover text for the tray icon."""
    label = STATUS_LABELS.get(status, status)
    lines = [f"ollama-sentinel · {label}"]
    if server:
        lines[0] = f"ollama-sentinel · {server} · {label}"
    if summary:
        lines.append(summary.strip())
    for alarm in (alarms or [])[:2]:
        msg = (alarm.get("message") or "").strip()
        if msg and msg not in lines:
            lines.append(msg)
    # Windows tray tooltips are short; keep under ~120 chars-ish.
    text = "\n".join(lines)
    if len(text) > 180:
        text = text[:177] + "…"
    return text


def start_tray(
    *,
    on_open: Callable[[], None] | None = None,
    on_restart: Callable[[], None] | None = None,
    on_quit: Callable[[], None] | None = None,
) -> pystray.Icon:
    items = [
        pystray.MenuItem("Open", lambda *_: on_open() if on_open else None, default=True),
    ]
    if on_restart:
        items.append(pystray.MenuItem("Restart", lambda *_: on_restart()))
    items.append(pystray.MenuItem("Quit", lambda *_: on_quit() if on_quit else None))
    tip = format_tray_tooltip(status="ok")
    icon = pystray.Icon(
        "ollama-sentinel",
        make_status_icon("ok"),
        tip,
        menu=pystray.Menu(*items),
    )
    icon._sentinel_status = "ok"  # type: ignore[attr-defined]
    icon._sentinel_tip = tip  # type: ignore[attr-defined]

    def run():
        icon.run()

    threading.Thread(target=run, daemon=True).start()
    return icon


def update_tray(
    icon: pystray.Icon,
    *,
    reachable: bool = True,
    alarms: list[dict[str, Any]] | None = None,
    phase: str | None = None,
    summary: str | None = None,
    server: str | None = None,
) -> str:
    """Refresh tray glyph + tooltip. Returns the resolved status key."""
    alarms = alarms or []
    status = resolve_tray_status(reachable=reachable, alarms=alarms, phase=phase)
    tip = format_tray_tooltip(
        status=status,
        server=server,
        summary=summary,
        alarms=alarms,
    )
    if getattr(icon, "_sentinel_status", None) != status:
        icon.icon = make_status_icon(status)
        icon._sentinel_status = status  # type: ignore[attr-defined]
    if getattr(icon, "_sentinel_tip", None) != tip:
        icon.title = tip
        icon._sentinel_tip = tip  # type: ignore[attr-defined]
    return status


def set_tray_color(icon: pystray.Icon, alarms: list[dict]) -> None:
    """Backward-compatible wrapper: alarm-only coloring."""
    update_tray(icon, reachable=True, alarms=alarms, phase=None)
