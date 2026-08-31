"""System tray icon (Windows-first)."""

from __future__ import annotations

import threading
from typing import Callable

from PIL import Image, ImageDraw
import pystray


def _make_icon(color: str) -> Image.Image:
    img = Image.new("RGB", (64, 64), color)
    draw = ImageDraw.Draw(img)
    draw.ellipse((8, 8, 56, 56), fill="white")
    return img


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
    icon = pystray.Icon(
        "ollama-sentinel",
        _make_icon("green"),
        "ollama-sentinel",
        menu=pystray.Menu(*items),
    )

    def run():
        icon.run()

    threading.Thread(target=run, daemon=True).start()
    return icon


def set_tray_color(icon: pystray.Icon, alarms: list[dict]) -> None:
    if not alarms:
        color = "green"
    elif any(a.get("type") == "paging" for a in alarms):
        color = "red"
    else:
        color = "orange"
    icon.icon = _make_icon(color)
