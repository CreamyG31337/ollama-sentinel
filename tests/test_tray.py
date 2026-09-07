"""Tests for system tray status helpers."""

from __future__ import annotations

import struct
from io import BytesIO

from PIL import Image

from ollama_sentinel.tray import (
    ICO_SIZES,
    STATUS_COLORS,
    find_icon_asset,
    format_tray_tooltip,
    make_status_icon,
    recolor_brand,
    render_brand_icon,
    resolve_tray_status,
    set_tray_color,
)


def test_resolve_priority():
    assert resolve_tray_status(reachable=False) == "down"
    assert (
        resolve_tray_status(
            reachable=True,
            alarms=[{"type": "paging", "message": "PCIe"}],
            phase="generating",
        )
        == "alarm"
    )
    assert (
        resolve_tray_status(
            reachable=True,
            alarms=[{"type": "spill", "message": "spill"}],
            phase="idle",
        )
        == "warn"
    )
    assert resolve_tray_status(reachable=True, alarms=[], phase="generating") == "busy"
    assert resolve_tray_status(reachable=True, alarms=[], phase="prompt") == "busy"
    assert resolve_tray_status(reachable=True, alarms=[], phase="idle") == "ok"


def test_tooltip_includes_server_and_summary():
    tip = format_tray_tooltip(
        status="busy",
        server="cr-desktop-3090",
        summary="qwen: Generating 100 tokens",
        alarms=[],
    )
    assert "cr-desktop-3090" in tip
    assert "Busy" in tip
    assert "Generating 100" in tip


def test_tooltip_includes_alarms():
    tip = format_tray_tooltip(
        status="alarm",
        alarms=[{"type": "paging", "message": "PCIe paging suspected"}],
    )
    assert "Paging" in tip
    assert "PCIe paging" in tip


def test_make_status_icon_sizes_and_colors():
    ok = make_status_icon("ok", size=32)
    assert ok.size == (32, 32)
    assert ok.mode == "RGBA"
    alarm = make_status_icon("alarm", size=32)
    # Alarm should have red-ish pixels where brand green was.
    reds = sum(
        1
        for r, g, b, a in alarm.getdata()
        if a > 40 and r > 180 and r > g + 40 and r > b + 40
    )
    greens = sum(
        1
        for r, g, b, a in ok.getdata()
        if a > 40 and g > 90 and g > r + 25 and g > b + 25
    )
    assert greens > 20
    assert reds > 20


def test_recolor_preserves_alpha_and_non_green():
    img = Image.new("RGBA", (4, 4), (32, 34, 38, 255))
    img.putpixel((1, 1), (64, 200, 120, 200))
    img.putpixel((2, 2), (255, 0, 0, 100))
    out = recolor_brand(img, STATUS_COLORS["busy"])
    assert out.getpixel((0, 0)) == (32, 34, 38, 255)
    assert out.getpixel((2, 2)) == (255, 0, 0, 100)
    r, g, b, a = out.getpixel((1, 1))
    assert a == 200
    assert (r, g, b) == STATUS_COLORS["busy"]


def test_set_tray_color_compat_wrapper():
    class Fake:
        def __init__(self):
            self.icon = None
            self.title = ""

    fake = Fake()
    set_tray_color(fake, [{"type": "spill", "message": "spill"}])
    assert fake._sentinel_status == "warn"
    assert "Warn" in fake.title


def test_render_brand_icon_is_antialiased():
    img = render_brand_icon(64)
    assert img.size == (64, 64)
    fringe = sum(1 for _r, _g, _b, a in img.getdata() if 1 <= a <= 200)
    assert fringe > 50


def test_brand_ico_has_antialiased_sizes():
    path = find_icon_asset()
    assert path is not None
    data = path.read_bytes()
    _reserved, _typ, count = struct.unpack_from("<HHH", data, 0)
    assert count >= len(ICO_SIZES)
    offset = 6
    seen: set[int] = set()
    for _ in range(count):
        w, _h, _c, _res, _planes, _bpp, size, off = struct.unpack_from("<BBBBHHII", data, offset)
        w = 256 if w == 0 else w
        seen.add(w)
        frame = Image.open(BytesIO(data[off : off + size])).convert("RGBA")
        fringe = sum(1 for _r, _g, _b, a in frame.getdata() if 1 <= a <= 200)
        # Posterized frames have zero fringe; every size should soft-edge.
        assert fringe > 0, f"{w}x{w} lacks antialiasing"
        offset += 16
    assert set(ICO_SIZES) <= seen
