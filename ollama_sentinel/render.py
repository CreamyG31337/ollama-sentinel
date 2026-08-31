"""Rich console rendering."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from ollama_sentinel.alarms import format_expires, gpu_pct
from ollama_sentinel.inventory import build_inventory, inventory_summary


def _alarm_text(active: list[dict[str, Any]]) -> Text:
    if not active:
        t = Text("OK")
        t.stylize("bold green")
        return t
    t = Text()
    for i, a in enumerate(active):
        if i:
            t.append("\n")
        t.append(a["message"], style="bold red")
    return t


def render_snapshot_plain(snapshots: list[dict[str, Any]], alarms: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for snap in snapshots:
        name = snap.get("server", "?")
        if not snap.get("reachable"):
            lines.append(f"[{name}] Ollama unreachable: {snap.get('error')}")
            continue
        lines.append(f"[{name}] Ollama {snap.get('version', '?')}")
        inv = build_inventory(snap)
        lines.append(f"  Library: {inventory_summary(inv)}")
        for m in snap.get("models") or []:
            mn = m.get("name") or "?"
            size = m.get("size") or 0
            sv = m.get("size_vram") or 0
            pct = gpu_pct(size, sv)
            exp = format_expires(m.get("expires_at"))
            lines.append(
                f"  {mn}: {size/1e9:.1f} GB total, {sv/1e9:.1f} GB VRAM, "
                f"{100-pct}% CPU / {pct}% GPU, expires {exp}"
            )
        for gpu in snap.get("gpus") or []:
            used = gpu.get("memory_used") or 0
            total = gpu.get("memory_total") or 0
            lines.append(
                f"  GPU {gpu.get('index')}: {used/1e9:.1f}/{total/1e9:.1f} GB, "
                f"util {gpu.get('utilization') or '?'}%, "
                f"power {gpu.get('power_draw') or '?'} / {gpu.get('power_limit') or '?'} W"
            )
    if alarms:
        lines.append("ALARMS:")
        for a in alarms:
            lines.append(f"  {a['message']}")
    else:
        lines.append("Alarms: OK")
    return "\n".join(lines)


def render_list_table(snapshots: list[dict[str, Any]]) -> Table:
    table = Table(title="Installed models")
    table.add_column("Server")
    table.add_column("Name")
    table.add_column("Size GB")
    table.add_column("Loaded")
    table.add_column("Fit")
    table.add_column("Split")
    for snap in snapshots:
        if not snap.get("reachable"):
            continue
        srv = snap.get("server", "?")
        for row in build_inventory(snap):
            fit = "—"
            if row.get("loaded"):
                fit = f"{row.get('gpu_pct', '?')}% GPU"
            elif row.get("would_spill") is True:
                fit = "would spill"
            elif row.get("would_spill") is False:
                fit = "fits"
            split = ""
            if row.get("loaded"):
                split = f"{row.get('gpu_pct')}% GPU"
            table.add_row(
                srv,
                row["name"],
                f"{row['size_gb']:.1f}",
                "yes" if row["loaded"] else "no",
                fit,
                split,
            )
    return table


def build_live_panel(snapshots: list[dict[str, Any]], alarms: list[dict[str, Any]]) -> Table:
    table = Table(title="ollama-sentinel", expand=True)
    table.add_column("Section", style="cyan")
    table.add_column("Details")
    for snap in snapshots:
        srv = snap.get("server", "?")
        if not snap.get("reachable"):
            table.add_row(srv, f"[red]unreachable[/red] {snap.get('error')}")
            continue
        inv = build_inventory(snap)
        table.add_row(srv, f"v{snap.get('version')} · {inventory_summary(inv)}")
        for m in snap.get("models") or []:
            mn = m.get("name") or "?"
            size = m.get("size") or 0
            sv = m.get("size_vram") or 0
            pct = gpu_pct(size, sv)
            table.add_row(
                "  model",
                f"{mn} · {size/1e9:.1f} GB · {sv/1e9:.1f} GB VRAM · {100-pct}% CPU / {pct}% GPU",
            )
        for gpu in snap.get("gpus") or []:
            used = gpu.get("memory_used") or 0
            total = gpu.get("memory_total") or 0
            table.add_row(
                "  gpu",
                f"#{gpu.get('index')} {used/1e9:.1f}/{total/1e9:.1f} GB · "
                f"{gpu.get('utilization') or '?'}% · {gpu.get('power_draw') or '?'} W",
            )
    alarm_str = _alarm_text(alarms)
    table.add_row("alarms", alarm_str)
    return table


class LiveRenderer:
    def __init__(self) -> None:
        self.console = Console()

    def run(self, poll_fn, interval: float) -> None:
        import time

        snapshots, alarms = poll_fn()
        with Live(build_live_panel(snapshots, alarms), refresh_per_second=4) as live:
            while True:
                time.sleep(interval)
                snapshots, alarms = poll_fn()
                live.update(build_live_panel(snapshots, alarms))
