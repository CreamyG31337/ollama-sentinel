"""Rich console rendering."""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from ollama_sentinel.alarms import format_expires, gpu_pct
from ollama_sentinel.activity import build_server_activity, model_detail_line
from ollama_sentinel.inventory import build_inventory, inventory_detail_line, inventory_summary
from ollama_sentinel.telemetry import format_gpu_line, format_poll_age, is_stale


def _free_vram_summary(gpus: list[dict[str, Any]] | None) -> tuple[float | None, float | None]:
    if not gpus:
        return None, None
    total = sum(g.get("memory_total") or 0 for g in gpus)
    free = sum(g.get("memory_free") or 0 for g in gpus)
    if total <= 0:
        return None, None
    return free / 1e9, 100 * free / total


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


def _format_process_vram(process_vram: dict[str, Any] | None, *, interval: float) -> list[str]:
    if not process_vram or not process_vram.get("enabled", True):
        return []
    lines: list[str] = []
    ts = process_vram.get("polled_at_ts")
    now = time.time()
    header = "Process VRAM"
    if ts is not None:
        age_line = format_poll_age(ts, now)
        if is_stale(ts, interval, now):
            header += f" STALE as of {age_line}"
        else:
            header += f" as of {age_line}"
    elif process_vram.get("error"):
        header += f" (error: {process_vram['error']})"
    lines.append(f"  {header}")
    rows = process_vram.get("rows") or []
    if not rows and not process_vram.get("error"):
        lines.append("  (no processes above threshold)")
    for row in rows:
        local_gb = row.get("bytes", 0) / 1e9
        util = row.get("engine_3d_pct")
        util_s = f", {util:.0f}% util" if util is not None else ""
        non_local = row.get("non_local_bytes")
        if non_local is not None:
            lines.append(
                f"  {row.get('pid')} {row.get('name')}: "
                f"{local_gb:.2f} GB local, {non_local / 1e9:.2f} GB non-local{util_s}"
            )
        else:
            lines.append(f"  {row.get('pid')} {row.get('name')}: {local_gb:.2f} GB{util_s}")
    return lines


def _format_activity(snap: dict[str, Any], proc_rows: list[dict[str, Any]] | None) -> list[str]:
    activity = snap.get("activity")
    if activity is None:
        if not snap.get("local_gpu", True):
            return []
        import sys

        if sys.platform == "win32":
            activity = build_server_activity(proc_rows=proc_rows).to_dict()
        else:
            return []
    lines = [f"  Activity: {activity.get('summary', '—')}"]
    for r in activity.get("runners") or []:
        util = r.get("engine_3d_pct") or 0
        tag = "busy" if r.get("busy") else "idle"
        lines.append(
            f"    runner pid {r.get('pid')} {r.get('vram_bytes', 0) / 1e9:.1f} GB "
            f"· {util:.0f}% util · {tag}"
        )
    last = activity.get("last_request")
    if last:
        lines.append(
            f"    last {last.get('method')} {last.get('path')} from {last.get('client')}"
        )
    return lines


def render_snapshot_plain(
    snapshots: list[dict[str, Any]],
    alarms: list[dict[str, Any]],
    *,
    poll_interval: float = 5.0,
    process_vram: dict[str, Any] | None = None,
    proc_vram_interval: float = 30.0,
    once: bool = False,
) -> str:
    lines: list[str] = []
    now = time.time()
    for snap in snapshots:
        name = snap.get("server", "?")
        polled_ts = snap.get("polled_at_ts")
        header = f"[{name}]"
        if polled_ts is not None:
            age = format_poll_age(polled_ts, now)
            stale = snap.get("stale") or (
                not once and is_stale(polled_ts, poll_interval, now)
            )
            if stale:
                header += f" STALE {age}"
            else:
                header += f" {age}"
        if not snap.get("reachable"):
            if snap.get("optional"):
                lines.append(f"{header} Ollama offline (optional host): {snap.get('error')}")
            else:
                lines.append(f"{header} Ollama unreachable: {snap.get('error')}")
            continue
        lines.append(f"{header} Ollama {snap.get('version', '?')}")
        inv = build_inventory(snap)
        free_gb, free_pct = _free_vram_summary(snap.get("gpus"))
        lines.append(f"  Library: {inventory_summary(inv, free_vram_gb=free_gb, free_vram_pct=free_pct)}")
        if snap.get("local_gpu", True):
            proc_rows = (process_vram or {}).get("rows") if process_vram else None
            lines.extend(_format_activity(snap, proc_rows))
        elif snap.get("activity"):
            lines.extend(_format_activity(snap, None))
        for m in snap.get("models") or []:
            mn = m.get("name") or "?"
            size = m.get("size") or 0
            sv = m.get("size_vram") or 0
            pct = gpu_pct(size, sv)
            exp = format_expires(m.get("expires_at"), server_url=snap.get("url"))
            detail = model_detail_line(m)
            lines.append(
                f"  {mn}: {detail} · {size/1e9:.1f} GB total, {sv/1e9:.1f} GB VRAM, "
                f"{100-pct}% CPU / {pct}% GPU, expires {exp}"
            )
        for gpu in snap.get("gpus") or []:
            lines.append(f"  GPU {format_gpu_line(gpu)}")
        if snap.get("local_gpu", True):
            lines.extend(
                _format_process_vram(process_vram, interval=proc_vram_interval)
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
    table.add_column("Details")
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
            elif row.get("would_spill") is None:
                fit = "fit unknown"
            split = ""
            if row.get("loaded"):
                split = f"{row.get('gpu_pct')}% GPU"
            table.add_row(
                srv,
                row["name"],
                inventory_detail_line(row),
                f"{row['size_gb']:.1f}",
                "yes" if row["loaded"] else "no",
                fit,
                split,
            )
    return table


def build_live_panel(
    snapshots: list[dict[str, Any]],
    alarms: list[dict[str, Any]],
    *,
    poll_interval: float = 5.0,
    process_vram: dict[str, Any] | None = None,
    proc_vram_interval: float = 30.0,
) -> Table:
    table = Table(title="ollama-sentinel", expand=True)
    table.add_column("Section", style="cyan")
    table.add_column("Details")
    now = time.time()
    for snap in snapshots:
        srv = snap.get("server", "?")
        polled_ts = snap.get("polled_at_ts")
        age_suffix = ""
        if polled_ts is not None:
            age = format_poll_age(polled_ts, now)
            if snap.get("stale") or is_stale(polled_ts, poll_interval, now):
                age_suffix = f" · [dim]STALE {age}[/dim]"
            else:
                age_suffix = f" · {age}"
        if not snap.get("reachable"):
            if snap.get("optional"):
                table.add_row(srv, f"[dim]offline (optional host)[/dim] {snap.get('error')}{age_suffix}")
            else:
                table.add_row(srv, f"[red]unreachable[/red] {snap.get('error')}{age_suffix}")
            continue
        inv = build_inventory(snap)
        free_gb, free_pct = _free_vram_summary(snap.get("gpus"))
        table.add_row(
            srv,
            f"v{snap.get('version')} · {inventory_summary(inv, free_vram_gb=free_gb, free_vram_pct=free_pct)}{age_suffix}",
        )
        if snap.get("local_gpu", True):
            proc_rows = (process_vram or {}).get("rows") if process_vram else None
            for act_line in _format_activity(snap, proc_rows):
                table.add_row("  activity", act_line.strip())
        elif snap.get("activity"):
            for act_line in _format_activity(snap, None):
                table.add_row("  activity", act_line.strip())
        for m in snap.get("models") or []:
            mn = m.get("name") or "?"
            size = m.get("size") or 0
            sv = m.get("size_vram") or 0
            pct = gpu_pct(size, sv)
            exp = format_expires(m.get("expires_at"), server_url=snap.get("url"))
            detail = model_detail_line(m)
            table.add_row(
                "  model",
                f"{mn} · {detail} · {size/1e9:.1f} GB · {sv/1e9:.1f} GB VRAM · "
                f"{100-pct}% CPU / {pct}% GPU · expires {exp}",
            )
        for gpu in snap.get("gpus") or []:
            table.add_row("  gpu", format_gpu_line(gpu))
    if any(s.get("local_gpu", True) for s in snapshots):
        for line in _format_process_vram(process_vram, interval=proc_vram_interval):
            if line.startswith("  Process VRAM"):
                table.add_row("proc vram", line.strip())
            else:
                table.add_row("  proc", line.strip())
    alarm_str = _alarm_text(alarms)
    table.add_row("alarms", alarm_str)
    return table


class LiveRenderer:
    def __init__(self) -> None:
        self.console = Console()

    def run(
        self,
        poll_fn,
        interval: float,
        *,
        proc_vram_interval: float = 30.0,
        get_process_vram=None,
    ) -> None:
        import time as time_mod

        snapshots, alarms = poll_fn()
        proc = get_process_vram() if get_process_vram else None
        with Live(
            build_live_panel(
                snapshots,
                alarms,
                poll_interval=interval,
                process_vram=proc,
                proc_vram_interval=proc_vram_interval,
            ),
            refresh_per_second=4,
        ) as live:
            while True:
                time_mod.sleep(interval)
                snapshots, alarms = poll_fn()
                proc = get_process_vram() if get_process_vram else None
                live.update(
                    build_live_panel(
                        snapshots,
                        alarms,
                        poll_interval=interval,
                        process_vram=proc,
                        proc_vram_interval=proc_vram_interval,
                    )
                )
