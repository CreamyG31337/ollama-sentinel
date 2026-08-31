"""CLI entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ollama_sentinel.alarms import AlarmState, evaluate_alarms
from ollama_sentinel.catalog import search_models, typeahead
from ollama_sentinel.config import build_parser, resolve_config, selected_servers
from ollama_sentinel.inventory import build_inventory
from ollama_sentinel.log import append_alarm_log
from ollama_sentinel.notify import notify_transition
from ollama_sentinel.poll import poll_all
from ollama_sentinel.pull import pull_model
from ollama_sentinel.render import LiveRenderer, render_list_table, render_snapshot_plain
from ollama_sentinel.smi import query_gpus
from ollama_sentinel.state import load_state, save_state


def _evaluate_all(
    snapshots: list[dict[str, Any]],
    state: AlarmState,
    thresholds,
) -> tuple[list[dict[str, Any]], AlarmState, list]:
    from ollama_sentinel.alarms import AlarmTransition

    streak = dict(state.paging_streak)
    all_active: list[dict[str, Any]] = []
    for snap in snapshots:
        partial_state = AlarmState(paging_streak=streak, active_ids=set())
        active, new_partial, _ = evaluate_alarms(snap, partial_state, thresholds)
        all_active.extend(active)
        streak = new_partial.paging_streak

    final_ids = {a["id"] for a in all_active}
    transitions = []
    for aid in final_ids - state.active_ids:
        msg = next((a["message"] for a in all_active if a["id"] == aid), aid)
        transitions.append(AlarmTransition("FIRE", aid, msg))
    for aid in state.active_ids - final_ids:
        transitions.append(AlarmTransition("RESOLVED", aid, f"RESOLVED {aid}"))

    new_state = AlarmState(paging_streak=streak, active_ids=final_ids)
    return all_active, new_state, transitions


def _poll(cfg) -> list[dict[str, Any]]:
    servers = selected_servers(cfg)
    return poll_all(
        [{"name": s.name, "url": s.url, "local_gpu": s.local_gpu} for s in servers],
        gpu_filter=cfg.gpu_filter,
        query_gpus_fn=query_gpus,
    )


def _exit_code(snapshots: list[dict[str, Any]], alarms: list[dict[str, Any]]) -> int:
    if snapshots and all(not s.get("reachable") for s in snapshots):
        return 2
    if alarms:
        return 1
    return 0


def cmd_search(args, cfg) -> int:
    from rich.console import Console
    from rich.table import Table

    query = getattr(args, "query", "") or ""
    results = search_models(
        query,
        sort=getattr(args, "sort", "trendingScore"),
        limit=getattr(args, "limit", 20),
        token=cfg.hf_token,
    )
    table = Table(title=f"HF search: {query or '(trending)'}")
    table.add_column("Model")
    table.add_column("Pull as")
    table.add_column("Downloads")
    for r in results:
        table.add_row(r["id"], r["pull_name"], str(r.get("downloads") or "—"))
    Console().print(table)
    return 0


def cmd_pull(args, cfg) -> int:
    servers = {s.name: s for s in cfg.servers}
    srv_name = args.server
    if srv_name not in servers:
        print(f"Unknown server: {srv_name}", file=sys.stderr)
        return 2
    srv = servers[srv_name]
    if not args.yes and srv.local_gpu:
        snaps = poll_all(
            [{"name": srv.name, "url": srv.url, "local_gpu": True}],
            gpu_filter=cfg.gpu_filter,
            query_gpus_fn=query_gpus,
        )
        inv = build_inventory(snaps[0]) if snaps else []
        # rough check: any unloaded model that would spill for this pull size unknown — skip if no free vram info
        from ollama_sentinel.inventory import free_vram_bytes

        free = free_vram_bytes(snaps[0].get("gpus") if snaps else None)
        if free is not None and free < 1e9:
            print(f"Warning: only {free/1e9:.1f} GB free VRAM on {srv_name}", file=sys.stderr)
            if not args.yes:
                ans = input("Continue pull? [y/N] ").strip().lower()
                if ans != "y":
                    return 0
    print(f"Pulling {args.model} to {srv_name} ({srv.url})...")
    for event in pull_model(srv.url, args.model):
        if "error" in event:
            print(event["error"], file=sys.stderr)
            return 2
        status = event.get("status") or event.get("digest") or event
        print(status)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = resolve_config(args)

    if args.gui or args.tray or args.tray_only:
        from ollama_sentinel.ui import run_gui

        run_gui(cfg, tray=args.tray or args.tray_only,
                start_hidden=args.tray_only)
        return 0

    if args.command == "search":
        return cmd_search(args, cfg)

    if args.command == "pull":
        return cmd_pull(args, cfg)

    state_path = cfg.state_file or Path("state.json")
    state = load_state(state_path)

    def cycle() -> tuple[list[dict[str, Any]], list[dict[str, Any]], AlarmState, list]:
        snapshots = _poll(cfg)
        active, new_state, transitions = _evaluate_all(snapshots, state, cfg.thresholds)
        return snapshots, active, new_state, transitions

    if args.once or args.json or args.list:
        snapshots, active, new_state, transitions = cycle()
        save_state(state_path, new_state)
        if args.toast:
            for t in transitions:
                notify_transition(t)
        if args.list:
            from rich.console import Console

            Console().print(render_list_table(snapshots))
            return _exit_code(snapshots, active)
        payload = {
            "snapshots": snapshots,
            "alarms": active,
            "inventory": {
                s.get("server"): build_inventory(s) for s in snapshots if s.get("reachable")
            },
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(render_snapshot_plain(snapshots, active))
        if args.log:
            append_alarm_log(
                Path(args.log),
                alarms=active,
                transitions=transitions,
                on_transition_only=False,
            )
        return _exit_code(snapshots, active)

    # Live mode
    renderer = LiveRenderer()

    def poll_fn():
        nonlocal state
        snapshots = _poll(cfg)
        active, new_state, transitions = _evaluate_all(snapshots, state, cfg.thresholds)
        if args.toast:
            for t in transitions:
                notify_transition(t)
        save_state(state_path, new_state)
        state = new_state
        if args.log:
            append_alarm_log(
                Path(args.log),
                alarms=active,
                transitions=transitions,
                on_transition_only=True,
            )
        return snapshots, active

    try:
        renderer.run(poll_fn, cfg.poll_interval)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
