"""CLI entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ollama_sentinel.metrics import make_metrics_store
from ollama_sentinel.activity import build_server_activity
from ollama_sentinel.alarms import AlarmState, evaluate_alarms
from ollama_sentinel.catalog import search_models, typeahead
from ollama_sentinel.config import build_parser, resolve_config, resolve_gui_options, selected_servers
from ollama_sentinel.instance import InstanceLock
from ollama_sentinel.inventory import build_inventory
from ollama_sentinel.log import append_alarm_log
from ollama_sentinel.notify import notify_transition
from ollama_sentinel.poll import poll_all
from ollama_sentinel.proc_vram import ProcessVramCollector, query_process_vram
from ollama_sentinel.telemetry import polled_at_iso
from ollama_sentinel.pull import pull_model
from ollama_sentinel.render import LiveRenderer, render_list_table, render_snapshot_plain
from ollama_sentinel.smi import query_gpus
from ollama_sentinel.state import load_state, save_state
from ollama_sentinel.doctor import (
    DoctorFinding,
    collect_doctor_inputs,
    evaluate_doctor_alarms,
    findings_exit_code,
    run_doctor,
)
from ollama_sentinel.doctor_win import kill_orphan_pids
from ollama_sentinel.unload import unload_model, unload_models


def _attach_activity(
    snapshots: list[dict[str, Any]],
    proc_rows: list[dict[str, Any]] | None,
    cfg,
) -> list[dict[str, Any]]:
    if sys.platform != "win32":
        return snapshots
    servers = {s.name: s for s in selected_servers(cfg)}
    out: list[dict[str, Any]] = []
    for snap in snapshots:
        if not snap.get("reachable"):
            out.append(snap)
            continue
        srv = servers.get(snap.get("server"))
        if srv is not None and not srv.local_gpu:
            out.append(snap)
            continue
        enriched = dict(snap)
        enriched["activity"] = build_server_activity(proc_rows=proc_rows).to_dict()
        out.append(enriched)
    return out


def _evaluate_all(
    snapshots: list[dict[str, Any]],
    state: AlarmState,
    thresholds,
    *,
    doctor_findings: list[DoctorFinding] | None = None,
) -> tuple[list[dict[str, Any]], AlarmState, list]:
    from ollama_sentinel.alarms import AlarmTransition

    streak = dict(state.paging_streak)
    all_active: list[dict[str, Any]] = []
    for snap in snapshots:
        partial_state = AlarmState(paging_streak=streak, active_ids=set())
        active, new_partial, _ = evaluate_alarms(snap, partial_state, thresholds)
        all_active.extend(active)
        streak = new_partial.paging_streak

    if doctor_findings:
        all_active.extend(evaluate_doctor_alarms(doctor_findings))

    final_ids = {a["id"] for a in all_active}
    transitions = []
    for aid in final_ids - state.active_ids:
        msg = next((a["message"] for a in all_active if a["id"] == aid), aid)
        transitions.append(AlarmTransition("FIRE", aid, msg))
    for aid in state.active_ids - final_ids:
        transitions.append(AlarmTransition("RESOLVED", aid, f"RESOLVED {aid}"))

    new_state = AlarmState(paging_streak=streak, active_ids=final_ids)
    return all_active, new_state, transitions


def _doctor_findings_for_snapshots(
    snapshots: list[dict[str, Any]],
    cfg,
    *,
    proc_rows: list[dict[str, Any]] | None = None,
) -> list[DoctorFinding]:
    """Run Check A/B (and full doctor when inputs available) for local_gpu snaps."""
    if sys.platform != "win32":
        return []
    local = [s for s in snapshots if s.get("reachable")]
    servers = {s.name: s for s in selected_servers(cfg)}
    findings: list[DoctorFinding] = []
    try:
        inputs = collect_doctor_inputs()
    except Exception:
        return []
    for snap in local:
        srv = servers.get(snap.get("server"))
        if srv is not None and not srv.local_gpu:
            continue
        # Default single-server local_gpu=True when no servers.json entry match
        if srv is None and snap.get("server") not in (None, "local"):
            continue
        findings.extend(
            run_doctor(
                snap,
                registry=inputs["registry"],
                log_cfg=inputs["log_cfg"],
                log_path=inputs["log_path"],
                runners=inputs["runners"],
                proc_rows=proc_rows,
                ollama_url=cfg.ollama_url,
                registry_mtime=inputs["registry_mtime"],
                restart_remedy=inputs["restart_remedy"],
            )
        )
        break  # one local doctor pass per cycle
    return findings


def _proc_vram_enabled(cfg) -> bool:
    return cfg.proc_vram and any(s.local_gpu for s in selected_servers(cfg))


def _make_proc_collector(cfg) -> ProcessVramCollector | None:
    if not _proc_vram_enabled(cfg):
        return None
    collector = ProcessVramCollector(
        interval=cfg.proc_vram_interval,
        enabled=True,
        min_bytes=cfg.proc_vram_min_mb * 1024 * 1024,
    )
    collector.start()
    return collector


def _process_vram_payload(
    collector: ProcessVramCollector | None,
    cfg,
    *,
    sync: bool = False,
) -> dict[str, Any] | None:
    if not _proc_vram_enabled(cfg) or (collector is None and not sync):
        return None
    if sync:
        import time as time_mod

        try:
            rows = query_process_vram(cfg.proc_vram_min_mb * 1024 * 1024)
            now = time_mod.time()
            return {
                "enabled": True,
                "polled_at": polled_at_iso(now),
                "polled_at_ts": now,
                "stale": False,
                "error": None,
                "rows": rows,
            }
        except Exception as exc:
            return {
                "enabled": True,
                "polled_at": None,
                "polled_at_ts": None,
                "stale": True,
                "error": str(exc),
                "rows": [],
            }
    snap = collector.get_snapshot()
    return {
        "enabled": True,
        "polled_at": snap.get("polled_at"),
        "polled_at_ts": snap.get("polled_at_ts"),
        "stale": snap.get("stale", False),
        "error": snap.get("error"),
        "rows": snap.get("rows") or [],
    }


def _poll(cfg, last_snapshots: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    servers = selected_servers(cfg)
    return poll_all(
        [{"name": s.name, "url": s.url, "local_gpu": s.local_gpu} for s in servers],
        gpu_filter=cfg.gpu_filter,
        query_gpus_fn=query_gpus,
        last_snapshots=last_snapshots,
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


def cmd_unload(args, cfg) -> int:
    servers = {s.name: s for s in cfg.servers}
    srv_name = args.server
    if srv_name not in servers:
        print(f"Unknown server: {srv_name}", file=sys.stderr)
        return 2
    srv = servers[srv_name]

    if args.all:
        snap = poll_all(
            [{"name": srv.name, "url": srv.url, "local_gpu": srv.local_gpu}],
            gpu_filter=cfg.gpu_filter,
            query_gpus_fn=query_gpus,
        )[0]
        names = [m.get("name") for m in snap.get("models") or [] if m.get("name")]
        if not names:
            print("No loaded models")
            return 0
        target = ", ".join(names)
        prompt = f"Unload all loaded models on {srv_name}? ({target})"
    else:
        if not args.model:
            print("Provide a model name or use --all", file=sys.stderr)
            return 2
        names = [args.model]
        prompt = f"Unload {args.model} from {srv_name}?"

    if not args.yes:
        ans = input(f"{prompt} [y/N] ").strip().lower()
        if ans != "y":
            return 0

    results = unload_models(srv.url, names)
    exit_code = 0
    for result in results:
        model = result.get("model", "?")
        if result.get("error"):
            print(f"{model}: {result['error']}", file=sys.stderr)
            exit_code = 2
        else:
            reason = result.get("done_reason") or "done"
            print(f"{model}: {reason}")
    return exit_code


def cmd_doctor(args, cfg) -> int:
    servers = {s.name: s for s in cfg.servers}
    srv_name = args.server
    if srv_name not in servers:
        print(f"Unknown server: {srv_name}", file=sys.stderr)
        return 2
    srv = servers[srv_name]
    if not srv.local_gpu:
        print("doctor only supports local_gpu servers", file=sys.stderr)
        return 2

    snap = poll_all(
        [{"name": srv.name, "url": srv.url, "local_gpu": srv.local_gpu}],
        gpu_filter=cfg.gpu_filter,
        query_gpus_fn=query_gpus,
    )[0]
    proc_rows: list[dict[str, Any]] = []
    if _proc_vram_enabled(cfg):
        try:
            proc_rows = query_process_vram(min_bytes=cfg.proc_vram_min_mb * 1024 * 1024)
        except Exception:
            proc_rows = []

    inputs = collect_doctor_inputs()
    findings = run_doctor(
        snap,
        registry=inputs["registry"],
        log_cfg=inputs["log_cfg"],
        log_path=inputs["log_path"],
        runners=inputs["runners"],
        proc_rows=proc_rows,
        ollama_url=cfg.ollama_url,
        registry_mtime=inputs["registry_mtime"],
        restart_remedy=inputs["restart_remedy"],
    )
    exit_code = findings_exit_code(findings)

    if args.json:
        print(
            json.dumps(
                {"findings": [f.to_dict() for f in findings], "exit_code": exit_code},
                indent=2,
            )
        )
    else:
        for f in findings:
            label = f.severity.upper()
            print(f"{label:7} {f.id}")
            print(f"        {f.message}")
            if f.remedy and f.severity in ("warn", "fail"):
                for line in f.remedy.strip().splitlines():
                    print(f"        remedy: {line}")
        needs_restart = any(
            f.id.startswith("config:drift:") and f.severity == "warn"
            or f.id == "config:footgun:stale_env"
            for f in findings
        )
        if needs_restart and inputs.get("restart_remedy"):
            print("\n--- Restart remedy (copy-paste) ---")
            print(inputs["restart_remedy"])

    if args.fix_orphans:
        orphan_pids = [
            int(f.id.rsplit(":", 1)[-1])
            for f in findings
            if f.check == "orphan" and f.severity == "warn" and f.id.startswith("runner:orphan:")
        ]
        if orphan_pids:
            if not args.yes:
                ans = input(f"Kill orphan PIDs {orphan_pids}? [y/N] ").strip().lower()
                if ans != "y":
                    return exit_code
            for row in kill_orphan_pids(orphan_pids):
                if row.get("ok"):
                    print(f"Killed pid {row['pid']}")
                else:
                    print(f"Failed pid {row['pid']}: {row.get('error')}", file=sys.stderr)
                    exit_code = max(exit_code, 2)
        elif not args.json:
            print("No orphan PIDs to fix")

    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = resolve_config(args)

    if args.gui:
        lock = InstanceLock()
        if not lock.try_acquire(InstanceLock.CONTINUOUS):
            lock.request_show()
            return 0
        tray, start_hidden = resolve_gui_options(args)
        from ollama_sentinel.ui import run_gui

        try:
            run_gui(cfg, tray=tray, start_hidden=start_hidden, instance_lock=lock)
        finally:
            lock.release()
        return 0

    if args.command == "search":
        return cmd_search(args, cfg)

    if args.command == "pull":
        return cmd_pull(args, cfg)

    if args.command == "unload":
        return cmd_unload(args, cfg)

    if args.command == "doctor":
        return cmd_doctor(args, cfg)

    state_path = cfg.state_file or Path("state.json")
    state = load_state(state_path)
    once_mode = args.once or args.json or args.list
    proc_collector = _make_proc_collector(cfg) if not once_mode else None

    def _doctor_proc_rows():
        block = _process_vram_payload(proc_collector, cfg, sync=once_mode)
        if not block:
            return None
        return block.get("rows") or []

    def cycle(last_snapshots=None):
        snapshots = _poll(cfg, last_snapshots=last_snapshots)
        # Passive doctor (A+B mapped to alarms) — do not affect --once exit codes below
        findings = _doctor_findings_for_snapshots(
            snapshots, cfg, proc_rows=_doctor_proc_rows()
        )
        active, new_state, transitions = _evaluate_all(
            snapshots, state, cfg.thresholds, doctor_findings=findings
        )
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
            if proc_collector:
                proc_collector.stop()
            return _exit_code(snapshots, active)
        proc_block = _process_vram_payload(proc_collector, cfg, sync=once_mode)
        proc_rows = (proc_block or {}).get("rows") if proc_block else None
        snapshots_out = _attach_activity(snapshots, proc_rows, cfg)
        payload = {
            "snapshots": snapshots_out,
            "alarms": active,
            "inventory": {
                s.get("server"): build_inventory(s) for s in snapshots if s.get("reachable")
            },
        }
        if proc_block is not None:
            payload["process_vram"] = proc_block
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(
                render_snapshot_plain(
                    snapshots_out,
                    active,
                    poll_interval=cfg.poll_interval,
                    process_vram=proc_block,
                    proc_vram_interval=cfg.proc_vram_interval,
                )
            )
        if args.log:
            append_alarm_log(
                Path(args.log),
                alarms=active,
                transitions=transitions,
                on_transition_only=False,
            )
        if proc_collector:
            proc_collector.stop()
        # Exit codes remain spill/paging/vram/unreachable only (ignore doctor alarms)
        resource_active = [
            a for a in active if a.get("type") not in ("config", "orphan")
        ]
        return _exit_code(snapshots, resource_active)

    # Live mode
    lock = InstanceLock()
    if not lock.try_acquire(InstanceLock.CONTINUOUS):
        print("Another ollama-sentinel monitor is already running.", file=sys.stderr)
        return 1
    renderer = LiveRenderer()
    last_by_server: dict[str, dict[str, Any]] = {}
    metrics_store = make_metrics_store(cfg)

    def poll_fn():
        nonlocal state, last_by_server
        snapshots = _poll(cfg, last_snapshots=last_by_server or None)
        for snap in snapshots:
            if snap.get("reachable"):
                last_by_server[snap["server"]] = snap
            if metrics_store is not None:
                metrics_store.ingest_snapshot(snap)
        if metrics_store is not None:
            proc_block = _process_vram_payload(proc_collector, cfg)
            if proc_block and proc_block.get("rows"):
                metrics_store.ingest_proc_vram(
                    proc_block["rows"],
                    ts=proc_block.get("polled_at_ts"),
                )
        findings = _doctor_findings_for_snapshots(
            snapshots, cfg, proc_rows=_doctor_proc_rows()
        )
        active, new_state, transitions = _evaluate_all(
            snapshots, state, cfg.thresholds, doctor_findings=findings
        )
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
        renderer.run(
            poll_fn,
            cfg.poll_interval,
            proc_vram_interval=cfg.proc_vram_interval,
            get_process_vram=lambda: _process_vram_payload(proc_collector, cfg),
        )
    except KeyboardInterrupt:
        pass
    finally:
        if proc_collector:
            proc_collector.stop()
        lock.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
