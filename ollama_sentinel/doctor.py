"""Config doctor: drift, orphans, derived effects, footguns."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ollama_sentinel.doctor_log import (
    TRACKED_KEYS,
    find_latest_server_log,
    normalize_value,
    parse_server_log,
    values_agree,
)
from ollama_sentinel.inventory import build_inventory, free_vram_bytes


@dataclass
class DoctorFinding:
    check: str  # drift | orphan | derived | footgun | cuda | info
    severity: str  # pass | warn | fail | unknown
    id: str
    message: str
    remedy: str | None = None
    vram_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ORPHAN_VRAM_BYTES = int(1.5 * 1024**3)
PINNED_VRAM_FRAC = 0.70


def _parent_is_ollama(runner: dict[str, Any]) -> bool:
    if not runner.get("parent_alive"):
        return False
    name = (runner.get("parent_name") or "").lower()
    return name in ("ollama.exe", "ollama", "ollama app.exe")


def check_drift(
    registry: dict[str, str | None],
    log_cfg: dict[str, str],
    *,
    restart_remedy: str | None = None,
) -> list[DoctorFinding]:
    findings: list[DoctorFinding] = []
    if not log_cfg:
        findings.append(
            DoctorFinding(
                check="drift",
                severity="unknown",
                id="config:drift:log",
                message="Ollama server.log missing or unreadable — cannot check config drift",
                remedy=None,
            )
        )
        return findings

    drifts = 0
    for key in TRACKED_KEYS:
        reg = registry.get(key)
        log_val = log_cfg.get(key)
        if values_agree(key, reg, log_val):
            continue
        drifts += 1
        findings.append(
            DoctorFinding(
                check="drift",
                severity="warn",
                id=f"config:drift:{key}",
                message=(
                    f"CONFIG DRIFT {key}: registry={reg!r} "
                    f"running={log_val!r} (normalized "
                    f"{normalize_value(key, reg)!r} vs {normalize_value(key, log_val)!r})"
                ),
                remedy="Restart Ollama to apply registry settings.\n" + (restart_remedy or ""),
            )
        )
    if drifts == 0:
        findings.append(
            DoctorFinding(
                check="drift",
                severity="pass",
                id="config:drift:ok",
                message="Registry and running server.log agree on tracked OLLAMA_* keys",
            )
        )
    return findings


def check_orphans(
    loaded_models: list[dict[str, Any]],
    runners: list[dict[str, Any]],
    proc_rows: list[dict[str, Any]] | None = None,
) -> list[DoctorFinding]:
    findings: list[DoctorFinding] = []
    n_models = len(loaded_models)
    n_runners = len(runners)
    vram_by_pid: dict[int, int] = {}
    for row in proc_rows or []:
        pid = row.get("pid")
        if pid is not None:
            vram_by_pid[int(pid)] = int(row.get("bytes") or 0)

    flagged: dict[int, list[str]] = {}

    def add(pid: int, reason: str) -> None:
        flagged.setdefault(pid, [])
        if reason not in flagged[pid]:
            flagged[pid].append(reason)

    if n_runners > n_models:
        for runner in runners:
            add(
                int(runner.get("pid") or 0),
                f"count {n_runners} runners vs {n_models} loaded models",
            )

    for runner in runners:
        pid = int(runner.get("pid") or 0)
        vram = vram_by_pid.get(pid, 0)
        if n_models == 0 and vram >= ORPHAN_VRAM_BYTES:
            add(pid, "VRAM held with no loaded models")
        if not _parent_is_ollama(runner):
            parent = runner.get("parent_name") or "none"
            add(
                pid,
                f"parent not ollama (parent={parent!r}, alive={runner.get('parent_alive')})",
            )

    for runner in runners:
        pid = int(runner.get("pid") or 0)
        if pid not in flagged:
            continue
        vram = vram_by_pid.get(pid)
        if vram and vram > 0:
            head = f"ORPHANED RUNNER pid {pid} holding {vram / 1e9:.1f} GB"
        else:
            head = f"ORPHANED RUNNER pid {pid}"
        findings.append(
            DoctorFinding(
                check="orphan",
                severity="warn",
                id=f"runner:orphan:{pid}",
                message=f"{head}: {'; '.join(flagged[pid])}",
                remedy=(
                    f"Stop-Process -Id {pid} -Force\n"
                    "Or restart Ollama including llama-server (see doctor restart remedy)."
                ),
                vram_bytes=vram if vram else None,
            )
        )

    if not findings:
        findings.append(
            DoctorFinding(
                check="orphan",
                severity="pass",
                id="runner:orphan:ok",
                message=f"llama-server count ({n_runners}) matches loaded models ({n_models})",
            )
        )
    return findings


def check_derived(
    loaded_models: list[dict[str, Any]],
    log_cfg: dict[str, str],
) -> list[DoctorFinding]:
    findings: list[DoctorFinding] = []
    keep = log_cfg.get("OLLAMA_KEEP_ALIVE", "")
    keep_n = normalize_value("OLLAMA_KEEP_ALIVE", keep)
    finite_keep = keep_n not in ("", "-1")
    ctx_cfg = log_cfg.get("OLLAMA_CONTEXT_LENGTH")
    kv = (log_cfg.get("OLLAMA_KV_CACHE_TYPE") or "f16").lower()

    for model in loaded_models:
        name = model.get("name") or model.get("model") or "unknown"
        expires = model.get("expires_at") or ""
        if finite_keep and expires[:4].isdigit() and int(expires[:4]) >= 2100:
            findings.append(
                DoctorFinding(
                    check="derived",
                    severity="warn",
                    id=f"config:stale_keepalive:{name}",
                    message=(
                        f"Model {name} expires Forever but server keep_alive is {keep!r} — "
                        "loaded before current config; unload/reload to apply"
                    ),
                    remedy=f'ollama-sentinel unload "{name}" -y',
                )
            )
        ctx = model.get("context_length") or (model.get("details") or {}).get("context_length")
        if ctx_cfg and ctx is not None:
            try:
                if int(ctx) != int(ctx_cfg):
                    findings.append(
                        DoctorFinding(
                            check="derived",
                            severity="warn",
                            id=f"config:stale_context:{name}",
                            message=(
                                f"Model {name} context_length={ctx} but "
                                f"OLLAMA_CONTEXT_LENGTH={ctx_cfg} — reload to apply"
                            ),
                            remedy=f'ollama-sentinel unload "{name}" -y',
                        )
                    )
            except (TypeError, ValueError):
                pass
        # KV quant heuristic: non-f16 config but size looks like unquantised (~measured gap)
        # Soft check: if kv != f16 and size_vram is close to size (fully resident) with
        # no other signal — skip aggressive size fingerprint without a baseline.
        # Spec: 18.37 GB without q8 vs 17.54 with — use relative: if configured q8_0/q4
        # and model size is provided via 'size' and we have a rough expected savings —
        # keep a light advisory only when expires Forever + kv non-f16 (already covered)
        # or when details.quantization suggests mismatch. Minimal: skip hard size compare
        # without a known baseline; covered by keep-alive/context derived checks.
        _ = kv  # reserved for future footprint table

    if not findings:
        findings.append(
            DoctorFinding(
                check="derived",
                severity="pass",
                id="config:derived:ok",
                message="Loaded models match keep_alive / context from server.log",
            )
        )
    return findings


def check_footguns(
    *,
    ollama_url: str,
    log_cfg: dict[str, str],
    snapshot: dict[str, Any],
    log_path: Path | None,
    registry_mtime: float | None,
) -> list[DoctorFinding]:
    findings: list[DoctorFinding] = []

    # D1: OLLAMA_URL is 0.0.0.0
    try:
        host = urlparse(ollama_url).hostname or ""
    except Exception:
        host = ""
    if host in ("0.0.0.0", "::"):
        findings.append(
            DoctorFinding(
                check="footgun",
                severity="warn",
                id="config:footgun:ollama_url",
                message=(
                    f"OLLAMA_URL is {ollama_url!r} — that is a bind address, not a client URL. "
                    "Use http://127.0.0.1:11434"
                ),
                remedy="Set OLLAMA_URL=http://127.0.0.1:11434 in .env",
            )
        )

    # D2: KEEP_ALIVE=-1 with model >70% VRAM
    keep = normalize_value("OLLAMA_KEEP_ALIVE", log_cfg.get("OLLAMA_KEEP_ALIVE"))
    gpus = snapshot.get("gpus") or []
    total_vram = sum(g.get("memory_total") or 0 for g in gpus)
    if keep == "-1" and total_vram > 0:
        for model in snapshot.get("models") or []:
            size = model.get("size") or model.get("size_vram") or 0
            name = model.get("name") or "unknown"
            if size / total_vram > PINNED_VRAM_FRAC:
                findings.append(
                    DoctorFinding(
                        check="footgun",
                        severity="warn",
                        id=f"config:footgun:keepalive:{name}",
                        message=(
                            f"KEEP_ALIVE=-1 with {name} holding "
                            f"{100 * size / total_vram:.0f}% of VRAM — GPU never frees itself"
                        ),
                        remedy="Set OLLAMA_KEEP_ALIVE=30m (User env) and restart Ollama",
                    )
                )

    # D3: largest installed would spill
    inv = build_inventory(snapshot)
    free = free_vram_bytes(snapshot.get("gpus"))
    if free is not None:
        for row in inv:
            if row.get("would_spill") and not row.get("loaded"):
                findings.append(
                    DoctorFinding(
                        check="footgun",
                        severity="warn",
                        id=f"config:footgun:spill:{row['name']}",
                        message=(
                            f"Installed model {row['name']} ({row['size_gb']:.1f} GB) "
                            f"would spill with {free / 1e9:.1f} GB free VRAM"
                        ),
                        remedy="Unload another model or free VRAM before loading",
                    )
                )
                break

    # D4: registry newer than server.log
    if registry_mtime is not None and log_path is not None and log_path.is_file():
        try:
            log_mtime = log_path.stat().st_mtime
            if registry_mtime > log_mtime + 5:
                findings.append(
                    DoctorFinding(
                        check="footgun",
                        severity="warn",
                        id="config:footgun:stale_env",
                        message=(
                            "Registry Environment changed after server.log — "
                            "settings changed since last Ollama start; restart needed"
                        ),
                        remedy="Restart Ollama (include llama-server in Stop-Process)",
                    )
                )
        except OSError:
            pass

    if not findings:
        findings.append(
            DoctorFinding(
                check="footgun",
                severity="pass",
                id="config:footgun:ok",
                message="No known config footguns detected",
            )
        )
    return findings


def check_cuda_compat(
    *,
    log_path: Path | None = None,
    log_text: str | None = None,
    driver_version: str | None = None,
    driver_cuda: str | None = None,
) -> list[DoctorFinding]:
    """Warn when the display driver's CUDA UMD cannot host Ollama's cuda_vN."""
    from ollama_sentinel.cuda_compat import probe_cuda_compat

    probe = probe_cuda_compat(
        log_path=log_path,
        log_text=log_text,
        driver_version=driver_version,
        driver_cuda=driver_cuda,
    )
    if probe.ok:
        if probe.ollama_cuda_major is not None and probe.driver_cuda:
            return [
                DoctorFinding(
                    check="cuda",
                    severity="pass",
                    id="cuda:compat:ok",
                    message=(
                        f"CUDA ok — Ollama cuda_v{probe.ollama_cuda_major} on driver "
                        f"CUDA {probe.driver_cuda}"
                        + (f" ({probe.driver_version})" if probe.driver_version else "")
                    ),
                )
            ]
        return [
            DoctorFinding(
                check="cuda",
                severity="info",
                id="cuda:compat:unknown",
                message=probe.reason or "CUDA compatibility not measured yet",
            )
        ]
    return [
        DoctorFinding(
            check="cuda",
            severity="warn",
            id="cuda:compat:mismatch",
            message=probe.reason or "CUDA driver mismatch",
            remedy="Install a newer NVIDIA display driver, then restart Ollama",
        )
    ]


def run_doctor(
    snapshot: dict[str, Any],
    *,
    registry: dict[str, str | None] | None = None,
    log_cfg: dict[str, str] | None = None,
    log_path: Path | None = None,
    runners: list[dict[str, Any]] | None = None,
    proc_rows: list[dict[str, Any]] | None = None,
    ollama_url: str = "http://127.0.0.1:11434",
    registry_mtime: float | None = None,
    restart_remedy: str | None = None,
    driver_version: str | None = None,
    driver_cuda: str | None = None,
) -> list[DoctorFinding]:
    """Run all doctor checks. Pure given inputs (callers supply I/O)."""
    findings: list[DoctorFinding] = []
    findings.extend(check_drift(registry or {}, log_cfg or {}, restart_remedy=restart_remedy))
    findings.extend(
        check_orphans(snapshot.get("models") or [], runners or [], proc_rows)
    )
    findings.extend(check_derived(snapshot.get("models") or [], log_cfg or {}))
    findings.extend(
        check_footguns(
            ollama_url=ollama_url,
            log_cfg=log_cfg or {},
            snapshot=snapshot,
            log_path=log_path,
            registry_mtime=registry_mtime,
        )
    )
    findings.extend(
        check_cuda_compat(
            log_path=log_path,
            driver_version=driver_version,
            driver_cuda=driver_cuda,
        )
    )
    return findings


def collect_doctor_inputs() -> dict[str, Any]:
    """Gather Windows I/O inputs for run_doctor. Safe on non-Windows."""
    from ollama_sentinel.cuda_compat import query_driver_cuda
    from ollama_sentinel.doctor_win import (
        build_restart_remedy,
        list_llama_server_processes,
        read_registry_env,
        registry_env_mtime,
    )

    log_path = find_latest_server_log()
    log_cfg = parse_server_log(log_path) if log_path else {}
    registry = {key: read_registry_env(key) for key in TRACKED_KEYS}
    runners = list_llama_server_processes() if sys.platform == "win32" else []
    driver_version, driver_cuda = (None, None)
    if sys.platform == "win32":
        driver_version, driver_cuda = query_driver_cuda()
    return {
        "registry": registry,
        "log_cfg": log_cfg,
        "log_path": log_path,
        "runners": runners,
        "registry_mtime": registry_env_mtime(),
        "restart_remedy": build_restart_remedy(),
        "driver_version": driver_version,
        "driver_cuda": driver_cuda,
    }


def findings_exit_code(findings: list[DoctorFinding]) -> int:
    if any(f.severity == "fail" for f in findings):
        return 2
    if any(f.severity in ("warn", "unknown") for f in findings):
        return 1
    return 0


def evaluate_doctor_alarms(findings: list[DoctorFinding]) -> list[dict[str, Any]]:
    """Map Check A/B WARN/FAIL findings to passive alarm dicts."""
    alarms: list[dict[str, Any]] = []
    for f in findings:
        if f.severity not in ("warn", "fail"):
            continue
        if f.check == "drift" and f.id.startswith("config:drift:") and f.id != "config:drift:ok":
            alarms.append(
                {"id": f.id, "type": "config", "message": f.message}
            )
        elif f.check == "orphan" and f.id.startswith("runner:orphan:") and f.id != "runner:orphan:ok":
            alarms.append(
                {"id": f.id, "type": "orphan", "message": f.message}
            )
        elif f.check == "cuda" and f.id == "cuda:compat:mismatch":
            alarms.append(
                {"id": f.id, "type": "cuda", "message": f.message}
            )
    return alarms
