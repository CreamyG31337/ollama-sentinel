"""Infer current Ollama processing state from server.log and GPU util."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ollama_sentinel.doctor_log import find_latest_server_log
from ollama_sentinel.gaming import is_ollama_busy

_GIN = re.compile(
    r'\[GIN\]\s+(\d{4}/\d{2}/\d{2} - \d{2}:\d{2}:\d{2})\s+\|\s+(\d+)\s+\|\s+(\S+)\s+\|\s+(\S+)\s+\|\s+(POST|GET)\s+"([^"]+)"'
)
_SLOT_TASK = re.compile(r"slot \S+: id\s+(\d+) \| task (\d+) \| processing task")
_PROMPT_TIMING = re.compile(
    r"print_timing: id\s+(\d+) \| task (\d+) \| prompt processing, "
    r"n_tokens =\s*(\d+), progress = ([\d.]+), t =\s*([\d.]+) s / ([\d.]+) tokens per second"
)
_GENERATION_TIMING = re.compile(
    r"print_timing: id\s+(\d+) \| task (\d+) \| (?:acc per pos|draft acceptance)"
)
_N_GEN = re.compile(
    r"print_timing: id\s+(\d+) \| task (\d+) \| n_gen =\s*(\d+), "
    r"tg =\s*([\d.]+) t/s(?:, tg_3s =\s*([\d.]+) t/s)?"
)
_NEW_PROMPT = re.compile(
    r"task\s+(\d+)\s*\|\s*new prompt.*?"
    r"n_ctx_slot\s*=\s*(\d+).*?"
    r"task\.n_tokens\s*=\s*(\d+)"
)
_RELEASE = re.compile(
    r"task\s+(\d+)\s*\|\s*stop processing:\s*"
    r"n_tokens\s*=\s*(\d+),\s*truncated\s*=\s*([01])"
)
_ABORT = "aborting completion request due to client closing the connection"

# Sentinel / health-check noise that drowns real chat traffic in GIN lines.
_MONITOR_GET_PATHS = frozenset(
    {
        "/api/ps",
        "/api/tags",
        "/api/version",
        "/api/v1/models",
    }
)
_INFERENCE_MARKERS = ("/chat", "/generate", "/embed", "/completions")


def _parse_gin_time(text: str) -> datetime | None:
    try:
        return datetime.strptime(text, "%Y/%m/%d - %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _read_log_tail(path: Path, *, max_bytes: int = 512_000) -> list[str]:
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                fh.readline()  # drop partial line
            data = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return data.splitlines()


def path_only(path: str) -> str:
    return (path or "").split("?", 1)[0]


def is_monitor_request(method: str, path: str) -> bool:
    p = path_only(path)
    if method == "GET" and p in _MONITOR_GET_PATHS:
        return True
    if method == "POST" and p == "/api/show":
        return True
    return False


def is_inference_request(method: str, path: str) -> bool:
    if method != "POST":
        return False
    p = path_only(path)
    return any(marker in p for marker in _INFERENCE_MARKERS)


def build_peer_name_map(clients: list[dict[str, Any]] | None) -> dict[str, str]:
    """Map client IP / host → friendly name from clients.json `addrs`."""
    out: dict[str, str] = {}
    for client in clients or []:
        name = str(client.get("name") or "").strip()
        if not name:
            continue
        for addr in client.get("addrs") or []:
            key = str(addr).strip()
            if key:
                out[key] = name
    return out


def format_peer(addr: str, names: dict[str, str] | None = None) -> str:
    names = names or {}
    label = names.get(addr)
    if label:
        return f"{label} ({addr})"
    return addr


@dataclass
class ActivityRequest:
    at: str
    method: str
    path: str
    status: int
    duration_s: float | None
    client: str
    client_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def client_label(self) -> str:
        if self.client_name:
            return f"{self.client_name} ({self.client})"
        return self.client


@dataclass
class RunnerActivity:
    pid: int
    name: str
    vram_bytes: int
    engine_3d_pct: float
    busy: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActivityPeer:
    addr: str
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def label(self) -> str:
        return format_peer(self.addr, {self.addr: self.name} if self.name else None)


@dataclass
class ServerActivity:
    phase: str  # idle | prompt | generating | embed | request
    summary: str
    slot_id: int | None = None
    task_id: int | None = None
    prompt_tokens: int | None = None
    prompt_progress: float | None = None
    prompt_tps: float | None = None
    n_ctx_slot: int | None = None
    ctx_fill: float | None = None
    n_gen: int | None = None
    gen_tps: float | None = None
    gen_tps_3s: float | None = None
    model: str | None = None
    aborted: bool = False
    last_request: ActivityRequest | None = None
    recent_requests: list[ActivityRequest] = field(default_factory=list)
    peers: list[ActivityPeer] = field(default_factory=list)
    runners: list[RunnerActivity] = field(default_factory=list)
    log_path: str | None = None
    stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        if self.last_request:
            out["last_request"] = self.last_request.to_dict()
        out["recent_requests"] = [r.to_dict() for r in self.recent_requests]
        out["peers"] = [p.to_dict() for p in self.peers]
        out["runners"] = [r.to_dict() for r in self.runners]
        return out


@dataclass
class ParsedLogActivity:
    """Raw parse result from a server.log tail."""

    requests: list[ActivityRequest]
    inference_requests: list[ActivityRequest]
    last_task: dict[str, Any] | None = None
    last_prompt: dict[str, Any] | None = None
    last_generation: dict[str, Any] | None = None
    last_ngen: dict[str, Any] | None = None
    last_new_prompt: dict[str, Any] | None = None
    last_release: dict[str, Any] | None = None
    aborted: bool = False
    # Task still open (saw work after its last release), if any.
    open_task_id: int | None = None
    open_slot_id: int | None = None
    open_phase: str = "idle"  # idle | prompt | generating


def _llama_rows(proc_rows: list[dict[str, Any]] | None) -> list[RunnerActivity]:
    out: list[RunnerActivity] = []
    for row in proc_rows or []:
        name = (row.get("name") or "").lower()
        if "llama-server" not in name:
            continue
        util = float(row.get("engine_3d_pct") or 0.0)
        out.append(
            RunnerActivity(
                pid=int(row.get("pid") or 0),
                name=row.get("name") or "llama-server",
                vram_bytes=int(row.get("bytes") or 0),
                engine_3d_pct=util,
                busy=is_ollama_busy(util, busy_util=5.0),
            )
        )
    out.sort(key=lambda r: (r.busy, r.vram_bytes), reverse=True)
    return out


def _parse_duration(dur_raw: str) -> float | None:
    try:
        if dur_raw.endswith("ms"):
            return float(dur_raw[:-2].strip()) / 1000.0
        if dur_raw.endswith("s"):
            return float(dur_raw[:-1].strip())
        return float(dur_raw)
    except ValueError:
        return None


def parse_server_log_activity(
    lines: list[str],
    *,
    now: datetime | None = None,
    fresh_seconds: float = 45.0,
    peer_names: dict[str, str] | None = None,
) -> ParsedLogActivity:
    """Parse GIN + llama.cpp slot lines into activity signals."""
    now_dt = now or datetime.now(timezone.utc)
    names = peer_names or {}
    requests: list[ActivityRequest] = []
    last_task: dict[str, Any] | None = None
    last_prompt: dict[str, Any] | None = None
    last_generation: dict[str, Any] | None = None
    last_ngen: dict[str, Any] | None = None
    last_new_prompt: dict[str, Any] | None = None
    last_release: dict[str, Any] | None = None
    aborted = False

    open_task_id: int | None = None
    open_slot_id: int | None = None
    open_phase = "idle"

    for line in lines:
        if _ABORT in line:
            aborted = True
            if open_phase != "idle":
                open_phase = "idle"
                open_task_id = None
                open_slot_id = None
            continue

        m = _GIN.search(line)
        if m:
            client = m.group(4)
            requests.append(
                ActivityRequest(
                    at=m.group(1),
                    method=m.group(5),
                    path=m.group(6),
                    status=int(m.group(2)),
                    duration_s=_parse_duration(m.group(3).strip()),
                    client=client,
                    client_name=names.get(client),
                )
            )
            continue

        m = _SLOT_TASK.search(line)
        if m:
            slot_id, task_id = int(m.group(1)), int(m.group(2))
            last_task = {"slot_id": slot_id, "task_id": task_id, "line": line}
            open_task_id = task_id
            open_slot_id = slot_id
            if open_phase == "idle":
                open_phase = "prompt"
            continue

        m = _NEW_PROMPT.search(line)
        if m:
            task_id = int(m.group(1))
            n_ctx = int(m.group(2))
            n_tokens = int(m.group(3))
            last_new_prompt = {
                "task_id": task_id,
                "n_ctx_slot": n_ctx,
                "tokens": n_tokens,
                "line": line,
            }
            open_task_id = task_id
            if open_phase == "idle":
                open_phase = "prompt"
            continue

        m = _PROMPT_TIMING.search(line)
        if m:
            slot_id, task_id = int(m.group(1)), int(m.group(2))
            progress = float(m.group(4))
            last_prompt = {
                "slot_id": slot_id,
                "task_id": task_id,
                "tokens": int(m.group(3)),
                "progress": progress,
                "elapsed_s": float(m.group(5)),
                "tps": float(m.group(6)),
                "line": line,
            }
            open_task_id = task_id
            open_slot_id = slot_id
            open_phase = "prompt" if progress < 1.0 else "generating"
            continue

        m = _N_GEN.search(line)
        if m:
            slot_id, task_id = int(m.group(1)), int(m.group(2))
            tg_3s = float(m.group(5)) if m.group(5) is not None else None
            last_ngen = {
                "slot_id": slot_id,
                "task_id": task_id,
                "n_gen": int(m.group(3)),
                "tg": float(m.group(4)),
                "tg_3s": tg_3s,
                "line": line,
            }
            open_task_id = task_id
            open_slot_id = slot_id
            open_phase = "generating"
            continue

        m = _GENERATION_TIMING.search(line)
        if m:
            slot_id, task_id = int(m.group(1)), int(m.group(2))
            last_generation = {
                "slot_id": slot_id,
                "task_id": task_id,
                "line": line,
            }
            open_task_id = task_id
            open_slot_id = slot_id
            open_phase = "generating"
            continue

        m = _RELEASE.search(line)
        if m:
            task_id = int(m.group(1))
            last_release = {
                "task_id": task_id,
                "n_tokens": int(m.group(2)),
                "truncated": m.group(3) == "1",
                "line": line,
            }
            if open_task_id == task_id:
                open_phase = "idle"
                open_task_id = None
                open_slot_id = None

    def _age_seconds(at_str: str) -> float | None:
        ts = _parse_gin_time(at_str)
        if ts is None:
            return None
        return (now_dt - ts).total_seconds()

    fresh: list[ActivityRequest] = []
    for req in requests:
        age = _age_seconds(req.at)
        if age is not None and age <= fresh_seconds:
            fresh.append(req)

    inference = [r for r in fresh if is_inference_request(r.method, r.path)]

    return ParsedLogActivity(
        requests=fresh[-8:],
        inference_requests=inference[-8:],
        last_task=last_task,
        last_prompt=last_prompt,
        last_generation=last_generation,
        last_ngen=last_ngen,
        last_new_prompt=last_new_prompt,
        last_release=last_release,
        aborted=aborted,
        open_task_id=open_task_id,
        open_slot_id=open_slot_id,
        open_phase=open_phase,
    )


def listen_port_from_url(url: str | None, default: int = 11434) -> int:
    if not url:
        return default
    try:
        port = urlparse(url).port
        return int(port) if port else default
    except (TypeError, ValueError):
        return default


def parse_ss_peers(text: str) -> list[str]:
    """Parse `ss -H -tn state established '( sport = :PORT )'` peers."""
    peers: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        parts = line.split()
        # ss: Netid State Recv-Q Send-Q Local Address:Port Peer Address:Port
        if len(parts) < 6:
            continue
        peer = parts[-1]
        if peer.startswith("["):
            # [::1]:54321
            end = peer.rfind("]:")
            addr = peer[1:end] if end > 0 else peer
        else:
            addr = peer.rsplit(":", 1)[0]
        if addr in ("*", "0.0.0.0", "::") or addr in seen:
            continue
        seen.add(addr)
        peers.append(addr)
    return peers


def parse_powershell_peers(text: str) -> list[str]:
    peers: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        addr = line.strip()
        if not addr or addr.lower() in ("remoteaddress", "---"):
            continue
        if addr in seen:
            continue
        seen.add(addr)
        peers.append(addr)
    return peers


def list_tcp_peers(
    port: int = 11434,
    *,
    peers: list[str] | None = None,
) -> list[str]:
    """Return remote addresses with ESTABLISHED connections to local `port`.

    `peers` injects a precomputed list for tests (skips the OS query).
    """
    if peers is not None:
        return list(peers)

    try:
        if sys.platform == "win32":
            from ollama_sentinel.smi import _no_window

            cmd = (
                f"Get-NetTCPConnection -LocalPort {int(port)} -State Established "
                "-ErrorAction SilentlyContinue | Select-Object -ExpandProperty RemoteAddress"
            )
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=5,
                **_no_window(),
            )
            if proc.returncode != 0:
                return []
            return parse_powershell_peers(proc.stdout or "")

        proc = subprocess.run(
            ["ss", "-H", "-tn", "state", "established", f"( sport = :{int(port)} )"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return []
        return parse_ss_peers(proc.stdout or "")
    except (OSError, subprocess.SubprocessError):
        return []


def model_name_from_ps(models: list[dict[str, Any]] | None) -> str | None:
    if not models:
        return None
    for m in models:
        name = m.get("name") or m.get("model")
        if name:
            return str(name)
    return None


def build_server_activity(
    *,
    log_path: Path | None = None,
    proc_rows: list[dict[str, Any]] | None = None,
    now: datetime | None = None,
    fresh_seconds: float = 45.0,
    model: str | None = None,
    models: list[dict[str, Any]] | None = None,
    peer_names: dict[str, str] | None = None,
    listen_port: int = 11434,
    tcp_peers: list[str] | None = None,
    include_peers: bool = True,
) -> ServerActivity:
    path = log_path or find_latest_server_log()
    runners = _llama_rows(proc_rows)
    any_busy = any(r.busy for r in runners)
    model = model or model_name_from_ps(models)
    names = peer_names or {}

    peers: list[ActivityPeer] = []
    if include_peers:
        for addr in list_tcp_peers(listen_port, peers=tcp_peers):
            peers.append(ActivityPeer(addr=addr, name=names.get(addr)))

    if path is None or not path.is_file():
        summary = "No server.log — activity unknown"
        phase = "idle"
        if any_busy:
            busy = next(r for r in runners if r.busy)
            phase = "generating"
            summary = f"Generating (pid {busy.pid} @ {busy.engine_3d_pct:.0f}% GPU util)"
        elif runners:
            summary = "Idle (models loaded)"
        else:
            summary = "Idle"
        if model and phase != "idle":
            summary = f"{model}: {summary}"
        return ServerActivity(
            phase=phase,
            summary=summary,
            model=model,
            peers=peers,
            runners=runners,
        )

    lines = _read_log_tail(path)
    parsed = parse_server_log_activity(
        lines, now=now, fresh_seconds=fresh_seconds, peer_names=names
    )
    inference = parsed.inference_requests
    last_api = inference[-1] if inference else None

    phase = "idle"
    summary = "Idle"
    slot_id = task_id = prompt_tokens = None
    prompt_progress = prompt_tps = None
    n_ctx_slot = ctx_fill = None
    n_gen = gen_tps = gen_tps_3s = None
    stale = False

    new_p = parsed.last_new_prompt
    if new_p and (
        parsed.open_task_id == new_p["task_id"]
        or (
            parsed.last_ngen
            and parsed.last_ngen["task_id"] == new_p["task_id"]
            and parsed.open_phase != "idle"
        )
        or (
            parsed.last_prompt
            and parsed.last_prompt["task_id"] == new_p["task_id"]
            and parsed.open_phase != "idle"
        )
    ):
        n_ctx_slot = new_p["n_ctx_slot"]
        prompt_tokens = new_p["tokens"]
        if n_ctx_slot:
            ctx_fill = prompt_tokens / n_ctx_slot

    if parsed.open_phase == "generating" and parsed.open_task_id is not None:
        phase = "generating"
        task_id = parsed.open_task_id
        slot_id = parsed.open_slot_id
        if parsed.last_ngen and parsed.last_ngen["task_id"] == task_id:
            n_gen = parsed.last_ngen["n_gen"]
            gen_tps = parsed.last_ngen["tg"]
            gen_tps_3s = parsed.last_ngen.get("tg_3s")
        util = runners[0].engine_3d_pct if runners else 0.0
        parts = [f"Generating tokens (task {task_id}"]
        if n_gen is not None:
            parts[0] = f"Generating {n_gen:,} tokens (task {task_id}"
        if gen_tps is not None:
            parts.append(f"{gen_tps:.0f} tok/s")
        if util > 0:
            parts.append(f"{util:.0f}% GPU util")
        summary = ", ".join(parts) + ")"
    elif parsed.open_phase == "prompt" and parsed.open_task_id is not None:
        phase = "prompt"
        task_id = parsed.open_task_id
        slot_id = parsed.open_slot_id
        if parsed.last_prompt and parsed.last_prompt["task_id"] == task_id:
            prompt_tokens = parsed.last_prompt["tokens"]
            prompt_progress = parsed.last_prompt["progress"]
            prompt_tps = parsed.last_prompt["tps"]
            summary = (
                f"Processing prompt {prompt_progress * 100:.0f}% "
                f"({prompt_tokens} tokens @ {prompt_tps:.0f} tok/s)"
            )
        elif new_p and new_p["task_id"] == task_id:
            if n_ctx_slot and prompt_tokens is not None:
                summary = f"Prompt loaded ({prompt_tokens:,}/{n_ctx_slot:,} ctx)"
            elif prompt_tokens is not None:
                summary = f"Prompt loaded ({prompt_tokens:,} tokens)"
            else:
                summary = f"Processing prompt (task {task_id})"
        else:
            summary = f"Processing prompt (task {task_id})"
    elif last_api:
        if "/embed" in last_api.path:
            phase = "embed"
            summary = f"Embedding via {last_api.path}"
        else:
            phase = "request"
            summary = f"API {last_api.method} {last_api.path}"
        if last_api.client_label:
            summary += f" from {last_api.client_label}"
    elif any_busy:
        phase = "generating"
        busy = next(r for r in runners if r.busy)
        summary = f"Generating (pid {busy.pid} @ {busy.engine_3d_pct:.0f}% GPU util)"
    elif runners:
        summary = "Idle (models loaded)"
    else:
        summary = "Idle"

    if phase == "idle" and any_busy:
        phase = "generating"
        busy = next(r for r in runners if r.busy)
        summary = f"Generating (pid {busy.pid} @ {busy.engine_3d_pct:.0f}% GPU util)"
        stale = bool(parsed.last_prompt or parsed.last_generation or parsed.last_ngen)

    if parsed.aborted and phase == "idle":
        summary = "Idle (last request aborted by client)"

    if model and phase != "idle":
        summary = f"{model}: {summary}"

    # Prefer ctx from new_prompt even when generating.
    if n_ctx_slot is None and new_p:
        # Keep last prompt size as context for the completed/current task display
        # when we still have matching open/last ngen task.
        if task_id is None or new_p["task_id"] == task_id:
            n_ctx_slot = new_p["n_ctx_slot"]
            if prompt_tokens is None:
                prompt_tokens = new_p["tokens"]
            if n_ctx_slot and prompt_tokens is not None:
                ctx_fill = prompt_tokens / n_ctx_slot

    return ServerActivity(
        phase=phase,
        summary=summary,
        slot_id=slot_id,
        task_id=task_id,
        prompt_tokens=prompt_tokens,
        prompt_progress=prompt_progress,
        prompt_tps=prompt_tps,
        n_ctx_slot=n_ctx_slot,
        ctx_fill=ctx_fill,
        n_gen=n_gen,
        gen_tps=gen_tps,
        gen_tps_3s=gen_tps_3s,
        model=model,
        aborted=parsed.aborted and phase != "generating",
        last_request=last_api,
        recent_requests=inference,
        peers=peers,
        runners=runners,
        log_path=str(path),
        stale=stale,
    )


def model_detail_line(model: dict[str, Any]) -> str:
    """Short context/quant summary for a loaded model row."""
    parts: list[str] = []
    ctx = model.get("context_length")
    if ctx:
        parts.append(f"ctx {ctx:,}")
    details = model.get("details") or {}
    quant = details.get("quantization_level")
    if quant:
        parts.append(str(quant))
    fam = details.get("family")
    if fam:
        parts.append(str(fam))
    params = details.get("parameter_size")
    if params:
        parts.append(str(params))
    return " · ".join(parts) if parts else "—"
