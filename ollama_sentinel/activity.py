"""Infer current Ollama processing state from server.log and GPU util."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


@dataclass
class ActivityRequest:
    at: str
    method: str
    path: str
    status: int
    duration_s: float | None
    client: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
class ServerActivity:
    phase: str  # idle | prompt | generating | embed | request
    summary: str
    slot_id: int | None = None
    task_id: int | None = None
    prompt_tokens: int | None = None
    prompt_progress: float | None = None
    prompt_tps: float | None = None
    last_request: ActivityRequest | None = None
    recent_requests: list[ActivityRequest] = field(default_factory=list)
    runners: list[RunnerActivity] = field(default_factory=list)
    log_path: str | None = None
    stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        if self.last_request:
            out["last_request"] = self.last_request.to_dict()
        out["recent_requests"] = [r.to_dict() for r in self.recent_requests]
        out["runners"] = [r.to_dict() for r in self.runners]
        return out


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


def parse_server_log_activity(
    lines: list[str],
    *,
    now: datetime | None = None,
    fresh_seconds: float = 45.0,
) -> tuple[
    list[ActivityRequest],
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Return recent API requests, latest slot task, latest timing line."""
    now_dt = now or datetime.now(timezone.utc)
    requests: list[ActivityRequest] = []
    last_task: dict[str, Any] | None = None
    last_prompt: dict[str, Any] | None = None
    last_generation: dict[str, Any] | None = None

    for line in lines:
        m = _GIN.search(line)
        if m:
            ts = _parse_gin_time(m.group(1))
            dur_raw = m.group(3).strip()
            duration: float | None
            try:
                if dur_raw.endswith("ms"):
                    duration = float(dur_raw[:-2].strip()) / 1000.0
                elif dur_raw.endswith("s"):
                    duration = float(dur_raw[:-1].strip())
                else:
                    duration = float(dur_raw)
            except ValueError:
                duration = None
            requests.append(
                ActivityRequest(
                    at=m.group(1),
                    method=m.group(5),
                    path=m.group(6),
                    status=int(m.group(2)),
                    duration_s=duration,
                    client=m.group(4),
                )
            )
            continue
        m = _SLOT_TASK.search(line)
        if m:
            last_task = {"slot_id": int(m.group(1)), "task_id": int(m.group(2)), "line": line}
            continue
        m = _PROMPT_TIMING.search(line)
        if m:
            last_prompt = {
                "slot_id": int(m.group(1)),
                "task_id": int(m.group(2)),
                "tokens": int(m.group(3)),
                "progress": float(m.group(4)),
                "elapsed_s": float(m.group(5)),
                "tps": float(m.group(6)),
                "line": line,
            }
            continue
        m = _GENERATION_TIMING.search(line)
        if m:
            last_generation = {
                "slot_id": int(m.group(1)),
                "task_id": int(m.group(2)),
                "line": line,
            }

    def _age_seconds(at_str: str) -> float | None:
        ts = _parse_gin_time(at_str)
        if ts is None:
            return None
        return (now_dt - ts).total_seconds()

    fresh_requests = []
    for req in requests:
        age = _age_seconds(req.at)
        if age is not None and age <= fresh_seconds:
            fresh_requests.append(req)

    return fresh_requests[-8:], last_task, last_prompt, last_generation


def build_server_activity(
    *,
    log_path: Path | None = None,
    proc_rows: list[dict[str, Any]] | None = None,
    now: datetime | None = None,
    fresh_seconds: float = 45.0,
) -> ServerActivity:
    path = log_path or find_latest_server_log()
    runners = _llama_rows(proc_rows)
    any_busy = any(r.busy for r in runners)

    if path is None or not path.is_file():
        summary = "No server.log — activity unknown"
        if any_busy:
            busy = next(r for r in runners if r.busy)
            summary = f"Generating (pid {busy.pid} @ {busy.engine_3d_pct:.0f}% GPU util)"
        elif runners:
            summary = "Idle (models loaded)"
        else:
            summary = "Idle"
        return ServerActivity(phase="idle" if not any_busy else "generating", summary=summary, runners=runners)

    lines = _read_log_tail(path)
    recent, last_task, last_prompt, last_generation = parse_server_log_activity(
        lines, now=now, fresh_seconds=fresh_seconds
    )
    last_api = recent[-1] if recent else None

    phase = "idle"
    summary = "Idle"
    slot_id = task_id = prompt_tokens = None
    prompt_progress = prompt_tps = None
    stale = False

    # Prefer log timing when task is fresh; GPU util corroborates.
    if last_generation and last_task and last_generation.get("task_id") == last_task.get("task_id"):
        phase = "generating"
        slot_id = last_generation.get("slot_id")
        task_id = last_generation.get("task_id")
        util = runners[0].engine_3d_pct if runners else 0.0
        summary = f"Generating tokens (task {task_id}"
        if util > 0:
            summary += f", {util:.0f}% GPU util"
        summary += ")"
    elif last_prompt and last_task and last_prompt.get("task_id") == last_task.get("task_id"):
        progress = last_prompt["progress"]
        if progress < 1.0:
            phase = "prompt"
            slot_id = last_prompt["slot_id"]
            task_id = last_prompt["task_id"]
            prompt_tokens = last_prompt["tokens"]
            prompt_progress = progress
            prompt_tps = last_prompt["tps"]
            summary = (
                f"Processing prompt {progress * 100:.0f}% "
                f"({prompt_tokens} tokens @ {prompt_tps:.0f} tok/s)"
            )
        else:
            phase = "generating"
            task_id = last_prompt["task_id"]
            summary = f"Generating tokens (task {task_id})"
    elif last_api and last_api.method == "POST":
        if "/embed" in last_api.path:
            phase = "embed"
            summary = f"Embedding via {last_api.path}"
        elif "/generate" in last_api.path or "/chat" in last_api.path:
            phase = "request"
            summary = f"API {last_api.method} {last_api.path}"
        else:
            phase = "request"
            summary = f"API {last_api.method} {last_api.path}"
    elif any_busy:
        phase = "generating"
        busy = next(r for r in runners if r.busy)
        summary = f"Generating (pid {busy.pid} @ {busy.engine_3d_pct:.0f}% GPU util)"
    elif runners:
        summary = "Idle (models loaded)"
    else:
        summary = "Idle"

    # GPU busy without fresh log lines — still show generating.
    if phase == "idle" and any_busy:
        phase = "generating"
        busy = next(r for r in runners if r.busy)
        summary = f"Generating (pid {busy.pid} @ {busy.engine_3d_pct:.0f}% GPU util)"
        stale = bool(last_prompt or last_generation)

    return ServerActivity(
        phase=phase,
        summary=summary,
        slot_id=slot_id,
        task_id=task_id,
        prompt_tokens=prompt_tokens,
        prompt_progress=prompt_progress,
        prompt_tps=prompt_tps,
        last_request=last_api,
        recent_requests=recent,
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
