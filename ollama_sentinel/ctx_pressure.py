"""Context-window pressure from Ollama server.log runtime lines (read-only).

`doctor_log` parses the *config* the server booted with. This module parses what
actually happened per request, because a correct config still truncates when a
client believes the window is bigger than the one being served.

Two authoritative signals, both emitted by llama.cpp itself:

* ``new prompt, n_ctx_slot = C, ..., task.n_tokens = P`` — the prompt occupies P
  of C tokens, so the answer gets at most ``C - P``. This fires *before*
  generation, which is what makes prevention possible.
* ``stop processing: n_tokens = N, truncated = 1`` — a generation hit the wall.
  Ground truth, not a heuristic.

The failure this was written for (2026-09-04): a client cached the model's
*architectural* context (262144) instead of the *served* window
(``OLLAMA_CONTEXT_LENGTH=65536``), so its compressor waited for a token count the
server could never reach. Prompts arrived at 65303, 65358, 65409, 65460, 65506 —
each retry appending the partial answer and climbing further into the wall.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# `slot   operator(): id  0 | task 147 | new prompt, n_ctx_slot = 65536, n_keep = 4, task.n_tokens = 16428`
_PROMPT_RE = re.compile(
    r"task\s+(?P<task>\d+)\s*\|\s*new prompt.*?"
    r"n_ctx_slot\s*=\s*(?P<slot>\d+).*?"
    r"task\.n_tokens\s*=\s*(?P<tokens>\d+)"
)

# `slot      release: id  0 | task 1731 | stop processing: n_tokens = 65535, truncated = 1`
_RELEASE_RE = re.compile(
    r"task\s+(?P<task>\d+)\s*\|\s*stop processing:\s*"
    r"n_tokens\s*=\s*(?P<tokens>\d+),\s*truncated\s*=\s*(?P<truncated>[01])"
)

_KV_SHIFT_DISABLED = "KV cache shifting is not supported"

# A prompt occupying this much of the window leaves too little room to answer.
WARN_FILL = 0.90
CRITICAL_FILL = 0.98
# Consecutive climbing near-ceiling prompts that mean "a client is retrying into the wall".
LADDER_MIN = 3
# How many recent requests count as "now". A resolved incident stays in the log
# for as long as the file is kept, and an alarm that cannot clear is one people
# learn to ignore.
RECENT_REQUESTS = 40

_RUNNER_START = "load_model: initializing"


@dataclass(frozen=True)
class PromptEvent:
    task: int
    n_ctx_slot: int
    n_tokens: int

    @property
    def fill(self) -> float:
        if self.n_ctx_slot <= 0:
            return 0.0
        return self.n_tokens / self.n_ctx_slot

    @property
    def headroom(self) -> int:
        return max(0, self.n_ctx_slot - self.n_tokens)


@dataclass(frozen=True)
class ReleaseEvent:
    task: int
    n_tokens: int
    truncated: bool


@dataclass
class CtxPressureReport:
    n_ctx_slot: int | None = None
    prompts: list[PromptEvent] = field(default_factory=list)
    truncated_tasks: list[int] = field(default_factory=list)
    kv_shift_disabled: bool = False
    ladder: list[PromptEvent] = field(default_factory=list)

    @property
    def worst(self) -> PromptEvent | None:
        return max(self.prompts, key=lambda p: p.fill) if self.prompts else None

    @property
    def truncated_count(self) -> int:
        return len(self.truncated_tasks)


def read_tail(path: Path, max_bytes: int = 2_000_000) -> str:
    """Last `max_bytes` of a log, decoded leniently.

    server.log grows without bound; only recent requests describe the state the
    user is actually in. Reading the whole file would also let a long-resolved
    incident keep firing an alarm forever.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                fh.readline()  # discard the partial first line
            raw = fh.read()
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace")


def find_ladder(prompts: list[PromptEvent], *, min_fill: float = CRITICAL_FILL,
                min_len: int = LADDER_MIN) -> list[PromptEvent]:
    """Longest run of consecutive near-full prompts with strictly growing size.

    This is the fingerprint of a client's retry loop: each attempt appends the
    partial response and re-sends, so the prompt grows *toward* the ceiling
    instead of backing off. Distinguishing it from a single large prompt matters
    because the remedy is different — the client's continuation logic is making
    things worse, not just its context accounting.
    """
    best: list[PromptEvent] = []
    run: list[PromptEvent] = []
    for ev in prompts:
        if ev.fill < min_fill:
            run = []
            continue
        if run and ev.n_tokens <= run[-1].n_tokens:
            run = [ev]
        else:
            run.append(ev)
        if len(run) > len(best):
            best = list(run)
    return best if len(best) >= min_len else []


def parse_ctx_pressure(text: str, *, recent: int = RECENT_REQUESTS) -> CtxPressureReport:
    """Extract per-request context pressure from recent server.log activity.

    Scoped twice, both to keep a fixed alarm from firing forever:

    * to the current runner — task numbers restart at 0 when a model reloads, so
      comparing across that boundary would mix unrelated requests;
    * to the last `recent` requests, so yesterday's resolved truncation stops
      counting once normal traffic has moved past it.
    """
    kv_shift = _KV_SHIFT_DISABLED in text
    marker = text.rfind(_RUNNER_START)
    if marker != -1:
        text = text[marker:]

    report = CtxPressureReport(kv_shift_disabled=kv_shift)

    for m in _PROMPT_RE.finditer(text):
        slot = int(m.group("slot"))
        report.prompts.append(
            PromptEvent(
                task=int(m.group("task")),
                n_ctx_slot=slot,
                n_tokens=int(m.group("tokens")),
            )
        )
        report.n_ctx_slot = slot

    if recent > 0 and len(report.prompts) > recent:
        report.prompts = report.prompts[-recent:]
    oldest_task = report.prompts[0].task if report.prompts else None

    for m in _RELEASE_RE.finditer(text):
        if m.group("truncated") != "1":
            continue
        task = int(m.group("task"))
        # A truncation older than the window we are reporting on is history.
        if oldest_task is not None and task < oldest_task:
            continue
        report.truncated_tasks.append(task)

    report.ladder = find_ladder(report.prompts)
    return report


def collect_ctx_pressure(log_path: Path | None = None) -> CtxPressureReport:
    """Best-effort read of the local server log. Empty report when unavailable."""
    from ollama_sentinel.doctor_log import find_latest_server_log

    path = log_path or find_latest_server_log()
    if path is None:
        return CtxPressureReport()
    return parse_ctx_pressure(read_tail(path))
