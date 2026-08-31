"""Rotating JSONL alarm log — alarms and transitions only."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ollama_sentinel.alarms import AlarmTransition

DEFAULT_MAX_BYTES = 1_048_576  # 1 MiB
DEFAULT_BACKUP_COUNT = 3


def _rotate(path: Path, backup_count: int) -> None:
    if backup_count <= 0:
        return
    oldest = Path(f"{path}.{backup_count}")
    if oldest.exists():
        oldest.unlink()
    for i in range(backup_count, 0, -1):
        src = path if i == 1 else Path(f"{path}.{i - 1}")
        dst = Path(f"{path}.{i}")
        if src.exists():
            src.replace(dst)


def _maybe_rotate(path: Path, max_bytes: int, backup_count: int) -> None:
    if max_bytes <= 0 or not path.is_file():
        return
    if path.stat().st_size >= max_bytes:
        _rotate(path, backup_count)


def append_alarm_log(
    path: Path,
    *,
    alarms: list[dict[str, Any]],
    transitions: list[AlarmTransition] | None = None,
    ts: float | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    on_transition_only: bool = False,
) -> bool:
    """Append one JSONL record. Returns True if a line was written.

    When on_transition_only is True (live polling), write only on FIRE/RESOLVED.
    When False (--once), write whenever any alarm is active.
    """
    transitions = transitions or []
    if on_transition_only:
        if not transitions:
            return False
    elif not alarms:
        return False

    entry: dict[str, Any] = {
        "ts": ts if ts is not None else time.time(),
        "alarms": alarms,
    }
    if transitions:
        entry["transitions"] = [
            {"kind": t.kind, "id": t.alarm_id, "message": t.message}
            for t in transitions
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    _maybe_rotate(path, max_bytes, backup_count)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
    return True
