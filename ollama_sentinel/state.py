"""Persist alarm state across --once runs."""

from __future__ import annotations

import json
from pathlib import Path

from ollama_sentinel.alarms import AlarmState


def load_state(path: Path) -> AlarmState:
    if not path.is_file():
        return AlarmState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return AlarmState()
    return AlarmState(
        paging_streak={k: int(v) for k, v in (data.get("paging_streak") or {}).items()},
        active_ids=set(data.get("active_ids") or []),
    )


def save_state(path: Path, state: AlarmState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "paging_streak": state.paging_streak,
        "active_ids": sorted(state.active_ids),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
