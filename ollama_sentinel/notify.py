"""Desktop notifications on alarm transitions."""

from __future__ import annotations

import shutil
import subprocess
import sys

from ollama_sentinel.alarms import AlarmTransition


def notify_transition(transition: AlarmTransition) -> None:
    title = "ollama-sentinel"
    body = transition.message
    if transition.kind == "RESOLVED":
        title = "ollama-sentinel — resolved"

    if sys.platform == "win32":
        try:
            from win11toast import toast

            toast(title, body)
            return
        except ImportError:
            print(f"notify: {title}: {body}", file=sys.stderr)
            return

    if shutil.which("notify-send"):
        subprocess.run(
            ["notify-send", title, body],
            check=False,
            capture_output=True,
        )
        return

    print(f"notify: {title}: {body}", file=sys.stderr)
