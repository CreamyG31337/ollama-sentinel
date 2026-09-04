"""Desktop notifications on alarm transitions."""

from __future__ import annotations

import shutil
import subprocess
import sys

from ollama_sentinel.alarms import AlarmTransition


def notify_transition(transition: AlarmTransition) -> None:
    # Checked here rather than at each call site so no future caller can bypass
    # the user's choice by forgetting the guard.
    try:
        from ollama_sentinel.settings import effective

        if not effective("notifications"):
            return
    except Exception:
        pass

    title = "ollama-sentinel"
    body = transition.message
    if transition.kind == "RESOLVED":
        title = "ollama-sentinel — resolved"

    if sys.platform == "win32":
        try:
            from win11toast import toast

            # Silent: banner only, no notification sound
            toast(title, body, audio={"silent": "true"})
            return
        except ImportError:
            print(f"notify: {title}: {body}", file=sys.stderr)
            return

    if shutil.which("notify-send"):
        subprocess.run(
            [
                "notify-send",
                "--urgency=low",
                "--hint=string:sound-name:",
                "--hint=string:suppress-sound:true",
                title,
                body,
            ],
            check=False,
            capture_output=True,
        )
        return

    print(f"notify: {title}: {body}", file=sys.stderr)
