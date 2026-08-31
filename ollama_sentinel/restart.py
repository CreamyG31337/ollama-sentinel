"""Restart the running GUI process (dev/testing helper)."""

from __future__ import annotations

import os
import subprocess
import sys

from ollama_sentinel.smi import _no_window


def restart_command() -> list[str]:
    """Build argv to relaunch with the same CLI flags."""
    args = list(sys.argv[1:])
    if args and args[0] == "-m":
        return [sys.executable, *args]
    return [sys.executable, "-m", "ollama_sentinel", *args]


def spawn_restart() -> None:
    """Start a fresh ollama-sentinel process, then call quit on the current one."""
    cmd = restart_command()
    cwd = os.getcwd()
    subprocess.Popen(cmd, cwd=cwd, close_fds=True, **_no_window())
