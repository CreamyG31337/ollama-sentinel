"""Restart the running GUI process (dev/testing helper)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from ollama_sentinel.smi import _no_window


def restart_command() -> list[str]:
    """Build argv to relaunch with the same CLI flags."""
    args = list(sys.argv[1:])
    if args and args[0] == "-m":
        return [sys.executable, *args]
    return [sys.executable, "-m", "ollama_sentinel", *args]


def _wait_and_spawn(parent_pid: int, cmd: list[str], cwd: str) -> None:
    """Wait for parent to exit, then start the replacement GUI."""
    from ollama_sentinel.instance import _pid_alive

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and _pid_alive(parent_pid):
        time.sleep(0.2)
    subprocess.Popen(cmd, cwd=cwd, close_fds=True, **_no_window())


def spawn_restart() -> None:
    """Relaunch after this process exits, then terminate immediately.

    Flet on Windows can leave a stray window if we spawn the replacement
    before the current process (and its flet child) are gone.
    """
    parent = os.getpid()
    cmd = restart_command()
    cwd = os.getcwd()
    helper = [
        sys.executable,
        "-m",
        "ollama_sentinel.restart",
        "--wait-spawn",
        str(parent),
        json.dumps(cmd),
        cwd,
    ]
    popen_kwargs: dict = {**_no_window()}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    subprocess.Popen(helper, **popen_kwargs)
    os._exit(0)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) >= 4 and args[0] == "--wait-spawn":
        _wait_and_spawn(int(args[1]), json.loads(args[2]), args[3])
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
