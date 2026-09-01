"""Restart the running GUI process (dev/testing helper)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

from ollama_sentinel.smi import _no_window

_TH32CS_SNAPPROCESS = 0x00000002


def restart_command() -> list[str]:
    """Build argv to relaunch with the same CLI flags."""
    args = list(sys.argv[1:])
    if args and args[0] == "-m":
        return [sys.executable, *args]
    return [sys.executable, "-m", "ollama_sentinel", *args]


def child_pids(parent_pid: int) -> list[int]:
    """Direct children of ``parent_pid``.

    ``ft.app()`` runs the actual window as a separate ``flet.exe`` child, and
    ``os._exit()`` does not take children with it. Without this the window
    outlives the restart and the user is left staring at a dead UI while a
    second copy starts behind it.
    """
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    class PROCESSENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", ctypes.c_char * 260),
        ]

    kernel32 = ctypes.windll.kernel32
    snapshot = kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if snapshot == -1:
        return []
    found: list[int] = []
    try:
        entry = PROCESSENTRY32()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
        if not kernel32.Process32First(snapshot, ctypes.byref(entry)):
            return []
        while True:
            if entry.th32ParentProcessID == parent_pid:
                found.append(int(entry.th32ProcessID))
            if not kernel32.Process32Next(snapshot, ctypes.byref(entry)):
                break
    finally:
        kernel32.CloseHandle(snapshot)
    return found


def _terminate(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except (OSError, ValueError, PermissionError):
        pass


def _wait_and_spawn(pids: list[int], cmd: list[str], cwd: str) -> None:
    """Wait for every old process to exit, then start the replacement GUI."""
    from ollama_sentinel.instance import _pid_alive

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and any(_pid_alive(p) for p in pids):
        time.sleep(0.2)
    subprocess.Popen(cmd, cwd=cwd, close_fds=True, **_no_window())


def spawn_restart() -> None:
    """Relaunch after this process *and its flet window* are gone.

    The helper waits on the whole set: this process plus the child processes
    flet owns. Waiting only on ``os.getpid()`` is not enough, because
    ``os._exit()`` below returns immediately while the window child keeps
    running.
    """
    parent = os.getpid()
    pids = [parent, *child_pids(parent)]
    cmd = restart_command()
    cwd = os.getcwd()
    helper = [
        sys.executable,
        "-m",
        "ollama_sentinel.restart",
        "--wait-spawn",
        json.dumps(pids),
        json.dumps(cmd),
        cwd,
    ]
    # Merge, don't clobber: _no_window() sets CREATE_NO_WINDOW, and overwriting
    # creationflags outright made the helper flash a console under pythonw.
    popen_kwargs: dict = {**_no_window()}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = (
            popen_kwargs.get("creationflags", 0)
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    subprocess.Popen(helper, **popen_kwargs)

    # os._exit() would strand the window child, so close it explicitly first.
    for pid in pids:
        if pid != parent:
            _terminate(pid)
    os._exit(0)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) >= 4 and args[0] == "--wait-spawn":
        _wait_and_spawn(json.loads(args[1]), json.loads(args[2]), args[3])
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
