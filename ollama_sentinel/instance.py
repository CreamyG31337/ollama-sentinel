"""Single-instance lock for continuous monitor modes."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, TextIO

from ollama_sentinel.paths import app_data_dir

LOCK_NAME = "continuous.lock"
SHOW_REQUEST_NAME = "show.request"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_lock_meta(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        return json.loads(text)
    except (OSError, json.JSONDecodeError):
        return None


def _try_file_lock(fh: TextIO) -> bool:
    import fcntl

    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock_file(fh: TextIO | None) -> None:
    if fh is None:
        return
    try:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


class InstanceLock:
    CONTINUOUS = "continuous"
    _MUTEX_NAME = "Local\\ollama-sentinel-continuous"

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base = base_dir or app_data_dir()
        self._base.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._base / LOCK_NAME
        self._show_path = self._base / SHOW_REQUEST_NAME
        self._fh: TextIO | None = None
        self._mutex: int | None = None
        self._owned = False

    def _mutex_name(self) -> str:
        if self._base.resolve() == app_data_dir().resolve():
            return self._MUTEX_NAME
        digest = hashlib.sha256(str(self._base.resolve()).encode()).hexdigest()[:16]
        return f"Local\\ollama-sentinel-{digest}"

    @staticmethod
    def is_holder_alive(path: Path) -> bool:
        meta = _read_lock_meta(path)
        if not meta:
            return False
        return _pid_alive(int(meta.get("pid", 0)))

    def _write_meta(self, mode: str) -> None:
        self._lock_path.write_text(
            json.dumps({"pid": os.getpid(), "mode": mode}),
            encoding="utf-8",
        )

    def _clear_stale_lock(self) -> bool:
        if not self._lock_path.exists():
            return True
        if self.is_holder_alive(self._lock_path):
            return False
        try:
            self._lock_path.unlink(missing_ok=True)
        except OSError:
            return False
        return True

    def _try_acquire_windows(self, mode: str) -> bool:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        mutex = kernel32.CreateMutexW(None, True, self._mutex_name())
        if not mutex:
            return False
        already_exists = kernel32.GetLastError() == 183  # ERROR_ALREADY_EXISTS
        if already_exists:
            kernel32.CloseHandle(mutex)
            if self._clear_stale_lock():
                mutex = kernel32.CreateMutexW(None, True, self._mutex_name())
                if not mutex or kernel32.GetLastError() == 183:
                    if mutex:
                        kernel32.CloseHandle(mutex)
                    return False
            else:
                return False
        self._mutex = mutex
        self._owned = True
        self._write_meta(mode)
        return True

    def _try_acquire_unix(self, mode: str) -> bool:
        for attempt in range(2):
            if attempt == 1 and not self._clear_stale_lock():
                return False
            fh = open(self._lock_path, "a+", encoding="utf-8")
            if not _try_file_lock(fh):
                fh.close()
                if attempt == 0 and self._clear_stale_lock():
                    continue
                return False
            fh.seek(0)
            fh.truncate(0)
            fh.write(json.dumps({"pid": os.getpid(), "mode": mode}))
            fh.flush()
            self._fh = fh
            self._owned = True
            return True
        return False

    def try_acquire(self, mode: str = CONTINUOUS) -> bool:
        if self._owned:
            return True
        if sys.platform == "win32":
            return self._try_acquire_windows(mode)
        return self._try_acquire_unix(mode)

    def release(self) -> None:
        if not self._owned:
            return
        if sys.platform == "win32":
            if self._mutex is not None:
                import ctypes

                ctypes.windll.kernel32.CloseHandle(self._mutex)
                self._mutex = None
        else:
            _unlock_file(self._fh)
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None
        try:
            self._lock_path.unlink(missing_ok=True)
        except OSError:
            pass
        self._owned = False

    def request_show(self) -> None:
        try:
            self._show_path.write_text("1", encoding="utf-8")
        except OSError:
            pass

    def consume_show_request(self) -> bool:
        try:
            if self._show_path.is_file():
                self._show_path.unlink()
                return True
        except OSError:
            pass
        return False
