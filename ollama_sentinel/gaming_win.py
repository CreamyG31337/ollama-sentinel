"""Windows gaming signal collectors (degrade-safe)."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Any

from ollama_sentinel.gaming import is_fullscreen_bounds

QUNS_RUNNING_D3D_FULL_SCREEN = 3
MONITOR_DEFAULTTONEAREST = 2


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", RECT),
        ("rcWork", RECT),
        ("dwFlags", wintypes.DWORD),
    ]


def query_exclusive_fullscreen() -> bool:
    """Signal A: SHQueryUserNotificationState == QUNS_RUNNING_D3D_FULL_SCREEN."""
    if sys.platform != "win32":
        return False
    try:
        state = wintypes.DWORD()
        hr = ctypes.windll.shell32.SHQueryUserNotificationState(ctypes.byref(state))
        if hr != 0:
            return False
        return int(state.value) == QUNS_RUNNING_D3D_FULL_SCREEN
    except Exception:
        return False


def _foreground_window_info() -> dict[str, Any] | None:
    if sys.platform != "win32":
        return None
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        rect = RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        hmon = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        if not user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
            return None
        path = _process_image_path(int(pid.value))
        name = path.rsplit("\\", 1)[-1] if path else None
        return {
            "pid": int(pid.value),
            "name": name,
            "exe_path": path,
            "win": (rect.left, rect.top, rect.right, rect.bottom),
            "mon": (
                mi.rcMonitor.left,
                mi.rcMonitor.top,
                mi.rcMonitor.right,
                mi.rcMonitor.bottom,
            ),
        }
    except Exception:
        return None


def _process_image_path(pid: int) -> str | None:
    try:
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        try:
            buf = ctypes.create_unicode_buffer(4096)
            size = wintypes.DWORD(len(buf))
            # QueryFullProcessImageNameW
            ok = ctypes.windll.kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
            if not ok:
                return None
            return buf.value
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    except Exception:
        return None


def query_borderless_fullscreen() -> dict[str, Any]:
    """Signal B + foreground pid/name. Degrades to not-fullscreen on failure."""
    empty: dict[str, Any] = {
        "borderless_fullscreen": False,
        "pid": None,
        "name": None,
        "exe_path": None,
    }
    info = _foreground_window_info()
    if not info:
        return empty
    win = info["win"]
    mon = info["mon"]
    fullscreen = is_fullscreen_bounds(*win, *mon)
    return {
        "borderless_fullscreen": fullscreen,
        "pid": info["pid"],
        "name": info["name"],
        "exe_path": info["exe_path"],
    }


def query_game_config_paths() -> set[str]:
    """Signal C: MatchedExeFullPath values from GameConfigStore."""
    if sys.platform != "win32":
        return set()
    try:
        import winreg
    except ImportError:
        return set()
    paths: set[str] = set()
    root = r"System\GameConfigStore\Children"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, root) as key:
            i = 0
            while True:
                try:
                    child = winreg.EnumKey(key, i)
                except OSError:
                    break
                i += 1
                try:
                    with winreg.OpenKey(key, child) as ck:
                        val, _ = winreg.QueryValueEx(ck, "MatchedExeFullPath")
                        if isinstance(val, str) and val.strip():
                            paths.add(_norm_path(val))
                except OSError:
                    continue
    except OSError:
        return set()
    return paths


def _norm_path(path: str) -> str:
    return path.replace("/", "\\").lower()


def path_in_game_list(exe_path: str | None, game_paths: set[str]) -> bool:
    if not exe_path or not game_paths:
        return False
    return _norm_path(exe_path) in game_paths


def collect_windows_signals() -> dict[str, Any]:
    """Gather A/B/C (and foreground identity). Never raises."""
    out: dict[str, Any] = {
        "exclusive_fullscreen": False,
        "borderless_fullscreen": False,
        "in_game_list": False,
        "pid": None,
        "name": None,
        "exe_path": None,
    }
    if sys.platform != "win32":
        return out
    try:
        out["exclusive_fullscreen"] = query_exclusive_fullscreen()
        fg = query_borderless_fullscreen()
        out["borderless_fullscreen"] = bool(fg.get("borderless_fullscreen"))
        out["pid"] = fg.get("pid")
        out["name"] = fg.get("name")
        out["exe_path"] = fg.get("exe_path")
        games = query_game_config_paths()
        out["in_game_list"] = path_in_game_list(out.get("exe_path"), games)
    except Exception:
        pass
    return out
