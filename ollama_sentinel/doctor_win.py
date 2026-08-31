"""Windows helpers for config doctor (registry, processes)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ollama_sentinel.smi import _no_window


def read_registry_env(name: str) -> str | None:
    """Read User then Machine environment variable from the registry."""
    if sys.platform != "win32":
        return None
    try:
        import winreg
    except ImportError:
        return None
    for hive, path in (
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    ):
        try:
            with winreg.OpenKey(hive, path) as key:
                val, _ = winreg.QueryValueEx(key, name)
                if isinstance(val, str) and val != "":
                    return val
        except OSError:
            continue
    return None


def registry_env_mtime() -> float | None:
    """Approximate last write time of HKCU Environment (Check D stale-env proxy)."""
    if sys.platform != "win32":
        return None
    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            # QueryInfoKey returns (subkeys, values, last_modified_filetime)
            info = winreg.QueryInfoKey(key)
            # Windows FILETIME (100ns since 1601) — convert via winreg helper if available
            last_write = info[2]
            # FILETIME to unix: (ft - 116444736000000000) / 1e7
            return (last_write - 116444736000000000) / 10_000_000
    except OSError:
        return None


def list_llama_server_processes() -> list[dict[str, Any]]:
    """Enumerate llama-server processes with parent info. Degrades to []."""
    if sys.platform != "win32":
        return []
    ps = (
        "$procs = Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\"; "
        "$rows = @(); "
        "foreach ($p in $procs) { "
        "$parentName = $null; $parentAlive = $false; "
        "try { "
        "$par = Get-CimInstance Win32_Process -Filter (\"ProcessId=\" + $p.ParentProcessId) -ErrorAction Stop; "
        "$parentName = $par.Name; $parentAlive = $true "
        "} catch { $parentAlive = $false }; "
        "$rows += [PSCustomObject]@{ "
        "pid=[int]$p.ProcessId; name=$p.Name; parent_pid=[int]$p.ParentProcessId; "
        "parent_name=$parentName; parent_alive=[bool]$parentAlive "
        "} }; "
        "if ($rows.Count -eq 0) { '[]' } else { $rows | ConvertTo-Json -Compress }"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=20,
            **_no_window(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    text = (result.stdout or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    out: list[dict[str, Any]] = []
    for row in data:
        out.append(
            {
                "pid": int(row.get("pid") or 0),
                "name": row.get("name") or "llama-server.exe",
                "parent_pid": int(row.get("parent_pid") or 0),
                "parent_name": row.get("parent_name"),
                "parent_alive": bool(row.get("parent_alive")),
            }
        )
    return out


def default_ollama_app_path() -> str:
    local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return str(Path(local) / "Programs" / "Ollama" / "ollama app.exe")


def build_restart_remedy(*, ollama_app: str | None = None) -> str:
    app = ollama_app or default_ollama_app_path()
    return (
        "Stop-Process -Name 'ollama app','ollama','llama-server' -Force -ErrorAction SilentlyContinue\n"
        "foreach ($v in 'OLLAMA_FLASH_ATTENTION','OLLAMA_KV_CACHE_TYPE','OLLAMA_KEEP_ALIVE',"
        "'OLLAMA_CONTEXT_LENGTH','OLLAMA_HOST') {\n"
        "  $val = [Environment]::GetEnvironmentVariable($v,'User')\n"
        "  if ($val) { Set-Item -Path \"Env:$v\" -Value $val }\n"
        "}\n"
        f"Start-Process '{app}'"
    )


def kill_orphan_pids(pids: list[int]) -> list[dict[str, Any]]:
    """Stop listed PIDs. Returns per-pid results. Doctor CLI only."""
    results: list[dict[str, Any]] = []
    if sys.platform != "win32" or not pids:
        return results
    for pid in pids:
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"Stop-Process -Id {int(pid)} -Force -ErrorAction Stop",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                **_no_window(),
            )
            if result.returncode == 0:
                results.append({"pid": pid, "ok": True})
            else:
                results.append({"pid": pid, "ok": False, "error": result.stderr.strip()})
        except (OSError, subprocess.TimeoutExpired) as exc:
            results.append({"pid": pid, "ok": False, "error": str(exc)})
    return results
