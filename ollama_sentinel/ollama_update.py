"""Pending-Ollama-update detection, and applying one while the server is idle.

Ollama's Windows tray checks hourly, downloads the new installer into
``%LOCALAPPDATA%\\Ollama\\updates_v2\\<sha>\\OllamaSetup.exe``, and then waits for
the user to click "Restart to update".

**Restarting the ollama process does not apply it.** The staged file is a full
Inno Setup installer (~1.5 GB); killing and relaunching ``ollama app.exe`` just
starts the old build again. The upgrade happens only when that installer runs —
this is how Ollama itself did it on 2026-08-28::

    /CLOSEAPPLICATIONS /FORCECLOSEAPPLICATIONS /SP /NOCANCEL /SILENT /VERYSILENT
    /SUPPRESSMSGBOXES /LOG=upgrade.log

which closes the running server, installs, and relaunches ``ollama app.exe``.
That took ~47 s end to end, during which every client — including remote ones —
loses the API.

Applying is therefore **opt-in and gated on idle**, and idle here means more than
"no local user typing": this server is consumed across the tailnet, so a loaded
model or a recent request is treated as someone else mid-conversation.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# `msg="New update available at https://github.com/ollama/ollama/releases/download/v0.33.3/OllamaSetup.exe"`
_UPDATE_URL_RE = re.compile(r"releases/download/v(?P<version>[0-9]+(?:\.[0-9]+)*)/", re.I)

INSTALLER_NAME = "OllamaSetup.exe"
UPDATES_DIRNAME = "updates_v2"

# Mirrors the flags Ollama's own updater used, so behaviour matches a tray-click
# upgrade rather than being a new install path with untested options.
INSTALL_ARGS = (
    "/CLOSEAPPLICATIONS",
    "/FORCECLOSEAPPLICATIONS",
    "/SP-",
    "/NOCANCEL",
    "/SILENT",
    "/VERYSILENT",
    "/SUPPRESSMSGBOXES",
    "/LOG=upgrade.log",
)

# A model still resident means someone was talking to it inside OLLAMA_KEEP_ALIVE.
DEFAULT_IDLE_SECONDS = 900.0


@dataclass(frozen=True)
class UpdateStatus:
    running_version: str | None = None
    staged_version: str | None = None
    installer: Path | None = None

    @property
    def pending(self) -> bool:
        """A staged installer that is not the version already running.

        An unreadable staged version still counts as pending: the file only
        exists because Ollama downloaded it, and reporting "no update" for
        something we merely failed to parse would hide the thing we were asked
        to watch for.
        """
        if self.installer is None:
            return False
        if self.staged_version and self.running_version:
            return _version_tuple(self.staged_version) > _version_tuple(self.running_version)
        return True

    @property
    def summary(self) -> str:
        if not self.pending:
            return f"Ollama {self.running_version or '?'} is current"
        return (
            f"Ollama {self.staged_version or 'update'} downloaded and waiting "
            f"(running {self.running_version or '?'})"
        )


def _version_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", raw or "")) or (0,)


def ollama_home(base: Path | None = None) -> Path:
    if base is not None:
        return base
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "Ollama"
    return Path.home() / ".ollama"


def find_staged_installer(home: Path | None = None) -> Path | None:
    """Newest staged installer, or None.

    Ollama keeps one directory per download hash and does not always clean old
    ones up, so the newest by mtime is the one it would actually run.
    """
    updates = ollama_home(home) / UPDATES_DIRNAME
    if not updates.is_dir():
        return None
    found = [p for p in updates.glob(f"*/{INSTALLER_NAME}") if p.is_file()]
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


def staged_version_from_app_log(home: Path | None = None) -> str | None:
    """Version Ollama announced, read from the tail of app.log.

    The installer's own file version would need a Windows version-resource call;
    the log line is the same fact from the source that downloaded it, and costs
    a regex. Returns None when the log has rotated past the announcement.
    """
    log = ollama_home(home) / "app.log"
    try:
        with log.open("rb") as fh:
            size = fh.seek(0, os.SEEK_END)
            fh.seek(max(0, size - 200_000))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    matches = _UPDATE_URL_RE.findall(text)
    return matches[-1] if matches else None


def update_status(
    running_version: str | None = None,
    home: Path | None = None,
) -> UpdateStatus:
    return UpdateStatus(
        running_version=running_version,
        staged_version=staged_version_from_app_log(home),
        installer=find_staged_installer(home),
    )


@dataclass(frozen=True)
class IdleVerdict:
    idle: bool
    reason: str


def _newest_request_age(activity, now: datetime | None = None) -> float | None:
    """Seconds since the most recent logged API request, or None if unknown.

    `ServerActivity.recent_requests` is already filtered to the `fresh_seconds`
    the caller built it with, so a caller wanting a 15-minute quiet window must
    build activity with `fresh_seconds=idle_seconds` — otherwise the default 45 s
    filter empties the list and everything looks idle.
    """
    from ollama_sentinel.activity import _parse_gin_time

    now_dt = now or datetime.now(timezone.utc)
    ages: list[float] = []
    for req in getattr(activity, "recent_requests", None) or []:
        ts = _parse_gin_time(getattr(req, "at", "") or "")
        if ts is not None:
            ages.append((now_dt - ts).total_seconds())
    return min(ages) if ages else None


def idle_verdict(
    snapshot: dict,
    activity=None,
    *,
    idle_seconds: float = DEFAULT_IDLE_SECONDS,
) -> IdleVerdict:
    """Whether it is safe to take the server down.

    Deliberately conservative, and it says *why* rather than returning a bare
    bool, because the caller has to be able to explain a refusal to a user
    staring at a pending update that never installs.
    """
    if not snapshot.get("reachable", False):
        return IdleVerdict(False, "server unreachable — not touching it")

    loaded = snapshot.get("models") or []
    if loaded:
        names = ", ".join(str(m.get("name") or m.get("model") or "?") for m in loaded)
        return IdleVerdict(False, f"model loaded ({names}) — someone may be mid-conversation")

    if activity is not None:
        phase = getattr(activity, "phase", None)
        if phase and phase != "idle":
            return IdleVerdict(False, f"server is {phase}")
        newest = _newest_request_age(activity)
        if newest is not None and newest < idle_seconds:
            return IdleVerdict(
                False, f"a request {int(newest)}s ago — quieter than {int(idle_seconds)}s required"
            )

    return IdleVerdict(True, "no models loaded and no recent requests")


def apply_update(
    installer: Path,
    *,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Run the staged installer. Returns (started, message).

    Does not wait for completion: the installer closes the very server this
    process may be talking to, and Inno relaunches ``ollama app.exe`` itself.
    """
    if not installer.is_file():
        return False, f"installer missing: {installer}"
    cmd = [str(installer), *INSTALL_ARGS]
    if dry_run:
        return False, "dry-run, would run: " + " ".join(cmd)
    try:
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = 0x00000008 | 0x00000200  # DETACHED | NEW_PROCESS_GROUP
        subprocess.Popen(cmd, cwd=str(installer.parent), close_fds=True, **kwargs)
    except OSError as exc:
        return False, f"failed to start installer: {exc}"
    note_update_started()
    return True, f"started {installer.name} (~1 min; the API drops while it runs)"


#: Set once an installer has been launched, so a poll loop cannot start a second
#: one while the first is still tearing the server down.
_APPLIED_THIS_PROCESS = False
_APPLIED_AT: float | None = None
# Installer run measured ~47s; keep explaining refuse/timeout as "updating"
# until well after Ollama should be back.
UPDATE_DOWN_GRACE_S = 120.0


def reset_auto_apply_guard() -> None:
    """Test hook — clears the once-per-process latch."""
    global _APPLIED_THIS_PROCESS, _APPLIED_AT
    _APPLIED_THIS_PROCESS = False
    _APPLIED_AT = None


def note_update_started(*, at: float | None = None) -> None:
    """Record that the installer was launched (API will drop for ~1 min)."""
    global _APPLIED_THIS_PROCESS, _APPLIED_AT
    _APPLIED_THIS_PROCESS = True
    _APPLIED_AT = time.time() if at is None else at


def update_in_progress(*, now: float | None = None) -> bool:
    """True while a just-started installer is expected to keep the API down."""
    if not _APPLIED_THIS_PROCESS or _APPLIED_AT is None:
        return False
    tick = time.time() if now is None else now
    return (tick - _APPLIED_AT) < UPDATE_DOWN_GRACE_S


def updating_unreachable_message() -> str:
    return (
        "Ollama update in progress — the installer stops the API for about a minute, "
        "then relaunches the server"
    )


def explain_unreachable_while_updating(
    error: str | None = None,
    *,
    local_gpu: bool = False,
    now: float | None = None,
) -> str | None:
    """Replace refuse/timeout copy when the local host is mid-update."""
    if not local_gpu or not update_in_progress(now=now):
        return None
    return updating_unreachable_message()


def maybe_auto_apply(
    snapshot: dict,
    activity=None,
    *,
    enabled: bool | None = None,
    idle_seconds: float | None = None,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """Install a staged update if the user opted in and the server is quiet.

    Returns ``(started, reason)``. Off by default: applying takes Ollama down
    for about a minute, and this host serves clients over the tailnet that get
    no say in the timing.
    """
    global _APPLIED_THIS_PROCESS
    if _APPLIED_THIS_PROCESS:
        return False, "already applied in this process"

    if enabled is None or idle_seconds is None:
        try:
            from ollama_sentinel.settings import effective

            if enabled is None:
                enabled = bool(effective("update_auto_apply"))
            if idle_seconds is None:
                idle_seconds = float(effective("update_idle_seconds"))
        except Exception:
            return False, "settings unavailable"
    if not enabled:
        return False, "auto-apply disabled"

    status = update_status(running_version=snapshot.get("version"))
    if not status.pending:
        return False, "no update staged"

    verdict = idle_verdict(snapshot, activity, idle_seconds=idle_seconds)
    if not verdict.idle:
        return False, verdict.reason

    started, message = apply_update(status.installer, dry_run=dry_run)
    # apply_update already latches via note_update_started()
    return started, message


def format_update_status_line(
    *,
    pending: bool,
    summary: str,
    auto_apply: bool,
    started: bool,
    reason: str,
    installing: bool = False,
) -> tuple[str, str]:
    """Status-page caption for a pending / just-started update.

    Returns ``(text, palette_key)``. Empty text means hide the line. When
    auto-apply is on but the idle gate refuses, ``reason`` is that refusal so
    the user can see why flipping the toggle did nothing.
    """
    if installing or started:
        detail = reason if (started and reason) else updating_unreachable_message()
        return f"Update: {detail}", "warn"
    if not pending:
        return "", "muted"
    if auto_apply and reason:
        return f"Update pending — {reason}", "warn"
    return f"Update: {summary}", "muted"
