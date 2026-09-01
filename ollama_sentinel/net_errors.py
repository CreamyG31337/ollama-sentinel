"""User-facing messages for HTTP and socket failures."""

from __future__ import annotations

import socket
import urllib.error
from typing import Any


class HubRequestError(RuntimeError):
    """Hugging Face Hub request failed."""

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause


def format_network_error(exc: BaseException, *, context: str | None = None) -> str:
    """Turn socket/HTTP exceptions into short UI copy."""
    prefix = f"{context}: " if context else ""

    if isinstance(exc, HubRequestError):
        return str(exc) if not context else f"{context}: {exc}"

    if isinstance(exc, TimeoutError):
        return f"{prefix}Timed out — check your connection and try again"

    if isinstance(exc, urllib.error.HTTPError):
        code = exc.code
        if code in (401, 403):
            return f"{prefix}Access denied (HTTP {code}) — check credentials or HF_TOKEN"
        if code == 429:
            return f"{prefix}Rate limited (HTTP 429) — wait a moment and retry"
        if code >= 500:
            return f"{prefix}Remote server error (HTTP {code}) — try again later"
        return f"{prefix}HTTP {code}"

    if isinstance(exc, urllib.error.URLError):
        reason: Any = exc.reason
        if isinstance(reason, TimeoutError):
            return f"{prefix}Timed out — check your connection and try again"
        if isinstance(reason, OSError):
            return prefix + _os_error_message(reason)
        text = str(reason or exc).strip()
        lowered = text.lower()
        if "timed out" in lowered or "timeout" in lowered:
            return f"{prefix}Timed out — check your connection and try again"
        if text:
            return prefix + text
        return f"{prefix}Network error"

    if isinstance(exc, (OSError, socket.timeout)):
        return prefix + _os_error_message(exc)

    return prefix + str(exc)


def _os_error_message(exc: BaseException) -> str:
    winerr = getattr(exc, "winerror", None)
    errno = getattr(exc, "errno", None)
    if winerr == 10061 or errno in (10061, 61, 111):
        return "Connection refused — is the service running?"
    if winerr == 10060 or errno in (10060, 60, 110):
        return "Timed out — host not responding"
    if winerr == 10054 or errno in (10054, 54, 104):
        return "Connection lost — remote host closed the connection"
    msg = str(exc).strip()
    lowered = msg.lower()
    if "connection refused" in lowered or "actively refused" in lowered:
        return "Connection refused — is the service running?"
    if "timed out" in lowered or "timeout" in lowered:
        return "Timed out — host not responding"
    return msg or "Network error"
