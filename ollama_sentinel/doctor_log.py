"""Parse Ollama server.log effective config (read-only)."""

from __future__ import annotations

import os
import re
from pathlib import Path

TRACKED_KEYS = (
    "OLLAMA_FLASH_ATTENTION",
    "OLLAMA_KV_CACHE_TYPE",
    "OLLAMA_KEEP_ALIVE",
    "OLLAMA_CONTEXT_LENGTH",
    "OLLAMA_HOST",
    "OLLAMA_NUM_PARALLEL",
    "OLLAMA_MAX_LOADED_MODELS",
    "OLLAMA_GPU_OVERHEAD",
)

# Known Ollama defaults when registry key is unset — agreement with log, not drift.
DEFAULTS: dict[str, str] = {
    "OLLAMA_FLASH_ATTENTION": "false",
    "OLLAMA_KV_CACHE_TYPE": "f16",
    "OLLAMA_KEEP_ALIVE": "5m0s",
    "OLLAMA_CONTEXT_LENGTH": "2048",
    "OLLAMA_HOST": "http://127.0.0.1:11434",
    "OLLAMA_NUM_PARALLEL": "1",
    "OLLAMA_MAX_LOADED_MODELS": "0",
    "OLLAMA_GPU_OVERHEAD": "0",
}

BOOL_KEYS = frozenset({"OLLAMA_FLASH_ATTENTION", "OLLAMA_SCHED_SPREAD", "OLLAMA_VULKAN"})
DURATION_KEYS = frozenset({"OLLAMA_KEEP_ALIVE"})
HOST_KEYS = frozenset({"OLLAMA_HOST"})

_KEY_LINE = re.compile(r"^(OLLAMA_[A-Z0-9_]+):(.+)$")
_KEY_TOKEN = re.compile(r"(OLLAMA_[A-Z0-9_]+):")


def ollama_log_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "Ollama"
    return Path.home() / "AppData" / "Local" / "Ollama"


def find_latest_server_log(log_dir: Path | None = None) -> Path | None:
    """Prefer server.log; else newest server*.log by mtime."""
    directory = log_dir or ollama_log_dir()
    preferred = directory / "server.log"
    if preferred.is_file():
        return preferred
    if not directory.is_dir():
        return None
    candidates = sorted(
        directory.glob("server*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _extract_keys_from_text(text: str) -> dict[str, str]:
    """Extract OLLAMA_*:value pairs from one-per-line or Go map dump lines."""
    result: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        match = _KEY_LINE.match(stripped)
        if match and " env=" not in stripped and 'msg="server config"' not in stripped:
            result[match.group(1)] = match.group(2).strip()
            continue

        tokens = list(_KEY_TOKEN.finditer(line))
        if not tokens:
            continue
        for i, tok in enumerate(tokens):
            key = tok.group(1)
            start = tok.end()
            if i + 1 < len(tokens):
                end = tokens[i + 1].start()
                raw = line[start:end].rstrip()
            else:
                rest = line[start:]
                if rest.startswith("["):
                    depth = 0
                    cut = len(rest)
                    for j, ch in enumerate(rest):
                        if ch == "[":
                            depth += 1
                        elif ch == "]":
                            depth -= 1
                            if depth == 0:
                                cut = j + 1
                                break
                    raw = rest[:cut]
                else:
                    # Take first whitespace-delimited token; empty if next char is end noise
                    raw = rest.split()[0] if rest.split() else ""
                    # Strip trailing slog closers glued to token
                    raw = raw.rstrip(']"\'')
            result[key] = raw.strip()
    return result


def parse_server_log(path: Path) -> dict[str, str]:
    """Parse KEY:value; last occurrence of each key wins (latest startup)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    return _extract_keys_from_text(text)


def _normalize_bool(raw: str | None) -> str:
    if raw is None or raw == "":
        return "false"
    lower = raw.strip().lower()
    if lower in ("1", "true", "yes", "on"):
        return "true"
    return "false"


def _normalize_duration(raw: str | None) -> str:
    if raw is None or raw == "":
        return ""
    s = raw.strip().lower()
    if s in ("-1", "-1m", "-1s"):
        return "-1"
    # Expand short forms to Go-like duration with zero subunits
    # 30m -> 30m0s; 1h -> 1h0m0s; 5m0s already fine
    if re.fullmatch(r"-?\d+h", s):
        return f"{s}0m0s"
    if re.fullmatch(r"-?\d+m", s):
        return f"{s}0s"
    if re.fullmatch(r"-?\d+s", s):
        return s
    return s


def _normalize_host(raw: str | None) -> str:
    if raw is None or raw == "":
        return ""
    s = raw.strip()
    if "://" not in s:
        s = f"http://{s}"
    return s.rstrip("/")


def normalize_value(key: str, raw: str | None) -> str:
    if raw is None:
        return ""
    raw = raw.strip()
    if key in BOOL_KEYS:
        return _normalize_bool(raw)
    if key in DURATION_KEYS:
        return _normalize_duration(raw)
    if key in HOST_KEYS:
        return _normalize_host(raw)
    return raw


def values_agree(key: str, registry_val: str | None, log_val: str | None) -> bool:
    """Unset registry + log default counts as agreement."""
    if registry_val is None or registry_val == "":
        if log_val is None or log_val == "":
            return True
        default = DEFAULTS.get(key)
        if default is None:
            return True
        return normalize_value(key, log_val) == normalize_value(key, default)
    if log_val is None or log_val == "":
        return False
    return normalize_value(key, registry_val) == normalize_value(key, log_val)
