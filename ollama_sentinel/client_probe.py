"""Read a client's *effective* context window off disk.

A static ``context_length`` in the client config records what a client was
believed to use, which is exactly the thing that goes stale. The failure this
guards against is drift: Hermes caches a context window in
``context_length_cache.yaml`` and rewrites that file whenever it re-probes the
model, so a value corrected by hand reverts silently and the truncation returns.

Pointing sentinel at the client's own file means the check reflects what the
client will actually do on its next request, not what someone once wrote down.

Config shape (in the client config JSON)::

    {"name": "hermes",
     "context_length_file": "%LOCALAPPDATA%/hermes/context_length_cache.yaml",
     "context_length_key": "context_lengths"}

``context_length_key`` is an optional dotted path. When the node it selects is a
mapping (Hermes keys its cache by ``model@base_url``), the largest integer leaf
wins — any entry above the served window is a request that can overrun it.

``context_length_match`` narrows that mapping to keys containing a substring.
Hermes caches every provider it has talked to in one file, and a cloud model
legitimately has a 200k window; without the filter that entry would be compared
against a local 65k server and report a problem that does not exist. Set it to
something identifying this server, e.g. ``localhost:11434``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

MAX_PROBE_BYTES = 1_000_000

# `  qwen3.8:27b-heretic@http://localhost:11434/v1: 262144`
# The key itself contains colons, so the greedy group deliberately runs to the
# LAST `: <int>` on the line rather than splitting on the first colon.
_YAML_INT_LINE = re.compile(r"^(?P<indent>[ 	]*)(?P<key>\S.*):[ 	]+(?P<val>-?\d+)[ 	]*$")
# Greedy to the LAST colon: a quoted key can itself contain colons
# ("qwen3.8:27b-heretic":). The trailing-colon anchor keeps `k: v` out.
_YAML_SECTION = re.compile(r"^(?P<key>\S.*):[ 	]*$")


def expand_path(raw: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(raw)))


def _parse_simple_yaml_ints(text: str) -> dict[str, Any]:
    """Integer scalars from a nested YAML mapping, tracked by indentation.

    PyYAML is not a dependency of this project (only ``rich`` is), and the files
    worth probing — Hermes' context cache and config — are plain nested maps of
    ``key: <int>``. Nesting has to be real rather than flattened: a config with
    several providers would otherwise merge their values together and a dotted
    lookup like ``providers.local-ollama.models`` could not select one of them.

    Only integer leaves are kept. Anything richer (lists, anchors, multi-line
    scalars) is skipped rather than guessed at, so an unparsed value reports
    "unknown" instead of a wrong number.
    """
    root: dict[str, Any] = {}
    # (indent, mapping) pairs; the last entry is the mapping we are filling.
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" 	"))
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        m = _YAML_INT_LINE.match(line)
        if m:
            parent[_clean_key(m.group("key"))] = int(m.group("val"))
            continue
        m = _YAML_SECTION.match(line.strip())
        if m:
            child: dict[str, Any] = {}
            parent[_clean_key(m.group("key"))] = child
            stack.append((indent, child))
    return root


def _clean_key(raw: str) -> str:
    """Strip the quotes YAML needs around keys that contain a colon."""
    key = raw.strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in ("'", '"'):
        return key[1:-1]
    return key


def _load_structured(path: Path) -> Any:
    try:
        if path.stat().st_size > MAX_PROBE_BYTES:
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # optional; not a project dependency
        except ImportError:
            return _parse_simple_yaml_ints(text)
        try:
            return yaml.safe_load(text)
        except Exception:
            return _parse_simple_yaml_ints(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _descend(node: Any, dotted: str | None) -> Any:
    if not dotted:
        return node
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _max_int_leaf(node: Any, depth: int = 0) -> int | None:
    """Largest positive integer anywhere under `node`.

    Booleans are excluded: ``isinstance(True, int)`` is True in Python, and a
    stray flag would otherwise be read as a one-token context window.
    """
    if depth > 6:
        return None
    if isinstance(node, bool):
        return None
    if isinstance(node, int) and node > 0:
        return node
    if isinstance(node, dict):
        values = [_max_int_leaf(v, depth + 1) for v in node.values()]
    elif isinstance(node, list):
        values = [_max_int_leaf(v, depth + 1) for v in node]
    else:
        return None
    found = [v for v in values if v is not None]
    return max(found) if found else None


def _filter_keys(node: Any, needle: str | None) -> Any:
    """Keep only mapping entries whose key contains `needle`.

    Returns None when nothing matches, so an over-narrow filter reports
    "unknown" rather than falling back to the unfiltered maximum.
    """
    if not needle or not isinstance(node, dict):
        return node
    kept = {k: v for k, v in node.items() if needle in str(k)}
    return kept or None


def probe_context_length(client: dict[str, Any]) -> int | None:
    """Effective context window for a client, or None when it cannot be read.

    None means "unknown", never "fine" — callers must not treat an unreadable
    file as agreement.
    """
    raw_path = client.get("context_length_file")
    if not raw_path:
        return None
    path = expand_path(str(raw_path))
    if not path.is_file():
        return None
    node = _descend(_load_structured(path), client.get("context_length_key"))
    node = _filter_keys(node, client.get("context_length_match"))
    return _max_int_leaf(node)


def resolve_client_context(client: dict[str, Any]) -> tuple[int | None, str]:
    """(context window, where it came from) — the probed file wins over a static value.

    The file is authoritative because it is what the client reads at request
    time; a declared number is a fallback for clients with no readable config.
    """
    probed = probe_context_length(client)
    if probed is not None:
        return probed, "file"
    declared = client.get("context_length")
    if isinstance(declared, int) and declared > 0:
        return declared, "declared"
    return None, "unknown"
