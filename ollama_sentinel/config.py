"""Configuration: .env, servers.json, argparse."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ollama_sentinel.alarms import Thresholds
from ollama_sentinel.paths import default_state_path

DEFAULT_URL = "http://127.0.0.1:11434"


@dataclass
class ServerConfig:
    name: str
    url: str
    local_gpu: bool = False


@dataclass
class AppConfig:
    ollama_url: str = DEFAULT_URL
    poll_interval: float = 5.0
    thresholds: Thresholds = field(default_factory=Thresholds)
    gpu_filter: int | None = None
    hf_token: str | None = None
    servers: list[ServerConfig] = field(default_factory=list)
    state_file: Path | None = None
    server: str | None = None
    proc_vram: bool = True
    proc_vram_interval: float = 30.0
    proc_vram_min_mb: int = 64
    gaming_yield: bool = False
    gaming_yield_observe: bool = True
    gaming_yield_interval: float = 12.0
    gaming_yield_exclude: str = "SolitaireCollection"
    gaming_yield_min_vram_mb: int = 1536
    gaming_yield_min_util: float = 50.0
    gaming_yield_busy_util: float = 20.0


def parse_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip().strip('"').strip("'")
    return env


def _find_env_files() -> list[Path]:
    paths: list[Path] = []
    cwd = Path.cwd() / ".env"
    if cwd.is_file():
        paths.append(cwd)
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        if appdata:
            p = Path(appdata) / "ollama-sentinel" / ".env"
            if p.is_file():
                paths.append(p)
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
        p = Path(xdg) / "ollama-sentinel" / ".env"
        if p.is_file():
            paths.append(p)
    return paths


def load_env() -> dict[str, str]:
    merged: dict[str, str] = {}
    for p in _find_env_files():
        merged.update(parse_dotenv(p))
    return merged


def _float_or_none(val: str | None) -> float | None:
    if val is None or val == "":
        return None
    return float(val)


def _int_or_none(val: str | None) -> int | None:
    if val is None or val == "":
        return None
    return int(val)


def config_from_env(env: dict[str, str]) -> AppConfig:
    th = Thresholds(
        paging_util_pct=float(env.get("PAGING_UTIL_PCT", 85)),
        paging_power_frac=float(env.get("PAGING_POWER_FRAC", 0.60)),
        paging_power_w=_float_or_none(env.get("PAGING_POWER_W")),
        paging_polls=int(env.get("PAGING_POLLS", 3)),
        vram_pressure=float(env.get("VRAM_PRESSURE", 0.95)),
    )
    proc_vram_raw = env.get("PROC_VRAM", "1")
    yield_raw = env.get("GAMING_YIELD", "0")
    observe_raw = env.get("GAMING_YIELD_OBSERVE", "1")
    return AppConfig(
        ollama_url=env.get("OLLAMA_URL", DEFAULT_URL),
        poll_interval=float(env.get("POLL_INTERVAL", 5)),
        thresholds=th,
        gpu_filter=_int_or_none(env.get("GPU")),
        hf_token=env.get("HF_TOKEN") or None,
        proc_vram=proc_vram_raw not in ("0", "false", "False", "no"),
        proc_vram_interval=float(env.get("PROC_VRAM_INTERVAL", 30)),
        proc_vram_min_mb=int(env.get("PROC_VRAM_MIN_MB", 64)),
        gaming_yield=yield_raw not in ("0", "false", "False", "no"),
        gaming_yield_observe=observe_raw not in ("0", "false", "False", "no"),
        gaming_yield_interval=float(env.get("GAMING_YIELD_INTERVAL", 12)),
        gaming_yield_exclude=env.get("GAMING_YIELD_EXCLUDE", "SolitaireCollection"),
        gaming_yield_min_vram_mb=int(env.get("GAMING_YIELD_MIN_VRAM_MB", 1536)),
        gaming_yield_min_util=float(env.get("GAMING_YIELD_MIN_UTIL", 50)),
        gaming_yield_busy_util=float(env.get("GAMING_YIELD_BUSY_UTIL", 20)),
    )


def load_servers(path: Path | None, fallback_url: str) -> list[ServerConfig]:
    if path and path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        servers = []
        for entry in data.get("servers", []):
            servers.append(
                ServerConfig(
                    name=entry["name"],
                    url=entry["url"],
                    local_gpu=bool(entry.get("local_gpu", False)),
                )
            )
        if servers:
            return servers
    return [ServerConfig(name="local", url=fallback_url, local_gpu=True)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ollama-sentinel", description="Ollama GPU companion")
    p.add_argument("--interval", type=float, help="Poll interval seconds")
    p.add_argument("--once", action="store_true", help="Single snapshot and exit")
    p.add_argument("--json", action="store_true", help="JSON output")
    p.add_argument("--list", action="store_true", help="Installed library table")
    p.add_argument(
        "--log",
        metavar="PATH",
        help="Append rotating JSONL alarm log (transitions in live mode; active alarms in --once)",
    )
    p.add_argument("--toast", action="store_true", help="Toast on alarm transitions")
    p.add_argument("--state-file", type=Path, help="Alarm state file")
    p.add_argument("--servers-file", type=Path, help="servers.json path")
    p.add_argument("--server", help="Pin to one server name")
    p.add_argument("--ollama-url", help="Default Ollama URL")
    p.add_argument("--gui", action="store_true", help="Open Flet window")
    p.add_argument(
        "--start-minimized",
        action="store_true",
        help="Start with window hidden (tray icon on Windows)",
    )
    p.add_argument("--no-tray", action="store_true", help="Disable system tray icon")
    sub = p.add_subparsers(dest="command")
    sp = sub.add_parser("search", help="Search Hugging Face")
    sp.add_argument("query", nargs="?", default="")
    sp.add_argument("--sort", default="trendingScore", help="Hub sort field")
    sp.add_argument("--limit", type=int, default=20)
    pp = sub.add_parser("pull", help="Pull model to a server")
    pp.add_argument("model", help="Model name or hf.co/...")
    pp.add_argument("--server", default="local", help="Target server name")
    pp.add_argument("-y", "--yes", action="store_true", help="Skip would-spill confirm")
    pu = sub.add_parser("unload", help="Unload model(s) from VRAM")
    pu.add_argument("model", nargs="?", help="Model name (omit with --all)")
    pu.add_argument("--all", action="store_true", help="Unload every loaded model")
    pu.add_argument("--server", default="local", help="Target server name")
    pu.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    doc = sub.add_parser("doctor", help="Diagnose Ollama config drift and orphaned runners")
    doc.add_argument("--json", action="store_true", help="JSON findings")
    doc.add_argument(
        "--fix-orphans",
        action="store_true",
        help="Kill orphaned llama-server PIDs after confirmation",
    )
    doc.add_argument("--server", default="local", help="Target server name")
    doc.add_argument("-y", "--yes", action="store_true", help="Skip --fix-orphans confirm")
    return p


def resolve_config(args: argparse.Namespace) -> AppConfig:
    env = load_env()
    cfg = config_from_env(env)
    if args.ollama_url:
        cfg.ollama_url = args.ollama_url
    if args.interval is not None:
        cfg.poll_interval = args.interval
    if args.state_file:
        cfg.state_file = args.state_file
    else:
        cfg.state_file = default_state_path()
    if args.server:
        cfg.server = args.server
    servers_path = args.servers_file or Path.cwd() / "servers.json"
    cfg.servers = load_servers(servers_path, cfg.ollama_url)
    return cfg


def resolve_gui_options(args: argparse.Namespace) -> tuple[bool, bool]:
    """Return (tray, start_hidden) for GUI mode."""
    start_hidden = bool(args.start_minimized)
    tray = not args.no_tray and sys.platform == "win32"
    return tray, start_hidden


def selected_servers(cfg: AppConfig) -> list[ServerConfig]:
    if cfg.server:
        return [s for s in cfg.servers if s.name == cfg.server]
    return cfg.servers
