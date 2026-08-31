"""Configuration: .env, servers.json, argparse."""

from __future__ import annotations

import argparse
import json
import os
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
    return AppConfig(
        ollama_url=env.get("OLLAMA_URL", DEFAULT_URL),
        poll_interval=float(env.get("POLL_INTERVAL", 5)),
        thresholds=th,
        gpu_filter=_int_or_none(env.get("GPU")),
        hf_token=env.get("HF_TOKEN") or None,
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
    p.add_argument("--log", metavar="PATH", help="Append JSONL log per poll")
    p.add_argument("--toast", action="store_true", help="Toast on alarm transitions")
    p.add_argument("--state-file", type=Path, help="Alarm state file")
    p.add_argument("--servers-file", type=Path, help="servers.json path")
    p.add_argument("--server", help="Pin to one server name")
    p.add_argument("--ollama-url", help="Default Ollama URL")
    p.add_argument("--gui", action="store_true", help="Open Flet window")
    p.add_argument("--tray", action="store_true", help="Open Flet window + tray icon")
    p.add_argument("--tray-only", action="store_true",
                   help="Tray icon only; start hidden, open from the tray menu")
    sub = p.add_subparsers(dest="command")
    sp = sub.add_parser("search", help="Search Hugging Face")
    sp.add_argument("query", nargs="?", default="")
    sp.add_argument("--sort", default="trendingScore", help="Hub sort field")
    sp.add_argument("--limit", type=int, default=20)
    pp = sub.add_parser("pull", help="Pull model to a server")
    pp.add_argument("model", help="Model name or hf.co/...")
    pp.add_argument("--server", default="local", help="Target server name")
    pp.add_argument("-y", "--yes", action="store_true", help="Skip would-spill confirm")
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


def selected_servers(cfg: AppConfig) -> list[ServerConfig]:
    if cfg.server:
        return [s for s in cfg.servers if s.name == cfg.server]
    return cfg.servers
