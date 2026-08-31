# ollama-sentinel

Monitor Ollama on NVIDIA GPUs for failure modes generic tools miss: **CPU/GPU spill**,
**PCIe paging** (high util, low power), and **VRAM pressure**. Includes a library view,
Hugging Face discover with typeahead, opt-in pull, Flet GUI, and optional tray icon.

Not a chat UI. Read-only monitoring except explicit `pull`.

## Why these alarms exist

| Alarm | Meaning |
|---|---|
| **SPILL** | A loaded model is not fully on the GPU (`size_vram < size`). With `keep_alive=-1` it stays slow forever. |
| **PCIe PAGING** | High GPU util at implausibly low power — buffers paging over PCIe. |
| **VRAM PRESSURE** | >95% VRAM used — the next load will likely spill. |

## Install

```bash
pip install -e .
# Windows toasts + GUI:
pip install -e ".[windows,gui]"
```

Copy config (never commit the copies):

```bash
copy .env.example .env
copy servers.example.json servers.json   # optional multi-server
```

**Do not commit `.env` or `servers.json`.** Do not `git add -f`.

Default Ollama URL is `http://127.0.0.1:11434`. Do **not** set `OLLAMA_HOST=0.0.0.0` in `.env` — that is a bind address, not a client URL.

## Usage

```bash
# Live console (default)
ollama-sentinel

# One shot for Task Scheduler / cron (exit 0=OK, 1=alarm, 2=unreachable)
ollama-sentinel --once --toast

# JSON snapshot
ollama-sentinel --once --json

# Installed library
ollama-sentinel --list

# Hugging Face search
ollama-sentinel search qwen --sort trendingScore

# Pull to a server
ollama-sentinel pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF --server local

# Unload from VRAM (installed files stay on disk)
ollama-sentinel unload qwen3.8:27b-mtp-q4_K_M --server local
ollama-sentinel unload --all --server local -y

# Config doctor (Windows + local_gpu): drift, orphans, footguns
ollama-sentinel doctor
ollama-sentinel doctor --json
ollama-sentinel doctor --fix-orphans -y   # opt-in kill of orphaned llama-server PIDs

# GUI (tray on Windows; close hides to tray when tray is enabled)
ollama-sentinel --gui
ollama-sentinel --gui --start-minimized    # autostart / logon; window hidden
ollama-sentinel --gui --no-tray            # window only (Linux default)
```

Only one continuous monitor (GUI or live console) may run at a time. A second `--gui` launch focuses the existing window. `--once` and other short commands are always safe alongside the tray app.

### Config doctor (Windows)

`ollama-sentinel doctor` compares User/Machine registry `OLLAMA_*` values to the effective config in `%LOCALAPPDATA%\Ollama\server.log`, looks for orphaned `llama-server.exe` processes, and flags common footguns (connect URL set to `0.0.0.0`, forever keep-alive pinning VRAM, registry changed after last start).

| Exit | Meaning |
|---|---|
| 0 | All checks pass |
| 1 | Any WARN / UNKNOWN |
| 2 | Any FAIL |

Read-only by default. `--fix-orphans` is the only destructive option and never runs in the tray/`--once` poll path. Restart remedies stop `ollama app`, `ollama`, **and** `llama-server` before relaunching the app.

Passive Check A (config drift) and Check B (orphans) also surface as soft alarms in the GUI/tray poll; Status shows `Doctor: N warnings`. Those alarms do **not** change `--once` exit codes (still spill/paging/vram/unreachable only).

### Windows Task Scheduler (every 15 min)

```
schtasks /Create /TN "ollama-sentinel" /TR "\"C:\Path\To\python.exe\" -m ollama_sentinel --once --toast" /SC MINUTE /MO 15 /F
```

### Linux systemd user timer

See `examples/env.linux.example`.

## Multi-server

Copy `servers.example.json` to `servers.json`. Set `local_gpu: true` only on the machine where this process runs **and** has the NVIDIA GPU. GPU paging/VRAM alarms use **local** `nvidia-smi` only.

## Linux

CLI, alarms, HF search, pull, and `--gui` work on Linux. Tray is enabled by default on Windows; use `--no-tray` if the icon fails.

## Status telemetry

The Status tab (and `--once` / live console) show expanded GPU telemetry from a single `nvidia-smi` call: temperature, fan, clocks, pstate, memory utilization, reserved VRAM, free VRAM, and throttle reasons.

Per-process VRAM attribution runs on a **background thread** (default every 30 s) because Windows `Get-Counter` is ~20× slower than `nvidia-smi`. Disable with `PROC_VRAM=0`.

| Variable | Default | Meaning |
|---|---|---|
| `PROC_VRAM` | `1` | `0` disables per-process VRAM collection |
| `PROC_VRAM_INTERVAL` | `30` | Seconds between background process-VRAM polls |
| `PROC_VRAM_MIN_MB` | `64` | Hide processes below this local VRAM usage |

JSON output (`--once --json`) adds `polled_at` on each snapshot and a top-level `process_vram` block with its own timestamp. Readings older than 3× their interval are marked `STALE`.

## Gaming yield (Windows)

When a real fullscreen game needs the GPU, the tray/GUI process can **detect** that and optionally **unload** Ollama models (`keep_alive: 0`) so VRAM frees for the game. The Ollama server stays up — Open WebUI reconnects with a ~13 s cold reload on the next chat.

| Variable | Default | Meaning |
|---|---|---|
| `GAMING_YIELD_OBSERVE` | `1` | Log `gaming_detected` / `gaming_cleared` to `%LOCALAPPDATA%\ollama-sentinel\gaming.jsonl` |
| `GAMING_YIELD` | `0` | `1` enables automatic unload (opt-in) |
| `GAMING_YIELD_INTERVAL` | `12` | Seconds between detection polls |
| `GAMING_YIELD_EXCLUDE` | `SolitaireCollection` | Comma-separated process names never treated as games |
| `GAMING_YIELD_MIN_VRAM_MB` | `1536` | Solitaire gate: need this much local VRAM **or** high 3D util |
| `GAMING_YIELD_MIN_UTIL` | `50` | Solitaire gate / signal D 3D util threshold |
| `GAMING_YIELD_BUSY_UTIL` | `20` | Do not unload while `llama-server` 3D util is at/above this |

Detection runs only in `--gui` (tray) — not in the 15-minute `--once` task. Status tab shows `Gaming: idle | detected | yielded`.

This is an intentional exception to the original read-only stance; unload stays in `unload.py` + the gaming watcher.

## License

MIT — see [LICENSE](LICENSE).
