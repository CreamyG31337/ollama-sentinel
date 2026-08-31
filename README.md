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

# GUI (Linux) or GUI + tray (Windows)
ollama-sentinel --gui
ollama-sentinel --tray
```

### Windows Task Scheduler (every 15 min)

```
schtasks /Create /TN "ollama-sentinel" /TR "\"C:\Path\To\python.exe\" -m ollama_sentinel --once --toast" /SC MINUTE /MO 15 /F
```

### Linux systemd user timer

See `examples/env.linux.example`.

## Multi-server

Copy `servers.example.json` to `servers.json`. Set `local_gpu: true` only on the machine where this process runs **and** has the NVIDIA GPU. GPU paging/VRAM alarms use **local** `nvidia-smi` only.

## Linux

CLI, alarms, HF search, pull, and `--gui` work on Linux. Tray (`--tray`) is best-effort; use `--gui` if the icon fails.

## License

MIT — see [LICENSE](LICENSE).
