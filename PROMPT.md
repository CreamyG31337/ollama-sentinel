# Build spec — hand this to Cursor

Build `ollama-watch`, a Windows console monitor for a local Ollama server and its GPU. Everything
below was measured on the target machine — treat it as fact, don't re-derive it.

## Goal

Catch three failure modes that have actually occurred here. Generic Ollama monitors show a list of
loaded models and total VRAM; none of them detect any of these. The point of this tool is the alarms,
not the model list.

## Environment

- Windows 11, single **RTX 3090 24 GB**, power limit **350 W**, driver 616.56.
- **Ollama 0.33.2** on this machine. It also serves Open WebUI on another host, so this tool must be
  strictly read-only — never unload, stop, or otherwise mutate server state.
- **Python 3.12.10** at `C:\Users\cream\AppData\Local\Programs\Python\Python312\python.exe`.
- Runs as a normal user. **No admin, no elevation.**

## Hard constraints

1. **Always connect to `http://127.0.0.1:11434` as a literal.** Do **not** read `OLLAMA_HOST` — on
   this machine it is `0.0.0.0:11434`, which is the server's *bind* address. Code that treats it as a
   connect address fails. This has already broken two other tools here.
2. **Read-only.** Only `GET /api/ps`, `GET /api/tags`, `GET /api/version`, and `nvidia-smi` queries.
   Never call `/api/generate`, never POST, never unload a model.
3. Standard library plus at most `rich` (console rendering) and one toast library. Pin versions in
   `requirements.txt`. No GPU libraries, no `nvidia-ml-py` — shell out to `nvidia-smi`.
4. Must survive Ollama being down, restarting, or returning no models, without crashing or spamming.

## Data sources and exact field semantics

**`GET /api/ps`** returns `{"models": [...]}`, each entry having:

- `name`, `model`, `digest`
- `size` — total bytes the model occupies
- `size_vram` — bytes resident in **GPU** memory
- `context_length` — the context the model was actually loaded with
- `expires_at` — RFC3339. With `OLLAMA_KEEP_ALIVE=-1` this is a year-2318 sentinel, meaning pinned
  forever. Render that as `Forever`, not as a date.

**`size_vram < size` is the spill condition.** The difference is what is sitting in system RAM and
being shuttled over PCIe. This is exactly what the `ollama ps` `PROCESSOR` column expresses as e.g.
`6%/94% CPU/GPU`. Compute and display that same percentage split yourself:
`gpu_pct = round(100 * size_vram / size)`.

**`nvidia-smi`**, one call per poll:

```
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw,power.limit,temperature.gpu --format=csv,noheader,nounits
```

Note `nvidia-smi` reports `[N/A]` for *per-process* VRAM on Windows, so do not attempt per-process
attribution.

## The three alarms

**A. SPILL — a loaded model is not fully on the GPU.**
Fire when any model has `size_vram < size`. Report the model, the GB on CPU, and the CPU/GPU split.
This is the most important alarm: with `OLLAMA_KEEP_ALIVE=-1` a model that loads spilled **stays
spilled forever**, silently running slow until something restarts it.

**B. PCIe PAGING — high GPU utilization at implausibly low power.**
Fire when `utilization.gpu > 85%` **and** `power.draw < 200 W` sustained for 3 consecutive polls.
A genuinely busy 3090 pulls ~345 W; ~94% utilization at ~147 W means the driver is paging GPU
buffers over PCIe. The real incident behind this alarm was a stalled Plex transcode holding 12 GB of
VRAM, which made inference roughly 30x slower with no error anywhere. Require the sustained count so
idle and light load don't trip it.

**C. VRAM PRESSURE — headroom is nearly gone.**
Fire when `memory.used / memory.total > 0.95`. This is the state that *causes* alarm A on the next
model load.

Alarms need hysteresis: fire once on entering the state, print a single `RESOLVED` line on leaving
it. Do not repeat the same alarm every poll.

## Output

Default: a `rich` live console view, refreshing every 5 seconds (`--interval` to change), showing

- Ollama version and reachability
- Per loaded model: name, size GB, VRAM GB, **CPU/GPU split**, context length, expires (`Forever`)
- GPU: memory used/total, utilization %, power draw / limit, temperature
- An alarm area — green `OK` when clear, otherwise the active alarms

Also provide:

- `--once` — print one plain-text snapshot and exit (for scripting and scheduled tasks)
- `--json` — machine-readable single snapshot
- `--log <path>` — append one JSON line per poll, for later inspection
- `--toast` — Windows toast notification on alarm transitions only, never per poll

Exit codes for `--once`: `0` clear, `1` one or more alarms active, `2` Ollama unreachable.

## Deliverables

- `ollama_watch.py` (single module is fine, but keep polling, alarm evaluation, and rendering as
  separate functions)
- `requirements.txt` with pinned versions
- `README.md` — what each alarm means and why it exists, plus the `schtasks` line to run
  `--once --toast` every 15 minutes
- `tests/test_alarms.py` — unit tests for alarm logic against **synthetic** `/api/ps` and
  `nvidia-smi` payloads. The alarm evaluator must be a pure function of parsed inputs so it is
  testable without a GPU or a running server. Cover at minimum: fully-resident model (no alarm),
  spilled model (alarm A), high-util/low-power sustained for 3 polls (alarm B fires on the 3rd, not
  the 1st), 96% memory used (alarm C), Ollama unreachable, and no models loaded.

## Explicit non-goals

Not a model manager. No pulling, deleting, unloading, or chatting. No web UI. No multi-host support —
this watches localhost only. Don't reimplement `ollama ps` as a prettier table; the alarms are the
product.
