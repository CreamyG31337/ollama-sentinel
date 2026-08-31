# Build spec — status page: VRAM attribution, free VRAM, timestamp, full telemetry

Extend the existing status view (console `--once`/live, and the Flet window in `ui.py`) to answer
three questions it currently cannot: **what is holding the VRAM, how much is actually free, and how
old is this reading.** Then round out the GPU telemetry.

Everything below was **measured on the target machine on 2026-08-30**. Treat it as fact and do not
re-derive it — one of these facts will cost you an afternoon if you rediscover it the hard way.

---

## 1. Per-process VRAM — the trap

**`nvidia-smi` cannot attribute VRAM per process on Windows.** This is not a flag or permissions
problem; the WDDM driver model does not expose it. Measured:

```
> nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
2888,  C:\Windows\System32\dwm.exe,                        [N/A]
45188, ...\Ollama\lib\ollama\llama-server.exe,             [N/A]
...43 processes, every single one [N/A]
```

The process list is real; **the memory column is always `[N/A]`**. Do not ship a feature that
depends on it, and do not "fix" it with `sudo`/elevation — it is elevated-proof.

### What does work on Windows: the performance counter

```powershell
(Get-Counter '\GPU Process Memory(*)\Local Usage').CounterSamples
```

Instance names look like `pid_45188_luid_0x00000000_0x0000ABCD_phys_0`. Extract the PID with
`pid_(\d+)`, then map to a process name via PID. Measured output:

```
PID    Process        GB
45188  llama-server   20.02
2888   dwm             0.53
```

That is the feature: `llama-server` holding 20.02 GB explains a 23.9/24.6 GB reading that the
model's own 17.5 GB does not. (The gap is the KV cache plus the desktop compositor.)

Notes:
- Sum instances **per PID** — a process can have several LUID/phys instances.
- Filter noise: drop anything under ~64 MB by default, make the floor configurable.
- The PID may be gone by the time you resolve the name; render `pid 1234 (exited)`, never crash.
- `Local Usage` is dedicated VRAM. There is also `\GPU Process Memory(*)\Non Local Usage` (spilled
  to system RAM) — worth showing as a second column, since it is *per-process spill* and complements
  the existing SPILL alarm nicely.

### Cost — this is why it needs its own cadence

Measured on this machine:

| Call | Time |
|---|---|
| `nvidia-smi --query-gpu=...` | **68 ms** |
| `Get-Counter '\GPU Process Memory(*)\Local Usage'` | **1,276 ms** |
| `typeperf` equivalent | 1,189 ms |

The counter is **~19x slower** than nvidia-smi. At the default 5 s poll it would burn 25% of the
interval and make the UI stutter.

**Required:** do not call it on the main poll path.
- Give it its own interval (`PROC_VRAM_INTERVAL`, default 30 s) and run it on a background thread.
- Cache the last result and render it with its own "as of" time.
- Make it opt-out via config (`PROC_VRAM=0`) for anyone who does not want a 1.3 s subprocess.
- Reuse `smi._no_window()` for the subprocess, or a console window will flash every time. This
  already bit us once: the 5 s `nvidia-smi` poll flashed a window under `pythonw` until
  `CREATE_NO_WINDOW` was added.

### Linux

On Linux `--query-compute-apps` **does** report real `used_memory`. Branch on `sys.platform`: use
nvidia-smi there (cheap, accurate) and the perf counter only on Windows. Keep both behind one
function returning the same shape, e.g.:

```python
def query_process_vram() -> list[dict]:
    """[{"pid": int, "name": str, "bytes": int, "non_local_bytes": int | None}, ...]"""
```

---

## 2. Free VRAM

`memory.total - memory.used`, shown in GB **and** as a percentage, next to the existing used figure.

Also surface **`memory.reserved`** (254 MiB here) — driver-reserved memory that is neither used nor
allocatable, so `free` never reaches `total` and that is not a bug.

The library view already computes `would_spill` against free VRAM; show the *number* that decision
was made from, so "would spill" is explainable rather than mysterious.

---

## 3. Timestamp and staleness

Every panel must say when its data was taken. Right now a frozen UI is indistinguishable from an
idle GPU.

- Show poll time as local time, `HH:MM:SS`, plus relative age (`3s ago`).
- The process-VRAM panel carries its **own** timestamp — it updates on a slower cadence (above).
- If a reading is older than **3x its interval**, grey it out and mark it `STALE`. If a poll throws,
  keep showing the last good values, marked stale, rather than blanking the panel.
- In `--json`, add ISO-8601 `polled_at` per snapshot and per process-VRAM block. Do not remove or
  rename existing keys — the scheduled task logs this shape.

---

## 4. Fuller GPU telemetry

One `nvidia-smi` call already returns all of this — **verified working** on driver 616.56, 68 ms:

```
nvidia-smi --query-gpu=name,temperature.gpu,fan.speed,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,pstate,memory.used,memory.total,memory.reserved,power.draw,power.limit,enforced.power.limit,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_power_cap --format=csv,noheader
```

Live sample:

```
NVIDIA GeForce RTX 3090, 33, 50 %, 1 %, 25 %, 210 MHz, 810 MHz, P5,
23296 MiB, 24576 MiB, 254 MiB, 38.40 W, 350.00 W, 350.00 W, Not Active, Not Active
```

Add to the status view: **temperature**, **fan %**, **SM / memory clocks**, **pstate**,
**memory utilization %** (distinct from GPU utilization — memory-bandwidth bound vs compute bound),
and **throttle reasons**.

Throttle reasons are the highest-value addition: `clocks_event_reasons.*` returning `Active`
directly explains low power at high utilization, which is exactly the ambiguity the PCIe PAGING
alarm exists to guess at. If `hw_thermal_slowdown` or `sw_power_cap` is `Active`, say so plainly
instead of making the user infer it.

Do **not** add HWiNFO64 or any other sensor dependency. nvidia-smi covers every GPU field here, and
HWiNFO would require its own running service and shared-memory access for no gain. Reconsider only
if CPU/board sensors are ever wanted in the same pane.

Parse defensively: some fields return `[N/A]` or `[Not Supported]` on other cards. Any unavailable
field must render `—` and never crash or alarm.

---

## Constraints

1. **Read-only.** Only `GET` against Ollama plus `nvidia-smi` / perf-counter reads. This server also
   feeds Open WebUI on another host — never unload, stop, or mutate anything.
2. **Never read `OLLAMA_HOST` as a connect address.** It is `0.0.0.0:11434` here — a *bind* address.
   Always dial `http://127.0.0.1:11434` unless `.env` overrides it.
3. Every Windows subprocess goes through `smi._no_window()`.
4. Do not block the UI thread on the 1.3 s counter call.
5. Keep existing `--json` keys and exit codes (`0` clear, `1` alarms, `2` unreachable) intact.

## Tests

Extend `tests/`, keeping everything offline and GPU-free — parsing and formatting must be pure
functions over synthetic payloads:

- perf-counter instance parsing: multiple instances for one PID summed correctly; malformed instance
  names ignored; a PID that no longer resolves renders `(exited)`
- `[N/A]` / `[Not Supported]` in **any** nvidia-smi field renders `—` rather than raising
- free VRAM and percentage arithmetic, including `memory.reserved`
- staleness: a reading older than 3x its interval is marked `STALE`; a failed poll retains last-good
  values instead of blanking
- Linux vs Windows process-VRAM source selection dispatches on `sys.platform`

## Non-goals

Not a model manager, not a chat UI, no web UI. Do not add a GPU library dependency
(`nvidia-ml-py`, `pynvml`) — shelling out to `nvidia-smi` is deliberate and keeps the install thin.
