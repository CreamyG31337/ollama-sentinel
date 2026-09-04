# ollama-sentinel

A cross-platform companion for Ollama on NVIDIA GPUs: spill / paging / VRAM alarms,
installed-vs-loaded library view, Hugging Face discover, and opt-in pull. Not a chat UI.

If `AGENTS.local.md` exists in this folder, read it for host-specific facts (GPU, Python path,
Ollama env). That file is gitignored.

See `PROMPT.md` for the original alarm spec.

## Two traps to respect

1. **`OLLAMA_HOST` is often a bind address (`0.0.0.0:11434`).** Never treat it as a connect
   address — always dial `http://127.0.0.1:11434` unless the user overrides in `.env`.
2. **`nvidia-smi` returns `[N/A]` for per-process VRAM on Windows.** Use whole-GPU queries only.

## Why this exists rather than forking an existing monitor

Surveyed 2026-08-30. All three candidates were dormant, and none reported the CPU/GPU split --
the single number that catches a silent spill:

| Project | Last push | Stars | License |
|---|---|---|---|
| ElBruno/ElBruno.OllamaMonitor | 2026-07-02 | 11 | MIT |
| ysfemreAlbyrk/ollama-monitor | 2026-06-03 | 30 | **none -- cannot legally fork** |
| yonie/ollama-monitor | 2026-01-18 | 2 | MIT |

Generic monitors show loaded models and total VRAM. The alarms are the product here, not the table.

## The third trap: the served context window is not the advertised one

Ollama reports a model's **architectural** context (`/api/show` → 262144) but serves whatever
`OLLAMA_CONTEXT_LENGTH` says (65536 here). A client that auto-detects the first number sizes its
history and its compaction threshold to a window that does not exist, fills the real one, and has
every reply truncated. Hit for real on 2026-09-04: Hermes cached 262144, so its compressor waited
for 131,072 tokens and never fired; prompts arrived at 65,303 → 65,506 against a 65,536 slot,
leaving 30 tokens to answer in.

`ctx_pressure.py` reads what actually happened, from lines llama.cpp emits per request:

| Line | Meaning |
|---|---|
| `new prompt, n_ctx_slot = C, ..., task.n_tokens = P` | the answer gets at most `C - P` — fires **before** generation |
| `stop processing: n_tokens = N, truncated = 1` | a generation hit the wall; ground truth, not a heuristic |
| `KV cache shifting is not supported` | a full window truncates hard instead of sliding |

The **retry ladder** is the signature worth knowing: consecutive near-ceiling prompts that grow
(`65303, 65358, 65409, 65460`). It means the client re-sends its partial reply on truncation, so
each retry makes the prompt *larger* — retrying walks into the wall instead of backing off. A
single oversized prompt is a different problem with a different remedy, so the two are kept apart.

**Scoping keeps a fixed problem from alarming forever**: only the current runner (task numbers
restart at 0 on reload) and only the last `RECENT_REQUESTS` requests count.

### Prevention that survives drift

`client_probe.py` reads a client's **own** config rather than trusting a number written into
`clients.json`. This matters because Hermes' `save_context_length()` rewrites
`context_length_cache.yaml` on every re-probe — a hand-applied fix reverts silently, which is
exactly how the incident would come back. Declare the file, not the value:

```json
{"name": "hermes",
 "context_length_file": "%LOCALAPPDATA%/hermes/context_length_cache.yaml",
 "context_length_key": "context_lengths",
 "context_length_match": "localhost:11434"}
```

`context_length_match` is not optional in practice: Hermes caches every provider it has talked to
in one file, and a cloud model really does have a 200k window. Without the filter that entry gets
compared against a 65k local server and the tool cries wolf.

PyYAML is **not** a dependency (only `rich` is), so the probe falls back to a small parser for
`key: <int>` maps. It splits on the **last** `: <int>` because the keys themselves contain colons
(`qwen3.8:27b-heretic@http://localhost:11434/v1`). Anything richer is ignored rather than guessed
at — an unreadable file reports "unknown", never "fine".

**The durable fix on the client side is `model.context_length` in Hermes' `config.yaml`**; editing
the cache alone does not hold.

## Pending Ollama updates: restarting does not apply them

Ollama's tray checks hourly and downloads the new installer to
`%LOCALAPPDATA%\Ollama\updates_v2\<sha>\OllamaSetup.exe`, then waits for a "Restart to update"
click. **Restarting the ollama process installs nothing** — the staged file is a full ~1.5 GB Inno
Setup installer, so killing and relaunching `ollama app.exe` just starts the old build again. The
upgrade happens only when that installer runs, which is what Ollama itself did on 2026-08-28:

```
/CLOSEAPPLICATIONS /FORCECLOSEAPPLICATIONS /SP /NOCANCEL /SILENT /VERYSILENT
/SUPPRESSMSGBOXES /LOG=upgrade.log
```

It closes the server, installs, and relaunches `ollama app.exe`. That run took **~47 s**, during
which the API is gone for every client.

`ollama_update.py` reports the state; `ollama-sentinel update` prints it and `--apply` installs.
Detection reads the running version from `/api/version` and the staged one from the
`releases/download/vX.Y.Z/` URL in `app.log` — the same fact from the process that downloaded it,
for the cost of a regex instead of a Windows version-resource call. A staged installer whose
version cannot be parsed still counts as pending: the file exists only because Ollama fetched it.

**The idle gate is strict on purpose, because a server may have remote consumers** across a LAN
or tailnet that get no say in the timing. It refuses while a model is resident — with `OLLAMA_KEEP_ALIVE` at 30 m that means
someone was talking to it recently — while the phase is not `idle`, or while any request is newer
than `--idle-seconds` (default 900). It returns a *reason*, not a bare boolean, so a refusal can
be explained to someone watching an update that never installs.

Note `build_server_activity(fresh_seconds=...)` must be passed the same window: `recent_requests`
is pre-filtered to 45 s by default, so a 15-minute check against the default list sees an empty
list and calls everything idle.

Unattended use is opt-in — `ollama-sentinel update --apply -y` from a scheduled task. Without
`-y` and with no stdin it refuses (exit 1) rather than raising `EOFError`, and when the server is
busy it declines with exit 1 so the next run simply retries. `--force` overrides the gate and will
drop in-flight requests, local and remote.

## Settings: a sparse store, not a second config file

`.env` is deployment config — hand-edited, lives in the repo folder, provisions the machine.
Toggles flipped in the GUI need somewhere else to live, or the app would have to rewrite a file
the user owns and lose their comments and ordering doing it. `settings.py` persists them to
`%LOCALAPPDATA%\ollama-sentinel\settings.json`.

**The store is sparse: it holds only keys the user actually changed.** Precedence is
CLI flag > settings.json > `.env` > declared default, and "nothing stored" stays distinguishable
from "stored as the default" — otherwise the first GUI visit would freeze every value and `.env`
could never own an untouched setting again. Deleting the file restores `.env`, not a set of
hardcoded defaults.

Settings are declared once in `SETTINGS`, and `settings_panel()` renders the GUI page from that
registry — a new feature flag is one entry with no UI code, and no toggle can quietly go missing
from the panel. `config_attr` maps a setting onto `AppConfig`; settings without one (`ctx_pressure`,
the update flags) are deliberately *not* read off the config object, so a same-named attribute
cannot be picked up by accident. A test asserts every declared `config_attr` exists on `AppConfig`.

Gates are applied at the lowest level rather than at call sites — `notify_transition()` checks
`notifications` itself, so no future caller can bypass the choice by forgetting a guard.

`update_auto_apply` defaults to **off** and is asserted off by a test. Turning it on lets the
monitor loop run the staged installer once the idle gate passes; a once-per-process latch stops a
poll loop launching a second installer while the first is still taking the server down. Numbers
from the GUI are clamped and fall back to their default when unparseable, so an empty text field
cannot persist as `0` and make every server look idle.
