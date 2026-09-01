---
name: Multi-host correctness fixes
overview: Four defects and a test-coverage gap found by running ollama-sentinel against two real Ollama hosts on 2026-08-31. Fix these before building the model advisor, which assumes the fleet view is truthful.
isProject: false
---

# Multi-host correctness fixes

**Audience:** an agent doing implementation work. Read this whole file before editing anything.

**Why this comes first:** the model advisor drafted in
`model_optimization_advisor_draft_b5012d92.plan.md` is built entirely on multi-host snapshots. Every
advisory it proposes reads free VRAM, residency and inventory *per server*. Two of the bugs below
make that data wrong in ways that read as reassuring — a fleet view that lies is worse than no fleet
view — so they are prerequisites, not cleanup.

---

## How this was found (reproduce it yourself first)

Create a two-entry servers file. The second host is a **real remote Ollama on AMD/ROCm with no
`nvidia-smi` anywhere**; that combination is what exposes the bugs.

```json
{
  "servers": [
    {"name": "cr-desktop-3090", "url": "http://127.0.0.1:11434", "local_gpu": true},
    {"name": "ubuntu-rx6800",   "url": "http://100.64.188.1:11434", "local_gpu": false}
  ]
}
```

```bash
python -m ollama_sentinel --servers-file /path/to/servers.json --once
```

If you have no access to a second host, the captured fixtures in `tests/fixtures/` (see
[Fixtures](#fixtures)) reproduce every case offline — and the tests you write **must** use those,
not the network.

Observed output, abbreviated, with the four defects marked:

```
[cr-desktop-3090] STALE 21:22:34 (15s ago) Ollama 0.33.2          <- BUG 3
  Library: 6 installed | 2 loaded | 2 would spill | free VRAM: 0.8 GB (3%)
  qwen3.8:27b-mtp-q4_K_M: ctx 65,536 ? Q4_K_M ? qwen35 ...        <- BUG 4 (should be ·)
[ubuntu-rx6800] STALE 21:22:34 (15s ago) Ollama 0.33.2
  Library: 20 installed | 2 loaded | 0 would spill                 <- BUG 2
  Activity: Generating tokens (task 2087)                          <- BUG 1
    runner pid 65220 21.5 GB ? 0% util ? idle                      <- BUG 1 (local pid!)
  Process VRAM as of 21:22:50 (0s ago)                             <- BUG 1
  65220 llama-server.exe: 21.50 GB local ...                       <- BUG 1
ALARMS:
  VRAM PRESSURE [cr-desktop-3090] GPU 0: 24.9/25.8 GB (97%)
  SPILL [ubuntu-rx6800] bge-m3:latest: 1.1 GB on CPU (84% CPU / 16% GPU)
```

The two alarms at the bottom are **correct** — that SPILL is a genuine live problem on the AMD host.
Do not change alarm semantics while fixing the rendering around them.

---

## Ground rules

1. **Do not touch alarm IDs, thresholds, or `--once` exit codes.** `alarms.py` and
   `findings_exit_code` are out of scope. Several of these hosts are monitored by scripts.
2. **Tests must be offline.** No network calls, no `nvidia-smi`, no `subprocess` to real tools. Use
   the fixtures.
3. **Keep the existing style.** `from __future__ import annotations`, type hints on public
   functions, small pure helpers that take data and return data. Match the surrounding file — do not
   introduce a new abstraction layer, a config framework, or a logging framework.
4. **Do not reformat files you are not otherwise changing**, and do not run a formatter across the
   repo. Diffs should be readable.
5. `python -m pytest tests/ -q` must stay green. It is **147 passing** as of this writing and runs
   in under a second; there is no excuse for leaving it red.
6. **Do not "fix" `OLLAMA_HOST` anywhere, in code or docs.** On these machines it is the *bind*
   address and is `0.0.0.0:11434` deliberately. See the advisor plan for why.
7. If a task turns out to be wrong or much larger than described, **stop and say so** rather than
   inventing scope. Note it in the task's checklist and move to the next one.

### Before you start: the working tree may be dirty

At the time of writing, `inventory.py`, `render.py`, `ui_widgets.py` and `tests/test_inventory.py`
carry an **unrelated in-progress feature** (a Library "Details" column adding
`inventory_detail_line()`, `parameter_size` and `context_length` to rows). It is complete and green.

Bugs 1 and 2 touch `render.py` and `inventory.py` directly. **Confirm that work is committed before
you begin** — if `git status` is dirty in those files, stop and ask. Do not stash, revert, or
absorb someone else's uncommitted work into your commits.

---

## BUG 1 — local process data is attributed to remote servers (highest priority)

> **STATUS 2026-08-31: FIXED** in `cb20889`. Verified: a `--once` across four servers shows no
> Activity or Process VRAM block under any remote host. Left here for the record; do not redo.

**Symptom.** In the `[ubuntu-rx6800]` block, sentinel prints `runner pid 65220 21.5 GB` and
`65220 llama-server.exe: 21.50 GB local`. Those are processes on **this Windows desktop**. The
remote host is a Linux box whose Ollama runs in a Docker container; it has no such PIDs, and
sentinel has no way to see its processes at all. The output is fabricated by attribution error.

**Root cause.** `__main__.py` is careful — `_attach_activity()` explicitly skips servers where
`local_gpu` is false, leaving `snap["activity"]` unset. But `render.py` then *reconstructs* it:

```python
# render.py:76
def _format_activity(snap, proc_rows):
    activity = snap.get("activity")
    if activity is None:
        ...
        activity = build_server_activity(proc_rows=proc_rows)   # <- defeats the guard
```

`proc_rows` is the **local** machine's process table, passed in unconditionally at `render.py:128`
(plain renderer) and `render.py:222` (table renderer). So every snapshot without pre-attached
activity gets local data synthesised into it.

**The missing piece:** the snapshot does not record whether its server is local. `poll_all()` knows
(`srv.get("local_gpu")` at `poll.py:93`) but discards it.

**Fix.**

1. In `poll.py`, stamp the flag onto the snapshot — `poll_server()` should accept `local_gpu: bool`
   and set `snapshot["local_gpu"]`, with `poll_all()` passing `bool(srv.get("local_gpu"))`. Default
   it to `True` when unspecified so single-server behaviour is unchanged.
2. In `render.py`, `_format_activity()` must **not** synthesise activity when
   `snap.get("local_gpu") is False`. Return `[]` (render no Activity block at all) rather than an
   empty-looking one.
3. The per-snapshot **Process VRAM** block has the same problem — it must only render for local
   snapshots.
4. Apply the fix to **both** renderers. `render_snapshot_plain` and the table path at
   `render.py:222` share the defect; fixing only the one you reproduced is not done.

**Acceptance.**

- A remote snapshot renders **no** Activity block and **no** Process VRAM block.
- A local snapshot is byte-identical to today's output.
- A regression test builds two snapshots (one `local_gpu: True`, one `False`), renders both with a
  non-empty `proc_rows`, and asserts no local PID string appears anywhere in the remote server's
  section. **This assertion is the whole point of the task** — write it first.

---

## BUG 2 — `0 would spill` on a host with no GPU data

> **STATUS 2026-08-31: FIXED** in `cb20889`. `inventory.py:128` now emits `fit unknown`, and the
> AMD host renders `20 installed | 2 loaded | fit unknown`. Left here for the record.

**Symptom.** The AMD host reports `20 installed | 2 loaded | 0 would spill` while holding
`dolphin-mixtral:8x7b` at **26.44 GB on a 16 GB card**. Zero is not merely imprecise, it is the
most dangerous possible answer: it reads as "everything fits".

**Root cause.** With `local_gpu: false` there is no `nvidia-smi`, so `snapshot["gpus"]` is `None`
and `free_vram_bytes()` returns `None`. In `build_inventory()`:

- unloaded rows correctly get `would_spill = None` (unknown), but
- **loaded** rows unconditionally get `would_spill = False` (`inventory.py`, the `if loaded:` branch)

Then `inventory_summary()` gates on `any(r.get("would_spill") is not None for r in rows)`. The two
loaded rows make that `any()` true, so the counter renders — as `0`, because the 18 genuinely
unknown rows count for nothing.

**Fix.** Distinguish "does not spill" from "cannot tell".

1. In `build_inventory()`, a loaded row should only claim `would_spill = False` on evidence.
   Residency is knowable from `/api/ps` regardless of GPU data (`size_vram` vs `size` — that is how
   the SPILL alarm works), so prefer deriving it from the row's own `gpu_pct`/`size_vram`. If
   neither is available, `None`.
2. In `inventory_summary()`, when any row is unknown, say so. Render
   `fit unknown (no GPU data)` instead of a count, or `N would spill, M unknown` — pick one and be
   consistent across the plain and table renderers.
3. The Library "Fit" column in the table renderer needs the same treatment: unknown must not look
   like a pass.

**Acceptance.**

- With `gpus: None`, the AMD fixture never renders the string `0 would spill`.
- With real GPU data present, output is unchanged from today.
- Tests cover: all-unknown, mixed known/unknown, and all-known.

---

## BUG 3 — `STALE (15s ago)` on a fresh `--once`

> **STATUS 2026-08-31: FIXED** on branch `fix/stale-and-encoding` (`ec5fa61`), not yet merged.
> Each snapshot is now stamped at its own poll completion instead of sharing one pre-loop
> timestamp, and the `--once` call site wires the `once` flag `render.py` already accepted.
> Do not start this.

**Symptom.** A single fresh snapshot is labelled STALE.

**What is already known — do not re-derive this** (this supersedes an earlier, wrong note in
this file that claimed the age was a constant 15s; it is not, it scales with how long the poll
takes):

- It appears **only with two or more servers**. Pinning `--server cr-desktop-3090` makes it vanish.
- The reported age **tracks total poll duration, not `--interval`**. Measured: 2 servers -> 15s;
  4 servers (2 of them unreachable, so each must hit its connect timeout) -> **55-68s**.
- `poll_all()` stamps `polled_at = time.time()` **once, before the loop** (`poll.py:89`) and shares
  that single timestamp across every snapshot. Everything after that stamp — the remaining servers'
  HTTP calls (`DEFAULT_TIMEOUT = 10` each, and unreachable hosts pay it in full), then
  `proc_vram` collection (`timeout=15` at `proc_vram.py:54`) — elapses before anything renders.
- So by render time, `now - polled_at` legitimately exceeds `3 * interval` (15s at the default
  interval of 5.0) even though every snapshot is as fresh as it can be. The staleness test is
  measuring **the app's own polling latency** and calling it staleness.

**Therefore the fix is about *what* is timed, not the threshold.** Options worth weighing: stamp
each snapshot with its own completion time rather than one shared start time; or compare against
poll *completion* rather than poll *start*; or exclude unreachable servers from the shared stamp.
Whichever you choose, an unreachable host must not be able to make healthy hosts look stale.

**Fix.** Trace where `stale` and `polled_at` reach the header for the multi-server path, and make a
snapshot polled moments ago never render as stale. Do not paper over it by raising the threshold.

**Acceptance.** `--once` with two reachable servers shows no STALE marker; a genuinely old snapshot
still does. Add a test at the unit level for whatever the actual mechanism turns out to be.

---

## BUG 4 — console encoding mojibake  ~~(NOT A BUG)~~

> **STATUS 2026-08-31: WITHDRAWN — this was my error, not a defect.** The app writes the
> separators correctly: a redirected run contains byte `0xB7` (cp1252 middle dot) and no literal
> `?` anywhere. The `?` glyphs in the transcript above came from the tool that captured the
> output decoding cp1252 bytes as UTF-8. **Do not 'fix' this** — changing the separators or
> forcing an encoding would be churn against working code.
>
> One genuine, much smaller point survives: `—` (U+2014, used as the empty-value placeholder)
> is not encodable in some OEM code pages such as cp437, so a redirect under those locales could
> raise `UnicodeEncodeError`. That is a latent robustness nit, not the reported symptom, and is
> worth at most a defensive `errors="replace"` if it ever actually bites someone.

**Symptom.** `·` and `°` render as `?` on the Windows console — e.g.
`ctx 65,536 ? Q4_K_M ? qwen35` and `temp 29?C`.

**Fix.** This is stdout encoding, not the strings. Rich should be writing UTF-8; find where the
console is constructed and make it explicit, or fall back to ASCII separators when the stream
cannot encode them. **Do not** globally replace `·` with `-` in source as the fix — that loses the
intent and will drift back.

**Acceptance.** Separators render correctly in Windows Terminal, and nothing raises
`UnicodeEncodeError` when stdout is redirected to a file or a pipe (test that case explicitly —
redirection is where this usually bites).

---

## TASK 5 — multi-server regression tests

The multi-host feature set is **essentially untested**, which is how bugs 1-3 survived. Only
`test_poll.py` and `test_config.py` touch it, and `test_config` exercises only the *fallback
single-server* case.

Add coverage for:

1. `load_servers()` with a real multi-entry file — names, urls and `local_gpu` all preserved.
2. `selected_servers()` — returns all servers when `cfg.server` is None; exactly one when pinned;
   empty when pinned to a name that does not exist (assert current behaviour, and if it is silently
   empty, note that as a finding rather than changing it here).
3. `poll_all()` across two servers where one is reachable and one is not — the unreachable one must
   not corrupt or inherit the other's data. `test_poll.py:56` has a related case to model on.
4. The bug-1 attribution guarantee (above).
5. The bug-2 unknown-fit rendering (above).
6. GPU data is attached **only** to `local_gpu` servers — `poll_all()` already gates this at
   `poll.py:93`; lock it in.

Prefer table-driven tests in the existing style. Do not introduce new test dependencies —
`unittest` and plain `pytest` only, matching what is already there.

---

## Fixtures

Captured from the live fleet on 2026-08-31 and committed to `tests/fixtures/`. These are real
payloads, not hand-written, and they are the reason these tests can be offline. Follow the existing
`server.log.sample` convention: if you add more, note the host and date at the top.

| File | What it is |
|---|---|
| `api_tags_cr_desktop_3090.json` | `/api/tags`, RTX 3090 host, 6 models |
| `api_ps_cr_desktop_3090.json` | `/api/ps`, both models 100% GPU |
| `api_tags_ubuntu_rx6800.json` | `/api/tags`, **AMD ROCm host, 20 models** — includes the 26.44 GB `dolphin-mixtral:8x7b` on a 16 GB card, and two models whose `quantization_level` is `"unknown"` |
| `api_ps_ubuntu_rx6800.json` | `/api/ps` showing **a real spill**: `bge-m3` at 0.20 GB VRAM of 1.26 GB total |
| `api_show_qwen38_mtp.json` | `/api/show` for an MTP model — `nextn_predict_layers`, `blk.*.nextn.*` tensors, `draft_num_predict 4`, `requires`, `projector_info` |
| `api_show_granite41.json` | `/api/show` where `/api/tags` said quant `unknown` but the GGUF header says `Q8_0` |

The `tensors` arrays in the two `show` fixtures are truncated to 6 entries plus a
`{"_truncated": N}` marker, and license text is stripped, to keep them reviewable. The two `show`
fixtures are **not needed for bugs 1-4** — they are here for the advisor work that follows, so do
not build tests that depend on the truncation shape.

To recapture against a live host, the exact endpoints are `/api/tags`, `/api/ps`, and
`POST /api/show {"model": "<name>"}`.

---

## Explicitly out of scope

Do not start these. They are real, they are planned, and they need design decisions that have not
been made:

- Making the **GUI** aggregate multiple servers. Today `ui.py:402` polls exactly one
  (`poll_all(target, ...)[0]`) and the dropdown is a selector. Changing that is a feature with a
  layout question attached, not a bug fix.
- A `rocm-smi` parser in `smi.py` for the AMD host.
- The `"optional": true` per-server flag (so an intentionally-offline gaming rig never alarms).
- Anything from the model advisor plan: `/api/show` caching, `AdvisorFinding`, MTP detection,
  HF cross-referencing.
- The `sentinel.jsonl` at the repo root, and the pre-existing "Details" column work.

---

## Definition of done

- [x] Bug 1 fixed in both renderers, with the attribution assertion test
- [x] Bug 2 fixed, unknown fit distinguished from "fits" everywhere it surfaces
- [x] Bug 3 root-caused (not threshold-tweaked) and covered — branch `fix/stale-and-encoding`
- [x] Bug 4 withdrawn — not a defect; see status note
- [ ] Task 5 tests added
- [ ] `python -m pytest tests/ -q` green, ≥147 tests (169 as of 2026-08-31)
- [ ] Manual check: `--once` against two servers shows no local PIDs under the remote host, no
      `0 would spill`, no STALE, and correct `·` separators
- [ ] One commit per bug, message explaining the *observed wrong behaviour*, not just the change
