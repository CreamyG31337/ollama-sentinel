---
name: GUI overnight leak fix
overview: "Long-running GUI goes blank white while Python stays healthy. Primary cause: Flet control mutations and page.update() from several worker threads racing the event loop inside Flet's unlocked patch_control(). Fix marshals every UI mutation onto the event loop, pauses painting while not visible and repaints on return, and cuts per-second control churn."
todos:
  - id: ui-call-helper
    content: Add ui_call() marshaling helper; move every worker-thread control mutation + update into on-loop paint closures
    status: pending
  - id: pause-when-hidden
    content: Skip paints while tray-hidden or minimized; repaint (rebuild content_area + kick_refresh) when visible again
    status: pending
  - id: live-activity-freshness
    content: Persistent in-place freshness banner; fingerprint-gated activity card rebuild; drop duplicate footer paint from charts loop
    status: pending
  - id: gpu-inplace
    content: Fingerprint-gate GPU card rebuilds (rounded values)
    status: pending
  - id: single-flight-refresh
    content: Coalescing single-flight refresh (rerun, never drop) so switches/clicks cannot stack refresh threads
    status: pending
  - id: cache-overlay-caps
    content: page.show_dialog/pop_dialog instead of overlay.append; ShowCache eviction; detail_cache cap
    status: pending
  - id: logging-tests
    content: Warn-once logging at failure sites; unit tests for helpers
    status: pending
isProject: false
---

# GUI overnight leak / draw-failure fix (revised 2026-09-10)

## Live evidence (instance PID 17184, up since 2026-09-07 02:35, ~3.5 days)

| Check | Result | Meaning |
|---|---|---|
| Window after restore | 960x640, **entirely white** below title bar | Not a title-bar collapse this time; render tree is gone |
| Window state before restore | minimized (iconic), not tray-hidden | Hidden-window gating alone would **not** have prevented this |
| `pythonw` private bytes | 54 MB | Python side is not leaking |
| `flet.exe` | 225 MB private, 219 threads, 1269 handles | Client is alive |
| Python<->flet TCP link | ESTABLISHED | Session was not dropped or replaced |
| Sleep/resume events since launch | none | Not a resume artifact |
| `py-spy dump` | event loop idle; `poll_loop`, `footer_tick_loop`, tray, proc-vram, gaming threads all alive | Data paths keep running; only drawing died |
| Thread counter | `Thread-961340 (_readerthread)` | ~1M short-lived subprocess reader threads (netstat/nvidia-smi); churn, not a leak |

## Root cause (ranked)

### 1. Cross-thread Flet patching — primary

Flet 0.86.5 `Page.update()` calls `session.patch_control()` **synchronously on the calling
thread**. That method diffs the control tree against its previous snapshot, then mutates
`session.__index` (pops removed ids, adds new ones) and sends the patch — with **no lock**.

Callers today, concurrently:

| Caller | Thread | What it does |
|---|---|---|
| `footer_tick_loop` (1 Hz) | worker | replaces `activity_host.content`, `freshness_host.content`, calls `.update()` |
| `poll_loop` -> `refresh()` (5 s) | worker | clears/rebuilds gpu/models/library/proc hosts, `page.update()` x2-3 |
| `probe_fleet_reachability` (5 s) | worker | rebuilds dropdown options, `current_server.update()` |
| `charts_live_loop` (2 s) | **event loop** | rebuilds GPU cards, replaces freshness banner, updates |
| pull / search / detail / unload workers | worker | `page.update()` |

Two threads diffing overlapping subtrees can send patches out of order or leave the client
referencing control ids the Python index already dropped. Once the Flutter side hits a bad
patch it stops rendering the tree — Python never finds out. Rare per tick, certain over ~10^5
ticks, which matches "fine for hours, blank after days".

Note: **property mutations race too**, not only `.update()` — the diff reads `.content`,
`.controls`, `.value` while another thread assigns them. The mutation and the update must run
together on the loop.

Note: sync event handlers (`on_click`, `on_select`, `on_nav`) already run **on the event loop**
in Flet 0.86 (`base_control.py` calls them inline). They need no marshaling.

### 2. Painting while not visible

Flet's own docstring on `Page.wait_until_visible`: *"While hidden, the client's render pipeline
is suspended, so a producer loop that keeps pushing frames only builds a backlog that floods the
client on resume."* We push ~2-3 patches/second into a hidden or minimized window for hours.

### 3. Control churn

Per 8 h: ~29k activity cards, ~43k freshness banners (footer 1 Hz **and** charts loop 0.5 Hz),
~14k GPU card rebuilds, ~5.8k full `refresh()` rebuilds. Every replacement is a mount/unmount in
both the Python index and the Flutter tree — more surface for #1 and needless work.

### Corrections to the first draft

- `poll_loop` calls `refresh()` **inline**, so it cannot stack threads overnight. Stacking only
  comes from manual Refresh / host switch / unload / pull — single-flight is lower priority.
- Single-flight must **coalesce** (run once more after the current pass), never drop: a host
  switch that arrives mid-refresh must still get its own refresh.
- Do **not** force 960x640 on every show — it stomps a user-resized window. Restore only when the
  height is clearly collapsed (< 300).
- Pinned `flet>=0.24` is stale: installed is 0.86.5 and the code already relies on 0.80+ APIs
  (`Dropdown.on_select`). Bump the lower bound to `>=0.80`.
- `page.overlay` leak is real but user-driven (one per unload confirm) — insurance, not the cause.

## Implementation

### Step 1 — `ui_call` helper and paint closures (highest priority)

In `ui.py`, inside `app(page)`:

```python
import asyncio, logging
log = logging.getLogger(__name__)

loop = page.session.connection.loop  # verify attribute path in flet 0.86.5; page.run_task uses it

def _on_loop() -> bool:
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False

def ui_call(fn, *, wait: bool = False, timeout: float = 10.0) -> None:
    """Run fn (mutations + update) on the Flet event loop."""
    if _on_loop():
        fn()          # already on loop (event handlers, charts_live_loop)
        return
    async def _run():
        fn()
    fut = page.run_task(_run)
    if wait:
        fut.result(timeout=timeout)   # only ever from a worker thread — never on-loop (deadlock)
```

Wrap `fn` so exceptions go through warn-once logging (step 7) instead of vanishing.

Then restructure every worker path as **compute off-loop, paint on-loop**:

- `refresh()`: keep polling, alarm evaluation, state save, update-status read, doctor and
  advisor I/O on the worker. Move every control assignment (`host_banner.*`, `alarm_host.content`,
  `gpu_host.controls`, `models_host.controls`, `activity_host.content`, `proc_vram_host.controls`,
  `library_host.controls`, `unload_all_btn.disabled`, status texts, `rebuild_server_options()`)
  plus `page.update()` into paint closures passed to `ui_call(..., wait=True)` — one for the main
  status paint, one after doctor, one after advisor. `wait=True` keeps the three phases ordered.
  Re-check `refresh_guard.still_current(...)` **inside** each closure before painting.
  The existing `page.run_task(_paint_charts)` folds into the first closure.
- `footer_tick_loop`: build `ServerActivity` on the worker; paint via `ui_call`. Tray update
  (`update_tray`, pystray) stays on the worker — it is not Flet.
- `probe_fleet_reachability`: TCP probe on worker; `rebuild_server_options()` + update via `ui_call`.
- `_run_unload`, `request_pull`, `load_detail_if_needed`, `do_search` workers: status text,
  `render_discover_results()`, `page.update()` via `ui_call`. Keep the non-worker lines (called
  from handlers) as-is.
- `charts_live_loop` is already on-loop; leave it, except step 3.

Acceptance: `grep -n "page.update()\|\.update()" ollama_sentinel/ui.py` — every hit is inside an
event handler, an `async def`, or a closure handed to `ui_call`.

### Step 2 — pause while not visible, repaint on return

- `ui_state = {"visible": not start_hidden}`; set in `show_window_async` / `hide_window_async`.
- `def can_paint() -> bool: return ui_state["visible"] and getattr(page, "app_visible", True)`
  (covers minimize via Flet's lifecycle state).
- Paint closures from **timers** (footer tick, poll refresh, charts loop, fleet probe) return
  early when `not can_paint()`. Data still updates (`last_snap`, `poll_state`, metrics store, tray).
  User-initiated paints (pull/search/unload status) are not gated.
- Recovery `repaint_all()` (on-loop): if `page.window.height` is known and < 300, set 960x640;
  `content_area.content = _page_column(pages[nav_state["index"]])`; `page.update()`; then
  `kick_refresh()` so every panel repaints from a fresh poll.
- Call `repaint_all()` from `show_window_async`, and on becoming visible again: hook
  `page.on_app_lifecycle_state_change` (visible when state not HIDE/PAUSE) and window events
  (`SHOW`, `RESTORE`, `FOCUS` — check which exist in `flet/controls/core/window.py`). The window
  close handler is already on `page.window.on_event`; extend it rather than replacing it. Debounce
  so a burst of focus events triggers one repaint (e.g. skip if one ran < 2 s ago).
- Log visibility transitions at INFO — if `app_visible` ever sticks at False on Windows we need
  to see it, because it would freeze the UI.

### Step 3 — in-place freshness banner, gated activity card

In `ui_widgets.py`:

- `class LiveFreshnessBanner`: builds the Container/Row/badge/label/hint **once**; exposes
  `.control` and `set(level, label) -> bool` that mutates badge text, badge bgcolor, container
  bgcolor, label value in place and returns whether anything changed. Keep `freshness_banner()`
  for callers/tests that need a one-off.
- `def activity_fingerprint(activity) -> tuple`: phase, summary, stale, model, n_gen, gen_tps
  (rounded), prompt_tokens, n_ctx_slot, and hashable summaries of runners / peers / recent
  requests as rendered by `activity_card`. Pure function, no Flet.

In `ui.py`:

- `freshness_host.content = live_freshness.control` once; `update_poll_footer()` calls
  `live_freshness.set(...)`; `blank_for_server` uses `set("unknown", caption)`.
- `paint_activity(act)`: rebuild `activity_host.content = activity_card(act)` **only** when the
  fingerprint differs from the last painted one; reset the stored fingerprint on host switch
  and in `repaint_all()`.
- Remove `update_poll_footer()` + freshness/footer `.update()` from `charts_live_loop` — the
  1 Hz footer tick owns it.

### Step 4 — GPU cards

`paint_live_gpu_cards()` and the GPU block in `refresh()`: compute a fingerprint from each GPU
dict with floats rounded to display precision (util %, W, GB to 0.1); skip the rebuild when equal
to the last painted fingerprint. Reset on host switch / `repaint_all()`.

### Step 5 — coalescing single-flight refresh

New small class (e.g. in `refresh_guard.py`, testable without Flet):

```python
class SingleFlight:
    """At most one run at a time; requests during a run cause exactly one rerun."""
    def request(self) -> bool: ...   # True -> caller should start a worker
    def finish(self) -> bool: ...    # True -> run again (a request arrived meanwhile)
```

`kick_refresh()` and `on_server_change` use it; the worker loops `refresh()` while `finish()`
says rerun. `poll_loop` goes through the same path so a manual refresh and the timer never
overlap. `RefreshGuard` stays — it still decides whether a result is applied.

### Step 6 — dialogs and caches

- `_confirm_dialog` / `_close_dialog`: use `page.show_dialog(dlg)` and `page.pop_dialog()` (Flet
  0.86 API; removes the dialog from the stack) instead of `page.overlay.append` + `open=False`.
- `ShowCache`: drop expired entries on write and cap at 200 entries (evict oldest). Guard with a
  lock — `fetch_all` calls `get` from 8 threads.
- `discover_state["detail_cache"]`: cap at 20 (evict oldest insertion), also drop matching
  `detail_errors`.
- `pyproject.toml`: `flet>=0.80,<1`.

### Step 7 — logging

Module logger in `ui.py`. `_warn_once(site: str, exc: BaseException)` logs WARNING with
traceback the first time a site fails, then at most once per 10 minutes per site. Use it in
`ui_call`'s wrapper and replace the bare `except Exception: pass` in the loops and paint paths.
Do not log per tick.

## Tests (no Flet window)

- `tests/test_ui_live.py`:
  - `activity_fingerprint` equal for identical activity, differs when `n_gen` / phase / peers change.
  - `LiveFreshnessBanner.set` mutates in place (same control object ids), returns False on a
    repeated identical set.
  - GPU fingerprint ignores sub-display-precision jitter.
- `tests/test_refresh_guard.py`: `SingleFlight` — request while running returns False and causes
  exactly one rerun; many requests during a run still coalesce to one rerun.
- `tests/test_show_cache.py`: expired entries evicted, cap enforced.

Run `python -m pytest -q` — baseline before the change: **334 passed, 1 skipped**.

## Verification (manual)

1. Launch `--gui` (tray). Leave it minimized for hours, and separately X-close to tray for hours.
2. Restore / tray Open -> live data within a few seconds, no white window.
3. Status tab: freshness counter ticks, activity updates during generation, charts move without clicking.
4. Host switch mid-refresh still lands on the new host.
5. Task Manager: `flet.exe` private bytes plateau.

Immediate workaround: tray **Restart** recovers (fresh Flet process); tray Open does not.

## Review round 1 (after first implementation pass) — fix these

Pass 1 landed Steps 1-7 (357 passed, 1 skipped). Flet names verified (`page.loop`,
`WindowEventType.RESTORE/SHOW/FOCUS`, `AppLifecycleState`, `on_app_lifecycle_state_change`).
The activity/GPU fingerprints cover everything the cards render. Remaining defects:

1. **Warnings go nowhere.** Nothing configures logging, and under `pythonw` `sys.stderr` is
   `None`, so `logging.lastResort` drops every `_warn_once` WARNING and every visibility INFO line.
   In `run_gui` (before `ft.app`), attach a `logging.handlers.RotatingFileHandler` to the
   `ollama_sentinel` logger: `%LOCALAPPDATA%\ollama-sentinel\gui.log` (fall back to
   `~/.ollama-sentinel/gui.log` off Windows), 1 MB x 3 backups, INFO, timestamped format. Idempotent
   (do not add a second handler if one for that path already exists). Make it a small function
   with a unit test that uses a temp dir.
2. **FOCUS repaints on every alt-tab.** `repaint_all()` rebuilds `content_area` and kicks a full
   HTTP refresh; wiring it to `FOCUS` means every window focus does that. Add
   `ui_state["missed"]`: set True whenever a gated paint returns early because `not can_paint()`.
   Window/lifecycle events (`SHOW`/`RESTORE`/`FOCUS`/lifecycle-visible) call `repaint_all()` only
   when `missed` is True; `show_window_async` (tray Open) always does. `repaint_all()` clears it.
3. **`do_search` worker still mutates a mounted control off-loop** — `pull_status.value = ...`
   at the result / "Search failed" lines. Compute the text on the worker; assign it inside the
   `paint()` closure.
4. **Unload and pull workers call `refresh()` directly**, bypassing `SingleFlight`. Use
   `kick_refresh()`.
5. **Footer tick diffs the whole page every second.** `paint_footer` and the paint in
   `rebuild_live_activity` call `page.update()`. Scope them: `page.update(freshness_host,
   poll_footer)` and `page.update(activity_host)` (Flet 0.86 `Page.update(*controls)`). Only call
   the activity update when `paint_activity` actually changed something (have it return bool), and
   the footer update only when `live_freshness.set` or the footer text/color changed.

## Out of scope / follow-ups

- `list_tcp_peers` spawns a subprocess every second (~1M reader threads in 3.5 days) — replace
  with a native connection-table read (`psutil` or `GetExtendedTcpTable`) or slow it to the poll
  interval. CPU churn, not the blank-window cause.
- `ft.app()` is deprecated since 0.80 — migrate to `ft.run()` separately.
- Full per-refresh library/models rebuild (~5.8k/day) — revisit only if blanking persists.
