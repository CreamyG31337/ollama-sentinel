---
name: GUI hold last good data between refreshes
overview: "Same-server refreshes blank the advisor notes (and doctor/advisor alarms, Library advisory badges) until the slow /api/show pass repaints them; a single failed poll blanks every panel. Wipe only on a host switch or after a sustained outage; otherwise keep the last real data on screen, clearly marked stale."
isProject: false
---

# Hold last good data between refreshes

## Intent (from the user)

The GUI must only show real data. That rule is right for a **host switch** — wipe everything
until the new host answers. It is wrong **between refreshes of the same host**, and wrong for a
**brief connection hiccup**. Sustained unreachability should still blank.

## Bug 1 — advisor notes flicker on every refresh

In `ollama_sentinel/ui.py` `refresh()` (same-server path):

- `last_advisories.clear()` runs before the first paint.
- `paint_main` sets `advisor_status.content = None`, `doctor_status.value = ""`, and builds the
  Library with no advisories and no `/api/show` enrichment.
- `paint_main`'s alarm banner is built from `evaluate_alarms` only — no doctor or advisor alarms.
- The doctor pass and the advisor pass (seconds later) paint them back.
- `paint_doctor` only paints when there ARE doctor alarms — it relies on `paint_main` having
  cleared the old ones. Once `paint_main` stops clearing, `paint_doctor` must paint its result
  **including the empty result** (reset `doctor_status` and the banner when the warnings go away).

### Fix

Keep the latest known results for the current host and reuse them until replaced:

- `last_advisories` (exists), plus new holders: `last_show_by_model: dict`, `last_doctor_alarms:
  list`, `last_advisor_alarms: list`.
- Do not clear them at the start of `refresh()`.
- `paint_main` / the first-pass compute:
  - Library: `build_inventory(snap)`, then `enrich_inventory_rows(inv, last_show_by_model)` when
    non-empty, advisories from `last_advisories`. Rows come from the fresh `tags`, so a model that
    was removed disappears immediately and a new one simply has no enrichment yet.
  - Alarm banner from `active + last_doctor_alarms + last_advisor_alarms`.
  - Do **not** touch `advisor_status` or `doctor_status`.
- Doctor pass: set `last_doctor_alarms` to the fresh list (possibly empty); always repaint the
  banner from the composed list and set `doctor_status` (text or `""`).
- Advisor pass: set `last_show_by_model`, `last_advisories`, `last_advisor_alarms`; repaint banner
  (composed), `advisor_status`, Library. If the pass raises, keep the previous values.
- One helper builds the composed alarm list so the three paints cannot disagree. Keep the existing
  `new_state.active_ids` / `save_state` behavior.
- Everything that clears on a host switch must clear the new holders too: `blank_for_server`
  and/or `clear_switch_state` (which is unit-tested in `tests/test_ui_host_switch.py`).

## Bug 2 — one failed poll blanks everything

Today the first unreachable result immediately empties GPU cards, loaded models, activity,
process VRAM and Library, replaces the alarm banner with "Unreachable", and sets
`poll_state["polled_ts"]` to the **failure** time — so the freshness strip reads
"Unreachable · last <now> (0s ago)", hiding how old the on-screen data really is.

### Fix

New pure module `ollama_sentinel/panel_hold.py` (no Flet imports), e.g.:

```python
UNREACHABLE_GRACE_S = 60.0

class UnreachableHold:
    """Decides whether a failed poll may keep the last good panels on screen."""
    def record(self, server: str, reachable: bool, now: float) -> str:
        """'fresh' (reachable), 'hold' (unreachable within grace of the last good
        poll for this server), or 'blank' (no good poll for this server yet, or
        the grace has run out)."""
    def last_good_ts(self, server: str) -> float | None: ...
    def reset(self) -> None: ...   # host switch
```

Grace is measured from the last **good** poll, not from the first failure. 60 s covers an Ollama
self-update (~47 s, see AGENTS.md) and a single 30 s timeout poll, and still blanks well within
the time anyone would act on stale numbers.

In `refresh()`:

- `fresh` → current behavior (with the Bug 1 changes).
- `hold` →
  - Do not overwrite `last_snap` with the failure snapshot (unload-all names, gaming yield and
    live GPU sampling read it). Keep the failure snapshot in a local.
  - `poll_state["reachable"] = False`, `poll_state["polled_ts"] = last_good_ts` (so the strip reads
    "Unreachable · last HH:MM:SS (Ns ago)" with the real age), `poll_state["live_ts"] = None`.
  - Paint: host banner in warn color, freshness footer, `unload_all_btn.disabled = True`. Leave GPU,
    models, activity, process VRAM, Library, advisor, doctor and the alarm banner exactly as they are
    — do not rebuild them.
  - Metrics ingest, `host_online`, tray update and alarm state evaluation are unchanged (the tray
    is the alarm surface; it may turn red immediately).
  - Skip the doctor and advisor passes (they need a reachable host).
- `blank` → current unreachable behavior (full wipe, "Unreachable" banner). Here `polled_ts` may be
  the last good ts if there is one, else the failure ts.
- `blank_for_server` (host switch) calls `hold.reset()`.

`rebuild_live_activity` already returns early while `poll_state["reachable"]` is False, so the
activity card stays put during a hold. `ingest_live_gpu_metrics` is likewise gated.

## Constraints

- Keep every mounted-control mutation inside the existing `ui_call` paint closures (the GUI was
  just fixed for cross-thread patching — see `.cursor/plans/gui_overnight_leak_fix_e6cfe02c.plan.md`).
  Keep the `refresh_guard.still_current(...)` and `paint_blocked()` checks at the top of each paint.
- No changes to alarm evaluation semantics, notifications, or the tray.
- No drive-by refactors.

## Tests

- `tests/test_panel_hold.py`:
  - reachable → `fresh`; first failure right after a good poll → `hold`.
  - failure with no good poll for that server → `blank`.
  - failures past `UNREACHABLE_GRACE_S` since the last good poll → `blank`, and it stays `blank`.
  - recovery → `fresh`, and a later failure gets a new grace window.
  - grace is per server; `reset()` forgets everything.
- `tests/test_ui_host_switch.py`: whatever clears on switch also clears the new holders.
- If the alarm composition is a module-level helper, test that it concatenates base + doctor +
  advisor without duplicating ids.

Run `python -m pytest -q` (baseline 360 passed, 1 skipped).

## Manual check

1. Status tab on the local host: advisor panel and Library badges never vanish across refreshes.
2. Stop Ollama for ~20 s and start it again: panels stay, freshness strip turns red and counts up,
   then returns to Live.
3. Leave it stopped > 60 s: panels blank with the Unreachable banner.
4. Switch hosts in the dropdown: immediate wipe to "Loading …" as before.
