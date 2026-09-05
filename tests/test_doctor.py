"""Tests for config doctor checks A–D."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from ollama_sentinel.doctor import (
    DoctorFinding,
    check_derived,
    check_drift,
    check_footguns,
    check_orphans,
    evaluate_doctor_alarms,
    findings_exit_code,
    run_doctor,
)
from ollama_sentinel.doctor_win import kill_orphan_pids


def test_drift_warn_on_mismatch():
    findings = check_drift(
        {"OLLAMA_KV_CACHE_TYPE": "q8_0"},
        {"OLLAMA_KV_CACHE_TYPE": "f16", "OLLAMA_FLASH_ATTENTION": "false"},
        restart_remedy="Stop-Process ...",
    )
    warns = [f for f in findings if f.severity == "warn"]
    assert any(f.id == "config:drift:OLLAMA_KV_CACHE_TYPE" for f in warns)
    assert warns[0].remedy and "Restart" in warns[0].remedy


def test_drift_pass_when_unset_matches_default():
    findings = check_drift(
        {},
        {
            "OLLAMA_FLASH_ATTENTION": "false",
            "OLLAMA_KV_CACHE_TYPE": "f16",
            "OLLAMA_KEEP_ALIVE": "5m0s",
            "OLLAMA_CONTEXT_LENGTH": "2048",
            "OLLAMA_HOST": "http://127.0.0.1:11434",
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_MAX_LOADED_MODELS": "0",
            "OLLAMA_GPU_OVERHEAD": "0",
        },
    )
    assert any(f.id == "config:drift:ok" and f.severity == "pass" for f in findings)


def test_drift_unknown_missing_log():
    findings = check_drift({}, {})
    assert findings[0].severity == "unknown"


def test_orphan_count_mismatch():
    runners = [
        {"pid": 10, "parent_alive": True, "parent_name": "ollama.exe"},
        {"pid": 11, "parent_alive": True, "parent_name": "ollama.exe"},
    ]
    findings = check_orphans([], runners)
    assert len([f for f in findings if f.severity == "warn"]) == 2


def test_orphan_matched_count_ok():
    runners = [
        {"pid": 10, "parent_alive": True, "parent_name": "ollama.exe"},
        {"pid": 11, "parent_alive": True, "parent_name": "ollama.exe"},
    ]
    models = [{"name": "a"}, {"name": "b"}]
    findings = check_orphans(models, runners)
    assert any(f.id == "runner:orphan:ok" for f in findings)


def test_orphan_dead_parent():
    runners = [{"pid": 42, "parent_alive": False, "parent_name": None}]
    findings = check_orphans([{"name": "m"}], runners)
    assert any(f.id == "runner:orphan:42" for f in findings)


def test_orphan_wrong_parent_name():
    runners = [{"pid": 7, "parent_alive": True, "parent_name": "explorer.exe"}]
    findings = check_orphans([{"name": "m"}], runners)
    assert any(f.id == "runner:orphan:7" for f in findings)


def test_orphan_vram_with_zero_models():
    runners = [{"pid": 9, "parent_alive": True, "parent_name": "ollama.exe"}]
    vram = int(15.7 * 1e9)
    rows = [{"pid": 9, "bytes": vram}]
    findings = check_orphans([], runners, rows)
    orphan = next(f for f in findings if f.id == "runner:orphan:9")
    assert "holding 15.7 GB" in orphan.message
    assert orphan.vram_bytes == vram


def test_orphan_vram_bytes_in_json():
    vram = int(5.4 * 1e9)
    runners = [{"pid": 99, "parent_alive": False, "parent_name": None}]
    rows = [{"pid": 99, "bytes": vram}]
    findings = check_orphans([], runners, rows)
    orphan = next(f for f in findings if f.id == "runner:orphan:99")
    payload = orphan.to_dict()
    assert payload["vram_bytes"] == vram
    assert "holding 5.4 GB" in payload["message"]


def test_derived_forever_expires_with_finite_keepalive():
    models = [{"name": "qwen", "expires_at": "2318-01-01T00:00:00Z", "context_length": 65536}]
    log_cfg = {"OLLAMA_KEEP_ALIVE": "30m0s", "OLLAMA_CONTEXT_LENGTH": "65536"}
    findings = check_derived(models, log_cfg)
    assert any("stale_keepalive" in f.id for f in findings)


def test_footgun_zero_zero_url():
    findings = check_footguns(
        ollama_url="http://0.0.0.0:11434",
        log_cfg={},
        snapshot={"models": [], "gpus": []},
        log_path=None,
        registry_mtime=None,
    )
    assert any(f.id == "config:footgun:ollama_url" for f in findings)


def test_run_doctor_missing_log_no_crash():
    findings = run_doctor(
        {"models": [], "gpus": [], "reachable": True},
        registry={},
        log_cfg={},
        log_path=Path("/no/such/server.log"),
        runners=[],
        ollama_url="http://127.0.0.1:11434",
    )
    assert findings
    assert findings_exit_code(findings) == 1  # unknown counts as warn-ish


def test_evaluate_doctor_alarms_maps_a_and_b_only():
    findings = [
        DoctorFinding("drift", "warn", "config:drift:OLLAMA_HOST", "drift"),
        DoctorFinding("orphan", "warn", "runner:orphan:1", "orphan"),
        DoctorFinding("derived", "warn", "config:stale_keepalive:x", "derived"),
        DoctorFinding("footgun", "warn", "config:footgun:ollama_url", "footgun"),
        DoctorFinding("cuda", "warn", "cuda:compat:mismatch", "cuda behind"),
        DoctorFinding("drift", "pass", "config:drift:ok", "ok"),
    ]
    alarms = evaluate_doctor_alarms(findings)
    ids = {a["id"] for a in alarms}
    assert ids == {
        "config:drift:OLLAMA_HOST",
        "runner:orphan:1",
        "cuda:compat:mismatch",
    }
    assert all(a["type"] in ("config", "orphan", "cuda") for a in alarms)


def test_fix_orphans_kills_only_flagged():
    with patch("ollama_sentinel.doctor_win.subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        results = kill_orphan_pids([100, 200])
    assert len(results) == 2
    assert all(r["ok"] for r in results)
    assert run.call_count == 2
    cmds = [" ".join(c.args[0]) for c in run.call_args_list]
    assert any("100" in c for c in cmds)
    assert any("200" in c for c in cmds)
