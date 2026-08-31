"""Tests for doctor_log parse / normalize."""

from __future__ import annotations

from pathlib import Path

from ollama_sentinel.doctor_log import (
    TRACKED_KEYS,
    find_latest_server_log,
    normalize_value,
    parse_server_log,
    values_agree,
)

FIXTURE = Path(__file__).parent / "fixtures" / "server.log.sample"


def test_parse_sample_extracts_tracked_keys():
    cfg = parse_server_log(FIXTURE)
    for key in TRACKED_KEYS:
        assert key in cfg, key
    assert cfg["OLLAMA_FLASH_ATTENTION"] == "true"
    assert cfg["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    assert cfg["OLLAMA_KEEP_ALIVE"] == "30m0s"
    assert cfg["OLLAMA_CONTEXT_LENGTH"] == "65536"
    assert cfg["OLLAMA_HOST"] == "http://0.0.0.0:11434"
    assert len([k for k in TRACKED_KEYS if k in cfg]) == 8


def test_normalize_bool_and_duration_and_host():
    assert normalize_value("OLLAMA_FLASH_ATTENTION", "1") == "true"
    assert normalize_value("OLLAMA_FLASH_ATTENTION", "true") == "true"
    assert normalize_value("OLLAMA_FLASH_ATTENTION", "0") == "false"
    assert normalize_value("OLLAMA_KEEP_ALIVE", "30m") == "30m0s"
    assert normalize_value("OLLAMA_KEEP_ALIVE", "30m0s") == "30m0s"
    assert normalize_value("OLLAMA_KEEP_ALIVE", "1h") == "1h0m0s"
    assert normalize_value("OLLAMA_HOST", "0.0.0.0:11434") == "http://0.0.0.0:11434"
    assert normalize_value("OLLAMA_HOST", "http://0.0.0.0:11434") == "http://0.0.0.0:11434"


def test_values_agree_unset_registry_vs_default():
    assert values_agree("OLLAMA_FLASH_ATTENTION", None, "false")
    assert values_agree("OLLAMA_KEEP_ALIVE", None, "5m0s")
    assert values_agree("OLLAMA_FLASH_ATTENTION", "1", "true")
    assert values_agree("OLLAMA_KEEP_ALIVE", "30m", "30m0s")
    assert values_agree("OLLAMA_HOST", "0.0.0.0:11434", "http://0.0.0.0:11434")


def test_values_disagree_on_mismatch():
    assert not values_agree("OLLAMA_KV_CACHE_TYPE", "q8_0", "f16")
    assert not values_agree("OLLAMA_FLASH_ATTENTION", "true", "false")


def test_find_latest_prefers_server_log(tmp_path: Path):
    older = tmp_path / "server-1.log"
    preferred = tmp_path / "server.log"
    older.write_text("OLLAMA_HOST:a\n", encoding="utf-8")
    preferred.write_text("OLLAMA_HOST:b\n", encoding="utf-8")
    assert find_latest_server_log(tmp_path) == preferred


def test_find_latest_fallback_by_mtime(tmp_path: Path):
    a = tmp_path / "server-old.log"
    b = tmp_path / "server-new.log"
    a.write_text("x\n", encoding="utf-8")
    b.write_text("y\n", encoding="utf-8")
    import os
    import time

    os.utime(a, (time.time() - 100, time.time() - 100))
    os.utime(b, (time.time(), time.time()))
    assert find_latest_server_log(tmp_path) == b


def test_missing_log_returns_empty():
    assert parse_server_log(Path("/nonexistent/server.log")) == {}


def test_parse_go_map_dump_line():
    line = (
        'time=2026-08-30T18:48:03.066-07:00 level=INFO msg="server config" '
        'env="map[OLLAMA_CONTEXT_LENGTH:65536 OLLAMA_FLASH_ATTENTION:true '
        "OLLAMA_KEEP_ALIVE:30m0s OLLAMA_KV_CACHE_TYPE:q8_0 "
        "OLLAMA_HOST:http://0.0.0.0:11434 OLLAMA_MAX_LOADED_MODELS:0 "
        'OLLAMA_NUM_PARALLEL:1 OLLAMA_GPU_OVERHEAD:0 OLLAMA_ORIGINS:[a b] ROCR_VISIBLE_DEVICES:]"'
    )
    from ollama_sentinel.doctor_log import _extract_keys_from_text

    cfg = _extract_keys_from_text(line)
    assert cfg["OLLAMA_FLASH_ATTENTION"] == "true"
    assert cfg["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    assert cfg["OLLAMA_KEEP_ALIVE"] == "30m0s"
    assert cfg["OLLAMA_CONTEXT_LENGTH"] == "65536"
    assert cfg["OLLAMA_HOST"] == "http://0.0.0.0:11434"
    assert cfg["OLLAMA_NUM_PARALLEL"] == "1"
    assert cfg["OLLAMA_GPU_OVERHEAD"] == "0"
    assert cfg["OLLAMA_MAX_LOADED_MODELS"] == "0"
    assert cfg["OLLAMA_ORIGINS"] == "[a b]"
