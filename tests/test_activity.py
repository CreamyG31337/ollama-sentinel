"""Tests for Ollama activity inference."""

from __future__ import annotations

from datetime import datetime, timezone

from ollama_sentinel.activity import (
    build_server_activity,
    model_detail_line,
    parse_server_log_activity,
)


SAMPLE = """
[GIN] 2026/08/30 - 21:16:42 | 200 |     41.3914ms |    100.64.188.1 | POST     "/api/embed"
slot launch_slot_: id  0 | task 11779 | processing task, is_child = 0
slot print_timing: id  0 | task 11779 | prompt processing, n_tokens =   4096, progress = 0.47, t =   3.96 s / 1033.35 tokens per second
slot print_timing: id  0 | task 11779 | acc per pos = (0.833, 0.867)
""".strip().splitlines()


def test_parse_gin_and_prompt():
    now = datetime(2026, 8, 30, 21, 17, 0, tzinfo=timezone.utc)
    recent, task, prompt, generation = parse_server_log_activity(
        SAMPLE, now=now, fresh_seconds=300
    )
    assert len(recent) == 1
    assert recent[0].path == "/api/embed"
    assert task and task["task_id"] == 11779
    assert prompt and prompt["progress"] == 0.47
    assert generation and generation["task_id"] == 11779


def test_build_activity_prompt_phase(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("\n".join(SAMPLE) + "\n", encoding="utf-8")
    proc_rows = [
        {"pid": 123, "name": "llama-server.exe", "bytes": 17_000_000_000, "engine_3d_pct": 42.0}
    ]
    act = build_server_activity(log_path=log, proc_rows=proc_rows)
    assert act.phase == "generating"
    assert "Generating" in act.summary
    assert act.runners[0].busy


def test_model_detail_line():
    model = {
        "context_length": 65536,
        "details": {
            "quantization_level": "Q4_K_M",
            "family": "qwen35",
            "parameter_size": "27.3B",
        },
    }
    text = model_detail_line(model)
    assert "ctx 65,536" in text
    assert "Q4_K_M" in text
    assert "qwen35" in text
