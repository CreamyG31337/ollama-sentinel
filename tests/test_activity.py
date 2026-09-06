"""Tests for Ollama activity inference."""

from __future__ import annotations

from datetime import datetime, timezone

from ollama_sentinel.activity import (
    build_peer_name_map,
    build_server_activity,
    is_inference_request,
    is_monitor_request,
    list_tcp_peers,
    model_detail_line,
    parse_powershell_peers,
    parse_server_log_activity,
    parse_ss_peers,
)


SAMPLE = """
[GIN] 2026/08/30 - 21:16:42 | 200 |     41.3914ms |    100.64.188.1 | POST     "/api/embed"
slot launch_slot_: id  0 | task 11779 | processing task, is_child = 0
slot print_timing: id  0 | task 11779 | prompt processing, n_tokens =   4096, progress = 0.47, t =   3.96 s / 1033.35 tokens per second
slot print_timing: id  0 | task 11779 | acc per pos = (0.833, 0.867)
""".strip().splitlines()

NOISE_AND_CHAT = """
[GIN] 2026/08/30 - 21:16:40 | 200 |            0s |       127.0.0.1 | GET      "/api/ps"
[GIN] 2026/08/30 - 21:16:40 | 200 |      1.2000ms |       127.0.0.1 | GET      "/api/tags"
[GIN] 2026/08/30 - 21:16:41 | 200 |      2.5000ms |       127.0.0.1 | POST     "/api/show"
[GIN] 2026/08/30 - 21:16:42 | 200 |       12.500s |    100.75.27.13 | POST     "/v1/chat/completions"
""".strip().splitlines()

N_GEN_SAMPLE = """
slot launch_slot_: id  0 | task 586686 | processing task, is_child = 0
slot   operator(): id  0 | task 586686 | new prompt, n_ctx_slot = 65536, n_keep = 4, task.n_tokens = 45259
slot print_timing: id  0 | task 586686 | n_gen =    846, tg =  30.68 t/s, tg_3s =  32.37 t/s
""".strip().splitlines()

COMPLETED_SAMPLE = """
slot launch_slot_: id  0 | task 100 | processing task, is_child = 0
slot   operator(): id  0 | task 100 | new prompt, n_ctx_slot = 65536, n_keep = 4, task.n_tokens = 1000
slot print_timing: id  0 | task 100 | n_gen =    50, tg =  30.00 t/s, tg_3s =  30.00 t/s
slot      release: id  0 | task 100 | stop processing: n_tokens = 1050, truncated = 0
[GIN] 2026/08/30 - 21:16:50 | 200 |       5.000s |       127.0.0.1 | POST     "/v1/chat/completions"
""".strip().splitlines()

ABORT_SAMPLE = """
slot launch_slot_: id  0 | task 200 | processing task, is_child = 0
slot print_timing: id  0 | task 200 | n_gen =    10, tg =  20.00 t/s, tg_3s =  20.00 t/s
time=2026-08-30T21:16:45.000-07:00 level=INFO source=llama_server.go:1538 msg="aborting completion request due to client closing the connection"
""".strip().splitlines()


def test_parse_gin_and_prompt():
    now = datetime(2026, 8, 30, 21, 17, 0, tzinfo=timezone.utc)
    parsed = parse_server_log_activity(SAMPLE, now=now, fresh_seconds=300)
    assert len(parsed.inference_requests) == 1
    assert parsed.inference_requests[0].path == "/api/embed"
    assert parsed.last_task and parsed.last_task["task_id"] == 11779
    assert parsed.last_prompt and parsed.last_prompt["progress"] == 0.47
    assert parsed.last_generation and parsed.last_generation["task_id"] == 11779
    assert parsed.open_phase == "generating"


def test_build_activity_prompt_phase(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("\n".join(SAMPLE) + "\n", encoding="utf-8")
    proc_rows = [
        {"pid": 123, "name": "llama-server.exe", "bytes": 17_000_000_000, "engine_3d_pct": 42.0}
    ]
    act = build_server_activity(
        log_path=log, proc_rows=proc_rows, include_peers=False, tcp_peers=[]
    )
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


def test_monitor_vs_inference_paths():
    assert is_monitor_request("GET", "/api/ps")
    assert is_monitor_request("POST", "/api/show")
    assert not is_monitor_request("POST", "/v1/chat/completions")
    assert is_inference_request("POST", "/v1/chat/completions")
    assert is_inference_request("POST", "/api/embed")
    assert not is_inference_request("GET", "/api/tags")


def test_noise_filtered_from_recent(tmp_path):
    now = datetime(2026, 8, 30, 21, 17, 0, tzinfo=timezone.utc)
    log = tmp_path / "server.log"
    log.write_text("\n".join(NOISE_AND_CHAT) + "\n", encoding="utf-8")
    act = build_server_activity(
        log_path=log,
        now=now,
        fresh_seconds=300,
        include_peers=False,
        tcp_peers=[],
        peer_names={"100.75.27.13": "open-webui"},
    )
    assert act.last_request is not None
    assert act.last_request.path == "/v1/chat/completions"
    assert act.last_request.client_name == "open-webui"
    assert all(not is_monitor_request(r.method, r.path) for r in act.recent_requests)


def test_n_gen_phase_without_mtp(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("\n".join(N_GEN_SAMPLE) + "\n", encoding="utf-8")
    act = build_server_activity(
        log_path=log,
        include_peers=True,
        tcp_peers=["100.75.27.13"],
        peer_names={"100.75.27.13": "open-webui"},
        models=[{"name": "qwen3.8:27b-heretic"}],
    )
    assert act.phase == "generating"
    assert act.n_gen == 846
    assert act.gen_tps == 30.68
    assert act.gen_tps_3s == 32.37
    assert act.n_ctx_slot == 65536
    assert act.prompt_tokens == 45259
    assert act.ctx_fill is not None and act.ctx_fill > 0.6
    assert act.model == "qwen3.8:27b-heretic"
    assert "qwen3.8:27b-heretic" in act.summary
    assert act.peers and act.peers[0].name == "open-webui"


def test_release_returns_idle_with_last_inference(tmp_path):
    now = datetime(2026, 8, 30, 21, 17, 0, tzinfo=timezone.utc)
    log = tmp_path / "server.log"
    log.write_text("\n".join(COMPLETED_SAMPLE) + "\n", encoding="utf-8")
    act = build_server_activity(
        log_path=log, now=now, fresh_seconds=300, include_peers=False, tcp_peers=[]
    )
    assert act.phase in ("idle", "request")
    assert act.n_gen is None or act.phase == "request"
    assert act.last_request and act.last_request.path == "/v1/chat/completions"


def test_abort_clears_generating(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("\n".join(ABORT_SAMPLE) + "\n", encoding="utf-8")
    act = build_server_activity(log_path=log, include_peers=False, tcp_peers=[])
    assert act.phase == "idle"
    assert "abort" in act.summary.lower()


def test_peer_name_map_and_parsers():
    names = build_peer_name_map(
        [{"name": "hermes", "addrs": ["127.0.0.1", "::1"]}, {"name": "x"}]
    )
    assert names["127.0.0.1"] == "hermes"
    assert names["::1"] == "hermes"

    assert parse_powershell_peers("RemoteAddress\n127.0.0.1\n100.1.2.3\n127.0.0.1\n") == [
        "127.0.0.1",
        "100.1.2.3",
    ]
    ss = (
        "tcp   ESTAB 0 0 0.0.0.0:11434 127.0.0.1:54321\n"
        "tcp   ESTAB 0 0 [::]:11434 [::1]:9999\n"
    )
    assert parse_ss_peers(ss) == ["127.0.0.1", "::1"]
    assert list_tcp_peers(11434, peers=["1.2.3.4"]) == ["1.2.3.4"]
