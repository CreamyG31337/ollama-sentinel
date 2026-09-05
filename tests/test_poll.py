"""Poll behavior when Ollama is unreachable."""

from __future__ import annotations

import time
from unittest.mock import patch

from ollama_sentinel.poll import poll_all, poll_server


def _unreachable(*_args, **_kwargs):
    return None, "Connection refused"


@patch("ollama_sentinel.poll._tcp_connect_error", return_value=None)
@patch("ollama_sentinel.poll._get_json", side_effect=_unreachable)
def test_unreachable_clears_ollama_data(mock_get, _mock_tcp):
    snap = poll_server("http://127.0.0.1:11434", "local")
    assert snap["reachable"] is False
    assert snap["models"] == []
    assert snap["tags"] == []
    assert snap["version"] is None
    assert snap["error"]
    assert snap.get("polled_at_ts") is not None
    assert mock_get.call_count == 3


@patch("ollama_sentinel.poll._tcp_connect_error", return_value="Connection refused — is the service running?")
@patch("ollama_sentinel.poll._get_json")
def test_tcp_fail_skips_http(mock_get, _mock_tcp):
    snap = poll_server("http://127.0.0.1:11434", "local")
    assert snap["reachable"] is False
    assert "refused" in (snap["error"] or "").lower()
    mock_get.assert_not_called()


@patch("ollama_sentinel.poll._tcp_connect_error", return_value="Connection refused — is the service running?")
@patch("ollama_sentinel.poll._get_json")
def test_tcp_fail_still_attaches_gpus(mock_get, _mock_tcp):
    def gpus(_gf):
        return [{"index": 0, "memory_used": 1, "memory_total": 2}]

    snap = poll_server(
        "http://127.0.0.1:11434",
        "local",
        attach_gpus=True,
        query_gpus_fn=gpus,
    )
    assert snap["reachable"] is False
    assert snap["gpus"] == [{"index": 0, "memory_used": 1, "memory_total": 2}]
    mock_get.assert_not_called()


@patch("ollama_sentinel.poll.poll_server")
def test_poll_all_does_not_reuse_last_snapshot(mock_poll):
    good = {
        "server": "local",
        "reachable": True,
        "models": [{"name": "stale-model"}],
        "tags": [{"name": "stale-model"}],
        "version": "0.1.0",
        "stale": False,
    }
    bad = {
        "server": "local",
        "reachable": False,
        "models": [],
        "tags": [],
        "version": None,
        "error": "Connection refused",
        "stale": False,
    }
    mock_poll.side_effect = [bad]

    snaps = poll_all([{"name": "local", "url": "http://127.0.0.1:11434", "local_gpu": False}])
    assert snaps[0]["reachable"] is False
    assert snaps[0]["models"] == []
    assert good["models"]  # unchanged — poll_all must not copy prior state


@patch("ollama_sentinel.poll._tcp_connect_error", return_value=None)
@patch("ollama_sentinel.poll._get_json")
def test_reachable_populates_snapshot(mock_get, _mock_tcp):
    def route(url, path, timeout=10):
        if path == "/api/version":
            return {"version": "0.5.0"}, None
        if path == "/api/ps":
            return {"models": [{"name": "llama3"}]}, None
        if path == "/api/tags":
            return {"models": [{"name": "llama3"}]}, None
        return None, "missing"

    mock_get.side_effect = route
    snap = poll_server("http://127.0.0.1:11434", "local")
    assert snap["reachable"] is True
    assert snap["version"] == "0.5.0"
    assert len(snap["models"]) == 1


def test_connect_timeout_fails_fast_against_refused_host():
    """Don't wait for Windows' ~2s SYN-retry refuse dance."""
    t0 = time.perf_counter()
    snap = poll_server(
        "http://127.0.0.1:59997",
        "dead",
        connect_timeout=0.5,
    )
    elapsed = time.perf_counter() - t0
    assert snap["reachable"] is False
    assert snap["error"]
    assert elapsed < 1.2, f"expected fail-fast, took {elapsed:.2f}s"
