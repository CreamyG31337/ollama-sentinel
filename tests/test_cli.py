"""CLI entry-point smoke tests — catch import/regression breaks."""

from __future__ import annotations

import importlib
import json
import pkgutil
from pathlib import Path

import pytest

from ollama_sentinel.config import ServerConfig


def _mock_snap() -> list[dict]:
    return [
        {
            "server": "local",
            "url": "http://127.0.0.1:11434",
            "reachable": True,
            "version": "0.0.0",
            "models": [
                {
                    "name": "test",
                    "size": 1_000_000_000,
                    "size_vram": 1_000_000_000,
                    "expires_at": "2026-08-30T20:00:00-08:00",
                }
            ],
            "tags": [],
            "gpus": None,
            "error": None,
            "stale": False,
            "polled_at": "2026-08-31T03:00:00+00:00",
            "polled_at_ts": 1_700_000_000.0,
        }
    ]


@pytest.fixture
def cli_mocks(monkeypatch, tmp_path: Path):
    state_file = tmp_path / "state.json"
    servers = [ServerConfig(name="local", url="http://127.0.0.1:11434", local_gpu=True)]

    monkeypatch.setattr("ollama_sentinel.__main__.poll_all", lambda *a, **k: _mock_snap())
    monkeypatch.setattr("ollama_sentinel.__main__.query_gpus", lambda *a, **k: [])
    monkeypatch.setattr("ollama_sentinel.__main__.query_process_vram", lambda *a, **k: [])
    monkeypatch.setattr("ollama_sentinel.__main__.selected_servers", lambda cfg: servers)
    monkeypatch.setattr("ollama_sentinel.__main__.collect_doctor_inputs", lambda: {})
    monkeypatch.setattr("ollama_sentinel.__main__._doctor_findings_for_snapshots", lambda *a, **k: [])

    return state_file


def test_once_runs_end_to_end(cli_mocks):
    from ollama_sentinel.__main__ import main

    code = main(["--once", "--state-file", str(cli_mocks)])
    assert code in (0, 1, 2)


def test_once_json_runs_and_preserves_iso(cli_mocks, capsys):
    from ollama_sentinel.__main__ import main

    code = main(["--json", "--once", "--state-file", str(cli_mocks)])
    assert code in (0, 1, 2)
    payload = json.loads(capsys.readouterr().out)
    snap = payload["snapshots"][0]
    assert snap["models"][0]["expires_at"] == "2026-08-30T20:00:00-08:00"
    assert "polled_at" in snap
    assert "T" in snap["models"][0]["expires_at"]


def test_list_runs_end_to_end(cli_mocks):
    from ollama_sentinel.__main__ import main

    code = main(["--list", "--state-file", str(cli_mocks)])
    assert code in (0, 1, 2)


def test_main_imports_load_state():
    """Regression: gaming-yield refactor dropped load_state import."""
    import ollama_sentinel.__main__ as entry

    assert callable(entry.load_state)
    assert callable(entry.save_state)


def test_all_package_modules_import():
    import ollama_sentinel

    skip = {"ollama_sentinel.ui"}  # flet optional at import in some envs
    failures: list[str] = []
    for info in pkgutil.walk_packages(ollama_sentinel.__path__, ollama_sentinel.__name__ + "."):
        if info.name in skip:
            continue
        try:
            importlib.import_module(info.name)
        except Exception as exc:
            failures.append(f"{info.name}: {exc}")
    assert not failures, "Import failures:\n" + "\n".join(failures)
