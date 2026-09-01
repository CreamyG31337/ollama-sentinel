"""Advisor heuristic tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from ollama_sentinel.advisor import (
    AdvisorFinding,
    advisories_for_model,
    evaluate_advisories,
    evaluate_advisor_alarms,
)
from ollama_sentinel.client_config import missing_client_models
from ollama_sentinel.show_parse import parse_show_bundle

FIXTURES = Path(__file__).parent / "fixtures"


def _snap(**overrides):
    base = {
        "server": "local",
        "reachable": True,
        "version": "0.33.2",
        "tags": [],
        "models": [],
        "gpus": [{"memory_used": 2e9, "memory_total": 24e9, "memory_free": 22e9}],
        "gpu_data_available": True,
    }
    base.update(overrides)
    return base


class TestAdvisor(unittest.TestCase):
    def test_mtp_dormant(self) -> None:
        raw = json.loads((FIXTURES / "api_show_qwen38_mtp.json").read_text(encoding="utf-8"))
        show = parse_show_bundle(raw)
        show["draft_num_predict"] = 0
        snap = _snap(tags=[{"name": "qwen3.8:27b-mtp-q4_K_M", "size": 18e9}])
        findings = evaluate_advisories(snap, show_by_model={"qwen3.8:27b-mtp-q4_K_M": show})
        ids = [f.id for f in findings]
        self.assertIn("model:mtp_dormant:qwen3.8:27b-mtp-q4_K_M", ids)

    def test_fit_would_spill(self) -> None:
        snap = _snap(
            tags=[{"name": "huge", "size": 30e9, "details": {"quantization_level": "Q8_0"}}],
            gpus=[{"memory_used": 22e9, "memory_total": 24e9}],
        )
        findings = evaluate_advisories(snap)
        self.assertTrue(any(f.id.startswith("fit:would_spill:") for f in findings))

    def test_gpu_unknown_skips_fit(self) -> None:
        snap = _snap(
            tags=[{"name": "big", "size": 30e9}],
            gpus=None,
            gpu_data_available=False,
        )
        findings = evaluate_advisories(snap, gpu_data_available=False)
        self.assertTrue(any(f.id.startswith("fit:gpu_unknown:") for f in findings))
        self.assertFalse(any(f.id.startswith("fit:would_spill:") for f in findings))

    def test_client_model_missing(self) -> None:
        missing = missing_client_models(
            [{"name": "open-webui", "models": ["missing:tag"]}],
            {"installed:tag"},
        )
        snap = _snap(tags=[{"name": "installed:tag", "size": 1e9}])
        findings = evaluate_advisories(snap, client_missing=missing)
        self.assertTrue(any(f.id.startswith("config:client_model_missing:") for f in findings))

    def test_advisor_alarms_skip_low_confidence(self) -> None:
        findings = [
            AdvisorFinding(
                category="name",
                severity="warn",
                confidence="low",
                id="name:generation_stale:x",
                message="stale",
            ),
            AdvisorFinding(
                category="fit",
                severity="warn",
                confidence="medium",
                id="fit:would_spill:x",
                message="tight",
            ),
        ]
        alarms = evaluate_advisor_alarms(findings)
        self.assertEqual(len(alarms), 1)
        self.assertEqual(alarms[0]["type"], "advisor")

    def test_advisories_for_model(self) -> None:
        findings = [
            AdvisorFinding("model", "warn", "high", "a", "one", model="m1"),
            AdvisorFinding("fit", "info", "high", "b", "two", model="m2"),
        ]
        self.assertEqual(len(advisories_for_model(findings, "m1")), 1)


if __name__ == "__main__":
    unittest.main()
