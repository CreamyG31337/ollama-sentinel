"""UI widget helper tests."""

import unittest

from ollama_sentinel.ui_widgets import (
    advisory_summary,
    advisor_detail_lines,
    advisor_headline,
    alarm_state,
    fit_label,
)


class TestFitLabel(unittest.TestCase):
    def test_loaded(self) -> None:
        text, color = fit_label({"loaded": True, "gpu_pct": 100})
        self.assertIn("100", text)
        self.assertIsNotNone(color)

    def test_would_spill(self) -> None:
        text, color = fit_label({"loaded": False, "would_spill": True})
        self.assertEqual(text, "would spill")
        self.assertIsNotNone(color)

    def test_fits(self) -> None:
        text, _color = fit_label({"loaded": False, "would_spill": False})
        self.assertEqual(text, "fits")

    def test_unknown(self) -> None:
        text, color = fit_label({"loaded": False})
        self.assertEqual(text, "—")
        self.assertIsNone(color)

    def test_fit_unknown(self) -> None:
        text, color = fit_label({"loaded": False, "would_spill": None})
        self.assertEqual(text, "fit unknown")
        self.assertIsNotNone(color)


class TestAdvisorySummary(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(advisory_summary([]), "—")

    def test_uses_message_not_id_token(self) -> None:
        from ollama_sentinel.advisor import AdvisorFinding

        findings = [
            AdvisorFinding(
                category="fit",
                severity="warn",
                confidence="medium",
                id="fit:would_spill:big",
                message="big may not fit in free VRAM",
            )
        ]
        summary = advisory_summary(findings)
        self.assertIn("may not fit", summary)
        self.assertNotIn("would_spill", summary)

    def test_strips_server_prefix(self) -> None:
        from ollama_sentinel.advisor import AdvisorFinding

        findings = [
            AdvisorFinding(
                category="fit",
                severity="info",
                confidence="high",
                id="fit:gpu_unknown:local",
                message="[local] No GPU telemetry — fit advisories skipped",
            )
        ]
        self.assertTrue(advisory_summary(findings).startswith("No GPU"))


class TestAdvisorPanelText(unittest.TestCase):
    def test_headline_notes(self) -> None:
        from ollama_sentinel.advisor import AdvisorFinding

        findings = [
            AdvisorFinding("info", "info", "high", "a", "one"),
            AdvisorFinding("info", "info", "high", "b", "two"),
        ]
        self.assertEqual(advisor_headline(findings), "Advisor: 2 notes")

    def test_headline_warnings(self) -> None:
        from ollama_sentinel.advisor import AdvisorFinding

        findings = [
            AdvisorFinding("runtime", "warn", "high", "a", "spill"),
            AdvisorFinding("info", "info", "high", "b", "note"),
        ]
        self.assertEqual(advisor_headline(findings), "Advisor: 1 warning")

    def test_detail_lines_include_messages_and_remedy(self) -> None:
        from ollama_sentinel.advisor import AdvisorFinding

        findings = [
            AdvisorFinding(
                "info",
                "info",
                "high",
                "info:mtp",
                "MTP note about speculative decoding",
            ),
            AdvisorFinding(
                "runtime",
                "warn",
                "high",
                "runtime:spill",
                "Model is spilling to system RAM",
                remedy="Unload another model or use a smaller quant",
            ),
        ]
        lines = advisor_detail_lines(findings)
        texts = [t for _, t in lines]
        self.assertEqual(lines[0][0], "warn")
        self.assertIn("spilling", texts[0])
        self.assertTrue(any(t.startswith("→ ") for t in texts))
        self.assertTrue(any("MTP" in t for t in texts))


class TestAlarmState(unittest.TestCase):
    def test_ok(self) -> None:
        title, body, key = alarm_state(True, [])
        self.assertEqual(title, "OK")
        self.assertEqual(key, "ok")

    def test_unreachable(self) -> None:
        title, _body, key = alarm_state(False, [])
        self.assertEqual(title, "Unreachable")
        self.assertEqual(key, "alarm")

    def test_alarms_body(self) -> None:
        title, body, key = alarm_state(
            True, [{"type": "spill", "message": "model spilled"}]
        )
        self.assertEqual(title, "Alarms")
        self.assertIn("spilled", body)
        self.assertEqual(key, "warn")

    def test_paging_alarm(self) -> None:
        active = [{"type": "paging", "message": "PAGING test"}]
        _title, _body, key = alarm_state(True, active)
        self.assertEqual(key, "alarm")


if __name__ == "__main__":
    unittest.main()
