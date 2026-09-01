"""UI widget helper tests."""

import unittest

from ollama_sentinel.ui_widgets import advisory_summary, alarm_state, fit_label


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

    def test_warn_token(self) -> None:
        from ollama_sentinel.advisor import AdvisorFinding

        findings = [
            AdvisorFinding(
                category="fit",
                severity="warn",
                confidence="medium",
                id="fit:would_spill:big",
                message="tight",
            )
        ]
        self.assertIn("big", advisory_summary(findings))


class TestAlarmState(unittest.TestCase):
    def test_ok(self) -> None:
        title, body, key = alarm_state(True, [])
        self.assertEqual(title, "OK")
        self.assertEqual(key, "ok")

    def test_unreachable(self) -> None:
        title, _body, key = alarm_state(False, [])
        self.assertEqual(title, "Unreachable")
        self.assertEqual(key, "alarm")

    def test_spill_warn(self) -> None:
        active = [{"type": "spill", "message": "SPILL test"}]
        title, body, key = alarm_state(True, active)
        self.assertEqual(title, "Alarms")
        self.assertEqual(key, "warn")
        self.assertIn("SPILL", body)

    def test_paging_alarm(self) -> None:
        active = [{"type": "paging", "message": "PAGING test"}]
        _title, _body, key = alarm_state(True, active)
        self.assertEqual(key, "alarm")


if __name__ == "__main__":
    unittest.main()
