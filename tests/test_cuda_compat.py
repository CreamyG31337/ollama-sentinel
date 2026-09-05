"""CUDA / display-driver compatibility probes."""

from __future__ import annotations

import unittest

from ollama_sentinel.cuda_compat import (
    evaluate_cuda_probe,
    parse_inference_compute,
    parse_smi_cuda_version,
)
from ollama_sentinel.doctor import check_cuda_compat, evaluate_doctor_alarms


INFERENCE_OK = (
    'time=2026-09-05T16:03:35.257-07:00 level=INFO source=types.go:32 '
    'msg="inference compute" id=0 filter_id=0 library=CUDA compute=8.6 name=CUDA0 '
    'description="NVIDIA GeForce RTX 3090" libdirs=ollama,cuda_v13 driver=13.4 '
    'pci_id=0000:01:00.0 type=discrete total="24.0 GiB" available="22.8 GiB"\n'
)

SMI_BANNER = """\
Sat Sep  5 16:07:31 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 616.56                 KMD Version: 616.56        CUDA UMD Version: 13.4     |
+-----------------------------------------+------------------------+----------------------+
"""


class ParseTests(unittest.TestCase):
    def test_parse_smi_cuda_umd(self):
        self.assertEqual(parse_smi_cuda_version(SMI_BANNER), "13.4")

    def test_parse_inference_compute(self):
        rec = parse_inference_compute(INFERENCE_OK)
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertEqual(rec["library"], "CUDA")
        self.assertEqual(rec["cuda_major"], 13)
        self.assertEqual(rec["driver"], "13.4")


class EvaluateTests(unittest.TestCase):
    def test_ok_when_aligned(self):
        rec = parse_inference_compute(INFERENCE_OK)
        probe = evaluate_cuda_probe(
            driver_cuda="13.4",
            inference=rec,
            driver_version="616.56",
        )
        self.assertTrue(probe.ok)

    def test_mismatch_when_smi_cuda_behind_bundle(self):
        rec = parse_inference_compute(INFERENCE_OK)
        probe = evaluate_cuda_probe(
            driver_cuda="12.4",
            inference=rec,
            driver_version="550.54",
        )
        self.assertFalse(probe.ok)
        self.assertIn("cuda_v13", probe.reason or "")

    def test_mismatch_when_library_not_cuda(self):
        bad = INFERENCE_OK.replace("library=CUDA", "library=CPU")
        rec = parse_inference_compute(bad)
        probe = evaluate_cuda_probe(driver_cuda="13.4", inference=rec)
        self.assertFalse(probe.ok)
        self.assertIn("library=CPU", probe.reason or "")


class DoctorWireTests(unittest.TestCase):
    def test_check_cuda_compat_pass(self):
        findings = check_cuda_compat(
            log_text=INFERENCE_OK,
            driver_version="616.56",
            driver_cuda="13.4",
        )
        self.assertEqual(findings[0].id, "cuda:compat:ok")
        self.assertEqual(findings[0].severity, "pass")

    def test_check_cuda_compat_warn_and_alarm(self):
        findings = check_cuda_compat(
            log_text=INFERENCE_OK,
            driver_version="550.54",
            driver_cuda="12.0",
        )
        self.assertEqual(findings[0].id, "cuda:compat:mismatch")
        self.assertEqual(findings[0].severity, "warn")
        alarms = evaluate_doctor_alarms(findings)
        self.assertEqual(len(alarms), 1)
        self.assertEqual(alarms[0]["type"], "cuda")


if __name__ == "__main__":
    unittest.main()
