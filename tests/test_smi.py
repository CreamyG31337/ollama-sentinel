"""nvidia-smi parse tests."""

import unittest

from ollama_sentinel.smi import parse_nvidia_smi_csv


class TestSmi(unittest.TestCase):
    def test_two_gpus(self):
        csv = (
            "RTX 3090, 1000, 24576, 50, 100, 350, 60\n"
            "RTX 3090, 2000, 24576, 30, 80, 350, 55\n"
        )
        gpus = parse_nvidia_smi_csv(csv)
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0]["index"], 0)

    def test_na_power_limit(self):
        csv = "GPU, 1000, 24576, 50, 100, [N/A], 60\n"
        gpus = parse_nvidia_smi_csv(csv)
        self.assertIsNone(gpus[0]["power_limit"])


if __name__ == "__main__":
    unittest.main()
