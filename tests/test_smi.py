"""nvidia-smi parse tests."""

import unittest

from ollama_sentinel.smi import parse_nvidia_smi_csv, parse_rocm_smi_vram

LIVE_SAMPLE = (
    "NVIDIA GeForce RTX 3090, 33, 50 %, 1 %, 25 %, 210 MHz, 810 MHz, P5, "
    "23296 MiB, 24576 MiB, 254 MiB, 38.40 W, 350.00 W, 350.00 W, Not Active, Not Active"
)


class TestSmi(unittest.TestCase):
    def test_live_sample(self):
        gpus = parse_nvidia_smi_csv(LIVE_SAMPLE)
        self.assertEqual(len(gpus), 1)
        gpu = gpus[0]
        self.assertEqual(gpu["name"], "NVIDIA GeForce RTX 3090")
        self.assertEqual(gpu["temperature"], 33)
        self.assertEqual(gpu["fan_speed"], 50)
        self.assertEqual(gpu["utilization"], 1)
        self.assertEqual(gpu["memory_utilization"], 25)
        self.assertEqual(gpu["clock_sm"], 210)
        self.assertEqual(gpu["clock_mem"], 810)
        self.assertEqual(gpu["pstate"], "P5")
        self.assertEqual(gpu["memory_used"], 23296 * 1024 * 1024)
        self.assertEqual(gpu["memory_total"], 24576 * 1024 * 1024)
        self.assertEqual(gpu["memory_reserved"], 254 * 1024 * 1024)
        self.assertAlmostEqual(gpu["memory_free"], (24576 - 23296) * 1024 * 1024, delta=1)
        self.assertEqual(gpu["power_draw"], 38.40)
        self.assertEqual(gpu["power_limit"], 350.00)
        self.assertEqual(gpu["throttle_hw_thermal"], "Not Active")
        self.assertEqual(gpu["throttle_sw_power_cap"], "Not Active")

    def test_throttle_active(self):
        csv = (
            "GPU, 40, 30 %, 99 %, 80 %, 1800 MHz, 9500 MHz, P0, "
            "20000 MiB, 24576 MiB, 254 MiB, 100 W, 350 W, 350 W, Active, Not Active"
        )
        gpu = parse_nvidia_smi_csv(csv)[0]
        self.assertEqual(gpu["throttle_hw_thermal"], "Active")
        self.assertEqual(gpu["throttle_sw_power_cap"], "Not Active")

    def test_na_fan_and_throttle(self):
        csv = (
            "GPU, 40, [N/A], 50 %, 25 %, 210 MHz, 810 MHz, P5, "
            "1000 MiB, 24576 MiB, 254 MiB, 50 W, [N/A], 350 W, [N/A], [Not Supported]"
        )
        gpu = parse_nvidia_smi_csv(csv)[0]
        self.assertIsNone(gpu["fan_speed"])
        self.assertIsNone(gpu["power_limit"])
        self.assertIsNone(gpu["throttle_hw_thermal"])
        self.assertIsNone(gpu["throttle_sw_power_cap"])

    def test_two_gpus(self):
        csv = LIVE_SAMPLE + "\n" + LIVE_SAMPLE.replace("3090", "3080")
        gpus = parse_nvidia_smi_csv(csv)
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0]["index"], 0)
        self.assertEqual(gpus[1]["index"], 1)

    def test_rocm_vram_parse(self):
        sample = """
GPU[0]          : VRAM Total Memory (B): 17163091968
GPU[0]          : VRAM Total Used Memory (B): 15032385536
GPU[1]          : VRAM Total Memory (B): 8589934592
GPU[1]          : VRAM Total Used Memory (B): 1073741824
"""
        gpus = parse_rocm_smi_vram(sample)
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0]["memory_total"], 17163091968)
        self.assertEqual(gpus[0]["memory_used"], 15032385536)
        self.assertEqual(gpus[0]["name"], "AMD GPU")


if __name__ == "__main__":
    unittest.main()
