"""Restart helper tests."""

import sys
import unittest
from unittest.mock import patch

from ollama_sentinel.restart import restart_command


class TestRestart(unittest.TestCase):
    @patch.object(sys, "argv", ["ollama_sentinel/__main__.py", "--gui", "--start-minimized"])
    @patch.object(sys, "executable", "pythonw.exe")
    def test_module_launch(self) -> None:
        self.assertEqual(
            restart_command(),
            ["pythonw.exe", "-m", "ollama_sentinel", "--gui", "--start-minimized"],
        )

    @patch.object(sys, "argv", ["pythonw.exe", "-m", "ollama_sentinel", "--gui"])
    @patch.object(sys, "executable", "pythonw.exe")
    def test_passthrough_m_form(self) -> None:
        self.assertEqual(
            restart_command(),
            ["pythonw.exe", "-m", "ollama_sentinel", "--gui"],
        )


if __name__ == "__main__":
    unittest.main()
