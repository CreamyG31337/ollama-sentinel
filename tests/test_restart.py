"""Restart helper tests."""

import json
import sys
import unittest
from unittest.mock import MagicMock, patch

from ollama_sentinel.restart import _wait_and_spawn, main, restart_command, spawn_restart


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

    @patch("ollama_sentinel.restart.subprocess.Popen")
    @patch("ollama_sentinel.instance._pid_alive", return_value=False)
    @patch("ollama_sentinel.restart.time.sleep")
    def test_wait_and_spawn_starts_after_parent_gone(
        self, _sleep: MagicMock, _alive: MagicMock, popen: MagicMock
    ) -> None:
        cmd = ["pythonw.exe", "-m", "ollama_sentinel", "--gui"]
        _wait_and_spawn([1234], cmd, "C:/proj")
        popen.assert_called_once()
        self.assertEqual(popen.call_args.kwargs["cwd"], "C:/proj")
        self.assertEqual(popen.call_args.args[0], cmd)

    @patch("ollama_sentinel.restart.os._exit")
    @patch("ollama_sentinel.restart.subprocess.Popen")
    @patch("ollama_sentinel.restart.restart_command", return_value=["py", "-m", "ollama_sentinel", "--gui"])
    @patch("ollama_sentinel.restart.os.getpid", return_value=999)
    @patch("ollama_sentinel.restart.child_pids", return_value=[1001])
    @patch("ollama_sentinel.restart._terminate")
    def test_spawn_restart_exits_current_process(
        self,
        terminate: MagicMock,
        _children: MagicMock,
        _pid: MagicMock,
        _cmd: MagicMock,
        popen: MagicMock,
        exit_: MagicMock,
    ) -> None:
        spawn_restart()
        popen.assert_called_once()
        helper = popen.call_args.args[0]
        self.assertEqual(helper[2], "ollama_sentinel.restart")
        self.assertEqual(helper[3], "--wait-spawn")
        # The helper must wait on the flet window child too, not just this pid.
        self.assertEqual(json.loads(helper[4]), [999, 1001])
        self.assertEqual(json.loads(helper[5]), ["py", "-m", "ollama_sentinel", "--gui"])
        # os._exit() would strand the window, so the child is killed explicitly.
        terminate.assert_called_once_with(1001)
        exit_.assert_called_once_with(0)

    @patch("ollama_sentinel.restart.subprocess.Popen")
    @patch("ollama_sentinel.instance._pid_alive")
    @patch("ollama_sentinel.restart.time.sleep")
    def test_wait_blocks_while_the_window_child_lives(
        self, _sleep: MagicMock, alive: MagicMock, popen: MagicMock
    ) -> None:
        """Regression: the parent dies instantly but flet.exe lingers."""
        # parent gone, window child still up for two polls, then gone.
        alive.side_effect = [False, True, False, True, False, False]
        _wait_and_spawn([999, 1001], ["py"], "C:/proj")
        popen.assert_called_once()
        self.assertGreater(alive.call_count, 2, "must keep polling while child lives")

    def test_wait_spawn_cli(self) -> None:
        with patch("ollama_sentinel.restart._wait_and_spawn") as wait:
            code = main(
                [
                    "--wait-spawn",
                    json.dumps([42]),
                    json.dumps(["py", "-m", "ollama_sentinel"]),
                    "C:/proj",
                ]
            )
        self.assertEqual(code, 0)
        wait.assert_called_once_with([42], ["py", "-m", "ollama_sentinel"], "C:/proj")


if __name__ == "__main__":
    unittest.main()
