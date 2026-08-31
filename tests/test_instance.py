"""Single-instance lock tests."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from ollama_sentinel.instance import InstanceLock, LOCK_NAME, SHOW_REQUEST_NAME


class TestInstanceLock(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(os.getenv("TEMP", "/tmp")) / f"sentinel-lock-test-{os.getpid()}"
        self._tmp.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        lock = InstanceLock(base_dir=self._tmp)
        lock.release()
        lock2 = InstanceLock(base_dir=self._tmp)
        lock2.release()
        import gc
        gc.collect()
        for name in (LOCK_NAME, SHOW_REQUEST_NAME):
            p = self._tmp / name
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

    def test_acquire_release_round_trip(self) -> None:
        lock = InstanceLock(base_dir=self._tmp)
        self.assertTrue(lock.try_acquire("gui"))
        lock.release()
        lock2 = InstanceLock(base_dir=self._tmp)
        self.assertTrue(lock2.try_acquire("gui"))
        lock2.release()

    def test_second_acquire_fails_while_held(self) -> None:
        first = InstanceLock(base_dir=self._tmp)
        second = InstanceLock(base_dir=self._tmp)
        self.assertTrue(first.try_acquire("gui"))
        self.assertFalse(second.try_acquire("gui"))
        first.release()

    def test_stale_lock_reclaimed(self) -> None:
        lock_path = self._tmp / LOCK_NAME
        lock_path.write_text('{"pid": 999999999, "mode": "gui"}', encoding="utf-8")
        lock = InstanceLock(base_dir=self._tmp)
        self.assertTrue(lock.try_acquire("gui"))
        lock.release()

    def test_show_request_round_trip(self) -> None:
        holder = InstanceLock(base_dir=self._tmp)
        self.assertTrue(holder.try_acquire("gui"))
        duplicate = InstanceLock(base_dir=self._tmp)
        duplicate.request_show()
        self.assertTrue(holder.consume_show_request())
        self.assertFalse(holder.consume_show_request())
        holder.release()

    def test_is_holder_alive_dead_pid(self) -> None:
        lock_path = self._tmp / LOCK_NAME
        lock_path.write_text('{"pid": 999999999, "mode": "gui"}', encoding="utf-8")
        self.assertFalse(InstanceLock.is_holder_alive(lock_path))

    def test_is_holder_alive_current_pid(self) -> None:
        lock_path = self._tmp / LOCK_NAME
        lock_path.write_text(f'{{"pid": {os.getpid()}, "mode": "gui"}}', encoding="utf-8")
        self.assertTrue(InstanceLock.is_holder_alive(lock_path))


class TestResolveGuiOptions(unittest.TestCase):
    def test_windows_tray_default(self) -> None:
        from argparse import Namespace

        from ollama_sentinel.config import resolve_gui_options

        with patch("ollama_sentinel.config.sys.platform", "win32"):
            tray, hidden = resolve_gui_options(Namespace(start_minimized=False, no_tray=False))
        self.assertTrue(tray)
        self.assertFalse(hidden)

    def test_no_tray(self) -> None:
        from argparse import Namespace

        from ollama_sentinel.config import resolve_gui_options

        with patch("ollama_sentinel.config.sys.platform", "win32"):
            tray, hidden = resolve_gui_options(Namespace(start_minimized=False, no_tray=True))
        self.assertFalse(tray)

    def test_start_minimized(self) -> None:
        from argparse import Namespace

        from ollama_sentinel.config import resolve_gui_options

        with patch("ollama_sentinel.config.sys.platform", "linux"):
            tray, hidden = resolve_gui_options(Namespace(start_minimized=True, no_tray=False))
        self.assertFalse(tray)
        self.assertTrue(hidden)


if __name__ == "__main__":
    unittest.main()
