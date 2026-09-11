"""The GUI log sink: under pythonw, stderr is None and lastResort drops everything."""

from __future__ import annotations

import logging
import logging.handlers
import tempfile
import unittest
from pathlib import Path

from ollama_sentinel.ui import setup_gui_logging


class SetupGuiLoggingTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("ollama_sentinel")
        self._saved_handlers = list(self.logger.handlers)
        self._saved_level = self.logger.level
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.log_dir = Path(self._tmp.name)

    def tearDown(self):
        # Restore exactly the pre-test handlers so the temp-dir handler (with
        # its open file on Windows) cannot leak into other tests.
        for handler in list(self.logger.handlers):
            if handler not in self._saved_handlers:
                self.logger.removeHandler(handler)
                handler.close()
        self.logger.setLevel(self._saved_level)

    def _rotating_for(self, path: Path) -> list[logging.handlers.RotatingFileHandler]:
        return [
            h
            for h in self.logger.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
            and h.baseFilename == str(path)
        ]

    def test_attaches_one_rotating_handler_and_is_idempotent(self):
        path = setup_gui_logging(self.log_dir)
        self.assertIsNotNone(path)
        self.assertEqual(path, self.log_dir / "gui.log")
        self.assertTrue(path.exists())
        first = self._rotating_for(path)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].maxBytes, 1_000_000)
        self.assertEqual(first[0].backupCount, 3)

        again = setup_gui_logging(self.log_dir)
        self.assertEqual(again, path)
        self.assertEqual(len(self._rotating_for(path)), 1, "no second handler for same path")

    def test_records_from_child_loggers_land_in_the_file(self):
        path = setup_gui_logging(self.log_dir)
        logging.getLogger("ollama_sentinel.ui").warning("sink round-trip check")
        for handler in self._rotating_for(path):
            handler.flush()
        text = path.read_text(encoding="utf-8")
        self.assertIn("sink round-trip check", text)
        self.assertIn("WARNING", text)

    def test_logger_level_is_raised_to_info(self):
        self.logger.setLevel(logging.WARNING)
        setup_gui_logging(self.log_dir)
        self.assertLessEqual(self.logger.level, logging.INFO)


if __name__ == "__main__":
    unittest.main()
