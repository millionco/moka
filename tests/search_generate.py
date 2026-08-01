import subprocess
import unittest
from unittest.mock import Mock

from go_model.config import SEARCH_TEACHER_SHUTDOWN_TIMEOUT_SECONDS
from go_model.search_generate import KataGoAnalysisEngine


class KataGoAnalysisEngineTest(unittest.TestCase):
    def create_engine(self) -> KataGoAnalysisEngine:
        engine = object.__new__(KataGoAnalysisEngine)
        engine.standard_input = Mock()
        engine.process = Mock()
        engine.error_thread = Mock()
        return engine

    def test_close_terminates_an_unresponsive_process(self) -> None:
        engine = self.create_engine()
        engine.process.wait.side_effect = [
            subprocess.TimeoutExpired(
                "katago",
                SEARCH_TEACHER_SHUTDOWN_TIMEOUT_SECONDS,
            ),
            0,
        ]

        engine.close()

        engine.standard_input.close.assert_called_once_with()
        engine.process.terminate.assert_called_once_with()
        engine.process.kill.assert_not_called()

    def test_close_kills_a_process_that_ignores_termination(self) -> None:
        engine = self.create_engine()
        engine.process.wait.side_effect = [
            subprocess.TimeoutExpired(
                "katago",
                SEARCH_TEACHER_SHUTDOWN_TIMEOUT_SECONDS,
            ),
            subprocess.TimeoutExpired(
                "katago",
                SEARCH_TEACHER_SHUTDOWN_TIMEOUT_SECONDS,
            ),
            0,
        ]

        engine.close()

        engine.process.terminate.assert_called_once_with()
        engine.process.kill.assert_called_once_with()
        self.assertEqual(engine.process.wait.call_count, 3)


if __name__ == "__main__":
    unittest.main()
