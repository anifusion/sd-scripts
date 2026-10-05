import json
import unittest

from library.operational_diagnostic import exception_diagnostic


class OperationalDiagnosticTests(unittest.TestCase):
    def test_keeps_bounded_message_cause_and_stack(self):
        try:
            try:
                raise ValueError("invalid parameter")
            except ValueError as cause:
                raise RuntimeError("prompt parse failed") from cause
        except RuntimeError as error:
            diagnostic = json.loads(exception_diagnostic(error, "prompt_parse"))
        self.assertEqual(diagnostic["stage"], "prompt_parse")
        self.assertEqual(diagnostic["message"], "prompt parse failed")
        self.assertEqual(diagnostic["cause_message"], "invalid parameter")
        self.assertTrue(diagnostic["frames"])

    def test_throwing_exception_string_fails_closed(self):
        class HostileError(RuntimeError):
            def __str__(self):
                raise RuntimeError("private content")

        diagnostic = json.loads(exception_diagnostic(HostileError(), "prompt_parse"))
        self.assertEqual(diagnostic, {"stage": "prompt_parse", "type": "UnknownError"})


if __name__ == "__main__":
    unittest.main()
