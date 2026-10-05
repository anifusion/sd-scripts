"""Bounded exception details for restricted training diagnostics."""

import json
import traceback


def exception_diagnostic(error, stage):
    try:
        details = {
            "stage": stage,
            "type": type(error).__name__[:80],
            "message": str(error)[:2000],
            "frames": [
                f"{frame.name}:{frame.lineno}"
                for frame in traceback.extract_tb(error.__traceback__)[-6:]
            ],
        }
        cause = error.__cause__ or error.__context__
        if isinstance(cause, BaseException) and cause is not error:
            details["cause_type"] = type(cause).__name__[:80]
            details["cause_message"] = str(cause)[:1000]
        return json.dumps(details, ensure_ascii=True)
    except Exception:
        return json.dumps({"stage": stage, "type": "UnknownError"})
