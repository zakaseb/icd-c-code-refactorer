"""Verification streams tokens while the model is still writing.

A buffered reply left the browser idle after script generation. These tests
do not call a model.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "api"))
os.environ.setdefault("WORKSPACE_DIR", tempfile.mkdtemp(prefix="icd_verify_stream_"))

import app as appmod  # noqa: E402

PASS = 0
FAIL = 0


def _check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}: {detail}")


def _frames(gen):
    frames = []
    while True:
        try:
            frames.append(next(gen))
        except StopIteration as stop:
            return frames, stop.value


def test_tokens_leave_before_the_continuation_pass():
    original = appmod._call_llm_stream
    pulled: list[str] = []

    def fake_stream(_system, user, max_tokens=0, meta=None):
        if "cut off" not in user:
            meta["finish_reason"] = "length"
            yield "HELLO-TOKEN"
            return
        already = any("HELLO-TOKEN" in frame for frame in pulled)
        meta["finish_reason"] = "stop"
        meta["already_streamed"] = already
        yield "TAIL-TOKEN"

    appmod._call_llm_stream = fake_stream
    try:
        gen = appmod._stream_llm_to_sse(
            "system",
            "verify this file",
            stage="verification",
            filename="boImuIn.c",
            max_tokens=64,
            max_passes=2,
        )
        while True:
            try:
                pulled.append(next(gen))
            except StopIteration as stop:
                full = stop.value
                break
    finally:
        appmod._call_llm_stream = original

    _check("two_passes_returned", full == "HELLO-TOKEN\nTAIL-TOKEN", repr(full))
    _check(
        "first_token_before_continuation",
        any("HELLO-TOKEN" in frame for frame in pulled[:-1]),
        repr(pulled),
    )
    payloads = []
    for frame in pulled:
        _check_prefix = frame.startswith("data: ")
        if not _check_prefix:
            payloads.append(None)
            continue
        payloads.append(json.loads(frame[len("data: "):].strip()))
    _check("sse_frames", all(payloads), repr(pulled))
    kinds = [(p or {}).get("type") for p in payloads]
    _check(
        "token_then_continue_then_token",
        kinds == ["token", "info", "token"],
        repr(kinds),
    )
    _check(
        "stage_and_file",
        all(
            (p or {}).get("stage") == "verification"
            and (p or {}).get("file") == "boImuIn.c"
            for p in payloads
        ),
        repr(payloads),
    )


def main() -> int:
    print("Running test_verification_stream")
    test_tokens_leave_before_the_continuation_pass()
    print(f"{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
