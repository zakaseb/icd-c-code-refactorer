"""Tests for `_call_llm_stream` hardening: prompt clamping, 400-body
surfacing, and automatic recovery on llama-server context overflow.

These tests do *not* talk to a real llama-server.  They monkeypatch
`httpx.Client` with a tiny stub that records the requests it receives
and lets each test assert what the production code sent (clamp,
retry, etc.) and how it reacted to canned 4xx responses.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

# Resolve the api/ package the same way the other dev tests do.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "api"))
os.environ.setdefault("WORKSPACE_DIR", tempfile.mkdtemp(prefix="icd_llm_test_"))

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


# ---------------------------------------------------------------------------
# Stub httpx.Client / stream response
# ---------------------------------------------------------------------------

class _FakeStreamResponse:
    def __init__(
        self,
        *,
        status_code: int,
        body_text: str = "",
        sse_events: Iterable[str] = (),
    ):
        self.status_code = status_code
        self._body_text = body_text
        self._events = list(sse_events)

    # context manager API used by httpx.Client.stream(...)
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._body_text.encode("utf-8")

    def iter_lines(self):
        for ev in self._events:
            yield ev


class _FakeHttpxClient:
    """Records each request, draining a shared scripted-response queue.

    Multiple `httpx.Client(...)` instances (the production code creates
    one per attempt) share the same response queue and request log so
    tests can inspect the full request sequence.
    """

    shared_queue: list[_FakeStreamResponse] = []
    shared_requests: list[dict] = []

    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url, *, json=None, headers=None):
        _FakeHttpxClient.shared_requests.append({
            "method": method,
            "url": url,
            "json": json,
            "headers": headers,
        })
        if not _FakeHttpxClient.shared_queue:
            raise AssertionError("No more scripted responses")
        return _FakeHttpxClient.shared_queue.pop(0)


def _install_fake_client(monkeypatch_responses):
    """Install a `_FakeHttpxClient` factory on appmod.httpx.Client."""
    _FakeHttpxClient.shared_queue = list(monkeypatch_responses)
    _FakeHttpxClient.shared_requests = []
    appmod.httpx.Client = _FakeHttpxClient  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Test 1: pure helper — `_clamp_chat_request`
# ---------------------------------------------------------------------------

def test_clamp_does_nothing_when_within_budget():
    sys_p = "system prompt"
    user_p = "small user prompt"
    sp, up, mt, clamped = appmod._clamp_chat_request(sys_p, user_p, max_tokens=512)
    _check(
        "clamp_no_op_when_fits",
        sp == sys_p and up == user_p and mt == 512 and clamped is False,
        f"got sp_len={len(sp)} up_len={len(up)} mt={mt} clamped={clamped}",
    )


def test_clamp_shrinks_max_tokens_when_overflow_small():
    chars_per_token = appmod.CHARS_PER_TOKEN
    ctx = appmod.CTX_SIZE_TOKENS
    safety = appmod.LLM_PROMPT_SAFETY_TOKENS
    floor = appmod.LLM_MIN_OUTPUT_TOKENS

    sys_p = "S"
    target_user_tokens = ctx - safety - 1500
    user_p = "u" * (target_user_tokens * chars_per_token)
    sp, up, mt, clamped = appmod._clamp_chat_request(sys_p, user_p, max_tokens=2000)

    _check(
        "clamp_reduces_max_tokens",
        clamped is True and floor <= mt < 2000 and len(up) <= len(user_p) + 200,
        f"clamped={clamped} mt={mt} up_grew={len(up) - len(user_p)}",
    )


def test_clamp_tail_truncates_user_prompt_when_far_too_big():
    chars_per_token = appmod.CHARS_PER_TOKEN
    ctx = appmod.CTX_SIZE_TOKENS
    sys_p = "S"
    user_p = "u" * (ctx * chars_per_token * 3) + "TAIL_MARKER"
    sp, up, mt, clamped = appmod._clamp_chat_request(sys_p, user_p, max_tokens=2000)

    fits_now = appmod._estimate_tokens(up) + appmod._estimate_tokens(sp) + mt + 1 \
        <= ctx
    _check(
        "clamp_tail_truncates_huge_prompt",
        clamped is True and len(up) < len(user_p) and fits_now and up.endswith("TAIL_MARKER"),
        f"clamped={clamped} up_len={len(up)} fits={fits_now} ends={up[-15:]!r}",
    )


# ---------------------------------------------------------------------------
# Test 2: `_is_context_overflow_error`
# ---------------------------------------------------------------------------

def test_is_context_overflow_error_detects_known_messages():
    msg1 = (
        '{"error":{"code":400,"message":"the request exceeds the available context size. '
        'try increasing the context size or enable context shift",'
        '"type":"invalid_request_error"}}'
    )
    msg2 = '{"error":{"message":"input is too large for n_ctx=98274"}}'
    msg3 = '{"error":{"message":"some unrelated 400 reason like bad json"}}'

    _check("overflow_msg1", appmod._is_context_overflow_error(400, msg1) is True)
    _check("overflow_msg2", appmod._is_context_overflow_error(400, msg2) is True)
    _check("overflow_msg3_negative", appmod._is_context_overflow_error(400, msg3) is False)
    _check(
        "overflow_500_negative",
        appmod._is_context_overflow_error(500, msg1) is False,
    )


# ---------------------------------------------------------------------------
# Test 3: `_call_llm_stream` — surfaces 400 body in raised exception
# ---------------------------------------------------------------------------

def test_call_llm_stream_surfaces_400_body():
    body = '{"error":{"code":400,"message":"some malformed json field","type":"invalid_request_error"}}'
    _install_fake_client([
        _FakeStreamResponse(status_code=400, body_text=body),
    ])

    raised: Exception | None = None
    try:
        for _ in appmod._call_llm_stream("sys", "user", max_tokens=128):
            pass
    except Exception as e:  # noqa: BLE001
        raised = e

    msg = str(raised) if raised else ""
    _check(
        "call_llm_stream_surfaces_400_body",
        raised is not None
        and "llama-server HTTP 400" in msg
        and "malformed json field" in msg,
        f"raised={raised!r}",
    )


# ---------------------------------------------------------------------------
# Test 4: `_call_llm_stream` — auto-retries on context-overflow 400
# ---------------------------------------------------------------------------

def test_call_llm_stream_retries_on_context_overflow():
    overflow_body = (
        '{"error":{"code":400,"message":"the request exceeds the available '
        'context size. try increasing the context size or enable context '
        'shift","type":"invalid_request_error"}}'
    )
    success_events = [
        'data: {"choices":[{"delta":{"content":"ok"}}]}',
        'data: {"choices":[{"finish_reason":"stop"}]}',
        'data: [DONE]',
    ]

    _install_fake_client([
        _FakeStreamResponse(status_code=400, body_text=overflow_body),
        _FakeStreamResponse(status_code=200, sse_events=success_events),
    ])

    out = list(appmod._call_llm_stream(
        "system small", "user content " * 100, max_tokens=2048,
    ))

    requests = _FakeHttpxClient.shared_requests
    requested_max_tokens = [r["json"]["max_tokens"] for r in requests]

    _check(
        "retry_emits_content_after_overflow",
        out == ["ok"],
        f"out={out!r}",
    )
    _check(
        "retry_request_count",
        len(requests) == 2,
        f"requests={len(requests)}",
    )
    _check(
        "retry_shrinks_max_tokens",
        len(requested_max_tokens) >= 2
        and requested_max_tokens[1] < requested_max_tokens[0]
        and requested_max_tokens[1] >= appmod.LLM_MIN_OUTPUT_TOKENS,
        f"max_tokens seq={requested_max_tokens}",
    )


# ---------------------------------------------------------------------------
# Test 5: iteration defaults bumped to 120
# ---------------------------------------------------------------------------

def test_iteration_caps_are_120():
    _check(
        "agentic_max_attempts_is_120",
        appmod.SANDBOX_AGENTIC_MAX_ATTEMPTS == 120,
        f"got {appmod.SANDBOX_AGENTIC_MAX_ATTEMPTS}",
    )
    # The orchestrator cap may be either an explicit >=120 or None
    # (unlimited): an upstream commit makes 0 / negative mean "no cap"
    # so debug runs can converge without an arbitrary ceiling.  Either
    # is acceptable for the user-facing "iterations >= 120" requirement.
    val = appmod.SANDBOX_ORCH_MAX_STEPS
    _check(
        "orchestrator_max_steps_is_120_or_unlimited",
        val is None or (isinstance(val, int) and val >= 120),
        f"got {val!r}",
    )


# ---------------------------------------------------------------------------
# Test 6: 80B coder-specialised model is the active default
# ---------------------------------------------------------------------------

def test_default_model_is_qwen3_coder_next_80b():
    _check(
        "default_hf_model_qwen3_coder_next",
        "Qwen3-Coder-Next" in appmod.HF_MODEL,
        f"got {appmod.HF_MODEL!r}",
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    print("Running test_llm_request_hardening")

    tests = [
        test_clamp_does_nothing_when_within_budget,
        test_clamp_shrinks_max_tokens_when_overflow_small,
        test_clamp_tail_truncates_user_prompt_when_far_too_big,
        test_is_context_overflow_error_detects_known_messages,
        test_call_llm_stream_surfaces_400_body,
        test_call_llm_stream_retries_on_context_overflow,
        test_iteration_caps_are_120,
        test_default_model_is_qwen3_coder_next_80b,
    ]

    for t in tests:
        _FakeHttpxClient.shared_queue = []
        _FakeHttpxClient.shared_requests = []
        try:
            t()
        except Exception as e:  # noqa: BLE001
            global FAIL
            FAIL += 1
            print(f"  FAIL  {t.__name__}: unexpected exception {e!r}")

    print(f"\nResults: {PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
