"""Tests for UI-controlled sandbox build retry budgets."""
from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "api"))

import app as appmod  # noqa: E402
import orchestrator as orchmod  # noqa: E402
import agentic_debug as agenticmod  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}  {detail}")


print("\n=== Test 1: _parse_sandbox_retries ===")
check("25 -> 25", appmod._parse_sandbox_retries("25") == 25)
check("1 -> 1", appmod._parse_sandbox_retries(1) == 1)
check("indefinite -> None", appmod._parse_sandbox_retries("indefinite") is None)
check("INF -> None", appmod._parse_sandbox_retries("INF") is None)
check("0 -> None", appmod._parse_sandbox_retries("0") is None)
check("-1 -> None", appmod._parse_sandbox_retries(-1) is None)
try:
    appmod._parse_sandbox_retries("abc")
    check("rejects garbage", False)
except ValueError:
    check("rejects garbage", True)

print("\n=== Test 2: _resolve_sandbox_max_retries ===")
check(
    "no raw + empty status -> UNSET",
    appmod._resolve_sandbox_max_retries(None, {}) is appmod._SANDBOX_RETRIES_UNSET,
)
check(
    "query indefinite wins",
    appmod._resolve_sandbox_max_retries("indefinite", {"sandbox_retries_mode": "finite", "sandbox_max_retries": 3}) is None,
)
check(
    "query int wins",
    appmod._resolve_sandbox_max_retries("7", {}) == 7,
)
check(
    "status indefinite when no query",
    appmod._resolve_sandbox_max_retries(None, {"sandbox_retries_mode": "indefinite"}) is None,
)
check(
    "status finite when no query",
    appmod._resolve_sandbox_max_retries(None, {"sandbox_retries_mode": "finite", "sandbox_max_retries": 12}) == 12,
)

print("\n=== Test 3: _sandbox_retry_plan ===")
plan_env = appmod._sandbox_retry_plan(appmod._SANDBOX_RETRIES_UNSET)
check("env mode", plan_env["mode"] == "env")
check("env orch_outer from constant", plan_env["orch_outer"] == appmod.SANDBOX_ORCH_OUTER_ROUNDS)
check("env orch_builds from constant", plan_env["orch_builds"] == appmod.SANDBOX_ORCH_MAX_BUILDS)

plan_inf = appmod._sandbox_retry_plan(None)
check("indefinite mode", plan_inf["mode"] == "indefinite")
check(
    "indefinite orch_outer is 1 (single campaign)",
    plan_inf["orch_outer"] == 1,
)
check("indefinite orch_builds None (unlimited builds)", plan_inf["orch_builds"] is None)
check("indefinite agentic_outer is 1", plan_inf["agentic_outer"] == 1)
check("indefinite agentic_attempts None", plan_inf["agentic_attempts"] is None)
check("indefinite legacy_max None", plan_inf["legacy_max"] is None)

plan_n = appmod._sandbox_retry_plan(10)
check("finite mode", plan_n["mode"] == "finite")
check("finite collapses to 1 orch round", plan_n["orch_outer"] == 1)
check("finite orch_builds == N", plan_n["orch_builds"] == 10)
check("finite agentic_attempts == N", plan_n["agentic_attempts"] == 10)
check("finite legacy_max == N", plan_n["legacy_max"] == 10)

print("\n=== Test 4: orchestrator accepts max_builds=None ===")
# Smoke: calling with unlimited budget must not TypeError on the comparison.
builds_seen = {"n": 0}

def _ok_runner():
    builds_seen["n"] += 1
    return True, "ok"

events = list(orchmod.run_orchestrator(
    sandbox_dir=Path(tempfile.mkdtemp()),
    build_info={"type": "make", "dir": Path(".")},
    sandbox_cc="gcc",
    is_cross=False,
    gen_files={},
    change_spec="",
    repo_knowledge="",
    file_index={},
    snapshots={},
    build_runner=_ok_runner,
    llm_stream=lambda *a, **k: iter(()),
    max_builds=None,
))
done = [e for e in events if e.get("type") == "done"]
check("orch with max_builds=None completes on initial success",
      len(done) == 1 and done[0].get("success") is True)

print("\n=== Test 5: agentic_debug accepts max_attempts=None loop gate ===")
# Verify the while-loop gate: with max_attempts=None and an immediate
# successful initial build, it should done(success) without attempting.
# run_agentic_debug always does an initial build first.
def _success_runner():
    return True, "build ok"

tmp = Path(tempfile.mkdtemp())
(tmp / "sandbox").mkdir()
events = list(agenticmod.run_agentic_debug(
    session_dir=tmp,
    sandbox_dir=tmp / "sandbox",
    build_info={"type": "make", "dir": tmp / "sandbox"},
    sandbox_cc="gcc",
    is_cross=False,
    gen_files={},
    change_spec="",
    repo_knowledge="",
    file_index={},
    snapshots={},
    build_runner=_success_runner,
    llm_stream=lambda *a, **k: iter(()),
    max_attempts=None,
))
done = [e for e in events if e.get("type") == "done"]
check("agentic with max_attempts=None succeeds on clean build",
      len(done) == 1 and done[0].get("success") is True)

print("\n=== Test 6: FastAPI routes accept sandbox_retries query ===")
from fastapi.testclient import TestClient  # noqa: E402
client = TestClient(appmod.app)
# 404 session is fine — we only care the query param is accepted (not 422)
r = client.get("/api/process/does-not-exist?sandbox_retries=5")
check("process accepts sandbox_retries query (404 not 422)", r.status_code == 404)
r2 = client.get("/api/process/does-not-exist?sandbox_retries=indefinite")
check("process accepts indefinite query", r2.status_code == 404)
r3 = client.get("/api/regenerate/does-not-exist?sandbox_retries=3")
check("regenerate accepts sandbox_retries query", r3.status_code == 404)

print("\n=== Test 7: UI markup / JS helpers present ===")
html = (REPO / "api" / "static" / "index.html").read_text()
js = (REPO / "api" / "static" / "app.js").read_text()
css = (REPO / "api" / "static" / "style.css").read_text()
check("index has sandbox-retries-input", 'id="sandbox-retries-input"' in html)
check("index has indefinite button", 'id="sandbox-retries-indefinite"' in html)
check("js has processStreamUrl", "function processStreamUrl" in js)
check("js passes query on process EventSource",
      "processStreamUrl('/api/process/'" in js)
check("js passes query on regenerate EventSource",
      "processStreamUrl('/api/regenerate/'" in js)
check("js has indefinite toggle", "setSandboxRetriesIndefinite" in js)
check("css styles the control", ".sandbox-retries-control" in css)

print("\n=== Test 8: _sandbox_build_iterate emits budget info ===")
# Drive just the early yield by giving an empty gen_dir (no .c/.h).
sess = Path(tempfile.mkdtemp())
gen = sess / "generated_code"
gen.mkdir()
repo = sess / "repo"
repo.mkdir()
(repo / "Makefile").write_text("all:\n\t@true\n")
events = []
for evt in appmod._sandbox_build_iterate(
    session_dir=sess,
    gen_dir=gen,
    repo_dir=repo,
    change_spec="",
    uploaded_names=set(),
    has_repo=True,
    sandbox_max_retries=3,
):
    payload = appmod._parse_sse_event(evt)
    if payload:
        events.append(payload)
msgs = [e.get("message", "") for e in events if e.get("type") == "info"]
check(
    "finite budget announced in SSE",
    any("3 reiteration" in m for m in msgs),
    str(msgs[:3]),
)

events2 = []
for evt in appmod._sandbox_build_iterate(
    session_dir=sess,
    gen_dir=gen,
    repo_dir=repo,
    change_spec="",
    uploaded_names=set(),
    has_repo=True,
    sandbox_max_retries=None,
):
    payload = appmod._parse_sse_event(evt)
    if payload:
        events2.append(payload)
msgs2 = [e.get("message", "") for e in events2 if e.get("type") == "info"]
check(
    "indefinite budget announced in SSE",
    any("indefinite" in m.lower() for m in msgs2),
    str(msgs2[:3]),
)

print("\n" + "=" * 60)
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
if FAIL:
    sys.exit(1)
print("All tests passed!")
