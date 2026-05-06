"""Python wrapper around the Node-based append-log-helper tests.

The bulk of the testing is in `test_append_log_helper.js` (executed via
Node) which exercises the actual JavaScript runtime semantics —
WeakMap, requestAnimationFrame, setTimeout, and DOM event dispatch.

This Python wrapper:

1. Runs the Node test suite, surfacing pass/fail counts to the
   project's standard development-test format.
2. Falls back to a structural sanity check on the source file when
   Node is not available, so regressions on the safety caps are
   still caught even on systems without Node installed.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HELPER = REPO_ROOT / "api" / "static" / "append_log_helper.js"
JS_TEST = REPO_ROOT / "tests" / "development" / "test_append_log_helper.js"


PASS = 0
FAIL = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}: {detail}")


def static_structural_checks() -> None:
    """Cheap structural assertions on the helper source.

    These guard against regressions even when Node is unavailable:
    if any of the safety caps or fallback hooks gets removed by
    accident, this test will fail.
    """
    src = HELPER.read_text(encoding="utf-8", errors="replace")

    required = [
        ("MAX_STEP_LOG_CHARS", "visible <pre> cap is declared"),
        ("MAX_PENDING_BUF_CHARS", "pending-buffer cap is declared"),
        ("MAX_CHUNK_CHARS", "single-chunk cap is declared"),
        ("BG_FLUSH_INTERVAL_MS", "setTimeout fallback interval is declared"),
        ("PENDING_BUF_HIGH_WATER_FACTOR", "amortized high-water factor is declared"),
        ("visibilitychange", "visibilitychange handler is registered"),
        ("pagehide", "pagehide handler is registered"),
        ("flushAllStepLogs", "flushAllStepLogs is exported"),
        ("__icdAppendLogConfig", "config introspection is exported"),
    ]
    for marker, label in required:
        check(f"static:{label}", marker in src, f"missing {marker!r} in {HELPER}")


def run_js_tests() -> bool:
    node = shutil.which("node")
    if node is None:
        print("  SKIP  node not available — JS runtime tests skipped")
        return True
    try:
        proc = subprocess.run(
            [node, str(JS_TEST)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        print("  FAIL  js_runtime_tests: timed out")
        if e.stdout:
            print(e.stdout[-2000:])
        if e.stderr:
            print(e.stderr[-2000:])
        return False

    if proc.stdout:
        for line in proc.stdout.splitlines():
            print(f"    {line}")
    if proc.returncode != 0:
        if proc.stderr:
            for line in proc.stderr.splitlines():
                print(f"    [stderr] {line}")
        return False
    return True


def main() -> int:
    print("Running test_append_log_helper_py")

    if not HELPER.is_file():
        check("helper_file_present", False, f"{HELPER} not found")
    else:
        check("helper_file_present", True)
        static_structural_checks()

    print("\n-- JS runtime tests (node) --")
    js_ok = run_js_tests()
    check("js_runtime_tests_ok", js_ok)

    print(f"\nResults: {PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
