"""
Test suite for the codebase context integration feature (v2).
Tests repo ZIP upload, distilled context building, budget-aware prompts,
structural verification, and the end-to-end pipeline — no Docker/LLM needed.
"""
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "api"))

os.environ["WORKSPACE_DIR"] = tempfile.mkdtemp(prefix="icd_test_")

from fastapi.testclient import TestClient
from app import (
    app,
    _build_repo_context,
    _build_file_repo_context,
    _build_repo_summary,
    _safe_extract_zip,
    _extract_includes,
    _find_repo_file,
    _structural_verify,
    _estimate_tokens,
    _assemble_prompt,
    _extract_fenced,
    _looks_complete_c_file,
    _sync_generated_from_sandbox,
    MAX_REPO_CONTEXT_CHARS,
    MAX_INPUT_TOKENS,
    CHARS_PER_TOKEN,
)

client = TestClient(app)
PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} — {detail}")


def make_repo_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project/include/comm.h", (
            "#ifndef COMM_H\n#define COMM_H\n\n"
            "#include <stdint.h>\n\n"
            "typedef struct {\n"
            "    uint32_t msg_id;\n"
            "    uint16_t payload_len;\n"
            "    uint8_t  payload[256];\n"
            "} CommMessage_t;\n\n"
            "int COMM_Init(void);\n"
            "int COMM_Send(const CommMessage_t *msg);\n"
            "int COMM_Recv(CommMessage_t *msg, uint32_t timeout_ms);\n\n"
            "#endif /* COMM_H */\n"
        ))
        zf.writestr("project/include/sensor.h", (
            "#ifndef SENSOR_H\n#define SENSOR_H\n\n"
            "#include <stdint.h>\n"
            '#include "comm.h"\n\n'
            "typedef enum {\n"
            "    SENSOR_TYPE_TEMP = 0,\n"
            "    SENSOR_TYPE_PRES = 1,\n"
            "    SENSOR_TYPE_ACCEL = 2,\n"
            "} SensorType_e;\n\n"
            "typedef struct {\n"
            "    SensorType_e type;\n"
            "    float value;\n"
            "    uint32_t timestamp;\n"
            "} SensorReading_t;\n\n"
            "int SENSOR_Init(SensorType_e type);\n"
            "int SENSOR_Read(SensorType_e type, SensorReading_t *reading);\n\n"
            "#endif /* SENSOR_H */\n"
        ))
        zf.writestr("project/src/comm.c", (
            '#include "comm.h"\n'
            '#include <string.h>\n\n'
            "static int s_initialized = 0;\n\n"
            "int COMM_Init(void) {\n"
            "    s_initialized = 1;\n"
            "    return 0;\n"
            "}\n\n"
            "int COMM_Send(const CommMessage_t *msg) {\n"
            "    if (!s_initialized) return -1;\n"
            "    return 0;\n"
            "}\n\n"
            "int COMM_Recv(CommMessage_t *msg, uint32_t timeout_ms) {\n"
            "    if (!s_initialized) return -1;\n"
            "    return 0;\n"
            "}\n"
        ))
        zf.writestr("project/src/sensor.c", (
            '#include "sensor.h"\n'
            '#include "comm.h"\n\n'
            "int SENSOR_Init(SensorType_e type) {\n"
            "    return 0;\n"
            "}\n\n"
            "int SENSOR_Read(SensorType_e type, SensorReading_t *reading) {\n"
            "    return 0;\n"
            "}\n"
        ))
        zf.writestr("project/Makefile", (
            "CC = arm-none-eabi-gcc\n"
            "CFLAGS = -Wall -Werror -Iinclude\n"
            "SRC = src/comm.c src/sensor.c\n"
        ))
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------
print("\n=== Test 1: Session creation ===")
r = client.post("/api/session/create")
check("status 200", r.status_code == 200)
data = r.json()
session_id = data["session_id"]
check("repo_zip field exists", "repo_zip" in client.get(f"/api/status/{session_id}").json())

# ---------------------------------------------------------------
print("\n=== Test 2: Upload repo ZIP ===")
zip_bytes = make_repo_zip()
r = client.post(
    f"/api/upload/repo-zip/{session_id}",
    files={"file": ("test_repo.zip", io.BytesIO(zip_bytes), "application/zip")},
)
check("upload 200", r.status_code == 200)
check("file_count == 5", r.json().get("file_count") == 5)

# ---------------------------------------------------------------
print("\n=== Test 3: _extract_includes ===")
source_with_includes = (
    '#include "comm.h"\n'
    '#include "sensor.h"\n'
    '#include <stdio.h>\n'
    '#include "driver/uart.h"\n'
)
incs = _extract_includes(source_with_includes)
check("finds 3 quoted includes", len(incs) == 3, f"got {incs}")
check("comm.h found", "comm.h" in incs)
check("sensor.h found", "sensor.h" in incs)
check("driver/uart.h found", "driver/uart.h" in incs)
check("stdio.h NOT found (angle bracket)", "stdio.h" not in incs)

# ---------------------------------------------------------------
print("\n=== Test 4: _find_repo_file ===")
with tempfile.TemporaryDirectory() as tmpdir:
    zp = Path(tmpdir) / "test.zip"
    zp.write_bytes(zip_bytes)
    dest = Path(tmpdir) / "repo"
    _safe_extract_zip(zp, dest)

    found = _find_repo_file(dest, "comm.h")
    check("finds comm.h", found is not None and found.name == "comm.h")
    found2 = _find_repo_file(dest, "sensor.h")
    check("finds sensor.h", found2 is not None)
    not_found = _find_repo_file(dest, "nonexistent.h")
    check("returns None for missing", not_found is None)

# ---------------------------------------------------------------
print("\n=== Test 5: _build_file_repo_context (distilled) ===")
with tempfile.TemporaryDirectory() as tmpdir:
    zp = Path(tmpdir) / "test.zip"
    zp.write_bytes(zip_bytes)
    dest = Path(tmpdir) / "repo"
    _safe_extract_zip(zp, dest)

    source_code = '#include "sensor.h"\n\nint main(void) {\n    SENSOR_Init(SENSOR_TYPE_TEMP);\n    return 0;\n}\n'
    ctx = _build_file_repo_context(dest, source_code, set())
    check("distilled context non-empty", len(ctx) > 0, f"got {len(ctx)} chars")
    check("sensor.h in context", "sensor.h" in ctx)
    check("SensorType_e in context", "SensorType_e" in ctx)
    check("comm.h pulled transitively", "comm.h" in ctx, "sensor.h includes comm.h")
    check("CommMessage_t via transitive", "CommMessage_t" in ctx)
    check("within budget", len(ctx) <= MAX_REPO_CONTEXT_CHARS + 100)

    ctx_no_deps = _build_file_repo_context(dest, "int x = 1;\n", set())
    check("no includes = empty context", ctx_no_deps == "", f"got '{ctx_no_deps[:50]}'")

# ---------------------------------------------------------------
print("\n=== Test 6: _build_repo_summary ===")
with tempfile.TemporaryDirectory() as tmpdir:
    zp = Path(tmpdir) / "test.zip"
    zp.write_bytes(zip_bytes)
    dest = Path(tmpdir) / "repo"
    _safe_extract_zip(zp, dest)

    summary = _build_repo_summary(dest)
    check("summary non-empty", len(summary) > 0)
    check("summary under 4K", len(summary) <= 4100)
    check("file structure in summary", "File Structure" in summary)
    check("type names in summary", "CommMessage_t" in summary or "SensorReading_t" in summary)
    check("function names in summary", "COMM_Init" in summary or "SENSOR_Init" in summary)

# ---------------------------------------------------------------
print("\n=== Test 7: _estimate_tokens ===")
check("1000 chars ~ 250 tokens", _estimate_tokens("x" * 1000) == 250)
check("empty = 0", _estimate_tokens("") == 0)

# ---------------------------------------------------------------
print("\n=== Test 8: _assemble_prompt (budget enforcement) ===")
small = ("small", "A" * 100, 0)
medium = ("medium", "B" * 2000, 1)
huge = ("huge", "C" * 200000, 2)

result = _assemble_prompt([small, medium, huge], max_input_tokens=1000)
check("small included fully", "A" * 100 in result)
check("medium included", "B" * 100 in result)
check("result within budget", _estimate_tokens(result) <= 1100)
check("huge truncated or dropped", len(result) < 200000)

result2 = _assemble_prompt([small, medium], max_input_tokens=10000)
check("both fit when budget large", "A" * 100 in result2 and "B" * 2000 in result2)

# ---------------------------------------------------------------
print("\n=== Test 9: _structural_verify ===")
good_code = (
    '#include "comm.h"\n\n'
    "int COMM_Init(void) {\n"
    "    return 0;\n"
    "}\n"
)
issues = _structural_verify(good_code, None, good_code, "comm.c")
check("good code no issues", len(issues) == 0, str(issues))

bad_braces = '#include "comm.h"\n\nint main() {\n    return 0;\n'
issues2 = _structural_verify(bad_braces, None, good_code, "main.c")
check("unbalanced braces detected", any("brace" in i.lower() for i in issues2), str(issues2))

bad_comment = '#include "test.h"\n\nint x; /* unclosed\n'
issues3 = _structural_verify(bad_comment, None, bad_comment, "test.c")
check("unclosed comment detected", any("comment" in i.lower() for i in issues3), str(issues3))

with tempfile.TemporaryDirectory() as tmpdir:
    zp = Path(tmpdir) / "test.zip"
    zp.write_bytes(zip_bytes)
    dest = Path(tmpdir) / "repo"
    _safe_extract_zip(zp, dest)

    code_with_bad_include = '#include "nonexistent_driver.h"\n\nint main() {\n    return 0;\n}\n'
    issues4 = _structural_verify(code_with_bad_include, dest, code_with_bad_include, "main.c")
    check("missing include detected", any("not found" in i for i in issues4), str(issues4))

    code_with_good_include = '#include "comm.h"\n\nint main() {\n    return 0;\n}\n'
    issues5 = _structural_verify(code_with_good_include, dest, code_with_good_include, "main.c")
    check("valid include passes", not any("not found" in i for i in issues5), str(issues5))

# ---------------------------------------------------------------
print("\n=== Test 10: Header guard check ===")
bad_header = "#ifndef TEST_H\n#define TEST_H\ntypedef int x_t;\n"
issues6 = _structural_verify(bad_header, None, bad_header, "test.h")
check("missing endif detected", any("endif" in i.lower() for i in issues6), str(issues6))

good_header = "#ifndef TEST_H\n#define TEST_H\ntypedef int x_t;\n#endif\n"
issues7 = _structural_verify(good_header, None, good_header, "test.h")
check("complete header passes", not any("endif" in i.lower() for i in issues7), str(issues7))

# ---------------------------------------------------------------
print("\n=== Test 11: Static files include new features ===")
r = client.get("/")
check("repo-zip card in HTML", "card-repo-zip" in r.text)

r = client.get("/static/app.js")
check("verification stage in JS", "verification" in r.text)
check("uploadRepoZip in JS", "uploadRepoZip" in r.text)

r = client.get("/static/style.css")
check("accent-purple in CSS", "accent-purple" in r.text)

# ---------------------------------------------------------------
print("\n=== Test 12: _looks_complete_c_file unchanged ===")
complete = (
    '#include "test.h"\n#include <stdio.h>\n#include <stdlib.h>\n\n'
    "static int g_counter = 0;\n\n"
    "void helper_function(int x) {\n"
    "    g_counter += x;\n"
    '    printf("counter: %d\\n", g_counter);\n'
    "}\n\n"
    "int main(void) {\n"
    "    helper_function(42);\n"
    "    return 0;\n"
    "}\n"
)
check("complete file passes", _looks_complete_c_file(complete, complete, "main.c"))
check("empty file fails", not _looks_complete_c_file("", complete, "main.c"))

# ---------------------------------------------------------------
print("\n=== Test 13: _extract_fenced handles prefixed report text ===")
verify_output = (
    "FIXES: corrected include path and enum type\n\n"
    "```c\n"
    '#include "imu.h"\n'
    "int IMU_Init(void) {\n"
    "    return 0;\n"
    "}\n"
    "```\n"
)
fenced = _extract_fenced(verify_output, "c")
check("extracts code block when report prefix exists", "int IMU_Init(void)" in fenced, fenced)
check("strips FIXES preamble", "FIXES:" not in fenced, fenced)

# ---------------------------------------------------------------
print("\n=== Test 14: Constants are reasonable ===")
check("MAX_REPO_CONTEXT_CHARS is 15000", MAX_REPO_CONTEXT_CHARS == 15_000)
check("MAX_INPUT_TOKENS > 20000", MAX_INPUT_TOKENS > 20000,
      f"got {MAX_INPUT_TOKENS}")
check("CHARS_PER_TOKEN is 4", CHARS_PER_TOKEN == 4)

# ---------------------------------------------------------------
print("\n=== Test 15: sandbox fixes sync to generated outputs ===")
with tempfile.TemporaryDirectory() as tmpdir:
    root = Path(tmpdir)
    gen_dir = root / "generated_code"
    sandbox_dir = root / "sandbox"
    gen_dir.mkdir()
    (sandbox_dir / "src").mkdir(parents=True)

    (gen_dir / "comm.c").write_text("stale generated code\n")
    fixed = sandbox_dir / "src" / "comm.c"
    fixed.write_text("fixed sandbox code\n")

    _sync_generated_from_sandbox(gen_dir, {"comm.c": fixed})

    check(
        "generated_code receives sandbox fix",
        (gen_dir / "comm.c").read_text() == "fixed sandbox code\n",
        (gen_dir / "comm.c").read_text(),
    )

# ---------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
if FAIL > 0:
    sys.exit(1)
else:
    print("All tests passed!")
    sys.exit(0)
