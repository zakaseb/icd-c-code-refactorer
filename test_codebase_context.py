"""
Test suite for the codebase context integration feature.
Tests the repo ZIP upload, context building, and verification pipeline
without requiring the full Docker/LLM infrastructure.
"""
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "webapp"))

os.environ["WORKSPACE_DIR"] = tempfile.mkdtemp(prefix="icd_test_")

from fastapi.testclient import TestClient
from app import (
    app,
    _build_repo_context,
    _safe_extract_zip,
    _looks_complete_c_file,
    MAX_REPO_CONTEXT_CHARS,
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
    """Create a mock repository ZIP with typical C project structure."""
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
            "#include <stdint.h>\n\n"
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
            "    /* ... implementation ... */\n"
            "    return 0;\n"
            "}\n\n"
            "int COMM_Recv(CommMessage_t *msg, uint32_t timeout_ms) {\n"
            "    if (!s_initialized) return -1;\n"
            "    /* ... implementation ... */\n"
            "    return 0;\n"
            "}\n"
        ))
        zf.writestr("project/src/sensor.c", (
            '#include "sensor.h"\n'
            '#include "comm.h"\n\n'
            "int SENSOR_Init(SensorType_e type) {\n"
            "    /* ... implementation ... */\n"
            "    return 0;\n"
            "}\n\n"
            "int SENSOR_Read(SensorType_e type, SensorReading_t *reading) {\n"
            "    /* ... implementation ... */\n"
            "    return 0;\n"
            "}\n"
        ))
        zf.writestr("project/Makefile", (
            "CC = arm-none-eabi-gcc\n"
            "CFLAGS = -Wall -Werror -Iinclude\n"
            "SRC = src/comm.c src/sensor.c\n"
            "OBJ = $(SRC:.c=.o)\n\n"
            "all: firmware.elf\n\n"
            "firmware.elf: $(OBJ)\n"
            "\t$(CC) $(CFLAGS) -o $@ $^\n"
        ))
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------
print("\n=== Test 1: Session creation includes repo_zip field ===")
r = client.post("/api/session/create")
check("status 200", r.status_code == 200)
data = r.json()
session_id = data["session_id"]
check("session_id present", bool(session_id))

r2 = client.get(f"/api/status/{session_id}")
status = r2.json()
check("repo_zip field exists", "repo_zip" in status, str(status))
check("repo_zip is False", status["repo_zip"] is False)

# ---------------------------------------------------------------
print("\n=== Test 2: Upload repo ZIP ===")
zip_bytes = make_repo_zip()
r = client.post(
    f"/api/upload/repo-zip/{session_id}",
    files={"file": ("test_repo.zip", io.BytesIO(zip_bytes), "application/zip")},
)
check("upload status 200", r.status_code == 200, str(r.text))
data = r.json()
check("file_count > 0", data.get("file_count", 0) > 0, str(data))
check("file_count == 5", data.get("file_count") == 5, f"got {data.get('file_count')}")

r3 = client.get(f"/api/status/{session_id}")
status = r3.json()
check("repo_zip is True", status.get("repo_zip") is True)
check("repo_zip_name set", status.get("repo_zip_name") == "test_repo.zip")

# ---------------------------------------------------------------
print("\n=== Test 3: Reject non-ZIP files ===")
r = client.post(
    f"/api/upload/repo-zip/{session_id}",
    files={"file": ("readme.txt", io.BytesIO(b"not a zip"), "text/plain")},
)
check("reject non-zip", r.status_code == 400)

# ---------------------------------------------------------------
print("\n=== Test 4: _safe_extract_zip ===")
with tempfile.TemporaryDirectory() as tmpdir:
    zip_path = Path(tmpdir) / "test.zip"
    zip_path.write_bytes(zip_bytes)
    dest = Path(tmpdir) / "extracted"
    stats = _safe_extract_zip(zip_path, dest)
    check("extract returns dict", isinstance(stats, dict))
    check("file_count == 5", stats["file_count"] == 5)
    check("comm.h exists", (dest / "project" / "include" / "comm.h").exists())
    check("sensor.c exists", (dest / "project" / "src" / "sensor.c").exists())
    check("Makefile exists", (dest / "project" / "Makefile").exists())

# ---------------------------------------------------------------
print("\n=== Test 5: _build_repo_context ===")
with tempfile.TemporaryDirectory() as tmpdir:
    zip_path = Path(tmpdir) / "test.zip"
    zip_path.write_bytes(zip_bytes)
    dest = Path(tmpdir) / "extracted"
    _safe_extract_zip(zip_path, dest)

    ctx = _build_repo_context(dest)
    check("context is non-empty", len(ctx) > 0)
    check("context within budget", len(ctx) <= MAX_REPO_CONTEXT_CHARS + 100)
    check("file tree present", "Repository File Structure" in ctx)
    check("headers section present", "Repository Headers" in ctx)
    check("comm.h content present", "CommMessage_t" in ctx)
    check("sensor.h content present", "SensorType_e" in ctx)
    check("source section present", "Repository Source" in ctx)
    check("COMM_Init in context", "COMM_Init" in ctx)
    check("build section present", "Build System" in ctx)
    check("Makefile content present", "arm-none-eabi-gcc" in ctx)

    ctx_excl = _build_repo_context(dest, exclude_names={"comm.c", "comm.h"})
    check("exclude works for comm.h", "### project/include/comm.h" not in ctx_excl)
    check("sensor.h still present", "sensor.h" in ctx_excl)

# ---------------------------------------------------------------
print("\n=== Test 6: Upload flow end-to-end (no LLM) ===")
r = client.post("/api/session/create")
sid2 = r.json()["session_id"]

c_file_content = (
    '#include "comm.h"\n\n'
    "int main(void) {\n"
    "    COMM_Init();\n"
    "    return 0;\n"
    "}\n"
)
r = client.post(
    f"/api/upload/code/{sid2}",
    files={"files": ("main.c", io.BytesIO(c_file_content.encode()), "text/plain")},
)
check("code upload ok", r.status_code == 200)

dummy_pdf = b"%PDF-1.4 dummy"
r = client.post(
    f"/api/upload/source-icd/{sid2}",
    files={"file": ("source.pdf", io.BytesIO(dummy_pdf), "application/pdf")},
)
check("source icd upload", r.status_code == 200 or "extract" in r.text.lower(),
      f"status={r.status_code}")

r = client.post(
    f"/api/upload/repo-zip/{sid2}",
    files={"file": ("repo.zip", io.BytesIO(zip_bytes), "application/zip")},
)
check("repo zip upload for session 2", r.status_code == 200)

status = client.get(f"/api/status/{sid2}").json()
check("session 2 has repo_zip", status.get("repo_zip") is True)
check("session 2 has files", len(status.get("files", [])) > 0)

# ---------------------------------------------------------------
print("\n=== Test 7: _looks_complete_c_file still works ===")
complete = (
    '#include "test.h"\n#include <stdio.h>\n#include <stdlib.h>\n\n'
    "static int g_counter = 0;\n\n"
    "void helper_function(int x) {\n"
    "    g_counter += x;\n"
    "    printf(\"counter: %d\\n\", g_counter);\n"
    "}\n\n"
    "int main(void) {\n"
    "    helper_function(42);\n"
    "    return 0;\n"
    "}\n"
)
check("complete file passes", _looks_complete_c_file(complete, complete, "main.c"))
check("empty file fails", not _looks_complete_c_file("", complete, "main.c"))
check("unbalanced braces fail", not _looks_complete_c_file(
    '#include "test.h"\nint main() {\n', complete, "main.c"
))

# ---------------------------------------------------------------
print("\n=== Test 8: Static files served ===")
r = client.get("/")
check("index.html served", r.status_code == 200)
check("repo-zip card in HTML", "card-repo-zip" in r.text)
check("input-repo-zip in HTML", "input-repo-zip" in r.text)
check("Repository Codebase label", "Repository Codebase" in r.text)

r = client.get("/static/app.js")
check("app.js served", r.status_code == 200)
check("repoZipFile in JS", "repoZipFile" in r.text)
check("uploadRepoZip in JS", "uploadRepoZip" in r.text)
check("verification stage in JS", "verification" in r.text)

r = client.get("/static/style.css")
check("style.css served", r.status_code == 200)
check("accent-purple in CSS", "accent-purple" in r.text)

# ---------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
if FAIL > 0:
    sys.exit(1)
else:
    print("All tests passed!")
    sys.exit(0)
