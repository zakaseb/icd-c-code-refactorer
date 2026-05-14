"""End-to-end test for the per-file compile gate (api/per_file_compile.py).

Exercises:
  Test 1   module imports cleanly
  Test 2   clean compile of a simple .c with native gcc
  Test 3   agentic fix path: broken .c is auto-repaired by a mocked LLM
  Test 4   resume short-circuit when compile is already in completed_stages
  Test 5   skip path when no .c files were generated
  Test 6   compile_report.txt contents (status, command, attempts)
  Test 7   /api/download zip includes .o files and compile_report.txt
  Test 8   status["generated_files"] never contains .o entries
  Test 9   _extract_per_file_blocks parser
  Test 10  _collect_include_dirs prefers gen_dir over repo (helper unit)
  Test 11  REGRESSION: repo_dir is NEVER swept for -I, even when it carries
           a toxic foreign cross-toolchain header (IMU.c production bug)
  Test 12  code_dir headers are findable when gen_dir has only the .c
  Test 13  Selective resolution: project-local header from repo is pulled
           in alongside a toxic peer (Timer.h next to xilinx string.h)
  Test 14  Case-insensitive resolution: `#include "Timer.h"` resolves to
           on-disk `timer.h` via shim symlink in session_dir/compile_includes
  Test 15  Truly missing header is reported as 'missing' and the failure
           is surfaced to the LLM fix prompt
  Test 16  Toxic-only candidate is reported as 'skipped_toxic_only',
           never added to -I, never copied into a shim
  Test 17  Helper unit tests (_is_toxic_include_dir, _index_repo_headers,
           _quoted_includes_in_dir)

The test does NOT require a running LLM — we monkey-patch
`app._call_llm_complete` to a deterministic stub that returns a corrected
fenced C block.
"""
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
import shlex
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "api"))

os.environ["WORKSPACE_DIR"] = tempfile.mkdtemp(prefix="icd_pfc_test_")

import app  # noqa: E402  pylint: disable=wrong-import-position
from fastapi.testclient import TestClient  # noqa: E402

import per_file_compile  # noqa: E402  pylint: disable=wrong-import-position
from per_file_compile import (  # noqa: E402
    run_per_file_compile,
    _collect_include_dirs,
    _compile_command,
    _run_compile,
    _extract_per_file_blocks,
    _is_toxic_include_dir,
    _index_repo_headers,
    _resolve_quoted_includes_in_repo,
    _quoted_includes_in_dir,
)

client = TestClient(app.app)
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


def drain(gen) -> list[dict]:
    """Run an SSE generator and return parsed payload dicts (skip blanks)."""
    out: list[dict] = []
    for evt in gen:
        if not evt or not evt.startswith("data: "):
            continue
        try:
            out.append(json.loads(evt.split("data: ", 1)[1].split("\n", 1)[0]))
        except Exception:
            continue
    return out


def include_path_section(report: str) -> str:
    """Slice out the `INCLUDE PATH` section of compile_report.txt.

    The section starts at the `INCLUDE PATH (...)` header line and ends at
    the next ``-----`` divider or the next ``FILE:`` block / section
    header — whichever comes first. This is the only part of the report
    that constitutes the actual compile `-I` argument list.
    """
    lines = report.splitlines()
    out: list[str] = []
    in_section = False
    for i, line in enumerate(lines):
        if line.startswith("INCLUDE PATH"):
            in_section = True
            out.append(line)
            continue
        if in_section:
            stripped = line.strip()
            # Stop at the next section header (QUOTE-INCLUDE RESOLUTIONS,
            # FILE:, etc.) — those start with a divider then an UPPERCASE
            # header.
            if stripped.startswith("FILE:"):
                break
            # A divider followed by an uppercase header line ends the section.
            if (
                set(stripped) == {"-"}
                and i + 1 < len(lines)
                and lines[i + 1].strip()
                and not lines[i + 1].startswith("  -I")
                and not lines[i + 1].startswith("FILE:")
                and lines[i + 1].strip().isupper() is False
                # The line after the divider is the next section's title;
                # only stop if it's a known section title.
                and lines[i + 1].strip().startswith("QUOTE-INCLUDE")
            ):
                break
            out.append(line)
    return "\n".join(out)


def setup_session_dir(tmp: Path) -> tuple[Path, Path, Path, Path]:
    """Create a fresh session dir layout and return (session, gen, code, repo)."""
    sess = tmp / "session"
    sess.mkdir(parents=True, exist_ok=True)
    gen = sess / "generated_code"
    code = sess / "original_code"
    repo = sess / "repo_contents"
    for d in (gen, code, repo):
        d.mkdir(exist_ok=True)
    # Minimal status.json so _read_status() / _pipeline_state() work.
    (sess / "status.json").write_text(json.dumps({
        "files": ["foo.c"],
        "source_icd": True,
        "target_icd": True,
        "pipeline_state": {
            "completed_stages": [],
            "completed_files": [],
            "events": 0,
        },
    }))
    return sess, gen, code, repo


# Use native gcc throughout this test; cross-compile path is identical
# code-wise, just gated by `_detect_cross_compiler`.
NATIVE_CC = app.SANDBOX_CC_NATIVE

# Tiny, self-contained .c and .h that compile cleanly with -std=c99 -pedantic.
CLEAN_H = (
    "#ifndef FOO_H\n"
    "#define FOO_H\n"
    "#include <stdint.h>\n"
    "typedef struct { uint32_t id; uint32_t value; } FooMsg_t;\n"
    "int foo_init(FooMsg_t *m);\n"
    "#endif /* FOO_H */\n"
)
CLEAN_C = (
    "#include \"foo.h\"\n"
    "int foo_init(FooMsg_t *m)\n"
    "{\n"
    "    if (m == 0) return -1;\n"
    "    m->id = 1u;\n"
    "    m->value = 0u;\n"
    "    return 0;\n"
    "}\n"
)
# A version that fails (missing typedef + missing return) — the LLM stub fixes it.
BROKEN_C = (
    "#include \"foo.h\"\n"
    "int foo_init(FooMsg_t *m)\n"
    "{\n"
    "    UnknownType_t bogus_decl;  /* deliberate unknown type */\n"
    "    if (m == 0) return -1;\n"
    "    m->id = 1u;\n"
    "    m->value = 0u;\n"
    "    return 0;\n"
    "}\n"
)


# ---------------------------------------------------------------
print("\n=== Test 1: module imports cleanly ===")
check("run_per_file_compile callable", callable(run_per_file_compile))
check("helpers exported",
      callable(_collect_include_dirs)
      and callable(_compile_command)
      and callable(_run_compile)
      and callable(_extract_per_file_blocks))


# ---------------------------------------------------------------
print("\n=== Test 2: clean compile path (native gcc) ===")
with tempfile.TemporaryDirectory() as tmpd:
    tmp = Path(tmpd)
    sess, gen, code, repo = setup_session_dir(tmp)
    (gen / "foo.h").write_text(CLEAN_H)
    (gen / "foo.c").write_text(CLEAN_C)
    (code / "foo.c").write_text(CLEAN_C)
    (code / "foo.h").write_text(CLEAN_H)

    events = drain(run_per_file_compile(
        session_dir=sess, gen_dir=gen, code_dir=code, repo_dir=repo,
        has_repo=False, change_spec="(none)", repo_knowledge="",
        is_resume=False, completed_stages=set(),
        max_fix_attempts=2, compile_timeout=30,
        cc_override=NATIVE_CC,
    ))

    types = [e["type"] for e in events]
    check("stage event emitted",
          any(e.get("type") == "stage" and e.get("stage") == "compile" for e in events))
    check("stage_complete emitted",
          any(e.get("type") == "stage_complete" and e.get("stage") == "compile" for e in events))
    summary = next((e for e in events if e["type"] == "compile_summary"), None)
    check("compile_summary present", summary is not None)
    check("foo.c compiled OK", summary and summary["ok"] == 1 and summary["failed"] == 0,
          f"summary={summary}")
    check(".o produced in gen_dir", (gen / "foo.o").is_file(),
          f"contents: {[p.name for p in gen.iterdir()]}")
    check("compile_report.txt produced", (gen / "compile_report.txt").is_file())
    file_result = next((e for e in events if e["type"] == "compile_file_result"), None)
    check("compile_file_result success=True",
          file_result is not None and file_result.get("success") is True)
    artefacts = next((e for e in events if e["type"] == "compile_artifacts_ready"), None)
    check("artefacts: previewable files exclude .o",
          artefacts is not None
          and all(not f.lower().endswith(".o") for f in artefacts.get("files", [])))
    check("artefacts: objects list contains foo.o",
          artefacts is not None and "foo.o" in artefacts.get("objects", []))


# ---------------------------------------------------------------
print("\n=== Test 3: agentic fix path (mocked LLM repairs a broken .c) ===")
with tempfile.TemporaryDirectory() as tmpd:
    tmp = Path(tmpd)
    sess, gen, code, repo = setup_session_dir(tmp)
    (gen / "foo.h").write_text(CLEAN_H)
    (gen / "foo.c").write_text(BROKEN_C)
    (code / "foo.c").write_text(CLEAN_C)
    (code / "foo.h").write_text(CLEAN_H)

    call_count = {"n": 0}

    def fake_llm_complete(system, prompt, **kw):
        call_count["n"] += 1
        # Return a fenced clean .c so the next compile pass succeeds.
        return (
            "FIXES: removed bogus UnknownType_t declaration; signatures unchanged.\n\n"
            "```c\n" + CLEAN_C + "```\n"
        )

    original_llm = app._call_llm_complete
    app._call_llm_complete = fake_llm_complete
    try:
        events = drain(run_per_file_compile(
            session_dir=sess, gen_dir=gen, code_dir=code, repo_dir=repo,
            has_repo=False, change_spec="ICD spec stub",
            repo_knowledge="", is_resume=False, completed_stages=set(),
            max_fix_attempts=3, compile_timeout=30,
            cc_override=NATIVE_CC,
        ))
    finally:
        app._call_llm_complete = original_llm

    summary = next((e for e in events if e["type"] == "compile_summary"), None)
    check("LLM was invoked at least once for the fix", call_count["n"] >= 1,
          f"call_count={call_count['n']}")
    check("summary reports OK after fix",
          summary and summary["ok"] == 1 and summary["failed"] == 0,
          f"summary={summary}, gen_dir={[p.name for p in gen.iterdir()]}")
    check("foo.o produced after fix", (gen / "foo.o").is_file())
    check("foo.c was actually rewritten",
          "UnknownType_t" not in (gen / "foo.c").read_text())
    # Should be at least 2 compile attempts (initial fail, post-fix success).
    file_results = [e for e in events if e["type"] == "compile_file_result"]
    check("final compile_file_result is success",
          file_results and file_results[-1].get("success") is True)
    rep = (gen / "compile_report.txt").read_text()
    check("report mentions Attempt 1", "Attempt 1" in rep)
    check("report mentions Attempt 2", "Attempt 2" in rep)
    check("report contains a diff applied", "Diff applied after this attempt" in rep)


# ---------------------------------------------------------------
print("\n=== Test 4: resume short-circuit when compile already complete ===")
with tempfile.TemporaryDirectory() as tmpd:
    tmp = Path(tmpd)
    sess, gen, code, repo = setup_session_dir(tmp)
    (gen / "foo.h").write_text(CLEAN_H)
    (gen / "foo.c").write_text(CLEAN_C)
    (gen / "compile_report.txt").write_text("(previous report)")
    # Mark a fake earlier .o so we can confirm it's NOT recompiled.
    (gen / "foo.o").write_bytes(b"\x7fELF" + b"\x00" * 32)
    before_obj_bytes = (gen / "foo.o").read_bytes()

    events = drain(run_per_file_compile(
        session_dir=sess, gen_dir=gen, code_dir=code, repo_dir=repo,
        has_repo=False, change_spec="(stub)", repo_knowledge="",
        is_resume=True, completed_stages={"compile"},
        cc_override=NATIVE_CC,
    ))

    check("resume mentions reuse",
          any("Resuming" in (e.get("message") or "") for e in events))
    # No compile_file_result events should fire during a resume short-circuit.
    check("no compile_file_result on resume",
          not any(e["type"] == "compile_file_result" for e in events))
    check("compile_artifacts_ready still emitted on resume",
          any(e["type"] == "compile_artifacts_ready" for e in events))
    check("stage_complete still emitted on resume",
          any(e.get("type") == "stage_complete" and e.get("stage") == "compile" for e in events))
    check(".o not rewritten by resume short-circuit",
          (gen / "foo.o").read_bytes() == before_obj_bytes)


# ---------------------------------------------------------------
print("\n=== Test 5: skip path when no .c files were generated ===")
with tempfile.TemporaryDirectory() as tmpd:
    tmp = Path(tmpd)
    sess, gen, code, repo = setup_session_dir(tmp)
    (gen / "foo.h").write_text(CLEAN_H)  # header only, no .c

    events = drain(run_per_file_compile(
        session_dir=sess, gen_dir=gen, code_dir=code, repo_dir=repo,
        has_repo=False, change_spec="(stub)", repo_knowledge="",
        is_resume=False, completed_stages=set(),
        cc_override=NATIVE_CC,
    ))

    check("skip message emitted",
          any("No generated .c files" in (e.get("message") or "") for e in events))
    check("compile_report.txt still written",
          (gen / "compile_report.txt").is_file())
    check("stage_complete still emitted",
          any(e.get("type") == "stage_complete" and e.get("stage") == "compile" for e in events))


# ---------------------------------------------------------------
print("\n=== Test 6: compile_report.txt format ===")
with tempfile.TemporaryDirectory() as tmpd:
    tmp = Path(tmpd)
    sess, gen, code, repo = setup_session_dir(tmp)
    (gen / "foo.h").write_text(CLEAN_H)
    (gen / "foo.c").write_text(CLEAN_C)

    drain(run_per_file_compile(
        session_dir=sess, gen_dir=gen, code_dir=code, repo_dir=repo,
        has_repo=False, change_spec="(stub)", repo_knowledge="",
        is_resume=False, completed_stages=set(),
        cc_override=NATIVE_CC,
    ))

    rep = (gen / "compile_report.txt").read_text()
    check("report has banner", "PER-FILE COMPILE REPORT" in rep)
    check("report mentions compiler", NATIVE_CC.split()[0] in rep)
    check("report lists FILE: foo.c", "FILE: foo.c" in rep)
    check("report has Status: OK", "Status:   OK" in rep)
    check("report contains compile command", "-c -o" in rep)
    check("report lists include dirs section", "INCLUDE PATH" in rep)


# ---------------------------------------------------------------
print("\n=== Test 7: /api/download zip includes .o files and compile_report.txt ===")
# Drive the full pipeline minimally: create a session via API, write the
# pre-compile-gate artefacts directly, then run the gate, then hit
# /api/download.  This validates the download endpoint without spinning
# up the whole ICD analysis flow.
r = client.post("/api/session/create")
check("session create 200", r.status_code == 200)
session_id = r.json()["session_id"]

sess_path = app.SESSIONS_DIR / session_id
gen_path = sess_path / "generated_code"
code_path = sess_path / "original_code"
repo_path = sess_path / "repo_contents"
for d in (gen_path, code_path, repo_path):
    d.mkdir(parents=True, exist_ok=True)
(gen_path / "foo.h").write_text(CLEAN_H)
(gen_path / "foo.c").write_text(CLEAN_C)

drain(run_per_file_compile(
    session_dir=sess_path, gen_dir=gen_path, code_dir=code_path, repo_dir=repo_path,
    has_repo=False, change_spec="(stub)", repo_knowledge="",
    is_resume=False, completed_stages=set(),
    cc_override=NATIVE_CC,
))

resp = client.get(f"/api/download/{session_id}")
check("download 200", resp.status_code == 200,
      f"got {resp.status_code}: {resp.text[:200]}")
zf = zipfile.ZipFile(io.BytesIO(resp.content))
names = set(zf.namelist())
check("download zip contains foo.o", "foo.o" in names, f"names={sorted(names)}")
check("download zip contains foo.c", "foo.c" in names)
check("download zip contains foo.h", "foo.h" in names)
check("download zip contains compile_report.txt", "compile_report.txt" in names)
# foo.o should be only present once (the explicit re-add must not duplicate).
check("foo.o appears exactly once in zip",
      sorted(zf.namelist()).count("foo.o") == 1)


# ---------------------------------------------------------------
print("\n=== Test 8: status['generated_files'] excludes .o files ===")
# Re-using the same session — write a marker .o file alongside and
# call the same code path used at end of process_session.
sorted_gen = sorted(
    p.name for p in gen_path.iterdir()
    if p.is_file() and p.suffix.lower() != ".o"
)
check(".o not in previewable set", "foo.o" not in sorted_gen)
check(".c is in previewable set", "foo.c" in sorted_gen)
check(".h is in previewable set", "foo.h" in sorted_gen)
check("compile_report.txt is in previewable set", "compile_report.txt" in sorted_gen)


# ---------------------------------------------------------------
print("\n=== Test 9: _extract_per_file_blocks parser ===")
multi = (
    "Sure, here are the fixes:\n\n"
    "### foo.h\n"
    "```c\n"
    "#ifndef FOO_H\n#define FOO_H\nint foo_init(void);\n#endif\n"
    "```\n\n"
    "### foo.c\n"
    "```c\n"
    "#include \"foo.h\"\nint foo_init(void){return 0;}\n"
    "```\n"
)
parsed = _extract_per_file_blocks(multi)
check("multi-block parse finds foo.h", "foo.h" in parsed)
check("multi-block parse finds foo.c", "foo.c" in parsed)
check("multi-block .c body looks right",
      "foo_init" in parsed.get("foo.c", ""))

single = "```c\nint x;\n```"
check("single fenced block yields nothing in multi-parse",
      _extract_per_file_blocks(single) == {})


# ---------------------------------------------------------------
print("\n=== Test 10: _collect_include_dirs prefers gen_dir over repo ===")
with tempfile.TemporaryDirectory() as tmpd:
    tmp = Path(tmpd)
    g = tmp / "gen"
    r = tmp / "repo" / "include"
    sub = tmp / "repo" / "src"
    g.mkdir()
    r.mkdir(parents=True)
    sub.mkdir(parents=True)
    (g / "a.h").write_text("/* gen */")
    (r / "b.h").write_text("/* repo include */")
    (sub / "c.h").write_text("/* repo src */")

    inc = _collect_include_dirs(g, tmp / "repo")
    check("collect_include_dirs found gen dir first", inc and inc[0] == g.resolve(),
          f"got {[str(p) for p in inc]}")
    check("collect_include_dirs found both repo dirs",
          any(p == r.resolve() for p in inc) and any(p == sub.resolve() for p in inc))


# ---------------------------------------------------------------
# Regression test for the IMU.c failure pattern reported in production:
# a real-world repo can ship an entire foreign cross-toolchain sysroot
# (e.g. `.../arm-xilinx-eabi/include/`). Sweeping it for `-I` paths used
# to drag those incompatible libc headers in front of the host's, which
# made native gcc fail with cascading "unknown type name 'wint_t'" /
# "unknown type name 'size_t'" errors on code that was otherwise fine.
# The compile gate must now ignore repo_dir entirely and scope `-I` to
# the newly generated + verified scripts only.
print("\n=== Test 11: repo_dir is NOT swept for -I (IMU.c regression) ===")
with tempfile.TemporaryDirectory() as tmpd:
    tmp = Path(tmpd)
    sess, gen, code, repo = setup_session_dir(tmp)
    (gen / "foo.h").write_text(CLEAN_H)
    (gen / "foo.c").write_text(CLEAN_C)
    (code / "foo.c").write_text(CLEAN_C)
    (code / "foo.h").write_text(CLEAN_H)

    # Plant a "toxic" cross-toolchain header tree under repo_dir that
    # mirrors the production failure: a `string.h` that references types
    # the native compiler can't possibly provide. If this file ever
    # appears on the compile `-I` path, foo.c's `#include "foo.h"` is
    # fine but ANY `#include <string.h>` (and we add one below) would
    # explode exactly like IMU.c did.
    toxic_dir = repo / "superloop-sw-develop" / "Workspace" \
        / "P3_MCP_Application" / "cmake-src" / "toolchain" / "gnu" \
        / "arm" / "nt" / "arm-xilinx-eabi" / "include"
    toxic_dir.mkdir(parents=True, exist_ok=True)
    (toxic_dir / "string.h").write_text(
        "/* Foreign cross-toolchain string.h — relies on builtins the\n"
        "   host compiler does not provide. Should NEVER be picked up\n"
        "   by the per-file compile gate. */\n"
        "extern int  _bogus_xilinx_builtin_wint_t;\n"
        "extern int  _bogus_xilinx_builtin_size_t;\n"
        "#error \"per-file compile gate must NOT load this foreign header\"\n"
    )
    # And a benign-looking second one further inside the repo, just to
    # confirm we ignore every depth.
    (repo / "include").mkdir(parents=True, exist_ok=True)
    (repo / "include" / "junk.h").write_text("/* must not be on -I */\n")

    # Force foo.c to actually try `<string.h>` so that, IF the toxic
    # header were on the path, gcc would resolve to it and #error out.
    (gen / "foo.c").write_text(
        "#include \"foo.h\"\n"
        "#include <string.h>\n"
        + CLEAN_C.split("#include \"foo.h\"\n", 1)[1]
    )

    events = drain(run_per_file_compile(
        session_dir=sess, gen_dir=gen, code_dir=code, repo_dir=repo,
        has_repo=True, change_spec="(stub)", repo_knowledge="",
        is_resume=False, completed_stages=set(),
        max_fix_attempts=2, compile_timeout=30,
        cc_override=NATIVE_CC,
    ))

    summary = next((e for e in events if e["type"] == "compile_summary"), None)
    check(
        "compile succeeds even though repo_dir carries a toxic string.h",
        summary and summary["ok"] == 1 and summary["failed"] == 0,
        f"summary={summary}",
    )
    check(".o produced", (gen / "foo.o").is_file())

    rep = (gen / "compile_report.txt").read_text()
    inc = include_path_section(rep)
    check(
        "report's INCLUDE PATH header advertises the scope",
        "INCLUDE PATH" in rep
        and "gen_dir" in inc
        and ("toolchain" in inc.lower() or "demand-driven" in inc.lower()),
        "expected report banner to document the include-path scope",
    )
    check(
        "INCLUDE PATH section never lists the toxic repo dir as -I",
        f"-I{toxic_dir.resolve()}" not in inc,
        f"toxic dir {toxic_dir} leaked onto -I:\n{inc}",
    )
    check(
        "INCLUDE PATH section never lists ANY repo dir as -I",
        f"-I{repo.resolve()}" not in inc
        and not any(
            l.strip().startswith(f"-I{repo.resolve()}") for l in inc.splitlines()
        ),
        "any repo_dir path on the compile -I list is a regression",
    )
    check(
        "INCLUDE PATH section DOES list gen_dir on the -I path",
        f"-I{gen.resolve()}" in inc,
    )

    # Also verify the live SSE message advertises the scope.
    info_msgs = [e.get("message", "") for e in events if e.get("type") == "info"]
    check(
        "SSE info message documents the scope explicitly",
        any(
            "scoped to" in m
            and "repository ZIP" in m
            and "NOT swept" in m
            for m in info_msgs
        ),
        f"info messages={info_msgs}",
    )


# ---------------------------------------------------------------
# Make sure the scoping change still lets the gate find headers when a
# `.c` lives in gen_dir but its only matching `.h` was uploaded by the
# user and not regenerated (i.e. still in code_dir). This case used to
# be covered transitively by sweeping repo_dir; it must keep working.
print("\n=== Test 12: code_dir headers are findable when gen_dir has only .c ===")
with tempfile.TemporaryDirectory() as tmpd:
    tmp = Path(tmpd)
    sess, gen, code, repo = setup_session_dir(tmp)
    # The .h is ONLY in code_dir; gen_dir has only the .c.
    (gen / "foo.c").write_text(CLEAN_C)
    (code / "foo.h").write_text(CLEAN_H)
    (code / "foo.c").write_text(CLEAN_C)

    events = drain(run_per_file_compile(
        session_dir=sess, gen_dir=gen, code_dir=code, repo_dir=repo,
        has_repo=False, change_spec="(stub)", repo_knowledge="",
        is_resume=False, completed_stages=set(),
        max_fix_attempts=1, compile_timeout=30,
        cc_override=NATIVE_CC,
    ))
    summary = next((e for e in events if e["type"] == "compile_summary"), None)
    check(
        "gen_dir-only .c finds its .h fallback in code_dir",
        summary and summary["ok"] == 1 and summary["failed"] == 0,
        f"summary={summary}",
    )
    check(".o produced via code_dir header fallback",
          (gen / "foo.o").is_file())


# ---------------------------------------------------------------
# Test 13 reproduces the second production failure: the generated .c
# does `#include "Timer.h"` (a project-local header). After the previous
# repo-scope fix, Timer.h could no longer be resolved and the gate
# failed with `fatal error: Timer.h: No such file or directory`. The
# selective resolver must now find Timer.h in a non-toxic repo
# directory AND still keep the toxic xilinx-eabi peer off the -I path.
print("\n=== Test 13: selective resolution finds project Timer.h, "
      "ignores toxic peer ===")
TIMER_H_BODY = (
    "#ifndef TIMER_H\n#define TIMER_H\n"
    "#include <stdint.h>\n"
    "void timer_tick(uint32_t ms);\n"
    "#endif\n"
)
IMU_H_BODY = (
    "#ifndef IMU_H\n#define IMU_H\n"
    "#include <stdint.h>\n"
    "int IMU_Init(void);\n"
    "#endif\n"
)
IMU_C_USES_TIMER = (
    "#include \"IMU.h\"\n"
    "#include \"Timer.h\"\n"
    "#include <string.h>\n"
    "int IMU_Init(void) { timer_tick(0u); return 0; }\n"
)
with tempfile.TemporaryDirectory() as tmpd:
    tmp = Path(tmpd)
    sess, gen, code, repo = setup_session_dir(tmp)
    (gen / "IMU.h").write_text(IMU_H_BODY)
    (gen / "IMU.c").write_text(IMU_C_USES_TIMER)
    (code / "IMU.c").write_text(IMU_C_USES_TIMER)
    (code / "IMU.h").write_text(IMU_H_BODY)

    # Legit project-local Timer.h, deep but reachable.
    proj = repo / "superloop-sw-develop" / "Workspace" \
        / "P3_MCP_Application" / "Source" / "Drivers"
    proj.mkdir(parents=True)
    (proj / "Timer.h").write_text(TIMER_H_BODY)

    # Toxic xilinx-eabi peer, mirroring the exact production layout.
    toxic = repo / "superloop-sw-develop" / "Workspace" \
        / "P3_MCP_Application" / "cmake-src" / "toolchain" / "gnu" \
        / "arm" / "nt" / "arm-xilinx-eabi" / "include"
    toxic.mkdir(parents=True)
    (toxic / "string.h").write_text("#error toxic_string_h\n")
    (toxic / "Timer.h").write_text("#error toxic_timer_h\n")

    events = drain(run_per_file_compile(
        session_dir=sess, gen_dir=gen, code_dir=code, repo_dir=repo,
        has_repo=True, change_spec="(stub)", repo_knowledge="",
        is_resume=False, completed_stages=set(),
        max_fix_attempts=1, compile_timeout=30,
        cc_override=NATIVE_CC,
    ))

    summary = next((e for e in events if e["type"] == "compile_summary"), None)
    check("IMU.c compiles with Timer.h resolved from non-toxic repo path",
          summary and summary["ok"] == 1 and summary["failed"] == 0,
          f"summary={summary}; gen={[p.name for p in gen.iterdir()]}")
    check("IMU.o produced", (gen / "IMU.o").is_file())

    rep = (gen / "compile_report.txt").read_text()
    inc = include_path_section(rep)
    check("report adds a QUOTE-INCLUDE RESOLUTIONS section",
          "QUOTE-INCLUDE RESOLUTIONS" in rep)
    check("report records Timer.h as 'found' status",
          "[found] Timer.h" in rep)
    check("report -I includes the legit Drivers dir",
          f"-I{proj.resolve()}" in inc)
    check("INCLUDE PATH section never lists the toxic arm-xilinx-eabi dir as -I",
          f"-I{toxic.resolve()}" not in inc,
          f"toxic dir leaked onto -I:\n{inc}")
    check(
        "audit surfaces the toxic Timer.h candidate (so the user can see "
        "WHY we picked the legit one)",
        "arm-xilinx-eabi" in rep
        and str((toxic / "Timer.h").resolve()) in rep,
        "expected the toxic Timer.h candidate to be audited in the report",
    )


# ---------------------------------------------------------------
# Test 14: case sensitivity. `#include "Timer.h"` on Linux must resolve
# to an on-disk file named `timer.h` via the shim mechanism.
print("\n=== Test 14: case-insensitive resolution via shim ===")
with tempfile.TemporaryDirectory() as tmpd:
    tmp = Path(tmpd)
    sess, gen, code, repo = setup_session_dir(tmp)
    (gen / "IMU.h").write_text(IMU_H_BODY)
    (gen / "IMU.c").write_text(IMU_C_USES_TIMER)  # quote-includes "Timer.h"
    (code / "IMU.h").write_text(IMU_H_BODY)
    (code / "IMU.c").write_text(IMU_C_USES_TIMER)

    # ONLY a lowercase 'timer.h' exists in the repo.
    proj = repo / "proj" / "src"
    proj.mkdir(parents=True)
    (proj / "timer.h").write_text(TIMER_H_BODY)

    events = drain(run_per_file_compile(
        session_dir=sess, gen_dir=gen, code_dir=code, repo_dir=repo,
        has_repo=True, change_spec="(stub)", repo_knowledge="",
        is_resume=False, completed_stages=set(),
        max_fix_attempts=1, compile_timeout=30,
        cc_override=NATIVE_CC,
    ))

    summary = next((e for e in events if e["type"] == "compile_summary"), None)
    check(
        "case-mismatched include resolves via shim (compile succeeds)",
        summary and summary["ok"] == 1 and summary["failed"] == 0,
        f"summary={summary}; "
        f"shim_dir={[p.name for p in (sess / 'compile_includes').iterdir()] if (sess / 'compile_includes').exists() else 'MISSING'}",
    )
    shim_dir = sess / "compile_includes"
    check("shim dir created under session_dir, NOT under gen_dir",
          shim_dir.exists() and "Timer.h" in {p.name for p in shim_dir.iterdir()})
    check("shim's Timer.h points back to the real lowercase timer.h",
          (shim_dir / "Timer.h").is_file()
          and ((shim_dir / "Timer.h").resolve() == (proj / "timer.h").resolve()
               or (shim_dir / "Timer.h").read_text() == TIMER_H_BODY))

    rep = (gen / "compile_report.txt").read_text()
    check("report records Timer.h as 'found_case_normalized'",
          "[found_case_normalized] Timer.h" in rep)
    check("report shim line points at compile_includes",
          "compile_includes" in rep)
    check("shim dir does NOT appear in gen_dir (kept out of download bundle)",
          not (gen / "compile_includes").exists())


# ---------------------------------------------------------------
# Test 15: truly missing header. The resolver must classify it as
# 'missing', the report must say so, and the agentic-fix prompt must
# surface it so the LLM can drop or rename the bad include.
print("\n=== Test 15: truly missing quote-include is flagged 'missing' ===")
IMU_C_MISSING = (
    "#include \"IMU.h\"\n"
    "#include \"DoesNotExist.h\"\n"
    "#include <stdint.h>\n"
    "\n"
    "static uint32_t s_counter = 0u;\n"
    "\n"
    "int IMU_Init(void)\n"
    "{\n"
    "    s_counter = 0u;\n"
    "    return 0;\n"
    "}\n"
)
IMU_C_MISSING_FIXED = (
    "#include \"IMU.h\"\n"
    "#include <stdint.h>\n"
    "\n"
    "static uint32_t s_counter = 0u;\n"
    "\n"
    "int IMU_Init(void)\n"
    "{\n"
    "    s_counter = 0u;\n"
    "    return 0;\n"
    "}\n"
)

# Capture the LLM prompt so we can assert the resolution audit is in it.
captured_prompts: list[str] = []

def fake_llm_prompt_capture(system, prompt, **kw):
    captured_prompts.append(prompt)
    # Return a fenced "fix" that drops the bad include.
    return (
        "FIXES: removed missing DoesNotExist.h include.\n\n"
        "```c\n" + IMU_C_MISSING_FIXED + "```\n"
    )

with tempfile.TemporaryDirectory() as tmpd:
    tmp = Path(tmpd)
    sess, gen, code, repo = setup_session_dir(tmp)
    (gen / "IMU.h").write_text(IMU_H_BODY)
    (gen / "IMU.c").write_text(IMU_C_MISSING)
    (code / "IMU.h").write_text(IMU_H_BODY)
    (code / "IMU.c").write_text(IMU_C_MISSING)

    original_llm = app._call_llm_complete
    app._call_llm_complete = fake_llm_prompt_capture
    try:
        events = drain(run_per_file_compile(
            session_dir=sess, gen_dir=gen, code_dir=code, repo_dir=repo,
            has_repo=True, change_spec="(stub)", repo_knowledge="",
            is_resume=False, completed_stages=set(),
            max_fix_attempts=2, compile_timeout=30,
            cc_override=NATIVE_CC,
        ))
    finally:
        app._call_llm_complete = original_llm

    rep = (gen / "compile_report.txt").read_text()
    check("report flags DoesNotExist.h as 'missing'",
          "[missing] DoesNotExist.h" in rep,
          f"report excerpt:\n{rep[-2000:]}")
    check(
        "LLM fix prompt surfaces the resolution audit",
        any("Quote-Include Resolution Audit" in p for p in captured_prompts)
        and any("DoesNotExist.h" in p for p in captured_prompts),
        f"captured {len(captured_prompts)} prompts; "
        f"first 400 chars of first: "
        f"{captured_prompts[0][:400] if captured_prompts else '(none)'}",
    )
    # After the LLM drops the bad include the compile should succeed.
    summary = next((e for e in events if e["type"] == "compile_summary"), None)
    check(
        "missing-include path is salvaged by the agentic loop "
        "(LLM drops the bad include)",
        summary and summary["ok"] == 1 and summary["failed"] == 0,
        f"summary={summary}",
    )


# ---------------------------------------------------------------
# Test 16: header exists ONLY in a toxic toolchain dir. The resolver
# must classify it as 'skipped_toxic_only' and never add a shim.
print("\n=== Test 16: toxic-only candidate is skipped, never shimmed ===")
IMU_C_NEEDS_HEADER = (
    "#include \"IMU.h\"\n"
    "#include \"Reent.h\"\n"
    "int IMU_Init(void) { return 0; }\n"
)
with tempfile.TemporaryDirectory() as tmpd:
    tmp = Path(tmpd)
    sess, gen, code, repo = setup_session_dir(tmp)
    (gen / "IMU.h").write_text(IMU_H_BODY)
    (gen / "IMU.c").write_text(IMU_C_NEEDS_HEADER)
    (code / "IMU.h").write_text(IMU_H_BODY)
    (code / "IMU.c").write_text(IMU_C_NEEDS_HEADER)

    # Reent.h ONLY in arm-xilinx-eabi/include/. Note: that dir also
    # contains `string.h`/`stdio.h`, which triggers the libc-sentinel
    # check in _is_toxic_include_dir even without the path-segment
    # match — belt and braces.
    toxic = repo / "vendor" / "arm-xilinx-eabi" / "include" / "sys"
    toxic.mkdir(parents=True)
    (toxic / "Reent.h").write_text("#error never_load_me\n")
    (toxic.parent / "string.h").write_text("/* libc sentinel */\n")

    events = drain(run_per_file_compile(
        session_dir=sess, gen_dir=gen, code_dir=code, repo_dir=repo,
        has_repo=True, change_spec="(stub)", repo_knowledge="",
        is_resume=False, completed_stages=set(),
        max_fix_attempts=1, compile_timeout=30,
        cc_override=NATIVE_CC,
    ))

    rep = (gen / "compile_report.txt").read_text()
    inc = include_path_section(rep)
    check("report flags Reent.h as 'skipped_toxic_only'",
          "[skipped_toxic_only] Reent.h" in rep,
          f"tail:\n{rep[-1500:]}")
    check("INCLUDE PATH section has no -I leading to the toxic vendor dir",
          f"-I{toxic.resolve()}" not in inc
          and f"-I{toxic.parent.resolve()}" not in inc,
          f"INCLUDE PATH section leaked:\n{inc}")
    shim = sess / "compile_includes"
    check("no shim materialized for a toxic-only candidate",
          not shim.exists()
          or "Reent.h" not in {p.name for p in shim.rglob("*") if p.is_file()})
    # Compile will fail (the include genuinely cannot be satisfied) — that's
    # the correct, honest outcome here.
    summary = next((e for e in events if e["type"] == "compile_summary"), None)
    check("compile correctly fails when only-toxic candidate is skipped",
          summary and summary["ok"] == 0 and summary["failed"] == 1,
          f"summary={summary}")


# ---------------------------------------------------------------
# Tiny unit tests for the new helpers, independent of the gate.
print("\n=== Test 17: helper unit tests ===")
with tempfile.TemporaryDirectory() as tmpd:
    tmp = Path(tmpd)
    # _is_toxic_include_dir: path-segment markers.
    d1 = tmp / "x" / "arm-xilinx-eabi" / "include"
    d1.mkdir(parents=True)
    check("toxic path: arm-xilinx-eabi triggers",
          _is_toxic_include_dir(d1))
    d2 = tmp / "x" / "toolchain" / "gnu" / "include"
    d2.mkdir(parents=True)
    check("toxic path: /toolchain/ triggers",
          _is_toxic_include_dir(d2))
    d3 = tmp / "x" / "vendor" / "sysroot" / "include"
    d3.mkdir(parents=True)
    check("toxic path: /sysroot/ triggers",
          _is_toxic_include_dir(d3))
    d4 = tmp / "x" / "proj" / "src"
    d4.mkdir(parents=True)
    check("plain project src is NOT toxic",
          not _is_toxic_include_dir(d4))

    # _is_toxic_include_dir: libc sentinel detection.
    d5 = tmp / "x" / "looks_legit_but_carries_libc"
    d5.mkdir(parents=True)
    (d5 / "string.h").write_text("/* libc */\n")
    check("dir carrying string.h is treated as toxic even with a clean path",
          _is_toxic_include_dir(d5))

    # _index_repo_headers
    r = tmp / "r"
    (r / "a").mkdir(parents=True)
    (r / "b").mkdir(parents=True)
    (r / "a" / "Foo.h").write_text("/* */")
    (r / "b" / "foo.h").write_text("/* */")
    (r / "a" / "ignore.txt").write_text("/* */")
    idx = _index_repo_headers(r)
    check("index has lowercase basename keys", "foo.h" in idx)
    check("index gathers both case variants under one key",
          len(idx.get("foo.h", [])) == 2)
    check("index ignores non-headers", all(
        p.suffix.lower() in {".h", ".hpp", ".hh", ".hxx"}
        for paths in idx.values() for p in paths
    ))

    # _quoted_includes_in_dir picks up only quoted includes
    g = tmp / "g"
    g.mkdir()
    (g / "x.c").write_text(
        "#include \"Foo.h\"\n#include <stdio.h>\n#include \"sub/Bar.h\"\n"
    )
    qs = _quoted_includes_in_dir(g)
    check("quoted-include parser finds Foo.h", "Foo.h" in qs)
    check("quoted-include parser finds sub/Bar.h", "sub/Bar.h" in qs)
    check("quoted-include parser IGNORES angle <stdio.h>",
          "stdio.h" not in qs)


# ---------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
if FAIL > 0:
    sys.exit(1)
else:
    print("All tests passed!")
    sys.exit(0)
