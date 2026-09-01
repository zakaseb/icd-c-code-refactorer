"""
Test suite for the codebase-informed transform stage.

Covers the deterministic symbol / cross-reference index (``api/codegraph.py``)
and the change-impact audit that gates the transform output in ``api/app.py``.
No Docker, no LLM, no network.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "api"))

os.environ["WORKSPACE_DIR"] = tempfile.mkdtemp(prefix="icd_cg_test_")

from codegraph import (  # noqa: E402
    CodeGraph,
    render_dependency_slices,
    render_public_surface,
    render_spec_targets,
)
from app import (  # noqa: E402
    _audit_transform_impact,
    _format_impact_report,
    _parse_impact_manifest,
)

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


COMM_H = """\
#ifndef COMM_H
#define COMM_H

#include <stdint.h>

#define MSG_MAX_PAYLOAD 64

typedef struct {
    uint32_t msg_id;      /* message identifier */
    uint16_t len;
    uint8_t  payload[MSG_MAX_PAYLOAD];
} MsgHeader;

typedef enum { MODE_IDLE = 0, MODE_RUN } CommMode;

int comm_send(const MsgHeader *hdr);
extern volatile uint32_t comm_tick;

#endif
"""

COMM_C = """\
#include "comm.h"
volatile uint32_t comm_tick;
static uint8_t scratch_buffer[16];
int comm_send(const MsgHeader *hdr) { return (int)hdr->msg_id; }
static int comm_private_helper(void) { return (int)scratch_buffer[0]; }
"""


def make_repo() -> Path:
    """Header defining several symbols, three consumers, one implementation."""
    root = Path(tempfile.mkdtemp(prefix="icd_cg_repo_"))
    (root / "inc").mkdir()
    (root / "src").mkdir()
    (root / "inc" / "comm.h").write_text(COMM_H)
    (root / "src" / "comm.c").write_text(COMM_C)
    for i, name in enumerate(["a.c", "b.c", "c.c"]):
        (root / "src" / name).write_text(
            '#include "comm.h"\n'
            f"int use{i}(MsgHeader *h) {{\n"
            f"    h->msg_id = {i};\n"
            "    h->len = MSG_MAX_PAYLOAD;\n"
            "    return comm_send(h) + (int)comm_tick + MODE_RUN;\n"
            "}\n"
        )
    # An unrelated module with its own local `len` — it must NOT be counted as
    # a use of MsgHeader::len.
    (root / "src" / "unrelated.c").write_text(
        "int measure(const char *s) {\n"
        "    int len = 0;\n"
        "    while (s[len]) { len++; }\n"
        "    return len;\n"
        "}\n"
    )
    return root


# ---------------------------------------------------------------------------

print("\n=== Test 1: CodeGraph symbol extraction ===")
repo = make_repo()
graph = CodeGraph.build(repo)
header = repo / "inc" / "comm.h"
batch = {header}

names = set(graph.symbols)
check("typedef extracted", "MsgHeader" in names)
check("macro extracted", "MSG_MAX_PAYLOAD" in names)
check("function extracted", "comm_send" in names)
check("global extracted", "comm_tick" in names)
check("struct field extracted", "msg_id" in names)
check("enumerator extracted", "MODE_RUN" in names)
check("enum typedef extracted", "CommMode" in names)
check(
    "include guard NOT treated as a macro",
    "COMM_H" not in names,
    f"symbols: {sorted(names)}",
)

fields = [s for s in graph.symbols.get("msg_id", [])]
check("field carries its parent record", bool(fields) and fields[0].parent == "MsgHeader",
      f"parent={fields[0].parent if fields else 'n/a'}")


print("\n=== Test 2: reverse dependency edges ===")
consumers = {p.name for p in graph.included_by.get(header, set())}
check("included_by populated", consumers == {"a.c", "b.c", "c.c", "comm.c"},
      f"got {sorted(consumers)}")
check("includes (forward) populated",
      header in graph.includes.get(repo / "src" / "a.c", set()))


print("\n=== Test 3: external reference counting ===")
imp_type = graph.impact_of("MsgHeader", batch)
check("MsgHeader is frozen", imp_type.frozen)
check("MsgHeader counted in 4 files", imp_type.external_files == 4,
      f"got {imp_type.external_files}")

imp_field = graph.impact_of("msg_id", batch)
check("msg_id is frozen", imp_field.frozen)
check("msg_id seen in 4 files", imp_field.external_files == 4,
      f"got {imp_field.external_files}")

imp_fn = graph.impact_of("comm_send", batch)
check("comm_send excludes its own definition site", imp_fn.external_files == 3,
      f"got {imp_fn.external_files}")

imp_local = graph.impact_of("comm_private_helper", batch)
check("file-local helper is not frozen", not imp_local.frozen)

# The precision case: a bare `len` in an unrelated function must not be
# mistaken for a use of MsgHeader::len, or every common field name in the
# tree would read as load-bearing.
imp_len = graph.impact_of("len", batch)
len_files = {p.name for p in graph.ref_counts.get("len", {})}
check("field `len` matched only via member access",
      "unrelated.c" not in len_files, f"counted in {sorted(len_files)}")
check("field `len` still matched in real consumers",
      len_files == {"a.c", "b.c", "c.c"}, f"got {sorted(len_files)}")


print("\n=== Test 4: declaration slices, not whole files ===")
a_c = repo / "src" / "a.c"
slices = render_dependency_slices(graph, a_c, a_c.read_text(), max_chars=4000)
check("slices reference MsgHeader", "MsgHeader" in slices)
check("slice contains the struct body", "uint32_t msg_id" in slices)
check("slice keeps declaration comments", "message identifier" in slices)
check("slice is far smaller than the header", len(slices) < len(COMM_H) * 4,
      f"{len(slices)} chars")
check("no truncation marker in slices", "TRUNCATED" not in slices)

tiny = render_dependency_slices(graph, a_c, a_c.read_text(), max_chars=260)
check("tiny budget drops whole blocks, never cuts one",
      tiny.count("```c") == tiny.count("```") - tiny.count("```c"),
      "unbalanced fences — a block was cut mid-way")
check("tiny budget reports what it dropped",
      not tiny or "omitted to fit" in tiny)


print("\n=== Test 5: public surface table ===")
surface = render_public_surface(graph, header, batch)
check("surface names the file", "comm.h" in surface)
check("MsgHeader marked FROZEN",
      any("FROZEN" in ln and "MsgHeader" in ln for ln in surface.splitlines()))
check("payload (unused field) marked local",
      any("local" in ln and "payload" in ln for ln in surface.splitlines()),
      surface)
check("surface explains what FROZEN means", "not being regenerated" in surface
      or "not part of this transform" in surface)
check("surface shows real call sites for frozen symbols",
      "How the FROZEN symbols are actually used" in surface, surface)
check("call site shows the consuming file and line",
      "a.c:" in surface or "b.c:" in surface, surface)
check("call site shows the actual usage expression",
      "msg_id" in surface and "->" in surface, surface)
check("local symbols get no usage block",
      "**MsgHeader.payload**" not in surface)

# A file nothing depends on has no contract to state.
empty_surface = render_public_surface(
    graph, repo / "src" / "unrelated.c", {repo / "src" / "unrelated.c"},
)
check("no surface section when nothing depends on the file",
      empty_surface == "", empty_surface[:120])


print("\n=== Test 6: change-spec symbol routing ===")
targets = render_spec_targets(
    graph, repo / "src" / "a.c",
    {"msg_id", "comm_send", "MSG_MAX_PAYLOAD", "NOT_A_REAL_SYMBOL"},
    batch,
)
check("spec symbols defined elsewhere are listed as consumed",
      "comm.h" in targets)
check("unknown spec tokens are dropped", "NOT_A_REAL_SYMBOL" not in targets)

owned = render_spec_targets(graph, header, {"msg_id", "comm_send"}, batch)
check("symbols owned by the file are marked as its own edits",
      "YOURS" in owned, owned)


print("\n=== Test 7: impact manifest parsing ===")
check("empty manifest on no marker", _parse_impact_manifest("just code") == {})
check(
    "parses a manifest after the fence",
    _parse_impact_manifest(
        '```c\nint x;\n```\nIMPACT-MANIFEST: {"renamed": '
        '[{"from": "msg_id", "to": "message_id", "reason": "ICD 4.2"}]}'
    ).get("renamed", [{}])[0].get("from") == "msg_id",
)
check("empty JSON manifest parses to {}",
      _parse_impact_manifest("IMPACT-MANIFEST: {}") == {})
check("malformed manifest degrades to empty, never raises",
      _parse_impact_manifest("IMPACT-MANIFEST: {oops") == {})


print("\n=== Test 8: change-impact audit ===")
renamed_header = COMM_H.replace("msg_id", "message_id")

rep = _audit_transform_impact(
    COMM_H, renamed_header, "comm.h", graph, batch, {},
)
check("undeclared rename of a frozen field is caught",
      len(rep["unauthorized"]) == 1,
      f"{[b['symbol'] for b in rep['unauthorized']]}")
if rep["unauthorized"]:
    b = rep["unauthorized"][0]
    check("break identifies the symbol", b["symbol"] == "msg_id")
    check("break reported as a rename, not delete+add",
          b["likely_renamed_to"] == "message_id", b["likely_renamed_to"])
    check("break carries the blast radius", b["external_files"] == 4,
          str(b["external_files"]))
    check("break carries sample call sites", len(b["sample_sites"]) > 0)

rep_declared = _audit_transform_impact(
    COMM_H, renamed_header, "comm.h", graph, batch,
    {"renamed": [{"from": "msg_id", "to": "message_id",
                  "reason": "Target ICD 4.2 renames the field"}]},
)
check("declared rename is authorized, not flagged",
      not rep_declared["unauthorized"] and len(rep_declared["authorized"]) == 1)
check("authorized break keeps the stated reason",
      "4.2" in rep_declared["authorized"][0].get("reason", ""))

# Additive change: the whole point of the contract is that this is free.
additive = COMM_H.replace(
    "uint16_t len;", "uint16_t len;\n    uint32_t crc32;"
)
rep_add = _audit_transform_impact(COMM_H, additive, "comm.h", graph, batch, {})
check("adding a field triggers no break",
      not rep_add["unauthorized"] and not rep_add["authorized"])

# Renaming something nothing depends on is not the audit's business.
local_rename = COMM_H.replace("payload", "data_bytes")
rep_local = _audit_transform_impact(
    COMM_H, local_rename, "comm.h", graph, batch, {},
)
check("renaming an unreferenced field is allowed",
      not rep_local["unauthorized"], str(rep_local["unauthorized"]))

# A dropped function is as breaking as a renamed one.
dropped = COMM_H.replace("int comm_send(const MsgHeader *hdr);\n", "")
rep_drop = _audit_transform_impact(COMM_H, dropped, "comm.h", graph, batch, {})
check("dropping a frozen function is caught",
      any(b["symbol"] == "comm_send" for b in rep_drop["unauthorized"]),
      str([b["symbol"] for b in rep_drop["unauthorized"]]))

check("audit degrades gracefully with no graph",
      _audit_transform_impact(
          COMM_H, renamed_header, "comm.h", None, batch, {},
      )["unauthorized"] == [])


print("\n=== Test 9: impact report rendering ===")
audited = [{"file": "comm.h", "frozen_symbols": 7, "breaks": 1},
           {"file": "comm.c", "frozen_symbols": 2, "breaks": 0}]
text = _format_impact_report([rep, rep_declared], audited)
check("report names the file", "comm.h" in text)
check("report separates authorized from unauthorized",
      "[UNAUTHORIZED]" in text and "[AUTHORIZED]" in text)
check("report shows dependent-file counts", "dependent file(s)" in text)
check("report states what it audited", "files audited            : 2" in text, text)

# The all-clean report must be distinguishable from a report that never ran —
# "no file on disk" is useless as evidence that the transform behaved.
clean = _format_impact_report([], audited)
check("clean run still produces a report", bool(clean.strip()))
check("clean run says so explicitly",
      "no cross-file breaks detected" in clean, clean)
check("clean run shows the frozen symbols it protected",
      "FROZEN symbols in scope  : 9" in clean, clean)

# Nothing frozen at all is a different situation and needs a different answer:
# usually it means no repository ZIP was uploaded.
no_frozen = _format_impact_report([], [
    {"file": "comm.h", "frozen_symbols": 0, "breaks": 0},
])
check("zero-frozen report explains why nothing was enforced",
      "nothing could be marked FROZEN" in no_frozen, no_frozen)
check("zero-frozen report points at the repository ZIP",
      "repository ZIP" in no_frozen, no_frozen)
check("non-zero-frozen report does NOT show the ZIP note",
      "nothing could be marked FROZEN" not in clean)

check("report tolerates a missing audited list (back-compat)",
      bool(_format_impact_report([rep]).strip()))


print("\n=== Test 10: works with uploaded files only (no repo ZIP) ===")
solo = Path(tempfile.mkdtemp(prefix="icd_cg_solo_"))
(solo / "comm.h").write_text(COMM_H)
(solo / "comm.c").write_text(COMM_C)
solo_graph = CodeGraph.build(solo, None)
check("graph builds from the uploaded dir alone",
      "MsgHeader" in solo_graph.symbols)
# Both files are in the batch, so nothing is frozen — both get regenerated.
solo_batch = {solo / "comm.h", solo / "comm.c"}
imp = solo_graph.impact_of("MsgHeader", solo_batch)
check("sibling-only use is not frozen", not imp.frozen)
check("sibling use is still reported", imp.sibling_sites > 0)


print("\n=== Test 11: robustness ===")
weird = Path(tempfile.mkdtemp(prefix="icd_cg_weird_"))
(weird / "empty.c").write_text("")
(weird / "binary.h").write_bytes(b"\xff\xfe\x00garbage\x00\x91\x92")
(weird / "unclosed.c").write_text("typedef struct { int a;\n/* never closed")
g2 = CodeGraph.build(weird)
check("malformed sources do not raise", isinstance(g2.symbols, dict))
check("empty render on a file with no deps",
      render_dependency_slices(g2, weird / "empty.c", "") == "")


print("\n" + "=" * 60)
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
if FAIL:
    print("FAILURES PRESENT")
    sys.exit(1)
print("All tests passed!")
