"""Offline sanity tests for the agentic multi-agent pipeline.

Covers (no live LLM needed — a scripted FakeLLM plays every agent):

  Test 1   package imports + endpoint registration
  Test 2   Blackboard: stage records, feedback routing, stale marking
  Test 3   LLM backend selection + resilient degradation
  Test 4   Deterministic router policy (blockers pull upstream, budgets cap)
  Test 5   Full mission end-to-end: stage order, .h-before-.c, artifacts,
           complete event contract
  Test 6   Distillation: change_spec is a genuine distillation of raw
  Test 7   Per-variant header generation path (multiple .h files)
  Test 8   NON-SEQUENTIAL reiteration: compile failure routes feedback to
           transform, which re-runs with the feedback injected, then the
           pipeline re-flows and completes
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "api"))

import app as real_app  # noqa: E402
from agentic_pipeline import (  # noqa: E402
    Blackboard,
    MissionController,
    PipelineContext,
    STAGE_ORDER,
    run_agentic_pipeline,
)
from agentic_pipeline.llm import (  # noqa: E402
    LocalLLMBackend,
    ResilientLLM,
    make_agent_llm,
)

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}  {detail}")


# ---------------------------------------------------------------------------
# Scripted LLM standing in for every agent
# ---------------------------------------------------------------------------

RAW_SPEC = (
    "## Change specification\n"
    "- sync: uint16_t 0xEB90 -> 0xEB91 in Imu.h (ImuHeader_t)\n"
    "- msg_id: uint8_t 0x10 unchanged\n"
    "- acc_x: int32_t scale 0.00012 at 1000 Hz (imu.c: imu_decode)\n"
    "- NEW field variant_tag uint8_t; macro IMU_BAUD_RATE = 115200\n"
    "- Repeated statement: sync constant changes from 0xEB90 to 0xEB91.\n"
    "- Verbose narration paragraph restating all of the above in prose "
    "for emphasis and repetition, adding no new technical content at all."
)
DISTILLED_SPEC = (
    "- sync uint16_t 0xEB90 -> 0xEB91 (Imu.h, ImuHeader_t)\n"
    "- msg_id uint8_t 0x10 unchanged\n"
    "- acc_x int32_t scale 0.00012 @1000 Hz (imu.c: imu_decode)\n"
    "- NEW variant_tag uint8_t; IMU_BAUD_RATE = 115200"
)

GEN_HEADER = (
    "/* Generated header for Target ICD */\n"
    "#ifndef IMU_H\n#define IMU_H\n#include <stdint.h>\n"
    "typedef struct {\n"
    "    uint16_t sync;      /* 0xEB91 */\n"
    "    uint8_t  msg_id;    /* 0x10 */\n"
    "    uint8_t  variant_tag;\n"
    "    uint32_t timestamp;\n"
    "} ImuHeader_t;\n"
    "#define IMU_BAUD_RATE 115200\n"
    "int imu_decode(const ImuHeader_t *hdr);\n"
    "#endif /* IMU_H */\n"
)
GEN_SOURCE = (
    "/* Generated source for Target ICD */\n"
    "#include \"Imu.h\"\n"
    "int imu_decode(const ImuHeader_t *hdr) {\n"
    "    if (hdr->sync != 0xEB91) { return -1; }\n"
    "    if (hdr->msg_id != 0x10) { return -2; }\n"
    "    return (int)hdr->variant_tag;\n"
    "}\n"
)


class FakeLLM:
    """Dispatches on system-prompt content; logs every call."""

    name = "fake-llm"

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, **kw) -> str:
        self.calls.append((system, user))
        s = system.lower()
        if "routing brain" in s:
            return '{"action": "proceed", "target": "", "reason": "ok"}'
        if "documentation analyst" in s:
            return "- field sync uint16_t 0xEB91; 1000 Hz sample rate"
        if "comparing two interface control documents" in s:
            return "sync 0xEB90 -> 0xEB91; NEW variant_tag uint8_t"
        if "merge per-section delta findings" in s or "merge" in s and "delta" in s:
            return RAW_SPEC
        if "distillation" in s or "shorter, more compact rewrite" in s:
            return DISTILLED_SPEC
        if "code reviewer" in s:
            return '{"compliant": true, "issues": [], "needs_spec_change": false}'
        if "repair" in s:
            return f"```c\n{GEN_SOURCE}```"
        if "software architect" in s:
            return "- Imu.h / imu.c: sync constant + variant_tag decode"
        if "expert c programmer" in s:
            if ".h\n\n```c" in user or "File to Transform: Imu.h" in user:
                return f"```c\n{GEN_HEADER}```"
            return f"```c\n{GEN_SOURCE}```"
        return "OK"


# ---------------------------------------------------------------------------
# Fake classic-pipeline module (deterministic helpers real, LLM ones stubbed)
# ---------------------------------------------------------------------------

_REAL_ATTRS = [
    "_build_source_scripts_context", "_split_text_chunks", "_truncate_text",
    "_distillation_fact_check", "_extract_fact_tokens", "_build_repo_knowledge",
    "_build_gitnexus_report", "_build_file_repo_context", "_build_repo_context",
    "_variant_slug", "_looks_complete_header", "_estimate_tokens",
    "_assemble_prompt", "_extract_fenced", "_looks_complete_c_file",
    "_structural_verify", "_extract_json_object", "_detect_build_system",
    "_detect_cross_compiler", "_parse_sse_event", "_read_status",
    "_write_status", "_sse",
    "ICD_CHUNK_CHARS", "MAX_CODE_CONTEXT_CHARS", "MAX_REPO_CONTEXT_CHARS",
    "MAX_INPUT_TOKENS",
    "PERIPHERAL_VARIATION_ANALYSIS_GUIDANCE",
    "PERIPHERAL_VARIATION_CODEGEN_GUIDANCE",
]


def make_fake_app(**overrides) -> SimpleNamespace:
    ns = SimpleNamespace(
        **{k: getattr(real_app, k) for k in _REAL_ATTRS}
    )
    ns.REMOTE_BUILD_ENABLED = False
    ns._detect_peripheral_variants = lambda *a, **k: []
    ns._generate_variant_header = lambda **k: ""

    def _fake_compile(**kw):
        gen_dir = kw["gen_dir"]
        n = len(list(gen_dir.glob("*.c")))
        (gen_dir / "compile_report.txt").write_text(
            f"Per-file compile gate\nstatus: OK ({n}/{n})\n"
        )
        yield real_app._sse({
            "type": "compile_summary", "stage": "compile",
            "ok": n, "failed": 0, "total": n, "message": "all good",
        })
        yield real_app._sse({"type": "stage_complete", "stage": "compile"})

    ns.run_per_file_compile = _fake_compile

    def _fake_sandbox(**kw):
        yield real_app._sse({
            "type": "sandbox_build_result", "stage": "sandbox_build",
            "success": True, "iterations": 1, "message": "built",
        })

    ns._sandbox_build_iterate = _fake_sandbox
    ns._remote_build_iterate = lambda *a, **k: iter(())
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def make_session(tmp: Path, with_variant_header: bool = False) -> dict:
    session = tmp / "session"
    code_dir = session / "original_code"
    gen_dir = session / "generated_code"
    code_dir.mkdir(parents=True)
    gen_dir.mkdir(parents=True)
    (session / "status.json").write_text(json.dumps({
        "state": "processing", "files": ["Imu.h", "imu.c"],
        "source_icd": True, "target_icd": True,
    }))
    (code_dir / "Imu.h").write_text(
        "#ifndef IMU_H\n#define IMU_H\n#include <stdint.h>\n"
        "typedef struct { uint16_t sync; uint8_t msg_id; } ImuHeader_t;\n"
        "int imu_decode(const ImuHeader_t *hdr);\n"
        "#endif\n"
    )
    (code_dir / "imu.c").write_text(
        "#include \"Imu.h\"\n"
        "int imu_decode(const ImuHeader_t *hdr) {\n"
        "    return hdr->sync == 0xEB90 ? 0 : -1;\n"
        "}\n"
    )
    return {
        "session_dir": session, "gen_dir": gen_dir, "code_dir": code_dir,
        "repo_dir": session / "repo_contents", "has_repo": False,
        "source_icd": "IMU frame: sync 0xEB90 uint16, msg_id 0x10 uint8.",
        "target_icd": (
            "IMU frame v2: sync 0xEB91 uint16, msg_id 0x10 uint8, NEW "
            "variant_tag uint8, acc_x int32 scale 0.00012 at 1000 Hz."
        ),
    }


def run_mission(kw: dict, fake_app: SimpleNamespace, llm=None) -> list[dict]:
    events = list(run_agentic_pipeline(
        llm=llm or FakeLLM(), app=fake_app, **kw,
    ))
    return events


# ===========================================================================
print("\n=== Test 1: imports + endpoint registration ===")
check("STAGE_ORDER has the six stages",
      STAGE_ORDER == ["analysis", "gitnexus", "transform",
                      "verification", "compile", "sandbox_build"])
routes = {r.path for r in real_app.app.routes}
check("/api/process-agentic/{session_id} route registered",
      "/api/process-agentic/{session_id}" in routes)
check("/api/process/{session_id} still registered",
      "/api/process/{session_id}" in routes)
check("AGENTIC_PIPELINE flag defaults to off",
      real_app.AGENTIC_PIPELINE is False)

# ===========================================================================
print("\n=== Test 2: Blackboard mechanics ===")
tmp = Path(tempfile.mkdtemp(prefix="agentic_bb_"))
bb = Blackboard(tmp, STAGE_ORDER)
check("all stages start pending",
      all(r.status == "pending" for r in bb.stages.values()))
check("next pending is analysis", bb.next_pending_stage() == "analysis")
bb.record_stage("analysis", "success", "ok")
bb.record_stage("gitnexus", "success", "ok")
bb.record_stage("transform", "success", "ok")
check("next pending is verification",
      bb.next_pending_stage() == "verification")
fb = bb.add_feedback(source_stage="verification", target_stage="transform",
                     severity="blocker", summary="missing struct")
check("blocker is pending", bb.pending_blockers() == [fb])
check("feedback digest mentions the issue",
      "missing struct" in bb.feedback_digest("transform"))
stale = bb.mark_stale_after("transform")
check("gitnexus NOT marked stale", "gitnexus" not in stale)
check("no downstream stages had run, so nothing stale", stale == [])
bb.record_stage("verification", "success", "ok")
stale = bb.mark_stale_after("transform")
check("verification marked stale after revisit of transform",
      "verification" in stale)
n = bb.resolve_feedback("transform")
check("resolve_feedback resolves 1 item", n == 1)
check("no pending blockers after resolve", bb.pending_blockers() == [])
p = bb.save()
check("blackboard snapshot saved", p.exists())
snap = json.loads(p.read_text())
check("snapshot has stages + feedback",
      "stages" in snap and "feedback" in snap)
shutil.rmtree(tmp)

# ===========================================================================
print("\n=== Test 3: LLM backend selection + degradation ===")
import os as _os
_os.environ["AGENTIC_LLM_BACKEND"] = "local"
llm = make_agent_llm(complete_fn=lambda s, u, **k: "local-ok")
check("explicit local backend", isinstance(llm, LocalLLMBackend))
check("local backend answers", llm.complete("s", "u") == "local-ok")
_os.environ["AGENTIC_LLM_BACKEND"] = "auto"


class _Boom:
    name = "boom"

    def complete(self, *a, **k):
        raise RuntimeError("primary down")


res = ResilientLLM(_Boom(), LocalLLMBackend(
    complete_fn=lambda s, u, **k: "fallback-ok"), max_primary_errors=2)
check("resilient falls back on primary error",
      res.complete("s", "u") == "fallback-ok")
res.complete("s", "u")
check("resilient degrades permanently after budget",
      "degraded" in res.name)
_os.environ.pop("AGENTIC_LLM_BACKEND", None)

# ===========================================================================
print("\n=== Test 4: deterministic router policy ===")
tmp = Path(tempfile.mkdtemp(prefix="agentic_route_"))
kw = make_session(tmp)
ctx = PipelineContext(llm=FakeLLM(), app=make_fake_app(), **kw)
mc = MissionController(ctx)
mc.bb.record_stage("analysis", "success")
mc.bb.record_stage("gitnexus", "success")
mc.bb.record_stage("transform", "success")
mc.bb.add_feedback(source_stage="compile", target_stage="transform",
                   severity="blocker", summary="won't compile")
d = mc._deterministic_route("compile")
check("blocker pulls mission back to transform",
      d["action"] == "revisit" and d["target"] == "transform")
mc.bb.add_feedback(source_stage="verification", target_stage="analysis",
                   severity="blocker", summary="spec ambiguous")
d = mc._deterministic_route("verification")
check("most upstream blocker target wins",
      d["target"] == "analysis")
# Exhaust the revisit budget for analysis (runs = 1 + 2 revisits = 3)
mc.bb.record_stage("analysis", "success")
mc.bb.record_stage("analysis", "success")
d = mc._deterministic_route("verification")
check("budget-exhausted stage is skipped by the router",
      d["target"] == "transform", f"got {d}")
mc.bb.resolve_feedback("transform")
mc.bb.resolve_feedback("analysis")
d = mc._deterministic_route("compile")
check("no blockers -> proceed", d["action"] == "proceed")
shutil.rmtree(tmp)

# ===========================================================================
print("\n=== Test 5: full mission end-to-end (offline) ===")
tmp = Path(tempfile.mkdtemp(prefix="agentic_e2e_"))
kw = make_session(tmp)
fake = make_fake_app()
events = run_mission(kw, fake)
types = [e.get("type") for e in events]
stages_seen = [e.get("stage") for e in events if e.get("type") == "stage"]
check("mission emitted events", len(events) > 10)
check("every stage team announced itself",
      all(s in {e.get("stage") for e in events} for s in STAGE_ORDER))
complete = [e for e in events if e.get("type") == "complete"]
check("exactly one complete event", len(complete) == 1)
check("complete lists generated files",
      "Imu.h" in complete[0]["files"] and "imu.c" in complete[0]["files"])
fc = [e["file"] for e in events if e.get("type") == "file_complete"]
check(".h generated before .c (headers-first order)",
      fc.index("Imu.h") < fc.index("imu.c"), f"order={fc}")
sd = kw["session_dir"]
for art in ["change_spec.txt", "change_spec_raw.txt", "target_summary.txt",
            "icd_analysis.txt", "gitnexus_report.txt"]:
    check(f"artifact {art} written", (sd / art).exists())
gd = kw["gen_dir"]
for art in ["Imu.h", "imu.c", "verification_report.txt",
            "compile_report.txt", "icd_analysis.txt",
            "gitnexus_report.txt"]:
    check(f"generated_code/{art} written", (gd / art).exists())
check("generated header has new sync constant",
      "0xEB91" in (gd / "Imu.h").read_text())
check("blackboard audit snapshot persisted",
      (sd / "agentic_pipeline" / "blackboard.json").exists())
status = json.loads((sd / "status.json").read_text())
check("status.state == completed", status.get("state") == "completed")
check("status.generated_files excludes .o",
      all(not f.endswith(".o") for f in status.get("generated_files", [])))
check("stage_complete emitted for every stage",
      all(s in [e.get("stage") for e in events
                if e.get("type") == "stage_complete"]
          for s in STAGE_ORDER))

# ===========================================================================
print("\n=== Test 6: change_spec is a genuine distillation of raw ===")
raw = (sd / "change_spec_raw.txt").read_text()
spec = (sd / "change_spec.txt").read_text()
check("raw spec saved with narration", "narration" in raw)
check("change_spec differs from raw", spec != raw)
check("change_spec is shorter", len(spec) < len(raw))
ok, missing, kept = real_app._distillation_fact_check(raw, spec)
check(f"distilled spec keeps facts ({kept:.0%})", ok, f"missing={missing}")
shutil.rmtree(tmp)

# ===========================================================================
print("\n=== Test 7: per-variant header generation ===")
tmp = Path(tempfile.mkdtemp(prefix="agentic_variant_"))
kw = make_session(tmp)
VARIANT_HDR = GEN_HEADER.replace("IMU_H", "IMU_VARIANT_H")
fake = make_fake_app(
    _detect_peripheral_variants=(
        lambda original, fname, *a, **k:
        ["High Rate", "Low Power"] if fname == "Imu.h" else []
    ),
    _generate_variant_header=lambda **k: VARIANT_HDR,
)
events = run_mission(kw, fake)
gd = kw["gen_dir"]
check("Imu_HighRate.h written", (gd / "Imu_HighRate.h").exists())
check("Imu_LowPower.h written", (gd / "Imu_LowPower.h").exists())
check("combined Imu.h NOT written (variants replace it)",
      not (gd / "Imu.h").exists())
fc = [e["file"] for e in events if e.get("type") == "file_complete"]
check("both variant headers precede imu.c",
      max(fc.index("Imu_HighRate.h"), fc.index("Imu_LowPower.h"))
      < fc.index("imu.c"), f"order={fc}")
shutil.rmtree(tmp)

# ===========================================================================
print("\n=== Test 8: NON-SEQUENTIAL reiteration via feedback routing ===")
tmp = Path(tempfile.mkdtemp(prefix="agentic_loop_"))
kw = make_session(tmp)

compile_calls = {"n": 0}


def _flaky_compile(**kwargs):
    compile_calls["n"] += 1
    gen_dir = kwargs["gen_dir"]
    n = len(list(gen_dir.glob("*.c")))
    failing = 1 if compile_calls["n"] == 1 else 0
    (gen_dir / "compile_report.txt").write_text(
        "FAILED: imu.c: error: unknown type name 'ImuHeader_t'\n"
        if failing else f"status: OK ({n}/{n})\n"
    )
    yield real_app._sse({
        "type": "compile_summary", "stage": "compile",
        "ok": n - failing, "failed": failing, "total": n,
        "message": "flaky",
    })
    yield real_app._sse({"type": "stage_complete", "stage": "compile"})


fake = make_fake_app(run_per_file_compile=_flaky_compile)
fake_llm = FakeLLM()
events = run_mission(kw, fake, llm=fake_llm)
complete = [e for e in events if e.get("type") == "complete"]
check("mission completed despite the compile failure", len(complete) == 1)
check("compile gate ran twice", compile_calls["n"] == 2,
      f"ran {compile_calls['n']}x")
transform_stage_msgs = [
    e for e in events
    if e.get("type") == "stage" and e.get("stage") == "transform"
    and e.get("index") is None
]
check("transform stage re-ran after compile feedback",
      len(transform_stage_msgs) >= 2,
      f"transform banner events: {len(transform_stage_msgs)}")
revisit_msgs = [
    e for e in events
    if "Revisiting the 'transform' stage" in str(e.get("message", ""))
]
check("MissionController announced the non-sequential revisit",
      len(revisit_msgs) == 1)
gen_prompts = [
    u for (s, u) in fake_llm.calls
    if "expert c programmer" in s.lower()
    and "Feedback from downstream teams" in u
]
check("re-run generation prompts carried the compile feedback",
      len(gen_prompts) >= 1)
verification_runs = [
    e for e in events
    if e.get("type") == "stage" and e.get("stage") == "verification"
]
check("verification re-flowed after the revisit",
      len(verification_runs) >= 2)
shutil.rmtree(tmp)

# ===========================================================================
print("\n" + "=" * 60)
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
if FAIL:
    sys.exit(1)
print("All tests passed!")
