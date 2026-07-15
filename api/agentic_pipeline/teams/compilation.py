"""Stage 5 — CompilationTeam (SSE stage: ``compile``).

Multi-agent system wrapping the per-file compile gate (.c -> .o).

Roster
    ToolchainScout — deterministic: detects the repository's build system
                     and cross compiler (arm-none-eabi / arm-xilinx /
                     aarch64-none-elf / mb-gcc, -std=gnu99) so RTOS /
                     embedded targets compile with the right toolchain.
    BuildOperator  — drives the proven per-file compile gate, which
                     internally runs its own agentic LLM fix loop per
                     failing translation unit.
    GateKeeper     — deterministic: reads the compile outcome and raises
                     targeted feedback (usually at generation) when files
                     still fail after the fix budget.

Artifacts (identical to the classic pipeline)
    generated_code/<base>.o, generated_code/compile_report.txt
"""

from __future__ import annotations

from typing import Iterator

from ..agents import Agent
from ..blackboard import Blackboard
from ..context import PipelineContext
from .base import Team


class CompilationTeam(Team):
    stage = "compile"
    title = "Compilation Team"

    def __init__(self):
        super().__init__()
        self.toolchain_scout = Agent(
            name="ToolchainScout",
            role="deterministic build-system / cross-compiler detection",
        )
        self.build_operator = Agent(
            name="BuildOperator",
            role="per-file compile gate with agentic fix loop",
        )
        self.gate_keeper = Agent(
            name="GateKeeper",
            role="deterministic outcome triage + feedback routing",
        )
        self.agents = [
            self.toolchain_scout, self.build_operator, self.gate_keeper,
        ]

    def run(self, ctx: PipelineContext, bb: Blackboard) -> Iterator[dict]:
        a = ctx.app
        yield self.evt_stage("Compiling generated files (per-file gate)…")
        yield self.evt_roster()

        # ---- ToolchainScout ----------------------------------------------
        if ctx.has_repo:
            build_info = a._detect_build_system(ctx.repo_dir)
            cc = a._detect_cross_compiler(build_info)
            yield self.evt_info(
                f"ToolchainScout: build system "
                f"{build_info.get('type', 'unknown')}, compiler '{cc}'."
            )
        else:
            yield self.evt_info(
                "ToolchainScout: no repository — using the native "
                "embedded-compatible toolchain (gcc -std=gnu99)."
            )

        # ---- BuildOperator: proven per-file compile gate -------------------
        summary: dict = {}
        for evt in a.run_per_file_compile(
            session_dir=ctx.session_dir,
            gen_dir=ctx.gen_dir,
            code_dir=ctx.code_dir,
            repo_dir=ctx.repo_dir,
            has_repo=ctx.has_repo,
            change_spec=str(bb.data.get("change_spec", "")),
            repo_knowledge=str(bb.data.get("repo_knowledge", "")),
            gitnexus_report=str(bb.data.get("gitnexus_report", "")),
            is_resume=False,
            completed_stages=set(),
        ):
            payload = a._parse_sse_event(evt) if isinstance(evt, str) else evt
            if payload is None:
                continue
            if payload.get("type") == "compile_summary":
                summary = payload
            # The gate emits its own stage_complete; the team owns it here.
            if payload.get("type") == "stage_complete":
                continue
            yield payload

        # ---- GateKeeper -----------------------------------------------------
        ok = int(summary.get("ok", 0))
        failed = int(summary.get("failed", 0))
        total = int(summary.get("total", 0))
        bb.data["compile_summary"] = summary

        if failed:
            details = ""
            report = ctx.gen_dir / "compile_report.txt"
            if report.exists():
                details = report.read_text()[-4000:]
            bb.add_feedback(
                source_stage=self.stage, target_stage="transform",
                severity="blocker",
                summary=(
                    f"{failed}/{total} generated file(s) still fail to "
                    "compile after the per-file fix budget."
                ),
                details=details,
            )
            yield self.evt_info(
                f"GateKeeper: {failed} file(s) failed the compile gate — "
                "routing feedback to the Code Generation Team."
            )

        bb.resolve_feedback(self.stage)
        bb.record_stage(
            self.stage,
            "success" if not failed else "partial",
            f"Per-file compile: {ok}/{total} succeeded"
            + (f", {failed} failing." if failed else "."),
            artifacts=["compile_report.txt"],
        )
        yield self.evt_stage_complete()
