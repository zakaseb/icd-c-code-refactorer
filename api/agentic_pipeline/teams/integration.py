"""Stage 6 — IntegrationTeam (SSE stage: ``sandbox_build``).

Multi-agent system building the generated files inside the codebase — the
mission's final goal.

Roster
    SandboxEngineer — copies the repository into a sandbox, injects the
                      generated files and drives the build (make/cmake with
                      the detected cross toolchain; remote Xilinx build when
                      ``REMOTE_BUILD_ENABLED``).
    DebugCrew       — the existing agentic debugger / ReAct orchestrator
                      (state-machine triage, hypothesis, patch, verify) that
                      iterates on build failures inside the sandbox.
    IntegrationJudge — deterministic: interprets the final build outcome and
                      raises targeted feedback for earlier stages when the
                      sandbox could not converge.

Artifacts (identical to the classic pipeline)
    built_repo.zip, sandbox_build_log.txt, agentic_attempts/, playbook.jsonl
"""

from __future__ import annotations

from typing import Iterator

from ..agents import Agent
from ..blackboard import Blackboard
from ..context import PipelineContext
from .base import Team


class IntegrationTeam(Team):
    stage = "sandbox_build"
    title = "Integration & Build Team"

    def __init__(self):
        super().__init__()
        self.sandbox_engineer = Agent(
            name="SandboxEngineer",
            role="sandbox setup + build driving",
        )
        self.debug_crew = Agent(
            name="DebugCrew",
            role="iterative build debugging (agentic debugger/orchestrator)",
        )
        self.integration_judge = Agent(
            name="IntegrationJudge",
            role="deterministic outcome triage + feedback routing",
        )
        self.agents = [
            self.sandbox_engineer, self.debug_crew, self.integration_judge,
        ]

    def run(self, ctx: PipelineContext, bb: Blackboard) -> Iterator[dict]:
        a = ctx.app

        if not (ctx.has_repo or a.REMOTE_BUILD_ENABLED):
            yield self.evt_info(
                "No repository uploaded and remote build disabled — "
                "skipping the integration build."
            )
            bb.data["sandbox_build_success"] = None
            bb.record_stage(
                self.stage, "success",
                "Skipped (no repository to integrate into).",
            )
            yield self.evt_stage_complete()
            return

        yield self.evt_stage(
            "Building generated code inside the repository sandbox…"
        )
        yield self.evt_roster()

        build_iter = (
            a._remote_build_iterate(
                ctx.session_dir, ctx.gen_dir,
                workspace_path=str(ctx.session_dir),
            )
            if a.REMOTE_BUILD_ENABLED
            else a._sandbox_build_iterate(
                session_dir=ctx.session_dir,
                gen_dir=ctx.gen_dir,
                repo_dir=ctx.repo_dir,
                change_spec=str(bb.data.get("change_spec", "")),
                uploaded_names=ctx.uploaded_names,
                has_repo=ctx.has_repo,
                repo_knowledge=str(bb.data.get("repo_knowledge", "")),
                gitnexus_report=str(bb.data.get("gitnexus_report", "")),
                sandbox_max_retries=a._resolve_sandbox_max_retries(
                    None, a._read_status(ctx.session_dir),
                ),
            )
        )

        success = None
        for evt in build_iter:
            payload = a._parse_sse_event(evt) if isinstance(evt, str) else evt
            if payload is None:
                continue
            if payload.get("type") == "sandbox_build_result":
                success = payload.get("success", False)
            yield payload

        bb.data["sandbox_build_success"] = success

        # ---- IntegrationJudge ----------------------------------------------
        if success:
            bb.resolve_feedback(self.stage)
            bb.record_stage(
                self.stage, "success",
                "Generated files built successfully inside the codebase.",
            )
        else:
            log_tail = ""
            log_file = ctx.session_dir / "sandbox_build_log.txt"
            if log_file.exists():
                log_tail = log_file.read_text()[-4000:]
            target = self._route_failure(log_tail, bb)
            bb.add_feedback(
                source_stage=self.stage, target_stage=target,
                severity="blocker",
                summary=(
                    "Sandbox build did not converge — routing to the "
                    f"'{target}' stage."
                ),
                details=log_tail,
            )
            yield self.evt_info(
                "IntegrationJudge: build did not converge — feedback "
                f"routed to the '{target}' stage."
            )
            bb.resolve_feedback(self.stage)
            bb.record_stage(
                self.stage, "failed",
                "Sandbox build failed after the debug crew's budget.",
            )
        yield self.evt_stage_complete()

    # ------------------------------------------------------------------
    def _route_failure(self, log_tail: str, bb: Blackboard) -> str:
        """Choose which stage should absorb a build failure.

        Errors inside generated files point at generation; anything else
        (include topology, repo-side breakage) goes to the compile gate
        which owns toolchain/shim mechanics.
        """
        gen_map = bb.data.get("generated_files")
        gen_names = set(gen_map.keys()) if isinstance(gen_map, dict) else set()
        if log_tail and any(name in log_tail for name in gen_names):
            return "transform"
        return "compile"
