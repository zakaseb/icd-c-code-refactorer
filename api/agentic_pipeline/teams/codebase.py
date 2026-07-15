"""Stage 2 — CodebaseTeam (SSE stage: ``gitnexus``).

Multi-agent system for codebase understanding.  Consumes the outputs of the
IngestionTeam (change specification) so its impact assessment is grounded
in what is actually changing — the user-facing point of step 2 taking part
in step 1's output.

Roster
    RepoCartographer     — deterministic: extracts the repository knowledge
                           inventory (structs, enums, signatures, globals,
                           macros, tree) in both full and distilled flavours.
    SystemsArchaeologist — deterministic: runs the GitNexus extractors
                           (ISR/task wiring, RTOS/superloop, drivers, state
                           machines, memory ownership, HAL/BSP, bootloader,
                           safety, include/global graphs).
    ImpactAssessor       — LLM: fuses the change specification with the
                           codebase knowledge into an impact map (which
                           modules/files are touched and why) shared with
                           downstream teams via the blackboard.

Artifacts (identical to the classic pipeline)
    repo_knowledge.txt, gitnexus_report.txt (session + generated_code),
    icd_analysis.txt enriched with the full repo knowledge.
"""

from __future__ import annotations

from typing import Iterator

from ..agents import Agent
from ..blackboard import Blackboard
from ..context import PipelineContext
from .base import Team


class CodebaseTeam(Team):
    stage = "gitnexus"
    title = "Codebase Understanding Team"

    def __init__(self):
        super().__init__()
        self.cartographer = Agent(
            name="RepoCartographer",
            role="deterministic repository inventory",
        )
        self.archaeologist = Agent(
            name="SystemsArchaeologist",
            role="deterministic embedded-relationship extraction (GitNexus)",
        )
        self.impact_assessor = Agent(
            name="ImpactAssessor",
            role="change-impact mapping",
            system_prompt=(
                "You are an embedded software architect. Given an ICD "
                "change specification and a codebase inventory, you map "
                "each specified change to the concrete modules, files, "
                "types and functions it impacts, including RTOS tasks, "
                "ISRs and drivers. Be concise and concrete; output a "
                "bullet list grouped by file/module."
            ),
        )
        self.agents = [
            self.cartographer, self.archaeologist, self.impact_assessor,
        ]

    def run(self, ctx: PipelineContext, bb: Blackboard) -> Iterator[dict]:
        a = ctx.app
        yield self.evt_stage(
            "Understanding the codebase (repository inventory + GitNexus)…"
        )
        yield self.evt_roster()

        artifacts: list[str] = []

        # ---- RepoCartographer -------------------------------------------
        repo_knowledge_full = ""
        repo_knowledge = ""
        if ctx.has_repo:
            yield self.evt_info(
                "RepoCartographer: extracting repository knowledge "
                "(structs, enums, signatures, globals, macros)…"
            )
            repo_knowledge_full = a._build_repo_knowledge(
                ctx.repo_dir, full=True,
            )
            repo_knowledge = a._build_repo_knowledge(
                ctx.repo_dir, full=False,
            )
            (ctx.session_dir / "repo_knowledge.txt").write_text(
                repo_knowledge_full
            )
            artifacts.append("repo_knowledge.txt")
        else:
            yield self.evt_info(
                "RepoCartographer: no repository uploaded — skipping "
                "repository inventory."
            )
        bb.data["repo_knowledge"] = repo_knowledge
        bb.data["repo_knowledge_full"] = repo_knowledge_full

        # ---- SystemsArchaeologist (GitNexus) -----------------------------
        yield self.evt_info(
            "SystemsArchaeologist: running GitNexus extractors…"
        )
        gitnexus_full = a._build_gitnexus_report(
            repo_dir=ctx.repo_dir if ctx.has_repo else None,
            code_dir=ctx.code_dir,
            full=True,
        )
        gitnexus_report = a._build_gitnexus_report(
            repo_dir=ctx.repo_dir if ctx.has_repo else None,
            code_dir=ctx.code_dir,
            full=False,
        )
        (ctx.session_dir / "gitnexus_report.txt").write_text(gitnexus_full)
        ctx.gen_dir.mkdir(parents=True, exist_ok=True)
        (ctx.gen_dir / "gitnexus_report.txt").write_text(gitnexus_full)
        artifacts.append("gitnexus_report.txt")
        bb.data["gitnexus_report"] = gitnexus_report
        bb.data["gitnexus_full"] = gitnexus_full

        # ---- Enrich icd_analysis.txt with the full repo knowledge --------
        # (same final artifact content as the classic pipeline: distilled
        # change spec followed by the full repository knowledge).
        change_spec = str(bb.data.get("change_spec", ""))
        if change_spec:
            parts = [change_spec]
            if repo_knowledge_full:
                parts.append("\n\n" + repo_knowledge_full)
            full_analysis = "\n".join(parts)
            (ctx.session_dir / "icd_analysis.txt").write_text(full_analysis)
            (ctx.gen_dir / "icd_analysis.txt").write_text(full_analysis)
            if "icd_analysis.txt" not in artifacts:
                artifacts.append("icd_analysis.txt")

        # ---- ImpactAssessor ----------------------------------------------
        impact_map = ""
        if change_spec:
            yield self.evt_info(
                "ImpactAssessor: mapping specified changes onto the "
                "codebase…"
            )
            feedback = self.feedback_section(bb)
            prompt = (
                f"## Change specification\n{change_spec[:12_000]}\n\n"
                f"## Repository knowledge (distilled)\n"
                f"{repo_knowledge[:8_000]}\n\n"
                f"## GitNexus report (distilled)\n"
                f"{gitnexus_report[:8_000]}\n\n"
            )
            if feedback:
                prompt += f"{feedback}\n\n"
            prompt += (
                "Produce the impact map: for each specified change, list "
                "the impacted files/modules/functions/types and any "
                "RTOS/ISR/driver relationships that constrain the edit."
            )
            try:
                impact_map = self.impact_assessor.act(
                    ctx.llm, prompt, max_tokens=2048, max_passes=3,
                ).strip()
            except Exception:  # noqa: BLE001
                impact_map = ""
        bb.data["impact_map"] = impact_map
        if impact_map:
            yield self.evt_info(
                f"ImpactAssessor: impact map ready ({len(impact_map):,} "
                "chars) — shared with downstream teams."
            )

        bb.resolve_feedback(self.stage)
        bb.record_stage(
            self.stage, "success",
            f"repo_knowledge {len(repo_knowledge_full):,} chars, gitnexus "
            f"{len(gitnexus_full):,} chars, impact map "
            f"{len(impact_map):,} chars.",
            artifacts=artifacts,
        )
        yield self.evt_stage_complete()
