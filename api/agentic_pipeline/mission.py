"""MissionController — the agent-of-agents orchestrating the stage teams.

The controller runs the six multi-agent teams and, between stage runs, asks
its Router agent where to go next.  Unlike the classic pipeline (strictly
sequential, each fix loop only looking one step back), the router may send
the mission BACK to any earlier stage that pending feedback targets —
analysis, codebase understanding, generation, verification, compilation or
the integration build — and then re-flows the invalidated downstream stages
in order.  Iteration is bounded by ``AGENTIC_MAX_STAGE_RUNS`` and
``AGENTIC_MAX_REVISITS_PER_STAGE`` so a mission always terminates.

Routing is LLM-assisted (the Router agent sees a mission digest and returns
a strict-JSON decision) with a fully deterministic fallback, so the mission
keeps moving even if the router output is unusable.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Iterator

from .agents import Agent
from .blackboard import Blackboard
from .context import PipelineContext
from .llm import make_agent_llm
from .teams import (
    CodebaseTeam,
    CompilationTeam,
    GenerationTeam,
    IngestionTeam,
    IntegrationTeam,
    VerificationTeam,
)

log = logging.getLogger("agentic_pipeline.mission")

STAGE_ORDER = [
    "analysis", "gitnexus", "transform",
    "verification", "compile", "sandbox_build",
]

MAX_STAGE_RUNS = int(os.environ.get("AGENTIC_MAX_STAGE_RUNS", "12"))
MAX_REVISITS_PER_STAGE = int(
    os.environ.get("AGENTIC_MAX_REVISITS_PER_STAGE", "2")
)
ROUTER_USE_LLM = os.environ.get(
    "AGENTIC_ROUTER_LLM", "1"
).strip().lower() not in ("0", "false", "no", "off")


class MissionController:
    """Agent-of-agents: runs teams, routes feedback, owns the mission goal.

    The mission goal is the project goal: code files that build successfully
    within their codebase.  The controller therefore treats a successful
    integration build (or, without a repository, a clean compile gate) as
    mission complete.
    """

    def __init__(self, ctx: PipelineContext):
        self.ctx = ctx
        self.bb = Blackboard(ctx.session_dir, STAGE_ORDER)
        self.teams = {
            "analysis": IngestionTeam(),
            "gitnexus": CodebaseTeam(),
            "transform": GenerationTeam(),
            "verification": VerificationTeam(),
            "compile": CompilationTeam(),
            "sandbox_build": IntegrationTeam(),
        }
        self.router = Agent(
            name="Router",
            role="mission routing (agent-of-agents)",
            system_prompt=(
                "You are the routing brain of an agent-of-agents system "
                "refactoring embedded C code to a new ICD. Given the "
                "mission digest, decide the next action. Respond with "
                "STRICT JSON only: "
                '{"action": "proceed" | "revisit", '
                '"target": "<stage name>", "reason": "<short reason>"}. '
                "Choose \"revisit\" with the most upstream stage whose "
                "output is causing pending blocker feedback; choose "
                "\"proceed\" when the pipeline should simply continue "
                "forward. Valid stages: "
                + ", ".join(STAGE_ORDER) + "."
            ),
        )

    # ------------------------------------------------------------------
    def _evt(self, message: str, stage: str, etype: str = "info") -> dict:
        return {"type": etype, "stage": stage, "message": message}

    # ------------------------------------------------------------------
    def run(self) -> Iterator[dict]:
        bb, ctx = self.bb, self.ctx
        yield self._evt(
            "[MissionController] Agentic pipeline engaged — 6 multi-agent "
            f"teams, LLM backend: {getattr(ctx.llm, 'name', 'custom')}. "
            "Mission goal: generated code that builds inside its codebase.",
            stage="analysis",
        )

        runs = 0
        current = bb.next_pending_stage()
        while current is not None and runs < MAX_STAGE_RUNS:
            team = self.teams[current]
            runs += 1
            try:
                yield from team.run(ctx, bb)
            except Exception as e:  # noqa: BLE001
                log.exception("Team %s crashed: %s", current, e)
                bb.record_stage(
                    current, "failed", f"Team crashed: {e}",
                )
                bb.add_feedback(
                    source_stage=current, target_stage=current,
                    severity="blocker",
                    summary=f"Team crashed and must be re-run: {e}",
                )
                yield self._evt(
                    f"[MissionController] {team.title} crashed ({e}) — "
                    "consulting the router.",
                    stage=current,
                )
            # Guarantee a consolidated team report exists for Download All /
            # preview even if the team crashed or forgot to emit one.
            yield from team.ensure_consolidated_report(ctx, bb)
            bb.save()

            decision = self._route(current)
            action = decision.get("action")
            target = decision.get("target", "")
            reason = decision.get("reason", "")

            if action == "revisit":
                yield self._evt(
                    f"[MissionController] Revisiting the '{target}' stage "
                    f"({reason}) — downstream stages will re-flow.",
                    stage=target,
                )
                bb.stages[target].status = "stale"
                bb.mark_stale_after(target)
                current = target
                continue

            current = bb.next_pending_stage()

        if runs >= MAX_STAGE_RUNS and bb.next_pending_stage() is not None:
            yield self._evt(
                f"[MissionController] Stage-run budget exhausted "
                f"({MAX_STAGE_RUNS}) — finalising with the best artifacts "
                "produced so far.",
                stage="sandbox_build",
            )

        yield from self._finalise()

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def _route(self, just_ran: str) -> dict:
        """Decide the next move after *just_ran* finished.

        Pending blocker feedback (with revisit budget) FORCES a revisit —
        the Router agent only arbitrates WHICH eligible target to revisit
        first; it may not veto the revisit itself. Without blockers the
        mission simply proceeds (no LLM call spent).
        """
        deterministic = self._deterministic_route(just_ran)
        if deterministic.get("action") == "proceed" or not ROUTER_USE_LLM:
            return deterministic
        eligible = self._eligible_revisit_targets()
        if len(eligible) > 1:
            llm_decision = self._llm_route(just_ran)
            if (
                llm_decision is not None
                and llm_decision.get("action") == "revisit"
                and llm_decision.get("target") in eligible
            ):
                return llm_decision
        return deterministic

    def _eligible_revisit_targets(self) -> list[str]:
        """Blocker-targeted stages that still have revisit budget."""
        bb = self.bb
        targets: list[str] = []
        for f in bb.pending_blockers():
            if f.target_stage not in STAGE_ORDER:
                continue
            if bb.stages[f.target_stage].runs >= MAX_REVISITS_PER_STAGE + 1:
                continue
            if f.target_stage not in targets:
                targets.append(f.target_stage)
        return targets

    def _deterministic_route(self, just_ran: str) -> dict:
        """Policy fallback: blockers pull the mission to the most upstream
        target stage with revisit budget; otherwise proceed."""
        bb = self.bb
        blockers = bb.pending_blockers()
        candidates = []
        for f in blockers:
            if f.target_stage not in STAGE_ORDER:
                continue
            if bb.stages[f.target_stage].runs >= MAX_REVISITS_PER_STAGE + 1:
                continue    # budget spent — stop bouncing to this stage
            candidates.append(f.target_stage)
        if candidates:
            target = min(candidates, key=STAGE_ORDER.index)
            return {
                "action": "revisit", "target": target,
                "reason": "pending blocker feedback targets this stage",
            }
        return {"action": "proceed", "target": "", "reason": "no blockers"}

    def _llm_route(self, just_ran: str) -> dict | None:
        """Ask the Router agent; validate hard before trusting it."""
        bb = self.bb
        digest = self._mission_digest(just_ran)
        try:
            out = self.router.act(
                self.ctx.llm, digest, max_tokens=256, max_passes=1,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("Router agent failed (%s); deterministic fallback.", e)
            return None
        data = self.ctx.app._extract_json_object(out)
        if not isinstance(data, dict):
            return None
        action = str(data.get("action", "")).strip().lower()
        target = str(data.get("target", "")).strip()
        if action == "proceed":
            return {
                "action": "proceed", "target": "",
                "reason": str(data.get("reason", ""))[:200],
            }
        if action == "revisit" and target in STAGE_ORDER:
            if not bb.stage_ran(target):
                return None       # cannot revisit a stage that never ran
            if bb.stages[target].runs >= MAX_REVISITS_PER_STAGE + 1:
                return None       # revisit budget exhausted
            return {
                "action": "revisit", "target": target,
                "reason": str(data.get("reason", ""))[:200],
            }
        return None

    def _mission_digest(self, just_ran: str) -> str:
        bb = self.bb
        lines = [
            f"Stage just finished: {just_ran}",
            "",
            "## Stage records",
        ]
        for s in STAGE_ORDER:
            r = bb.stages[s]
            lines.append(
                f"- {s}: status={r.status}, runs={r.runs}, "
                f"summary={r.summary[:160]}"
            )
        fb = bb.pending_feedback()
        lines.append("")
        lines.append(f"## Pending feedback ({len(fb)})")
        for f in fb[:10]:
            lines.append(
                f"- [{f.severity}] {f.source_stage} -> {f.target_stage}: "
                f"{f.summary[:200]}"
            )
        lines.append("")
        lines.append(
            "Decide the next action (strict JSON). Revisit budget per "
            f"stage: {MAX_REVISITS_PER_STAGE} extra run(s)."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _finalise(self) -> Iterator[dict]:
        """Write final status + emit the classic ``complete`` event."""
        ctx, bb = self.ctx, self.bb
        bb.save()
        sandbox_success = bb.data.get("sandbox_build_success", None)

        status = ctx.app._read_status(ctx.session_dir)
        status["pause_requested"] = False
        status["state"] = "completed"
        if sandbox_success is not None:
            status["sandbox_build_success"] = sandbox_success
        status["generated_files"] = sorted(
            p.name for p in ctx.gen_dir.iterdir()
            if p.is_file() and p.suffix.lower() != ".o"
        ) if ctx.gen_dir.exists() else []
        ctx.app._write_status(ctx.session_dir, status)

        payload = {"type": "complete", "files": status["generated_files"]}
        if sandbox_success is not None:
            payload["sandbox_build"] = bool(sandbox_success)
        yield payload


def run_agentic_pipeline(ctx: PipelineContext | None = None, **kw) -> Iterator[dict]:
    """Entry point: build the context (if needed) and run the mission.

    Accepts either a ready :class:`PipelineContext` or the keyword
    arguments to build one (``session_dir``, ``gen_dir``, ``code_dir``,
    ``repo_dir``, ``has_repo``, ``source_icd``, ``target_icd`` and
    optionally ``llm``/``app``).
    """
    if ctx is None:
        if "llm" not in kw or kw["llm"] is None:
            kw["llm"] = make_agent_llm()
        ctx = PipelineContext(**kw)
    controller = MissionController(ctx)
    yield from controller.run()
