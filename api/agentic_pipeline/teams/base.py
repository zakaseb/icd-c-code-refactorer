"""Common machinery for stage teams."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

from ..agents import Agent, roster_line
from ..blackboard import Blackboard
from ..context import PipelineContext

log = logging.getLogger("agentic_pipeline.teams")

TEAM_REPORTS_DIRNAME = "team_reports"


class Team:
    """Base class for a multi-agent stage team.

    Subclasses define ``stage`` (the SSE stage name the classic pipeline and
    web UI already know), ``agents`` (the roster) and implement ``run()``,
    a generator yielding plain SSE-payload dicts.  ``run()`` must finish by
    calling ``bb.record_stage(...)`` with the outcome and may raise
    :class:`~agentic_pipeline.blackboard.Feedback` items targeting ANY stage.

    Every team also writes a single consolidated findings report via
    :meth:`emit_consolidated_report` so Download All / preview stay current
    during the mission (same live-deliverables path as sandbox edits).
    """

    stage: str = ""
    title: str = ""

    def __init__(self):
        self.agents: list[Agent] = []

    # ------------------------------------------------------------------
    # Event helpers — mirror the classic pipeline's SSE payload shapes.
    # ------------------------------------------------------------------
    def evt_stage(self, message: str, **kw) -> dict:
        return {"type": "stage", "stage": self.stage, "message": message, **kw}

    def evt_info(self, message: str, **kw) -> dict:
        return {"type": "info", "stage": self.stage, "message": message, **kw}

    def evt_stage_complete(self, **kw) -> dict:
        return {"type": "stage_complete", "stage": self.stage, **kw}

    def evt_roster(self) -> dict:
        return self.evt_info(
            f"[{self.title or type(self).__name__}] agents on deck: "
            f"{roster_line(self.agents)}"
        )

    def feedback_section(self, bb: Blackboard) -> str:
        """Digest of pending feedback aimed at this stage (for prompts)."""
        return bb.feedback_digest(self.stage)

    # ------------------------------------------------------------------
    # Consolidated team findings report (downloadable deliverable)
    # ------------------------------------------------------------------
    def report_basename(self) -> str:
        """Filename used under ``generated_code/`` and the team_reports dir."""
        return f"team_report_{self.stage}.md"

    def team_reports_dir(self, ctx: PipelineContext) -> Path:
        d = ctx.session_dir / "agentic_pipeline" / TEAM_REPORTS_DIRNAME
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_consolidated_report(
        self,
        ctx: PipelineContext,
        bb: Blackboard,
        findings: str | list[str],
        *,
        status: str = "",
        summary: str = "",
        artifacts: list[str] | None = None,
    ) -> Path:
        """Write one consolidated Markdown report for this team's run.

        Durable copies:
          * ``session_dir/agentic_pipeline/team_reports/{stage}_team_report.md``
          * ``generated_code/team_report_{stage}.md`` (preview + Download All)
        """
        if isinstance(findings, list):
            formatted: list[str] = []
            for line in findings:
                s = str(line)
                if not s.strip():
                    formatted.append("")
                elif (
                    "\n" in s
                    or s.startswith("#")
                    or s.startswith("```")
                    or s.startswith("- ")
                    or s.startswith("* ")
                    or s.startswith("  ")
                ):
                    # Preserve structured markdown / multi-line excerpts as-is.
                    formatted.append(s)
                else:
                    formatted.append(f"- {s}")
            findings_body = "\n".join(formatted)
        else:
            findings_body = str(findings).rstrip()

        rec = bb.stages.get(self.stage)
        status = status or (rec.status if rec else "unknown")
        summary = summary or (rec.summary if rec else "")
        artifacts = list(
            artifacts
            if artifacts is not None
            else (rec.artifacts if rec else [])
        )

        pending = bb.pending_feedback(self.stage)
        raised = [
            f for f in bb.feedback
            if f.source_stage == self.stage and not f.resolved
        ]

        lines = [
            f"# {self.title or self.stage} — consolidated findings",
            "",
            f"- **Stage:** `{self.stage}`",
            f"- **Status:** {status}",
            f"- **Runs (this stage):** {rec.runs if rec else 0}",
            f"- **Updated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
            f"- **Agents:** {roster_line(self.agents)}",
            "",
            "## Summary",
            "",
            summary or "(no summary)",
            "",
            "## Findings",
            "",
            findings_body or "(no detailed findings recorded)",
            "",
        ]
        if artifacts:
            lines.extend([
                "## Artifacts produced",
                "",
                *[f"- `{name}`" for name in artifacts],
                "",
            ])
        if raised:
            lines.append("## Feedback raised by this team")
            lines.append("")
            for fb in raised:
                lines.append(
                    f"- **[{fb.severity}] → `{fb.target_stage}`:** {fb.summary}"
                )
                if fb.details:
                    detail = fb.details.strip()
                    if len(detail) > 1500:
                        detail = detail[:1500] + " …"
                    lines.append(f"  ```\n  {detail}\n  ```")
            lines.append("")
        if pending:
            lines.append("## Pending feedback targeting this stage")
            lines.append("")
            for fb in pending:
                lines.append(
                    f"- **[{fb.severity}] from `{fb.source_stage}`:** {fb.summary}"
                )
            lines.append("")

        text = "\n".join(lines).rstrip() + "\n"
        archive = self.team_reports_dir(ctx) / f"{self.stage}_team_report.md"
        archive.write_text(text)
        ctx.gen_dir.mkdir(parents=True, exist_ok=True)
        live = ctx.gen_dir / self.report_basename()
        live.write_text(text)
        log.info(
            "Team report written: stage=%s archive=%s live=%s (%d chars)",
            self.stage, archive, live, len(text),
        )
        return live

    def emit_consolidated_report(
        self,
        ctx: PipelineContext,
        bb: Blackboard,
        findings: str | list[str],
        *,
        status: str = "",
        summary: str = "",
        artifacts: list[str] | None = None,
    ) -> Iterator[dict]:
        """Write the team report and notify the UI / download path."""
        path = self.write_consolidated_report(
            ctx, bb, findings,
            status=status, summary=summary, artifacts=artifacts,
        )
        yield self.evt_info(
            f"Consolidated team report ready: {path.name}"
        )
        # Same live-deliverables channel used during sandbox_build.
        files = sorted(
            p.name for p in ctx.gen_dir.iterdir()
            if p.is_file() and p.suffix.lower() != ".o"
        ) if ctx.gen_dir.is_dir() else [path.name]
        yield {
            "type": "deliverables_updated",
            "stage": self.stage,
            "files": files,
            "synced": [path.name],
            "reason": f"team_report:{self.stage}",
        }

    def ensure_consolidated_report(
        self, ctx: PipelineContext, bb: Blackboard,
    ) -> Iterator[dict]:
        """Write a fallback report from the blackboard if none exists yet."""
        live = ctx.gen_dir / self.report_basename()
        if live.exists() and live.stat().st_size > 0:
            return
        rec = bb.stages.get(self.stage)
        findings = [
            "Auto-generated fallback report (team did not emit detailed findings).",
            f"Stage status: {rec.status if rec else 'unknown'}",
        ]
        yield from self.emit_consolidated_report(
            ctx, bb, findings,
            status=rec.status if rec else "unknown",
            summary=rec.summary if rec else "",
            artifacts=list(rec.artifacts) if rec else None,
        )

    # ------------------------------------------------------------------
    def run(self, ctx: PipelineContext, bb: Blackboard) -> Iterator[dict]:
        raise NotImplementedError
