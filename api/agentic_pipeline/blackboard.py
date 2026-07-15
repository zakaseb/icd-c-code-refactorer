"""Shared blackboard for the agentic pipeline.

The blackboard is the single source of truth that every team reads from and
writes to.  It carries three kinds of state:

* ``data``      — large in-memory artifacts (change_spec, repo_knowledge,
                  generated file map, …) shared between teams.  Never
                  persisted wholesale; the artifact FILES on disk are the
                  durable copies, exactly as in the classic pipeline.
* ``stages``    — one :class:`StageRecord` per pipeline stage describing its
                  latest execution (status, summary, artifacts, run count).
* ``feedback``  — :class:`Feedback` items raised by any team and TARGETED at
                  any stage.  This is what makes reiteration non-sequential:
                  a build failure can send the pipeline straight back to
                  analysis or generation, not merely the previous step.

A compact JSON snapshot (without the large ``data`` texts) is persisted to
``session_dir/agentic_pipeline/blackboard.json`` after every stage run so a
human (or a future resume feature) can audit the mission.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Feedback:
    """A routable piece of feedback raised by one team about another stage."""

    source_stage: str
    target_stage: str
    severity: str          # "info" | "warning" | "blocker"
    summary: str
    details: str = ""
    resolved: bool = False
    created_at: float = field(default_factory=time.time)

    def is_blocker(self) -> bool:
        return self.severity == "blocker" and not self.resolved


@dataclass
class StageRecord:
    """Latest execution record for one pipeline stage."""

    stage: str
    status: str = "pending"   # pending | success | partial | failed | stale
    summary: str = ""
    artifacts: list[str] = field(default_factory=list)
    runs: int = 0
    updated_at: float = 0.0


class Blackboard:
    """Shared mission state for all teams + the MissionController."""

    def __init__(self, session_dir: Path, stage_order: list[str]):
        self.session_dir = session_dir
        self.stage_order = list(stage_order)
        self.stages: dict[str, StageRecord] = {
            s: StageRecord(stage=s) for s in stage_order
        }
        self.feedback: list[Feedback] = []
        self.data: dict[str, object] = {}
        self.history: list[dict] = []      # chronological stage executions
        self._store_dir = session_dir / "agentic_pipeline"

    # ------------------------------------------------------------------
    # Stage records
    # ------------------------------------------------------------------
    def record_stage(
        self, stage: str, status: str, summary: str = "",
        artifacts: list[str] | None = None,
    ) -> StageRecord:
        rec = self.stages[stage]
        rec.status = status
        rec.summary = summary
        if artifacts is not None:
            rec.artifacts = list(artifacts)
        rec.runs += 1
        rec.updated_at = time.time()
        self.history.append({
            "stage": stage, "status": status, "summary": summary,
            "run": rec.runs, "ts": rec.updated_at,
        })
        return rec

    def mark_stale_after(self, stage: str) -> list[str]:
        """Mark every stage AFTER *stage* as stale (must re-run).

        Called when the mission revisits *stage*: its downstream stages
        consumed the now-outdated artifacts, so they are queued to re-flow
        in order once the revisited stage completes.
        """
        idx = self.stage_order.index(stage)
        stale: list[str] = []
        for s in self.stage_order[idx + 1:]:
            if self.stages[s].status not in ("pending",):
                self.stages[s].status = "stale"
                stale.append(s)
        return stale

    def next_pending_stage(self) -> str | None:
        """First stage (in pipeline order) that still needs a run."""
        for s in self.stage_order:
            if self.stages[s].status in ("pending", "stale"):
                return s
        return None

    def stage_ran(self, stage: str) -> bool:
        return self.stages[stage].runs > 0

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------
    def add_feedback(
        self, *, source_stage: str, target_stage: str, severity: str,
        summary: str, details: str = "",
    ) -> Feedback:
        fb = Feedback(
            source_stage=source_stage, target_stage=target_stage,
            severity=severity, summary=summary, details=details,
        )
        self.feedback.append(fb)
        return fb

    def pending_feedback(self, target_stage: str | None = None) -> list[Feedback]:
        items = [f for f in self.feedback if not f.resolved]
        if target_stage is not None:
            items = [f for f in items if f.target_stage == target_stage]
        return items

    def pending_blockers(self) -> list[Feedback]:
        return [f for f in self.feedback if f.is_blocker()]

    def resolve_feedback(self, target_stage: str) -> int:
        """Mark all pending feedback aimed at *target_stage* as resolved.

        Called after that stage re-ran with the feedback injected into its
        prompts — the re-run is the response to the feedback.
        """
        n = 0
        for f in self.feedback:
            if not f.resolved and f.target_stage == target_stage:
                f.resolved = True
                n += 1
        return n

    def feedback_digest(self, target_stage: str, max_chars: int = 6000) -> str:
        """Human/LLM-readable digest of pending feedback for one stage."""
        items = self.pending_feedback(target_stage)
        if not items:
            return ""
        parts = [
            "## Feedback from downstream teams (address ALL of these)\n",
        ]
        for f in items:
            parts.append(
                f"- [{f.severity}] from {f.source_stage}: {f.summary}"
            )
            if f.details:
                parts.append(f"  Details: {f.details[:1500]}")
        return "\n".join(parts)[:max_chars]

    # ------------------------------------------------------------------
    # Persistence (audit snapshot; large texts excluded)
    # ------------------------------------------------------------------
    def save(self) -> Path:
        self._store_dir.mkdir(parents=True, exist_ok=True)
        snap = {
            "stage_order": self.stage_order,
            "stages": {s: asdict(r) for s, r in self.stages.items()},
            "feedback": [asdict(f) for f in self.feedback],
            "history": self.history[-200:],
            "data_keys": {
                k: (len(v) if isinstance(v, str) else str(type(v).__name__))
                for k, v in self.data.items()
            },
        }
        out = self._store_dir / "blackboard.json"
        out.write_text(json.dumps(snap, indent=2))
        return out
