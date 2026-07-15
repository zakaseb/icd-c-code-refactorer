"""Common machinery for stage teams."""

from __future__ import annotations

import logging
from typing import Iterator

from ..agents import Agent, roster_line
from ..blackboard import Blackboard
from ..context import PipelineContext

log = logging.getLogger("agentic_pipeline.teams")


class Team:
    """Base class for a multi-agent stage team.

    Subclasses define ``stage`` (the SSE stage name the classic pipeline and
    web UI already know), ``agents`` (the roster) and implement ``run()``,
    a generator yielding plain SSE-payload dicts.  ``run()`` must finish by
    calling ``bb.record_stage(...)`` with the outcome and may raise
    :class:`~agentic_pipeline.blackboard.Feedback` items targeting ANY stage.
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
    def run(self, ctx: PipelineContext, bb: Blackboard) -> Iterator[dict]:
        raise NotImplementedError
