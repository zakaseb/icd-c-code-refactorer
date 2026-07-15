"""Agentic multi-agent pipeline for the ICD C code refactorer.

This package reimagines the classic sequential pipeline as an ecosystem of
cooperating multi-agent teams orchestrated by an agent-of-agents
(:class:`~agentic_pipeline.mission.MissionController`):

    Stage (SSE name)   Team                    Purpose
    -----------------  ----------------------  ---------------------------------
    analysis           IngestionTeam           ICD ingestion + delta analysis
    gitnexus           CodebaseTeam            repository / codebase understanding
    transform          GenerationTeam          .h then .c code generation
    verification       VerificationTeam        structural + spec-compliance audit
    compile            CompilationTeam         per-file compile gate (.c -> .o)
    sandbox_build      IntegrationTeam         build generated files in the repo

Every team is a small multi-agent system (planner / worker / critic roles)
that reads from and writes to a shared :class:`~agentic_pipeline.blackboard.
Blackboard`.  Teams publish :class:`~agentic_pipeline.blackboard.Feedback`
items that may target ANY stage — not just the previous one — and the
MissionController routes execution non-sequentially to whichever stage the
feedback says needs fixing, then re-flows the stale downstream stages.

The stage names, SSE event contract and on-disk artifacts are identical to
the classic pipeline, so the existing web UI, download ZIP and QA tooling
work unchanged.
"""

from .blackboard import Blackboard, Feedback, StageRecord  # noqa: F401
from .context import PipelineContext  # noqa: F401
from .mission import STAGE_ORDER, MissionController, run_agentic_pipeline  # noqa: F401
