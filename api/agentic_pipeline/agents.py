"""Agent primitives for the multi-agent teams."""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("agentic_pipeline.agents")


@dataclass(frozen=True)
class Agent:
    """A single specialised agent: a named role bound to a system prompt.

    Deterministic agents (critics that run structural checks, toolchain
    scouts, …) may never call :meth:`act`; they still appear in the team
    roster so the mission log shows who did what.
    """

    name: str
    role: str
    system_prompt: str = ""

    def act(
        self, llm, user_prompt: str, *,
        max_tokens: int = 4096, max_passes: int = 4,
    ) -> str:
        """One reasoning/acting turn for this agent."""
        log.debug("Agent %s acting (%d prompt chars)", self.name, len(user_prompt))
        return llm.complete(
            self.system_prompt, user_prompt,
            max_tokens=max_tokens, max_passes=max_passes,
        )


def roster_line(agents: list[Agent]) -> str:
    return ", ".join(f"{a.name} ({a.role})" for a in agents)
