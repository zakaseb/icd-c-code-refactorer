"""LLM backends for the agentic pipeline agents.

Two interchangeable backends drive the SAME local model:

* :class:`ClaudeSDKBackend` — routes agent calls through the Claude Agent
  SDK (``claude-agent-sdk``), pointed at a LOCAL Anthropic-compatible
  endpoint via ``ANTHROPIC_BASE_URL``.  Both Ollama (>= 0.14, native
  ``/v1/messages``) and the LiteLLM proxy that ships with this project's
  Docker deployment speak that protocol, so the agents run on the exact
  same local model as the classic pipeline — no cloud traffic.

* :class:`LocalLLMBackend` — calls the project's existing
  ``_call_llm_complete`` helper (llama-server, OpenAI-compatible) directly.
  This is the zero-dependency fallback and behaves identically to the
  classic pipeline's LLM path.

Selection (env ``AGENTIC_LLM_BACKEND``):

* ``auto``  (default) — Claude Agent SDK when importable, else local.
* ``claude``          — force the SDK (still degrades to local per-call on
                        runtime errors so a mission never dies mid-flight).
* ``local``           — force the direct llama-server path.

Related env vars:

* ``AGENTIC_ANTHROPIC_BASE_URL`` / ``ANTHROPIC_BASE_URL`` — the local
  Anthropic-compatible endpoint (default: the project's LiteLLM proxy,
  ``http://127.0.0.1:4000``; use ``http://127.0.0.1:11434`` for Ollama).
* ``AGENTIC_CLAUDE_MODEL`` — model name announced to the endpoint
  (default: the same ``HF_MODEL``-derived name the classic pipeline uses).
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger("agentic_pipeline.llm")


class LocalLLMBackend:
    """Direct llama-server path — identical to the classic pipeline."""

    name = "local-llama-server"

    def __init__(self, complete_fn=None):
        # ``complete_fn`` injectable for tests; defaults to app helper.
        self._complete_fn = complete_fn

    def _fn(self):
        if self._complete_fn is None:
            import app as _app  # lazy: avoid import cycle at module load
            self._complete_fn = _app._call_llm_complete
        return self._complete_fn

    def complete(
        self, system_prompt: str, user_prompt: str, *,
        max_tokens: int = 4096, max_passes: int = 4,
    ) -> str:
        return self._fn()(
            system_prompt, user_prompt,
            max_tokens=max_tokens, max_passes=max_passes,
        )


class ClaudeSDKBackend:
    """Claude Agent SDK backend targeting a local Anthropic-compatible endpoint."""

    name = "claude-agent-sdk"

    def __init__(self, *, model: str, base_url: str, auth_token: str = "ollama"):
        self.model = model
        self.base_url = base_url
        self.auth_token = auth_token

    def _env(self) -> dict[str, str]:
        # Model-tier mapping prevents the SDK from requesting Anthropic
        # tier names the local endpoint doesn't serve.
        return {
            "ANTHROPIC_BASE_URL": self.base_url,
            "ANTHROPIC_AUTH_TOKEN": self.auth_token,
            "ANTHROPIC_API_KEY": "",
            "ANTHROPIC_MODEL": self.model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": self.model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": self.model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": self.model,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }

    def complete(
        self, system_prompt: str, user_prompt: str, *,
        max_tokens: int = 4096, max_passes: int = 4,
    ) -> str:
        import anyio
        from claude_agent_sdk import (  # type: ignore[import-not-found]
            AssistantMessage,
            ClaudeAgentOptions,
            TextBlock,
            query,
        )

        options = ClaudeAgentOptions(
            model=self.model,
            system_prompt=system_prompt,
            max_turns=1,
            allowed_tools=[],
            env=self._env(),
        )

        async def _run() -> str:
            parts: list[str] = []
            async for message in query(prompt=user_prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            parts.append(block.text)
            return "".join(parts).strip()

        return anyio.run(_run)


class ResilientLLM:
    """Primary backend with per-call degradation to a fallback backend.

    A mission spans dozens of agent calls; a transient SDK/CLI failure must
    not kill it.  After ``max_primary_errors`` consecutive primary failures
    the fallback becomes permanent for the rest of the mission.
    """

    def __init__(self, primary, fallback, max_primary_errors: int = 2):
        self.primary = primary
        self.fallback = fallback
        self._primary_errors = 0
        self._max_primary_errors = max_primary_errors

    @property
    def name(self) -> str:
        if self.fallback is not None and self._degraded():
            return f"{self.fallback.name} (degraded from {self.primary.name})"
        return self.primary.name

    def _degraded(self) -> bool:
        return self._primary_errors >= self._max_primary_errors

    def complete(self, system_prompt: str, user_prompt: str, **kw) -> str:
        if self.fallback is None or not self._degraded():
            try:
                out = self.primary.complete(system_prompt, user_prompt, **kw)
                self._primary_errors = 0
                return out
            except Exception as e:  # noqa: BLE001
                self._primary_errors += 1
                log.warning(
                    "Agent LLM primary backend %s failed (%d/%d): %s",
                    self.primary.name, self._primary_errors,
                    self._max_primary_errors, e,
                )
                if self.fallback is None:
                    raise
        return self.fallback.complete(system_prompt, user_prompt, **kw)


def _claude_sdk_available() -> bool:
    try:
        import claude_agent_sdk  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def make_agent_llm(complete_fn=None):
    """Build the agent LLM per ``AGENTIC_LLM_BACKEND`` (auto|claude|local)."""
    choice = os.environ.get("AGENTIC_LLM_BACKEND", "auto").strip().lower()
    local = LocalLLMBackend(complete_fn=complete_fn)
    if choice == "local":
        return local

    if choice in ("auto", "claude"):
        if _claude_sdk_available():
            try:
                import app as _app
                default_base = _app.LLM_BASE_URL_LITELLM
                default_model = _app.MODEL_NAME
            except Exception:  # noqa: BLE001
                default_base = "http://127.0.0.1:4000"
                default_model = "local-model"
            base_url = (
                os.environ.get("AGENTIC_ANTHROPIC_BASE_URL")
                or os.environ.get("ANTHROPIC_BASE_URL")
                or default_base
            )
            model = os.environ.get("AGENTIC_CLAUDE_MODEL", default_model)
            auth = os.environ.get("ANTHROPIC_AUTH_TOKEN", "ollama")
            claude = ClaudeSDKBackend(
                model=model, base_url=base_url, auth_token=auth,
            )
            log.info(
                "Agentic LLM: claude-agent-sdk -> %s (model=%s), "
                "local fallback armed", base_url, model,
            )
            return ResilientLLM(claude, local)
        if choice == "claude":
            log.warning(
                "AGENTIC_LLM_BACKEND=claude but claude-agent-sdk is not "
                "installed — falling back to the local llama-server backend."
            )
    return local
