---
title: Agentic Multi-Agent Pipeline
description: MissionController, six teams, blackboard, Claude Agent SDK → local LLM
tags: [agentic, multiagent, mission]
---

# Agentic Multi-Agent Pipeline

On this branch the default `/api/process` path is a **multi-agent mission**: six teams coordinated by an agent-of-agents `MissionController`, sharing a blackboard, with optional revisits to earlier stages based on feedback.

This is **not** the same as the sandbox hypothesis loop in `api/agentic_debug.py` (documented in [agentic-debug-pipeline.md](../agentic-debug-pipeline.md)). That debug loop can still run *inside* IntegrationTeam when `SANDBOX_USE_AGENTIC=1`.

## Default-on

| Layer | Behaviour |
|-------|-----------|
| `api/app.py` | `AGENTIC_PIPELINE` defaults to **on** (`1`); unset/`true` → agentic |
| Docker | `ENV AGENTIC_PIPELINE=1` |
| Serve scripts | Forward `AGENTIC_PIPELINE="${AGENTIC_PIPELINE:-1}"` |
| Opt-out | `AGENTIC_PIPELINE=0` → classic sequential generator in `process()` |
| Always-on route | `GET /api/process-agentic/{session_id}` → agentic regardless of flag |

Entry: `_agentic_streaming_response()` → `run_agentic_pipeline()` → `MissionController.run()`.

## Package layout

```text
api/agentic_pipeline/
├── mission.py       # MissionController, STAGE_ORDER, run_agentic_pipeline
├── agents.py        # Agent dataclass + act()
├── blackboard.py    # Blackboard, Feedback, StageRecord
├── context.py       # PipelineContext (paths, llm, app handle)
├── llm.py           # ClaudeSDKBackend, LocalLLMBackend, ResilientLLM, make_agent_llm
└── teams/
    ├── ingestion.py      # analysis
    ├── codebase.py       # gitnexus
    ├── generation.py     # transform
    ├── verification.py   # verification
    ├── compilation.py    # compile
    └── integration.py    # sandbox_build
```

## Mission loop

1. Pick next pending stage from `STAGE_ORDER`.
2. Run the mapped team (`team.run`).
3. After each run, ensure a consolidated team findings report exists (`Team.ensure_consolidated_report` fallback if the team did not emit one).
4. Router decides `proceed` vs `revisit` (blockers force revisit; LLM router optional via `AGENTIC_ROUTER_LLM`).
5. On revisit, `mark_stale_after` so downstream stages re-run.
6. Bounds: `AGENTIC_MAX_STAGE_RUNS` (default 12), `AGENTIC_MAX_REVISITS_PER_STAGE` (default 2).
7. `_finalise()` writes `status.json` and emits classic SSE `{type: "complete", ...}`.

SSE stage names stay compatible with the UI: `analysis`, `gitnexus`, `transform`, `verification`, `compile`, `sandbox_build`.

## Consolidated team reports

Every team writes **one** Markdown findings report at the end of its run via `Team.emit_consolidated_report` in `teams/base.py`:

| Location | Name pattern | Purpose |
|----------|--------------|---------|
| `generated_code/` | `team_report_{stage}.md` | Live preview + Download All (same path as other deliverables) |
| `agentic_pipeline/team_reports/` | `{stage}_team_report.md` | Durable archive copy |

Reports include status, summary, findings, artefacts, feedback raised/pending, and the agent roster. Emitting a report also sends SSE `deliverables_updated` (same channel as mid-sandbox sync) so the UI/download list stays current without waiting for mission complete.

`GET /api/download/{session_id}` packs both the live `team_report_*.md` files from `generated_code/` and the `agentic_pipeline/team_reports/` archive folder into the ZIP.

## Teams & agents

| SSE stage | Team | Agents (roles) | Core work |
|-----------|------|----------------|-----------|
| `analysis` | `IngestionTeam` | TargetSummarizer, DocumentAnalyst, SpecSynthesizer, Distiller, FactAuditor | ICD delta → `change_spec_raw` → distilled `change_spec` (≥80% fact retention) |
| `gitnexus` | `CodebaseTeam` | RepoCartographer, SystemsArchaeologist, ImpactAssessor | Repo inventory + GitNexus extractors + impact map |
| `transform` | `GenerationTeam` | VariantScout, InterfaceArchitect, ImplementationEngineer, CompletionCritic | Headers (incl. per-variant) then `.c`; completeness gates |
| `verification` | `VerificationTeam` | StructuralAuditor, ComplianceReviewer, RepairEngineer | Structural + ICD compliance; repair or feedback upstream |
| `compile` | `CompilationTeam` | ToolchainScout, BuildOperator, GateKeeper | Per-file compile gate; blocker can send work back to `transform` |
| `sandbox_build` | `IntegrationTeam` | SandboxEngineer, DebugCrew, IntegrationJudge | Sandbox/remote build + optional agentic debug; route failures |

Teams reuse proven helpers via `ctx.app` (`_structural_verify`, `run_per_file_compile`, `_sandbox_build_iterate`, etc.).

## LLM backends (`llm.py`)

| Class | Role |
|-------|------|
| `ClaudeSDKBackend` | `claude_agent_sdk.query` → Anthropic-compatible endpoint (LiteLLM `:4000`) → same GGUF |
| `LocalLLMBackend` | Direct `app._call_llm_complete` (llama-server) |
| `ResilientLLM` | Primary + fallback; sticks to fallback after consecutive primary failures |
| `make_agent_llm()` | `AGENTIC_LLM_BACKEND=auto\|claude\|local` |

**Default (`auto` + SDK installed):** Claude Agent SDK → LiteLLM → GGUF; degrade to local llama-server if SDK/LiteLLM fails. Missing SDK install falls back to local automatically.

## Artefacts

Classic artefacts plus mission audit:

```text
session_dir/agentic_pipeline/blackboard.json   # stages, feedback, history
session_dir/agentic_pipeline/team_reports/     # {stage}_team_report.md archive
session_dir/change_spec.txt / change_spec_raw.txt
session_dir/repo_knowledge.txt / gitnexus_report.txt
generated_code/…  team_report_{stage}.md  verification_report.txt  compile_report.txt
built_repo.zip  sandbox_build_log.txt  agentic_attempts/ (if debug crew on)
```

## Configuration (mission-specific)

| Variable | Default | Meaning |
|----------|---------|---------|
| `AGENTIC_PIPELINE` | `1` | Route `/api/process` through mission |
| `AGENTIC_MAX_STAGE_RUNS` | `12` | Cap total team runs |
| `AGENTIC_MAX_REVISITS_PER_STAGE` | `2` | Extra runs per stage |
| `AGENTIC_ROUTER_LLM` | `1` | LLM router when multiple revisit targets |
| `AGENTIC_LLM_BACKEND` | `auto` | `auto` / `claude` / `local` |
| `AGENTIC_ANTHROPIC_BASE_URL` / `ANTHROPIC_BASE_URL` | LiteLLM `:4000` | Anthropic-compatible base |
| `AGENTIC_CLAUDE_MODEL` / `ANTHROPIC_MODEL` | app model / `openai/$HF_MODEL` | Model id |

See also [building/configuration.md](../building/configuration.md).

## Docker notes

- `requirements.txt` / image install `claude-agent-sdk` (+ optional `@anthropic-ai/claude-code` CLI).
- Entrypoint still starts llama-server + LiteLLM + uvicorn; LiteLLM is the Anthropic front for the SDK.
- Rebuild after pulling this branch: `./deployments/docker/build.sh && ./scripts/serve_web.sh`.

## Testing

- `tests/development/test_agentic_pipeline.py` — default-on flag, opt-out, FakeLLM mission coverage.
- `tests/development/test_team_consolidated_reports.py` — per-team report write, SSE `deliverables_updated`, Download ZIP packaging.

## Relationship diagram

```text
UI Transform → /api/process
                 ├─ AGENTIC_PIPELINE=1 → MissionController → six teams
                 └─ AGENTIC_PIPELINE=0 → classic sequential process()

IntegrationTeam.sandbox_build
                 └─ may call _sandbox_build_iterate
                        ├─ SANDBOX_USE_AGENTIC=1 → agentic_debug.py
                        └─ else orchestrator / legacy
```
