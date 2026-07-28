---
title: Sandbox Debugging
description: Orchestrator vs agentic debug, env budgets, remote build
tags: [sandbox, orchestrator, agentic]
---

# Sandbox Debugging

After verification and the per-file compile gate, IntegrationTeam (agentic path) or classic `process()` calls `_sandbox_build_iterate` to produce a clean build in session `sandbox/` (or on a remote host).

This is separate from the **mission** orchestrator (`MissionController` in `api/agentic_pipeline/`). See [agentic/multiagent-pipeline.md](../agentic/multiagent-pipeline.md).

## Backend selection (`_sandbox_build_iterate`)

1. If `REMOTE_BUILD_ENABLED=1` → `_remote_build_iterate`.
2. Else if `SANDBOX_USE_AGENTIC=1` → `run_agentic_debug` (**wins over orchestrator**).
3. Else if `SANDBOX_USE_ORCHESTRATOR=1` (default) → `run_orchestrator`.
4. Else → legacy per-file rewrite loop in `app.py`.

## Orchestrator (`api/orchestrator.py`)

ReAct tool loop. Tools: `read_file`, `list_dir`, `search`, `find_files`, `patch`, `write_file`, `reset_file`, `build`, `note`, `done`.

| Knob | Env / meaning |
|------|----------------|
| `max_steps` | `SANDBOX_ORCH_MAX_STEPS` (`0` = unlimited steps) |
| `max_builds` | `SANDBOX_ORCH_MAX_BUILDS` |
| Outer rounds | `SANDBOX_ORCH_OUTER_ROUNDS` |

Yielded events: `step`, `thought`, `action`, `observation`, `build`, `done`, `raw_token`, `warning`.

## Agentic debug (`api/agentic_debug.py`)

Phased loop: `build` → `triage` → `root_cause` → `plan` → `patch` → `superloop_validation` → `verify` → `decide` → `done`.

Artefacts: `<session>/agentic_attempts/attempt_NN/` (`patch.diff`, logs, `hypothesis.md`, `metrics.json`).

Env: `SANDBOX_AGENTIC_MAX_ATTEMPTS`, `SANDBOX_AGENTIC_NO_PROGRESS`, `SANDBOX_AGENTIC_OSCILLATION`, `SANDBOX_AGENTIC_EDIT_BUDGET`, `SANDBOX_AGENTIC_OUTER_ROUNDS`.

Deep dive: [agentic-debug-pipeline.md](../agentic-debug-pipeline.md).

## UI retry budget maths

| Mode | Outer rounds | Builds / attempts |
|------|--------------|-------------------|
| Finite `N` | 1 | `N` |
| Indefinite | Unlimited (with backend safety caps) | Unlimited |
| Env (query omitted) | From `SANDBOX_*_OUTER_ROUNDS` | From `SANDBOX_ORCH_MAX_BUILDS` / agentic max attempts |

Mission-level revisits still use `AGENTIC_MAX_STAGE_RUNS` / `AGENTIC_MAX_REVISITS_PER_STAGE`.

## Remote build

`REMOTE_BUILD_ENABLED`, `REMOTE_BUILD_HOST`, `REMOTE_BUILD_USER`, `REMOTE_BUILD_PASS`, `REMOTE_BUILD_SRC`, `REMOTE_BUILD_SCRIPT`, `REMOTE_BUILD_MODE`.
