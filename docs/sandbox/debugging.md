---
title: Sandbox Debugging
description: Orchestrator vs agentic debug, retry budgets, remote build
tags: [sandbox, orchestrator, agentic]
---

# Sandbox Debugging

After verification and the per-file compile gate, the pipeline tries to produce a clean build inside a session `sandbox/` (or on a remote host).

## Backend selection (`_sandbox_build_iterate`)

Order of precedence in `api/app.py`:

1. If `REMOTE_BUILD_ENABLED=1` → `_remote_build_iterate` (SSH/remote script path).
2. Else if `SANDBOX_USE_AGENTIC=1` (**default**) → `run_agentic_debug` (**wins over orchestrator**).
3. Else if `SANDBOX_USE_ORCHESTRATOR=1` → `run_orchestrator`.
4. Else → legacy per-file rewrite loop in `app.py`.

Set `SANDBOX_USE_AGENTIC=0` to fall back to the ReAct orchestrator.

## Orchestrator (`api/orchestrator.py`)

`run_orchestrator` is a ReAct tool loop. Tools include:

`read_file`, `list_dir`, `search`, `find_files`, `patch`, `write_file`, `reset_file`, `build`, `note`, `done`.

Caps:

| Knob | Meaning |
|------|---------|
| `max_steps` | Thought/action iterations (`SANDBOX_ORCH_MAX_STEPS`; `0` = unlimited steps) |
| `max_builds` | Build/attempt budget from UI or `SANDBOX_ORCH_MAX_BUILDS` |
| Outer rounds | `SANDBOX_ORCH_OUTER_ROUNDS` when using env mode |

When a **finite** UI budget is set, exhausting it **hard-stops** the orchestrator (further `build` calls are blocked and the loop ends) so patching cannot continue indefinitely after the Nth failed build.

Yielded event kinds: `step`, `thought`, `action`, `observation`, `build`, `done`, `raw_token`, `warning`.

## Agentic debug (`api/agentic_debug.py`)

`run_agentic_debug` runs a phased state machine:

`build` → `triage` → `root_cause` → `plan` → `patch` → `superloop_validation` → `verify` → `decide` → `done`.

Notable pieces: `CodebaseIndex`, build-log parsing / error clustering, playbook hypotheses, lane splits, Xilinx/BSP audits, superloop checks.

Attempt artefacts land under:

```text
<session>/agentic_attempts/attempt_NN/
  patch.diff, build logs, hypothesis.md, metrics.json
```

Env knobs: `SANDBOX_AGENTIC_MAX_ATTEMPTS`, `SANDBOX_AGENTIC_NO_PROGRESS`, `SANDBOX_AGENTIC_OSCILLATION`, `SANDBOX_AGENTIC_EDIT_BUDGET`, `SANDBOX_AGENTIC_OUTER_ROUNDS`.

Deep dive: [agentic-debug-pipeline.md](../agentic-debug-pipeline.md).

## UI retry budget maths

| Mode | Outer rounds | Builds / attempts |
|------|--------------|-------------------|
| Finite `N` | 1 | `N` |
| Indefinite | Unlimited (with backend safety caps) | Unlimited |
| Env (query omitted) | From `SANDBOX_*_OUTER_ROUNDS` | From `SANDBOX_ORCH_MAX_BUILDS` / agentic max attempts |

## Remote build

When enabled, local docker sandbox is skipped in favour of remote host settings:

`REMOTE_BUILD_ENABLED`, `REMOTE_BUILD_HOST`, `REMOTE_BUILD_USER`, `REMOTE_BUILD_PASS`, `REMOTE_BUILD_SRC`, `REMOTE_BUILD_SCRIPT`, `REMOTE_BUILD_MODE`.
