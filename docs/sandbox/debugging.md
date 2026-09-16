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

## HEX / VirtuosoNext RTOS SDK injection

`_sandbox_build_iterate` in `api/app.py` calls `hex_sdk.discover(...)` at the top of each run. When a `VisualDesigner-HEX-*` tree is located (via `HEX_SDK_DIR`, sibling of the repo, or a common install location — see [Configuration → HEX / VirtuosoNext RTOS SDK](../building/configuration.md#hex--virtuosonext-rtos-sdk)) three things happen:

1. **Cross-compiler upgrade** — if the SDK targets an arch that needs a cross-compiler (e.g. `arm-cortex-a9` → `arm-none-eabi-gcc`) and that toolchain is on `$PATH`, `sandbox_cc` is upgraded from `gcc` to the cross-compiler even when the user's `Makefile` doesn't declare it. `_detect_cross_compiler` also parses HEX `environment.mk` / `PROJECT_GEN` values as a fallback.
2. **Compile-time flag injection** — `_run_sandbox_build` prepends the SDK's `-I<targets/<platform>/include>` and `-D<PLATFORM>` / `-DVIRTUOSO_NEXT` / `-DVN_*` defines to `CFLAGS` (make) or `CMAKE_C_FLAGS` (cmake). The compile-gate (`api/per_file_compile.py`) adds the same flags via its `extra_flags` argument, so a `#include <L1_api.h>` in a generated file resolves in step 3.5 just as it does in the full sandbox build.
3. **Link-time flag injection** — the shortlisted static archives (`libHEX_<VARIANT>_CO<opt>[_D<dbg>][_PL<prot>]_Kernel.a` + companions) are turned into `-L<targets/<platform>/lib> -Wl,--start-group -l... -Wl,--end-group` and appended to `LDFLAGS` / `CMAKE_EXE_LINKER_FLAGS`. If a `.a` for the selected build key is missing, it's silently dropped from the link line — the compile-only fallback (`-c`) still passes.

The LLM sees a compact summary of the discovered SDK (version, platform, defines, a snippet of `L1_api.h`) via the `hex_sdk` section in the transform + fix prompts (priority `1`, higher than repo dependencies) so it will not invent kernel APIs. Set `HEX_SDK_DISABLE=1` to force a pure-native build when the sandbox does not have the SDK installed.
