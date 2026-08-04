---
type: "Reference"
title: "Agentic Debug Pipeline"
openwiki_generated: true
---

# Agentic Debug Pipeline

A plain-English walkthrough of the build-debug-repair loop introduced on
the `feature/agentic-debug-loop` branch.

This document covers **only** the debug / convergence layer of the
sandbox build stage. ICD ingestion, diffing, first-pass code generation,
verification, pause/resume, downloads, Docker, and the UI are all
unchanged.

---

## The big picture

Before this change, the sandbox debug loop worked like this: the LLM saw
"build failed, here are some errors" and was asked to rewrite a whole
file. If that didn't work, it tried again, and again — often making
things worse, because it was patching already-broken code with no
memory of what it had tried.

The new architecture treats **debugging as a search problem with strict
rules**, the same way an experienced engineer would do it: read the
error, find the smallest root cause, propose ONE small fix, try it, see
if it helped, decide what to do next. It splits this into seven distinct
phases that run in a loop until the build is green.

Implementation lives in [`api/agentic_debug.py`](../api/agentic_debug.py)
and is wired into the existing sandbox entry point (`_sandbox_build_iterate`
in `api/app.py`) behind the `SANDBOX_USE_AGENTIC` environment flag.

---

## The seven phases (one "attempt" = one full lap)

```
   ┌───────────────────────────────────────────────────────────────────┐
   │                                                                   │
   ▼                                                                   │
[BUILD] → [TRIAGE] → [ROOT_CAUSE] → [PLAN] → [PATCH] → [VERIFY] → [DECIDE]
                                                                       │
                                                              success ─┴─ DONE
```

### 1. BUILD — "What's broken?"

Runs the actual compiler (`make`, `cmake`, or the cross-compile shell
script for Xilinx). Captures the entire raw output to disk so we never
lose information. If the build is already green on the very first call,
we exit immediately with a `done` event — no LLM cost, no work to do.

### 2. TRIAGE — "Turn raw text into structured data"

The compiler dumps thousands of lines of unstructured text. The triage
step parses every diagnostic into a typed record:

```json
{
  "tool": "compile",
  "severity": "error",
  "file": "src/comms.c",
  "line": 42,
  "column": 5,
  "symbol": "PeripheralX_Config",
  "message": "unknown type name 'PeripheralX_Config'",
  "error_class": "unknown_type"
}
```

This conversion is critical — the rest of the pipeline reasons about
*errors as objects*, not strings. The triage step also:

- Drops warnings (they don't drive fix planning).
- Recognises both modern (`file:line:col:`) and old-style (`file:line:`)
  GCC formats so it works on Xilinx GCC 7.3.1.
- Catches both `undefined reference to \`...\`` and the `obj.o: in
  function \`...\`` forms produced by the linker.

### 3. ROOT_CAUSE — "Find the actual problem"

This is where the magic happens. Most build loops fail because they
chase **downstream noise** — one missing typedef can produce 50 errors,
but they're all really just one bug.

The root-cause stage:

- **Groups** errors by the same symbol (e.g. all errors mentioning
  `PeripheralX_Config` collapse into one cluster).
- **Ranks** clusters by *centrality*: how many files reference that
  symbol (so a missing type used in 30 files outranks a single bad cast
  in one file). Errors that appear early in the build log get a small
  bonus, because GCC tends to abort right after the first fatal error
  per file — so early errors are usually causes, not effects.
- **Suggests** a fix strategy per error class. For example:
  - `unknown_type` → "add a typedef bridge in a header"
  - `signature_mismatch` → "add an adapter function with the OLD signature"
  - `link_undefined` for BSP symbols (`Xil_*`, `xil_printf`) → "ignore — compile-only fallback handles this"

The output is a ranked list of clusters, each with a clear root-cause
label and a suggestion. **One** of these is what the planner will tackle
next.

The nine error classes the classifier recognises map directly to the
policy table in the design brief (section D):

| Class                   | Typical fix                                |
| ----------------------- | ------------------------------------------ |
| `missing_include`       | Restore an alias header / `#include`       |
| `unknown_type`          | Typedef bridge or forward declaration      |
| `macro_mismatch`        | Re-define / wrapper macro                  |
| `signature_mismatch`    | Adapter function with old signature        |
| `link_undefined`        | Stub or wrapper (BSP symbols ignored)      |
| `qualifier_mismatch`    | const/volatile/packed in declarations      |
| `enum_value_drift`      | Re-add enum value or `#define` alias       |
| `struct_layout`         | Restore field, validate with `static_assert` |
| `calling_convention`    | Match `__attribute__` to new ICD           |

### 4. PLAN — "Pick the smallest, safest fix"

A specialised LLM prompt (the **Fix Strategy Agent**) gets:

- The ranked cluster list,
- Recent metrics (how many errors before/after the last few attempts),
- Which "correction layer" we're in (more on this below),
- Hints from past successful fixes on this session (the **playbook**).

It must output **exactly ONE** hypothesis as JSON, with this shape:

```json
{
  "id": "H1",
  "layer": "shim",
  "kind": "header_typedef_bridge",
  "title": "Bridge PeripheralX_Config to PeripheralX2_Config",
  "rationale": "Old ICD used PeripheralX_Config; new ICD renames it ...",
  "target_files": ["include/comms_compat.h"],
  "risk": "low",
  "expected_resolved": ["unknown_type: symbol 'PeripheralX_Config'"],
  "notes": ""
}
```

The "one hypothesis per turn" rule is the **single most important
guardrail**. It means every attempt produces a clean trace of "edit X
fixed error family Y" — which we can verify, roll back, and learn from.
No more "rewrite the whole file and pray".

The **three correction layers** force a sensible order of operations:

| Layer       | Goal                                                         |
| ----------- | ------------------------------------------------------------ |
| `shim`      | Make it compile/link with compatibility adapters             |
| `semantic`  | Replace shims with proper ICD mappings                       |
| `cleanup`   | Remove dead aliases, tighten types, enforce warnings-as-errors |

We never advance to the next layer until the current one is exhausted.
This stops the LLM from trying to do "make it correct" and "make it
elegant" simultaneously, which is when it usually fails.

The planner also has a built-in `"kind": "stop"` escape hatch so it can
explicitly decline to make further changes when none would help — this
is treated as a clean exit, not a failure.

### 5. PATCH — "Write the actual edit"

A second LLM call (the **Patch Agent**) is given:

- The approved hypothesis,
- The *current* contents of the target files,
- The active correction layer,
- The top error clusters,
- The ICD change spec.

It must produce the new full content for ONLY the files listed in
`target_files`, in a strict per-file fenced format:

````
### include/comms_compat.h
```c
#ifndef COMMS_COMPAT_H
#define COMMS_COMPAT_H
typedef PeripheralX2_Config PeripheralX_Config;
#endif
```
````

The applier then:

- **Whitelists** writes — if the LLM tries to write to a file not in
  `target_files`, it's rejected.
- **Blocks path traversal** — `../../etc/passwd` style escapes are
  rejected with a `ValueError`-equivalent rejection record.
- **Snapshots** every file before overwriting (so we can roll back
  instantly).
- Writes a `patch.diff` summary to disk for traceability.

### 6. VERIFY — "Did it actually help?"

We rebuild. Then:

- Parse the new build output the same way we parsed the original.
- Compute **error fingerprint** — a hash of the error set with line
  numbers normalised. This means the same logical errors look identical
  across attempts even if the patch shifted lines around.
- Compare counts: how many errors were resolved, how many new ones
  were introduced?

The build call itself reuses the existing `_run_sandbox_build` runner,
so cross-compilation, compile-only fallback for BSP linker errors, and
build-system detection all behave exactly as before.

### 7. DECIDE — "What just happened?"

This is the convergence brain. Four possible verdicts:

| Outcome                                         | Action                                                                |
| ----------------------------------------------- | --------------------------------------------------------------------- |
| Build is green                                  | **DONE** — record success in playbook, package the repo               |
| Patch reduced errors                            | Keep the patch, continue to next attempt                              |
| Patch introduced new errors with no resolutions | **Roll back** instantly — the snapshot makes this free                |
| Same fingerprint as before (oscillating)        | **Escalate**: advance to next correction layer, or stop if at deepest |

Three hard stop conditions prevent infinite loops:

- **No-progress limit** (default `3`) — three attempts in a row with no
  net error reduction.
- **Oscillation limit** (default `2`) — same error fingerprint reappears.
- **Edit budget** (default `60`) — too many distinct files have been
  touched.

When all rounds fail, we still package a "best-effort" ZIP plus all
artifacts — **the user always gets something downloadable**.

---

## What you can inspect after a run

Every attempt drops six files into `agentic_attempts/attempt_NN/`:

| File                            | What it tells you                                         |
| ------------------------------- | --------------------------------------------------------- |
| `build.log.raw`                 | The exact compiler output                                 |
| `build.log.structured.json`     | The same errors as parsed, typed records                  |
| `error_clusters.json`           | Grouped root causes with centrality scores                |
| `hypothesis.md`                 | The plan: what was changed, why, expected impact          |
| `patch.diff`                    | Diff-style summary of what the applier wrote              |
| `metrics.json`                  | `errors_total`, `new_errors`, `resolved_errors`, `files_touched`, `duration_s`, `success` |

Plus a session-wide `playbook.jsonl` that records
`(error_class, symbol) → patch_kind` for every successful fix, so
subsequent attempts (and future runs on similar code) get a head start.
All of this is bundled into the download ZIP, so you can audit *exactly*
why the loop did what it did.

---

## Why this should converge faster on real inputs

The old loop's failure modes, and how this fixes them:

| Old failure                                                       | Fix in new loop                                                            |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------- |
| LLM rewrote whole files → introduced new bugs each iteration      | Patch Agent only writes files explicitly listed in the hypothesis; applier whitelists + snapshots |
| 50 errors from one missing typedef were treated as 50 unrelated bugs | Cascade clustering collapses them into one cluster ranked by centrality |
| LLM had no memory of what it tried                                | Playbook + metrics history are folded into every planner prompt           |
| Oscillation went undetected (line numbers shifted, errors looked different) | Fingerprint normalises line numbers, so true oscillation is caught |
| Bad patches lingered, polluting subsequent attempts               | Pure regressions are rolled back from snapshots — zero cost                |
| LLM mixed "make it work" + "make it pretty"                       | Three-layer correction strategy forbids this                              |
| No way to tell why a fix worked or failed                         | Per-attempt artifacts make every decision auditable                       |

---

## Component map

| Brief component (from design)        | Implementation                                                          |
| ------------------------------------ | ----------------------------------------------------------------------- |
| **Orchestrator / State Machine**     | `run_agentic_debug` driving `Phase.BUILD → TRIAGE → ROOT_CAUSE → PLAN → PATCH → VERIFY → DECIDE` with budgets and one-hypothesis-per-iteration |
| **Codebase Context Service**         | `CodebaseIndex` — basename + symbol → files index (`#define`, `typedef`, `struct`, function defs); `references_to(symbol)` for impact analysis |
| **Build Harness**                    | Wraps existing `_run_sandbox_build`; captures raw + structured (`BuildError`) logs |
| **Debug Intelligence Layer**         | `parse_build_log` → typed tuples; `classify_error` covers all 9 error classes; `cluster_errors` does cascade dedup with centrality ranking |
| **Patch Planner + Applier**          | `_planner_prompt` + `_PLANNER_SYSTEM_PROMPT` produce ONE typed `Hypothesis` per turn; `_patcher_prompt` produces multi-file content; `apply_hypothesis` is path-traversal-safe, whitelisted to `target_files`, with snapshots + `rollback` on regression |
| **Verification Suite**               | `run_pre_build_checks` (include guard / `#pragma once`); regression check via fingerprint comparison; rollback when an edit introduces only new errors |
| **Memory & Learning Store**          | `Playbook` JSONL keyed by `(error_class, symbol)` → `patch_kind`; future iterations get hints from prior successful fixes |
| **Three-layer correction**           | Each hypothesis is tagged `shim → semantic → cleanup`; layer auto-advances on oscillation |
| **Per-attempt artifacts**            | Under `agentic_attempts/attempt_NN/`: `patch.diff`, `build.log.raw`, `build.log.structured.json`, `error_clusters.json`, `hypothesis.md`, `metrics.json` |
| **Convergence**                      | Stop on success, no-net-improvement-for-N, oscillation, edit budget; `errors_fingerprint` is line-number-invariant |

---

## How to run it

The agentic debug pipeline is **on by default** for the sandbox stage
(`SANDBOX_USE_AGENTIC=1` in `api/app.py` and the Docker image). It takes
precedence over the ReAct orchestrator. To restore the orchestrator:

```bash
SANDBOX_USE_AGENTIC=0 ./scripts/serve_web.sh
```

Tunables (all optional, defaults shown):

| Variable                            | Default | Meaning                                                          |
| ----------------------------------- | ------- | ---------------------------------------------------------------- |
| `SANDBOX_USE_AGENTIC`               | `1`     | Master switch — when on, takes precedence over the orchestrator  |
| `SANDBOX_AGENTIC_MAX_ATTEMPTS`      | `120`   | Max hypothesis attempts per round                                |
| `SANDBOX_AGENTIC_NO_PROGRESS`       | `3`     | Stop after N iterations without error reduction                  |
| `SANDBOX_AGENTIC_OSCILLATION`       | `2`     | Escalate layer when fingerprint repeats                          |
| `SANDBOX_AGENTIC_EDIT_BUDGET`       | `60`    | Max distinct files modified across the run                       |
| `SANDBOX_AGENTIC_OUTER_ROUNDS`      | `2`     | Full-reset rounds before giving up                               |

---

## Verification

The module ships with focused unit + smoke tests covering:

1. `parse_build_log` on a representative GCC log (modern, no-column, and
   linker forms).
2. `classify_error` policy mapping for all nine error classes.
3. `cluster_errors` cascade dedup + centrality ranking — the same
   symbol in two errors must produce a single cluster.
4. `errors_fingerprint` stability across line-number shifts.
5. `parse_hypothesis` tolerance (fenced JSON + bare-JSON fallback).
6. `parse_patch` per-file routing.
7. `apply_hypothesis` whitelist enforcement and path-traversal guard.
8. End-to-end `run_agentic_debug` on a synthetic red→green session
   with a fake LLM stream and a fake build runner — verifies that the
   state machine produces all six per-attempt artifacts and writes a
   playbook entry on success.

All eight tests pass cleanly. With `SANDBOX_USE_AGENTIC=0` the sandbox
stage falls back to the ReAct orchestrator; the rest of the pipeline is
unchanged.

---

## Glossary

| Term                  | Plain-English meaning                                              |
| --------------------- | ------------------------------------------------------------------ |
| **Hypothesis**        | One proposed fix. Has a layer, kind, target files, and risk score. |
| **Cluster**           | A group of errors that almost certainly share one root cause.      |
| **Fingerprint**       | A line-number-invariant signature of the current error set.        |
| **Centrality**        | Rough measure of how many files reference a symbol — used for ranking. |
| **Layer**             | The current correction strategy: `shim`, `semantic`, or `cleanup`. |
| **Playbook**          | An append-only log of `(error_class, symbol) → fix kind` outcomes. |
| **Snapshot**          | Pre-edit copy of every file the applier touches; enables free rollback. |
| **Outer round**       | A full restart from the initial snapshot when an inner loop stalls. |
