---
title: API Surface
description: FastAPI routes, SSE behaviour, and sandbox_retries
tags: [api, rest, sse]
---

# API Surface

Primary implementation: `api/app.py`.

## Session & uploads

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/session/create` | Create session; returns `{session_id}` |
| `POST` | `/api/upload/code/{session_id}` | Multipart `.c` / `.h` files |
| `POST` | `/api/upload/source-icd/{session_id}` | Source ICD PDF → text |
| `POST` | `/api/upload/target-icd/{session_id}` | Target ICD PDF → text |
| `POST` | `/api/upload/repo-zip/{session_id}` | Optional surrounding repo ZIP |
| `GET` | `/api/status/{session_id}` | Session status / progress |
| `DELETE` | `/api/session/{session_id}` | Tear down session |

## Pipeline control

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/process/{session_id}` | **SSE** full pipeline (agentic mission when `AGENTIC_PIPELINE=1`) |
| `GET` | `/api/process-agentic/{session_id}` | **SSE** always agentic mission |
| `GET` | `/api/regenerate/{session_id}` | **SSE** conversational regen |
| `POST` | `/api/pause/{session_id}` | Pause processing |
| `POST` | `/api/resume/{session_id}` | Resume processing |

### Query: `sandbox_retries`

Accepted on both `/api/process/{session_id}` and `/api/regenerate/{session_id}`.

| Value | Behaviour |
|-------|-----------|
| Positive integer `N` | Finite budget: one outer round, `N` builds/attempts |
| `indefinite`, `inf`, `infinite`, `unlimited`, `0`, `-1` | Unlimited until success/cancel |
| Omitted | Env defaults (`SANDBOX_ORCH_*` / `SANDBOX_AGENTIC_*`) |

Parsed by `_parse_sandbox_retries` / `_resolve_sandbox_max_retries` / `_sandbox_retry_plan`. Persisted on the session as:

- `sandbox_retries_mode` ∈ `{env, indefinite, finite}`
- `sandbox_max_retries`

Finite budgets hard-stop the orchestrator when exhausted (no endless patch loop after the last failed build).

## Results & conversation

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/download/{session_id}` | Generated result ZIP |
| `GET` | `/api/download-repo/{session_id}` | Built sandbox repo ZIP |
| `GET` | `/api/preview/{session_id}/{filename}` | Preview a generated file |
| `POST` | `/api/conversation/{session_id}` | Append feedback (`ChatMessage`) |
| `GET` | `/api/conversation/{session_id}` | Read conversation history |

## Static UI

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Serves `api/static/index.html` |
| `GET` | `/static/app.js` | Explicit `app.js` route (cache/patch friendly) |

## SSE event shapes (typical)

Events emitted during process/regenerate include (non-exhaustive):

- `stage` / `stage_complete` — stage transitions
- `token` / batched tokens — streamed LLM output for UI
- `file_complete` — per-file transform progress
- `sandbox_build_result` / build log chunks — sandbox outcome
- Orchestrator events: `step`, `thought`, `action`, `observation`, `build`, `done`, `warning`
- Agentic events: phase transitions, attempt artefacts under `agentic_attempts/`

UI token batching is controlled by `SSE_UI_TOKEN_BATCH_*` to avoid browser OOM on long runs. Sandbox log streaming is capped (`SANDBOX_SSE_MAX_BUILD_LOG_CHARS`).
