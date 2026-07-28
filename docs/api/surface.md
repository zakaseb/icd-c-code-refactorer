---
title: API Surface
description: FastAPI routes, SSE behaviour, and sandbox_retries
tags: [api, rest, sse]
---

# API Surface

Primary implementation: `api/app.py`. On this branch, `/api/process` defaults to the agentic mission (`AGENTIC_PIPELINE=1`).

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
| `GET` | `/api/process/{session_id}` | **SSE** — agentic mission when `AGENTIC_PIPELINE=1` (default); classic sequential when `0` |
| `GET` | `/api/process-agentic/{session_id}` | **SSE** — always agentic mission |
| `GET` | `/api/regenerate/{session_id}` | **SSE** conversational regen |
| `POST` | `/api/pause/{session_id}` | Pause processing |
| `POST` | `/api/resume/{session_id}` | Resume processing |

### Query: `sandbox_retries`

Accepted on `/api/process`, `/api/process-agentic`, and `/api/regenerate`.

| Value | Behaviour |
|-------|-----------|
| Positive integer `N` | Finite budget: one outer round, `N` builds/attempts |
| `indefinite`, `inf`, `infinite`, `unlimited`, `0`, `-1` | Unlimited until success/cancel |
| Omitted | Env defaults (`SANDBOX_ORCH_*` / `SANDBOX_AGENTIC_*`) |

Persisted on the session as `sandbox_retries_mode` ∈ `{env, indefinite, finite}` and `sandbox_max_retries`. Mission revisits remain bounded by `AGENTIC_MAX_STAGE_RUNS` / `AGENTIC_MAX_REVISITS_PER_STAGE`.

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

- `stage` / `stage_complete` — including live **`gitnexus`** from CodebaseTeam
- `token` / batched tokens — streamed LLM output
- `file_complete` — per-file transform progress
- `sandbox_build_result` / build log chunks
- Mission / team info lines; orchestrator or agentic-debug events during `sandbox_build`
- Final `{type: "complete", files, sandbox_build?}`

UI token batching: `SSE_UI_TOKEN_BATCH_*`. Sandbox log cap: `SANDBOX_SSE_MAX_BUILD_LOG_CHARS`.

Mission audit on disk: `session_dir/agentic_pipeline/blackboard.json`.
