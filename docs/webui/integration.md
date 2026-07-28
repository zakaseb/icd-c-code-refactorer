---
title: Web UI Integration
description: How api/static talks to the FastAPI backend on this branch
tags: [webui, frontend]
---

# Web UI Integration

Location: `api/static/`.

| File | Role |
|------|------|
| `index.html` | Upload forms, stage indicators, conversation box |
| `app.js` | Session lifecycle, uploads, EventSource streaming, regenerate, downloads |
| `append_log_helper.js` | rAF-batched log DOM updates (reduces jank / OOM risk) |
| `style.css` | Layout and theming |

## User flow

1. Page load → `POST /api/session/create`.
2. User selects `.c`/`.h`, Source ICD, Target ICD, optional repo ZIP → matching `/api/upload/...` routes.
3. **Transform Code** opens EventSource on `/api/process/{session_id}` (no retry query string on this branch).
4. UI handles SSE stages: `analysis`, **`gitnexus`**, `transform` (per-file steps), `verification`, `compile`, `sandbox_build`, and `regeneration` on re-run.
5. User can pause/resume, preview files, download artefacts, send conversation feedback, then regenerate.

## Stages the UI knows about

`app.js` creates/updates steps for:

| Stage | Notes |
|-------|-------|
| `analysis` | ICD delta / change_spec |
| `gitnexus` | Live CodebaseTeam stage on the agentic path |
| `transform` | One UI step per generated file |
| `verification` | Structural + compliance |
| `compile` | Per-file compile gate |
| `sandbox_build` | Integration / sandbox |
| `regeneration` | Conversational re-generate path |

There are **no** sandbox-retries number/indefinite controls in this branch’s UI.

## Streaming robustness

- Token batching and log helper batching keep overnight runs from freezing/crashing the tab.
- Sandbox build log characters streamed to the browser are capped server-side.

## Conversation & regenerate

- Feedback is posted to `/api/conversation/{session_id}`.
- **Re-generate** opens EventSource on `/api/regenerate/{session_id}`.
- Regen reuses ICD/repo context and conversation history; it does not redo full ICD delta analysis from scratch.
