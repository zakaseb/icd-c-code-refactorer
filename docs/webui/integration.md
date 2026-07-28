---
title: Web UI Integration
description: How api/static talks to the FastAPI backend
tags: [webui, frontend]
---

# Web UI Integration

Location: `api/static/`.

| File | Role |
|------|------|
| `index.html` | Upload forms, stage indicators, sandbox-retries controls, conversation box |
| `app.js` | Session lifecycle, uploads, EventSource streaming, regenerate, downloads |
| `append_log_helper.js` | rAF-batched log DOM updates (reduces jank / OOM risk) |
| `style.css` | Layout and theming |

## User flow

1. Page load → `POST /api/session/create`.
2. User selects `.c`/`.h`, Source ICD, Target ICD, optional repo ZIP → matching `/api/upload/...` routes.
3. **Transform Code** builds an EventSource URL via `processStreamUrl()`, appending `?sandbox_retries=` from:
   - number input (default **25**), or
   - **Indefinite** toggle → `indefinite`.
4. UI handles SSE stages: `analysis`, `transform`, `header_doc`, `verification`, `compile`, `sandbox_build` (and `regeneration` on re-run). A `gitnexus` stage label may appear in the UI even though that report is not currently invoked from the live `process` stream.
5. User can pause/resume, preview files, download artefacts, send conversation feedback, then regenerate.

## Sandbox retries control

HTML elements (names may vary slightly):

- `#sandbox-retries-input` — integer attempts
- `#sandbox-retries-indefinite` — unlimited mode

These map directly to the `sandbox_retries` query parameter documented in [api/surface.md](../api/surface.md).

## Streaming robustness

- Token batching and log helper batching keep overnight runs from freezing/crashing the tab.
- Sandbox build log characters streamed to the browser are capped server-side.
- Stage UI folds `header_doc` tokens into file/analysis presentation where appropriate.

## Conversation & regenerate

- Feedback is posted to `/api/conversation/{session_id}`.
- **Re-generate** opens EventSource on `/api/regenerate/{session_id}` with the same retries query string.
- Regen reuses ICD/repo context and conversation history; it does not redo full ICD delta analysis from scratch.
