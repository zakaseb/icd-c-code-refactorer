---
title: Web UI Integration
description: How api/static talks to the FastAPI backend
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
3. **Transform Code** opens an EventSource on `/api/process/{session_id}`. The page does not send a sandbox retry budget.
4. UI handles SSE stages: `analysis`, `transform`, `header_doc`, `verification`, `compile`, `sandbox_build` (and `regeneration` on re-run). After the last script is transformed, `header_doc` is its own step, then verification, compile, and sandbox. A `gitnexus` stage label may appear in the UI even though that report is not currently invoked from the live `process` stream.
5. If analysis emits `variant_choice_required`, the stream closes and `#variant-choice` lists one radio group per peripheral. **Continue with this variant** posts `/api/variant-choice/{session_id}` and opens the process stream again. A Target ICD with one variant skips this step.
6. User can pause/resume, preview files, download artefacts, send conversation feedback, then regenerate.

## Streaming robustness

- Token batching and log helper batching keep overnight runs from freezing/crashing the tab.
- Sandbox build log characters streamed to the browser are capped server-side.
- `header_doc` tokens are shown on their own step after the transformed scripts. Verification tokens are streamed as the model writes them, so the process stream stays open through compile and sandbox.

## Conversation & regenerate

- Feedback is posted to `/api/conversation/{session_id}`.
- **Re-generate** opens EventSource on `/api/regenerate/{session_id}`.
- Regen reuses ICD/repo context and conversation history; it does not redo full ICD delta analysis from scratch.
