---
title: Web UI Integration
description: How api/static talks to the FastAPI backend on this branch
tags: [webui, frontend]
---

# Web UI Integration

Location: `api/static/`. Visual language matches the Halcon-themed UI (colors, typography, layout) inherited from main.

| File | Role |
|------|------|
| `index.html` | Upload forms, stage indicators, sandbox-retries controls, conversation box |
| `app.js` | Session lifecycle, uploads, EventSource streaming, regenerate, downloads |
| `append_log_helper.js` | rAF-batched log DOM updates |
| `style.css` | Halcon theme styles |

## User flow

1. Page load → `POST /api/session/create`.
2. Upload `.c`/`.h`, Source ICD, Target ICD, optional repo ZIP.
3. **Transform Code** opens EventSource via `processStreamUrl()`, appending `?sandbox_retries=` from:
   - number input (default **25**), or
   - **Indefinite** toggle → `indefinite`.
4. UI handles SSE stages: `analysis`, `gitnexus`, `transform` (per-file), `verification`, `compile`, `sandbox_build`, and `regeneration` on re-run.
5. Pause/resume, preview, download, conversation feedback, then regenerate (same retries query).

## Sandbox retries control

- `#sandbox-retries-input` — integer attempts
- `#sandbox-retries-indefinite` — unlimited mode

Maps to the `sandbox_retries` query on process / process-agentic / regenerate.

## Streaming robustness

Token batching + log helper batching; server-side sandbox log caps.
