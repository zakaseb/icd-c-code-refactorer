---
title: Architecture Overview
description: Major components, data flow, and session layout
tags: [architecture, overview]
---

# Architecture Overview

## Purpose

`icd-c-code-refactorer` automates updating embedded C code when an Interface Control Document changes. Inference stays on-box: browser → FastAPI → LiteLLM → llama-server (GGUF).

## Component diagram

```text
┌─────────────┐     SSE/REST      ┌──────────────────┐
│  Web UI     │ ◄──────────────► │  FastAPI (8081)  │
│ api/static/ │                  │  api/app.py      │
└─────────────┘                  └────────┬─────────┘
                                          │
                     ┌────────────────────┼────────────────────┐
                     ▼                    ▼                    ▼
              llama-server          LiteLLM (4000)      workspace/sessions/
              (24000, GGUF)         Anthropic-compat    <uuid>/artifacts
```

Inside the Docker entrypoint (`deployments/docker/entrypoint.sh`), three tmux panes typically run:

| Pane | Process | Default port |
|------|---------|--------------|
| 0 | `llama-server` | `24000` (`LLAMA_ARG_PORT`) |
| 1 | `litellm` | `4000` |
| 2 | `uvicorn app:app` | `8081` |

## Major modules

| Path | Role |
|------|------|
| `api/app.py` | Routes; classic sequential process **or** agentic entry (`_agentic_streaming_response`) |
| `api/agentic_pipeline/` | **Default mission path:** MissionController, blackboard, six teams, LLM backends |
| `api/per_file_compile.py` | Per-file `.c` → `.o` compile gate (`run_per_file_compile`) |
| `api/orchestrator.py` | ReAct sandbox debugger (`run_orchestrator`) |
| `api/agentic_debug.py` | Hypothesis/phase sandbox debugger (`run_agentic_debug`) — used inside IntegrationTeam when enabled |
| `api/gitnexus.py` / CodebaseTeam | Embedded codebase-context report (live as the `gitnexus` mission stage) |
| `api/static/` | SPA: `index.html`, `app.js`, `style.css`, `append_log_helper.js` |
| `src/utils/hf_download.py` | Downloads GGUF into `models/` at container start |
| `scripts/serve_web.sh` | Recommended host launcher (GPU preflight + docker run) |

Most of `src/pipelines/`, `src/dataset/`, and `src/evaluation/` are placeholders — the live pipeline lives under `api/`.

Deep dive: [agentic/multiagent-pipeline.md](../agentic/multiagent-pipeline.md).

## Session layout

Sessions are created under `WORKSPACE_DIR` (default `<repo>/workspace`), typically:

```text
workspace/sessions/<session_id>/
  original_code/          # uploaded .c/.h
  source_icd.txt          # extracted Source ICD text
  target_icd.txt          # extracted Target ICD text
  repo_contents/          # optional unzipped repo
  generated_code/         # transformed sources
  change_spec.txt         # distilled delta
  change_spec_raw.txt     # fuller analysis artefact
  verification_report.txt
  compile_report.txt
  sandbox/                # build workspace
  built_repo.zip
  sandbox_build_log.txt
  agentic_attempts/       # when agentic backend is on
  status.json / pipeline_events.jsonl
```

## Control flow (happy path)

1. `POST /api/session/create`
2. Upload code + ICDs (+ optional repo ZIP)
3. `GET /api/process/{id}` (SSE) — **agentic mission by default**; or `/api/process-agentic/{id}`
4. Optional conversation + `GET /api/regenerate/{id}`
5. `GET /api/download/{id}` and/or `/api/download-repo/{id}`

Pause/resume is supported via `/api/pause` and `/api/resume`.

## Design notes

- **Agentic-first on this branch:** `AGENTIC_PIPELINE=1` routes process through teams + blackboard; set `0` for classic sequential.
- **Streaming first:** long stages emit SSE tokens/events so the UI stays responsive on UMA hosts.
- **Local-only LLM:** llama-server (GGUF) + LiteLLM Anthropic proxy for Claude Agent SDK; resilient fallback to direct local calls.
- **Sandbox backends are swappable:** orchestrator (default), agentic debug, or legacy rewrite loop — see [sandbox/debugging.md](../sandbox/debugging.md).
