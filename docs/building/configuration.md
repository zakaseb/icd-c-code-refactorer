---
title: Configuration
description: Environment variables and in-app constants
tags: [configuration, env]
---

# Configuration

## LLM / host

| Variable | Typical default | Purpose |
|----------|-----------------|---------|
| `OPENAI_BASE_URL` | `http://127.0.0.1:24000` | llama-server OpenAI-compatible API |
| `LITELLM_BASE_URL` | `http://127.0.0.1:4000` | LiteLLM gateway |
| `OPENAI_API_KEY` / `LLAMA_API_KEY` | local stub key | Auth for local servers |
| `HF_MODEL` / `HF_REPO_ID` | Qwen3-Coder-Next GGUF | Model identity for download |
| `LLAMA_ARG_CTX_SIZE` | Dockerfile large / scripts often `32768` | Context window |
| `LLAMA_ARG_N_GPU_LAYERS` | `auto` | GPU offload depth (`0` = CPU) |
| `LLAMA_ARG_*` | various | Host, port, threads, flash attn, split mode |
| `WORKSPACE_DIR` | `<repo>/workspace` | Session root |
| `LLM_PROMPT_SAFETY_TOKENS` | `1024` | Prompt clamp headroom |
| `LLM_MIN_OUTPUT_TOKENS` | `768` | Minimum generation budget |
| `SSE_UI_TOKEN_BATCH_*` | batched | Browser token batching |

## Sandbox backends

| Variable | Default | Purpose |
|----------|---------|---------|
| `SANDBOX_USE_ORCHESTRATOR` | `1` | Enable ReAct orchestrator |
| `SANDBOX_ORCH_MAX_STEPS` | `0` (unlimited) | Orchestrator step cap |
| `SANDBOX_ORCH_MAX_BUILDS` | `25` | Build budget when UI omits retries |
| `SANDBOX_ORCH_OUTER_ROUNDS` | `4` | Env-mode outer rounds |
| `SANDBOX_USE_AGENTIC` | `0` | Prefer agentic debug pipeline |
| `SANDBOX_AGENTIC_MAX_ATTEMPTS` | `120` | Agentic attempt cap |
| `SANDBOX_AGENTIC_NO_PROGRESS` | `3` | Stop on stalled progress |
| `SANDBOX_AGENTIC_OSCILLATION` | `2` | Oscillation detector |
| `SANDBOX_AGENTIC_EDIT_BUDGET` | `60` | Edit budget |
| `SANDBOX_AGENTIC_OUTER_ROUNDS` | `2` | Agentic outer rounds |
| `SANDBOX_SSE_MAX_BUILD_LOG_CHARS` | `200000` | Cap streamed build logs |

UI `sandbox_retries` overrides the effective build/attempt budget for a run — see [api/surface.md](../api/surface.md).

## Remote build

`REMOTE_BUILD_ENABLED`, `REMOTE_BUILD_HOST`, `REMOTE_BUILD_USER`, `REMOTE_BUILD_PASS`, `REMOTE_BUILD_SRC`, `REMOTE_BUILD_SCRIPT`, `REMOTE_BUILD_MODE`.

## In-app constants (not env)

Examples from `api/app.py`: ICD chunk size (`ICD_CHUNK_CHARS`), context caps, sandbox compilers (`SANDBOX_CC_NATIVE` / `SANDBOX_CC_ARM`), `SANDBOX_BUILD_TIMEOUT` (often 120s).

## OpenWiki recurrence

Scheduled docs refresh: `.github/workflows/openwiki-update.yml` (`openwiki code --update --print`). Requires `OPENROUTER_API_KEY` (and optional LangSmith) secrets in GitHub.
