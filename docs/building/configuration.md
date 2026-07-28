---
title: Configuration
description: Environment variables and in-app constants
tags: [configuration, env]
---

# Configuration

## Agentic multi-agent mission (this branch)

| Variable | Default | Purpose |
|----------|---------|---------|
| `AGENTIC_PIPELINE` | `1` | Route `/api/process` through MissionController |
| `AGENTIC_MAX_STAGE_RUNS` | `12` | Cap total team runs |
| `AGENTIC_MAX_REVISITS_PER_STAGE` | `2` | Extra runs per stage after first |
| `AGENTIC_ROUTER_LLM` | `1` | LLM router when multiple revisit targets |
| `AGENTIC_LLM_BACKEND` | `auto` | `auto` / `claude` / `local` |
| `AGENTIC_ANTHROPIC_BASE_URL` / `ANTHROPIC_BASE_URL` | LiteLLM `:4000` | Anthropic-compatible base for Claude Agent SDK |
| `AGENTIC_CLAUDE_MODEL` / `ANTHROPIC_MODEL` | app / `openai/$HF_MODEL` | Model id announced to endpoint |
| `ANTHROPIC_AUTH_TOKEN` | local stub | Auth for local Anthropic proxy |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | `1` (Docker) | Keep SDK off cloud extras |

Full mission docs: [agentic/multiagent-pipeline.md](../agentic/multiagent-pipeline.md).

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
| `SANDBOX_USE_AGENTIC` | `0` | Prefer agentic debug pipeline inside sandbox |
| `SANDBOX_AGENTIC_MAX_ATTEMPTS` | `120` | Agentic attempt cap |
| `SANDBOX_AGENTIC_NO_PROGRESS` | `3` | Stop on stalled progress |
| `SANDBOX_AGENTIC_OSCILLATION` | `2` | Oscillation detector |
| `SANDBOX_AGENTIC_EDIT_BUDGET` | `60` | Edit budget |
| `SANDBOX_AGENTIC_OUTER_ROUNDS` | `2` | Agentic outer rounds |
| `SANDBOX_SSE_MAX_BUILD_LOG_CHARS` | `200000` | Cap streamed build logs |

## Remote build

`REMOTE_BUILD_ENABLED`, `REMOTE_BUILD_HOST`, `REMOTE_BUILD_USER`, `REMOTE_BUILD_PASS`, `REMOTE_BUILD_SRC`, `REMOTE_BUILD_SCRIPT`, `REMOTE_BUILD_MODE`.

## OpenWiki recurrence

Scheduled docs refresh: `.github/workflows/openwiki-update.yml` (`openwiki code --update --print`). Requires `OPENROUTER_API_KEY` (and optional LangSmith) secrets in GitHub.
