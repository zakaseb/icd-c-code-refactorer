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
| `SANDBOX_USE_AGENTIC` | `1` | Prefer agentic debug pipeline (`0` → ReAct orchestrator) |
| `SANDBOX_AGENTIC_MAX_ATTEMPTS` | `120` | Agentic attempt cap |
| `SANDBOX_AGENTIC_NO_PROGRESS` | `3` | Stop on stalled progress |
| `SANDBOX_AGENTIC_OSCILLATION` | `2` | Oscillation detector |
| `SANDBOX_AGENTIC_EDIT_BUDGET` | `60` | Edit budget |
| `SANDBOX_AGENTIC_OUTER_ROUNDS` | `2` | Agentic outer rounds |
| `SANDBOX_SSE_MAX_BUILD_LOG_CHARS` | `200000` | Cap streamed build logs |

UI `sandbox_retries` overrides the effective build/attempt budget for a run — see [api/surface.md](../api/surface.md).

## Remote build

`REMOTE_BUILD_ENABLED`, `REMOTE_BUILD_HOST`, `REMOTE_BUILD_USER`, `REMOTE_BUILD_PASS`, `REMOTE_BUILD_SRC`, `REMOTE_BUILD_SCRIPT`, `REMOTE_BUILD_MODE`.

## HEX / VirtuosoNext RTOS SDK

The `api/hex_sdk.py` module auto-discovers a locally installed HALCON HEX / VirtuosoNext SDK (typically `VisualDesigner-HEX-<version>/`) at runtime. When found, its `targets/<platform>/include` roots and preprocessor defines are pushed into the compile-gate (`api/per_file_compile.py`) and the sandbox build (`api/app.py`) `CFLAGS`/`LDFLAGS`, and a compact API summary is inlined into the transform + fix prompts so the LLM never invents kernel APIs. The SDK tree itself is **not** committed — it's ~415 MB — so the repository's `.gitignore` blocks `VisualDesigner-HEX-*/`, `VirtuosoNext*/`, `HEX-*/` and `hex-sdk/`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `HEX_SDK_DIR` | (unset) | Absolute path override. Takes highest priority. Skip discovery/search when set. |
| `HEX_SDK_DISABLE` | (unset) | Set to `1`/`true` to skip discovery entirely (native builds only). |
| `HEX_SDK_PLATFORM` | auto | e.g. `arm-cortex-a9`, `win64`, `posix32`. Auto-picked from the cross-compiler hint and available `targets/*` dirs. |
| `HEX_SDK_VARIANT` | `SP` | `SP` (single-processor) or `MP` (multi-processor) kernel. |
| `HEX_SDK_COMPILER` | `COs` | `CO0`, `CO3`, or `COs` — must match a library-suffix under `targets/<platform>/lib/`. |
| `HEX_SDK_DEBUG` | (unset) | `D1` or `D2` for debug-instrumented archives. |
| `HEX_SDK_PROTECTION` | (unset) | `PLNONE` or `PLSPACE` (space-partitioning enable). |

Discovery order: `HEX_SDK_DIR` → sibling of repo root → child of repo root → `~/HEX2/VirtuosoNext` / `~/VirtuosoNext` / `/opt/VirtuosoNext` / `/opt/hex-sdk` / `/usr/local/VirtuosoNext`. See `docs/sandbox/debugging.md` for how the discovery result flows through the build.

## In-app constants (not env)

Examples from `api/app.py`: ICD chunk size (`ICD_CHUNK_CHARS`), context caps, sandbox compilers (`SANDBOX_CC_NATIVE` / `SANDBOX_CC_ARM`), `SANDBOX_BUILD_TIMEOUT` (often 120s).

## OpenWiki recurrence

Scheduled docs refresh: `.github/workflows/openwiki-update.yml` (`openwiki code --update --print`). Requires `OPENROUTER_API_KEY` (and optional LangSmith) secrets in GitHub.
