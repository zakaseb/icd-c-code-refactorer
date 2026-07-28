---
title: Operational Runbooks
description: Common ops tasks for the agentic multi-agent branch
tags: [operations, runbooks]
---

# Operational Runbooks

## Agentic mission not running / classic path unexpected

- Confirm `AGENTIC_PIPELINE` is unset or `1` inside the container (`docker exec … env | grep AGENTIC`).
- Hit `GET /api/process-agentic/{session_id}` to force the mission path.
- If Claude Agent SDK is missing, agents should still run via `LocalLLMBackend`; check LiteLLM on `:4000` if using `claude` backend.
- Inspect `session_dir/agentic_pipeline/blackboard.json` for stage/feedback history.

## Start / stop the stack

```bash
./scripts/serve_web.sh
docker ps --filter ancestor=icd-c-code-refactorer:llama.cpp
```

UI: http://localhost:8081 — llama-server `:24000`, LiteLLM `:4000`.

## Port already in use

`serve_web.sh` exits if **8081** is taken. Stop the occupant or change `WEB_PORT`.

## GPU preflight failures

1. `nvidia-smi` — if **GPU Recovery Action: Reboot**, reboot the host.
2. Re-run `serve_web.sh` (retries CUDA listing).
3. CPU-only fallback only as last resort.

## First-boot model download stuck

- `models/` writable with ~50 GB free; check `hf_download.py` logs; later starts reuse cache.

## Browser tab dies on long runs

- Use current `app.js` + `append_log_helper.js`; keep a single EventSource tab per heavy session.

## Sandbox never finishes

- Check `SANDBOX_USE_ORCHESTRATOR` / `SANDBOX_USE_AGENTIC` and related `SANDBOX_*` caps.
- IntegrationTeam may emit blackboard feedback targeting `transform` or `compile`.
- Review `sandbox_build_log.txt` and (if agentic debug) `agentic_attempts/`.

## Mission stuck revisiting stages

- Caps: `AGENTIC_MAX_STAGE_RUNS`, `AGENTIC_MAX_REVISITS_PER_STAGE`.
- Set `AGENTIC_ROUTER_LLM=0` for deterministic-only routing while debugging.
- Read `blackboard.json` history/feedback.

## llama-server HTTP 400 / context overflow

- Tune `LLAMA_ARG_CTX_SIZE` / `LLM_PROMPT_SAFETY_TOKENS`; see hardening tests.

## Rebuild after code changes

```bash
./deployments/docker/build.sh
./scripts/serve_web.sh
```

Required after pulling `claude-agent-sdk` / Dockerfile changes on this branch.

## Keep documentation current

When you add or change a feature on this branch, update the matching pages under `docs/` (exposed as `openwiki/`) in the **same change**. Follow [INSTRUCTIONS.md](../INSTRUCTIONS.md). Recurring refresh:

```bash
openwiki code --update --print
```

or `.github/workflows/openwiki-update.yml` (needs OpenRouter secret). Preserve [agentic-debug-pipeline.md](../agentic-debug-pipeline.md) and [agentic/multiagent-pipeline.md](../agentic/multiagent-pipeline.md).
