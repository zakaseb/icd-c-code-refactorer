---
title: Operational Runbooks
description: Common ops tasks and failure recovery
tags: [operations, runbooks]
---

# Operational Runbooks

## Start / stop the stack

```bash
./scripts/serve_web.sh          # start (blocks / foreground docker)
# stop: Ctrl-C or docker stop the running container
docker ps --filter ancestor=icd-c-code-refactorer:llama.cpp
```

UI: http://localhost:8081 — llama-server `:24000`, LiteLLM `:4000` inside the container network.

## Port already in use

`serve_web.sh` exits if **8081** is taken. Find and stop the occupant, or change `WEB_PORT` in the script.

## GPU preflight failures

1. Run `nvidia-smi`. If **GPU Recovery Action: Reboot**, reboot the host before retrying.
2. Re-run `serve_web.sh` (script retries CUDA device listing up to 3 times).
3. If still failing, inspect preflight output and consider CPU-only fallback only as a last resort (very slow).

## First-boot model download stuck / `huggingface_hub` import errors

- Rebuild with current `deployments/docker/Dockerfile` (venv at `/opt/venv`, `PATH` includes `/opt/venv/bin`). A system `pip install` on Ubuntu 24.04 fails with PEP 668 `externally-managed-environment`.
- If build dies on `groupadd: GID '1000' already exists`, you are on an old Dockerfile — use the `getent`/`groupmod` user setup.
- Entrypoint must call `/opt/venv/bin/python hf_download.py` (not bare system `python3` without the venv).
- Ensure `models/` is writable and has ~50 GB free; check entrypoint/`hf_download.py` logs.
- Confirm `HF_REPO_ID` / `HF_MODEL` match a file under `models/$HF_REPO_ID/`.

## Browser tab dies on long runs

- Prefer current `app.js` + `append_log_helper.js` (batched logs).
- Confirm server-side SSE token batching and sandbox log caps are enabled.
- Avoid opening multiple EventSource tabs for the same heavy session.

## Sandbox never finishes / ignores retry count

- Confirm the UI is sending `?sandbox_retries=N` (network tab).
- Session `status.json` should show `sandbox_retries_mode` / `sandbox_max_retries`.
- Finite mode must hard-stop after budget exhaustion (see `tests/development/test_orchestrator_build_budget.py`).
- Agentic debug is the default (`SANDBOX_USE_AGENTIC=1`); agentic caps apply. Set `SANDBOX_USE_AGENTIC=0` for the ReAct orchestrator.

## llama-server HTTP 400 / context overflow

- Lower prompt size / raise ctx (`LLAMA_ARG_CTX_SIZE`) carefully for VRAM.
- App path clamps prompts and retries on overflow (`LLM_PROMPT_SAFETY_TOKENS`, hardening tests).

## Rebuild after code changes

If the image `COPY`s `api/` rather than bind-mounting it:

```bash
./deployments/docker/build.sh
./scripts/serve_web.sh
```

Hot-copying a single module into a running container is possible for emergency debugging but is not the durable workflow.

## Refresh documentation

Locally (with a capable OpenWiki provider configured):

```bash
openwiki code --update --print
```

Or rely on `.github/workflows/openwiki-update.yml` (needs OpenRouter secret). Preserve [agentic-debug-pipeline.md](../agentic-debug-pipeline.md) and follow [INSTRUCTIONS.md](../INSTRUCTIONS.md).
