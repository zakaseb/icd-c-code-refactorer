---
title: Docker & LLM Deployment
description: Build image, serve_web.sh, llama-server and LiteLLM
tags: [docker, deployment, llm]
---

# Docker & LLM Deployment

## Recommended path

```bash
./deployments/docker/build.sh
./scripts/serve_web.sh
# → http://localhost:8081
```

Image tag: `icd-c-code-refactorer:llama.cpp` (FROM `ghcr.io/ggml-org/llama.cpp:server-cuda`).

## What `serve_web.sh` does

1. Requires `models/` (create or symlink a model cache).
2. Ensures `workspace/` exists and port **8081** is free.
3. Reserves host CPU cores for the OS (`LLAMA_THREADS = nproc - 2`).
4. Checks NVIDIA “GPU Recovery Action: Reboot” and runs a CUDA preflight (`llama-server --list-devices`) with retries.
5. Starts the container with GPU device 0, mounting `models/` and `workspace/`.

Related scripts:

| Script | Use |
|--------|-----|
| `scripts/serve.sh` | Minimal container launch |
| `scripts/serve_web_split_gpu.sh` | Split llama vs web containers across GPUs |
| `scripts/monitor_host_during_ui.sh` | Host resource monitoring during UI runs |

## Entrypoint services

`deployments/docker/entrypoint.sh` brings up:

| Service | Port | Role |
|---------|------|------|
| llama-server | `24000` | Local GGUF inference (OpenAI-compatible) |
| LiteLLM | `4000` | Model gateway / Anthropic-compatible surface for tools |
| uvicorn `app:app` | `8081` | FastAPI + static UI |

Model download on first boot: `src/utils/hf_download.py` using `HF_REPO_ID` / `HF_MODEL` (Qwen3-Coder-Next GGUF by default).

## Hardware expectations

- **Recommended:** NVIDIA GPU with ~48 GB+ VRAM for full offload of the bundled quant.
- Smaller GPUs work with partial offload (`LLAMA_ARG_N_GPU_LAYERS`); CPU-only is possible but slow.
- Context size and sampling are controlled via `LLAMA_ARG_*` / `LLAMA_SAMPLING_*` (serve scripts often use a smaller ctx than the Dockerfile maximum).

## Local Python (advanced)

```bash
pip install -r requirements.txt
# run llama-server + litellm yourself, then:
cd api && uvicorn app:app --host 0.0.0.0 --port 8081
```

Docker remains the supported path for matching production behaviour.
