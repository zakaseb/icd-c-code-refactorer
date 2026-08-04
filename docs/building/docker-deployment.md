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

## Image build notes (PEP 668 + host UID)

Ubuntu’s system Python is **externally managed** (PEP 668). The Dockerfile therefore:

1. Installs `python3-venv` and creates `/opt/venv`.
2. `pip install`s app deps (`huggingface_hub`, FastAPI, LiteLLM, …) **into that venv**.
3. Prepends `/opt/venv/bin` to `PATH`.
4. Creates/renames the `developer` user to match `--build-arg USER_ID/GROUP_ID` (defaults `1000`). The base image already has GID `1000`, so the Dockerfile uses `getent`/`groupmod`/`usermod` instead of a naive `groupadd` (which fails with `GID already exists`).

`build.sh` passes your host `id -u` / `id -g` so the bind-mounted `models/` and `workspace/` stay writable.

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

### Model download (`hf_download.py`)

On first boot, `entrypoint.sh` runs `/opt/venv/bin/python hf_download.py` (copy of `src/utils/hf_download.py`):

| Env | Role |
|-----|------|
| `HF_REPO_ID` | Hugging Face repo (default `unsloth/Qwen3-Coder-Next-GGUF`) |
| `HF_MODEL` | Exact GGUF filename under that repo |
| `HOME` | Cache root → `$HOME/models/$HF_REPO_ID/` |

`serve_web.sh` mounts host `models/` → `/home/developer/models`, so downloads persist across restarts. The script skips the network call when the file already exists (exact path or recursive glob), fails fast if `huggingface_hub` is missing (wrong Python / broken image), and errors if the download finishes without the expected filename.

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
