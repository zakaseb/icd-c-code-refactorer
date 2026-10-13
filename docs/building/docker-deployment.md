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

Image tag: `icd-c-code-refactorer:llama.cpp`. The base is pinned to llama.cpp **b11580** (`ghcr.io/ggml-org/llama.cpp:server-cuda@sha256:c4348afb91499b10fad12f92626c7a8ca8b69d6475d10c793b810cb61237ce00`). Qwen3.8's GGUF includes an MTP block at layer 64. Older llama.cpp (the previous image was b9102) looks for `blk.64.ssm_conv1d.weight` and exits before the server is ready. b11580 ignores that block unless speculative MTP is turned on. This service does not turn it on.

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
6. If a HEX SDK is present (`HEX_SDK_DIR`, or `VisualDesigner-HEX-*` / `VirtuosoNext*` / `HEX-*` next to the repo), bind-mounts it read-only at `/opt/hex-sdk` and sets `HEX_SDK_DIR`. The compile gate and sandbox build need that tree for `#include <L1_api.h>`. Restart `serve_web.sh` after adding the SDK; the image itself does not contain it.

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
| `HF_REPO_ID` | Hugging Face repo (default `unsloth/Qwen3.8-27B-GGUF`) |
| `HF_MODEL` | Exact GGUF filename (default `Qwen3.8-27B-UD-Q4_K_XL.gguf`, about 17.6 GB) |
| `HOME` | Cache root → `$HOME/models/$HF_REPO_ID/` |

`serve_web.sh` mounts host `models/` → `/home/developer/models`, so downloads persist across restarts. The script skips the network call when the file already exists (exact path or recursive glob), fails fast if `huggingface_hub` is missing (wrong Python / broken image), and errors if the download finishes without the expected filename.

## Hardware expectations

- **Recommended:** NVIDIA GPU with ~24 GB VRAM for the bundled Qwen3.8-27B quant (~17.6 GB of weights). The Dockerfile context also needs KV-cache memory; `LLAMA_ARG_N_GPU_LAYERS=auto` spills layers when VRAM is tight.
- Smaller GPUs work with partial offload (`LLAMA_ARG_N_GPU_LAYERS`); CPU-only is possible but slow.
- Context size and sampling are controlled via `LLAMA_ARG_*` / `LLAMA_SAMPLING_*` (serve scripts often use a smaller ctx than the Dockerfile maximum).

## Local Python (advanced)

```bash
pip install -r requirements.txt
# run llama-server + litellm yourself, then:
cd api && uvicorn app:app --host 0.0.0.0 --port 8081
```

Docker remains the supported path for matching production behaviour.
