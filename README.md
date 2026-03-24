# ICD C Code Refactorer

Transform C source code between Interface Control Document (ICD) versions using AI-powered local inference.

## Overview

This tool automates the refactoring of C code (`.c` and `.h` files) when the governing Interface Control Document changes between versions. Instead of manually diffing two ICDs and hand-editing every struct, enum, message format, and function signature, you upload three things and let the AI handle it:

| Upload Zone | What to Provide |
|---|---|
| **Source Code** | Your existing `.c` and `.h` files |
| **Source ICD** | PDF of the ICD version your code currently implements |
| **Target ICD** | PDF of the ICD version you want the code to conform to |

The tool then:

1. **Extracts** text from both ICD PDFs
2. **Analyzes** the differences — struct field changes, new/removed messages, enum updates, protocol changes, etc.
3. **Transforms** each code file to match the target ICD, preserving architecture and coding style
4. **Delivers** downloadable `.c` and `.h` files that conform to the new ICD

## Quick Start

### Prerequisites

- Docker with NVIDIA GPU support
- NVIDIA GPU with sufficient VRAM (recommended: 24GB+)

### Build

```bash
chmod +x build.sh run.sh run_web.sh entrypoint.sh
./build.sh
```

### Run

**Speed:** For a model this size, **GPU inference is much faster than CPU**—the hot path (matrix math) runs on the accelerator. CPU-only mode is mainly a fallback when VRAM is insufficient.

**Sustainable defaults:** The image configures **llama-server** to use the GPU aggressively but safely:

- **`LLAMA_ARG_N_GPU_LAYERS=9999`** — request all layers on GPU. llama.cpp's built-in `--fit` mechanism (on by default) automatically reduces this to what actually fits in VRAM, leaving a ~1 GiB margin. On GPUs smaller than the model, this produces **partial offload** (most layers on GPU, remainder on CPU) which is still far faster than CPU-only.
- **`LLAMA_ARG_FLASH_ATTN=on`** and **quantized KV caches** (`LLAMA_ARG_CACHE_TYPE_*`) — reduce VRAM footprint and improve throughput.
- **Normal process priority** (`--prio 0` in `entrypoint.sh`) — avoids realtime scheduling that can freeze the desktop.
- **`run_web.sh` / `run.sh`** pass **`LLAMA_ARG_THREADS=(nproc - 2)`** (minimum 1) so the host keeps cores free for other work.

If you still hit out-of-memory errors, lower context or layers at runtime, for example:

```bash
docker run -ti --rm --network=host --gpus all \
  -e LLAMA_ARG_N_GPU_LAYERS=48 \
  -e LLAMA_ARG_CTX_SIZE=16384 \
  -e LLAMA_ARG_N_PREDICT=16384 \
  -v "$PWD/models":/home/developer/models \
  -v "$PWD/workspace":/home/developer/workspace \
  icd-c-code-refactorer:llama.cpp
```

```bash
./run_web.sh
```

Then open [http://localhost:8081](http://localhost:8081) in your browser.

## How to Use

1. **Upload Source Code** — Drag-and-drop your `.c` and `.h` files into the first card
2. **Upload Source ICD** — Drag-and-drop the PDF of the ICD your code currently implements
3. **Upload Target ICD** — Drag-and-drop the PDF of the ICD you want the code to implement
4. **Click "Transform Code"** — The pipeline will:
   - Analyze the two ICDs to produce a change specification
   - Transform each uploaded file against the target ICD
   - Stream progress in real time
5. **Preview & Download** — Inspect each generated file and download them all as a ZIP

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (UI)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ .c / .h  │  │Source ICD│  │Target ICD│  [Transform]  │
│  │  upload   │  │  (PDF)   │  │  (PDF)   │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└────────────────────┬────────────────────────────────────┘
                     │ REST + SSE
┌────────────────────▼────────────────────────────────────┐
│              FastAPI Backend (app.py)                    │
│  • PDF text extraction (PyMuPDF)                        │
│  • Session management                                   │
│  • LLM orchestration (ICD diff → code transform)        │
│  • ZIP packaging                                        │
└────────────────────┬────────────────────────────────────┘
                     │ OpenAI-compatible API
┌────────────────────▼────────────────────────────────────┐
│  LiteLLM proxy  →  llama.cpp  →  Qwen3-Coder-30B       │
│              (all running locally)                       │
└─────────────────────────────────────────────────────────┘
```

- **Frontend**: Vanilla HTML / CSS / JS with drag-and-drop uploads
- **Backend**: Python FastAPI with SSE streaming for real-time progress
- **PDF Processing**: PyMuPDF for reliable text extraction from ICD PDFs
- **LLM**: Qwen3-Coder-30B via llama.cpp (fully local, no data leaves your machine)
- **Proxy**: LiteLLM for OpenAI API compatibility

## Configuration

Environment variables (set in Docker or shell):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_BASE_URL` | `http://127.0.0.1:24000` | llama.cpp server URL |
| `OPENAI_API_KEY` | `sk-1234-miaw` | LLM API key |
| `LITELLM_BASE_URL` | `http://127.0.0.1:4000` | LiteLLM proxy URL |
| `HF_MODEL` | `Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf` | Model filename |
| `WORKSPACE_DIR` | `./workspace` | Session/file storage directory |

### GPU acceleration (llama.cpp)

These are read by **llama-server** (see [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)). Defaults favor **fast GPU inference** with **VRAM headroom** and **shared-machine** behavior.

| Variable | Default in image | Description |
|---|---|---|
| `LLAMA_ARG_N_GPU_LAYERS` | `9999` | Request all layers on GPU; llama.cpp's `--fit` auto-reduces to what fits. Lower this number to free more VRAM for other apps. |
| `LLAMA_ARG_FLASH_ATTN` | `on` | Flash attention on GPU when supported. |
| `LLAMA_ARG_CTX_SIZE` | `98274` (image); `32768` in `run_web.sh` / `run.sh`) | Prompt context length (affects KV cache size on GPU). |
| `LLAMA_ARG_N_PREDICT` | Same as context in each file | Max tokens per generation (`-1` = unlimited in llama.cpp). |
| `LLAMA_ARG_THREADS` | `-1` in image; overridden by `run_web.sh` / `run.sh` | CPU threads for llama (prefill/decode helpers); scripts reserve 2 cores for the host. |
| `LLAMA_ARG_SPLIT_MODE` | `row` | Multi-GPU tensor split mode (`row`, `layer`, or `none`). |
| `LLAMA_ARG_MAIN_GPU` | `0` | Primary GPU index when using multiple devices. |

## Derived From

This project is forked from [qwen-claude-code-sw2](https://github.com/zakaseb/qwen-claude-code-sw2), which provides the local LLM inference infrastructure (Docker + llama.cpp + LiteLLM).

## License

MIT License — see [LICENSE](LICENSE) for details.
