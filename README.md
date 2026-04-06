# ICD C Code Refactorer

Transform C source code between Interface Control Document (ICD) versions using AI-powered **fully local** inference. No data ever leaves your machine.

## Overview

When the ICD governing your embedded C codebase changes between versions, every struct, enum, message format, constant, and function signature potentially needs updating. Doing this by hand is tedious and error-prone.

This tool automates the entire workflow. You upload your existing C files, both ICD PDFs, and optionally a ZIP of the surrounding repository. The AI then analyzes the differences, generates conforming code, verifies it, and lets you iterate through a conversational feedback loop until the output builds cleanly.

### Key Features

- **ICD Delta Analysis** — Automatically extracts and compares two ICD PDFs to produce an exhaustive, structured change specification covering structs, enums, constants, function signatures, protocol changes, and more.
- **Repository-Aware Code Generation** — When a repository ZIP is provided, the tool extracts detailed codebase knowledge (types, function signatures, macros, naming conventions) and uses it as ground truth during transformation.
- **Multi-Pass Verification** — Every generated file goes through structural checks (brace matching, header guards, include resolution, function presence) and an LLM verification pass for ICD compliance and repository compatibility.
- **Conversational Feedback Loop** — After reviewing the output (e.g. pasting build errors), send feedback and click **Re-generate** to produce corrected code that considers your feedback holistically alongside all ICD and repository context.
- **Variable Inventory** — Verification reports include a complete inventory of all variables, macros, and function parameters in the generated code.
- **Detailed Verification Reports** — Every run produces a report documenting structural checks, verification outcomes, unified diffs of all changes, and the variable inventory.
- **Real-Time Streaming** — All pipeline stages stream progress via Server-Sent Events so you see analysis, transformation, and verification happen token by token.
- **Fully Local** — Runs Qwen3-Coder-30B via llama.cpp on your own GPU. No cloud APIs, no data exfiltration.

## Quick Start

### Prerequisites

| Requirement | Notes |
|---|---|
| Docker with NVIDIA GPU support | `nvidia-container-toolkit` installed and configured |
| NVIDIA GPU | 24 GB+ VRAM recommended for full GPU offload |
| ~20 GB disk space | For the quantised GGUF model (auto-downloaded on first run) |

### Build and Run

```bash
# Make scripts executable
chmod +x build.sh run.sh run_web.sh entrypoint.sh

# Build the Docker image (bakes in app code and dependencies)
./build.sh

# Launch the container (downloads model on first run, starts all services)
./run_web.sh
```

Open **[http://localhost:8081](http://localhost:8081)** in your browser once the container finishes starting up.

> `run_web.sh` runs a GPU preflight check, reserves CPU cores for the host, and validates port 8081 is free before starting. If the GPU is unavailable it offers a CPU-only fallback (significantly slower).

## How to Use

### 1. Upload Files

| Upload Zone | What to Provide | Required? |
|---|---|---|
| **Source Code** | Your `.c` and `.h` files to transform | Yes |
| **Source ICD** | PDF of the ICD version the code currently implements | Yes |
| **Target ICD** | PDF of the ICD version you want the code to conform to | Yes |
| **Repository ZIP** | ZIP of the surrounding codebase for context (headers, types, naming conventions) | Optional but recommended |

All upload zones support drag-and-drop and file picker dialogs.

### 2. Transform Code

Click **Transform Code** to start the pipeline. Progress streams in real time:

1. **ICD Analysis** — Both PDFs are extracted, chunked if large, and compared to produce a change specification. When a repository ZIP is provided, the codebase knowledge (file structure, struct definitions, enum values, function signatures, macros) is extracted and included in the analysis report.
2. **Code Transformation** — Each uploaded file is transformed against the target ICD, using the change specification, repository dependency headers, and codebase knowledge as context. Incomplete outputs are automatically continued.
3. **Verification** — Each generated file undergoes structural checks and an LLM verification pass. Corrections are applied automatically when possible.

### 3. Review Results

- **Preview** each generated file in the browser using the tab strip.
- **Download All** as a ZIP containing the transformed `.c`/`.h` files, the ICD analysis report (with repository codebase knowledge), and the verification report (with variable inventory).

### 4. Conversational Feedback (Iterate)

After the initial generation, a **conversation panel** appears below the results:

1. **Paste feedback** — Copy build errors, compiler warnings, test failures, or any other issues into the text area and click **Send Feedback** (or press `Ctrl+Enter`).
2. **Re-generate** — Click **Re-generate Code** to produce corrected output. The regeneration considers your feedback **holistically** alongside the ICD change specification, repository headers, and codebase knowledge to avoid error loops.
3. **Verification runs again** — Every regeneration round runs the full structural + LLM verification pipeline.
4. **Download the latest** — The **Download All** button always provides the most recent generated version.

You can repeat this cycle as many times as needed.

## Processing Pipeline

```
                        ┌──────────────────────────────────────────────┐
                        │              Upload Phase                    │
                        │  .c/.h files  +  Source ICD  +  Target ICD  │
                        │          (+ optional repo ZIP)               │
                        └──────────────────┬───────────────────────────┘
                                           │
                        ┌──────────────────▼───────────────────────────┐
                        │         ICD Analysis (Step 1)                │
                        │  • Extract PDF text (PyMuPDF)                │
                        │  • Direct comparison or chunked map-reduce   │
                        │  • Multi-pass continuation for completeness  │
                        │  • Build repo codebase knowledge             │
                        │  → change_spec.txt + icd_analysis.txt        │
                        └──────────────────┬───────────────────────────┘
                                           │
                        ┌──────────────────▼───────────────────────────┐
                        │     Code Transformation (Step 2)             │
                        │  Per file:                                   │
                        │  • Assemble prioritised prompt (file, spec,  │
                        │    repo deps, repo knowledge, target ICD,    │
                        │    cross-file context)                       │
                        │  • Stream generation + auto-continue         │
                        │  • Completeness validation                   │
                        └──────────────────┬───────────────────────────┘
                                           │
                        ┌──────────────────▼───────────────────────────┐
                        │         Verification (Step 3)                │
                        │  Per file:                                   │
                        │  • Structural checks (braces, guards,        │
                        │    includes, functions)                      │
                        │  • LLM verification + auto-correction        │
                        │  • Targeted fix pass if needed               │
                        │  • Variable inventory extraction             │
                        │  → verification_report.txt                   │
                        └──────────────────┬───────────────────────────┘
                                           │
                        ┌──────────────────▼───────────────────────────┐
                        │         Results + Download                   │
                        │  • Preview files in browser                  │
                        │  • Download ZIP (code + analysis + report)   │
                        └──────────────────┬───────────────────────────┘
                                           │
                           ┌───────────────▼──────────────────┐
                           │   Conversational Feedback Loop    │
                           │   (optional, repeatable)          │
                           │                                   │
                           │  1. User sends feedback           │
                           │  2. Re-generate with holistic     │
                           │     context (ICD + repo + feedback)│
                           │  3. Full verification re-runs     │
                           │  4. Download updated output       │
                           │         ↻ repeat                  │
                           └───────────────────────────────────┘
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Browser (localhost:8081)                      │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ .c / .h  │  │Source ICD│  │Target ICD│  │ Repo ZIP │           │
│  │  upload   │  │  (PDF)   │  │  (PDF)   │  │ (optional)│          │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘           │
│                                                                     │
│  [Transform Code]    [Preview / Download]    [Conversation Panel]   │
└────────────────────────────┬────────────────────────────────────────┘
                             │ REST API + SSE streams
┌────────────────────────────▼────────────────────────────────────────┐
│                    FastAPI Backend (app.py)                          │
│                                                                     │
│  • PDF text extraction (PyMuPDF)                                    │
│  • Session management (workspace/sessions/<uuid>/)                  │
│  • Repository context extraction (dependency tracing, knowledge)    │
│  • LLM orchestration (ICD analysis → transform → verify)            │
│  • Conversational feedback + holistic regeneration                   │
│  • Variable inventory extraction for verification reports            │
│  • ZIP packaging for downloads                                      │
└────────────────────────────┬────────────────────────────────────────┘
                             │ OpenAI-compatible streaming API
┌────────────────────────────▼────────────────────────────────────────┐
│                    llama.cpp (llama-server)                          │
│             Qwen3-Coder-30B-A3B quantised (Q4_K_XL)                 │
│                    GPU-accelerated (CUDA)                            │
└─────────────────────────────────────────────────────────────────────┘
```

| Layer | Technology | Role |
|---|---|---|
| **Frontend** | Vanilla HTML / CSS / JS | Drag-and-drop uploads, real-time SSE streaming, conversation UI |
| **Backend** | Python 3, FastAPI, uvicorn | Session management, PDF processing, LLM orchestration, verification |
| **PDF Extraction** | PyMuPDF (fitz) | Reliable text extraction from ICD PDFs |
| **LLM Inference** | llama.cpp `llama-server` | Local GPU inference, OpenAI-compatible streaming API |
| **Model** | Qwen3-Coder-30B-A3B-Instruct (GGUF Q4_K_XL) | Code-specialised LLM (~20 GB quantised) |
| **LLM Proxy** | LiteLLM | Anthropic-compatible API proxy (for Claude Code tooling) |
| **Container** | Docker + NVIDIA Container Toolkit | Reproducible deployment with GPU passthrough |

## API Reference

All endpoints are served by the FastAPI backend on port **8081**.

### Session Management

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/session/create` | Create a new session, returns `{ session_id }` |
| `GET` | `/api/status/{session_id}` | Get session status (uploaded files, state) |
| `DELETE` | `/api/session/{session_id}` | Delete a session and all its data |

### File Upload

| Method | Endpoint | Accepts | Description |
|---|---|---|---|
| `POST` | `/api/upload/code/{session_id}` | `.c`, `.h` (multipart) | Upload source code files |
| `POST` | `/api/upload/source-icd/{session_id}` | `.pdf` (multipart) | Upload source ICD PDF |
| `POST` | `/api/upload/target-icd/{session_id}` | `.pdf` (multipart) | Upload target ICD PDF |
| `POST` | `/api/upload/repo-zip/{session_id}` | `.zip` (multipart) | Upload repository ZIP for context |

### Processing

| Method | Endpoint | Response | Description |
|---|---|---|---|
| `GET` | `/api/process/{session_id}` | SSE stream | Run the full pipeline (analysis → transform → verify) |
| `GET` | `/api/regenerate/{session_id}` | SSE stream | Re-generate with conversation feedback |

SSE events use JSON payloads with a `type` field: `stage`, `token`, `info`, `file_complete`, `stage_complete`, `error`, `complete`.

### Results

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/preview/{session_id}/{filename}` | Preview a single generated file |
| `GET` | `/api/download/{session_id}` | Download all generated files as ZIP |

### Conversation

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/conversation/{session_id}` | Send feedback message (`{ "message": "..." }`) |
| `GET` | `/api/conversation/{session_id}` | Retrieve conversation history |

## Session Data Layout

Each session creates a directory under `workspace/sessions/<uuid>/`:

```
workspace/sessions/<uuid>/
├── status.json                 # Session state, file lists, regeneration count
├── original_code/              # Uploaded .c and .h files (untouched)
├── generated_code/             # Latest generated files + reports
│   ├── *.c / *.h               # Transformed source files
│   ├── icd_analysis.txt        # ICD change spec + repository codebase knowledge
│   └── verification_report.txt # Structural checks, diffs, variable inventory
├── source_icd.pdf / .txt       # Source ICD (PDF and extracted text)
├── target_icd.pdf / .txt       # Target ICD (PDF and extracted text)
├── repo.zip                    # Uploaded repository archive (if provided)
├── repo_contents/              # Extracted repository files
├── change_spec.txt             # Raw ICD change specification
├── target_summary.txt          # Consolidated target ICD summary
├── repo_knowledge.txt          # Extracted codebase knowledge (high + low level)
├── conversation.json           # Feedback messages and system responses
├── generated_code_v0/          # Archive of round 0 output (created on first regen)
├── generated_code_v1/          # Archive of round 1 output (created on second regen)
└── ...
```

## Project Structure

```
icd-c-code-refactorer/
├── webapp/
│   ├── app.py                  # FastAPI backend (all endpoints and LLM orchestration)
│   ├── requirements.txt        # Python dependencies (for local development)
│   └── static/
│       ├── index.html          # Single-page application HTML
│       ├── app.js              # Client-side JavaScript (uploads, SSE, conversation)
│       └── style.css           # Dark-themed responsive styles
├── Dockerfile                  # Multi-layer image: llama.cpp CUDA + Python + app
├── build.sh                    # Build the Docker image
├── run_web.sh                  # Launch container with GPU preflight and port checks
├── run.sh                      # Minimal container launch (no preflight)
├── entrypoint.sh               # Container entrypoint: model download + tmux services
├── hf_download.py              # Hugging Face model downloader
├── test_codebase_context.py    # Unit tests for context-building helpers
├── models/                     # Model storage (symlink or directory, gitignored)
├── workspace/                  # Runtime session data (gitignored)
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## Configuration

### Application Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_BASE_URL` | `http://127.0.0.1:24000` | llama.cpp server URL (direct, bypasses proxy) |
| `OPENAI_API_KEY` | `sk-1234-miaw` | API key for llama-server authentication |
| `LITELLM_BASE_URL` | `http://127.0.0.1:4000` | LiteLLM proxy URL |
| `HF_MODEL` | `Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf` | GGUF model filename |
| `WORKSPACE_DIR` | `./workspace` | Root directory for session storage |

### GPU / llama.cpp Environment Variables

These are read by `llama-server` inside the container. Defaults favour fast GPU inference with VRAM headroom on shared machines.

| Variable | Default | Description |
|---|---|---|
| `LLAMA_ARG_N_GPU_LAYERS` | `auto` | GPU layer offload — `auto` fits to available VRAM. Set a number (e.g. `40`) to cap. |
| `LLAMA_ARG_FLASH_ATTN` | `on` | Flash attention on GPU (CUDA). |
| `LLAMA_ARG_CTX_SIZE` | `98274` (image) / `32768` (run scripts) | Context window size. Affects KV cache VRAM usage. |
| `LLAMA_ARG_N_PREDICT` | Same as `CTX_SIZE` | Maximum tokens per generation. |
| `LLAMA_ARG_THREADS` | `nproc - 2` (set by run scripts) | CPU threads. Scripts reserve 2 cores for the host. |
| `LLAMA_ARG_CACHE_TYPE_K` | `q8_0` | KV cache key quantisation (reduces VRAM). |
| `LLAMA_ARG_CACHE_TYPE_V` | `q8_0` | KV cache value quantisation (reduces VRAM). |
| `LLAMA_ARG_SPLIT_MODE` | `none` | Single GPU. Use `row` or `layer` for multi-GPU setups. |
| `LLAMA_ARG_MAIN_GPU` | `0` | Primary GPU index for multi-GPU configurations. |

### Troubleshooting VRAM Issues

If you encounter out-of-memory errors, reduce context size or GPU layers:

```bash
docker run -ti --rm --network=host --gpus all \
  -e LLAMA_ARG_N_GPU_LAYERS=48 \
  -e LLAMA_ARG_CTX_SIZE=16384 \
  -e LLAMA_ARG_N_PREDICT=16384 \
  -v "$PWD/models":/home/developer/models \
  -v "$PWD/workspace":/home/developer/workspace \
  icd-c-code-refactorer:llama.cpp
```

## Container Services

The `entrypoint.sh` starts three services inside a tmux session:

| Pane | Service | Port | Purpose |
|---|---|---|---|
| 1 | `llama-server` | 24000 | LLM inference engine (CUDA-accelerated) |
| 2 | `litellm` | 4000 | Anthropic-compatible API proxy |
| 3 | `uvicorn` (FastAPI) | 8081 | Web application backend |

You can attach to the tmux session inside the running container for debugging:

```bash
docker exec -it icd-c-code-refactorer tmux attach -t llama-server
```

## Verification Report Contents

Each run produces a `verification_report.txt` included in the download ZIP:

1. **Checks Performed** — Lists all checks (structural integrity, include resolution, function presence, ICD compliance, repo compatibility, variable inventory).
2. **Results per File** — Structural check pass/fail with specific issues, verification outcome, and unified diffs of any corrections applied.
3. **Changes from User Feedback** (regeneration only) — Diffs showing what changed between rounds, with user feedback quoted.
4. **Variable Inventory** — Complete listing of all variables extracted from each generated file, organised by scope (global, local, parameter, macro) with types and initialiser values.

## Derived From

This project is forked from [qwen-claude-code-sw2](https://github.com/zakaseb/qwen-claude-code-sw2), which provides the local LLM inference infrastructure (Docker + llama.cpp + LiteLLM).

## License

MIT License — see [LICENSE](LICENSE) for details.
