**This project follows the [Project Structure Guidelines](https://hal-confluence.edgegroup.ae/spaces/AIENG/pages/409174019/Project+Structure+Guidelines)**

# ICD C Code Refactorer

Transform C source code between Interface Control Document (ICD) versions using AI-powered fully local LLM inference.


# Table of Contents

* [Project Overview](#project-overview)
* [Hardware](#hardware)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Repository Layout](#repository-layout)
* [Usage](#usage)
* [Model / Data Card & Licenses](#model--data-card--licenses)
* [Testing](#testing)
* [Serving & Deployment](#serving--deployment)
* [Support](#support)
* [Contributing](#contributing)
* [Authors and Acknowledgment](#authors-and-acknowledgment)
* [License](#license)
* [Project Status](#project-status)



# Project Overview

* **Problem:** When the ICD governing your embedded C codebase changes between versions, every struct, enum, message format, constant, and function signature potentially needs updating. Doing this by hand is tedious and error-prone.
* **Approach:** Local LLM inference pipeline tool that automates the entire workflow. You upload your existing C files, both ICD PDFs, and optionally a ZIP of the surrounding repository. The AI then analyzes the differences, generates conforming code, verifies it, and lets you iterate through a conversational feedback loop until the output builds cleanly.
* **Features:**

- **ICD Delta Analysis** — Automatically extracts and compares two ICD PDFs to produce an exhaustive, structured change specification covering structs, enums, constants, function signatures, protocol changes, and more.
- **Repository-Aware Code Generation** — When a repository ZIP is provided, the tool extracts detailed codebase knowledge (types, function signatures, macros, naming conventions) and uses it as ground truth during transformation.
- **GitNexus Codebase Understanding** — Immediately after the ICD change specification is produced, GitNexus deterministically extracts embedded-systems-specific relationships from the uploaded repository (ISR ↔ task wiring, drivers ↔ peripherals, RTOS/superloop task interactions, state-machine transitions, communication-stack dependencies, memory ownership, HAL/BSP boundaries, bootloader → firmware hand-off, firmware-update flow, safety-critical execution chains, cross-module #include graph, global-variable read/write graph, and build-script dependencies). The resulting `gitnexus_report.txt` is added to the context fed into every .c/.h generation, verification, per-file compile fix, sandbox build and regeneration prompt — alongside the existing ICD change spec, repository codebase knowledge and dependency headers.
- **Multi-Pass Verification** — Every generated file goes through structural checks (brace matching, header guards, include resolution, function presence) and an LLM verification pass for ICD compliance and repository compatibility.
- **Conversational Feedback Loop** — After reviewing the output (e.g. pasting build errors), send feedback and click **Re-generate** to produce corrected code that considers your feedback holistically alongside all ICD and repository context.
- **Variable Inventory** — Verification reports include a complete inventory of all variables, macros, and function parameters in the generated code.
- **Detailed Verification Reports** — Every run produces a report documenting structural checks, verification outcomes, unified diffs of all changes, and the variable inventory.
- **Real-Time Streaming** — All pipeline stages stream progress via Server-Sent Events so you see analysis, transformation, and verification happen token by token.
- **Fully Local** — Runs Qwen3-Coder-Next (80B-A3B coder-specialised) via llama.cpp on your own GPU. No cloud APIs, no data exfiltration.


# Hardware

Device: `Workstation`, `AI-Laptop-Dell`, `AI-Laptop-MSI`

Requires an NVIDIA GPU with 48 GB+ VRAM (e.g. A6000, L40, A100 40/80 GB, H100) for full GPU offload of the Qwen3-Coder-Next model at the bundled `UD-Q4_K_XL` quant (~49.3 GB). The number of GPU layers offloaded is controlled by `LLAMA_ARG_N_GPU_LAYERS` (default: `auto`), so smaller GPUs (e.g. 24 GB) still work by spilling layers to CPU RAM with reduced throughput. Because Qwen3-Coder-Next is a Mixture-of-Experts model with ~3 B active parameters per token (the "A3B" suffix), per-token latency is comparable to a dense ~3 B model when fully offloaded — much faster than its 80 B total-parameter count would suggest. CPU-only fallback is available but significantly slower.

Please refer to [Reproducible Experiments](https://hal-confluence.edgegroup.ae/spaces/AIENG/pages/424772002/Reproducible+Experiments+in+PyTorch) for settings on reproducible results.


# Prerequisites

* Docker with NVIDIA GPU support (`nvidia-container-toolkit` installed and configured)
* NVIDIA GPU with 48 GB+ VRAM recommended (24 GB works with partial offload)
* ~50 GB free disk space for the quantised GGUF model (~49.3 GB at `UD-Q4_K_XL`, auto-downloaded on first run)
* Tools:
  * Docker + NVIDIA Container Toolkit
  * `nvidia-smi` accessible on the host


# Installation

## Option 1: Docker (recommended)

```bash
# Make scripts executable
chmod +x deployments/docker/build.sh deployments/docker/entrypoint.sh scripts/serve.sh scripts/serve_web.sh

# Create the models directory (downloaded automatically on first run)
mkdir -p models

# Build the Docker image
./deployments/docker/build.sh

# Launch the container
./scripts/serve_web.sh
```

Open **[http://localhost:8081](http://localhost:8081)** in your browser once the container finishes starting up.

**Transform Code** uses the **agentic multi-agent pipeline** by default (Claude Agent SDK → local LiteLLM proxy → the same Qwen GGUF, with llama-server fallback). To use the classic sequential pipeline instead, rebuild/run with `AGENTIC_PIPELINE=0` (see [Agentic Multi-Agent Pipeline](#agentic-multi-agent-pipeline)).

> On first run, `entrypoint.sh` downloads the Qwen3-Coder-Next GGUF model (~49.3 GB at `UD-Q4_K_XL`) into `models/`. Subsequent starts are fast as the model is cached.

> `run_web.sh` runs a GPU preflight check, reserves CPU cores for the host, and validates port 8081 is free before starting. If the GPU is unavailable it offers a CPU-only fallback (significantly slower).

## Option 2: Local Python (advanced)

```bash
pip install -r requirements.txt
cd api && uvicorn app:app --host 0.0.0.0 --port 8081
```

> You will also need to run `llama-server` and `litellm` separately. Docker is strongly recommended.


# Repository Layout

```text
icd-c-code-refactorer/
│
├── api/                               # FastAPI backend
│   ├── app.py                         # All endpoints and LLM orchestration
│   ├── gitnexus.py                    # GitNexus codebase-understanding extractor
│   ├── per_file_compile.py            # Per-file .c → .o compile gate
│   ├── orchestrator.py                # Sandbox-build agentic debug loop
│   ├── agentic_debug.py               # Agentic-AI single-hypothesis debug pipeline
│   ├── agentic_pipeline/              # Multi-agent teams + agent-of-agents mission
│   │   ├── mission.py                 # MissionController (non-sequential routing)
│   │   ├── blackboard.py              # Shared state + routable feedback
│   │   ├── llm.py                     # Claude Agent SDK / local llama-server backends
│   │   └── teams/                     # One multi-agent team per pipeline stage
│   └── static/
│       ├── index.html                 # Single-page application
│       ├── app.js                     # Uploads, SSE streaming, conversation UI
│       └── style.css                  # Dark-themed responsive styles
│
├── src/
│   └── utils/
│       └── hf_download.py             # Hugging Face model downloader
│
├── tests/
│   └── development/
│       └── test_codebase_context.py   # Unit tests for context-building helpers
│
├── scripts/
│   ├── serve.sh                       # Minimal container launch
│   └── serve_web.sh                   # Launch with GPU preflight and port checks
│
├── deployments/
│   ├── docker/
│   │   ├── Dockerfile                 # Multi-layer image: llama.cpp CUDA + Python + app
│   │   ├── build.sh                   # Build the Docker image
│   │   ├── entrypoint.sh              # Model download + tmux services startup
│   │   └── .dockerignore
│   ├── edge_device/
│   └── k8s/
│
├── models/                            # GGUF model storage (gitignored, ~20 GB)
├── workspace/                         # Runtime session data (gitignored)
│
├── .gitignore
├── .gitattributes
├── requirements.txt
├── README.md
└── pyproject.toml
```


# Usage

## 1. Upload Files

| Upload Zone | What to Provide | Required? |
|---|---|---|
| **Source Code** | Your `.c` and `.h` files to transform | Yes |
| **Source ICD** | PDF of the ICD version the code currently implements | Yes |
| **Target ICD** | PDF of the ICD version you want the code to conform to | Yes |
| **Repository ZIP** | ZIP of the surrounding codebase for context (headers, types, naming conventions) | Optional but recommended |

All upload zones support drag-and-drop and file picker dialogs.

## 2. Transform Code

Click **Transform Code** to start the pipeline. Progress streams in real time:

1. **ICD Analysis** — Both PDFs are extracted, chunked if large, and compared to produce a change specification. When a repository ZIP is provided, the codebase knowledge (file structure, struct definitions, enum values, function signatures, macros) is extracted and included in the analysis report.
2. **GitNexus Codebase Report** — Runs immediately after the ICD change spec. Deterministically scans the uploaded repository and source scripts to extract ISR↔task wiring, driver/peripheral access, RTOS or superloop task interactions, state-machine transitions, communication-stack dependencies, memory ownership, HAL/BSP boundary, bootloader/firmware-update hooks, safety-critical chains, cross-module dependencies, the global-variable read/write graph and script dependencies. The output is saved as `gitnexus_report.txt` and added to the context for every later prompt.
3. **Code Transformation** — Each uploaded file is transformed against the target ICD, using the change specification, repository dependency headers, codebase knowledge **and the GitNexus report** as context. Incomplete outputs are automatically continued.
4. **Verification** — Each generated file undergoes structural checks and an LLM verification pass. Corrections are applied automatically when possible.

## 3. Review & Download

- **Preview** each generated file in the browser using the tab strip.
- **Download All** as a ZIP containing the transformed `.c`/`.h` files, the ICD analysis report (with repository codebase knowledge), and the verification report (with variable inventory).

## 4. Conversational Feedback (Iterate)

After the initial generation, a **conversation panel** appears below the results:

1. **Paste feedback** — Copy build errors, compiler warnings, test failures, or any other issues into the text area and click **Send Feedback** (or press `Ctrl+Enter`).
2. **Re-generate** — Click **Re-generate Code** to produce corrected output. The regeneration considers your feedback **holistically** alongside the ICD change specification, repository headers, and codebase knowledge to avoid error loops.
3. **Verification runs again** — Every regeneration round runs the full structural + LLM verification pipeline.
4. **Download the latest** — The **Download All** button always provides the most recent generated version.

You can repeat this cycle as many times as needed.

## 5. Verification Report Contents

Each run produces a `verification_report.txt` included in the download ZIP:

1. **Checks Performed** — lists all checks run: structural integrity, include resolution, function presence, ICD compliance, repo compatibility, and variable inventory.
2. **Results per File** — structural check pass/fail with specific issues, verification outcome, and unified diffs of any corrections applied automatically.
3. **Changes from User Feedback** *(regeneration rounds only)* — diffs showing what changed between rounds, with the user feedback quoted.
4. **Variable Inventory** — complete listing of all variables extracted from each generated file, organised by scope (global, local, parameter, macro) with types and initialiser values.

# Agentic Multi-Agent Pipeline

The classic sequential pipeline has an agentic re-imagining in
`api/agentic_pipeline/`: every stage is a **multi-agent team**, and the
whole mission is orchestrated by an **agent-of-agents**
(`MissionController`). All teams share a blackboard and publish feedback
that can target ANY stage — so reiteration is non-sequential: a failed
sandbox build can send the mission straight back to code generation or even
ICD analysis, not just the previous step. Stage names, SSE events and
artifacts are identical to the classic pipeline, so the web UI works
unchanged.

| Stage (SSE) | Team | Agents |
|---|---|---|
| `analysis` | Ingestion & Analysis | TargetSummarizer, DocumentAnalyst, SpecSynthesizer, Distiller, FactAuditor |
| `gitnexus` | Codebase Understanding | RepoCartographer, SystemsArchaeologist, ImpactAssessor |
| `transform` | Code Generation (.h first, then .c) | VariantScout, InterfaceArchitect, ImplementationEngineer, CompletionCritic |
| `verification` | Verification | StructuralAuditor, ComplianceReviewer, RepairEngineer |
| `compile` | Compilation | ToolchainScout, BuildOperator, GateKeeper |
| `sandbox_build` | Integration & Build | SandboxEngineer, DebugCrew, IntegrationJudge |

**How to run it**

- **Default** — `./scripts/serve_web.sh` then **Transform Code** uses the
  agentic pipeline (Claude Agent SDK → local LiteLLM proxy → same GGUF
  model, with llama-server fallback).
- `GET /api/process-agentic/{session_id}` — always available (same path).
- `AGENTIC_PIPELINE=0` — restores the classic sequential pipeline on
  `/api/process`.

**LLM backend** (same local model as the classic pipeline)

- `AGENTIC_LLM_BACKEND=auto|claude|local` (default `auto`). With `claude`,
  agent calls go through the **Claude Agent SDK** pointed at a LOCAL
  Anthropic-compatible endpoint — Ollama (>= 0.14, native `/v1/messages`)
  or the LiteLLM proxy shipped in the Docker deployment. `local` calls
  llama-server directly (identical to the classic pipeline). `auto` picks
  the SDK when installed, with per-call degradation to `local`.
- `AGENTIC_ANTHROPIC_BASE_URL` (default: LiteLLM `http://127.0.0.1:4000`;
  use `http://127.0.0.1:11434` for Ollama) and `AGENTIC_CLAUDE_MODEL`
  (default: the configured `HF_MODEL` name).

**Mission bounds**: `AGENTIC_MAX_STAGE_RUNS` (default 12) and
`AGENTIC_MAX_REVISITS_PER_STAGE` (default 2) guarantee termination;
`AGENTIC_ROUTER_LLM=0` disables the LLM router in favour of the
deterministic feedback-routing policy. The RTOS/embedded toolchain
contract (Xilinx SDK 2018 / GCC 7.3.1 / gnu99 / newlib, cross-compiler
detection) is inherited from the classic pipeline helpers.

# Processing Pipeline

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
                        │   GitNexus Codebase Report (Step 1b)         │
                        │  Deterministic extraction of:                │
                        │  • ISR ↔ task wiring                         │
                        │  • Drivers ↔ peripherals & comm stacks       │
                        │  • RTOS / superloop task interactions        │
                        │  • State machine transitions                 │
                        │  • Memory ownership, HAL/BSP boundary        │
                        │  • Bootloader → firmware handoff             │
                        │  • Firmware update flow                      │
                        │  • Safety-critical execution chains          │
                        │  • Cross-module #include graph               │
                        │  • Global variable read/write graph          │
                        │  • Build / script dependencies               │
                        │  → gitnexus_report.txt                       │
                        └──────────────────┬───────────────────────────┘
                                           │
                        ┌──────────────────▼───────────────────────────┐
                        │     Code Transformation (Step 2)             │
                        │  Per file:                                   │
                        │  • Assemble prioritised prompt (file, spec,  │
                        │    repo deps, repo knowledge, GitNexus,      │
                        │    target ICD, cross-file context)           │
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

# Architecture

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
│        Qwen3-Coder-Next (80B-A3B coder) quantised (UD-Q4_K_XL)      │
│                    GPU-accelerated (CUDA)                            │
└─────────────────────────────────────────────────────────────────────┘
```

| Layer | Technology | Role |
|---|---|---|
| **Frontend** | Vanilla HTML / CSS / JS | Drag-and-drop uploads, real-time SSE streaming, conversation UI |
| **Backend** | Python 3, FastAPI, uvicorn | Session management, PDF processing, LLM orchestration, verification |
| **PDF Extraction** | PyMuPDF (fitz) | Reliable text extraction from ICD PDFs |
| **LLM Inference** | llama.cpp `llama-server` | Local GPU inference, OpenAI-compatible streaming API |
| **Model** | Qwen3-Coder-Next (GGUF UD-Q4_K_XL) | Coder-specialised MoE LLM trained for agentic coding workflows; ~80 B total / ~3 B active params per token (~49.3 GB quantised) |
| **LLM Proxy** | LiteLLM | Anthropic-compatible API proxy (for Claude Code tooling) |
| **Container** | Docker + NVIDIA Container Toolkit | Reproducible deployment with GPU passthrough |

# Model / Data Card & Licenses

| | Details |
|---|---|
| **Model** | Qwen3-Coder-Next (GGUF UD-Q4_K_XL) |
| **Version** | `unsloth/Qwen3-Coder-Next-GGUF` |
| **Source** | [Hugging Face — unsloth/Qwen3-Coder-Next-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF) |
| **License** | Apache 2.0 |
| **Usage restrictions** | No PII or sensitive data should be included in uploaded ICD PDFs or source files if operating under data residency constraints. All inference is local — no data leaves the machine. |


# Testing

Run the development test suite (no Docker or live LLM required):

```bash
pytest tests/development/
```

Or run directly:

```bash
python tests/development/test_codebase_context.py
```

The test suite covers session creation, repo ZIP upload, include extraction, structural verification, prompt budget enforcement, and static file integrity across 14 test groups.


# Serving & Deployment

* **API:** FastAPI + uvicorn on port 8081
* **LLM Inference:** llama.cpp `llama-server` on port 24000 (CUDA-accelerated)
* **Proxy:** LiteLLM on port 4000 (Anthropic-compatible API)
* **Container:** Docker + NVIDIA Container Toolkit

The container runs all three services inside a tmux session. To attach for debugging:

```bash
docker exec -it icd-c-code-refactorer tmux attach -t llama-server
```

### GPU Troubleshooting

If you hit VRAM errors, reduce context size or GPU layers:

```bash
docker run -ti --rm --network=host --gpus all \
  -e LLAMA_ARG_N_GPU_LAYERS=48 \
  -e LLAMA_ARG_CTX_SIZE=16384 \
  -e LLAMA_ARG_N_PREDICT=16384 \
  -v "$PWD/models":/home/developer/models \
  -v "$PWD/workspace":/home/developer/workspace \
  icd-c-code-refactorer:llama.cpp
```


# Support

* Open an issue on the repository
* Contact the maintainers directly


# Contributing

```bash
git clone <repo>
git checkout -b feature/your-feature
```

Kindly refer to [Git Guidelines](https://hal-confluence.edgegroup.ae/x/AwEwE)


# Authors and Acknowledgment

Built on top of [qwen-claude-code-sw2](https://github.com/zakaseb/qwen-claude-code-sw2) which provides the local LLM inference infrastructure (Docker + llama.cpp + LiteLLM).


# License

MIT License — see [LICENSE](LICENSE) for details.


# Project Status

`Active`
