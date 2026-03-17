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

```bash
./run_web.sh
```

Then open [http://localhost:8080](http://localhost:8080) in your browser.

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

## Derived From

This project is forked from [qwen-claude-code-sw2](https://github.com/zakaseb/qwen-claude-code-sw2), which provides the local LLM inference infrastructure (Docker + llama.cpp + LiteLLM).

## License

MIT License — see [LICENSE](LICENSE) for details.
