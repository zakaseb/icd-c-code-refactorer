---
title: Quick Start
description: Index for icd-c-code-refactorer OpenWiki documentation
tags: [quickstart, index]
---

# Quick Start — ICD C Code Refactorer

Transform C source between Interface Control Document (ICD) versions using a fully local LLM pipeline (FastAPI + llama-server + LiteLLM).

On **this branch**, Transform Code defaults to the **agentic multi-agent mission** (`AGENTIC_PIPELINE=1`): six teams under `MissionController` with blackboard feedback and revisits.

## Run the app

```bash
chmod +x deployments/docker/build.sh deployments/docker/entrypoint.sh scripts/serve_web.sh
mkdir -p models workspace
./deployments/docker/build.sh
./scripts/serve_web.sh
```

Open **http://localhost:8081**. First start downloads the GGUF model into `models/` (~49 GB).

## Documentation map

| Topic | Page |
|-------|------|
| **Agentic multi-agent pipeline (default)** | [agentic/multiagent-pipeline.md](agentic/multiagent-pipeline.md) |
| Architecture & session layout | [architecture/overview.md](architecture/overview.md) |
| End-to-end pipeline stages | [pipeline/stages.md](pipeline/stages.md) |
| HTTP / SSE API surface | [api/surface.md](api/surface.md) |
| Web UI ↔ backend | [webui/integration.md](webui/integration.md) |
| Sandbox orchestrator & agentic debug | [sandbox/debugging.md](sandbox/debugging.md) |
| Agentic debug deep dive (sandbox) | [agentic-debug-pipeline.md](agentic-debug-pipeline.md) |
| Docker & local LLM deployment | [building/docker-deployment.md](building/docker-deployment.md) |
| Configuration & environment variables | [building/configuration.md](building/configuration.md) |
| Testing | [testing/overview.md](testing/overview.md) |
| Operational runbooks | [operations/runbooks.md](operations/runbooks.md) |
| OpenWiki brief | [INSTRUCTIONS.md](INSTRUCTIONS.md) |

## One-minute mental model

1. **Upload** `.c`/`.h`, Source ICD PDF, Target ICD PDF (optional repo ZIP).
2. **Process** streams SSE stages via MissionController teams: `analysis` → `gitnexus` → `transform` → `verification` → `compile` → `sandbox_build` (with possible revisits).
3. **Review** generated code, optionally chat feedback, then **Re-generate**.
4. **Download** result ZIP and/or built repo ZIP.

Set `AGENTIC_PIPELINE=0` to force the classic sequential path. Sandbox attempt budgets are env-driven (`SANDBOX_ORCH_*` / `SANDBOX_AGENTIC_*`); there is no UI `sandbox_retries` control on this branch.
