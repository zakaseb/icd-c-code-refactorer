# OpenWiki brief for icd-c-code-refactorer

Documentation under `docs/` (symlink `openwiki/` → `docs/`) **must describe the currently checked-out branch**, not another feature branch.

## Priorities

- End-to-end pipeline: ICD upload → analysis → code generation → verification → per-file compile → sandbox build
- API surface (`api/`), web UI (`api/static/`), orchestrator and agentic debug flows (`api/agentic_debug.py` when enabled)
- Docker / local LLM (llama-server, LiteLLM) deployment and how to run the web app
- Key modules, data flow, configuration, and operational runbooks
- Preserve existing docs (e.g. `agentic-debug-pipeline.md`); do not delete them

## Feature → docs

**Every new feature or behaviour change must update OpenWiki in the same PR/commit set.**

| Code area | Doc page |
|-----------|----------|
| Routes / SSE | `docs/api/surface.md` |
| Web UI | `docs/webui/integration.md` |
| Sandbox / orchestrator / agentic debug | `docs/sandbox/debugging.md`, `docs/agentic-debug-pipeline.md` |
| Env / Docker / serve scripts | `docs/building/configuration.md`, `docs/building/docker-deployment.md` |
| Tests | `docs/testing/overview.md` |
| Ops / failure modes | `docs/operations/runbooks.md` |

Write all wiki pages as Markdown under `docs/` (exposed as `/openwiki`).
