# OpenWiki brief for icd-c-code-refactorer

Generate comprehensive repository documentation for human developers and coding agents.

Priorities:
- End-to-end pipeline: ICD upload → analysis → code generation → verification → per-file compile → sandbox build
- API surface (`api/`), web UI (`api/static/`), orchestrator and agentic debug flows
- Docker / local LLM (llama-server, LiteLLM) deployment and how to run the web app
- Key modules, data flow, configuration, and operational runbooks
- Preserve existing docs (e.g. agentic-debug-pipeline.md); do not delete them

Write all wiki pages as Markdown under this documentation tree (the repository `docs/` folder exposed as `/openwiki`).
