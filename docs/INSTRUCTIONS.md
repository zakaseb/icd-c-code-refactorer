# OpenWiki brief for icd-c-code-refactorer

Generate and maintain comprehensive repository documentation for human developers and coding agents.

## Branch accuracy (required)

Documentation under `docs/` (symlink `openwiki/` → `docs/`) **must describe the currently checked-out branch**, not another feature branch or an outdated mainline snapshot.

- Do not document APIs, UI controls, env vars, tests, or stages that do not exist in this tree.
- When code diverges from older docs, update or delete the stale claims in the same change.
- Prefer concrete file paths, route names, class/function names, and artefacts from this branch.

## Feature → docs (required)

**Every new feature or behaviour change must update OpenWiki in the same PR/commit set.**

| If you change… | Also update… |
|----------------|--------------|
| Pipeline / teams / stages | `docs/pipeline/stages.md`, `docs/agentic/multiagent-pipeline.md`, `docs/quickstart.md` |
| HTTP routes / SSE | `docs/api/surface.md` |
| Web UI | `docs/webui/integration.md` |
| Sandbox / orchestrator / agentic debug | `docs/sandbox/debugging.md`, `docs/agentic-debug-pipeline.md` if needed |
| Env / Docker / serve | `docs/building/configuration.md`, `docs/building/docker-deployment.md` |
| Tests | `docs/testing/overview.md` |
| Ops failure modes | `docs/operations/runbooks.md` |

Add new pages under `docs/` when a feature deserves a dedicated deep dive; link them from `docs/quickstart.md`.

## Content priorities (this branch)

- Agentic multi-agent mission (`api/agentic_pipeline/`): MissionController, six teams, blackboard, Claude Agent SDK → local LLM (`AGENTIC_PIPELINE=1` default)
- End-to-end stages: upload → analysis → gitnexus → transform → verification → compile → sandbox_build (with revisits)
- API surface, web UI, sandbox backends, Docker / llama-server / LiteLLM
- Preserve hand-authored deep dives (`agentic-debug-pipeline.md`, `agentic/multiagent-pipeline.md`); do not delete them

Write all wiki pages as Markdown under `docs/` (exposed as `/openwiki`).
