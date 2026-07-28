---
title: Pipeline Stages
description: ICD upload through sandbox build — agentic teams vs classic
tags: [pipeline, stages]
---

# Pipeline Stages

Upload endpoints run **before** the SSE process stream.

Default on this branch: `GET /api/process/{session_id}` → **agentic mission** (`MissionController` + teams). Classic sequential path only when `AGENTIC_PIPELINE=0`. Details: [agentic/multiagent-pipeline.md](../agentic/multiagent-pipeline.md).

## Stage table

| Order | SSE `stage` | Agentic team (default) | Classic path | Key artefacts |
|------:|-------------|------------------------|--------------|---------------|
| 0 | *(upload)* | — | `upload_*` routes; PDF → text | `original_code/`, `source_icd.txt`, `target_icd.txt`, `repo_contents/` |
| 1 | `analysis` | `IngestionTeam` | ICD compare + distill in `app.py` | `change_spec.txt`, `change_spec_raw.txt`, `target_summary.txt`, `team_report_analysis.md` |
| 2 | `gitnexus` | `CodebaseTeam` | Optional / may be absent on classic | `repo_knowledge.txt`, `gitnexus_report.txt`, `team_report_gitnexus.md` |
| 3 | `transform` | `GenerationTeam` | Per-file codegen + variants | `generated_code/*.c|.h` (incl. per-variant headers), `team_report_transform.md` |
| 4 | `verification` | `VerificationTeam` | `_structural_verify` + LLM | `verification_report.txt`, `team_report_verification.md` |
| 5 | `compile` | `CompilationTeam` | `run_per_file_compile` | `*.o`, `compile_report.txt`, `team_report_compile.md` |
| 6 | `sandbox_build` | `IntegrationTeam` | `_sandbox_build_iterate` | `built_repo.zip`, `sandbox_build_log.txt`, `team_report_sandbox_build.md` |

In agentic mode each team also archives a copy under `agentic_pipeline/team_reports/{stage}_team_report.md` and emits `deliverables_updated` so Download All can pick up the report immediately. Agentic mode may **revisit** earlier stages when blackboard feedback has blockers (bounded by `AGENTIC_MAX_*`). There is no separate `header_doc` SSE stage on this branch; header work is part of GenerationTeam / classic transform.

## Analysis & generation

- ICD text may be chunked (`ICD_CHUNK_CHARS`, map-reduce).
- Uploaded source scripts are preferred context over sweeping the whole repo.
- Distillation + FactAuditor aim for high fact retention from `change_spec_raw` → `change_spec`.
- Peripheral variants can yield separate `*_Variant.h` files.

## Verification & compile

- Structural checks before / with LLM compliance.
- Per-file compile scopes to generated/code dirs — not a full repo sweep.
- Compile failures can feed blackboard feedback back to `transform`.

## Sandbox build

See [sandbox/debugging.md](../sandbox/debugging.md) for backend selection and `sandbox_retries` budgets.

## Regeneration

`GET /api/regenerate/{session_id}` — typically `regeneration` / `transform` → `verification` → `sandbox_build` (no full ICD re-analysis).

## Pause / resume

`POST /api/pause/{session_id}` and `POST /api/resume/{session_id}`; progress via session state / `pipeline_events.jsonl`.
