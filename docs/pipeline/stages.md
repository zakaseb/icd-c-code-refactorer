---
title: Pipeline Stages
description: ICD upload through sandbox build — stages, functions, artefacts
tags: [pipeline, stages]
---

# Pipeline Stages

Upload endpoints run **before** the SSE process stream.

On this branch, `GET /api/process/{session_id}` defaults to the **agentic mission** (`MissionController` + teams). Classic sequential `event_stream()` runs only when `AGENTIC_PIPELINE=0`. See [agentic/multiagent-pipeline.md](../agentic/multiagent-pipeline.md).

## Stage table (SSE names — both paths)

| Order | SSE `stage` | Agentic team (default) | Classic path | Key artefacts |
|------:|-------------|------------------------|--------------|---------------|
| 0 | *(upload)* | — | upload_* routes; PDF → text | `original_code/`, `source_icd.txt`, `target_icd.txt`, `repo_contents/` |
| 1 | `analysis` | `IngestionTeam` | ICD compare + distill in `app.py` | `change_spec.txt`, `change_spec_raw.txt`, `target_summary.txt` |
| 1b | `gitnexus` | `CodebaseTeam` | May be skipped / unwired on classic | `repo_knowledge.txt`, `gitnexus_report.txt` |
| 2 | `transform` | `GenerationTeam` | Per-file codegen + variants | `generated_code/*.c|.h` |
| 2b | `header_doc` | (folded into generation / classic-only) | Header documentation pass | Updated `.h` comments |
| 3 | `verification` | `VerificationTeam` | `_structural_verify` + LLM | `verification_report.txt` |
| 3.5 | `compile` | `CompilationTeam` | `run_per_file_compile` | `*.o`, `compile_report.txt` |
| 4 | `sandbox_build` | `IntegrationTeam` | `_sandbox_build_iterate` | `built_repo.zip`, `sandbox_build_log.txt` |

Agentic mode may **revisit** earlier stages when blackboard feedback has blockers (bounded by `AGENTIC_MAX_*`).

## Analysis details

- ICD text is chunked when large (`ICD_CHUNK_CHARS`, map-reduce path).
- Uploaded source scripts are preferred context over sweeping the whole repo directory.
- Distillation keeps factual tokens from the raw analysis so `change_spec` is shorter but not empty of requirements.
- Multiple peripheral variations in one Target ICD can yield separate generated headers.

## Transform & variants

- Each uploaded `.c`/`.h` is transformed against the change spec and available repo knowledge.
- When variants are detected, the pipeline can emit a distinct `.h` per variation instead of collapsing them into one ambiguous header.

## Verification & compile gate

- Structural verification catches brace imbalance, missing guards, unresolved includes, and missing expected symbols before the LLM pass.
- The per-file compile gate scopes to `generated_code` / code dir — it must not compile the entire uploaded repo tree.
- Failures produce actionable diagnostics that later sandbox agents can consume.

## Sandbox build

See [sandbox/debugging.md](../sandbox/debugging.md) for backend selection and retry budgets (`sandbox_retries`).

## Regeneration path

`GET /api/regenerate/{session_id}` re-runs generation using conversation feedback plus existing ICD/repo context. Typical SSE stages: regeneration/transform → verification → sandbox_build (full ICD analysis is not repeated). Same `sandbox_retries` query param as process.

## Resume / pause

- `POST /api/pause/{session_id}` and `POST /api/resume/{session_id}`
- Progress is tracked in session state (`completed_stages`, `completed_files`) and `pipeline_events.jsonl` so long overnight runs can continue after interruption.
