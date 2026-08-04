---
title: Pipeline Stages
description: ICD upload through sandbox build — stages, functions, artefacts
tags: [pipeline, stages]
---

# Pipeline Stages

Upload endpoints run **before** the SSE process stream. Processing is driven by `GET /api/process/{session_id}` → `event_stream()` in `api/app.py`.

## Stage table

| Order | SSE `stage` | What happens | Key artefacts |
|------:|-------------|--------------|---------------|
| 0 | *(upload)* | `upload_code`, `upload_source_icd`, `upload_target_icd`, optional `upload_repo_zip`; PDFs → text via `extract_pdf_text` | `original_code/`, `source_icd.txt`, `target_icd.txt`, `repo_contents/` |
| 1 | `analysis` | Compare Source vs Target ICD (direct or map-reduce `_call_llm_stream`); build repo/source-script context (`_build_repo_knowledge`, `_build_source_scripts_context`); distill `change_spec` from `change_spec_raw` | `change_spec.txt`, `change_spec_raw.txt`, `target_summary.txt`, `icd_analysis.txt`, optional `repo_knowledge.txt` |
| 2 | `transform` | Per-file LLM codegen; peripheral variant detection (`_detect_peripheral_variants`, `_generate_variant_header`); completeness checks (`_looks_complete_c_file`) | `generated_code/*.c|.h` |
| 2b | `header_doc` | ICD-grounded documentation pass for headers (`_build_header_documentation_prompt`) | Updated `.h` comments in `generated_code/` |
| 3 | `verification` | Structural checks (`_structural_verify`) + LLM verify/fix; variable inventory (`_extract_c_variables`) | `verification_report.txt`, in-place fixes |
| 3.5 | `compile` | `run_per_file_compile()` in `api/per_file_compile.py` — compile each `.c` to `.o` with project-local includes | `*.o`, `compile_report.txt` |
| 4 | `sandbox_build` | `_sandbox_build_iterate` (or `_remote_build_iterate` if remote). Chooses agentic / orchestrator / legacy | `sandbox/`, `built_repo.zip`, `sandbox_build_log.txt` |

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
