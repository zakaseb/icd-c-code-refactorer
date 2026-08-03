---
title: Testing
description: Development tests on the agentic multi-agent branch
tags: [testing]
---

# Testing

Primary suite: `tests/development/` (as present on this branch).

| Test module | Focus |
|-------------|-------|
| `test_deliverables_live_download.py` | Mid-build sandbox→gen_dir sync for Download All |
| `test_team_consolidated_reports.py` | Per-team findings report → gen_dir + archive + Download ZIP |
| `test_agentic_pipeline.py` | Default-on `AGENTIC_PIPELINE`, opt-out, FakeLLM mission coverage |
| `test_sandbox_retry_ui.py` | Parsing/resolving `sandbox_retries`, UI/API wiring |
| `test_orchestrator_build_budget.py` | Finite budget hard-stop in orchestrator |
| `test_orchestrator_gitnexus_kwarg.py` | `run_orchestrator` / agentic accept `gitnexus_report` (no TypeError crash loop) |
| `test_orchestrator_action_parse.py` | Parse canonical + Claude `<tool_call>` / bare JSON action formats |
| `test_indefinite_sandbox_outer_rounds.py` | Indefinite budget = one unlimited campaign; crash loop terminates (no SSE flood) |
| `test_agentic_xilinx_profile.py` | Agentic Xilinx / superloop profile behaviour |
| `test_per_file_compile.py` | Per-file compile gate scoping and diagnostics |
| `test_llm_request_hardening.py` | Prompt clamp, overflow retry, error surfaces |
| `test_codebase_context.py` | Source-script / repo context builders |
| `test_gitnexus.py` | GitNexus-style embedded context report |
| `test_append_log_helper_py.py` / `test_append_log_helper.js` | Log helper batching |

## Running

```bash
python -m pytest tests/development/ -q
```

Mission-focused slice:

```bash
python -m pytest tests/development/test_agentic_pipeline.py -q
```

## What “good” looks like

- Default env → `/api/process` uses the agentic mission.
- `AGENTIC_PIPELINE=0` → classic sequential processing.
- Mission writes `agentic_pipeline/blackboard.json`.
- Each team writes `team_report_{stage}.md` (preview + Download All) and archives under `agentic_pipeline/team_reports/`.
- UI receives a live `gitnexus` stage on the agentic path.
- Per-file compile does not indiscriminately sweep `repo_dir`.
