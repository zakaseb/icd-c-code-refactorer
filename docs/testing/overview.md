---
title: Testing
description: Development tests covering pipeline-critical behaviour
tags: [testing]
---

# Testing

Primary suite: `tests/development/`.

| Test module | Focus |
|-------------|-------|
| `test_sandbox_retry_ui.py` | Parsing/resolving `sandbox_retries`, UI/API wiring |
| `test_orchestrator_build_budget.py` | Finite budget hard-stop in orchestrator |
| `test_per_file_compile.py` | Per-file compile gate scoping and diagnostics |
| `test_llm_request_hardening.py` | Prompt clamp, overflow retry, error surfaces |
| `test_codebase_context.py` | Source-script / repo context builders |
| `test_agentic_xilinx_profile.py` | Agentic Xilinx/superloop profile behaviour |
| `test_gitnexus.py` | GitNexus-style embedded context report |
| `test_append_log_helper_py.py` | Log helper batching behaviour |
| `test_hf_download.py` | Model cache lookup / env guards for Docker entrypoint `hf_download.py` |

## Running

From the repo root (with project venv / container Python as appropriate):

```bash
python -m pytest tests/development/ -q
```

Target a slice:

```bash
python -m pytest tests/development/test_sandbox_retry_ui.py tests/development/test_orchestrator_build_budget.py -q
```

## What “good” looks like for docs-related changes

- Finite `sandbox_retries=N` stops sandbox after ~N builds.
- Indefinite continues until success/cancel (subject to safety caps).
- Omitting the query preserves env defaults.
- Per-file compile does not sweep `repo_dir` indiscriminately.
