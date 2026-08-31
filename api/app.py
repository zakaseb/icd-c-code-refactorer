"""
ICD-Based C Code Refactorer — Transform C code between ICD versions.

Upload original .c/.h files, source ICD (PDF), and target ICD (PDF).
The tool analyzes both ICDs, understands the differences, and transforms
the code to conform to the target ICD.
"""
import difflib
import json
import logging
import os
import re
import time
import uuid
import shutil
import subprocess
import zipfile
import io
from datetime import datetime, timezone

import httpx

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
import fitz  # PyMuPDF
from pathlib import Path
from typing import Iterable, Iterator, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Query

try:
    from api.orchestrator import run_orchestrator as _run_orchestrator
except ImportError:  # tolerate flat layout (uvicorn app.app:app)
    from orchestrator import run_orchestrator as _run_orchestrator
try:
    from api.agentic_debug import run_agentic_debug as _run_agentic_debug
except ImportError:
    from agentic_debug import run_agentic_debug as _run_agentic_debug
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from per_file_compile import run_per_file_compile, _extract_per_file_blocks

app = FastAPI(title="ICD C Code Refactorer", docs_url=None, redoc_url=None)

STATIC_DIR = Path(__file__).parent / "static"
# /static is mounted at end of this module so /static/app.js can be overridden
# with a patched version (root-owned on-disk file cannot be edited in some envs).

WORKSPACE_DIR = Path(
    os.environ.get("WORKSPACE_DIR", str(Path(__file__).parent.parent / "workspace"))
)
SESSIONS_DIR = WORKSPACE_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------
LLM_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:24000")
LLM_BASE_URL_LITELLM = os.environ.get("LITELLM_BASE_URL", "http://127.0.0.1:4000")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-1234-miaw")
HF_MODEL = os.environ.get(
    "HF_MODEL", "Qwen3-Coder-Next-UD-Q4_K_XL.gguf"
)
MODEL_NAME = HF_MODEL.replace(".gguf", "") if HF_MODEL.endswith(".gguf") else HF_MODEL
MODEL_NAME_LITELLM = f"openai/{HF_MODEL}"

HTTPX_STREAM_TIMEOUT = httpx.Timeout(timeout=600.0, connect=60.0)

# Disable direct full-text compare by default on CPU runs.
# In practice the prompt prefill for very large full-ICD prompts can take a long
# time before the first streamed token appears. The chunked path yields analysis
# chunk-by-chunk so users see progress sooner.
DIRECT_COMPARE_MAX_CHARS = 0

# Fallback chunked map-reduce for very large ICDs.
ICD_CHUNK_CHARS = 3_000
MAX_CODE_CONTEXT_CHARS = 8_000
MAX_REPO_CONTEXT_CHARS = 15_000
# Byte cap for the "old scripts" (.c / .h) context injected into the ICD-delta
# analysis prompt. Larger than `MAX_REPO_CONTEXT_CHARS` because reasoning about
# the *Impact on C code* benefits from seeing concrete source, not just headers.
SOURCE_SCRIPTS_MAX_CHARS = 25_000

CHARS_PER_TOKEN = 3
CTX_SIZE_TOKENS = 49152
MAX_OUTPUT_TOKENS = 4096
MAX_INPUT_TOKENS = CTX_SIZE_TOKENS - MAX_OUTPUT_TOKENS
LLM_PROMPT_SAFETY_TOKENS = int(os.environ.get("LLM_PROMPT_SAFETY_TOKENS", "1024"))
LLM_MIN_OUTPUT_TOKENS = int(os.environ.get("LLM_MIN_OUTPUT_TOKENS", "768"))

# ---------------------------------------------------------------------------
# Compile-gate (step 3.5) fix-prompt budget
# ---------------------------------------------------------------------------
# The transform stage was fixed by dropping its repo-dependency and
# repo-knowledge sections (see `sec_repo` / `sec_knowledge` / `sec_code` in the
# transform loop, now hard-set to ""). The per-file compile gate never got the
# same treatment: it still builds both, at _assemble_prompt priorities 1 and 2,
# ahead of `change_spec` at 3. Together they consume the entire
# MAX_INPUT_TOKENS budget, which
#   (a) drops `change_spec` outright, and
#   (b) pushes prompt_tokens past CTX_SIZE_TOKENS - LLM_PROMPT_SAFETY_TOKENS,
#       so _clamp_chat_request cuts max_tokens 4096 -> ~3048.
# ~3048 output tokens cannot hold a full .c plus its .h, so every fix pass ends
# finish_reason == "length"; the continuation pass (see _call_llm_complete)
# only sees the last 2000 chars and replies with unfenced text, so
# _extract_per_file_blocks reports "no triple-fenced code blocks found" or
# "parser saw filenames none" — which is the observed 0/2 failure.
#
# 1 (default) = lean prompt: compile gate drops repo_ctx + repo_knowledge.
# 0           = previous behaviour, for an A/B comparison.
COMPILE_FIX_LEAN_PROMPT = os.environ.get(
    "COMPILE_FIX_LEAN_PROMPT", "1"
).strip().lower() not in ("0", "false", "no", "")

# Hard ceiling on the compile-gate fix prompt, independent of MAX_INPUT_TOKENS.
# Leaves room for a full MAX_OUTPUT_TOKENS reply so the clamp never fires.
COMPILE_FIX_MAX_INPUT_TOKENS = int(os.environ.get(
    "COMPILE_FIX_MAX_INPUT_TOKENS",
    str(CTX_SIZE_TOKENS - MAX_OUTPUT_TOKENS - LLM_PROMPT_SAFETY_TOKENS - 512),
))

# ---------------------------------------------------------------------------
# Compile-gate include path
# ---------------------------------------------------------------------------
# By default the compile gate resolves quote-includes on demand and never
# sweeps the uploaded repository, because a blind sweep can put a foreign
# cross-toolchain sysroot on the -I path and shadow libc headers such as
# `string.h` (the failure `_is_toxic_include_dir` guards against).
#
# Set COMPILE_INCLUDE_SWEEP_REPO=1 to add EVERY header-bearing directory under
# the uploaded ZIP to the -I list. Directories that look like a toolchain
# sysroot are still excluded — that filter is what keeps the sweep survivable,
# so it is deliberately not made optional.
COMPILE_INCLUDE_SWEEP_REPO = os.environ.get(
    "COMPILE_INCLUDE_SWEEP_REPO", "0"
).strip().lower() not in ("0", "false", "no", "")

# Safety valve: a huge vendor SDK can produce thousands of -I flags and blow
# past the shell ARG_MAX. 0 disables the cap.
COMPILE_INCLUDE_SWEEP_MAX_DIRS = int(os.environ.get(
    "COMPILE_INCLUDE_SWEEP_MAX_DIRS", "400"
))

SANDBOX_BUILD_TIMEOUT = 120
# Cap compiler/build output in SSE: large make/cmake logs on high-core hosts
# can be multi-MB and overwhelm browser JSON parsing and DOM if sent whole.
SANDBOX_SSE_MAX_BUILD_LOG_CHARS = int(
    os.environ.get("SANDBOX_SSE_MAX_BUILD_LOG_CHARS", "200000")
)
# Orchestrator agent settings — drives the iterative debugging loop.
SANDBOX_USE_ORCHESTRATOR = os.environ.get(
    "SANDBOX_USE_ORCHESTRATOR", "1"
).strip().lower() not in ("0", "false", "no", "off")
_orch_steps_raw = os.environ.get("SANDBOX_ORCH_MAX_STEPS", "0").strip()
SANDBOX_ORCH_MAX_STEPS: int | None = int(_orch_steps_raw) if _orch_steps_raw else 0
if SANDBOX_ORCH_MAX_STEPS is not None and SANDBOX_ORCH_MAX_STEPS <= 0:
    SANDBOX_ORCH_MAX_STEPS = None
SANDBOX_ORCH_MAX_BUILDS = int(os.environ.get("SANDBOX_ORCH_MAX_BUILDS", "25"))
SANDBOX_ORCH_OUTER_ROUNDS = int(
    os.environ.get("SANDBOX_ORCH_OUTER_ROUNDS", "4")
)
# Agentic-AI debug pipeline (state-machine, single-hypothesis-per-iteration).
# Default ON: takes precedence over SANDBOX_USE_ORCHESTRATOR. Set
# SANDBOX_USE_AGENTIC=0 to restore the ReAct orchestrator path.
SANDBOX_USE_AGENTIC = os.environ.get(
    "SANDBOX_USE_AGENTIC", "1"
).strip().lower() not in ("0", "false", "no", "off")
SANDBOX_AGENTIC_MAX_ATTEMPTS = int(
    os.environ.get("SANDBOX_AGENTIC_MAX_ATTEMPTS", "120")
)
SANDBOX_AGENTIC_NO_PROGRESS = int(
    os.environ.get("SANDBOX_AGENTIC_NO_PROGRESS", "3")
)
SANDBOX_AGENTIC_OSCILLATION = int(
    os.environ.get("SANDBOX_AGENTIC_OSCILLATION", "2")
)
SANDBOX_AGENTIC_EDIT_BUDGET = int(
    os.environ.get("SANDBOX_AGENTIC_EDIT_BUDGET", "60")
)
SANDBOX_AGENTIC_OUTER_ROUNDS = int(
    os.environ.get("SANDBOX_AGENTIC_OUTER_ROUNDS", "2")
)
# Sentinel: caller did not supply a UI / query retry budget — keep the
# classic env-driven outer-round × per-round budgets.
_SANDBOX_RETRIES_UNSET = object()
# Default finite budget offered by the UI when the user has not chosen
# "indefinite" (matches the orchestrator's historic per-round build cap).
SANDBOX_RETRIES_UI_DEFAULT = SANDBOX_ORCH_MAX_BUILDS


def _parse_sandbox_retries(raw: str | int | None) -> int | None:
    """Parse a user-supplied sandbox retry budget.

    Returns:
      * ``None``  — run indefinitely until the build succeeds
      * ``int``   — maximum number of build/fix reiterations (>= 1)

    Accepts ``"indefinite"`` / ``"inf"`` / ``"infinite"`` / ``"0"`` / ``"-1"``
    for the unlimited mode, or a positive integer string.
    """
    if raw is None:
        raise ValueError("sandbox_retries value is required")
    if isinstance(raw, int):
        if raw < 0:
            return None
        if raw == 0:
            return None
        return raw
    text = str(raw).strip().lower()
    if text in ("indefinite", "inf", "infinite", "unlimited", "0", "-1"):
        return None
    try:
        n = int(text)
    except ValueError as e:
        raise ValueError(
            f"sandbox_retries must be a positive integer or 'indefinite' "
            f"(got {raw!r})"
        ) from e
    if n < 0:
        return None
    if n == 0:
        return None
    return n


def _resolve_sandbox_max_retries(
    raw: str | None, status: dict | None = None,
) -> object:
    """Resolve the effective sandbox retry budget for a process/regenerate run.

    Priority: explicit query/raw value → persisted status → unset (env defaults).
    Return value is ``None`` (indefinite), a positive ``int``, or
    ``_SANDBOX_RETRIES_UNSET`` (preserve classic env-driven limits).
    """
    if raw is not None and str(raw).strip() != "":
        return _parse_sandbox_retries(raw)
    if status:
        mode = status.get("sandbox_retries_mode")
        if mode == "indefinite":
            return None
        stored = status.get("sandbox_max_retries")
        if isinstance(stored, int) and stored >= 1:
            return stored
        if mode == "finite" and stored is None:
            # Defensive: finite mode without a value → UI default
            return SANDBOX_RETRIES_UI_DEFAULT
    return _SANDBOX_RETRIES_UNSET


def _sandbox_retry_plan(sandbox_max_retries: object) -> dict:
    """Translate a retry budget into per-backend limits.

    When the budget is unset, mirror the classic env-driven behaviour
    (multiple outer rounds × per-round caps). When the user supplies a
    finite N, collapse to a single round with N builds/attempts. When
    indefinite, outer rounds and per-round caps are both unlimited.
    """
    if sandbox_max_retries is _SANDBOX_RETRIES_UNSET:
        return {
            "mode": "env",
            "label": (
                f"env defaults (orch rounds={SANDBOX_ORCH_OUTER_ROUNDS} "
                f"× builds={SANDBOX_ORCH_MAX_BUILDS}; agentic rounds="
                f"{SANDBOX_AGENTIC_OUTER_ROUNDS} × attempts="
                f"{SANDBOX_AGENTIC_MAX_ATTEMPTS})"
            ),
            "orch_outer": SANDBOX_ORCH_OUTER_ROUNDS,
            "orch_builds": SANDBOX_ORCH_MAX_BUILDS,
            "agentic_outer": SANDBOX_AGENTIC_OUTER_ROUNDS,
            "agentic_attempts": SANDBOX_AGENTIC_MAX_ATTEMPTS,
            "legacy_max": None,
        }
    if sandbox_max_retries is None:
        return {
            "mode": "indefinite",
            "label": "indefinite (until build succeeds)",
            "orch_outer": None,
            "orch_builds": None,
            "agentic_outer": None,
            "agentic_attempts": None,
            "legacy_max": None,
        }
    n = int(sandbox_max_retries)
    return {
        "mode": "finite",
        "label": f"{n} reiteration(s)",
        "orch_outer": 1,
        "orch_builds": n,
        "agentic_outer": 1,
        "agentic_attempts": n,
        "legacy_max": n,
    }
# Coalesce tiny llama-server deltas into fewer SSE messages (less JSON.parse +
# DOM pressure in the browser — important on unified-memory hosts).
SSE_UI_TOKEN_BATCH_MIN_CHARS = int(os.environ.get("SSE_UI_TOKEN_BATCH_MIN_CHARS", "4096"))
SSE_UI_TOKEN_BATCH_MAX_INTERVAL_S = float(
    os.environ.get("SSE_UI_TOKEN_BATCH_MAX_INTERVAL_S", "0.12")
)
SSE_UI_TOKEN_BATCH_MAX_BURST_CHARS = int(
    os.environ.get("SSE_UI_TOKEN_BATCH_MAX_BURST_CHARS", "98304")
)

# ---------------------------------------------------------------------------
# Remote build configuration (feature/remote-xilinx-build)
# ---------------------------------------------------------------------------
REMOTE_BUILD_ENABLED = os.environ.get("REMOTE_BUILD_ENABLED", "0").strip().lower() \
    not in ("0", "false", "no", "off")
REMOTE_BUILD_HOST    = os.environ.get("REMOTE_BUILD_HOST", "")
REMOTE_BUILD_USER    = os.environ.get("REMOTE_BUILD_USER", "")
REMOTE_BUILD_PASS    = os.environ.get("REMOTE_BUILD_PASS", "")
REMOTE_BUILD_SRC     = os.environ.get("REMOTE_BUILD_SRC", "")
REMOTE_BUILD_SCRIPT  = os.environ.get("REMOTE_BUILD_SCRIPT", "python build.py")
REMOTE_BUILD_MODE    = os.environ.get("REMOTE_BUILD_MODE", "debug")   # "debug" | "release"

# ---------------------------------------------------------------------------
# Peripheral-variation awareness
# ---------------------------------------------------------------------------
# A single Target ICD sometimes describes MULTIPLE variations (variants /
# configurations / modes / build options) of the SAME peripheral — e.g. the
# same logical device with different message layouts, field sets, scale
# factors, sample rates or register maps.  The pipeline must recognise this
# and (a) report each variation separately and (b) emit one clearly-commented
# struct (and matching .c code) per variation so an integrating engineer can
# keep only the variation they need.  These two reusable instruction blocks
# are threaded into the analysis/report prompts and the code-generation prompts
# respectively so the behaviour is consistent across every stage.

PERIPHERAL_VARIATION_ANALYSIS_GUIDANCE = (
    "\n\nPERIPHERAL VARIATIONS WITHIN A SINGLE ICD:\n"
    "A single Target ICD may describe MULTIPLE VARIATIONS (variants / "
    "configurations / modes / build or hardware options) of the SAME "
    "peripheral — for example different message layouts, field sets, scale "
    "factors, sample rates, or register maps for one logical device.\n"
    "- Determine whether the Target ICD defines more than one variation of a "
    "peripheral. State EXPLICITLY how many variations exist and name each one "
    "using the ICD's own terminology.\n"
    "- If multiple variations exist, organise the change specification PER "
    "VARIATION: give each variation its own clearly-labelled subsection and, "
    "for each, report the same categories of detail you normally would "
    "(structs/fields/types/sizes/alignment, enums/constants/message IDs, "
    "function signatures, behaviour/protocol/timing). Use bullet points or a "
    "comparison table so the per-variation differences are easy to read.\n"
    "- Call out the SHARED parts (common to every variation) separately from "
    "the variation-specific parts.\n"
    "- If only ONE variation exists, report normally and do NOT invent "
    "variations."
)

PERIPHERAL_VARIATION_CODEGEN_GUIDANCE = (
    "\n\nPERIPHERAL VARIATIONS (MULTIPLE VARIANTS IN ONE ICD):\n"
    "When the change specification / Target ICD describes MORE THAN ONE "
    "variation (variant / configuration / mode) of the same peripheral, you "
    "MUST represent EACH variation separately instead of merging them or "
    "silently picking one:\n"
    "- In headers (.h): emit a SEPARATE, self-contained .h FILE for EACH "
    "variation rather than one combined header. Give each file a distinct, "
    "descriptive name derived from the original header and the ICD variation "
    "name (e.g. `<Peripheral>_<Variation>.h`). Each file must have its own "
    "include guard, its own #includes, a clear banner comment naming the "
    "variation and what makes it different, and only that variation's struct / "
    "enums / constants / macros — so the integrating engineer simply keeps the "
    "one header file for the variation they need and deletes the others.\n"
    "- In sources (.c): provide the corresponding per-variation code within "
    "the file (separate, clearly-commented functions / definitions per "
    "variation). Do NOT collapse the variations into a single implementation.\n"
    "- Keep logic that is common to all variations shared and clearly marked "
    "as common.\n"
    "- If only ONE variation is present, generate normally (a single header), "
    "exactly as before."
)

# ---------------------------------------------------------------------------
# Header Documentation System Prompt
# ---------------------------------------------------------------------------
HEADER_DOCUMENTATION_SYSTEM_PROMPT = (
    "You are an expert embedded C developer specialising in ICD-driven firmware. "
    "Your ONLY task is to add documentation comments to an existing C header file. "
    "The header was generated by an automated tool and already conforms to the "
    "target ICD. Your job is purely additive — you enrich the file with comments; "
    "you do NOT restructure, reorder, rename, or retype anything.\n\n"

    "## ABSOLUTE CONSTRAINTS — violation is a critical failure\n"
    "1. Every `typedef struct`, `typedef enum`, field name, field type, array size, "
    "   `#define` name, `#define` value, include guard, and `#include` directive in "
    "   the input MUST appear in the output UNCHANGED. Do NOT rename, retype, reorder, "
    "   add, or remove any declaration.\n"
    "2. Do NOT add any new `#include` lines.\n"
    "3. Do NOT change any numeric values — scale factors, limits, macro values, "
    "   array dimensions — even if you believe the ICD implies a different value.\n"
    "4. Do NOT collapse struct fields into bitfields or unions unless they already "
    "   appear that way in the input.\n"
    "5. Output the COMPLETE file. Do not truncate, summarise, or use '// ...' or "
    "   '/* ... */' as placeholders for omitted content.\n"
    "6. Wrap the entire output in a single ```c ... ``` fenced block. "
    "   No prose before or after the fence.\n\n"

    "## What you MUST add\n\n"

    "### A. File header block\n"
    "Replace or insert a file-level block comment at the very top (before the "
    "include guard) using this exact layout:\n"
    "```c\n"
    "/* ============================================================\n"
    " * File:    <filename>.h\n"
    " * Module:  <MODULE_NAME in SCREAMING_SNAKE_CASE>\n"
    " * ICD Ref: <ICD document name and version from the change specification>\n"
    " * Brief:   <one-line description of what this module implements>\n"
    " * Author:  Auto-generated from ICD\n"
    " * ============================================================ */\n"
    "```\n\n"

    "### B. Struct and enum block comments\n"
    "Before each `typedef struct` or `typedef enum`, insert a block comment:\n"
    "```c\n"
    "/**\n"
    " * @brief   <what this message/packet/enum represents>\n"
    " * @details <ICD message name, ID if present, and purpose — one to three sentences>\n"
    " * @note    Total size: <N> bytes   (computed from field types; omit if unknown)\n"
    " */\n"
    "```\n\n"

    "### C. Inline field comments\n"
    "Every struct field and every enum member MUST have an inline `/* ... */` comment "
    "on the same line. Each comment must include ALL of the following that are "
    "defined in the ICD for that field — omit a sub-item only if the ICD truly does "
    "not specify it:\n"
    "  - Human-readable description (what the field represents)\n"
    "  - Unit  (e.g. `unit: deg/s`, `unit: m/s²`, `unit: ms`)\n"
    "  - Range (e.g. `range: [-32768, 32767]`)\n"
    "  - LSB weight / scale factor / conversion formula "
    "    (e.g. `conv: raw * 0.01 = deg/s`)\n"
    "  - Resolution (e.g. `res: 0.01 deg/s`)\n"
    "  - Special values (e.g. `0xFF = not available`)\n"
    "Example:\n"
    "```c\n"
    "    int16_t  AngularRateX;  /* Angular rate X-axis | unit: deg/s | "
    "range: [-400, 400] | conv: raw * 0.01220703125 = deg/s | res: 0.012 deg/s */\n"
    "```\n\n"

    "### D. Macro comments\n"
    "Every `#define` MUST have an inline `/* ... */` comment that states:\n"
    "  - What the constant represents\n"
    "  - Its unit or meaning (e.g. `/* Maximum payload length in bytes */`)\n\n"

    "### E. Closing size comment on structs\n"
    "After the closing brace of every `typedef struct`, append a size comment:\n"
    "```c\n"
    "} sMyStruct;  /* Total: <N> bytes */\n"
    "```\n"
    "Compute N by summing the byte sizes of all fields. If a field is a nested "
    "struct or array, expand accordingly. If you cannot determine the size with "
    "certainty, write `/* Total: ? bytes — verify against sizeof() */`.\n\n"

    "## Quality self-check — apply before outputting\n"
    "1. Is every declaration from the input present and byte-for-byte identical "
    "   in the output?\n"
    "2. Does every struct field have an inline comment with description, unit, "
    "   range, and conversion (where the ICD provides them)?\n"
    "3. Does every `#define` have an inline comment?\n"
    "4. Does every `typedef struct` / `typedef enum` have a `@brief` block comment?\n"
    "5. Is the file-header block present at the top?\n"
    "6. Are there zero new `#include` lines?\n"
    "7. Is the output complete — no truncation, no placeholder comments?\n"
    "If any check fails, fix it before outputting."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _strip_c_comments(code: str) -> str:
    result = []
    i = 0
    n = len(code)
    in_string = False
    in_char   = False
    while i < n:
        c = code[i]
        if in_string:
            result.append(c)
            if c == '\\' and i + 1 < n:
                i += 1
                result.append(code[i])
            elif c == '"':
                in_string = False
            i += 1
            continue
        if in_char:
            result.append(c)
            if c == '\\' and i + 1 < n:
                i += 1
                result.append(code[i])
            elif c == "'":
                in_char = False
            i += 1
            continue
        if c == '"':
            in_string = True
            result.append(c)
            i += 1
            continue
        if c == "'":
            in_char = True
            result.append(c)
            i += 1
            continue
        if c == '/' and i + 1 < n and code[i + 1] == '*':
            i += 2
            while i < n - 1:
                if code[i] == '*' and code[i + 1] == '/':
                    i += 2
                    break
                i += 1
            else:
                i = n
            continue
        if c == '/' and i + 1 < n and code[i + 1] == '/':
            i += 2
            while i < n and code[i] != '\n':
                i += 1
            continue
        result.append(c)
        i += 1
    return re.sub(r'\n{3,}', '\n\n', ''.join(result))

def _tail_truncate_for_sse(text: str, max_chars: int, label: str) -> str:
    """Return the *end* of *text*, suitable for error-heavy compiler output.

    Differing from :func:`_truncate_text` (head-only), the UI needs the
    final diagnostics when logs are huge.
    """
    if len(text) <= max_chars:
        return text
    log.warning(
        "%s: SSE build log uses tail only (%d chars, showing last %d)",
        label, len(text), max_chars,
    )
    head = f"[… {len(text) - max_chars} leading characters omitted …]\n\n"
    return head + text[-(max(0, max_chars - len(head))):]


def _truncate_text(text: str, max_chars: int, label: str) -> str:
    if len(text) <= max_chars:
        return text
    log.warning("%s truncated from %d to %d chars", label, len(text), max_chars)
    return text[:max_chars] + f"\n\n[... TRUNCATED {label} ...]"


def _split_text_chunks(text: str, max_chars: int) -> list[str]:
    """Split large ICD text into bounded chunks, preserving paragraph boundaries."""
    blocks = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for block in blocks:
        candidate = (current + "\n\n" + block).strip() if current else block
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(block) <= max_chars:
            current = block
        else:
            # Hard split exceptionally large block.
            for i in range(0, len(block), max_chars):
                chunks.append(block[i:i + max_chars])
    if current:
        chunks.append(current)
    return chunks or [text]


def _call_llm_text(
    system_prompt: str, user_prompt: str, max_tokens: int = 1024,
    meta: dict | None = None,
    on_chunk=None,
) -> str:
    parts: list[str] = []
    for piece in _call_llm_stream(system_prompt, user_prompt,
                                  max_tokens=max_tokens, meta=meta):
        parts.append(piece)
        if on_chunk is not None:
            on_chunk(piece)
    return "".join(parts).strip()


def _call_llm_complete(
    system_prompt: str, user_prompt: str,
    max_tokens: int = 4096, max_passes: int = 4,
    on_chunk=None,
) -> str:
    """Call LLM and automatically continue if the response is truncated
    (``finish_reason == "length"``).  Returns the concatenated full text."""
    result = ""
    for attempt in range(max_passes):
        meta: dict = {}
        if attempt == 0:
            prompt = user_prompt
        else:
            tail = result[-2000:] if len(result) > 2000 else result
            prompt = (
                "Your previous response was cut off due to length limits. "
                "Here is the end of what you wrote:\n\n"
                f"---\n{tail}\n---\n\n"
                "Continue EXACTLY from where you left off. "
                "Do not repeat already-written content."
            )
        text = _call_llm_text(system_prompt, prompt,
                              max_tokens=max_tokens, meta=meta, on_chunk=on_chunk)
        result = (result + "\n" + text).strip() if result else text
        if meta.get("finish_reason") != "length":
            break
        log.info("_call_llm_complete: pass %d/%d truncated, continuing…",
                 attempt + 1, max_passes)
    return result


def _looks_complete_c_file(generated: str, original: str, filename: str) -> bool:
    text = generated.strip()
    if not text:
        return False
    if "[... TRUNCATED" in text or text.endswith("..."):
        return False
    if text.count("{") != text.count("}"):
        return False
    if text.count("/*") > text.count("*/"):
        return False
    if filename.endswith(".h"):
        has_ifndef = re.search(r"^\s*#\s*ifn?def\b", text, flags=re.MULTILINE)
        has_endif = re.search(r"^\s*#\s*endif\b", text, flags=re.MULTILINE)
        if has_ifndef and not has_endif:
            return False
    min_len = max(120, int(len(original.strip()) * 0.35))
    if len(text) < min_len:
        return False
    last = text.splitlines()[-1].strip()
    if not (
        last.endswith("}")
        or last.endswith(";")
        or last.endswith("*/")
        or last.startswith("#endif")
    ):
        return False
    return True


def _looks_complete_header(code: str) -> bool:
    """Lightweight completeness check for a SINGLE generated header file.

    Used when a .h transform is split into one file per peripheral
    variation: each variation file is naturally smaller than the original
    combined header, so the size-vs-original ratio used by
    :func:`_looks_complete_c_file` would wrongly reject valid per-variation
    headers. This check only validates self-consistency (balanced braces /
    comments, include guard, sane ending).
    """
    text = code.strip()
    if len(text) < 60:
        return False
    if "[... TRUNCATED" in text or text.endswith("..."):
        return False
    if text.count("{") != text.count("}"):
        return False
    if text.count("/*") > text.count("*/"):
        return False
    has_ifndef = re.search(r"^\s*#\s*ifn?def\b", text, flags=re.MULTILINE)
    has_endif = re.search(r"^\s*#\s*endif\b", text, flags=re.MULTILINE)
    if has_ifndef and not has_endif:
        return False
    last = text.splitlines()[-1].strip()
    if not (
        last.endswith("}")
        or last.endswith(";")
        or last.endswith("*/")
        or last.startswith("#endif")
        or last.endswith("#endif")
    ):
        return False
    return True


def _split_variation_header_files(
    raw_output: str, base_fname: str,
) -> dict[str, str]:
    """Extract per-variation header files from a .h transform output.

    When a target ICD describes multiple variations of the same
    peripheral, the model is asked to emit a SEPARATE, self-contained .h
    file per variation, each introduced by a ``### <filename>.h`` marker
    followed by a ```c fenced block. This returns ``{filename: code}`` for
    every such header.

    Returns an empty dict for the normal single-header case (fewer than two
    distinct .h files), so the caller falls back to the existing
    single-file write path. Only applies to ``.h`` inputs; ``.c`` files are
    never split.
    """
    if not base_fname.lower().endswith(".h"):
        return {}
    blocks = _extract_per_file_blocks(raw_output)
    headers = {
        name: body
        for name, body in blocks.items()
        if name.lower().endswith(".h") and body.strip()
    }
    if len(headers) >= 2:
        return headers
    return {}


def _variant_slug(name: str) -> str:
    """Turn a human variant name into a filename-safe CamelCase fragment.

    e.g. ``"High-Rate"`` -> ``"HighRate"``, ``"Low power mode"`` ->
    ``"LowPowerMode"``. Returns ``""`` if nothing usable remains.
    """
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", name or "") if p]
    if not parts:
        return ""
    slug = "".join(p[:1].upper() + p[1:] for p in parts)
    if slug and slug[0].isdigit():
        slug = "V" + slug
    return slug


def _extract_json_object(text: str) -> dict | None:
    """Best-effort parse of the first JSON object in *text*.

    Tolerates ```json fences and surrounding prose by scanning for the
    first balanced ``{ ... }`` span. Returns ``None`` on failure.
    """
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
    candidate = m.group(1) if m else text
    start = candidate.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(candidate)):
        ch = candidate[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(candidate[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    break
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# Regex for extracting "technical fact tokens" from a change-spec text.
# Used by the distillation validator to confirm that the distilled output
# preserves the substantive facts present in the raw spec (hex constants,
# numeric literals, multi-part identifiers, common embedded units). Tokens
# that appear in BOTH raw and distilled are evidence of fact preservation.
_FACT_TOKEN_RE = re.compile(
    r"""
      0[xX][0-9A-Fa-f]+                       # hex constants (0xEB90)
    | \b\d+(?:\.\d+)?\b                       # numeric literals (1000, 0.001)
    | \b[A-Za-z]\w*_\w+\b                     # any identifier containing '_'
                                              # (uint32_t, acc_x, msg_id,
                                              #  ImuHeader_t, High_Rate,
                                              #  IMU_BAUD_RATE)
    | \b[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+\b      # CamelCase ≥2 words (no '_')
    | \b[A-Z]{2,}\b                           # ACRONYMS
    | \b(?:Hz|kHz|MHz|GHz|us|ms|ns|dps|rps|bps|kbps|Mbps|MB|KB|GB)\b
    """,
    re.VERBOSE,
)


def _extract_fact_tokens(text: str) -> set[str]:
    """Extract a set of distinctive "fact tokens" from *text*.

    Returns the set of substantive tokens (hex / numeric constants, multi-part
    identifiers, common embedded units) that a faithful distillation must
    preserve. Single-letter and 1-char tokens are excluded as noise.
    """
    if not text:
        return set()
    return {
        m.group(0)
        for m in _FACT_TOKEN_RE.finditer(text)
        if len(m.group(0)) >= 2
    }


def _distillation_fact_check(
    raw: str, distilled: str, min_ratio: float = 0.80,
) -> tuple[bool, list[str], float]:
    """Verify that *distilled* preserves most fact tokens from *raw*.

    Returns ``(ok, missing_tokens_sorted, ratio_kept)``. ``ok`` is ``True``
    when the kept-ratio meets ``min_ratio``. Missing tokens are sorted for
    deterministic retry prompts.
    """
    raw_facts = _extract_fact_tokens(raw)
    if not raw_facts:
        return True, [], 1.0
    distilled_facts = _extract_fact_tokens(distilled)
    missing = raw_facts - distilled_facts
    kept = (len(raw_facts) - len(missing)) / len(raw_facts)
    return kept >= min_ratio, sorted(missing), kept


def _detect_peripheral_variants(
    original: str, fname: str, change_spec: str, target_summary: str,
) -> list[str]:
    """Detect distinct variants of the peripheral implemented by *fname*.

    Returns the list of variant names (using the ICD's own terminology) when
    the Target ICD defines TWO OR MORE variants of the same peripheral, else
    an empty list (single-variant / normal case). Names are de-duplicated by
    slug. Any failure degrades gracefully to ``[]``.
    """
    if not fname.lower().endswith(".h"):
        return []
    detect_system = (
        "You analyse ICD change specifications to decide whether a given C "
        "header's peripheral is defined in MULTIPLE distinct variants "
        "(variants / configurations / modes / hardware or build options) in "
        "the Target ICD. You respond with STRICT JSON only, no prose."
    )
    detect_prompt = (
        f"Header file: {fname}\n\n"
        f"## Original header (pre-change)\n```c\n{original[:4000]}\n```\n\n"
        f"## Change specification (Source ICD -> Target ICD)\n{change_spec}\n\n"
        f"## Target ICD summary\n{target_summary[:6000]}\n\n"
        "Question: does the Target ICD define MORE THAN ONE distinct variant "
        "of the peripheral implemented by THIS header? Count only true "
        "variants of the SAME peripheral (e.g. different message layouts, "
        "field sets, scale factors, sample rates, or register maps for one "
        "logical device) — NOT unrelated peripherals, and NOT mere version "
        "bumps of a single layout.\n"
        "Respond with STRICT JSON exactly in this shape:\n"
        '{"has_variants": true, "variants": ["<Name1>", "<Name2>"]}\n'
        "Use the ICD's own names for each variant. If there is only one "
        'variant, respond {"has_variants": false, "variants": []}.'
    )
    try:
        out = _call_llm_complete(
            detect_system, detect_prompt, max_tokens=512, max_passes=1,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Variant detection for %s failed: %s", fname, e)
        return []
    data = _extract_json_object(out)
    if not data or not data.get("has_variants"):
        return []
    raw_names = data.get("variants") or []
    if not isinstance(raw_names, list):
        return []
    seen: set[str] = set()
    uniq: list[str] = []
    for n in raw_names:
        name = str(n).strip()
        slug = _variant_slug(name)
        if name and slug and slug.lower() not in seen:
            seen.add(slug.lower())
            uniq.append(name)
    return uniq if len(uniq) >= 2 else []


def _generate_variant_header(
    *,
    base_sections: list[tuple[str, str, int]],
    transform_system: str,
    base_header_name: str,
    original: str,
    variant_name: str,
    all_variants: list[str],
    header_filename: str,
) -> str:
    """Generate ONE complete, self-contained .h for a single peripheral variant.

    Reuses the shared prompt *base_sections* (change spec, repo context, etc.)
    and adds a variant-scoped file instruction. Returns the cleaned header
    code (possibly empty / incomplete; the caller validates with
    :func:`_looks_complete_header`).
    """
    guard = re.sub(r"[^A-Za-z0-9]", "_", header_filename).upper()
    others = [v for v in all_variants if v != variant_name]
    sec_file = (
        f"## File to Transform: {base_header_name} -> {header_filename} "
        f"(VARIANT: {variant_name})\n\n```c\n{original}\n```\n\n"
        f"The Target ICD defines multiple variants of this peripheral: "
        f"{', '.join(all_variants)}.\n"
        f"Generate the header for ONLY the '{variant_name}' variant. Output a "
        f"SINGLE complete, self-contained C header file (to be saved as "
        f"{header_filename}) that conforms to the Target ICD for this variant:\n"
        f"- Use include guard {guard}.\n"
        f"- Start with a banner comment naming the '{variant_name}' variant and "
        f"what makes it distinct.\n"
        f"- Define ONLY this variant's structs / enums / constants / macros, "
        f"plus any definitions shared by ALL variants. Do NOT define the other "
        f"variants ({', '.join(others) if others else 'none'}).\n"
        f"- Use the exact repository type names, function signatures, and "
        f"#include paths.\n"
        "Output ONLY the header code in ```c fences. Do not summarise or "
        "truncate; include the full ending (#endif)."
    )
    sections = list(base_sections) + [("file_to_transform", sec_file, 0)]
    system_tokens = _estimate_tokens(transform_system)
    prompt = _assemble_prompt(
        sections, max_input_tokens=MAX_INPUT_TOKENS - system_tokens,
    )
    clean = ""
    for attempt in range(1, 3):
        attempt_prompt = prompt
        if attempt > 1 and clean:
            attempt_prompt = (
                f"{prompt}\n\n"
                "The previous output was incomplete/truncated. Continue from "
                "the exact point where it stopped, output ONLY the missing "
                "remainder, and ensure all braces/comments are closed and the "
                "file ends with #endif.\n\n"
                f"Previous partial output:\n```c\n{clean}\n```"
            )
        try:
            out = _call_llm_complete(
                transform_system, attempt_prompt, max_tokens=4096, max_passes=2,
            )
        except Exception as e:  # noqa: BLE001
            log.warning(
                "Variant header generation for %s (%s) failed: %s",
                header_filename, variant_name, e,
            )
            break
        chunk = _extract_fenced(out, "c").strip()
        clean = chunk if attempt == 1 else (
            clean.rstrip() + "\n" + chunk.lstrip()
        ).strip()
        if _looks_complete_header(clean):
            break
    return clean


# Encodings seen in real embedded-C sources. Order matters: utf-8-sig also
# accepts plain UTF-8 (and strips a BOM if present); cp1252 is the usual source
# of 0x91-0x97 smart quotes / en-dashes emitted by Windows editors; latin-1
# never fails, so it is the terminal fallback.
TEXT_DECODE_CANDIDATES = ("utf-8-sig", "cp1252", "latin-1")


def _decode_bytes(data: bytes) -> tuple[str, str]:
    """Decode source bytes to text, returning ``(text, encoding_used)``.

    Never raises. Embedded C files routinely carry Windows-1252 punctuation in
    comment banners, and a UnicodeDecodeError raised inside the transform
    stage kills the SSE generator *without* emitting an ``error`` event, so
    the stream simply goes silent and the UI appears to hang.
    """
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16"), "utf-16"
        except UnicodeDecodeError:
            pass
    for enc in TEXT_DECODE_CANDIDATES:
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace"), "utf-8/replace"


def _read_text_safe(path: Path) -> str:
    """Encoding-tolerant replacement for ``Path.read_text()``."""
    return _decode_bytes(path.read_bytes())[0]


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    parts: list[str] = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            parts.append(f"--- Page {page_num + 1} ---\n{text}")
    doc.close()
    return "\n\n".join(parts)


_CONTEXT_OVERFLOW_MARKERS = (
    "exceeds the available context",
    "exceed the available context",
    "context size",
    "n_ctx",
    "context window",
    "input is too large",
    "prompt is too long",
    "tokens, exceed",
)


def _is_context_overflow_error(status_code: int, body_text: str) -> bool:
    """Heuristically detect a llama-server context-window 4xx error."""
    if status_code != 400:
        return False
    low = (body_text or "").lower()
    return any(marker in low for marker in _CONTEXT_OVERFLOW_MARKERS)


def _tail_truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Trim *text* head-down so its estimated token count fits *max_tokens*.

    Used when the user prompt alone exceeds the per-request budget; we
    keep the *end* of the prompt because the most actionable instructions
    and the file-to-transform live near the tail of our assembled
    sections.
    """
    if max_tokens <= 0:
        return ""
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    head_marker = (
        f"[... {len(text) - max_chars} leading characters truncated to fit "
        f"llama-server context window ...]\n\n"
    )
    keep = max(0, max_chars - len(head_marker))
    return head_marker + text[-keep:]


def _clamp_chat_request(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> tuple[str, str, int, bool]:
    """Pre-flight clamp a chat request to fit the llama-server context.

    Returns ``(system_prompt, user_prompt, max_tokens, was_clamped)``.

    Strategy (in order):

    1. If everything fits with safety margin, return unchanged.
    2. Otherwise, reduce ``max_tokens`` toward ``LLM_MIN_OUTPUT_TOKENS``
       to free room for the prompt.
    3. If the prompt *still* doesn't fit, tail-truncate the user prompt
       (the system prompt is small and stable, so we keep it intact).
    """
    sys_tokens = _estimate_tokens(system_prompt)
    user_tokens = _estimate_tokens(user_prompt)
    available = max(0, CTX_SIZE_TOKENS - LLM_PROMPT_SAFETY_TOKENS - sys_tokens)
    target_user_tokens = max(0, available - max_tokens)

    if user_tokens <= target_user_tokens:
        return system_prompt, user_prompt, max_tokens, False

    overflow = (sys_tokens + user_tokens + max_tokens
                + LLM_PROMPT_SAFETY_TOKENS) - CTX_SIZE_TOKENS

    new_max_tokens = max_tokens
    if overflow > 0:
        new_max_tokens = max(
            LLM_MIN_OUTPUT_TOKENS, max_tokens - overflow,
        )

    user_budget = max(
        0, CTX_SIZE_TOKENS - LLM_PROMPT_SAFETY_TOKENS - sys_tokens - new_max_tokens,
    )
    if user_tokens > user_budget:
        user_prompt = _tail_truncate_to_tokens(user_prompt, user_budget)
        log.warning(
            "Clamping LLM request: tail-truncated user prompt %d -> ~%d tokens "
            "(sys=%d, max_tokens=%d, ctx=%d).",
            user_tokens, user_budget, sys_tokens, new_max_tokens,
            CTX_SIZE_TOKENS,
        )
    if new_max_tokens != max_tokens:
        log.warning(
            "Clamping LLM request: max_tokens %d -> %d to fit context.",
            max_tokens, new_max_tokens,
        )
    return system_prompt, user_prompt, new_max_tokens, True


def _call_llm_stream(
    system_prompt: str, user_prompt: str, max_tokens: int = 16384,
    meta: dict | None = None,
):
    """Yield text chunks via SSE streaming from LLM.

    Uses direct llama-server to avoid proxy-side stalls.
    If *meta* dict is provided, ``meta["finish_reason"]`` is set to the
    finish_reason reported by the last SSE chunk (e.g. ``"stop"`` or
    ``"length"``).

    Defensive behavior:
    * The full response body is read on any non-2xx status so the actual
      llama-server error message (e.g. context-window overflow) surfaces
      to the caller / SSE client instead of a generic ``Client error
      '400 Bad Request'`` from httpx.
    * The request is pre-flight clamped against ``CTX_SIZE_TOKENS`` so
      that ``prompt_tokens + max_tokens`` cannot exceed the configured
      ``LLAMA_ARG_CTX_SIZE`` (with a safety margin).  When clamping is
      not enough, the user prompt is tail-truncated.
    * On a context-overflow 400 (rare race when our token estimate is
      wrong), we shrink ``max_tokens`` and the user prompt and retry
      once before failing.
    """
    url = f"{LLM_BASE_URL.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    sys_p, usr_p, eff_max_tokens, _ = _clamp_chat_request(
        system_prompt, user_prompt, max_tokens,
    )

    yielded = False
    finish_reason_last: str | None = None
    last_err: Exception | None = None
    last_400_body: str | None = None
    max_attempts = 4
    overflow_retries = 0
    max_overflow_retries = 2

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p},
            ],
            "max_tokens": eff_max_tokens,
            "temperature": 0.2,
            "stream": True,
        }
        try:
            with httpx.Client(timeout=HTTPX_STREAM_TIMEOUT) as client:
                with client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code >= 400:
                        try:
                            body_bytes = resp.read()
                        except Exception:
                            body_bytes = b""
                        body_text = body_bytes.decode("utf-8", errors="replace")
                        if (
                            _is_context_overflow_error(resp.status_code, body_text)
                            and overflow_retries < max_overflow_retries
                        ):
                            overflow_retries += 1
                            new_max = max(
                                LLM_MIN_OUTPUT_TOKENS, eff_max_tokens // 2,
                            )
                            new_user_budget = max(
                                0,
                                CTX_SIZE_TOKENS
                                - LLM_PROMPT_SAFETY_TOKENS * 2
                                - _estimate_tokens(sys_p)
                                - new_max,
                            )
                            usr_p = _tail_truncate_to_tokens(usr_p, new_user_budget)
                            log.warning(
                                "llama-server returned context-overflow 400 "
                                "(retry %d/%d): shrinking max_tokens %d -> %d "
                                "and tail-truncating user prompt to ~%d tokens.",
                                overflow_retries, max_overflow_retries,
                                eff_max_tokens, new_max, new_user_budget,
                            )
                            eff_max_tokens = new_max
                            attempt -= 1
                            continue
                        last_400_body = body_text
                        snippet = body_text.strip().splitlines()[:5]
                        snippet_joined = " | ".join(s.strip() for s in snippet if s.strip())
                        if not snippet_joined:
                            snippet_joined = body_text[:500]
                        raise RuntimeError(
                            f"llama-server HTTP {resp.status_code}: {snippet_joined[:1500]}"
                        )
                    for line in resp.iter_lines():
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                choice = data.get("choices", [{}])[0]
                                delta = choice.get("delta", {})
                                part = delta.get("content", "")
                                fr = choice.get("finish_reason")
                                if fr:
                                    finish_reason_last = fr
                                if part:
                                    yielded = True
                                    yield part
                            except (json.JSONDecodeError, KeyError):
                                pass
            break
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError) as e:
            last_err = e
            if attempt == max_attempts:
                break
            backoff = min(2 ** (attempt - 1), 6)
            log.warning(
                "LLM stream connect attempt %d/%d failed (%s). Retrying in %ss...",
                attempt, max_attempts, e, backoff,
            )
            time.sleep(backoff)
            continue

    if meta is not None:
        meta["finish_reason"] = finish_reason_last
    if last_err is not None and not yielded:
        raise RuntimeError(
            "Failed to connect to llama-server after retries. "
            "Check container logs/tmux for llama-server startup errors."
        ) from last_err
    if last_400_body is not None and not yielded:
        # raise_for_status was bypassed because we already raised RuntimeError above;
        # this branch only reached if the loop fell through without yielding.
        raise RuntimeError(
            f"llama-server rejected request: {last_400_body[:1500]}"
        )
    if not yielded:
        raise RuntimeError("LLM returned empty response from direct llama-server")


def _batched_stream_text(
    chunks: Iterable[str],
    *,
    min_chars: int | None = None,
    max_interval_s: float | None = None,
    max_burst_chars: int | None = None,
) -> Iterator[str]:
    """Merge many small strings from an LLM stream before sending to the UI.

    llama-server often emits token-sized ``content`` deltas; forwarding each
    as its own SSE event floods the browser's main thread (layout + JSON).
    """
    lo = SSE_UI_TOKEN_BATCH_MIN_CHARS if min_chars is None else min_chars
    mx_iv = SSE_UI_TOKEN_BATCH_MAX_INTERVAL_S if max_interval_s is None else max_interval_s
    mx_burst = SSE_UI_TOKEN_BATCH_MAX_BURST_CHARS if max_burst_chars is None else max_burst_chars

    buf: list[str] = []
    size = 0
    t_flush = time.monotonic()
    for chunk in chunks:
        if not chunk:
            continue
        buf.append(chunk)
        size += len(chunk)
        now = time.monotonic()
        if (
            size >= mx_burst
            or size >= lo
            or (size > 0 and now - t_flush >= mx_iv)
        ):
            yield "".join(buf)
            buf.clear()
            size = 0
            t_flush = now
    if buf:
        yield "".join(buf)


def _wait_for_llm_ready(timeout_s: int = 300) -> None:
    """Wait until llama-server responds to a health endpoint."""
    health_url = f"{LLM_BASE_URL.rstrip('/')}/health"
    models_url = f"{LLM_BASE_URL.rstrip('/')}/v1/models"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    deadline = time.time() + timeout_s
    last_err: Exception | None = None

    with httpx.Client(timeout=httpx.Timeout(timeout=5.0, connect=2.0)) as client:
        while time.time() < deadline:
            try:
                r = client.get(health_url)
                if r.status_code == 200:
                    return
            except Exception as e:
                last_err = e
            try:
                r2 = client.get(models_url, headers=headers)
                if r2.status_code == 200:
                    return
            except Exception as e:
                last_err = e
            time.sleep(2)

    raise RuntimeError(
        "llama-server is not reachable. It may still be loading the model, or startup may have failed "
        "(often GPU/CUDA init/OOM)."
    ) from last_err


def _extract_fenced(text: str, lang_hint: str = "") -> str:
    """Extract first markdown fenced block anywhere in text.

    Falls back to raw text when no code fence is present.
    """
    text = text.strip()
    if not text:
        return text
    lang_re = rf"(?:{re.escape(lang_hint)})" if lang_hint else r"[A-Za-z0-9_+-]*"
    pattern = re.compile(
        rf"```[ \t]*{lang_re}[ \t]*\n(.*?)\n```",
        flags=re.DOTALL,
    )
    m = pattern.search(text)
    if m:
        return m.group(1).strip()
    fallback = re.search(r"```[^\n]*\n(.*?)\n```", text, flags=re.DOTALL)
    if fallback:
        return fallback.group(1).strip()
    return text


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _generate_diff(old_text: str, new_text: str, old_label: str, new_label: str) -> str:
    """Produce a unified diff between two texts."""
    diff_lines = list(difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=old_label,
        tofile=new_label,
    ))
    return "".join(diff_lines) if diff_lines else "(no changes)\n"


def _build_conversation_context(messages: list[dict]) -> str:
    """Format conversation messages into LLM prompt context."""
    if not messages:
        return ""
    parts = [
        "## User Feedback and Error Logs\n",
        "The following messages were provided by the user after reviewing the "
        "previously generated code. These typically contain build errors, compiler "
        "warnings, test failures, or other issues found when integrating the "
        "generated code into the repository. Fix ALL issues described below "
        "while maintaining full compliance with the ICD change specification "
        "and repository conventions. Do not introduce regressions in areas "
        "that were previously correct.\n",
    ]
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"\n### {role.title()} Message\n```\n{content}\n```\n")
    return "\n".join(parts)


class ChatMessage(BaseModel):
    message: str


def _build_header_documentation_prompt(
    fname: str,
    header_code: str,
    target_summary: str,
) -> str:
    """Build the user prompt for the header documentation pass.

    Parameters
    ----------
    fname           : filename, e.g. "IMU.h"
    header_code     : the full content of the generated .h file
    target_summary  : the target ICD consolidated summary

        """
    sections = [
        f"## Header file to document: `{fname}`\n\n"
        f"This file was generated by an automated ICD transform tool. "
        f"Add documentation comments as specified. "
        f"Do NOT alter any declaration — only add comments.\n\n"
        f"```c\n{header_code}\n```",

        "## Target ICD Consolidated Summary\n"
        "Supplementary context for document references, module names, "
        "and version information.\n\n"
        f"{target_summary}",
    ]

    sections.append(
        "## Your task\n"
        f"Output the fully documented version of `{fname}`. "
        "Every struct field must have an inline comment with its ICD-defined "
        "description, unit, range, and conversion formula. "
        "Every `#define` must have a comment. "
        "The file-header block and section dividers must be present. "
        "Output nothing except the single ```c fenced block."
    )

    return "\n\n".join(sections)

# ---------------------------------------------------------------------------
# Repository context helpers
# ---------------------------------------------------------------------------

_HEADER_EXTS = {'.h', '.hpp', '.hh', '.hxx'}
_SOURCE_EXTS = {'.c', '.cpp', '.cc', '.cxx'}
_BUILD_NAMES = {'Makefile', 'CMakeLists.txt'}
_BUILD_EXTS = {'.mk', '.cmake', '.mak'}

_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> dict:
    """Extract a ZIP safely into *dest_dir*, returning file statistics."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    file_count = 0
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            target = (dest_dir / info.filename).resolve()
            if not str(target).startswith(str(dest_dir.resolve())):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, 'wb') as dst:
                dst.write(src.read())
            file_count += 1
    return {"file_count": file_count}


def _extract_includes(source: str) -> list[str]:
    """Parse #include "..." directives from C source text."""
    return _INCLUDE_RE.findall(source)


def _find_repo_file(repo_dir: Path, include_name: str) -> Path | None:
    """Locate a header in the repository by its include name (e.g. 'comm.h')."""
    basename = Path(include_name).name
    for candidate in repo_dir.rglob(basename):
        if candidate.is_file():
            rel = str(candidate.relative_to(repo_dir))
            if rel.endswith(include_name) or candidate.name == basename:
                return candidate
    return None


def _build_file_repo_context(
    repo_dir: Path, source_text: str, exclude_names: set[str],
    max_chars: int = MAX_REPO_CONTEXT_CHARS,
) -> str:
    """Build a *distilled* repo context for a specific file being transformed.

    Instead of dumping the entire repo, this parses the file's #include
    directives, resolves them in the repository, and includes only the
    headers (and their transitive level-1 includes) that are actual
    dependencies.  This produces a small, highly relevant context.
    """
    direct_includes = _extract_includes(source_text)
    if not direct_includes:
        return ""

    seen: set[str] = set()
    resolved: list[tuple[str, str]] = []  # (rel_path, content)
    budget = max_chars

    def _resolve(include_name: str, depth: int) -> None:
        nonlocal budget
        if include_name in seen or budget <= 0 or depth > 2:
            return
        seen.add(include_name)
        fp = _find_repo_file(repo_dir, include_name)
        if not fp or fp.name in exclude_names:
            return
        try:
            content = fp.read_text(errors="replace")
        except Exception:
            return
        rel = str(fp.relative_to(repo_dir))
        entry_len = len(rel) + len(content) + 30
        if entry_len > budget:
            trim = content[: budget - len(rel) - 80]
            resolved.append((rel, trim + "\n/* ... truncated ... */"))
            budget = 0
            return
        resolved.append((rel, content))
        budget -= entry_len
        if depth < 2:
            for sub in _extract_includes(content):
                _resolve(sub, depth + 1)

    for inc in direct_includes:
        _resolve(inc, 0)

    if not resolved:
        return ""

    parts = ["## Dependency Headers from Repository\n"]
    for rel, content in resolved:
        parts.append(f"\n### {rel}\n```c\n{content}\n```\n")
    return "".join(parts)


def _build_source_scripts_context(
    code_dir: Path,
    max_chars: int = SOURCE_SCRIPTS_MAX_CHARS,
) -> str:
    """Build a context containing the ACTUAL .c and .h source files.

    The input ``code_dir`` is the directory of UPLOADED user code files
    (``session_dir / "original_code"``) — i.e. the very ``.c`` and ``.h``
    files that will be refactored against the Target ICD.  Passing the
    user's uploaded sources (instead of an entire repository ZIP) keeps
    the LLM grounded in the precise files under change without being
    diluted by unrelated repository code.

    This is the sole code-scripts context used by the ICD-delta analysis
    stage: instead of a heuristic summary of identifiers, the LLM is given
    the real source so it can ground the *Impact on C code* portion of the
    change specification in concrete struct layouts, function bodies, enum
    values, macro expansions and ``#include`` topology.

    Headers are emitted first (highest signal-per-byte), then implementation
    files, until the byte budget is exhausted.  The first section is a flat
    file tree so the model knows what exists even if some files get trimmed.
    """
    all_files = sorted(p for p in code_dir.rglob("*") if p.is_file())
    if not all_files:
        return ""

    tree_lines = [str(f.relative_to(code_dir)) for f in all_files]
    headers = [f for f in all_files if f.suffix.lower() in _HEADER_EXTS]
    sources = [f for f in all_files if f.suffix.lower() in _SOURCE_EXTS]
    if not headers and not sources:
        return ""

    parts: list[str] = [
        "## Existing C Source Scripts (PRE-CHANGE \"OLD\" CODE)\n",
        "The blocks below are the actual .c / .h files uploaded for "
        "refactoring as they exist BEFORE the ICD change. Treat them as the "
        "ground-truth current implementation when assessing impact on C "
        "code.\n\n",
        "### Uploaded Source Scripts File Structure\n```\n"
        + "\n".join(tree_lines[:120])
        + ("\n... (more files omitted)" if len(tree_lines) > 120 else "")
        + "\n```\n",
    ]
    budget = max_chars - sum(len(p) for p in parts)

    def _emit(fp: Path, kind: str) -> bool:
        """Append one file. Returns False when the budget is exhausted."""
        nonlocal budget
        try:
            content = _strip_c_comments(fp.read_text(errors="replace"))
        except Exception:
            return True
        rel = str(fp.relative_to(code_dir))
        entry = f"\n### {kind}: {rel}\n```c\n{content}\n```\n"
        if len(entry) <= budget:
            parts.append(entry)
            budget -= len(entry)
            return True
        if budget > 300:
            trim = content[: max(0, budget - 200)]
            parts.append(
                f"\n### {kind}: {rel}\n```c\n{trim}\n"
                "/* ... truncated (source-scripts budget exhausted) ... */\n```\n"
            )
            budget = 0
        return False

    if headers:
        parts.append("\n## Headers (.h)\n")
        for fp in headers:
            if not _emit(fp, "Header"):
                break

    if sources and budget > 0:
        parts.append("\n## Implementation Files (.c)\n")
        for fp in sources:
            if not _emit(fp, "Source"):
                break

    result = "".join(parts)
    return _truncate_text(result, max_chars, "source_scripts")


def _build_repo_knowledge(
    repo_dir: Path,
    max_chars: int = 10_000,
    *,
    full: bool = False,
) -> str:
    """Extract comprehensive high-level and low-level knowledge from the repo.

    High-level: file structure, module organisation, naming conventions.
    Low-level: struct/union definitions with fields, enum definitions with
    values, full function signatures, global variable declarations, and
    macro definitions with their values.

    Two flavours are supported:

      * ``full=False`` (default) — the **distilled** view fed into LLM
        prompts.  Per-category lists are capped (30 structs, 20 enums,
        40 function signatures, 30 globals, 40 macros, 100 tree lines)
        and the final string is clipped to ``max_chars`` so the prompt
        budget stays under control.

      * ``full=True`` — the **untruncated** view written to
        ``repo_knowledge.txt`` and shipped in the downloadable ZIP for
        QA review.  No per-category caps are applied and ``max_chars``
        is intentionally ignored.
    """
    all_files = sorted(p for p in repo_dir.rglob("*") if p.is_file())
    code_exts = _HEADER_EXTS | _SOURCE_EXTS
    tree_lines = [str(f.relative_to(repo_dir)) for f in all_files]

    struct_re = re.compile(
        r'typedef\s+(?:struct|union)\s*\w*\s*\{([^}]*)\}\s*(\w+)\s*;', re.DOTALL)
    enum_re = re.compile(
        r'typedef\s+enum\s*\w*\s*\{([^}]*)\}\s*(\w+)\s*;', re.DOTALL)
    func_sig_re = re.compile(
        r'^(?:(?:static|extern|inline|const|unsigned|void)\s+)*'
        r'[A-Za-z_]\w*(?:\s*\*)*\s+([A-Za-z_]\w*)\s*\(([^)]*)\)',
        re.MULTILINE)
    global_var_re = re.compile(
        r'^(?:(?:static|extern|volatile|const)\s+)+'
        r'([A-Za-z_]\w*(?:\s*\*)*)\s+([A-Za-z_]\w*)'
        r'(?:\s*\[[^\]]*\])*'
        r'(?:\s*=\s*[^;]+)?;',
        re.MULTILINE)
    define_re = re.compile(
        r'^\s*#\s*define\s+([A-Z_][A-Z0-9_]+)[ \t]+(.+?)$', re.MULTILINE)

    structs: list[str] = []
    enums: list[str] = []
    func_sigs: list[str] = []
    global_vars: list[str] = []
    macros: list[str] = []

    for f in all_files:
        if f.suffix.lower() not in code_exts:
            continue
        try:
            content = f.read_text(errors="replace")
        except Exception:
            continue
        for m in struct_re.finditer(content):
            fields = " ".join(m.group(1).split())
            structs.append(f"  {m.group(2)}: {{ {fields} }}")
        for m in enum_re.finditer(content):
            vals = " ".join(m.group(1).split())
            enums.append(f"  {m.group(2)}: {{ {vals} }}")
        for m in func_sig_re.finditer(content):
            params = " ".join(m.group(2).split())
            func_sigs.append(f"  {m.group(1)}({params})")
        for m in global_var_re.finditer(content):
            global_vars.append(f"  {m.group(1)} {m.group(2)}")
        for m in define_re.finditer(content):
            macros.append(f"  {m.group(1)} = {m.group(2).strip()}")

    func_sigs_sorted = sorted(set(func_sigs))
    global_vars_sorted = sorted(set(global_vars))
    if full:
        structs_out = structs
        enums_out = enums
        func_sigs_out = func_sigs_sorted
        global_vars_out = global_vars_sorted
        macros_out = macros
        tree_out = tree_lines
    else:
        structs_out = structs[:30]
        enums_out = enums[:20]
        func_sigs_out = func_sigs_sorted[:40]
        global_vars_out = global_vars_sorted[:30]
        macros_out = macros[:40]
        tree_out = tree_lines[:100]

    flavour = "FULL (untruncated, QA download)" if full else "DISTILLED (LLM context)"
    parts = [
        "=" * 65 + "\n",
        f"REPOSITORY CODEBASE KNOWLEDGE [{flavour}]\n",
        "=" * 65 + "\n\n",
        "## High-Level: File Structure & Organisation\n```\n",
        "\n".join(tree_out),
        "\n```\n",
    ]
    if not full and len(tree_lines) > len(tree_out):
        parts.append(
            f"_(file tree truncated to first {len(tree_out)} of "
            f"{len(tree_lines)} entries — full list in repo_knowledge.txt)_\n"
        )
    if structs_out:
        parts.append("\n## Low-Level: Struct / Union Definitions\n")
        parts.append("\n".join(structs_out) + "\n")
        if not full and len(structs) > len(structs_out):
            parts.append(
                f"_(+{len(structs) - len(structs_out)} more in "
                f"repo_knowledge.txt)_\n"
            )
    if enums_out:
        parts.append("\n## Low-Level: Enum Definitions\n")
        parts.append("\n".join(enums_out) + "\n")
        if not full and len(enums) > len(enums_out):
            parts.append(
                f"_(+{len(enums) - len(enums_out)} more in "
                f"repo_knowledge.txt)_\n"
            )
    if func_sigs_out:
        parts.append("\n## Low-Level: Function Signatures\n")
        parts.append("\n".join(func_sigs_out) + "\n")
        if not full and len(func_sigs_sorted) > len(func_sigs_out):
            parts.append(
                f"_(+{len(func_sigs_sorted) - len(func_sigs_out)} more in "
                f"repo_knowledge.txt)_\n"
            )
    if global_vars_out:
        parts.append("\n## Low-Level: Global Variable Declarations\n")
        parts.append("\n".join(global_vars_out) + "\n")
        if not full and len(global_vars_sorted) > len(global_vars_out):
            parts.append(
                f"_(+{len(global_vars_sorted) - len(global_vars_out)} more in "
                f"repo_knowledge.txt)_\n"
            )
    if macros_out:
        parts.append("\n## Low-Level: Macro Definitions\n")
        parts.append("\n".join(macros_out) + "\n")
        if not full and len(macros) > len(macros_out):
            parts.append(
                f"_(+{len(macros) - len(macros_out)} more in "
                f"repo_knowledge.txt)_\n"
            )

    result = "".join(parts)
    # ``full`` reports are never length-clipped — that flavour is the
    # canonical record for the downloadable QA report.  Only the
    # distilled flavour is bounded by ``max_chars``.
    if full:
        return result
    return _truncate_text(result, max_chars, "repo_knowledge")


def _extract_c_variables(code: str, filename: str) -> list[dict]:
    """Extract variable declarations from C source code.

    Returns a list of dicts with keys: name, type, scope, extra.
    """
    results: list[dict] = []
    seen: set[str] = set()

    global_re = re.compile(
        r'^(?:(?:static|extern|volatile|const|unsigned|signed|short|long|register)\s+)*'
        r'([A-Za-z_]\w*(?:\s*\*)*)\s*'
        r'\*?([A-Za-z_]\w*)'
        r'(\s*\[[^\]]*\])?'
        r'(?:\s*=\s*([^;]+))?;',
        re.MULTILINE)

    func_re = re.compile(
        r'^(?:(?:static|extern|inline|const|unsigned|void)\s+)*'
        r'[A-Za-z_]\w*(?:\s*\*)*\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*\{',
        re.MULTILINE)

    define_re = re.compile(
        r'^\s*#\s*define\s+([A-Za-z_]\w+)(?:[ \t]+(.+?))?$', re.MULTILINE)

    brace_depth = 0
    in_function = False
    current_func = ""
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
            pass
        fm = func_re.match(line)
        if fm and brace_depth == 0:
            in_function = True
            current_func = fm.group(1)
            params = fm.group(2).strip()
            if params and params != "void":
                for param in params.split(","):
                    parts = param.strip().rsplit(None, 1)
                    if len(parts) == 2:
                        ptype, pname = parts
                        pname = pname.lstrip("*")
                        key = f"param:{current_func}:{pname}"
                        if key not in seen:
                            seen.add(key)
                            results.append({
                                "name": pname,
                                "type": ptype.strip(),
                                "scope": f"parameter of {current_func}()",
                                "extra": "",
                            })
        brace_depth += stripped.count("{") - stripped.count("}")

        if brace_depth <= 0:
            in_function = False
            current_func = ""
            brace_depth = max(brace_depth, 0)

    for m in global_re.finditer(code):
        vtype = m.group(1).strip()
        vname = m.group(2)
        arr = (m.group(3) or "").strip()
        init = (m.group(4) or "").strip()
        pos = m.start()
        depth = code[:pos].count("{") - code[:pos].count("}")
        scope = "global" if depth == 0 else "local"
        key = f"{scope}:{vname}"
        if vname in ("if", "while", "for", "switch", "return", "sizeof",
                      "else", "case", "break", "continue", "goto"):
            continue
        if key not in seen:
            seen.add(key)
            extra = ""
            if arr:
                extra = f"array{arr}"
            if init:
                extra += (", " if extra else "") + f"init={init[:60]}"
            results.append({
                "name": vname,
                "type": vtype,
                "scope": scope,
                "extra": extra,
            })

    for m in define_re.finditer(code):
        mname = m.group(1)
        mval = (m.group(2) or "").strip()
        key = f"macro:{mname}"
        if key not in seen:
            seen.add(key)
            results.append({
                "name": mname,
                "type": "#define",
                "scope": "macro",
                "extra": mval[:80] if mval else "(flag)",
            })

    return results


def _format_variable_inventory(variables: list[dict], filename: str) -> str:
    """Format extracted variables into a report section."""
    if not variables:
        return f"  (no variables extracted from {filename})\n"
    lines = []
    by_scope: dict[str, list[dict]] = {}
    for v in variables:
        by_scope.setdefault(v["scope"], []).append(v)
    for scope in sorted(by_scope):
        lines.append(f"  [{scope}]")
        for v in by_scope[scope]:
            extra = f"  ({v['extra']})" if v["extra"] else ""
            lines.append(f"    {v['type']:30s} {v['name']}{extra}")
    return "\n".join(lines) + "\n"


def _build_repo_context(repo_dir: Path, exclude_names: set[str] | None = None) -> str:
    """Build a structured context string from the full repository (fallback).

    Used when per-file dependency tracing is not applicable.
    """
    exclude = exclude_names or set()
    all_files = sorted(p for p in repo_dir.rglob("*") if p.is_file())

    headers: list[Path] = []
    tree_lines: list[str] = []

    for f in all_files:
        rel = str(f.relative_to(repo_dir))
        tree_lines.append(rel)
        if f.name in exclude:
            continue
        if f.suffix.lower() in _HEADER_EXTS:
            headers.append(f)

    sections: list[str] = []
    budget = MAX_REPO_CONTEXT_CHARS

    tree_text = "## Repository File Structure\n```\n" + "\n".join(tree_lines) + "\n```\n"
    sections.append(tree_text)
    budget -= len(tree_text)

    if headers and budget > 0:
        hdr_parts = ["## Repository Headers\n"]
        for fp in headers:
            try:
                content = fp.read_text(errors="replace")
            except Exception:
                continue
            rel = str(fp.relative_to(repo_dir))
            entry = f"\n### {rel}\n```c\n{content}\n```\n"
            if len(entry) > budget:
                if budget > 300:
                    trim = content[: budget - 200]
                    hdr_parts.append(f"\n### {rel}\n```c\n{trim}\n/* ... truncated ... */\n```\n")
                break
            hdr_parts.append(entry)
            budget -= len(entry)
        sections.append("".join(hdr_parts))

    result = "\n".join(sections)
    if len(result) > MAX_REPO_CONTEXT_CHARS:
        result = result[:MAX_REPO_CONTEXT_CHARS] + "\n[... TRUNCATED repo_context ...]"
    return result


def _structural_verify(generated: str, repo_dir: Path | None,
                       original: str, filename: str) -> list[str]:
    """Run deterministic structural checks on generated C code.

    Returns a list of issue descriptions (empty = all checks passed).
    """
    issues: list[str] = []
    text = generated.strip()

    if text.count("{") != text.count("}"):
        issues.append(f"Unbalanced braces: {text.count('{')} open vs {text.count('}')} close")
    if text.count("/*") > text.count("*/"):
        issues.append(f"Unclosed block comment: {text.count('/*')} open vs {text.count('*/')} close")
    if filename.endswith(".h"):
        has_guard = re.search(r"^\s*#\s*ifn?def\b", text, re.MULTILINE)
        has_endif = re.search(r"^\s*#\s*endif\b", text, re.MULTILINE)
        if has_guard and not has_endif:
            issues.append("Header guard #ifndef without matching #endif")

    if repo_dir and repo_dir.exists():
        includes = _extract_includes(generated)
        for inc in includes:
            if not _find_repo_file(repo_dir, inc):
                issues.append(f'#include "{inc}" not found in repository')

    orig_funcs = set(re.findall(r'\b([A-Z_]\w+)\s*\(', original))
    gen_funcs = set(re.findall(r'\b([A-Z_]\w+)\s*\(', generated))
    missing = orig_funcs - gen_funcs - {'if', 'while', 'for', 'switch', 'sizeof', 'return'}
    if missing and len(missing) <= 5:
        issues.append(f"Functions from original not in generated: {', '.join(sorted(missing))}")

    return issues


# ---------------------------------------------------------------------------
# Sandbox build helpers
# ---------------------------------------------------------------------------

def _detect_build_system(
    repo_dir: Path,
    injected_paths: list[Path] | None = None,
) -> dict:
    """Detect the build system used by a repository.

    When *injected_paths* is given, we first walk **up** from each injected
    file's parent towards *repo_dir*, looking for Makefile / CMakeLists.txt in
    a direct ancestor directory.  This ensures that for large repos with many
    build files (e.g. BSP sub-projects) we pick the one that actually governs
    the generated code.

    Falls back to a global search (preferring shallowest depth) only when no
    ancestor build file is found.

    Returns a dict with keys ``type`` (``"make"`` | ``"cmake"`` | ``"none"``),
    ``path`` (the build file found), and ``build_dir`` (directory containing
    it).
    """
    # -- Phase 1: ancestor walk from injected file locations -----------------
    if injected_paths:
        ancestor_hits: list[tuple[int, str, Path]] = []
        seen: set[Path] = set()
        for ip in injected_paths:
            d = ip.parent if ip.is_file() else ip
            while True:
                if d in seen:
                    break
                seen.add(d)
                for name in ("CMakeLists.txt", "Makefile"):
                    cand = d / name
                    if cand.is_file():
                        depth = len(d.relative_to(repo_dir).parts) if d != repo_dir else 0
                        btype = "cmake" if name == "CMakeLists.txt" else "make"
                        ancestor_hits.append((depth, btype, cand))
                if d == repo_dir:
                    break
                d = d.parent

        if ancestor_hits:
            ancestor_hits.sort(key=lambda c: (-c[0], c[1]))
            _, btype, bpath = ancestor_hits[0]
            return {"type": btype, "path": bpath, "build_dir": bpath.parent}

    # -- Phase 2: global scan (original behaviour) ---------------------------
    candidates: list[tuple[int, str, Path]] = []
    for p in repo_dir.rglob("*"):
        if not p.is_file():
            continue
        depth = len(p.relative_to(repo_dir).parts)
        if p.name == "Makefile":
            candidates.append((depth, "make", p))
        elif p.name == "CMakeLists.txt":
            candidates.append((depth, "cmake", p))
        elif p.suffix in _BUILD_EXTS:
            btype = "cmake" if p.suffix == ".cmake" else "make"
            candidates.append((depth + 100, btype, p))

    if not candidates:
        return {"type": "none", "path": None, "build_dir": repo_dir}

    type_priority = {"make": 0, "cmake": 1}
    candidates.sort(key=lambda c: (type_priority.get(c[1], 99), c[0]))
    _, btype, bpath = candidates[0]
    return {"type": btype, "path": bpath, "build_dir": bpath.parent}


def _find_file_in_repo(repo_dir: Path, filename: str) -> list[Path]:
    """Find all files in *repo_dir* whose basename matches *filename*.

    Results are sorted by depth (shallowest first) so callers can prefer
    the most likely match.
    """
    matches = [
        p for p in repo_dir.rglob(filename)
        if p.is_file() and p.name == filename
    ]
    matches.sort(key=lambda p: len(p.relative_to(repo_dir).parts))
    return matches


# Per-file compile + sandbox build flags.
#
# We deliberately use ``-std=gnu99`` (NOT ``-std=c99 -pedantic``) for
# two reasons that came out of the IMU.c production failure:
#
#  1. Glibc's ``<math.h>`` only exposes ``M_PI`` / ``M_E`` / etc. when
#     ``__USE_MISC`` is defined, which requires ``_DEFAULT_SOURCE`` /
#     ``_BSD_SOURCE``. Under ``-std=c99`` glibc sets ``__STRICT_ANSI__``
#     and hides those macros — so the perfectly-correct fix
#     ``#include <math.h>; ...M_PI...`` still fails to compile and
#     the agentic loop has to guess at workarounds. ``-std=gnu99``
#     drops ``__STRICT_ANSI__`` and ``M_PI`` becomes visible
#     immediately.
#
#  2. Real-world embedded C uses GCC extensions extensively (unnamed
#     structs/unions, ``__attribute__``, statement expressions, extra
#     ``;``s after struct typedefs). Under ``-pedantic`` each of those
#     constructs floods the compile log with ``-Wpedantic`` warnings,
#     which then crowd the structured punch list and the LLM's input
#     context with non-actionable noise. ``-std=gnu99`` keeps the
#     C99 semantics we want for correctness checks while making the
#     compile gate match the way the user's actual toolchain (e.g.
#     ``arm-xilinx-eabi-gcc``) is configured in practice.
#
# Real errors (missing struct members, wrong types, missing
# declarations, missing headers) are still caught — they're language
# semantics, not portability warnings.
SANDBOX_CC_NATIVE = "gcc -std=gnu99"
SANDBOX_CC_ARM = "arm-none-eabi-gcc -std=gnu99"

_CC_RE = re.compile(
    r"^\s*CC\s*[:?]?=\s*(.+?)\s*$", re.MULTILINE,
)
_CROSS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"arm-none-eabi-gcc", re.I), SANDBOX_CC_ARM),
    (re.compile(r"arm-xilinx-eabi-gcc", re.I), SANDBOX_CC_ARM),
    (re.compile(r"aarch64-none-elf-gcc", re.I), SANDBOX_CC_ARM),
    (re.compile(r"mb-gcc|microblaze.*gcc", re.I), SANDBOX_CC_ARM),
]
_LINKER_ERROR_RE = re.compile(
    r"undefined reference to|"
    r"cannot find -l|"
    r"ld returned \d+ exit status|"
    r"cannot open output file|"
    r"collect2: error",
    re.I,
)


def _detect_cross_compiler(build_info: dict) -> str:
    """Detect whether the project needs a cross-compiler.

    Reads the Makefile (or CMakeLists.txt) and checks the ``CC`` variable
    for known cross-compiler prefixes.  Returns the sandbox ``CC`` string
    to use (cross or native).
    """
    bpath = build_info.get("path")
    if not bpath or not bpath.exists():
        return SANDBOX_CC_NATIVE

    try:
        content = bpath.read_text(errors="replace")
    except OSError:
        return SANDBOX_CC_NATIVE

    for m in _CC_RE.finditer(content):
        cc_val = m.group(1)
        for pattern, sandbox_cc in _CROSS_PATTERNS:
            if pattern.search(cc_val):
                return sandbox_cc

    for pattern, sandbox_cc in _CROSS_PATTERNS:
        if pattern.search(content):
            return sandbox_cc

    return SANDBOX_CC_NATIVE


def _is_linker_only_failure(build_output: str) -> bool:
    """Return True if the build output contains only linker errors (no
    compile errors).  This means the source code is valid but linking
    failed because of missing BSP libraries / linker scripts."""
    has_linker = bool(_LINKER_ERROR_RE.search(build_output))
    has_compile = bool(re.search(
        r":\d+:\d+:\s*error:", build_output,
    ))
    return has_linker and not has_compile


def _run_sandbox_build(
    sandbox_dir: Path,
    build_info: dict,
    timeout: int = SANDBOX_BUILD_TIMEOUT,
) -> tuple[bool, str]:
    """Execute the build command inside *sandbox_dir*.

    Automatically detects whether the project uses a cross-compiler
    (arm-none-eabi-gcc, mb-gcc, etc.) and uses the matching toolchain.

    If a full build fails with only linker errors (missing BSP libraries
    / linker scripts), a compile-only pass (``-c``) is attempted.  When
    all source files compile to object files successfully, the build is
    considered a pass — linking requires BSP artifacts that may not be in
    the uploaded repository.

    Returns ``(success, combined_output)``.
    """
    btype = build_info["type"]
    bdir = build_info["build_dir"]
    sandbox_cc = _detect_cross_compiler(build_info)

    if btype == "make":
        full_cmd = (
            f"make -C {bdir} CC='{sandbox_cc}' clean 2>/dev/null; "
            f"make -C {bdir} CC='{sandbox_cc}' 2>&1"
        )
    elif btype == "cmake":
        cmake_build = bdir / "_cmake_build"
        cc_bin = sandbox_cc.split()[0]
        cc_flags = " ".join(sandbox_cc.split()[1:])
        full_cmd = (
            f"cmake -S {bdir} -B {cmake_build} "
            f"-DCMAKE_C_COMPILER={cc_bin} "
            f'-DCMAKE_C_FLAGS="{cc_flags}" 2>&1 && '
            f"cmake --build {cmake_build} 2>&1"
        )
    else:
        return False, "No supported build system detected in repository."

    try:
        result = subprocess.run(
            full_cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=str(sandbox_dir),
        )
        output = (result.stdout or "") + (result.stderr or "")

        if result.returncode == 0:
            return True, output.strip()

        if not _is_linker_only_failure(output):
            return False, output.strip()

        log.info("Full build had only linker errors — attempting compile-only pass")
        compile_output = output + "\n\n--- Linker errors detected; retrying compile-only ---\n"

        c_sources = sorted(bdir.rglob("*.c"))
        if not c_sources:
            return False, output.strip()

        include_dirs: set[str] = set()
        for inc in bdir.rglob("*.h"):
            include_dirs.add(str(inc.parent))

        # Write a shell script to compile each source individually,
        # avoiding shell argument-length limits on large repos.
        script = bdir / "_sandbox_compile.sh"
        lines = ["#!/bin/sh", "set -e", f'CC="{sandbox_cc}"']
        inc_args = " ".join(f'"-I{d}"' for d in sorted(include_dirs))
        lines.append(f"INC={inc_args}")
        for src in c_sources:
            obj = src.with_suffix(".o")
            lines.append(f'$CC $INC -c -o "{obj}" "{src}" 2>&1')
        script.write_text("\n".join(lines) + "\n")
        script.chmod(0o755)

        result2 = subprocess.run(
            ["sh", str(script)], capture_output=True, text=True,
            timeout=timeout, cwd=str(sandbox_dir),
        )
        compile_output += (result2.stdout or "") + (result2.stderr or "")

        if result2.returncode == 0:
            compile_output += (
                "\n\n--- Compile-only PASSED (all .c → .o succeeded) ---\n"
                "Linking skipped: BSP libraries/linker scripts not in repository.\n"
            )
            return True, compile_output.strip()

        return False, compile_output.strip()

    except subprocess.TimeoutExpired:
        return False, f"Build timed out after {timeout} seconds."
    except Exception as e:
        return False, f"Build execution error: {e}"


_GCC_FILE_RE = re.compile(r"^(\S+?\.[ch]):\d+:\d+:\s*(?:error|warning)", re.MULTILINE)

def _parse_all_build_errors(
    build_output: str,
    sandbox_dir: Path,
    file_index: dict[str, list[Path]] | None = None,
) -> dict[Path, str]:
    """Extract per-file error sections from compiler output for ALL files.

    Returns ``{absolute_path: relevant_error_lines}`` for every ``.c``/``.h``
    file mentioned in the compiler output.  Paths are resolved relative to
    *sandbox_dir*.

    When *file_index* (``{basename: [abs_paths]}``) is provided, it is used
    for fast name-based lookup instead of ``rglob`` (critical for large repos).
    """
    mentioned: set[str] = set()
    for m in _GCC_FILE_RE.finditer(build_output):
        mentioned.add(m.group(1))

    def _resolve(rel: str) -> Path:
        abs_path = (sandbox_dir / rel).resolve()
        if abs_path.exists():
            return abs_path
        basename = Path(rel).name
        if file_index and basename in file_index:
            return file_index[basename][0]
        return abs_path

    file_errors: dict[Path, list[str]] = {}
    for rel in mentioned:
        file_errors[_resolve(rel)] = []

    for line in build_output.splitlines():
        for rel in mentioned:
            if rel in line:
                file_errors.setdefault(_resolve(rel), []).append(line)

    return {p: "\n".join(lines) for p, lines in file_errors.items() if lines}


def _parse_error_files(build_output: str, gen_filenames: set[str]) -> dict[str, str]:
    """Backwards-compatible wrapper: errors keyed by generated filename only."""
    file_errors: dict[str, list[str]] = {f: [] for f in gen_filenames}
    for line in build_output.splitlines():
        for fname in gen_filenames:
            if fname in line:
                file_errors[fname].append(line)

    result = {f: "\n".join(lines) for f, lines in file_errors.items() if lines}
    if not result and gen_filenames:
        first = sorted(gen_filenames)[0]
        result[first] = build_output
    return result


_ERROR_LINE_RE = re.compile(r"^(.+?:\d+:\d+:\s*(?:error|warning):\s*)(.+)$", re.MULTILINE)

def _normalise_error_signature(build_output: str) -> str:
    """Produce a stable fingerprint of compiler errors in *build_output*.

    Strips line numbers and paths so that the same logical errors across
    different iterations (where line numbers may shift due to code edits)
    still compare as equal.  Returns an empty string if no errors found.
    """
    msgs = sorted({m.group(2).strip() for m in _ERROR_LINE_RE.finditer(build_output)})
    return "\n".join(msgs)


def _sandbox_build_iterate(
    session_dir: Path,
    gen_dir: Path,
    repo_dir: Path,
    change_spec: str,
    uploaded_names: set[str],
    has_repo: bool,
    repo_knowledge: str = "",
    sandbox_max_retries: object = _SANDBOX_RETRIES_UNSET,
):
    """Generator that yields SSE dicts for the sandbox build loop.

    Copies the repo, injects generated files, builds, and iteratively fixes
    compiler errors via the LLM until the build succeeds.

    ``sandbox_max_retries`` controls the user-facing reiteration budget:
      * ``_SANDBOX_RETRIES_UNSET`` — classic env-driven outer-round × caps
      * ``None`` — keep iterating indefinitely until the build succeeds
      * positive ``int`` — at most that many build/fix reiterations

    Key convergence strategies:
    - Generated files are fixed using the original working code as a reference,
      preventing error drift.
    - Non-generated repo files that fail to compile (due to API changes from
      the ICD transformation) are also patched with a minimal-change prompt.
    - On stall (same errors repeating), all files are reset to their initial
      state and the LLM is given accumulated error history to force a
      different approach.
    """
    retry_plan = _sandbox_retry_plan(sandbox_max_retries)
    yield _sse({
        "type": "info",
        "stage": "sandbox_build",
        "message": f"Sandbox build retry budget: {retry_plan['label']}.",
    })
    sandbox_dir = session_dir / "sandbox"
    if sandbox_dir.exists():
        shutil.rmtree(sandbox_dir)
    yield _sse({
        "type": "info",
        "stage": "sandbox_build",
        "message": "Copying repository into sandbox…",
    })
    shutil.copytree(repo_dir, sandbox_dir)

    gen_files = sorted(
        p for p in gen_dir.iterdir()
        if p.is_file() and p.suffix in ('.c', '.h')
    )
    if not gen_files:
        yield _sse({
            "type": "info",
            "stage": "sandbox_build",
            "message": "No generated .c/.h files to inject — skipping sandbox build.",
        })
        return

    gen_filenames = {gf.name for gf in gen_files}

    # --- Map generated files to their sandbox locations and inject -----------
    replacement_map: dict[str, Path] = {}
    original_repo_code: dict[str, str] = {}
    initial_gen_code: dict[str, str] = {}

    for gf in gen_files:
        initial_gen_code[gf.name] = _read_text_safe(gf)
        matches = _find_file_in_repo(sandbox_dir, gf.name)
        if matches:
            replacement_map[gf.name] = matches[0]
            original_repo_code[gf.name] = _read_text_safe(matches[0])
            matches[0].write_text(initial_gen_code[gf.name])
            yield _sse({
                "type": "info",
                "stage": "sandbox_build",
                "message": (
                    f"Replaced {matches[0].relative_to(sandbox_dir)} "
                    f"with generated {gf.name}"
                ),
            })
        else:
            yield _sse({
                "type": "info",
                "stage": "sandbox_build",
                "message": (
                    f"Warning: {gf.name} not found in repository — "
                    "copying to sandbox root."
                ),
            })
            (sandbox_dir / gf.name).write_text(initial_gen_code[gf.name])
            replacement_map[gf.name] = sandbox_dir / gf.name

    # --- Detect build system relative to injected file locations -------------
    injected_locations = [p for p in replacement_map.values()]
    yield _sse({
        "type": "info",
        "stage": "sandbox_build",
        "message": "Detecting build system…",
    })
    build_info = _detect_build_system(sandbox_dir, injected_paths=injected_locations)
    sandbox_cc = _detect_cross_compiler(build_info)
    is_cross = sandbox_cc != SANDBOX_CC_NATIVE
    cc_label = sandbox_cc.split()[0]

    yield _sse({
        "type": "info",
        "stage": "sandbox_build",
        "message": (
            f"Detected build system: {build_info['type']}"
            + (f" ({build_info['path'].relative_to(sandbox_dir)})"
               if build_info['path'] else "")
            + (f" | cross-compiler: {cc_label}" if is_cross else "")
        ),
    })

    # --- Scope: only track .c/.h files under the build directory tree -------
    build_root = build_info["build_dir"] if build_info["type"] != "none" else sandbox_dir
    yield _sse({
        "type": "info",
        "stage": "sandbox_build",
        "message": (
            f"Indexing source files under "
            f"{build_root.relative_to(sandbox_dir) or '.'}…"
        ),
    })

    original_repo_files: dict[Path, str] = {}
    _file_index: dict[str, list[Path]] = {}
    for p in build_root.rglob("*.[ch]"):
        if p.is_file():
            original_repo_files[p] = p.read_text(errors="replace")
            _file_index.setdefault(p.name, []).append(p)

    if build_info["type"] == "none":
        yield _sse({
            "type": "info",
            "stage": "sandbox_build",
            "message": (
                "No Makefile or CMakeLists.txt found — "
                "skipping compilation loop. Repository packaged as-is."
            ),
        })
        _package_sandbox_zip(session_dir, sandbox_dir)
        yield _sse({
            "type": "info",
            "stage": "sandbox_build",
            "message": "Repository ZIP packaged (no build verification).",
        })
        return

    build_log_lines: list[str] = [
        "=" * 65,
        "SANDBOX BUILD LOG",
        "=" * 65,
        f"\nBuild system:  {build_info['type']}",
        f"Build file:    {build_info['path'].relative_to(sandbox_dir) if build_info['path'] else 'N/A'}",
        f"Compiler:      {sandbox_cc}",
        f"Cross-compile: {'yes' if is_cross else 'no'}",
        f"Files injected: {', '.join(sorted(gen_filenames))}",
        "",
    ]

    # --- System prompts for generated-file fixes and repo-file patches ------
    cross_note = (
        f"The sandbox is cross-compiling with {cc_label}. "
        "If you see linker errors about missing BSP symbols (Xil_*, "
        "xil_printf, etc.) those are expected — the sandbox only "
        "validates that source files compile to .o successfully. "
        "Focus on fixing COMPILE errors, not link errors.\n"
        if is_cross else ""
    )

    gen_fix_system = (
        "You are an expert C programmer. You are transforming C code to "
        "comply with a new ICD (Interface Control Document) version. The "
        "transformed code failed to compile inside its repository.\n\n"
        "You will receive:\n"
        "- The ORIGINAL working code (compiled successfully before ICD changes)\n"
        "- The current transformed code that failed to compile\n"
        "- The compiler errors\n"
        "- The ICD change specification\n\n"
        "Your job: produce a corrected version that applies ALL ICD changes "
        "from the change specification AND compiles cleanly.\n\n"
        "TARGET TOOLCHAIN:\n"
        f"- Compiler: {sandbox_cc}\n"
        "- Xilinx SDK 2018.x with GCC 7.3.1 (arm-none-eabi / mb-gcc)\n"
        "- C standard: C99 with GCC extensions (-std=gnu99) — "
        "GCC-specific constructs commonly used in embedded code are "
        "ACCEPTED, including unnamed structs/unions, `__attribute__`, "
        "statement expressions, and M_PI / M_E from <math.h>\n"
        "- C library: newlib (NOT glibc)\n"
        "- Use <stdint.h> fixed-width types\n"
        "- No POSIX headers — embedded freestanding\n"
        f"{cross_note}\n"
        "RULES:\n"
        "1. Output ONLY the complete, corrected C source file\n"
        "2. Start from the ORIGINAL working code and apply ICD changes\n"
        "3. Fix every compiler error — do NOT reproduce the same mistakes\n"
        "4. Preserve the code's architecture and naming conventions\n"
        "5. Ensure #include paths match the repository\n"
        "6. Wrap the output in ```c ... ``` fences"
    )

    repo_fix_system = (
        "You are an expert C programmer. A repository's API headers were "
        "updated for a new ICD version. Some repository source files that "
        "depend on these headers now fail to compile.\n\n"
        "Your job: make MINIMAL changes to the given source file so it "
        "compiles with the updated headers. Do NOT change the program's "
        "logic — only adapt it to use the new types, struct fields, "
        "function signatures, enum values, and macros from the new headers.\n\n"
        "TARGET TOOLCHAIN:\n"
        "- Xilinx SDK 2018.x with GCC 7.3.1\n"
        "- C standard: C99\n\n"
        "RULES:\n"
        "1. Output ONLY the complete, corrected source file\n"
        "2. Change ONLY what is necessary to fix compiler errors\n"
        "3. Initialise any new required struct fields to sensible defaults\n"
        "4. Preserve all existing program logic and behaviour\n"
        "5. Wrap the output in ```c ... ``` fences"
    )

    STALL_THRESHOLD = 3
    prev_error_sig: str | None = None
    stall_count = 0
    escalation_level = 0
    error_history: list[str] = []
    repo_files_patched: dict[Path, str] = {}

    # ---------------------------------------------------------------------
    # Agentic-AI debugging pipeline (state machine, one hypothesis per iter)
    #
    # Treats debugging as a search problem over constrained edits:
    #   BUILD → TRIAGE → ROOT_CAUSE → PLAN → PATCH → VERIFY → DECIDE
    # with snapshots+rollback, per-attempt artifacts, and a memory store.
    # Default ON (SANDBOX_USE_AGENTIC=1); takes precedence over the
    # ReAct-style orchestrator. Set SANDBOX_USE_AGENTIC=0 to opt out.
    # ---------------------------------------------------------------------
    if SANDBOX_USE_AGENTIC:
        gen_files_brief: dict[str, str] = {}
        for gname, gpath in replacement_map.items():
            try:
                rel = str(gpath.relative_to(sandbox_dir))
            except ValueError:
                rel = gname
            gen_files_brief[rel] = gname

        def _agentic_runner_factory(local_log: list[str]):
            attempts = {"n": 0}

            def _runner() -> tuple[bool, str]:
                attempts["n"] += 1
                local_log.append("-" * 65)
                local_log.append(f"BUILD ATTEMPT {attempts['n']}")
                local_log.append("-" * 65)
                ok, out = _run_sandbox_build(sandbox_dir, build_info)
                local_log.append(f"\n{out}\n")
                local_log.append(
                    f"\n>>> BUILD {'SUCCEEDED' if ok else 'FAILED'}"
                    f" on attempt {attempts['n']}\n"
                )
                return ok, out
            return _runner, attempts

        agentic_total_steps = 0
        agentic_total_builds = 0
        success_via_agentic = False
        agentic_outer = retry_plan["agentic_outer"]
        agentic_attempts = retry_plan["agentic_attempts"]

        outer_round = 0
        while True:
            outer_round += 1
            if agentic_outer is not None and outer_round > agentic_outer:
                break
            round_label = (
                f"{outer_round}/∞" if agentic_outer is None
                else f"{outer_round}/{agentic_outer}"
            )
            attempts_label = (
                "unlimited" if agentic_attempts is None
                else str(agentic_attempts)
            )
            yield _sse({
                "type": "info",
                "stage": "sandbox_build",
                "message": (
                    f"Starting agentic debug pipeline "
                    f"(round {round_label}) — "
                    f"max {attempts_label} attempts, "
                    f"no-progress limit {SANDBOX_AGENTIC_NO_PROGRESS}…"
                ),
            })
            build_log_lines.append("=" * 65)
            build_log_lines.append(
                f"AGENTIC ROUND {outer_round}"
            )
            build_log_lines.append("=" * 65)

            runner, attempts_state = _agentic_runner_factory(build_log_lines)

            round_done_event: dict | None = None
            try:
                for evt in _run_agentic_debug(
                    session_dir=session_dir,
                    sandbox_dir=sandbox_dir,
                    build_info=build_info,
                    sandbox_cc=sandbox_cc,
                    is_cross=is_cross,
                    gen_files=gen_files_brief,
                    change_spec=change_spec,
                    repo_knowledge=repo_knowledge,
                    file_index=_file_index,
                    snapshots=original_repo_files,
                    build_runner=runner,
                    llm_stream=_call_llm_stream,
                    max_attempts=agentic_attempts,
                    no_progress_limit=SANDBOX_AGENTIC_NO_PROGRESS,
                    oscillation_limit=SANDBOX_AGENTIC_OSCILLATION,
                    edit_budget_files=SANDBOX_AGENTIC_EDIT_BUDGET,
                ):
                    et = evt.get("type")
                    if et == "step":
                        yield _sse({
                            "type": "info",
                            "stage": "sandbox_build",
                            "message": (
                                f"agentic attempt {evt['step']}/"
                                f"{attempts_label}"
                            ),
                        })
                    elif et == "phase":
                        build_log_lines.append(
                            f"[phase:{evt['step']}] {evt['phase']}"
                        )
                        yield _sse({
                            "type": "info",
                            "stage": "sandbox_build",
                            "message": (
                                f"step {evt['step']}: phase={evt['phase']}"
                            ),
                        })
                    elif et == "thought":
                        snippet = evt["text"]
                        if len(snippet) > 800:
                            snippet = snippet[:800] + " …"
                        build_log_lines.append(
                            f"[think:{evt['step']}] {snippet}"
                        )
                        yield _sse({
                            "type": "token",
                            "stage": "sandbox_build",
                            "token": (
                                f"\n--- think (step {evt['step']}) ---\n"
                                f"{snippet}\n"
                            ),
                        })
                    elif et == "action":
                        try:
                            args_summary = json.dumps(
                                evt["args"], ensure_ascii=False,
                            )
                        except Exception:
                            args_summary = str(evt["args"])
                        if len(args_summary) > 600:
                            args_summary = args_summary[:600] + " …"
                        build_log_lines.append(
                            f"[action:{evt['step']}] {evt['tool']} "
                            f"{args_summary}"
                        )
                        yield _sse({
                            "type": "info",
                            "stage": "sandbox_build",
                            "message": (
                                f"step {evt['step']}: {evt['tool']}"
                            ),
                        })
                    elif et == "observation":
                        otext = evt["text"]
                        if len(otext) > 1500:
                            otext = otext[:1500] + " …"
                        build_log_lines.append(
                            f"[obs:{evt['step']}{' ERR' if evt.get('error') else ''}] "
                            f"{otext}"
                        )
                        sse_obs = _tail_truncate_for_sse(
                            evt["text"],
                            SANDBOX_SSE_MAX_BUILD_LOG_CHARS,
                            "agentic_observation",
                        )
                        yield _sse({
                            "type": "token",
                            "stage": "sandbox_build",
                            "token": (
                                f"\n--- observation (step {evt['step']}) ---\n"
                                f"{sse_obs}\n"
                            ),
                        })
                    elif et == "build":
                        build_log_lines.append(
                            f"[build call #{evt['calls']}] "
                            f"{'OK' if evt['success'] else 'FAIL'}"
                        )
                        yield _sse({
                            "type": "info",
                            "stage": "sandbox_build",
                            "message": (
                                f"build call {evt['calls']}: "
                                f"{'success' if evt['success'] else 'failed'}"
                            ),
                        })
                    elif et == "raw_token":
                        yield _sse({
                            "type": "token",
                            "stage": "sandbox_build",
                            "token": evt["text"],
                        })
                    elif et == "warning":
                        build_log_lines.append(
                            f"[warning] {evt['message']}"
                        )
                        yield _sse({
                            "type": "info",
                            "stage": "sandbox_build",
                            "message": evt["message"],
                        })
                    elif et == "done":
                        round_done_event = evt
                        break
            except Exception as e:
                log.exception("Agentic round %d crashed", outer_round)
                build_log_lines.append(f"[agentic crash] {e}")
                yield _sse({
                    "type": "info",
                    "stage": "sandbox_build",
                    "message": f"Agentic pipeline crashed: {e}",
                })
                round_done_event = {
                    "type": "done",
                    "success": False,
                    "reason": f"crash: {e}",
                    "steps": 0,
                    "builds": attempts_state["n"],
                }

            if round_done_event is None:
                round_done_event = {
                    "type": "done",
                    "success": False,
                    "reason": "round ended without explicit done event",
                    "steps": (
                        agentic_attempts
                        if agentic_attempts is not None
                        else attempts_state["n"]
                    ),
                    "builds": attempts_state["n"],
                }

            agentic_total_steps += int(round_done_event.get("steps", 0))
            agentic_total_builds += int(round_done_event.get("builds", 0))
            build_log_lines.append(
                f"\n>>> Round {outer_round} ended: "
                f"{round_done_event['reason']}\n"
            )
            yield _sse({
                "type": "info",
                "stage": "sandbox_build",
                "message": (
                    f"agentic round {outer_round} ended — "
                    f"{round_done_event['reason']}"
                ),
            })

            if round_done_event.get("success"):
                success_via_agentic = True
                break

            # Round failed: reset to initial snapshot before next round.
            more_rounds = (
                agentic_outer is None or outer_round < agentic_outer
            )
            if more_rounds:
                yield _sse({
                    "type": "info",
                    "stage": "sandbox_build",
                    "message": (
                        "Resetting all files to initial state for next "
                        "agentic round…"
                    ),
                })
                for fname in gen_filenames:
                    if fname in initial_gen_code:
                        (gen_dir / fname).write_text(initial_gen_code[fname])
                        if fname in replacement_map:
                            replacement_map[fname].write_text(
                                initial_gen_code[fname]
                            )
                for rpath, orig_content in original_repo_files.items():
                    if rpath.name not in gen_filenames:
                        try:
                            rpath.write_text(orig_content)
                        except OSError:
                            pass
                build_log_lines.append(
                    "   Reset all files to initial snapshot before next round.\n"
                )

        if not success_via_agentic:
            rounds_done = outer_round
            rounds_label = (
                "indefinite" if agentic_outer is None
                else str(agentic_outer)
            )
            build_log_lines.extend([
                "",
                "=" * 65,
                "SUMMARY (AGENTIC)",
                "=" * 65,
                f"Agentic rounds: {rounds_done} (budget={rounds_label})",
                f"Total agent steps: {agentic_total_steps}",
                f"Total build calls: {agentic_total_builds}",
                f"Result: BUILD STILL FAILING (best effort packaged)",
                "",
                "Per-attempt artifacts: see ./agentic_attempts/attempt_NN/",
                "Memory store:           see ./playbook.jsonl",
                "",
            ])
            yield _sse({
                "type": "info",
                "stage": "sandbox_build",
                "message": (
                    "Agentic pipeline did not converge — packaging "
                    "best-effort version of the repository for download."
                ),
            })
            yield _sse({
                "type": "info",
                "stage": "sandbox_build",
                "message": "Packaging best-effort repository…",
            })
            _package_sandbox_zip(session_dir, sandbox_dir)
            build_log_path = session_dir / "sandbox_build_log.txt"
            build_log_path.write_text("\n".join(build_log_lines) + "\n")
            yield _sse({
                "type": "sandbox_build_result",
                "stage": "sandbox_build",
                "success": False,
                "iterations": agentic_total_builds,
                "message": (
                    "Sandbox build did not converge after "
                    f"{rounds_done} agentic round(s); "
                    "best-effort repository packaged."
                ),
            })
            return

        build_log_lines.extend([
            "",
            "=" * 65,
            "SUMMARY (AGENTIC)",
            "=" * 65,
            f"Agentic rounds: {outer_round}",
            f"Total agent steps: {agentic_total_steps}",
            f"Total build calls: {agentic_total_builds}",
            f"Result: BUILD SUCCEEDED",
            "",
            "Per-attempt artifacts: see ./agentic_attempts/attempt_NN/",
            "Memory store:           see ./playbook.jsonl",
            "",
        ])
        yield _sse({
            "type": "info",
            "stage": "sandbox_build",
            "message": "Packaging built repository…",
        })
        _package_sandbox_zip(session_dir, sandbox_dir)
        build_log_path = session_dir / "sandbox_build_log.txt"
        build_log_path.write_text("\n".join(build_log_lines) + "\n")
        yield _sse({
            "type": "sandbox_build_result",
            "stage": "sandbox_build",
            "success": True,
            "iterations": agentic_total_builds,
            "message": (
                f"Sandbox build succeeded via agentic round "
                f"{outer_round} — repository packaged."
            ),
        })
        return

    # ---------------------------------------------------------------------
    # Orchestrator-driven debugging loop
    # ---------------------------------------------------------------------
    if SANDBOX_USE_ORCHESTRATOR:
        orch_started = time.monotonic()
        gen_files_brief = {}
        for gname, gpath in replacement_map.items():
            try:
                rel = str(gpath.relative_to(sandbox_dir))
            except ValueError:
                rel = gname
            gen_files_brief[rel] = gname

        def _build_runner_factory(local_log: list[str]):
            attempts = {"n": 0}

            def _runner() -> tuple[bool, str]:
                attempts["n"] += 1
                local_log.append("-" * 65)
                local_log.append(f"BUILD ATTEMPT {attempts['n']}")
                local_log.append("-" * 65)
                ok, out = _run_sandbox_build(sandbox_dir, build_info)
                local_log.append(f"\n{out}\n")
                local_log.append(
                    f"\n>>> BUILD {'SUCCEEDED' if ok else 'FAILED'}"
                    f" on attempt {attempts['n']}\n"
                )
                return ok, out
            return _runner, attempts

        success_via_orchestrator = False
        orch_total_steps = 0
        orch_total_builds = 0
        orch_outer = retry_plan["orch_outer"]
        orch_builds = retry_plan["orch_builds"]

        outer_round = 0
        while True:
            outer_round += 1
            if orch_outer is not None and outer_round > orch_outer:
                break
            round_label = (
                f"{outer_round}/∞" if orch_outer is None
                else f"{outer_round}/{orch_outer}"
            )
            builds_label = (
                "unlimited" if orch_builds is None else str(orch_builds)
            )
            step_phrase = (
                f"max {SANDBOX_ORCH_MAX_STEPS} steps, "
                if SANDBOX_ORCH_MAX_STEPS is not None
                else "no step limit, "
            )
            yield _sse({
                "type": "info",
                "stage": "sandbox_build",
                "message": (
                    "Starting debugging orchestrator "
                    f"(round {round_label}) — "
                    f"{step_phrase}{builds_label} builds…"
                ),
            })
            build_log_lines.append("=" * 65)
            build_log_lines.append(
                f"ORCHESTRATOR ROUND {outer_round}"
            )
            build_log_lines.append("=" * 65)

            runner, attempts_state = _build_runner_factory(build_log_lines)

            round_done_event: dict | None = None
            try:
                for evt in _run_orchestrator(
                    sandbox_dir=sandbox_dir,
                    build_info=build_info,
                    sandbox_cc=sandbox_cc,
                    is_cross=is_cross,
                    gen_files=gen_files_brief,
                    change_spec=change_spec,
                    repo_knowledge=repo_knowledge,
                    file_index=_file_index,
                    snapshots=original_repo_files,
                    build_runner=runner,
                    llm_stream=_call_llm_stream,
                    max_steps=SANDBOX_ORCH_MAX_STEPS,
                    max_builds=orch_builds,
                    max_input_tokens=MAX_INPUT_TOKENS,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                ):
                    et = evt.get("type")
                    if et == "step":
                        if SANDBOX_ORCH_MAX_STEPS is None:
                            msg = f"orchestrator step {evt['step']}"
                        else:
                            msg = (
                                f"orchestrator step {evt['step']}/"
                                f"{SANDBOX_ORCH_MAX_STEPS}"
                            )
                        yield _sse({
                            "type": "info",
                            "stage": "sandbox_build",
                            "message": msg,
                        })
                    elif et == "thought":
                        snippet = evt["text"]
                        if len(snippet) > 800:
                            snippet = snippet[:800] + " …"
                        build_log_lines.append(
                            f"[think:{evt['step']}] {snippet}"
                        )
                        yield _sse({
                            "type": "token",
                            "stage": "sandbox_build",
                            "token": (
                                f"\n--- think (step {evt['step']}) ---\n"
                                f"{snippet}\n"
                            ),
                        })
                    elif et == "action":
                        try:
                            args_summary = json.dumps(
                                evt["args"], ensure_ascii=False,
                            )
                        except Exception:
                            args_summary = str(evt["args"])
                        if len(args_summary) > 600:
                            args_summary = args_summary[:600] + " …"
                        build_log_lines.append(
                            f"[action:{evt['step']}] {evt['tool']} "
                            f"{args_summary}"
                        )
                        yield _sse({
                            "type": "info",
                            "stage": "sandbox_build",
                            "message": (
                                f"step {evt['step']}: {evt['tool']}"
                            ),
                        })
                    elif et == "observation":
                        otext = evt["text"]
                        if len(otext) > 1500:
                            otext = otext[:1500] + " …"
                        build_log_lines.append(
                            f"[obs:{evt['step']}{' ERR' if evt.get('error') else ''}] "
                            f"{otext}"
                        )
                        sse_obs = _tail_truncate_for_sse(
                            evt["text"],
                            SANDBOX_SSE_MAX_BUILD_LOG_CHARS,
                            "orchestrator_observation",
                        )
                        yield _sse({
                            "type": "token",
                            "stage": "sandbox_build",
                            "token": (
                                f"\n--- observation (step {evt['step']}) ---\n"
                                f"{sse_obs}\n"
                            ),
                        })
                    elif et == "build":
                        build_log_lines.append(
                            f"[build call #{evt['calls']}] "
                            f"{'OK' if evt['success'] else 'FAIL'}"
                        )
                        yield _sse({
                            "type": "info",
                            "stage": "sandbox_build",
                            "message": (
                                f"build call {evt['calls']}: "
                                f"{'success' if evt['success'] else 'failed'}"
                            ),
                        })
                    elif et == "raw_token":
                        yield _sse({
                            "type": "token",
                            "stage": "sandbox_build",
                            "token": evt["text"],
                        })
                    elif et == "warning":
                        build_log_lines.append(
                            f"[warning] {evt['message']}"
                        )
                        yield _sse({
                            "type": "info",
                            "stage": "sandbox_build",
                            "message": evt["message"],
                        })
                    elif et == "done":
                        round_done_event = evt
                        break
            except Exception as e:
                log.exception("Orchestrator round %d crashed", outer_round)
                build_log_lines.append(f"[orchestrator crash] {e}")
                yield _sse({
                    "type": "info",
                    "stage": "sandbox_build",
                    "message": f"Orchestrator crashed: {e}",
                })
                round_done_event = {
                    "type": "done",
                    "success": False,
                    "reason": f"crash: {e}",
                    "steps": 0,
                    "builds": attempts_state["n"],
                }

            if round_done_event is None:
                round_done_event = {
                    "type": "done",
                    "success": False,
                    "reason": "round ended without explicit done event",
                    "steps": SANDBOX_ORCH_MAX_STEPS,
                    "builds": attempts_state["n"],
                }

            orch_total_steps += int(round_done_event.get("steps", 0))
            orch_total_builds += int(round_done_event.get("builds", 0))
            build_log_lines.append(
                f"\n>>> Round {outer_round} ended: "
                f"{round_done_event['reason']}\n"
            )
            yield _sse({
                "type": "info",
                "stage": "sandbox_build",
                "message": (
                    f"orchestrator round {outer_round} ended — "
                    f"{round_done_event['reason']}"
                ),
            })

            if round_done_event.get("success"):
                success_via_orchestrator = True
                break

            # --- Round failed: reset and let next round try fresh -----------
            more_rounds = orch_outer is None or outer_round < orch_outer
            if more_rounds:
                yield _sse({
                    "type": "info",
                    "stage": "sandbox_build",
                    "message": (
                        "Resetting all files to initial state for next round…"
                    ),
                })
                for fname in gen_filenames:
                    if fname in initial_gen_code:
                        (gen_dir / fname).write_text(initial_gen_code[fname])
                        if fname in replacement_map:
                            replacement_map[fname].write_text(
                                initial_gen_code[fname]
                            )
                for rpath, orig_content in original_repo_files.items():
                    if rpath.name not in gen_filenames:
                        try:
                            rpath.write_text(orig_content)
                        except OSError:
                            pass
                build_log_lines.append(
                    "   Reset all files to initial snapshot before next round.\n"
                )

        iteration = max(1, orch_total_builds)
        orch_runtime_s = max(0.0, time.monotonic() - orch_started)
        orch_runtime_h = orch_runtime_s / 3600.0
        if not success_via_orchestrator:
            rounds_done = outer_round
            rounds_label = (
                "indefinite" if orch_outer is None else str(orch_outer)
            )
            build_log_lines.extend([
                "",
                "=" * 65,
                "SUMMARY",
                "=" * 65,
                f"Orchestrator rounds: {rounds_done} (budget={rounds_label})",
                f"Total agent steps: {orch_total_steps}",
                f"Total build calls: {orch_total_builds}",
                f"Runtime (hours): {orch_runtime_h:.2f}",
                f"Result: BUILD STILL FAILING (best effort packaged)",
                "",
            ])
            yield _sse({
                "type": "info",
                "stage": "sandbox_build",
                "message": (
                    "Orchestrator did not converge — packaging best-effort "
                    "version of the repository for download."
                ),
            })
            yield _sse({
                "type": "info",
                "stage": "sandbox_build",
                "message": "Packaging best-effort repository…",
            })
            _package_sandbox_zip(session_dir, sandbox_dir)
            build_log_path = session_dir / "sandbox_build_log.txt"
            build_log_path.write_text("\n".join(build_log_lines) + "\n")
            yield _sse({
                "type": "sandbox_build_result",
                "stage": "sandbox_build",
                "success": False,
                "iterations": orch_total_builds,
                "runtime_hours": round(orch_runtime_h, 3),
                "message": (
                    "Sandbox build did not converge after "
                    f"{rounds_done} orchestrator round(s); "
                    f"best-effort repository packaged. Runtime: {orch_runtime_h:.2f}h."
                ),
            })
            return

        build_log_lines.extend([
            "",
            "=" * 65,
            "SUMMARY",
            "=" * 65,
            f"Orchestrator rounds: {outer_round}",
            f"Total agent steps: {orch_total_steps}",
            f"Total build calls: {orch_total_builds}",
            f"Runtime (hours): {orch_runtime_h:.2f}",
            f"Result: BUILD SUCCEEDED",
            "",
        ])

        yield _sse({
            "type": "info",
            "stage": "sandbox_build",
            "message": "Packaging built repository…",
        })
        _package_sandbox_zip(session_dir, sandbox_dir)
        build_log_path = session_dir / "sandbox_build_log.txt"
        build_log_path.write_text("\n".join(build_log_lines) + "\n")
        yield _sse({
            "type": "sandbox_build_result",
            "stage": "sandbox_build",
            "success": True,
            "iterations": orch_total_builds,
            "runtime_hours": round(orch_runtime_h, 3),
            "message": (
                f"Sandbox build succeeded after orchestrator round "
                f"{outer_round} — repository packaged. Runtime: {orch_runtime_h:.2f}h."
            ),
        })
        return

    # ---------------------------------------------------------------------
    # Legacy per-file fix loop (set SANDBOX_USE_ORCHESTRATOR=0 to use it)
    # ---------------------------------------------------------------------
    iteration = 0
    legacy_max = retry_plan["legacy_max"]
    while True:
        iteration += 1
        if legacy_max is not None and iteration > legacy_max:
            build_log_lines.append(
                f"\n>>> BUILD RETRY BUDGET EXHAUSTED after {legacy_max} "
                "attempt(s)\n"
            )
            yield _sse({
                "type": "info",
                "stage": "sandbox_build",
                "message": (
                    f"Sandbox retry budget exhausted after {legacy_max} "
                    "attempt(s) — packaging best-effort repository."
                ),
            })
            _package_sandbox_zip(session_dir, sandbox_dir)
            build_log_path = session_dir / "sandbox_build_log.txt"
            build_log_path.write_text("\n".join(build_log_lines) + "\n")
            yield _sse({
                "type": "sandbox_build_result",
                "stage": "sandbox_build",
                "success": False,
                "iterations": legacy_max,
                "message": (
                    f"Sandbox build did not succeed within {legacy_max} "
                    "reiteration(s); best-effort repository packaged."
                ),
            })
            return
        yield _sse({
            "type": "info",
            "stage": "sandbox_build",
            "message": (
                f"Build attempt {iteration}"
                + (f"/{legacy_max}" if legacy_max is not None else "")
                + "…"
            ),
        })

        build_log_lines.append("-" * 65)
        build_log_lines.append(f"BUILD ATTEMPT {iteration}")
        build_log_lines.append("-" * 65)

        success, output = _run_sandbox_build(sandbox_dir, build_info)

        build_log_lines.append(f"\n{output}\n")

        sse_out = _tail_truncate_for_sse(
            output, SANDBOX_SSE_MAX_BUILD_LOG_CHARS, "sandbox_build",
        )
        yield _sse({
            "type": "token",
            "stage": "sandbox_build",
            "token": f"\n--- Build output (attempt {iteration}) ---\n{sse_out}\n",
        })

        if success:
            build_log_lines.append(f"\n>>> BUILD SUCCEEDED on attempt {iteration}\n")
            yield _sse({
                "type": "info",
                "stage": "sandbox_build",
                "message": f"Build succeeded on attempt {iteration}.",
            })
            break

        build_log_lines.append(f"\n>>> BUILD FAILED on attempt {iteration}\n")

        # --- Stall detection & escalation -----------------------------------
        error_sig = _normalise_error_signature(output)
        if error_sig and error_sig == prev_error_sig:
            stall_count += 1
        else:
            stall_count = 0
        prev_error_sig = error_sig

        escalated = False
        if stall_count >= STALL_THRESHOLD:
            escalation_level += 1
            stall_count = 0

            error_history.append(
                f"[Escalation {escalation_level}] Errors persisting after "
                f"attempt {iteration}:\n{_truncate_text(output, 3000, 'build_output')}"
            )

            build_log_lines.append(
                f"\n>>> STALL DETECTED — escalating fix strategy "
                f"(escalation {escalation_level})\n"
            )
            yield _sse({
                "type": "info",
                "stage": "sandbox_build",
                "message": (
                    f"Same errors repeated {STALL_THRESHOLD + 1} times — "
                    f"escalating fix strategy (level {escalation_level})…"
                ),
            })
            escalated = True

            # Reset generated files to the initial transform output so the
            # LLM starts fresh without accumulated drift.
            for fname in gen_filenames:
                if fname in initial_gen_code:
                    (gen_dir / fname).write_text(initial_gen_code[fname])
                    if fname in replacement_map:
                        replacement_map[fname].write_text(initial_gen_code[fname])

            # Reset non-generated repo files to originals so stale patches
            # don't compound.
            for rpath, orig_content in original_repo_files.items():
                if rpath.name not in gen_filenames:
                    rpath.write_text(orig_content)
            repo_files_patched.clear()

            build_log_lines.append(
                "   Reset all files to initial state for fresh approach.\n"
            )

        yield _sse({
            "type": "info",
            "stage": "sandbox_build",
            "message": f"Build failed — feeding errors to LLM for attempt {iteration + 1}…",
        })

        # --- Parse errors for ALL files -------------------------------------
        all_errors = _parse_all_build_errors(output, sandbox_dir, file_index=_file_index)

        gen_errors: dict[str, str] = {}
        repo_errors: dict[Path, str] = {}
        for fpath, err_text in all_errors.items():
            if fpath.name in gen_filenames:
                gen_errors[fpath.name] = err_text
            else:
                repo_errors[fpath] = err_text

        if not gen_errors and not repo_errors:
            first = sorted(gen_filenames)[0]
            gen_errors[first] = output

        # Collect the current generated headers to show repo-file fixer
        current_gen_headers = ""
        for fname in sorted(gen_filenames):
            if fname.endswith(".h"):
                hpath = replacement_map.get(fname)
                if hpath and hpath.exists():
                    current_gen_headers += (
                        f"### {fname}\n```c\n{_read_text_safe(hpath)}\n```\n\n"
                    )

        # --- Fix generated files (ICD re-transform with error feedback) -----
        for fname, errors in gen_errors.items():
            current_code = _read_text_safe(gen_dir / fname)
            orig_code = original_repo_code.get(fname, "")

            build_log_lines.append(
                f"\n--- LLM fix (generated): {fname} (after attempt {iteration}) ---"
            )

            file_repo_ctx = ""
            if has_repo:
                file_repo_ctx = _build_file_repo_context(
                    repo_dir, current_code, uploaded_names,
                    max_chars=MAX_REPO_CONTEXT_CHARS,
                )
                if not file_repo_ctx:
                    file_repo_ctx = _build_repo_context(
                        repo_dir, exclude_names=uploaded_names,
                    )

            sec_original = (
                f"## Original Working Code ({fname})\n"
                f"This code compiled successfully before ICD changes:\n"
                f"```c\n{orig_code}\n```"
                if orig_code else ""
            )
            sec_current = (
                f"## Current Transformed Code ({fname})\n"
                f"This version failed to compile:\n"
                f"```c\n{current_code}\n```\n\n"
                "Produce a corrected version that applies ALL ICD changes "
                "and compiles cleanly."
            )
            sec_errors = (
                f"## Compiler Errors\n```\n"
                f"{_truncate_text(errors, 6000, 'compiler_errors')}\n```"
            )
            sec_repo_ctx = (
                f"## Repository Dependency Context\n{file_repo_ctx}"
                if file_repo_ctx else ""
            )
            sec_knowledge = (
                f"## Repository Codebase Knowledge\n{repo_knowledge}"
                if repo_knowledge else ""
            )
            sec_change = f"## Change Specification\n{change_spec}"

            sec_escalation = ""
            if escalated and error_history:
                history_text = "\n\n".join(error_history[-3:])
                sec_escalation = (
                    f"## CRITICAL — Previous Fix Attempts Failed\n"
                    f"The following errors have persisted across multiple fix "
                    f"attempts (escalation {escalation_level}). You MUST try "
                    f"a fundamentally different strategy.\n\n"
                    f"### Error History\n{history_text}"
                )

            fix_sections = [
                ("original", sec_original, 0),
                ("current", sec_current, 0),
                ("errors", sec_errors, 0),
                ("escalation", sec_escalation, 0) if sec_escalation else ("escalation", "", 99),
                ("repo_ctx", sec_repo_ctx, 1),
                ("knowledge", sec_knowledge, 2),
                ("change_spec", sec_change, 3),
            ]
            sys_tokens = _estimate_tokens(gen_fix_system)
            fix_prompt = _assemble_prompt(
                fix_sections,
                max_input_tokens=MAX_INPUT_TOKENS - sys_tokens,
            )

            yield _sse({
                "type": "info",
                "stage": "sandbox_build",
                "message": f"LLM fixing generated file {fname}…",
            })

            try:
                fix_parts: list[str] = []
                meta: dict = {}
                for batch in _batched_stream_text(
                    _call_llm_stream(
                        gen_fix_system, fix_prompt,
                        max_tokens=MAX_OUTPUT_TOKENS, meta=meta,
                    )
                ):
                    fix_parts.append(batch)
                    yield _sse({
                        "type": "token",
                        "stage": "sandbox_build",
                        "token": batch,
                    })
                fix_output = "".join(fix_parts).strip()

                for cont_pass in range(1, 4):
                    if meta.get("finish_reason") != "length":
                        break
                    log.info("Sandbox fix for %s: pass %d truncated, continuing…",
                             fname, cont_pass)
                    yield _sse({
                        "type": "info",
                        "stage": "sandbox_build",
                        "message": f"Response truncated — continuing pass {cont_pass + 1}…",
                    })
                    tail = fix_output[-2000:] if len(fix_output) > 2000 else fix_output
                    cont_prompt = (
                        "Your previous response was cut off. "
                        f"End of what you wrote:\n---\n{tail}\n---\n\n"
                        "Continue EXACTLY from where you left off."
                    )
                    meta = {}
                    for batch in _batched_stream_text(
                        _call_llm_stream(
                            gen_fix_system, cont_prompt,
                            max_tokens=MAX_OUTPUT_TOKENS, meta=meta,
                        )
                    ):
                        fix_parts.append(batch)
                        yield _sse({
                            "type": "token",
                            "stage": "sandbox_build",
                            "token": batch,
                        })
                    fix_output = "".join(fix_parts).strip()

                fixed_code = _extract_fenced(fix_output, "c").strip()
                ref_code = orig_code if orig_code else current_code
                if fixed_code and _looks_complete_c_file(
                    fixed_code, ref_code, fname
                ):
                    (gen_dir / fname).write_text(fixed_code)
                    if fname in replacement_map:
                        replacement_map[fname].write_text(fixed_code)
                    build_log_lines.append(f"Applied LLM fix to {fname}")
                    yield _sse({
                        "type": "info",
                        "stage": "sandbox_build",
                        "message": f"Updated {fname} with LLM fix.",
                    })
                else:
                    build_log_lines.append(
                        f"LLM fix for {fname} was incomplete — kept previous version"
                    )
                    yield _sse({
                        "type": "info",
                        "stage": "sandbox_build",
                        "message": (
                            f"LLM fix for {fname} was incomplete — "
                            "keeping previous version for next attempt."
                        ),
                    })
            except Exception as e:
                log.warning("Sandbox LLM fix failed for %s: %s", fname, e)
                build_log_lines.append(f"LLM fix error for {fname}: {e}")
                yield _sse({
                    "type": "info",
                    "stage": "sandbox_build",
                    "message": f"LLM fix error for {fname}: {e}",
                })

        # --- Fix non-generated repo files (API adaptation) ------------------
        for fpath, errors in repo_errors.items():
            rel_name = str(fpath.relative_to(sandbox_dir))
            current_code = _read_text_safe(fpath)

            build_log_lines.append(
                f"\n--- LLM fix (repo): {rel_name} (after attempt {iteration}) ---"
            )

            sec_file = (
                f"## Source File ({rel_name})\n```c\n{current_code}\n```\n\n"
                "Make MINIMAL changes so this file compiles with the updated headers."
            )
            sec_headers = (
                f"## Updated Headers\n{current_gen_headers}"
                if current_gen_headers else ""
            )
            sec_errors_r = (
                f"## Compiler Errors\n```\n"
                f"{_truncate_text(errors, 6000, 'compiler_errors')}\n```"
            )

            repo_fix_sections = [
                ("file", sec_file, 0),
                ("errors", sec_errors_r, 0),
                ("headers", sec_headers, 0),
            ]
            sys_tokens = _estimate_tokens(repo_fix_system)
            repo_fix_prompt = _assemble_prompt(
                repo_fix_sections,
                max_input_tokens=MAX_INPUT_TOKENS - sys_tokens,
            )

            yield _sse({
                "type": "info",
                "stage": "sandbox_build",
                "message": f"LLM patching repo file {rel_name}…",
            })

            try:
                rfix_parts: list[str] = []
                rmeta: dict = {}
                for batch in _batched_stream_text(
                    _call_llm_stream(
                        repo_fix_system, repo_fix_prompt,
                        max_tokens=MAX_OUTPUT_TOKENS, meta=rmeta,
                    )
                ):
                    rfix_parts.append(batch)
                    yield _sse({
                        "type": "token",
                        "stage": "sandbox_build",
                        "token": batch,
                    })
                rfix_output = "".join(rfix_parts).strip()

                for cont_pass in range(1, 4):
                    if rmeta.get("finish_reason") != "length":
                        break
                    tail = rfix_output[-2000:] if len(rfix_output) > 2000 else rfix_output
                    cont_prompt = (
                        "Your previous response was cut off. "
                        f"End of what you wrote:\n---\n{tail}\n---\n\n"
                        "Continue EXACTLY from where you left off."
                    )
                    rmeta = {}
                    for batch in _batched_stream_text(
                        _call_llm_stream(
                            repo_fix_system, cont_prompt,
                            max_tokens=MAX_OUTPUT_TOKENS, meta=rmeta,
                        )
                    ):
                        rfix_parts.append(batch)
                        yield _sse({
                            "type": "token",
                            "stage": "sandbox_build",
                            "token": batch,
                        })
                    rfix_output = "".join(rfix_parts).strip()

                fixed_repo_code = _extract_fenced(rfix_output, "c").strip()
                if fixed_repo_code and len(fixed_repo_code) > 20:
                    fpath.write_text(fixed_repo_code)
                    repo_files_patched[fpath] = fixed_repo_code
                    build_log_lines.append(f"Applied repo patch to {rel_name}")
                    yield _sse({
                        "type": "info",
                        "stage": "sandbox_build",
                        "message": f"Patched repo file {rel_name}.",
                    })
                else:
                    build_log_lines.append(
                        f"Repo patch for {rel_name} was incomplete — skipped"
                    )
                    yield _sse({
                        "type": "info",
                        "stage": "sandbox_build",
                        "message": f"Repo patch for {rel_name} incomplete — skipped.",
                    })
            except Exception as e:
                log.warning("Sandbox repo fix failed for %s: %s", rel_name, e)
                build_log_lines.append(f"Repo fix error for {rel_name}: {e}")
                yield _sse({
                    "type": "info",
                    "stage": "sandbox_build",
                    "message": f"Repo fix error for {rel_name}: {e}",
                })

    build_log_lines.extend([
        "",
        "=" * 65,
        "SUMMARY",
        "=" * 65,
        f"Total build attempts: {iteration}",
        f"Escalations triggered: {escalation_level}",
        f"Result: BUILD SUCCEEDED",
        "",
    ])

    yield _sse({
        "type": "info",
        "stage": "sandbox_build",
        "message": "Packaging built repository…",
    })
    _package_sandbox_zip(session_dir, sandbox_dir)

    build_log_path = session_dir / "sandbox_build_log.txt"
    build_log_path.write_text("\n".join(build_log_lines) + "\n")

    yield _sse({
        "type": "sandbox_build_result",
        "stage": "sandbox_build",
        "success": True,
        "iterations": iteration,
        "message": f"Sandbox build succeeded on attempt {iteration} — repository packaged.",
    })

def _remote_build_iterate(
    session_dir: Path,
    gen_dir: Path,
    workspace_path: str,   # local abs path that maps to the session workspace
):
    """
    Generator yielding SSE dicts.
    Mirrors _sandbox_build_iterate()'s event contract so the caller
    (process / regenerate endpoints) works unchanged.
    """
    import paramiko   # lazy import — only required when remote build is used

    yield _sse({"type": "info", "stage": "sandbox_build",
                "message": "Remote build: connecting via SSH…"})

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(REMOTE_BUILD_HOST, username=REMOTE_BUILD_USER,
                    password=REMOTE_BUILD_PASS)
    except Exception as e:
        yield _sse({"type": "error",
                    "message": f"Remote build: SSH connection failed — {e}"})
        return

    sftp = ssh.open_sftp()

    # 1. Copy modified files to remote
    gen_files = sorted(p for p in gen_dir.iterdir()
                       if p.is_file() and p.suffix in ('.c', '.h'))
    for gf in gen_files:
        remote_path = f"{REMOTE_BUILD_SRC}/src/{gf.name}"   # adjust mapping as needed
        local_path  = str(session_dir / "generated_code" / gf.name)
        yield _sse({"type": "info", "stage": "sandbox_build",
                    "message": f"Remote build: uploading {gf.name}…"})
        try:
            sftp.put(local_path, remote_path)
        except Exception as e:
            yield _sse({"type": "error",
                        "message": f"Remote build: SFTP put {gf.name} failed — {e}"})
            ssh.close(); return

    # 2. Trigger build
    build_arg = "release" if REMOTE_BUILD_MODE == "release" else ""
    cmd = f"cd {REMOTE_BUILD_SRC} && {REMOTE_BUILD_SCRIPT} {build_arg}"
    yield _sse({"type": "info", "stage": "sandbox_build",
                "message": f"Remote build: running `{cmd}`…"})

    stdin, stdout, stderr = ssh.exec_command(cmd)
    output_lines = []
    for line in stdout:
        stripped = line.strip()
        output_lines.append(stripped)
        yield _sse({"type": "token", "stage": "sandbox_build", "token": stripped + "\n"})

    errors = stderr.read().decode()
    exit_status = stdout.channel.recv_exit_status()
    ssh.close()

    success = (exit_status == 0)
    if errors:
        yield _sse({"type": "token", "stage": "sandbox_build",
                    "token": f"\nSTDERR:\n{errors}"})

    yield _sse({
        "type": "sandbox_build_result",
        "stage": "sandbox_build",
        "success": success,
        "iterations": 1,
        "message": (
            "Remote Xilinx SDK build succeeded."
            if success else
            f"Remote build failed (exit {exit_status}). See output above."
        ),
    })

def _package_sandbox_zip(session_dir: Path, sandbox_dir: Path) -> Path:
    """Write the sandbox directory to ``session_dir/built_repo.zip``."""
    zip_path = session_dir / "built_repo.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(sandbox_dir.rglob("*")):
            if fpath.is_file():
                arcname = str(fpath.relative_to(sandbox_dir))
                zf.write(fpath, arcname)
    log.info("Packaged sandbox as %s (%d bytes)", zip_path, zip_path.stat().st_size)
    return zip_path


def _assemble_prompt(
    sections: list[tuple[str, str, int]],
    max_input_tokens: int = MAX_INPUT_TOKENS,
) -> str:
    """Assemble a prompt from prioritised sections within a token budget.

    Each section is ``(label, text, priority)`` where lower priority number
    means higher importance.  Sections are included in priority order;
    lower-priority sections are truncated or dropped to stay within budget.
    """
    sorted_secs = sorted(sections, key=lambda s: s[2])
    result_parts: list[str] = []
    remaining = max_input_tokens

    for label, text, _prio in sorted_secs:
        tokens = _estimate_tokens(text)
        if tokens <= remaining:
            result_parts.append(text)
            remaining -= tokens
        elif remaining > 200:
            trim_chars = remaining * CHARS_PER_TOKEN
            trimmed = text[:trim_chars] + f"\n\n[... {label} TRUNCATED to fit context ...]"
            result_parts.append(trimmed)
            remaining = 0
        else:
            log.warning("Dropping section '%s' (%d tokens) — no budget left", label, tokens)

    assembled = "\n\n".join(result_parts)
    log.info(
        "_assemble_prompt: %d/%d input tokens used (%d sections in, %d kept)",
        _estimate_tokens(assembled), max_input_tokens,
        len(sorted_secs), len(result_parts),
    )
    return assembled


# ---------------------------------------------------------------------------
# Web UI: bounded pipeline logs (append_log_helper.js + patched app.js)
# ---------------------------------------------------------------------------

_APP_JS_PATCHED: str | None = None


def _patched_app_js() -> str:
    """Serve app.js with appendStepLog(...) instead of unbounded textContent +=.

    On-disk :file:`static/app.js` may be read-only; patches are applied in memory.
    """
    global _APP_JS_PATCHED
    if _APP_JS_PATCHED is not None:
        return _APP_JS_PATCHED
    t = (STATIC_DIR / "app.js").read_text(encoding="utf-8", errors="replace")
    repls: list[tuple[str, str]] = [
        (
            "            if (o.textContent.startsWith('Connecting to LLM')) o.textContent = '';\n"  # noqa: E501
            "            o.textContent += msg.token;\n"
            "            o.scrollTop = o.scrollHeight;",
            "            if (o.textContent.startsWith('Connecting to LLM')) o.textContent = '';\n"  # noqa: E501
            "            appendStepLog(o, msg.token);",
        ),
        (
            "            var o = step.querySelector('.step-output');\n"
            "            o.textContent += msg.token;\n"
            "            o.scrollTop = o.scrollHeight;",
            "            var o = step.querySelector('.step-output');\n"
            "            appendStepLog(o, msg.token);",
        ),
        (
            "            if (out.textContent.startsWith('Connecting to LLM')) out.textContent = '';\n"  # noqa: E501
            "            out.textContent += '\\n' + msg.message + '\\n';\n"
            "            out.scrollTop = out.scrollHeight;",
            "            if (out.textContent.startsWith('Connecting to LLM')) out.textContent = '';\n"  # noqa: E501
            "            appendStepLog(out, '\\n' + msg.message + '\\n');",
        ),
        (
            "            const out = verificationStep.querySelector('.step-output');\n"
            "            out.textContent += '\\n' + msg.message + '\\n';\n"
            "            out.scrollTop = out.scrollHeight;",
            "            const out = verificationStep.querySelector('.step-output');\n"
            "            appendStepLog(out, '\\n' + msg.message + '\\n');",
        ),
        (
            "            const out = sandboxStep.querySelector('.step-output');\n"
            "            out.textContent += '\\n' + msg.message + '\\n';\n"
            "            out.scrollTop = out.scrollHeight;",
            "            const out = sandboxStep.querySelector('.step-output');\n"
            "            appendStepLog(out, '\\n' + msg.message + '\\n');",
        ),
        (
            "            const out = fileSteps[msg.file].querySelector('.step-output');\n"
            "            out.textContent += '\\n' + msg.message + '\\n';\n"
            "            out.scrollTop = out.scrollHeight;",
            "            const out = fileSteps[msg.file].querySelector('.step-output');\n"
            "            appendStepLog(out, '\\n' + msg.message + '\\n');",
        ),
        (
            "            const out = fileSteps[msg.file].querySelector('.step-output');\n"
            "            out.textContent += '\\nError: ' + msg.message + '\\n';",
            "            const out = fileSteps[msg.file].querySelector('.step-output');\n"
            "            appendStepLog(out, '\\nError: ' + msg.message + '\\n');",
        ),
        (
            "            if (out.textContent.startsWith('Connecting to LLM')) out.textContent = '';\n"  # noqa: E501
            "            out.textContent += 'Error: ' + msg.message;",
            "            if (out.textContent.startsWith('Connecting to LLM')) out.textContent = '';\n"  # noqa: E501
            "            appendStepLog(out, 'Error: ' + msg.message);",
        ),
        (
            "      if (out.textContent.startsWith('Connecting to LLM')) out.textContent = '';\n"  # noqa: E501
            "      out.textContent += '\\nError: connection to processing stream was interrupted. Please retry.';",  # noqa: E501
            "      if (out.textContent.startsWith('Connecting to LLM')) out.textContent = '';\n"  # noqa: E501
            "      appendStepLog(out, '\\nError: connection to processing stream was interrupted. Please retry.');",  # noqa: E501
        ),
        (
            "            var out = target.querySelector('.step-output');\n"
            "            out.textContent += '\\n' + msg.message + '\\n';\n"
            "            out.scrollTop = out.scrollHeight;",
            "            var out = target.querySelector('.step-output');\n"
            "            appendStepLog(out, '\\n' + msg.message + '\\n');",
        ),
        (
            "            outE.textContent += '\\nError: ' + msg.message + '\\n';",
            "            appendStepLog(outE, '\\nError: ' + msg.message + '\\n');",
        ),
        (
            "            outR.textContent += '\\nError: ' + msg.message + '\\n';",
            "            appendStepLog(outR, '\\nError: ' + msg.message + '\\n');",
        ),
        (
            "      outErr.textContent += '\\nError: connection interrupted. Please retry.';",  # noqa: E501
            "      appendStepLog(outErr, '\\nError: connection interrupted. Please retry.');",  # noqa: E501
        ),
    ]
    for i, (old, new) in enumerate(repls):
        if old not in t:
            log.error("app.js in-memory patch failed at step %d", i)
            raise RuntimeError("static/app.js no longer matches expected fragments")
        t = t.replace(old, new, 1)
    _APP_JS_PATCHED = t
    return t


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

PIPELINE_EVENTS_FILE = "pipeline_events.jsonl"
PAUSE_SUMMARY_FILE = "pause_summary.txt"


def _read_status(session_dir: Path) -> dict:
    status_path = session_dir / "status.json"
    if not status_path.exists():
        return {}
    return json.loads(status_path.read_text())


def _write_status(session_dir: Path, status: dict) -> None:
    (session_dir / "status.json").write_text(json.dumps(status))


def _pipeline_state(status: dict) -> dict:
    state = status.setdefault("pipeline_state", {})
    state.setdefault("completed_stages", [])
    state.setdefault("completed_files", [])
    state.setdefault("events", 0)
    return state


def _append_unique(items: list, value) -> None:
    if value not in items:
        items.append(value)


def _parse_sse_event(evt: str) -> dict | None:
    if not evt.startswith("data: "):
        return None
    try:
        return json.loads(evt.split("data: ", 1)[1].split("\n", 1)[0])
    except Exception:
        return None


def _record_pipeline_event(session_dir: Path, payload: dict) -> None:
    events_path = session_dir / PIPELINE_EVENTS_FILE
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": payload,
        }) + "\n")


def _checkpoint_pipeline_event(session_dir: Path, payload: dict) -> None:
    status = _read_status(session_dir)
    state = _pipeline_state(status)
    state["events"] = int(state.get("events", 0)) + 1
    if payload.get("stage"):
        state["last_stage"] = payload.get("stage")
    if payload.get("type") == "stage_complete" and payload.get("stage"):
        _append_unique(state["completed_stages"], payload["stage"])
    if payload.get("type") == "file_complete" and payload.get("file"):
        _append_unique(state["completed_files"], payload["file"])
    if payload.get("type") == "sandbox_build_result":
        state["sandbox_build_success"] = payload.get("success", False)
        if payload.get("success"):
            _append_unique(state["completed_stages"], "sandbox_build")
    if payload.get("type") == "complete":
        state["completed"] = True
        status["state"] = "completed"
    _write_status(session_dir, status)


def _is_pause_requested(session_dir: Path) -> bool:
    return bool(_read_status(session_dir).get("pause_requested"))


def _write_pause_summary(session_dir: Path, reason: str = "User paused processing") -> None:
    status = _read_status(session_dir)
    state = _pipeline_state(status)
    gen_dir = session_dir / "generated_code"
    generated = sorted(p.name for p in gen_dir.iterdir() if p.is_file()) if gen_dir.exists() else []
    report_lines = [
        "=" * 65,
        "PAUSE SUMMARY",
        "=" * 65,
        f"\nPaused at: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"Reason: {reason}",
        f"Last stage: {state.get('last_stage', 'N/A')}",
        "",
        "Completed stages:",
        *(f"  - {s}" for s in state.get("completed_stages", [])),
        "",
        "Completed generated files:",
        *(f"  - {f}" for f in state.get("completed_files", [])),
        "",
        "Generated artifacts currently available:",
        *(f"  - {f}" for f in generated),
        "",
        "This session can be resumed from the UI. Completed stages/files are",
        "reused; the interrupted stage is rerun with all previous artifacts",
        "and event history still available in the session directory.",
        "",
    ]
    (session_dir / PAUSE_SUMMARY_FILE).write_text("\n".join(report_lines) + "\n")


def _pausable_stream(session_dir: Path, events):
    """Yield SSE events while recording checkpoints and honoring pause requests."""
    for evt in events:
        payload = _parse_sse_event(evt)
        if payload:
            _record_pipeline_event(session_dir, payload)
            _checkpoint_pipeline_event(session_dir, payload)
        yield evt
        if _is_pause_requested(session_dir):
            status = _read_status(session_dir)
            status["state"] = "paused"
            status["paused_at"] = datetime.now(timezone.utc).isoformat()
            _write_status(session_dir, status)
            _write_pause_summary(session_dir)
            paused_payload = {
                "type": "paused",
                "message": (
                    "Processing paused. Current reports and generated artifacts "
                    "are available for download; click Resume to continue."
                ),
                # Previewable subset only — .o object files ship in the zip
                # but are not opened in the UI preview pane.
                "files": sorted(
                    p.name for p in (session_dir / "generated_code").iterdir()
                    if p.is_file() and p.suffix.lower() != ".o"
                ) if (session_dir / "generated_code").exists() else [],
            }
            _record_pipeline_event(session_dir, paused_payload)
            yield _sse(paused_payload)
            return

@app.get("/static/app.js")
async def static_app_js():
    return Response(_patched_app_js(), media_type="application/javascript")


@app.get("/")
async def index():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8", errors="replace")
    if "append_log_helper.js" not in html:
        html = html.replace(
            '  <script src="/static/app.js"></script>',
            '  <script src="/static/append_log_helper.js"></script>\n'
            '  <script src="/static/app.js"></script>',
        )
    return HTMLResponse(html)


@app.post("/api/session/create")
async def create_session():
    session_id = str(uuid.uuid4())
    session_dir = SESSIONS_DIR / session_id
    (session_dir / "original_code").mkdir(parents=True, exist_ok=True)
    (session_dir / "generated_code").mkdir(parents=True, exist_ok=True)
    status = {
        "state": "created",
        "files": [],
        "source_icd": False,
        "target_icd": False,
        "repo_zip": False,
        "pause_requested": False,
        "pipeline_state": {
            "completed_stages": [],
            "completed_files": [],
            "events": 0,
        },
    }
    (session_dir / "status.json").write_text(json.dumps(status))
    return {"session_id": session_id}


@app.post("/api/upload/code/{session_id}")
async def upload_code(session_id: str, files: List[UploadFile] = File(...)):
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    code_dir = session_dir / "original_code"
    uploaded: list[str] = []
    encodings: dict[str, str] = {}
    for f in files:
        if not f.filename or not (
            f.filename.endswith(".c") or f.filename.endswith(".h")
        ):
            continue
        dest = code_dir / f.filename
        content = await f.read()
        # Normalise to UTF-8 at the boundary so every downstream read_text()
        # in the pipeline is safe, rather than patching ~27 call sites.
        text, enc = _decode_bytes(content)
        if enc != "utf-8-sig":
            log.info("Upload %s: decoded as %s, transcoding to UTF-8",
                     f.filename, enc)
        dest.write_text(text, encoding="utf-8")
        encodings[f.filename] = enc
        uploaded.append(f.filename)

    status = json.loads((session_dir / "status.json").read_text())
    status["files"] = sorted(p.name for p in code_dir.iterdir() if p.is_file())
    (session_dir / "status.json").write_text(json.dumps(status))

    return {
        "uploaded": uploaded,
        "total_files": len(status["files"]),
        "encodings": encodings,
    }


@app.post("/api/upload/source-icd/{session_id}")
async def upload_source_icd(session_id: str, file: UploadFile = File(...)):
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Source ICD must be a PDF file")

    pdf_path = session_dir / "source_icd.pdf"
    pdf_path.write_bytes(await file.read())

    try:
        text = extract_pdf_text(pdf_path)
        (session_dir / "source_icd.txt").write_text(text)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to extract PDF text: {e}"
        )

    status = json.loads((session_dir / "status.json").read_text())
    status["source_icd"] = True
    status["source_icd_name"] = file.filename
    (session_dir / "status.json").write_text(json.dumps(status))

    return {"filename": file.filename, "text_length": len(text)}


@app.post("/api/upload/target-icd/{session_id}")
async def upload_target_icd(session_id: str, file: UploadFile = File(...)):
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Target ICD must be a PDF file")

    pdf_path = session_dir / "target_icd.pdf"
    pdf_path.write_bytes(await file.read())

    try:
        text = extract_pdf_text(pdf_path)
        (session_dir / "target_icd.txt").write_text(text)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to extract PDF text: {e}"
        )

    status = json.loads((session_dir / "status.json").read_text())
    status["target_icd"] = True
    status["target_icd_name"] = file.filename
    (session_dir / "status.json").write_text(json.dumps(status))

    return {"filename": file.filename, "text_length": len(text)}


@app.post("/api/upload/repo-zip/{session_id}")
async def upload_repo_zip(session_id: str, file: UploadFile = File(...)):
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Repository archive must be a ZIP file")

    zip_path = session_dir / "repo.zip"
    zip_path.write_bytes(await file.read())

    repo_dir = session_dir / "repo_contents"
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    try:
        stats = _safe_extract_zip(zip_path, repo_dir)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to extract ZIP archive: {e}"
        )

    status = json.loads((session_dir / "status.json").read_text())
    status["repo_zip"] = True
    status["repo_zip_name"] = file.filename
    status["repo_zip_files"] = stats["file_count"]
    (session_dir / "status.json").write_text(json.dumps(status))

    return {
        "filename": file.filename,
        "file_count": stats["file_count"],
    }


@app.get("/api/status/{session_id}")
async def get_status(session_id: str):
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    return json.loads((session_dir / "status.json").read_text())


@app.post("/api/pause/{session_id}")
async def pause_session(session_id: str):
    """Request cooperative pause of the active SSE processing stream."""
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    status = _read_status(session_dir)
    status["pause_requested"] = True
    status["state"] = "pause_requested"
    status["pause_requested_at"] = datetime.now(timezone.utc).isoformat()
    _write_status(session_dir, status)
    _write_pause_summary(session_dir, "Pause requested by user")
    return {
        "ok": True,
        "state": status["state"],
        "message": "Pause requested; processing will stop at the next safe checkpoint.",
    }


@app.post("/api/resume/{session_id}")
async def resume_session(session_id: str):
    """Clear pause state; the frontend then reconnects to /api/process."""
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    status = _read_status(session_dir)
    status["pause_requested"] = False
    status["state"] = "resuming"
    status["resumed_at"] = datetime.now(timezone.utc).isoformat()
    _write_status(session_dir, status)
    return {
        "ok": True,
        "state": status["state"],
        "message": "Resume accepted; reconnect to the processing stream.",
    }


@app.get("/api/process/{session_id}")
async def process(
    session_id: str,
    sandbox_retries: str | None = Query(
        None,
        description=(
            "Sandbox build reiteration budget: a positive integer, or "
            "'indefinite' to keep iterating until the build succeeds."
        ),
    ),
):
    """Analyze ICDs and transform every code file. Returns an SSE stream."""
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    status = json.loads((session_dir / "status.json").read_text())
    if not status.get("files"):
        raise HTTPException(status_code=400, detail="No code files uploaded")
    if not status.get("source_icd"):
        raise HTTPException(status_code=400, detail="Source ICD not uploaded")
    if not status.get("target_icd"):
        raise HTTPException(status_code=400, detail="Target ICD not uploaded")

    try:
        sandbox_max_retries = _resolve_sandbox_max_retries(
            sandbox_retries, status,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Persist so resume / regenerate can reuse the user's choice.
    if sandbox_max_retries is _SANDBOX_RETRIES_UNSET:
        status["sandbox_retries_mode"] = "env"
        status["sandbox_max_retries"] = None
    elif sandbox_max_retries is None:
        status["sandbox_retries_mode"] = "indefinite"
        status["sandbox_max_retries"] = None
    else:
        status["sandbox_retries_mode"] = "finite"
        status["sandbox_max_retries"] = int(sandbox_max_retries)
    _write_status(session_dir, status)

    code_dir = session_dir / "original_code"
    gen_dir = session_dir / "generated_code"
    source_icd = _read_text_safe(session_dir / "source_icd.txt")
    target_icd = _read_text_safe(session_dir / "target_icd.txt")
    code_files = sorted(p for p in code_dir.iterdir() if p.is_file())

    repo_dir = session_dir / "repo_contents"
    has_repo = repo_dir.exists() and any(repo_dir.rglob("*"))
    # The .c/.h scripts uploaded for refactoring are ALWAYS the
    # authoritative pre-change source for the ICD-delta analysis,
    # regardless of whether a broader repository ZIP was also provided.
    code_source_scripts = _build_source_scripts_context(code_dir)
    # Two flavours of repo_knowledge are produced:
    #
    #   * ``repo_knowledge_full`` — every extracted fact, no caps and no
    #     ``max_chars`` clip.  This is what gets persisted to
    #     ``repo_knowledge.txt`` and shipped in the downloadable QA ZIP.
    #
    #   * ``repo_knowledge``      — the distilled view (per-category caps
    #     plus the ``max_chars`` budget) used by every LLM prompt so the
    #     context window stays under control.
    repo_knowledge_full = ""
    repo_knowledge = ""
    if has_repo:
        repo_knowledge_full = _build_repo_knowledge(repo_dir, full=True)
        repo_knowledge = _build_repo_knowledge(repo_dir, full=False)
        (session_dir / "repo_knowledge.txt").write_text(repo_knowledge_full)
    log.info(
        "Source scripts for ICD analysis: %d chars, repo knowledge: "
        "distilled=%d chars, full=%d chars (saved to repo_knowledge.txt)",
        len(code_source_scripts), len(repo_knowledge), len(repo_knowledge_full),
    )

    uploaded_names = {p.name for p in code_files}
    log.info(
        "Process %s: %d code files, source_icd=%d chars, target_icd=%d chars, has_repo=%s",
        session_id[:8], len(code_files), len(source_icd), len(target_icd), has_repo,
    )

    is_resume = status.get("state") == "resuming"
    if not is_resume:
        status["pause_requested"] = False
        status["state"] = "processing"
        status["pipeline_state"] = {
            "completed_stages": [],
            "completed_files": [],
            "events": 0,
        }
        events_path = session_dir / PIPELINE_EVENTS_FILE
        if events_path.exists():
            events_path.unlink()
        _write_status(session_dir, status)

    def event_stream():
        yield _sse({
            "type": "info",
            "stage": "analysis",
            "message": "Checking llama-server availability...",
        })
        try:
            _wait_for_llm_ready(timeout_s=300)
        except Exception as e:
            yield _sse({
                "type": "error",
                "message": (
                    f"{e} This usually means llama-server failed during startup. "
                    "Check container terminal/tmux logs for CUDA/GPU errors."
                ),
            })
            return

        # ---- Step 1: Analyse ICD delta --------------------------------
        resume_state = _pipeline_state(_read_status(session_dir))
        completed_stages = set(resume_state.get("completed_stages", []))
        analysis_files_ready = (
            (session_dir / "change_spec.txt").exists()
            and (session_dir / "target_summary.txt").exists()
            and (session_dir / "icd_analysis.txt").exists()
        )
        if is_resume and "analysis" in completed_stages and analysis_files_ready:
            change_spec = _read_text_safe(session_dir / "change_spec.txt")
            target_summary = _read_text_safe(session_dir / "target_summary.txt")
            yield _sse({
                "type": "stage",
                "stage": "analysis",
                "message": "Resuming: reusing completed ICD analysis from checkpoint...",
            })
            yield _sse({
                "type": "info",
                "stage": "analysis",
                "message": "Loaded prior change_spec.txt, target_summary.txt, and icd_analysis.txt.",
            })
            yield _sse({"type": "stage_complete", "stage": "analysis"})

        else:
            # ---- Step 1: Analyse ICD delta --------------------------------
            yield _sse({
                "type": "stage",
                "stage": "analysis",
                "message": "Analyzing ICD differences\u2026",
            })

            repo_analysis_hint = ""
            if code_source_scripts:
                repo_analysis_hint = (
                    "\n\n"
                    "You are also given the ACTUAL existing C source code of the "
                    "project — the \"old scripts\" — that will be refactored to "
                    "match the Target ICD. Treat these scripts as the ground-truth "
                    "pre-change implementation.\n\n"
                    "When you build the change specification you MUST:\n"
                    "  * Read the old scripts carefully BEFORE listing impacts.\n"
                    "  * Quote the EXACT type names, struct fields, enum members, "
                    "function signatures, macro names, and #include paths as they "
                    "appear in the old scripts.\n"
                    "  * For every 'Impact on C code' item, name the specific file "
                    "(e.g. `include/comm.h`, `src/sensor.c`) and the specific "
                    "symbol or block that must change, mapped from its OLD form in "
                    "the scripts to its NEW form required by the Target ICD.\n"
                    "  * Do NOT invent symbols, types, or files that do not appear "
                    "in the old scripts; if the ICD introduces something brand-new, "
                    "say so explicitly and indicate where it should be added.\n"
                    "  * Flag any ICD requirements that have no clear hook in the "
                    "current code as 'NEW' so a downstream refactor agent knows to "
                    "create them rather than edit existing code.\n\n"
                    f"{code_source_scripts}\n"
                )

            compare_system = (
                "You are an expert systems engineer and C programmer specializing in ICD-driven changes.\n"
                "Compare the Source ICD vs Target ICD and produce a COMPLETE and EXHAUSTIVE\n"
                "code-impact change specification. Cover ALL of the following:\n"
                "- structs/fields/types/sizes/alignment changes\n"
                "- enums/constants/message IDs and payload formats\n"
                "- function signatures/APIs/callbacks changes\n"
                "- behavior/protocol/state/timing constraints\n"
                "- header/source synchronization requirements\n"
                "- any additions, removals, or modifications between versions\n\n"
                "Be thorough and detailed. Do not abbreviate or summarize. "
                "List every single change with specific old and new values. "
                "Ground the 'Impact on C code' section in the actual old scripts "
                "provided below — cite real files and symbols, never invented ones."
                f"{PERIPHERAL_VARIATION_ANALYSIS_GUIDANCE}"
                f"{repo_analysis_hint}"
            )

            combined_icd_len = len(source_icd) + len(target_icd)
            use_direct = combined_icd_len <= DIRECT_COMPARE_MAX_CHARS

            try:
                if use_direct:
                    # ---- DIRECT path: one streaming comparison with full ICD texts ----
                    log.info("Using direct comparison (combined %d chars)", combined_icd_len)
                    yield _sse({
                        "type": "info",
                        "stage": "analysis",
                        "message": (
                            f"ICDs fit in context ({combined_icd_len:,} chars). "
                            "Performing direct full-text comparison…"
                        ),
                    })

                    grounding_note = (
                        "\n\nThe system prompt contains the project's actual existing "
                        "C source scripts (the 'old scripts'). Use them as the "
                        "authoritative pre-change implementation when filling in the "
                        "'Impact on C code' field — quote real file paths and real "
                        "symbol names, and explicitly mark anything brand-new."
                        if code_source_scripts else ""
                    )
                    base_compare_prompt = (
                        "Compare these two complete ICD documents and produce a COMPLETE "
                        "and EXHAUSTIVE code-impact change specification that will be used "
                        "to refactor C source code.\n\n"
                        f"## Source ICD (Full Text)\n{source_icd}\n\n"
                        f"## Target ICD (Full Text)\n{target_icd}\n\n"
                        "List EVERY difference between the two ICDs. For each change, state:\n"
                        "1. What it was in the Source ICD (old)\n"
                        "2. What it is in the Target ICD (new)\n"
                        "3. Impact on C code (structs, enums, functions, constants, etc.) — "
                        "cite the specific file and symbol from the old scripts, mapping the "
                        "OLD form to the NEW form."
                        f"{PERIPHERAL_VARIATION_ANALYSIS_GUIDANCE}"
                        f"{grounding_note}"
                    )
                    target_summary = _truncate_text(target_icd, 12_000, "target_icd_for_transform")

                else:
                    # ---- MAP-REDUCE path: for very large ICDs ----
                    log.info("Using map-reduce (combined %d chars > %d threshold)",
                             combined_icd_len, DIRECT_COMPARE_MAX_CHARS)
                    map_system = (
                        "You are an ICD analyst. Extract exhaustive technical facts from this ICD chunk.\n"
                        "Capture structs, enums, constants, message IDs, payload layouts, field sizes/types,\n"
                        "function/interface signatures, protocol/state/timing requirements, and constraints.\n"
                        "Return concise bullet points with concrete values; no filler text."
                    )
                    merge_system = (
                        "You are consolidating multiple ICD chunk notes from the SAME ICD document.\n"
                        "Merge them into one complete, deduplicated technical summary while preserving\n"
                        "every concrete detail (names, values, sizes, types)."
                    )

                    source_chunks = _split_text_chunks(source_icd, ICD_CHUNK_CHARS)
                    target_chunks = _split_text_chunks(target_icd, ICD_CHUNK_CHARS)
                    yield _sse({
                        "type": "info",
                        "stage": "analysis",
                        "message": (
                            f"Large ICDs — using chunked analysis: "
                            f"source={len(source_chunks)} chunks, "
                            f"target={len(target_chunks)} chunks. "
                            "This will take a while…"
                        ),
                    })

                    source_notes: list[str] = []
                    for idx, chunk_text in enumerate(source_chunks, start=1):
                        yield _sse({
                            "type": "info",
                            "stage": "analysis",
                            "message": f"Extracting facts from source ICD chunk {idx}/{len(source_chunks)}…",
                        })
                        note = _call_llm_complete(
                            map_system,
                            (
                                f"Source ICD chunk {idx}/{len(source_chunks)}:\n\n"
                                f"{chunk_text}\n\n"
                                "Extract all code-relevant facts from this chunk."
                            ),
                            max_tokens=2048,
                            max_passes=2,
                        )
                        source_notes.append(note)
                        yield _sse({
                            "type": "token",
                            "stage": "analysis",
                            "token": (
                                f"\n\n[Source chunk {idx}/{len(source_chunks)} analysis]\n"
                                f"{note}\n"
                            ),
                        })

                    target_notes: list[str] = []
                    for idx, chunk_text in enumerate(target_chunks, start=1):
                        yield _sse({
                            "type": "info",
                            "stage": "analysis",
                            "message": f"Extracting facts from target ICD chunk {idx}/{len(target_chunks)}…",
                        })
                        note = _call_llm_complete(
                            map_system,
                            (
                                f"Target ICD chunk {idx}/{len(target_chunks)}:\n\n"
                                f"{chunk_text}\n\n"
                                "Extract all code-relevant facts from this chunk."
                            ),
                            max_tokens=2048,
                            max_passes=2,
                        )
                        target_notes.append(note)
                        yield _sse({
                            "type": "token",
                            "stage": "analysis",
                            "token": (
                                f"\n\n[Target chunk {idx}/{len(target_chunks)} analysis]\n"
                                f"{note}\n"
                            ),
                        })

                    yield _sse({
                        "type": "info",
                        "stage": "analysis",
                        "message": "Consolidating source ICD summary…",
                    })
                    source_summary = _call_llm_complete(
                        merge_system,
                        "Merge these Source ICD notes into one complete technical summary:\n\n"
                        + "\n\n".join(
                            f"## Source chunk note {i}\n{n}"
                            for i, n in enumerate(source_notes, 1)
                        ),
                        max_tokens=4096,
                        max_passes=3,
                    )
                    yield _sse({
                        "type": "token",
                        "stage": "analysis",
                        "token": f"\n\n[Consolidated Source ICD Summary]\n{source_summary}\n",
                    })

                    yield _sse({
                        "type": "info",
                        "stage": "analysis",
                        "message": "Consolidating target ICD summary…",
                    })
                    target_summary = _call_llm_complete(
                        merge_system,
                        "Merge these Target ICD notes into one complete technical summary:\n\n"
                        + "\n\n".join(
                            f"## Target chunk note {i}\n{n}"
                            for i, n in enumerate(target_notes, 1)
                        ),
                        max_tokens=4096,
                        max_passes=3,
                    )
                    yield _sse({
                        "type": "token",
                        "stage": "analysis",
                        "token": f"\n\n[Consolidated Target ICD Summary]\n{target_summary}\n",
                    })

                    yield _sse({
                        "type": "info",
                        "stage": "analysis",
                        "message": "Generating detailed change specification…",
                    })

                    grounding_note = (
                        "\n\nThe system prompt contains the project's actual existing "
                        "C source scripts (the 'old scripts'). Use them as the "
                        "authoritative pre-change implementation when filling in the "
                        "'Impact on C code' field — quote real file paths and real "
                        "symbol names, and explicitly mark anything brand-new."
                        if code_source_scripts else ""
                    )
                    base_compare_prompt = (
                        "Produce a COMPLETE and EXHAUSTIVE code-impact change specification "
                        "comparing the Source ICD to the Target ICD.  This specification will "
                        "be used to refactor C code, so it must cover every single difference.\n\n"
                        f"## Source ICD full summary\n{source_summary}\n\n"
                        f"## Target ICD full summary\n{target_summary}\n\n"
                        "List EVERY difference between the two ICDs. For each change, state:\n"
                        "1. What it was in the Source ICD (old)\n"
                        "2. What it is in the Target ICD (new)\n"
                        "3. Impact on C code (structs, enums, functions, constants, etc.) — "
                        "cite the specific file and symbol from the old scripts, mapping the "
                        "OLD form to the NEW form."
                        f"{PERIPHERAL_VARIATION_ANALYSIS_GUIDANCE}"
                        f"{grounding_note}"
                    )

                # ---- Streaming comparison (used by both paths) ----
                analysis_parts: list[str] = []
                complete_analysis = ""
                max_analysis_passes = 6
                compare_max_tokens = 4096
                last_meta: dict = {}

                for attempt in range(1, max_analysis_passes + 1):
                    meta: dict = {}
                    if attempt == 1:
                        prompt = base_compare_prompt
                    else:
                        yield _sse({
                            "type": "info",
                            "stage": "analysis",
                            "message": (
                                f"Analysis output was truncated — continuing "
                                f"(pass {attempt}/{max_analysis_passes})…"
                            ),
                        })
                        tail_len = 3000
                        tail = (complete_analysis[-tail_len:]
                                if len(complete_analysis) > tail_len
                                else complete_analysis)
                        prompt = (
                            "Your previous analysis was cut off due to length limits. "
                            "Here is the end of what you wrote:\n\n"
                            f"---\n{tail}\n---\n\n"
                            "Continue EXACTLY from where you left off. Do not repeat "
                            "content already written. Cover all remaining differences "
                            "between the Source and Target ICDs."
                        )

                    pass_parts: list[str] = []
                    for batch in _batched_stream_text(
                        _call_llm_stream(
                            compare_system, prompt,
                            max_tokens=compare_max_tokens, meta=meta,
                        )
                    ):
                        pass_parts.append(batch)
                        yield _sse({
                            "type": "token",
                            "stage": "analysis",
                            "token": batch,
                        })

                    pass_text = "".join(pass_parts)
                    analysis_parts.append(pass_text)
                    complete_analysis = "".join(analysis_parts)
                    last_meta = meta

                    if meta.get("finish_reason") != "length":
                        log.info(
                            "Analysis pass %d done (finish_reason=%s)",
                            attempt, meta.get("finish_reason"),
                        )
                        break
                    log.info("Analysis pass %d hit token limit, continuing…", attempt)

                if last_meta.get("finish_reason") == "length":
                    log.warning(
                        "ICD analysis may be incomplete after %d passes",
                        max_analysis_passes,
                    )
                    yield _sse({
                        "type": "info",
                        "stage": "analysis",
                        "message": (
                            "Note: analysis reached maximum continuation passes. "
                            "Some minor details may be incomplete."
                        ),
                    })

                raw_change_spec = complete_analysis.strip()
                # Default to raw; only overwrite below when distillation
                # genuinely produces a shorter/cleaner version that survives
                # the fact-preservation check.
                change_spec = raw_change_spec
                if raw_change_spec:
                    yield _sse({
                        "type": "info",
                        "stage": "analysis",
                        "message": (
                            "Distilling change specification for conciseness "
                            "while preserving every technical fact..."
                        ),
                    })
                    distill_system = (
                        "You are an expert embedded C refactoring analyst. You "
                        "produce a SHORTER, more compact rewrite of ICD delta "
                        "specifications. Your output is a distillation — it must "
                        "be different from the input, removing redundancy and "
                        "verbose prose, while preserving EVERY technical fact "
                        "verbatim (hex constants, numeric literals, type names, "
                        "field names, identifiers, units, scale factors). An "
                        "engineer must be able to refactor C code from the "
                        "distilled spec alone."
                    )
                    base_distill_prompt = (
                        "Produce a DISTILLED version of the following ICD change "
                        "specification.\n\n"
                        "MANDATORY RULES:\n"
                        "1. Your output MUST be shorter than the input (aim for "
                        "40-60% of the original length). It MUST be different "
                        "from the input — never echo it back verbatim.\n"
                        "2. Preserve EVERY technical fact verbatim: every hex "
                        "constant (e.g. 0xEB90), numeric literal, type "
                        "(uint8_t/uint16_t/uint32_t), field name, struct/enum/"
                        "macro/function identifier, message ID, scale factor, "
                        "sample rate, unit and OLD -> NEW mapping. Do NOT "
                        "rename or paraphrase identifiers.\n"
                        "3. Preserve file/symbol citations (which .c/.h and "
                        "which symbol each change touches) and any item marked "
                        "NEW.\n"
                        "4. Preserve behaviour / protocol / state / timing "
                        "constraints.\n"
                        "5. If the spec describes MULTIPLE variations (variants "
                        "/ configurations / modes) of the same peripheral, "
                        "PRESERVE the per-variation structure: a clearly-"
                        "labelled subsection per variation with that variation's "
                        "FULL field set, plus a separate SHARED section. Do NOT "
                        "merge variations and do NOT drop any.\n"
                        "6. REMOVE: duplicated statements, restated context, "
                        "verbose narration and filler prose — never substantive "
                        "detail. Use compact bullet points / tables.\n"
                        "7. Do NOT introduce new changes. Output ONLY the "
                        "distilled specification text (no preamble, no JSON, no "
                        "code fences around the whole thing).\n\n"
                        "## Original change specification\n"
                        f"{raw_change_spec}"
                    )
                    raw_facts_count = len(_extract_fact_tokens(raw_change_spec))

                    def _call_distill(prompt_text: str) -> str:
                        try:
                            return _call_llm_complete(
                                distill_system, prompt_text,
                                max_tokens=4096, max_passes=6,
                            ).strip()
                        except Exception as e:        # noqa: BLE001
                            log.warning(
                                "change_spec distillation call failed: %s", e,
                            )
                            return ""

                    distilled = _call_distill(base_distill_prompt)
                    ok, missing, kept_ratio = _distillation_fact_check(
                        raw_change_spec, distilled,
                    )
                    needs_retry_reason = ""
                    if not distilled:
                        needs_retry_reason = "previous attempt produced no output"
                    elif distilled == raw_change_spec:
                        needs_retry_reason = (
                            "previous attempt returned the input verbatim; you "
                            "MUST condense it"
                        )
                    elif not ok:
                        sample = ", ".join(missing[:60])
                        needs_retry_reason = (
                            "previous attempt dropped required technical "
                            f"tokens — re-include them verbatim: {sample}"
                        )
                    if needs_retry_reason:
                        yield _sse({
                            "type": "info",
                            "stage": "analysis",
                            "message": (
                                "Distillation needs a second pass — facts "
                                f"kept: {kept_ratio:.0%}, retrying with focused "
                                "guidance."
                            ),
                        })
                        retry_prompt = (
                            f"{base_distill_prompt}\n\n"
                            f"## Retry guidance\n{needs_retry_reason}.\n"
                            "Produce a NEW, shorter distillation that satisfies "
                            "every rule above."
                        )
                        distilled2 = _call_distill(retry_prompt)
                        ok2, missing2, kept2 = _distillation_fact_check(
                            raw_change_spec, distilled2,
                        )
                        # Accept the retry whenever it produces a usable,
                        # genuinely distilled output. We MUST swap when the
                        # first attempt was empty or was the raw verbatim
                        # (raw-verbatim trivially has 1.0 fact coverage but
                        # is NOT a distillation). Otherwise accept the retry
                        # when its fact coverage is at least as good.
                        if distilled2 and distilled2 != raw_change_spec:
                            first_unusable = (
                                not distilled or distilled == raw_change_spec
                            )
                            if first_unusable or kept2 >= kept_ratio:
                                distilled, ok, missing, kept_ratio = (
                                    distilled2, ok2, missing2, kept2,
                                )

                    if distilled and distilled != raw_change_spec:
                        change_spec = distilled
                        if ok:
                            log.info(
                                "Distilled change_spec: %d -> %d chars "
                                "(facts kept %.0f%%, %d/%d).",
                                len(raw_change_spec), len(change_spec),
                                kept_ratio * 100,
                                raw_facts_count - len(missing),
                                raw_facts_count,
                            )
                            yield _sse({
                                "type": "info",
                                "stage": "analysis",
                                "message": (
                                    f"Distilled change_spec: "
                                    f"{len(raw_change_spec):,} -> "
                                    f"{len(change_spec):,} chars "
                                    f"({kept_ratio:.0%} of technical facts "
                                    "preserved). Raw comprehensive version "
                                    "saved to change_spec_raw.txt."
                                ),
                            })
                        else:
                            log.warning(
                                "Distilled change_spec accepted with reduced "
                                "fact coverage: %d -> %d chars "
                                "(facts kept %.0f%%, missing %d tokens). "
                                "Comprehensive version available in "
                                "change_spec_raw.txt.",
                                len(raw_change_spec), len(change_spec),
                                kept_ratio * 100, len(missing),
                            )
                            yield _sse({
                                "type": "info",
                                "stage": "analysis",
                                "message": (
                                    f"Distilled spec accepted with "
                                    f"{kept_ratio:.0%} fact coverage; the "
                                    "comprehensive raw version is in "
                                    "change_spec_raw.txt."
                                ),
                            })
                    else:
                        log.warning(
                            "Distillation could not produce a usable shorter "
                            "version (empty=%s, equals_raw=%s); change_spec "
                            "will mirror change_spec_raw for this run.",
                            not distilled, distilled == raw_change_spec,
                        )
                        yield _sse({
                            "type": "info",
                            "stage": "analysis",
                            "message": (
                                "Distillation did not produce a usable shorter "
                                "version this time — change_spec will mirror "
                                "the comprehensive raw spec."
                            ),
                        })
                log.info(
                    "ICD analysis complete (%d chars, %d passes, mode=%s)",
                    len(change_spec), len(analysis_parts),
                    "direct" if use_direct else "map-reduce",
                )
            except Exception as e:
                log.exception("LLM analysis failed: %s", e)
                yield _sse({"type": "error", "message": str(e)})
                return

            (session_dir / "change_spec.txt").write_text(change_spec)
            # Distilled spec is what downstream LLM prompts consume;
            # the pre-distillation comprehensive analysis is kept separately
            # for QA / audit (same pattern as full vs distilled repo knowledge).
            (session_dir / "change_spec_raw.txt").write_text(raw_change_spec)
            (session_dir / "target_summary.txt").write_text(target_summary)
            log.info(
                "Saved change_spec (distilled): %d chars, change_spec_raw "
                "(comprehensive): %d chars",
                len(change_spec), len(raw_change_spec),
            )

            # ``icd_analysis.txt`` is a downloadable QA artefact, so it
            # embeds the FULL repo_knowledge (no caps / no clip), not the
            # distilled view fed into LLM prompts.
            analysis_report_parts = [change_spec]
            if repo_knowledge_full:
                analysis_report_parts.append("\n\n" + repo_knowledge_full)
            elif repo_knowledge:
                analysis_report_parts.append("\n\n" + repo_knowledge)
            full_analysis = "\n".join(analysis_report_parts)
            (session_dir / "icd_analysis.txt").write_text(full_analysis)
            (gen_dir / "icd_analysis.txt").write_text(full_analysis)
            yield _sse({"type": "stage_complete", "stage": "analysis"})

        # Gather cross-file context (the uploaded source files)
        all_code_ctx = ""
        for cf in code_files:
            all_code_ctx += f"\n### File: {cf.name}\n```c\n{_read_text_safe(cf)}\n```\n"
        all_code_ctx = _truncate_text(all_code_ctx, MAX_CODE_CONTEXT_CHARS, "all_code_ctx")

        # ---- Step 2: Transform each file ------------------------------
        for i, code_file in enumerate(code_files):
            fname = code_file.name
            if (
                is_resume
                and fname in set(_pipeline_state(_read_status(session_dir)).get("completed_files", []))
                and (gen_dir / fname).exists()
            ):
                yield _sse({
                    "type": "stage",
                    "stage": "transform",
                    "file": fname,
                    "index": i,
                    "total": len(code_files),
                    "message": f"Resuming: reusing completed transform for {fname}...",
                })
                yield _sse({
                    "type": "file_complete",
                    "file": fname,
                    "size": (gen_dir / fname).stat().st_size,
                })
                continue
            yield _sse({
                "type": "stage",
                "stage": "transform",
                "file": fname,
                "index": i,
                "total": len(code_files),
                "message": f"Transforming {fname}\u2026",
            })

            original = _read_text_safe(code_file)

            transform_system = (
                "You are an expert C programmer specializing in embedded systems "
                "and interface implementations governed by Interface Control "
                "Documents.\n\n"
                "TARGET TOOLCHAIN:\n"
                "- Xilinx SDK 2018.x with GCC 7.3.1 (arm-none-eabi / mb-gcc)\n"
                "- C standard: C99 with GCC extensions (-std=gnu99) — "
                "GCC-specific constructs commonly used in embedded code "
                "are ACCEPTED, including unnamed structs/unions, "
                "`__attribute__`, statement expressions, and M_PI / M_E "
                "from <math.h>\n"
                "- C library: newlib (NOT glibc) — do NOT use glibc-specific "
                "functions (e.g. asprintf, getline, strdup, strndup, vasprintf)\n"
                "- Use <stdint.h> fixed-width types (uint8_t, uint16_t, uint32_t)\n"
                "- No POSIX headers (unistd.h, sys/*.h) — embedded freestanding\n"
                "- Avoid GCC extensions added after GCC 7 (no __attribute__((access)), "
                "no __builtin_expect_with_probability, etc.)\n\n"
                "RULES:\n"
                "1. Output ONLY the complete, transformed C source code\n"
                "2. Preserve the overall code architecture, style, and conventions\n"
                "3. Apply ALL changes required by the target ICD per the change "
                "specification\n"
                "4. Update data structures, function signatures, constants, enums, "
                "macros\n"
                "5. Update comments/doc-strings to reflect the new ICD version\n"
                "6. Ensure type correctness and compilability with GCC 7.3.1\n"
                "7. Keep header/source consistency across the project\n"
                "8. Do NOT add prose explanations — only output C code\n"
                "9. Wrap the entire output in ```c ... ``` fences\n"
                "10. Match naming conventions, coding style, variable naming, "
                "and communication patterns from the repository codebase\n"
                "11. Ensure #include directives reference correct repository headers\n"
                "12. Maintain compatibility with all dependent modules in the repository"
                f"{PERIPHERAL_VARIATION_CODEGEN_GUIDANCE}"
            )

            file_repo_ctx = ""
            if has_repo:
                file_repo_ctx = _build_file_repo_context(
                    repo_dir, original, uploaded_names,
                    max_chars=MAX_REPO_CONTEXT_CHARS,
                )
                if not file_repo_ctx:
                    file_repo_ctx = _build_repo_context(
                        repo_dir, exclude_names=uploaded_names,
                    )
                log.info("Repo context for %s: %d chars (distilled=%s)",
                         fname, len(file_repo_ctx), bool(file_repo_ctx))

            sec_change = (
                f"## Change Specification (Source ICD -> Target ICD)\n\n{change_spec}"
            )
            sec_target = (
                f"## Target ICD Consolidated Summary\n\n{target_summary}"
            )
            sec_repo = ""
            # if file_repo_ctx:
            #     sec_repo = (
            #         f"## Repository Dependency Context\n"
            #         f"These are the actual headers and modules this file depends on. "
            #         f"Use the exact type names, function signatures, macros, and "
            #         f"naming conventions from these files.\n\n{file_repo_ctx}"
            #     )
            sec_knowledge = ""
            # if repo_knowledge:
            #     sec_knowledge = (
            #         f"## Repository Codebase Knowledge\n"
            #         f"Detailed inventory of types, functions, variables, and macros "
            #         f"from the repository. Use these as ground truth for naming, "
            #         f"types, and conventions.\n\n{repo_knowledge}"
            #     )
            sec_code = ""
            # sec_code = f"## All Project Files (cross-file context)\n{all_code_ctx}"

            base_sections = [
                ("repo_dependencies", sec_repo, 1),
                ("change_spec", sec_change, 1),
                ("repo_knowledge", sec_knowledge, 2),
                ("target_summary", sec_target, 3),
                ("cross_file_ctx", sec_code, 4),
            ]

            is_header = fname.lower().endswith(".h")

            # ---- Deterministic per-variant header generation ----
            # When the Target ICD defines multiple variants of THIS header's
            # peripheral, generate one self-contained .h FILE per variant
            # (named after the variant) using a focused call per variant.
            # This is far more reliable than asking the model to emit several
            # files with markers inside one capped response.
            if is_header:
                variant_names = _detect_peripheral_variants(
                    original, fname, change_spec, target_summary,
                )
                if len(variant_names) >= 2:
                    base_stem = fname[:-2]  # drop trailing ".h"
                    yield _sse({
                        "type": "info",
                        "stage": "transform",
                        "file": fname,
                        "message": (
                            f"{fname}: Target ICD defines {len(variant_names)} "
                            f"variants ({', '.join(variant_names)}) — generating "
                            "one header file per variant…"
                        ),
                    })
                    variant_files: dict[str, str] = {}
                    for vname in variant_names:
                        slug = _variant_slug(vname)
                        if not slug:
                            continue
                        hfn = f"{base_stem}_{slug}.h"
                        yield _sse({
                            "type": "info",
                            "stage": "transform",
                            "file": hfn,
                            "message": f"Generating {hfn} for variant '{vname}'…",
                        })
                        vcode = _generate_variant_header(
                            base_sections=base_sections,
                            transform_system=transform_system,
                            base_header_name=fname,
                            original=original,
                            variant_name=vname,
                            all_variants=variant_names,
                            header_filename=hfn,
                        )
                        if vcode and _looks_complete_header(vcode):
                            variant_files[hfn] = vcode
                        else:
                            yield _sse({
                                "type": "info",
                                "stage": "transform",
                                "file": hfn,
                                "message": (
                                    f"Variant '{vname}' header looked incomplete; "
                                    "it will be skipped."
                                ),
                            })
                    if len(variant_files) >= 2:
                        for hfn, vcode in sorted(variant_files.items()):
                            (gen_dir / hfn).write_text(vcode)
                            yield _sse({
                                "type": "file_complete",
                                "file": hfn,
                                "size": len(vcode),
                            })
                        log.info(
                            "Transform %s: emitted %d per-variant header files: %s",
                            fname, len(variant_files),
                            ", ".join(sorted(variant_files)),
                        )
                        yield _sse({
                            "type": "info",
                            "stage": "transform",
                            "file": fname,
                            "message": (
                                f"{fname}: generated {len(variant_files)} separate "
                                f"per-variant header files "
                                f"({', '.join(sorted(variant_files))})."
                            ),
                        })
                        continue
                    log.warning(
                        "Transform %s: deterministic per-variant generation "
                        "produced <2 complete headers; falling back to single "
                        "header output.", fname,
                    )
                    yield _sse({
                        "type": "info",
                        "stage": "transform",
                        "file": fname,
                        "message": (
                            f"{fname}: could not generate separate variant headers "
                            "reliably — falling back to a single header."
                        ),
                    })

            if is_header:
                base_stem = fname[:-2]  # drop trailing ".h"
                variation_instr = (
                    "If the change specification describes MULTIPLE variations of "
                    "the same peripheral, output a SEPARATE, COMPLETE .h file for "
                    "EACH variation instead of one combined header. Precede each "
                    "file with a line of the form `### <filename>` (for example "
                    f"`### {base_stem}_<Variation>.h`) immediately followed by its "
                    "```c fenced block. Each header MUST be fully self-contained: "
                    "its own include guard, its own #includes, a banner comment "
                    "naming the variation, and ONLY that variation's struct / "
                    "enums / constants / macros. Do NOT emit any other `###` "
                    "headings. If only one variation applies, output a single "
                    "complete header normally with NO `### ` marker. "
                )
            else:
                variation_instr = (
                    "If the change specification lists multiple variations of the "
                    "same peripheral, emit a separate, clearly-commented struct (and "
                    "matching code) for EACH variation so the engineer can keep only "
                    "the one they need — do not merge or drop variations. "
                )
            sec_file = (
                f"## File to Transform: {fname}\n\n```c\n{original}\n```\n\n"
                "Transform this file so it fully conforms to the Target ICD. "
                "Apply every relevant change from the change specification. "
                "Ensure the generated code is FULLY COMPATIBLE with the repository "
                "codebase — use the exact type names, function signatures, and "
                "#include paths from the dependency headers above. "
                f"{variation_instr}"
                "Output the complete file — do not omit any sections. "
                "Do not summarize. Do not truncate. Include the full ending of the file."
            )

            system_tokens = _estimate_tokens(transform_system)
            prompt_sections = [
                ("file_to_transform", sec_file, 0),
                ("repo_dependencies", sec_repo, 1),
                ("change_spec", sec_change, 1),
                ("repo_knowledge", sec_knowledge, 2),
                ("target_summary", sec_target, 3),
                ("cross_file_ctx", sec_code, 4),
            ]
            transform_prompt = _assemble_prompt(
                prompt_sections,
                max_input_tokens=MAX_INPUT_TOKENS - system_tokens,
            )
            log.info("Transform prompt for %s: %d chars (%d est. tokens)",
                     fname, len(transform_prompt), _estimate_tokens(transform_prompt))
            clean = ""
            last_chunk = ""
            last_raw = ""
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                file_parts: list[str] = []
                attempt_prompt = transform_prompt
                if attempt > 1:
                    yield _sse({
                        "type": "info",
                        "stage": "transform",
                        "file": fname,
                        "message": f"Detected incomplete output; requesting continuation (pass {attempt}/{max_attempts}).",
                    })
                    attempt_prompt = (
                        f"{transform_prompt}\n\n"
                        "The previous output was incomplete/truncated. "
                        "Continue from the exact point where it stopped, output ONLY the missing remainder, "
                        "and ensure braces/comments/preprocessor blocks are closed.\n\n"
                        f"Previous partial output:\n```c\n{clean}\n```"
                    )
                meta: dict = {}
                try:
                    max_tokens=max(MAX_OUTPUT_TOKENS, int(_estimate_tokens(original) * 1.15))
                    for batch in _batched_stream_text(
                        _call_llm_stream(
                            transform_system, attempt_prompt,
                            max_tokens=max_tokens, meta=meta,
                        )
                    ):
                        file_parts.append(batch)
                        yield _sse({
                            "type": "token",
                            "stage": "transform",
                            "file": fname,
                            "token": batch,
                        })
                except Exception as e:
                    yield _sse({
                        "type": "error",
                        "message": f"Error transforming {fname}: {e}",
                        "file": fname,
                    })
                    break

                last_raw = "".join(file_parts)
                last_chunk = _extract_fenced(last_raw, "c").strip()
                if attempt == 1:
                    clean = last_chunk
                else:
                    clean = (clean.rstrip() + "\n" + last_chunk.lstrip()).strip()

                # Diagnostic: distinguish "model hit the token limit" from
                # "model stopped on its own but the gate rejected the output".
                finish = meta.get("finish_reason")
                gate_min = max(120, int(len(original.strip()) * 0.35))
                log.info(
                    "Transform %s pass %d/%d: finish_reason=%r requested=%d "
                    "raw=%d chars, fenced=%d chars, accumulated=%d chars "
                    "(original=%d, gate needs >=%d)",
                    fname, attempt, max_attempts, finish, max_tokens,
                    len(last_raw), len(last_chunk), len(clean),
                    len(original.strip()), gate_min,
                )
                if not _looks_complete_c_file(clean, original, fname):
                    log.warning(
                        "Transform %s pass %d rejected by _looks_complete_c_file: "
                        "braces {=%d }=%d | comments /*=%d */=%d | len=%d/%d | "
                        "last_line=%r",
                        fname, attempt,
                        clean.count("{"), clean.count("}"),
                        clean.count("/*"), clean.count("*/"),
                        len(clean.strip()), gate_min,
                        (clean.strip().splitlines() or [""])[-1][:60],
                    )

                if _looks_complete_c_file(clean, original, fname):
                    break

                # finish_reason "stop" means the model ended deliberately -- there
                # is no remainder to continue from, and asking anyway makes it
                # re-emit content that the concatenation above then duplicates.
                if finish is not None and finish != "length":
                    log.warning(
                        "Transform %s: not retrying as a continuation "
                        "(finish_reason=%r, not 'length').", fname, finish,
                    )
                    break

            # When the target ICD describes multiple variations of a
            # peripheral, a .h transform is emitted as one self-contained
            # header file per variation (### <name>.h + fenced block).
            # Write each such file separately; otherwise fall through to the
            # normal single-file path below.
            variation_headers = _split_variation_header_files(last_raw, fname)
            if variation_headers:
                incomplete = [
                    n for n, c in variation_headers.items()
                    if not _looks_complete_header(c)
                ]
                if not incomplete:
                    for vname, vcode in sorted(variation_headers.items()):
                        (gen_dir / vname).write_text(vcode)
                        yield _sse({
                            "type": "file_complete",
                            "file": vname,
                            "size": len(vcode),
                        })
                    log.info(
                        "Transform %s: emitted %d per-variation header files: %s",
                        fname, len(variation_headers),
                        ", ".join(sorted(variation_headers)),
                    )
                    yield _sse({
                        "type": "info",
                        "stage": "transform",
                        "file": fname,
                        "message": (
                            f"{fname}: detected multiple peripheral variations — "
                            f"generated {len(variation_headers)} separate header "
                            f"files ({', '.join(sorted(variation_headers))})."
                        ),
                    })
                    continue
                log.warning(
                    "Transform %s: per-variation header split incomplete (%s); "
                    "falling back to single-file output.",
                    fname, ", ".join(incomplete),
                )

            if not _looks_complete_c_file(clean, original, fname):
                yield _sse({
                    "type": "error",
                    "message": (
                        f"{fname} still appears incomplete after retries. "
                        "Try with smaller ICD PDFs or a GPU-enabled llama runtime."
                    ),
                    "file": fname,
                })
                continue

            (gen_dir / fname).write_text(clean)
            yield _sse({
                "type": "file_complete",
                "file": fname,
                "size": len(clean),
            })

        # ---- Step 2b: Header documentation pass ---------------------------
        # Runs only on .h files. Adds ICD-grounded inline comments to every
        # struct field, enum member, and macro. The code structure is frozen —
        # this pass is purely additive.
        for i, code_file in enumerate(code_files):
            fname = code_file.name
            if not fname.lower().endswith(".h"):
                continue
            gen_path = gen_dir / fname
            if not gen_path.exists():
                continue

            yield _sse({
                "type": "stage",
                "stage": "header_doc",
                "file": fname,
                "index": i,
                "total": len(code_files),
                "message": f"Adding ICD documentation comments to {fname}…",
            })

            header_code = _read_text_safe(gen_path)

            doc_prompt = _build_header_documentation_prompt(
                fname=fname,
                header_code=header_code,
                target_summary=target_summary,
            )

            doc_parts: list[str] = []
            try:
                for batch in _batched_stream_text(
                    _call_llm_stream(
                        HEADER_DOCUMENTATION_SYSTEM_PROMPT,
                        doc_prompt,
                        max_tokens=MAX_OUTPUT_TOKENS,
                    )
                ):
                    doc_parts.append(batch)
                    yield _sse({
                        "type": "token",
                        "stage": "header_doc",
                        "file": fname,
                        "token": batch,
                    })
            except Exception as e:
                yield _sse({
                    "type": "info",
                    "stage": "header_doc",
                    "file": fname,
                    "message": f"Documentation pass failed for {fname}: {e} — keeping undocumented version.",
                })
                continue

            documented = _extract_fenced("".join(doc_parts), "c").strip()

            # Safety gate: reject if the LLM changed any declarations.
            # A documented file should always be longer than the original.
            # If it's shorter or empty, the LLM likely truncated — keep original.
            if documented and len(documented) >= len(header_code) * 0.9:
                gen_path.write_text(documented)
                yield _sse({
                    "type": "file_complete",
                    "stage": "header_doc",
                    "file": fname,
                    "size": len(documented),
                })
            else:
                yield _sse({
                    "type": "info",
                    "stage": "header_doc",
                    "file": fname,
                    "message": (
                        f"{fname}: documented output was too short or empty "
                        f"({len(documented)} chars vs {len(header_code)} original) "
                        "— keeping undocumented version."
                    ),
                })

        # ---- Step 3: Verification --------------------------------------
        gen_files = sorted(p for p in gen_dir.iterdir()
                           if p.is_file() and p.suffix in ('.c', '.h'))
        verification_done = (
            is_resume
            and "verification" in set(_pipeline_state(_read_status(session_dir)).get("completed_stages", []))
            and (gen_dir / "verification_report.txt").exists()
        )
        if verification_done:
            yield _sse({
                "type": "stage",
                "stage": "verification",
                "message": "Resuming: reusing completed verification report...",
            })
            yield _sse({
                "type": "info",
                "stage": "verification",
                "message": "Loaded prior verification_report.txt.",
            })
            yield _sse({"type": "stage_complete", "stage": "verification"})
        elif gen_files:
            yield _sse({
                "type": "stage",
                "stage": "verification",
                "message": "Verifying generated code against ICDs, original scripts, and repository\u2026",
            })

            verification_reports: list[str] = []
            verify_diffs: dict[str, str] = {}
            structural_results: dict[str, list[str]] = {}

            for gf in gen_files:
                gfname = gf.name
                generated_code = _read_text_safe(gf)
                pre_verify_code = generated_code
                orig_path = code_dir / gfname
                original_code = _read_text_safe(orig_path) if orig_path.exists() else ""

                yield _sse({
                    "type": "info",
                    "stage": "verification",
                    "file": gfname,
                    "message": f"Running structural checks on {gfname}\u2026",
                })

                # Phase 1: Deterministic structural checks
                structural_issues = _structural_verify(
                    generated_code,
                    repo_dir if has_repo else None,
                    original_code,
                    gfname,
                )
                structural_results[gfname] = structural_issues

                if structural_issues:
                    issue_text = "\n".join(f"  - {i}" for i in structural_issues)
                    yield _sse({
                        "type": "token",
                        "stage": "verification",
                        "file": gfname,
                        "token": f"\nStructural issues in {gfname}:\n{issue_text}\n",
                    })
                else:
                    yield _sse({
                        "type": "token",
                        "stage": "verification",
                        "file": gfname,
                        "token": f"\n{gfname}: structural checks passed.\n",
                    })

                # Phase 2: Focused LLM verification — only when there are
                # structural issues OR when repo context is available for
                # deeper compatibility checking.
                needs_llm = bool(structural_issues) or has_repo
                if needs_llm:
                    yield _sse({
                        "type": "info",
                        "stage": "verification",
                        "file": gfname,
                        "message": f"LLM verification pass on {gfname}\u2026",
                    })

                    issue_guidance = ""
                    if structural_issues:
                        issue_guidance = (
                            "\n\nThe following structural issues were detected — "
                            "you MUST fix these:\n"
                            + "\n".join(f"- {i}" for i in structural_issues)
                            + "\n"
                        )

                    verify_repo = ""
                    if has_repo:
                        verify_repo = _build_file_repo_context(
                            repo_dir, original_code, uploaded_names,
                            max_chars=MAX_REPO_CONTEXT_CHARS,
                        )
                        if not verify_repo:
                            verify_repo = _build_repo_context(
                                repo_dir, exclude_names=uploaded_names,
                            )

                    verify_system = (
                        "You are a code verification expert. Fix structural issues "
                        "and ensure compatibility with the repository headers. "
                        "Output ONLY the corrected complete C file in ```c fences. "
                        "Before the code, write a one-line summary starting with "
                        "'FIXES:' listing corrections made, or 'NO FIXES NEEDED' "
                        "if the code is correct. "
                        "The output MUST include a complete fenced C file. "
                        "Peripheral variations: if this generated HEADER file is "
                        "specific to ONE variation of a peripheral (its name and/or "
                        "banner comment identify a single variation), keep it scoped "
                        "to that one variation — do NOT pull in or merge the other "
                        "variations even though the change spec lists them. If a .c "
                        "file contains multiple per-variation structs, PRESERVE all "
                        "of them and their comments — do not merge, dedupe, or delete "
                        "variations."
                    )

                    sec_issues = f"## Issues to Fix{issue_guidance}" if issue_guidance else ""
                    sec_repo_v = (
                        f"## Repository Headers (must be compatible)\n{verify_repo}"
                        if verify_repo else ""
                    )
                    sec_spec_v = f"## Change Specification (must be applied)\n{change_spec}"
                    sec_code_v = (
                        f"## Code to Verify ({gfname})\n```c\n{generated_code}\n```\n\n"
                        "Output the verified/corrected file."
                    )

                    v_sys_tokens = _estimate_tokens(verify_system)
                    verify_prompt = _assemble_prompt(
                        [
                            ("code_to_verify", sec_code_v, 0),
                            ("issues", sec_issues, 1),
                            ("repo_headers", sec_repo_v, 2),
                            ("change_spec", sec_spec_v, 3),
                        ],
                        max_input_tokens=MAX_INPUT_TOKENS - v_sys_tokens,
                    )

                    try:
                        verify_pieces = []
                        # Verification re-emits the GENERATED file, so the
                        # budget must scale with that, not with the flat
                        # MAX_OUTPUT_TOKENS. A transform-expanded header is
                        # routinely several times MAX_OUTPUT_TOKENS, and a flat
                        # cap silently truncates it.
                        verify_max_tokens = max(
                            MAX_OUTPUT_TOKENS,
                            int(_estimate_tokens(generated_code) * 1.15),
                        )
                        verify_output = _call_llm_complete(
                            verify_system,
                            verify_prompt,
                            max_tokens=verify_max_tokens,
                            max_passes=4,
                            on_chunk=lambda c: verify_pieces.append(c),
                        )
                        for batch in _batched_stream_text(verify_pieces):
                            yield _sse({
                                "type": "token",
                                "stage": "verification",
                                "file": gfname,
                                "token": batch,
                            })
                        verified_code = _extract_fenced(verify_output, "c").strip()
                        verify_issues = (
                            _structural_verify(
                                verified_code, repo_dir if has_repo else None,
                                original_code, gfname,
                            ) if verified_code else ["empty verification output"]
                        )

                        # Guard against a truncated verification pass silently
                        # replacing a good file. The gate below compares against
                        # `generated_code` (what verification was actually given),
                        # not `original_code` -- an expanded header would
                        # otherwise only need 35% of the ORIGINAL upload, so a
                        # badly truncated rewrite sails through.
                        shrink_ratio = (
                            len(verified_code) / len(generated_code)
                            if verified_code and generated_code else 0.0
                        )
                        shrank_badly = shrink_ratio < 0.9
                        if shrank_badly:
                            log.warning(
                                "Verification of %s returned %d chars vs %d given "
                                "(%.0f%%) -- refusing to overwrite.",
                                gfname, len(verified_code), len(generated_code),
                                shrink_ratio * 100,
                            )
                            yield _sse({
                                "type": "warning",
                                "stage": "verification",
                                "file": gfname,
                                "message": (
                                    f"{gfname}: verification returned "
                                    f"{len(verified_code):,} chars vs "
                                    f"{len(generated_code):,} given "
                                    f"({shrink_ratio*100:.0f}%). Keeping the "
                                    f"pre-verification file."
                                ),
                            })

                        if verified_code and not shrank_badly and _looks_complete_c_file(
                            verified_code, generated_code, gfname
                        ) and not verify_issues:
                            gf.write_text(verified_code)
                            verify_diffs[gfname] = _generate_diff(
                                pre_verify_code, verified_code,
                                f"{gfname} (pre-verification)",
                                f"{gfname} (post-verification)",
                            )
                            verification_reports.append(
                                f"{gfname}: LLM verification applied corrections"
                            )
                            yield _sse({
                                "type": "info",
                                "stage": "verification",
                                "file": gfname,
                                "message": f"{gfname} verification complete — updated.",
                            })
                        else:
                            # Final targeted correction pass: constrain to known issues only.
                            fix_prompt = (
                                f"## Structural issues to fix\n"
                                + "\n".join(f"- {i}" for i in verify_issues[:12])
                                + "\n\n"
                                + f"## Repository dependency headers\n{verify_repo}\n\n"
                                + f"## Current code ({gfname})\n```c\n{generated_code}\n```\n\n"
                                + "Rewrite this file to fix the listed issues while preserving ICD-required behavior. "
                                  "Output only one complete ```c fenced file."
                            )
                            fix_output = _call_llm_complete(
                                verify_system,
                                fix_prompt,
                                max_tokens=verify_max_tokens,
                                max_passes=3,
                            )
                            fixed_code = _extract_fenced(fix_output, "c").strip()
                            fixed_issues = (
                                _structural_verify(
                                    fixed_code, repo_dir if has_repo else None,
                                    original_code, gfname,
                                ) if fixed_code else ["empty targeted-fix output"]
                            )
                            fix_ratio = (
                                len(fixed_code) / len(generated_code)
                                if fixed_code and generated_code else 0.0
                            )
                            if fix_ratio and fix_ratio < 0.9:
                                log.warning(
                                    "Targeted fix for %s returned %d chars vs %d "
                                    "given (%.0f%%) -- refusing to overwrite.",
                                    gfname, len(fixed_code), len(generated_code),
                                    fix_ratio * 100,
                                )
                            if (
                                fixed_code and fix_ratio >= 0.9
                                and _looks_complete_c_file(
                                    fixed_code, generated_code, gfname
                                )
                                and not fixed_issues
                            ):
                                gf.write_text(fixed_code)
                                verify_diffs[gfname] = _generate_diff(
                                    pre_verify_code, fixed_code,
                                    f"{gfname} (pre-verification)",
                                    f"{gfname} (post-verification)",
                                )
                                verification_reports.append(
                                    f"{gfname}: LLM verification applied targeted fix pass"
                                )
                                yield _sse({
                                    "type": "info",
                                    "stage": "verification",
                                    "file": gfname,
                                    "message": f"{gfname} verification complete — targeted fixes applied.",
                                })
                            else:
                                verification_reports.append(
                                    f"{gfname}: verification could not produce a complete compilable correction"
                                )
                                yield _sse({
                                    "type": "info",
                                    "stage": "verification",
                                    "file": gfname,
                                    "message": (
                                        f"{gfname} verification still incomplete after targeted retry; "
                                        "keeping generated version."
                                    ),
                                })
                    except Exception as e:
                        log.warning("Verification failed for %s: %s", gfname, e)
                        verification_reports.append(f"{gfname}: error — {e}")
                        yield _sse({
                            "type": "info",
                            "stage": "verification",
                            "file": gfname,
                            "message": f"Verification error for {gfname}: {e} — keeping generated version.",
                        })
                else:
                    verification_reports.append(f"{gfname}: all structural checks passed")

            report_path = gen_dir / "verification_report.txt"
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            report_lines = [
                "=" * 65,
                "VERIFICATION REPORT",
                "=" * 65,
                f"\nGenerated: {now}",
                "Round: Initial Generation",
                "",
                "-" * 65,
                "CHECKS PERFORMED",
                "-" * 65,
                "1. Structural integrity (matching braces, block comments, header guards)",
                "2. #include path resolution against repository codebase",
                "3. Function/symbol presence compared to original source files",
                "4. ICD change specification compliance (LLM-verified)",
                "5. Repository naming and type compatibility (LLM-verified)",
                "6. Variable inventory of generated code files",
                "",
                "-" * 65,
                "RESULTS",
                "-" * 65,
            ]
            for gf_r in gen_files:
                gfn = gf_r.name
                report_lines.append(f"\n### {gfn}\n")
                issues = structural_results.get(gfn, [])
                report_lines.append("Structural Checks:")
                if not issues:
                    report_lines.append("  [PASS] All structural checks passed")
                else:
                    for iss in issues:
                        report_lines.append(f"  [ISSUE] {iss}")
                status_line = next(
                    (r for r in verification_reports if r.startswith(gfn + ":")), ""
                )
                if status_line:
                    summary = status_line.split(": ", 1)[1] if ": " in status_line else status_line
                    report_lines.append(f"\nVerification Outcome: {summary}")
                diff_text = verify_diffs.get(gfn)
                if diff_text and diff_text != "(no changes)\n":
                    report_lines.append("\nChanges applied during verification:")
                    report_lines.append(diff_text)
                else:
                    report_lines.append("\nNo changes applied during verification.")

            report_lines.extend([
                "",
                "-" * 65,
                "VARIABLE INVENTORY",
                "-" * 65,
            ])
            for gf_r in gen_files:
                gfn = gf_r.name
                final_code = _read_text_safe(gf_r)
                variables = _extract_c_variables(final_code, gfn)
                report_lines.append(f"\n### {gfn}\n")
                report_lines.append(_format_variable_inventory(variables, gfn))

            report_path.write_text("\n".join(report_lines) + "\n")
            yield _sse({"type": "stage_complete", "stage": "verification"})

        # ---- Step 3.5: Per-file compile gate (.c -> .o) ----------------
        # Compile each revised .c on its own with the project's toolchain
        # so the .c/.h/.o triples and the per-file compile_report are
        # downloadable BEFORE the long-running sandbox build kicks in.
        # The same agentic LLM-fix loop used by the sandbox build is
        # applied here, scoped to one .c (+ its companion .h) at a time.
        completed_now = set(_pipeline_state(_read_status(session_dir)).get("completed_stages", []))
        for evt in run_per_file_compile(
            session_dir=session_dir,
            gen_dir=gen_dir,
            code_dir=code_dir,
            repo_dir=repo_dir,
            has_repo=has_repo,
            change_spec=change_spec,
            repo_knowledge=repo_knowledge,
            is_resume=is_resume,
            completed_stages=completed_now,
        ):
            yield evt

        # ---- Step 4: Sandbox build ------------------------------------
        sandbox_build_success = None
        if has_repo or REMOTE_BUILD_ENABLED:
            sandbox_done = (
                is_resume
                and "sandbox_build" in set(
                    _pipeline_state(_read_status(session_dir)).get("completed_stages", [])
                )
                and (session_dir / "built_repo.zip").exists()
            )
            if sandbox_done:
                sandbox_build_success = bool(
                    _pipeline_state(_read_status(session_dir)).get(
                        "sandbox_build_success", True
                    )
                )
                yield _sse({
                    "type": "stage",
                    "stage": "sandbox_build",
                    "message": "Resuming: reusing completed sandbox build artifacts...",
                })
                yield _sse({
                    "type": "sandbox_build_result",
                    "stage": "sandbox_build",
                    "success": sandbox_build_success,
                    "iterations": 0,
                    "message": "Reused prior built_repo.zip and sandbox_build_log.txt.",
                })
                yield _sse({"type": "stage_complete", "stage": "sandbox_build"})
            else:
                yield _sse({
                    "type": "stage",
                    "stage": "sandbox_build",
                    "message": "Building generated code inside repository sandbox…",
                })
                build_iter = (
                    _remote_build_iterate(session_dir, gen_dir, workspace_path=str(session_dir))
                    if REMOTE_BUILD_ENABLED
                    else _sandbox_build_iterate(
                        session_dir=session_dir,
                        gen_dir=gen_dir,
                        repo_dir=repo_dir,
                        change_spec=change_spec,
                        uploaded_names=uploaded_names,
                        has_repo=has_repo,
                        repo_knowledge=repo_knowledge,
                        sandbox_max_retries=sandbox_max_retries,
                    )
                )
                for evt in build_iter:
                    yield evt
                    try:
                        payload = json.loads(
                            evt.split("data: ", 1)[1].split("\n", 1)[0]
                        )
                        if payload.get("type") == "sandbox_build_result":
                            sandbox_build_success = payload.get("success", False)
                    except Exception:
                        pass
                yield _sse({"type": "stage_complete", "stage": "sandbox_build"})

            status = _read_status(session_dir)
            status["sandbox_build_success"] = sandbox_build_success
            _write_status(session_dir, status)

        # ---- Done -----------------------------------------------------
        status = _read_status(session_dir)
        status["pause_requested"] = False
        status["state"] = "completed"
        # `generated_files` drives the UI preview tabs, so .o files (binary
        # objects from the per-file compile gate) are excluded — they are
        # still in `gen_dir` and ship in the /api/download zip.
        status["generated_files"] = sorted(
            p.name for p in gen_dir.iterdir()
            if p.is_file() and p.suffix.lower() != ".o"
        )
        _write_status(session_dir, status)
        complete_payload = {"type": "complete", "files": status["generated_files"]}
        if sandbox_build_success is not None:
            complete_payload["sandbox_build"] = sandbox_build_success
        yield _sse(complete_payload)

    return StreamingResponse(
        _pausable_stream(session_dir, event_stream()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/download/{session_id}")
async def download_all(session_id: str):
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    gen_dir = session_dir / "generated_code"
    files = sorted(p for p in gen_dir.iterdir() if p.is_file())
    report_candidates = [
        session_dir / "icd_analysis.txt",
        session_dir / "change_spec.txt",
        session_dir / "change_spec_raw.txt",
        session_dir / "target_summary.txt",
        session_dir / "repo_knowledge.txt",
        session_dir / "sandbox_build_log.txt",
        session_dir / PAUSE_SUMMARY_FILE,
        session_dir / PIPELINE_EVENTS_FILE,
        session_dir / "status.json",
        gen_dir / "verification_report.txt",
        gen_dir / "compile_report.txt",
        gen_dir / "icd_analysis.txt",
    ]
    if not files and not any(p.exists() for p in report_candidates):
        raise HTTPException(status_code=404, detail="No generated files or reports found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if f.name in ("icd_analysis.txt", "verification_report.txt"):
                continue
            zf.write(f, f.name)
        for candidate in [gen_dir / "icd_analysis.txt",
                          session_dir / "icd_analysis.txt"]:
            if candidate.exists():
                zf.write(candidate, "icd_analysis.txt")
                log.info("Added icd_analysis.txt from %s", candidate)
                break
        else:
            log.warning("icd_analysis.txt not found for session %s", session_id[:8])
        vr = gen_dir / "verification_report.txt"
        if vr.exists():
            zf.write(vr, "verification_report.txt")
            log.info("Added verification_report.txt")
        # Per-file compile gate report (added by name above already, but
        # this also handles the resume-only case where gen_dir lists the
        # report among `files` and we want to ensure it's present.)
        compile_rep = gen_dir / "compile_report.txt"
        if compile_rep.exists() and "compile_report.txt" not in zf.namelist():
            zf.write(compile_rep, "compile_report.txt")
            log.info("Added compile_report.txt")
        built_repo = session_dir / "built_repo.zip"
        if built_repo.exists():
            zf.write(built_repo, "built_repo.zip")
            log.info("Added built_repo.zip")
        build_log = session_dir / "sandbox_build_log.txt"
        if build_log.exists():
            zf.write(build_log, "sandbox_build_log.txt")
            log.info("Added sandbox_build_log.txt")
        for report in [
            session_dir / PAUSE_SUMMARY_FILE,
            session_dir / PIPELINE_EVENTS_FILE,
            session_dir / "status.json",
            session_dir / "change_spec.txt",
            session_dir / "change_spec_raw.txt",
            session_dir / "target_summary.txt",
            session_dir / "repo_knowledge.txt",
            session_dir / "playbook.jsonl",
        ]:
            if report.exists():
                zf.write(report, report.name)
                log.info("Added %s", report.name)
        # Per-attempt artifacts from the agentic debug pipeline (small JSON
        # blobs + diffs); kept under ./agentic_attempts/ in the ZIP.
        agentic_attempts_dir = session_dir / "agentic_attempts"
        if agentic_attempts_dir.exists() and agentic_attempts_dir.is_dir():
            added = 0
            for af in sorted(agentic_attempts_dir.rglob("*")):
                if af.is_file():
                    arc = "agentic_attempts/" + str(
                        af.relative_to(agentic_attempts_dir)
                    )
                    zf.write(af, arc)
                    added += 1
            if added:
                log.info("Added %d agentic_attempts/* files", added)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f"attachment; filename=refactored_code_{session_id[:8]}.zip"
            )
        },
    )


@app.get("/api/download-repo/{session_id}")
async def download_repo(session_id: str):
    """Download just the built repository ZIP (sandbox output)."""
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    built_repo = session_dir / "built_repo.zip"
    if not built_repo.exists():
        raise HTTPException(
            status_code=404,
            detail="No built repository available (sandbox build may not have run)",
        )

    buf = io.BytesIO(built_repo.read_bytes())
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f"attachment; filename=built_repo_{session_id[:8]}.zip"
            )
        },
    )


@app.get("/api/preview/{session_id}/{filename}")
async def preview_file(session_id: str, filename: str):
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    file_path = (session_dir / "generated_code" / filename).resolve()
    if not file_path.is_relative_to((session_dir / "generated_code").resolve()):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return {"filename": filename, "content": _read_text_safe(file_path)}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    session_dir = SESSIONS_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
    return {"deleted": True}


# ---------------------------------------------------------------------------
# Conversation endpoints
# ---------------------------------------------------------------------------

@app.post("/api/conversation/{session_id}")
async def add_conversation_message(session_id: str, msg: ChatMessage):
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    if not msg.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    conv_path = session_dir / "conversation.json"
    conversation = json.loads(conv_path.read_text()) if conv_path.exists() else []
    conversation.append({
        "role": "user",
        "content": msg.message.strip(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    conv_path.write_text(json.dumps(conversation, indent=2))
    return {"messages": conversation}


@app.get("/api/conversation/{session_id}")
async def get_conversation(session_id: str):
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    conv_path = session_dir / "conversation.json"
    if not conv_path.exists():
        return {"messages": []}
    return {"messages": json.loads(conv_path.read_text())}


# ---------------------------------------------------------------------------
# Re-generation endpoint
# ---------------------------------------------------------------------------

@app.get("/api/regenerate/{session_id}")
async def regenerate(
    session_id: str,
    sandbox_retries: str | None = Query(
        None,
        description=(
            "Sandbox build reiteration budget: a positive integer, or "
            "'indefinite' to keep iterating until the build succeeds."
        ),
    ),
):
    """Re-generate code incorporating conversation feedback. Returns SSE stream."""
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    status = json.loads((session_dir / "status.json").read_text())
    if status.get("state") != "completed":
        raise HTTPException(status_code=400, detail="Initial processing must complete first")

    try:
        sandbox_max_retries = _resolve_sandbox_max_retries(
            sandbox_retries, status,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if sandbox_max_retries is _SANDBOX_RETRIES_UNSET:
        pass  # keep whatever was stored (or env)
    elif sandbox_max_retries is None:
        status["sandbox_retries_mode"] = "indefinite"
        status["sandbox_max_retries"] = None
        _write_status(session_dir, status)
    else:
        status["sandbox_retries_mode"] = "finite"
        status["sandbox_max_retries"] = int(sandbox_max_retries)
        _write_status(session_dir, status)

    conv_path = session_dir / "conversation.json"
    conversation = json.loads(conv_path.read_text()) if conv_path.exists() else []
    if not conversation:
        raise HTTPException(status_code=400, detail="No feedback messages to incorporate")

    change_spec = _read_text_safe(session_dir / "change_spec.txt")
    code_dir = session_dir / "original_code"
    gen_dir = session_dir / "generated_code"
    code_files = sorted(p for p in code_dir.iterdir() if p.is_file())

    repo_dir = session_dir / "repo_contents"
    has_repo = repo_dir.exists() and any(repo_dir.rglob("*"))
    uploaded_names = {p.name for p in code_files}

    regen_count = status.get("regeneration_count", 0) + 1

    target_summary_path = session_dir / "target_summary.txt"
    target_summary = (
        _read_text_safe(target_summary_path) if target_summary_path.exists()
        else _truncate_text(
            _read_text_safe(session_dir / "target_icd.txt"), 12_000, "target_icd"
        )
    )

    # ``repo_knowledge.txt`` on disk is
    # intentionally the FULL (untruncated, QA download) flavour, so we
    # cannot just read them into memory — that would defeat the
    # prompt-budget protection.  Instead we always re-derive the
    # DISTILLED flavour from the live ``repo_dir`` / ``code_dir`` (a
    # cheap regex pass) when source is available, and only fall back to
    # the saved file when source is missing.  We also regenerate the
    # saved FULL flavour if it's missing so the downloadable ZIP stays
    # complete across regeneration rounds.
    repo_knowledge_path = session_dir / "repo_knowledge.txt"
    repo_knowledge = ""
    if has_repo and repo_dir.exists():
        repo_knowledge = _build_repo_knowledge(repo_dir, full=False)
        if not repo_knowledge_path.exists():
            try:
                full = _build_repo_knowledge(repo_dir, full=True)
                repo_knowledge_path.write_text(full)
            except Exception as e:                   # noqa: BLE001
                log.warning(
                    "Regenerate: could not regenerate full "
                    "repo_knowledge.txt: %s", e,
                )
    elif repo_knowledge_path.exists():
        # No live repo to re-distill from — best-effort use of the
        # saved full report.  This puts more pressure on the prompt
        # budget but is preferable to losing repo knowledge entirely.
        repo_knowledge = _read_text_safe(repo_knowledge_path)

    prev_dir = session_dir / f"generated_code_v{regen_count - 1}"
    if not prev_dir.exists():
        shutil.copytree(gen_dir, prev_dir)

    conversation_ctx = _build_conversation_context(conversation)

    log.info(
        "Regenerate %s: round %d, %d code files, %d conversation messages",
        session_id[:8], regen_count, len(code_files), len(conversation),
    )

    def event_stream():
        yield _sse({
            "type": "stage",
            "stage": "regeneration",
            "message": f"Re-generating code (round {regen_count}) incorporating user feedback\u2026",
        })

        yield _sse({
            "type": "info",
            "stage": "regeneration",
            "message": "Checking llama-server availability\u2026",
        })
        try:
            _wait_for_llm_ready(timeout_s=300)
        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})
            return

        yield _sse({
            "type": "info",
            "stage": "regeneration",
            "message": f"Incorporating {len(conversation)} feedback message(s) into re-generation\u2026",
        })

        all_code_ctx = ""
        for cf in code_files:
            all_code_ctx += f"\n### File: {cf.name}\n```c\n{_read_text_safe(cf)}\n```\n"
        all_code_ctx = _truncate_text(all_code_ctx, MAX_CODE_CONTEXT_CHARS, "all_code_ctx")

        regen_diffs: dict[str, str] = {}

        for i, code_file in enumerate(code_files):
            fname = code_file.name
            original = _read_text_safe(code_file)

            prev_gen_path = prev_dir / fname
            prev_generated = _read_text_safe(prev_gen_path) if prev_gen_path.exists() else ""

            yield _sse({
                "type": "stage",
                "stage": "transform",
                "file": fname,
                "index": i,
                "total": len(code_files),
                "message": f"Re-generating {fname} with user feedback\u2026",
            })

            transform_system = (
                "You are an expert C programmer specializing in embedded systems "
                "and interface implementations governed by Interface Control Documents.\n\n"
                "TARGET TOOLCHAIN:\n"
                "- Xilinx SDK 2018.x with GCC 7.3.1 (arm-none-eabi / mb-gcc)\n"
                "- C standard: C99 with GCC extensions (-std=gnu99) — "
                "GCC-specific constructs commonly used in embedded code "
                "are ACCEPTED, including unnamed structs/unions, "
                "`__attribute__`, statement expressions, and M_PI / M_E "
                "from <math.h>\n"
                "- C library: newlib (NOT glibc) — do NOT use glibc-specific "
                "functions (e.g. asprintf, getline, strdup, strndup, vasprintf)\n"
                "- Use <stdint.h> fixed-width types (uint8_t, uint16_t, uint32_t)\n"
                "- No POSIX headers (unistd.h, sys/*.h) — embedded freestanding\n"
                "- Avoid GCC extensions added after GCC 7 (no __attribute__((access)), "
                "no __builtin_expect_with_probability, etc.)\n\n"
                "The user has previously generated code that had issues (build errors, "
                "warnings, or other problems). You must re-generate the code fixing ALL "
                "reported issues while maintaining full ICD compliance.\n\n"
                "CRITICAL: You must consider ALL provided context HOLISTICALLY — the "
                "ICD change specification, repository codebase knowledge, dependency "
                "headers, AND user feedback — to produce correct code. Do NOT focus "
                "solely on user-reported errors at the expense of ICD compliance or "
                "repository compatibility. Fixing one error must not introduce "
                "regressions elsewhere. Use the repository's actual type names, "
                "function signatures, macros, and variable conventions as ground truth.\n\n"
                "RULES:\n"
                "1. Output ONLY the complete, transformed C source code\n"
                "2. Fix ALL issues described in the user feedback/error logs\n"
                "3. Apply ALL changes required by the target ICD per the change specification\n"
                "4. Preserve the overall code architecture, style, and conventions\n"
                "5. Ensure type correctness and compilability with GCC 7.3.1\n"
                "6. Keep header/source consistency across the project\n"
                "7. Do NOT add prose explanations — only output C code\n"
                "8. Wrap the entire output in ```c ... ``` fences\n"
                "9. Match naming conventions from the repository codebase\n"
                "10. Ensure #include directives reference correct repository headers\n"
                "11. Maintain compatibility with all dependent modules in the repository\n"
                "12. Cross-check every fix against the change spec and repo headers "
                "to prevent error loops"
                f"{PERIPHERAL_VARIATION_CODEGEN_GUIDANCE}"
            )

            file_repo_ctx = ""
            if has_repo:
                file_repo_ctx = _build_file_repo_context(
                    repo_dir, original, uploaded_names,
                    max_chars=MAX_REPO_CONTEXT_CHARS,
                )
                if not file_repo_ctx:
                    file_repo_ctx = _build_repo_context(
                        repo_dir, exclude_names=uploaded_names,
                    )

            sec_conv = conversation_ctx
            sec_prev = ""
            if prev_generated:
                sec_prev = (
                    f"## Previously Generated Code ({fname}) \u2014 HAS ISSUES\n"
                    f"```c\n{prev_generated}\n```\n"
                )
            sec_change = f"## Change Specification (Source ICD -> Target ICD)\n\n{change_spec}"
            sec_target_v = f"## Target ICD Consolidated Summary\n\n{target_summary}"
            sec_repo = ""
            if file_repo_ctx:
                sec_repo = (
                    f"## Repository Dependency Context\n"
                    f"Use exact type names, function signatures, macros, and naming "
                    f"conventions from these files.\n\n{file_repo_ctx}"
                )
            sec_knowledge = ""
            # if repo_knowledge:
            #     sec_knowledge = (
            #         f"## Repository Codebase Knowledge\n"
            #         f"Detailed inventory of types, functions, variables, and macros "
            #         f"from the repository. Use these as ground truth for naming, "
            #         f"types, and conventions.\n\n{repo_knowledge}"
            #     )
            sec_code = f"## All Project Files (cross-file context)\n{all_code_ctx}"
            is_header = fname.lower().endswith(".h")
            if is_header:
                base_stem = fname[:-2]  # drop trailing ".h"
                regen_variation_instr = (
                    "If the change specification describes MULTIPLE variations of "
                    "the same peripheral, output a SEPARATE, COMPLETE .h file for "
                    "EACH variation instead of one combined header. Precede each "
                    "file with a line of the form `### <filename>` (for example "
                    f"`### {base_stem}_<Variation>.h`) immediately followed by its "
                    "```c fenced block. Each header MUST be self-contained (own "
                    "include guard, #includes, banner comment naming the variation, "
                    "and only that variation's declarations). Do NOT emit any other "
                    "`###` headings. If only one variation applies, output a single "
                    "complete header with NO `### ` marker. "
                )
            else:
                regen_variation_instr = (
                    "If the change specification lists multiple variations of the "
                    "same peripheral, keep a separate, clearly-commented struct "
                    "(and matching code) for EACH variation — do not merge or drop "
                    "variations. "
                )
            sec_file = (
                f"## Original File: {fname}\n\n```c\n{original}\n```\n\n"
                "Re-generate this file to conform to the Target ICD while fixing "
                "ALL issues from the user feedback. Use the change specification "
                "and repository knowledge as ground truth — do not introduce "
                "regressions. "
                f"{regen_variation_instr}"
                "Output the complete file in "
                "```c fences. Do not omit any sections."
            )

            system_tokens = _estimate_tokens(transform_system)
            transform_prompt = _assemble_prompt(
                [
                    ("file_to_transform", sec_file, 0),
                    ("user_feedback", sec_conv, 0),
                    ("change_spec", sec_change, 1),
                    ("previous_generated", sec_prev, 1),
                    ("repo_dependencies", sec_repo, 2),
                    ("repo_knowledge", sec_knowledge, 2),
                    ("target_summary", sec_target_v, 3),
                    ("cross_file_ctx", sec_code, 4),
                ],
                max_input_tokens=MAX_INPUT_TOKENS - system_tokens,
            )
            log.info("Regen prompt for %s: %d chars (%d est. tokens)",
                     fname, len(transform_prompt), _estimate_tokens(transform_prompt))

            clean = ""
            last_raw = ""
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                file_parts: list[str] = []
                attempt_prompt = transform_prompt
                if attempt > 1:
                    yield _sse({
                        "type": "info",
                        "stage": "transform",
                        "file": fname,
                        "message": f"Incomplete output; continuation pass {attempt}/{max_attempts}.",
                    })
                    attempt_prompt = (
                        f"{transform_prompt}\n\n"
                        "The previous output was incomplete/truncated. "
                        "Continue from the exact point where it stopped, output ONLY "
                        "the missing remainder, and ensure braces/comments/preprocessor "
                        "blocks are closed.\n\n"
                        f"Previous partial output:\n```c\n{clean}\n```"
                    )
                try:
                    max_tokens=max(MAX_OUTPUT_TOKENS, int(_estimate_tokens(original) * 1.15))
                    for batch in _batched_stream_text(
                        _call_llm_stream(
                            transform_system, attempt_prompt, max_tokens=max_tokens,
                        )
                    ):
                        file_parts.append(batch)
                        yield _sse({
                            "type": "token",
                            "stage": "transform",
                            "file": fname,
                            "token": batch,
                        })
                except Exception as e:
                    yield _sse({
                        "type": "error",
                        "message": f"Error re-generating {fname}: {e}",
                        "file": fname,
                    })
                    break

                last_raw = "".join(file_parts)
                last_chunk = _extract_fenced(last_raw, "c").strip()
                if attempt == 1:
                    clean = last_chunk
                else:
                    clean = (clean.rstrip() + "\n" + last_chunk.lstrip()).strip()

                if _looks_complete_c_file(clean, original, fname):
                    break

            # Multiple peripheral variations -> one self-contained .h file
            # per variation (### <name>.h + fenced block).
            variation_headers = _split_variation_header_files(last_raw, fname)
            if variation_headers:
                incomplete = [
                    n for n, c in variation_headers.items()
                    if not _looks_complete_header(c)
                ]
                if not incomplete:
                    for vname, vcode in sorted(variation_headers.items()):
                        prev_vtext = (
                            _read_text_safe(gen_dir / vname)
                            if (gen_dir / vname).exists() else ""
                        )
                        (gen_dir / vname).write_text(vcode)
                        if prev_vtext:
                            regen_diffs[vname] = _generate_diff(
                                prev_vtext, vcode,
                                f"{vname} (round {regen_count - 1})",
                                f"{vname} (round {regen_count})",
                            )
                        yield _sse({
                            "type": "file_complete",
                            "file": vname,
                            "size": len(vcode),
                        })
                    log.info(
                        "Regenerate %s: emitted %d per-variation header files: %s",
                        fname, len(variation_headers),
                        ", ".join(sorted(variation_headers)),
                    )
                    continue
                log.warning(
                    "Regenerate %s: per-variation header split incomplete (%s); "
                    "falling back to single-file output.",
                    fname, ", ".join(incomplete),
                )

            if not _looks_complete_c_file(clean, original, fname):
                yield _sse({
                    "type": "error",
                    "file": fname,
                    "message": (
                        f"{fname} still appears incomplete after retries. "
                        "Try with smaller ICD PDFs or a GPU-enabled llama runtime."
                    ),
                })
                continue

            (gen_dir / fname).write_text(clean)

            if prev_generated:
                regen_diffs[fname] = _generate_diff(
                    prev_generated, clean,
                    f"{fname} (round {regen_count - 1})",
                    f"{fname} (round {regen_count})",
                )

            yield _sse({
                "type": "file_complete",
                "file": fname,
                "size": len(clean),
            })

        # ---- Verification of re-generated code -------------------------
        gen_files = sorted(
            p for p in gen_dir.iterdir()
            if p.is_file() and p.suffix in ('.c', '.h')
        )
        if gen_files:
            yield _sse({
                "type": "stage",
                "stage": "verification",
                "message": "Verifying re-generated code\u2026",
            })

            verification_reports: list[str] = []
            verify_diffs: dict[str, str] = {}
            structural_results: dict[str, list[str]] = {}

            for gf in gen_files:
                gfname = gf.name
                generated_code = _read_text_safe(gf)
                pre_verify_code = generated_code
                orig_path = code_dir / gfname
                original_code = _read_text_safe(orig_path) if orig_path.exists() else ""

                yield _sse({
                    "type": "info",
                    "stage": "verification",
                    "file": gfname,
                    "message": f"Structural checks on {gfname}\u2026",
                })

                structural_issues = _structural_verify(
                    generated_code,
                    repo_dir if has_repo else None,
                    original_code,
                    gfname,
                )
                structural_results[gfname] = structural_issues

                if structural_issues:
                    issue_text = "\n".join(f"  - {si}" for si in structural_issues)
                    yield _sse({
                        "type": "token",
                        "stage": "verification",
                        "file": gfname,
                        "token": f"\nStructural issues in {gfname}:\n{issue_text}\n",
                    })
                else:
                    yield _sse({
                        "type": "token",
                        "stage": "verification",
                        "file": gfname,
                        "token": f"\n{gfname}: structural checks passed.\n",
                    })

                needs_llm = bool(structural_issues) or has_repo
                if needs_llm:
                    yield _sse({
                        "type": "info",
                        "stage": "verification",
                        "file": gfname,
                        "message": f"LLM verification pass on {gfname}\u2026",
                    })

                    issue_guidance = ""
                    if structural_issues:
                        issue_guidance = (
                            "\n\nThe following structural issues were detected \u2014 "
                            "you MUST fix these:\n"
                            + "\n".join(f"- {si}" for si in structural_issues)
                            + "\n"
                        )

                    verify_repo = ""
                    if has_repo:
                        verify_repo = _build_file_repo_context(
                            repo_dir, original_code, uploaded_names,
                            max_chars=MAX_REPO_CONTEXT_CHARS,
                        )
                        if not verify_repo:
                            verify_repo = _build_repo_context(
                                repo_dir, exclude_names=uploaded_names,
                            )

                    verify_system = (
                        "You are a code verification expert. Fix structural issues "
                        "and ensure compatibility with the repository headers. "
                        "The user has reported specific build errors/issues \u2014 ensure "
                        "the code fixes those problems as well. "
                        "Output ONLY the corrected complete C file in ```c fences. "
                        "Before the code, write a one-line summary starting with "
                        "'FIXES:' listing corrections made, or 'NO FIXES NEEDED' "
                        "if the code is correct. "
                        "The output MUST include a complete fenced C file. "
                        "Peripheral variations: if this generated HEADER file is "
                        "specific to ONE variation of a peripheral (its name and/or "
                        "banner comment identify a single variation), keep it scoped "
                        "to that one variation — do NOT pull in or merge the other "
                        "variations even though the change spec lists them. If a .c "
                        "file contains multiple per-variation structs, PRESERVE all "
                        "of them and their comments — do not merge, dedupe, or delete "
                        "variations."
                    )

                    sec_issues = f"## Issues to Fix{issue_guidance}" if issue_guidance else ""
                    sec_repo_v = (
                        f"## Repository Headers (must be compatible)\n{verify_repo}"
                        if verify_repo else ""
                    )
                    sec_spec_v = f"## Change Specification (must be applied)\n{change_spec}"
                    sec_conv_v = conversation_ctx
                    sec_code_v = (
                        f"## Code to Verify ({gfname})\n```c\n{generated_code}\n```\n\n"
                        "Output the verified/corrected file."
                    )

                    v_sys_tokens = _estimate_tokens(verify_system)
                    verify_prompt = _assemble_prompt(
                        [
                            ("code_to_verify", sec_code_v, 0),
                            ("issues", sec_issues, 1),
                            ("user_feedback", sec_conv_v, 1),
                            ("repo_headers", sec_repo_v, 2),
                            ("change_spec", sec_spec_v, 3),
                        ],
                        max_input_tokens=MAX_INPUT_TOKENS - v_sys_tokens,
                    )

                    try:
                        verify_pieces: list[str] = []
                        verify_output = _call_llm_complete(
                            verify_system,
                            verify_prompt,
                            max_tokens=MAX_OUTPUT_TOKENS,
                            max_passes=4,
                            on_chunk=lambda c: verify_pieces.append(c),
                        )
                        for batch in _batched_stream_text(verify_pieces):
                            yield _sse({
                                "type": "token",
                                "stage": "verification",
                                "file": gfname,
                                "token": batch,
                            })
                        verified_code = _extract_fenced(verify_output, "c").strip()
                        v_issues = (
                            _structural_verify(
                                verified_code, repo_dir if has_repo else None,
                                original_code, gfname,
                            ) if verified_code else ["empty verification output"]
                        )

                        if verified_code and _looks_complete_c_file(
                            verified_code, original_code, gfname
                        ) and not v_issues:
                            gf.write_text(verified_code)
                            verify_diffs[gfname] = _generate_diff(
                                pre_verify_code, verified_code,
                                f"{gfname} (pre-verification)",
                                f"{gfname} (post-verification)",
                            )
                            verification_reports.append(
                                f"{gfname}: LLM verification applied corrections"
                            )
                        else:
                            fix_prompt = (
                                f"## Structural issues to fix\n"
                                + "\n".join(f"- {fi}" for fi in v_issues[:12])
                                + "\n\n"
                                + f"## Repository dependency headers\n{verify_repo}\n\n"
                                + f"## Current code ({gfname})\n```c\n{generated_code}\n```\n\n"
                                + "Rewrite this file to fix the listed issues while preserving "
                                  "ICD-required behavior. Output only one complete ```c fenced file."
                            )
                            fix_output = _call_llm_complete(
                                verify_system, fix_prompt,
                                max_tokens=MAX_OUTPUT_TOKENS, max_passes=3,
                            )
                            fixed_code = _extract_fenced(fix_output, "c").strip()
                            fixed_issues = (
                                _structural_verify(
                                    fixed_code, repo_dir if has_repo else None,
                                    original_code, gfname,
                                ) if fixed_code else ["empty targeted-fix output"]
                            )
                            if fixed_code and _looks_complete_c_file(
                                fixed_code, original_code, gfname
                            ) and not fixed_issues:
                                gf.write_text(fixed_code)
                                verify_diffs[gfname] = _generate_diff(
                                    pre_verify_code, fixed_code,
                                    f"{gfname} (pre-verification)",
                                    f"{gfname} (post-verification)",
                                )
                                verification_reports.append(
                                    f"{gfname}: LLM verification applied targeted fix pass"
                                )
                            else:
                                verification_reports.append(
                                    f"{gfname}: verification could not produce a complete compilable correction"
                                )
                    except Exception as e:
                        log.warning("Verification failed for %s: %s", gfname, e)
                        verification_reports.append(f"{gfname}: error \u2014 {e}")
                else:
                    verification_reports.append(f"{gfname}: all structural checks passed")

            # ---- Build comprehensive report ----
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            report_lines = [
                "=" * 65,
                "VERIFICATION REPORT",
                "=" * 65,
                f"\nGenerated: {now}",
                f"Re-generation Round: {regen_count}",
                "",
                "-" * 65,
                "CHECKS PERFORMED",
                "-" * 65,
                "1. Structural integrity (matching braces, block comments, header guards)",
                "2. #include path resolution against repository codebase",
                "3. Function/symbol presence compared to original source files",
                "4. ICD change specification compliance (LLM-verified)",
                "5. Repository naming and type compatibility (LLM-verified)",
                "6. Variable inventory of generated code files",
                "",
            ]

            if regen_diffs:
                report_lines.extend([
                    "-" * 65,
                    f"CHANGES FROM USER FEEDBACK (Round {regen_count})",
                    "-" * 65,
                    "",
                    "User provided the following feedback:",
                ])
                for cmsg in conversation:
                    if cmsg.get("role") == "user":
                        content_preview = cmsg.get("content", "")[:2000]
                        report_lines.append(f"\n> {content_preview}")
                report_lines.append("")

                for rd_fname in sorted(regen_diffs):
                    report_lines.extend([
                        f"\n### {rd_fname}",
                        "\nChanges applied based on user feedback:",
                        regen_diffs[rd_fname],
                    ])
                report_lines.append("")

            report_lines.extend([
                "-" * 65,
                "VERIFICATION RESULTS",
                "-" * 65,
            ])
            for gf_r in gen_files:
                gfn = gf_r.name
                report_lines.append(f"\n### {gfn}\n")
                issues = structural_results.get(gfn, [])
                report_lines.append("Structural Checks:")
                if not issues:
                    report_lines.append("  [PASS] All structural checks passed")
                else:
                    for iss in issues:
                        report_lines.append(f"  [ISSUE] {iss}")
                status_line = next(
                    (r for r in verification_reports if r.startswith(gfn + ":")), ""
                )
                if status_line:
                    summary = status_line.split(": ", 1)[1] if ": " in status_line else status_line
                    report_lines.append(f"\nVerification Outcome: {summary}")
                diff_text = verify_diffs.get(gfn)
                if diff_text and diff_text != "(no changes)\n":
                    report_lines.append("\nChanges applied during verification:")
                    report_lines.append(diff_text)
                else:
                    report_lines.append("\nNo changes applied during verification.")

            report_lines.extend([
                "",
                "-" * 65,
                "VARIABLE INVENTORY",
                "-" * 65,
            ])
            for gf_r in gen_files:
                gfn = gf_r.name
                final_code = _read_text_safe(gf_r)
                variables = _extract_c_variables(final_code, gfn)
                report_lines.append(f"\n### {gfn}\n")
                report_lines.append(_format_variable_inventory(variables, gfn))

            (gen_dir / "verification_report.txt").write_text(
                "\n".join(report_lines) + "\n"
            )
            yield _sse({"type": "stage_complete", "stage": "verification"})

        # ---- Sandbox build (re-generation) ----
        sandbox_build_success = None
        if has_repo or REMOTE_BUILD_ENABLED:
            yield _sse({
                "type": "stage",
                "stage": "sandbox_build",
                "message": "Building re-generated code inside repository sandbox\u2026",
            })

            build_iter = (
                _remote_build_iterate(session_dir, gen_dir, workspace_path=str(session_dir))
                if REMOTE_BUILD_ENABLED
                else _sandbox_build_iterate(
                    session_dir=session_dir,
                    gen_dir=gen_dir,
                    repo_dir=repo_dir,
                    change_spec=change_spec,
                    uploaded_names=uploaded_names,
                    has_repo=has_repo,
                    repo_knowledge=repo_knowledge,
                    sandbox_max_retries=sandbox_max_retries,
                )
            )
            for evt in build_iter:
                yield evt
                try:
                    payload = json.loads(
                        evt.split("data: ", 1)[1].split("\n", 1)[0]
                    )
                    if payload.get("type") == "sandbox_build_result":
                        sandbox_build_success = payload.get("success", False)
                except Exception:
                    pass
            yield _sse({"type": "stage_complete", "stage": "sandbox_build"})

            status["sandbox_build_success"] = sandbox_build_success
            (session_dir / "status.json").write_text(json.dumps(status))

        # ---- Done ----
        conv_updated = json.loads(conv_path.read_text()) if conv_path.exists() else []
        conv_updated.append({
            "role": "system",
            "content": (
                f"Re-generation round {regen_count} completed. "
                f"{len(status.get('generated_files', []))} files processed. "
                "See verification report for detailed changes."
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        conv_path.write_text(json.dumps(conv_updated, indent=2))

        status["state"] = "completed"
        status["regeneration_count"] = regen_count
        # Filter .o objects: not previewable in the UI; still in the zip.
        status["generated_files"] = sorted(
            p.name for p in gen_dir.iterdir()
            if p.is_file() and p.suffix.lower() != ".o"
        )
        (session_dir / "status.json").write_text(json.dumps(status))
        complete_payload = {"type": "complete", "files": status["generated_files"]}
        if sandbox_build_success is not None:
            complete_payload["sandbox_build"] = sandbox_build_success
        yield _sse(complete_payload)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Registered last so /static/app.js and other explicit routes take precedence
# where the path overlaps with StaticFiles.
app.mount(
    "/static", StaticFiles(directory=str(STATIC_DIR)), name="static",
)
