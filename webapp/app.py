"""
ICD-Based C Code Refactorer — Transform C code between ICD versions.

Upload original .c/.h files, source ICD (PDF), and target ICD (PDF).
The tool analyzes both ICDs, understands the differences, and transforms
the code to conform to the target ICD.
"""
import json
import logging
import os
import re
import time
import uuid
import shutil
import zipfile
import io

import httpx

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
import fitz  # PyMuPDF
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="ICD C Code Refactorer", docs_url=None, redoc_url=None)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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
    "HF_MODEL", "Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _call_llm_stream(
    system_prompt: str, user_prompt: str, max_tokens: int = 16384,
    meta: dict | None = None,
):
    """Yield text chunks via SSE streaming from LLM.

    Uses direct llama-server to avoid proxy-side stalls.
    If *meta* dict is provided, ``meta["finish_reason"]`` is set to the
    finish_reason reported by the last SSE chunk (e.g. ``"stop"`` or
    ``"length"``).
    """
    url = f"{LLM_BASE_URL.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True,
        # Disable hidden reasoning stream so content tokens appear promptly.
        "reasoning_format": "none",
        "reasoning_in_content": True,
    }
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    yielded = False
    finish_reason_last: str | None = None
    last_err: Exception | None = None
    max_attempts = 4

    for attempt in range(1, max_attempts + 1):
        try:
            with httpx.Client(timeout=HTTPX_STREAM_TIMEOUT) as client:
                with client.stream("POST", url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
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
    if not yielded:
        raise RuntimeError("LLM returned empty response from direct llama-server")


def _wait_for_llm_ready(timeout_s: int = 120) -> None:
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
        "llama-server is not reachable. It may have failed to start (often GPU/CUDA init failure)."
    ) from last_err


def _extract_fenced(text: str, lang_hint: str = "") -> str:
    """Strip markdown fences if the LLM wraps output in ```."""
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.split("\n")
    out: list[str] = []
    in_block = False
    for line in lines:
        stripped = line.strip()
        if not in_block and stripped.startswith("```"):
            in_block = True
            continue
        if in_block and stripped == "```":
            break
        if in_block:
            out.append(line)
    return "\n".join(out) if out else text


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


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
    for f in files:
        if not f.filename or not (
            f.filename.endswith(".c") or f.filename.endswith(".h")
        ):
            continue
        dest = code_dir / f.filename
        content = await f.read()
        dest.write_bytes(content)
        uploaded.append(f.filename)

    status = json.loads((session_dir / "status.json").read_text())
    status["files"] = sorted(p.name for p in code_dir.iterdir() if p.is_file())
    (session_dir / "status.json").write_text(json.dumps(status))

    return {"uploaded": uploaded, "total_files": len(status["files"])}


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


@app.get("/api/status/{session_id}")
async def get_status(session_id: str):
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    return json.loads((session_dir / "status.json").read_text())


@app.get("/api/process/{session_id}")
async def process(session_id: str):
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

    code_dir = session_dir / "original_code"
    gen_dir = session_dir / "generated_code"
    source_icd = (session_dir / "source_icd.txt").read_text()
    target_icd = (session_dir / "target_icd.txt").read_text()
    code_files = sorted(p for p in code_dir.iterdir() if p.is_file())
    log.info("Process %s: %d code files, source_icd=%d chars, target_icd=%d chars",
             session_id[:8], len(code_files), len(source_icd), len(target_icd))

    def event_stream():
        yield _sse({
            "type": "info",
            "stage": "analysis",
            "message": "Checking llama-server availability...",
        })
        try:
            _wait_for_llm_ready(timeout_s=120)
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
        yield _sse({
            "type": "stage",
            "stage": "analysis",
            "message": "Analyzing ICD differences\u2026",
        })

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
            "List every single change with specific old and new values."
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

                base_compare_prompt = (
                    "Compare these two complete ICD documents and produce a COMPLETE "
                    "and EXHAUSTIVE code-impact change specification that will be used "
                    "to refactor C source code.\n\n"
                    f"## Source ICD (Full Text)\n{source_icd}\n\n"
                    f"## Target ICD (Full Text)\n{target_icd}\n\n"
                    "List EVERY difference between the two ICDs. For each change, state:\n"
                    "1. What it was in the Source ICD (old)\n"
                    "2. What it is in the Target ICD (new)\n"
                    "3. Impact on C code (structs, enums, functions, constants, etc.)"
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

                base_compare_prompt = (
                    "Produce a COMPLETE and EXHAUSTIVE code-impact change specification "
                    "comparing the Source ICD to the Target ICD.  This specification will "
                    "be used to refactor C code, so it must cover every single difference.\n\n"
                    f"## Source ICD full summary\n{source_summary}\n\n"
                    f"## Target ICD full summary\n{target_summary}\n\n"
                    "List EVERY difference between the two ICDs. For each change, state:\n"
                    "1. What it was in the Source ICD (old)\n"
                    "2. What it is in the Target ICD (new)\n"
                    "3. Impact on C code (structs, enums, functions, constants, etc.)"
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
                for piece in _call_llm_stream(
                    compare_system, prompt,
                    max_tokens=compare_max_tokens, meta=meta,
                ):
                    pass_parts.append(piece)
                    yield _sse({
                        "type": "token",
                        "stage": "analysis",
                        "token": piece,
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

            change_spec = complete_analysis.strip()
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
        (session_dir / "icd_analysis.txt").write_text(change_spec)
        (gen_dir / "icd_analysis.txt").write_text(change_spec)
        yield _sse({"type": "stage_complete", "stage": "analysis"})

        # Gather cross-file context
        all_code_ctx = ""
        for cf in code_files:
            all_code_ctx += f"\n### File: {cf.name}\n```c\n{cf.read_text()}\n```\n"
        all_code_ctx = _truncate_text(all_code_ctx, MAX_CODE_CONTEXT_CHARS, "all_code_ctx")

        # ---- Step 2: Transform each file ------------------------------
        for i, code_file in enumerate(code_files):
            fname = code_file.name
            yield _sse({
                "type": "stage",
                "stage": "transform",
                "file": fname,
                "index": i,
                "total": len(code_files),
                "message": f"Transforming {fname}\u2026",
            })

            original = code_file.read_text()

            transform_system = (
                "You are an expert C programmer specializing in embedded systems "
                "and interface implementations governed by Interface Control "
                "Documents.\n\n"
                "RULES:\n"
                "1. Output ONLY the complete, transformed C source code\n"
                "2. Preserve the overall code architecture, style, and conventions\n"
                "3. Apply ALL changes required by the target ICD per the change "
                "specification\n"
                "4. Update data structures, function signatures, constants, enums, "
                "macros\n"
                "5. Update comments/doc-strings to reflect the new ICD version\n"
                "6. Ensure type correctness and compilability\n"
                "7. Keep header/source consistency across the project\n"
                "8. Do NOT add prose explanations — only output C code\n"
                "9. Wrap the entire output in ```c ... ``` fences"
            )

            transform_prompt = (
                f"## Change Specification (Source ICD -> Target ICD)\n\n"
                f"{change_spec}\n\n"
                f"## Target ICD Consolidated Summary\n\n{target_summary}\n\n"
                f"## All Project Files (cross-file context)\n{all_code_ctx}\n\n"
                f"## File to Transform: {fname}\n\n```c\n{original}\n```\n\n"
                "Transform this file so it fully conforms to the Target ICD. "
                "Apply every relevant change from the change specification. "
                "Output the complete file — do not omit any sections. "
                "Do not summarize. Do not truncate. Include the full ending of the file."
            )
            clean = ""
            last_chunk = ""
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
                try:
                    for chunk in _call_llm_stream(
                        transform_system, attempt_prompt, max_tokens=4096
                    ):
                        file_parts.append(chunk)
                        yield _sse({
                            "type": "token",
                            "stage": "transform",
                            "file": fname,
                            "token": chunk,
                        })
                except Exception as e:
                    yield _sse({
                        "type": "error",
                        "message": f"Error transforming {fname}: {e}",
                        "file": fname,
                    })
                    break

                last_chunk = _extract_fenced("".join(file_parts), "c").strip()
                if attempt == 1:
                    clean = last_chunk
                else:
                    clean = (clean.rstrip() + "\n" + last_chunk.lstrip()).strip()

                if _looks_complete_c_file(clean, original, fname):
                    break

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

        # ---- Done -----------------------------------------------------
        status["state"] = "completed"
        status["generated_files"] = sorted(
            p.name for p in gen_dir.iterdir() if p.is_file()
        )
        (session_dir / "status.json").write_text(json.dumps(status))
        yield _sse({"type": "complete", "files": status["generated_files"]})

    return StreamingResponse(
        event_stream(),
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
    if not files:
        raise HTTPException(status_code=404, detail="No generated files found")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if f.name == "icd_analysis.txt":
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

    return {"filename": filename, "content": file_path.read_text()}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    session_dir = SESSIONS_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
    return {"deleted": True}
