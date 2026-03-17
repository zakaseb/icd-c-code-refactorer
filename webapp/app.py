"""
ICD-Based C Code Refactorer — Transform C code between ICD versions.

Upload original .c/.h files, source ICD (PDF), and target ICD (PDF).
The tool analyzes both ICDs, understands the differences, and transforms
the code to conform to the target ICD.
"""
import json
import os
import uuid
import shutil
import zipfile
import io

import httpx
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

HTTPX_STREAM_TIMEOUT = httpx.Timeout(timeout=None, connect=300.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    system_prompt: str, user_prompt: str, max_tokens: int = 16384
):
    """Yield text chunks via SSE streaming from LLM.

    Tries LiteLLM proxy first, then falls back to the direct llama-server.
    """
    for base_url, model in [
        (LLM_BASE_URL_LITELLM, MODEL_NAME_LITELLM),
        (LLM_BASE_URL, MODEL_NAME),
    ]:
        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        try:
            yielded = False
            with httpx.Client(timeout=HTTPX_STREAM_TIMEOUT) as client:
                with client.stream(
                    "POST", url, json=payload, headers=headers
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                delta = (
                                    data.get("choices", [{}])[0].get("delta", {})
                                )
                                part = delta.get("content", "")
                                if part:
                                    yielded = True
                                    yield part
                            except (json.JSONDecodeError, KeyError):
                                pass
            if yielded:
                return
        except Exception:
            if base_url == LLM_BASE_URL:
                raise
            continue
    raise RuntimeError("All LLM backends returned empty responses")


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

    def event_stream():
        # ---- Step 1: Analyse ICD delta --------------------------------
        yield _sse({
            "type": "stage",
            "stage": "analysis",
            "message": "Analyzing ICD differences\u2026",
        })

        analysis_system = (
            "You are an expert systems engineer and C programmer specializing in "
            "Interface Control Documents (ICDs) and embedded systems.\n"
            "Analyze the differences between the Source ICD and Target ICD.\n"
            "Focus on:\n"
            "- Data structure changes (added/removed/modified fields, types, sizes)\n"
            "- Message format changes (new/removed/modified messages)\n"
            "- Protocol and behavioral changes\n"
            "- Interface parameter changes (function signatures, callbacks)\n"
            "- Enumeration and constant value changes\n"
            "- Timing or sequencing requirement changes\n\n"
            "Produce a precise, actionable change specification that a C programmer "
            "can use to update source code."
        )
        analysis_prompt = (
            f"## Source ICD (Original Version)\n\n{source_icd}\n\n"
            f"## Target ICD (New Version)\n\n{target_icd}\n\n"
            "Produce a detailed change specification listing ALL differences "
            "between these two ICD versions that would affect C code. "
            "Be specific about struct fields, enum values, function signatures, "
            "message IDs, sizes, and any other concrete code-level changes."
        )

        analysis_parts: list[str] = []
        try:
            for chunk in _call_llm_stream(
                analysis_system, analysis_prompt, max_tokens=8192
            ):
                analysis_parts.append(chunk)
                yield _sse({
                    "type": "token",
                    "stage": "analysis",
                    "token": chunk,
                })
        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})
            return

        change_spec = "".join(analysis_parts)
        (session_dir / "change_spec.txt").write_text(change_spec)
        yield _sse({"type": "stage_complete", "stage": "analysis"})

        # Gather cross-file context
        all_code_ctx = ""
        for cf in code_files:
            all_code_ctx += f"\n### File: {cf.name}\n```c\n{cf.read_text()}\n```\n"

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
                f"## Target ICD Reference\n\n{target_icd}\n\n"
                f"## All Project Files (cross-file context)\n{all_code_ctx}\n\n"
                f"## File to Transform: {fname}\n\n```c\n{original}\n```\n\n"
                "Transform this file so it fully conforms to the Target ICD. "
                "Apply every relevant change from the change specification. "
                "Output the complete file — do not omit any sections."
            )

            file_parts: list[str] = []
            try:
                for chunk in _call_llm_stream(
                    transform_system, transform_prompt, max_tokens=32768
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
                })
                continue

            clean = _extract_fenced("".join(file_parts), "c")
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
            zf.write(f, f.name)
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
