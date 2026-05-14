"""Per-file compilation gate.

Runs between the *verification* and *sandbox_build* stages.  Every generated
``.c`` file is compiled to a ``.o`` object file using the **same** toolchain
the sandbox build will use (cross compiler when the repo's Makefile /
CMakeLists declare one, native ``gcc`` otherwise).  The artefacts land in
``gen_dir`` so they are picked up by ``/api/download/<session>``: the user
can grab the refactored ``.c`` / ``.h`` files plus the freshly compiled
``.o`` objects **before** the long-running sandbox build finishes.

If a ``.c`` fails to compile, an agentic fix loop iteratively asks the LLM
to repair the offending ``.c`` (and matching ``.h``, when relevant) using
the same prompting strategy as ``_sandbox_build_iterate``:

  - feed the ORIGINAL working code (when available) as a stable anchor,
  - feed the CURRENT generated code that failed,
  - feed the COMPILER errors,
  - feed the ICD change spec, repository knowledge and the file's
    dependency headers,
  - on stall (same error signature twice), reset the file to the
    pre-compile-gate version before re-attempting.

The full audit trail (per file, per attempt, with unified diffs) is written
to ``gen_dir / "compile_report.txt"``.

This module is intentionally self-contained.  All ``api/app.py`` helpers
are imported **lazily** inside the entry-point function so the module can
be imported during ``app.py`` module load without circular-import errors.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Generator, Optional

_HEADER_EXTS = {'.h', '.hpp', '.hh', '.hxx'}
_SOURCE_EXTS = {'.c', '.cpp', '.cc', '.cxx'}


def _sse(payload: dict) -> str:
    """SSE wire-format helper, duplicated locally to avoid circular import."""
    return f"data: {json.dumps(payload)}\n\n"


def _collect_include_dirs(*roots: Optional[Path]) -> list[Path]:
    """Every directory under *roots* that contains at least one header.

    The result is deduplicated and sorted by ``(depth_of_path, str_path)``
    so shallow / canonical include roots come first.  The order of *roots*
    is preserved: dirs discovered under an earlier root rank ahead of dirs
    discovered later (after the within-root depth sort).
    """
    seen: set[Path] = set()
    ordered: list[Path] = []
    for root in roots:
        if not root or not root.exists():
            continue
        per_root: list[Path] = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in _HEADER_EXTS:
                d = p.parent.resolve()
                if d not in seen:
                    seen.add(d)
                    per_root.append(d)
        per_root.sort(key=lambda d: (len(d.parts), str(d)))
        ordered.extend(per_root)
    return ordered


def _compile_command(cc: str, src: Path, obj: Path,
                     include_dirs: list[Path]) -> str:
    """Build the per-file compile shell command (single .c → .o)."""
    inc_args = " ".join(f'"-I{d}"' for d in include_dirs)
    return f'{cc} {inc_args} -c -o "{obj}" "{src}" 2>&1'


def _run_compile(cc: str, src: Path, obj: Path,
                 include_dirs: list[Path], timeout: int,
                 cwd: Path) -> tuple[bool, str, str]:
    """Run one compile.  Returns ``(success, command, combined_output)``.

    Removes any stale ``obj`` first so a missing ``.o`` after the run is an
    unambiguous failure signal.
    """
    try:
        if obj.exists():
            obj.unlink()
    except OSError:
        pass
    cmd = _compile_command(cc, src, obj, include_dirs)
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=str(cwd),
        )
        output = (r.stdout or "") + (r.stderr or "")
        ok = r.returncode == 0 and obj.exists()
        return ok, cmd, output.strip()
    except subprocess.TimeoutExpired:
        return False, cmd, f"Per-file compile timed out after {timeout}s."
    except Exception as e:
        return False, cmd, f"Per-file compile execution error: {e}"


_ERR_KEY_RE = re.compile(
    r"^([^:\n]+:\d+:\d+:\s*(?:error|fatal error):\s*[^\n]+)",
    re.MULTILINE,
)


def _error_signature(output: str) -> str:
    """Cheap stable fingerprint for stall detection (set of error lines)."""
    keys = sorted({m.group(1).strip() for m in _ERR_KEY_RE.finditer(output)})
    if keys:
        return "\n".join(keys)
    return "\n".join(sorted({l.strip() for l in output.splitlines() if l.strip()})[:8])


def _looks_compile_clean(output: str) -> bool:
    """True if no ``error:`` / ``fatal error:`` lines remain in the output."""
    return not bool(_ERR_KEY_RE.search(output or ""))


def run_per_file_compile(
    *,
    session_dir: Path,
    gen_dir: Path,
    code_dir: Path,
    repo_dir: Path,
    has_repo: bool,
    change_spec: str,
    repo_knowledge: str,
    is_resume: bool,
    completed_stages: set[str],
    max_fix_attempts: int = 4,
    compile_timeout: int = 60,
    cc_override: Optional[str] = None,
) -> Generator[str, None, None]:
    """Run the per-file compile gate, yielding SSE strings.

    Side effects on success: writes ``<base>.o`` files into ``gen_dir`` and
    writes ``gen_dir / "compile_report.txt"``.  On *partial* failure (some
    files still won't compile after the fix budget is spent), the report
    explicitly records the failures and the pipeline is allowed to proceed.

    Resume semantics: if ``is_resume`` and ``"compile"`` is already in
    ``completed_stages`` and the report file exists, the stage emits a
    short "reusing existing artefacts" sequence and returns.
    """
    # Lazy imports — avoid circular dependency with api/app.py.
    from app import (
        SANDBOX_CC_NATIVE,
        MAX_REPO_CONTEXT_CHARS,
        MAX_INPUT_TOKENS,
        MAX_OUTPUT_TOKENS,
        _detect_build_system,
        _detect_cross_compiler,
        _build_file_repo_context,
        _build_repo_context,
        _extract_fenced,
        _looks_complete_c_file,
        _structural_verify,
        _truncate_text,
        _estimate_tokens,
        _assemble_prompt,
        _generate_diff,
        _call_llm_complete,
        log,
    )

    report_path = gen_dir / "compile_report.txt"

    def _artefact_summary_event() -> dict:
        """Snapshot of `gen_dir` artefacts split for the UI.

        - `files` contains only items the UI can preview (.c / .h / .txt)
          so the existing showResults() can wire them up as tabs.
        - `objects` lists the freshly produced `.o` files for diagnostics;
          they are downloadable via the existing /api/download zip but
          should NOT be opened as text in the preview pane.
        """
        all_in_gen = [p for p in gen_dir.iterdir() if p.is_file()]
        previewable_exts = _HEADER_EXTS | _SOURCE_EXTS | {".txt"}
        return {
            "type": "compile_artifacts_ready",
            "stage": "compile",
            "files": sorted(
                p.name for p in all_in_gen
                if p.suffix.lower() in previewable_exts
            ),
            "objects": sorted(
                p.name for p in all_in_gen if p.suffix.lower() == ".o"
            ),
        }

    # ---- Resume short-circuit ----------------------------------------------
    if is_resume and "compile" in completed_stages and report_path.exists():
        yield _sse({
            "type": "stage",
            "stage": "compile",
            "message": "Resuming: reusing existing per-file compile artefacts...",
        })
        yield _sse({
            "type": "info",
            "stage": "compile",
            "message": "Loaded prior compile_report.txt and .o files.",
        })
        yield _sse(_artefact_summary_event())
        yield _sse({"type": "stage_complete", "stage": "compile"})
        return

    yield _sse({
        "type": "stage",
        "stage": "compile",
        "message": "Compiling generated .c files to .o object files…",
    })

    # ---- 1. Discover .c sources --------------------------------------------
    c_sources = sorted(
        p for p in gen_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _SOURCE_EXTS
    )
    if not c_sources:
        yield _sse({
            "type": "info",
            "stage": "compile",
            "message": "No generated .c files — skipping per-file compile gate.",
        })
        report_path.write_text(
            "PER-FILE COMPILE REPORT\n"
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
            "\nNo .c files were generated — compile stage skipped.\n"
        )
        yield _sse(_artefact_summary_event())
        yield _sse({"type": "stage_complete", "stage": "compile"})
        return

    # ---- 2. Detect toolchain (same logic as sandbox-build) ------------------
    if cc_override:
        sandbox_cc = cc_override
        build_info = {"type": "none", "path": None, "build_dir": repo_dir or gen_dir}
        is_cross = sandbox_cc.strip().split()[0] != SANDBOX_CC_NATIVE.split()[0]
    elif has_repo and repo_dir and repo_dir.exists():
        # Use repo build files (Makefile/CMakeLists.txt) for cross-compiler hints
        injected_locations: list[Path] = []
        for cs in c_sources:
            # If the .c exists in the repo, prefer that location for build-system detection.
            matches = list(repo_dir.rglob(cs.name))
            if matches:
                injected_locations.append(matches[0])
        build_info = _detect_build_system(repo_dir, injected_paths=injected_locations or None)
        sandbox_cc = _detect_cross_compiler(build_info)
        is_cross = sandbox_cc != SANDBOX_CC_NATIVE
    else:
        build_info = {"type": "none", "path": None, "build_dir": gen_dir}
        sandbox_cc = SANDBOX_CC_NATIVE
        is_cross = False

    cc_bin = sandbox_cc.split()[0]
    yield _sse({
        "type": "info",
        "stage": "compile",
        "message": (
            f"Toolchain: {sandbox_cc}"
            + (" (cross-compile)" if is_cross else " (native)")
        ),
    })

    # Pre-flight: confirm the compiler binary is on PATH.
    try:
        which = subprocess.run(
            ["which", cc_bin], capture_output=True, text=True, timeout=5,
        )
        cc_available = which.returncode == 0 and which.stdout.strip() != ""
    except Exception:
        cc_available = False

    if not cc_available:
        # The sandbox build will most likely also fail under this missing
        # toolchain — but we shouldn't block the pipeline. Skip cleanly and
        # surface the reason in the report.
        msg = (
            f"Compiler '{cc_bin}' not available on this host — "
            "skipping per-file compile gate. The sandbox build stage will "
            "still attempt the full build with the same toolchain."
        )
        yield _sse({"type": "info", "stage": "compile", "message": msg})
        report_path.write_text(
            "PER-FILE COMPILE REPORT\n"
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
            f"Toolchain detected: {sandbox_cc}\n"
            f"Cross-compile:      {'yes' if is_cross else 'no'}\n"
            f"\nStatus: SKIPPED — {msg}\n"
        )
        yield _sse(_artefact_summary_event())
        yield _sse({"type": "stage_complete", "stage": "compile"})
        return

    # ---- 3. Build include-dir list (gen_dir wins over repo) -----------------
    include_dirs = _collect_include_dirs(
        gen_dir,
        repo_dir if has_repo and repo_dir and repo_dir.exists() else None,
    )
    # Always include the gen_dir itself even if it has no .h yet (gen_dir
    # often hosts the canonical generated headers next to the .c sources).
    gd_resolved = gen_dir.resolve()
    if gd_resolved not in include_dirs:
        include_dirs.insert(0, gd_resolved)
    yield _sse({
        "type": "info",
        "stage": "compile",
        "message": f"Resolved {len(include_dirs)} include directories.",
    })

    # ---- 4. Snapshot the pre-compile-gate generated files (for stall reset)
    snapshots: dict[str, str] = {}
    for cs in c_sources:
        snapshots[cs.name] = cs.read_text()
    for hp in gen_dir.iterdir():
        if hp.is_file() and hp.suffix.lower() in _HEADER_EXTS:
            snapshots[hp.name] = hp.read_text()

    # ---- 5. Per-file compile + agentic fix loop ----------------------------
    uploaded_names = {p.name for p in code_dir.iterdir() if p.is_file()} if code_dir.exists() else set()
    per_file_results: dict[str, dict] = {}
    per_file_attempts: dict[str, list[dict]] = {}

    fix_system = (
        "You are an expert C programmer. The transformed C file failed "
        "per-file compilation (one .c → .o pass with the project's "
        "toolchain).\n\n"
        "You will receive:\n"
        "- The ORIGINAL working code (if available — compiled before ICD changes)\n"
        "- The CURRENT transformed code that failed to compile\n"
        "- The MATCHING header (.h) the .c depends on, if generated\n"
        "- The compiler errors\n"
        "- The ICD change specification\n"
        "- Repository dependency headers and codebase knowledge\n\n"
        "Your job: produce a corrected version that applies ALL ICD changes "
        "AND compiles cleanly with the per-file compile command shown.\n\n"
        f"TARGET COMPILER:\n- {sandbox_cc}\n"
        f"- Cross-compile: {'yes' if is_cross else 'no'}\n"
        "- C standard: C99 (-std=c99 compatible)\n"
        "- Use <stdint.h> fixed-width types\n\n"
        "RULES:\n"
        "1. Output ONLY the complete, corrected file\n"
        "2. Fix EVERY compiler error\n"
        "3. Preserve naming conventions and #include paths\n"
        "4. If both .c and .h need changes, output exactly two fenced blocks "
        "labelled with `### <filename>` headers, otherwise output one fenced block\n"
        "5. Wrap each file in ```c ... ``` fences"
    )

    for cs in c_sources:
        fname = cs.name
        base = cs.stem
        obj_path = gen_dir / f"{base}.o"
        original_code = ""
        orig_repo_match = (code_dir / fname)
        if orig_repo_match.exists():
            try:
                original_code = orig_repo_match.read_text()
            except OSError:
                original_code = ""
        elif has_repo and repo_dir and repo_dir.exists():
            for m in repo_dir.rglob(fname):
                if m.is_file():
                    try:
                        original_code = m.read_text()
                    except OSError:
                        original_code = ""
                    break

        # Companion .h (same stem) gets bundled into the LLM prompt if present.
        companion_h = gen_dir / f"{base}.h"
        if not companion_h.exists():
            companion_h = None

        attempts_log: list[dict] = []
        per_file_attempts[fname] = attempts_log

        prev_sig: Optional[str] = None
        stall = 0
        result: dict = {"status": "unknown"}

        yield _sse({
            "type": "info",
            "stage": "compile",
            "file": fname,
            "message": f"Compiling {fname}…",
        })

        for attempt in range(1, max_fix_attempts + 1):
            ok, cmd, output = _run_compile(
                sandbox_cc, cs, obj_path, include_dirs, compile_timeout,
                cwd=gen_dir,
            )
            attempts_log.append({
                "attempt": attempt,
                "command": cmd,
                "success": ok,
                "output": output,
                "diff": "",
            })
            sse_out = output
            if len(sse_out) > 4000:
                sse_out = sse_out[:4000] + "\n[... output truncated ...]"
            yield _sse({
                "type": "token",
                "stage": "compile",
                "file": fname,
                "token": (
                    f"\n--- {fname}: compile attempt {attempt} "
                    f"({'OK' if ok else 'FAIL'}) ---\n{sse_out}\n"
                ),
            })

            if ok:
                yield _sse({
                    "type": "compile_file_result",
                    "stage": "compile",
                    "file": fname,
                    "success": True,
                    "attempt": attempt,
                    "object": obj_path.name,
                    "message": f"{fname} compiled to {obj_path.name} on attempt {attempt}.",
                })
                result = {
                    "status": "ok",
                    "attempts": attempt,
                    "object": obj_path.name,
                }
                break

            # Stall detection across consecutive identical errors.
            sig = _error_signature(output)
            if sig and sig == prev_sig:
                stall += 1
            else:
                stall = 0
            prev_sig = sig

            if attempt == max_fix_attempts:
                yield _sse({
                    "type": "compile_file_result",
                    "stage": "compile",
                    "file": fname,
                    "success": False,
                    "attempt": attempt,
                    "message": (
                        f"{fname} still fails to compile after "
                        f"{max_fix_attempts} agentic fix attempts."
                    ),
                })
                result = {
                    "status": "failed",
                    "attempts": attempt,
                    "last_output": output[-2000:],
                }
                break

            # Stall reset: rewind to the pre-gate snapshot before re-prompting,
            # so the LLM doesn't compound its prior misguided edits.
            if stall >= 1:
                if fname in snapshots:
                    cs.write_text(snapshots[fname])
                if companion_h and companion_h.name in snapshots:
                    companion_h.write_text(snapshots[companion_h.name])
                stall = 0
                yield _sse({
                    "type": "info",
                    "stage": "compile",
                    "file": fname,
                    "message": (
                        f"Same compile errors persisted — reset {fname} "
                        f"(and matching .h) to the pre-compile-gate snapshot "
                        f"before re-prompting."
                    ),
                })

            # ---- Agentic fix prompt --------------------------------------
            pre_fix_c = cs.read_text()
            pre_fix_h = companion_h.read_text() if companion_h else ""

            file_repo_ctx = ""
            if has_repo and repo_dir and repo_dir.exists():
                file_repo_ctx = _build_file_repo_context(
                    repo_dir, pre_fix_c, uploaded_names,
                    max_chars=MAX_REPO_CONTEXT_CHARS,
                )
                if not file_repo_ctx:
                    file_repo_ctx = _build_repo_context(
                        repo_dir, exclude_names=uploaded_names,
                    )

            errors_trimmed = output if len(output) <= 4000 else (output[-4000:])

            sec_original = (
                f"## Original Working Code ({fname})\n"
                f"This code compiled cleanly before ICD changes:\n"
                f"```c\n{original_code}\n```"
                if original_code else ""
            )
            sec_current = (
                f"## Current Transformed Code ({fname})\n"
                f"This version failed the per-file compile:\n"
                f"```c\n{pre_fix_c}\n```"
            )
            sec_header = (
                f"## Matching Header ({companion_h.name})\n```c\n{pre_fix_h}\n```"
                if companion_h else ""
            )
            sec_cmd = (
                "## Per-file Compile Command\n```\n" + cmd + "\n```"
            )
            sec_errors = (
                f"## Compiler Errors\n```\n{errors_trimmed}\n```"
            )
            sec_repo_ctx = (
                f"## Repository Dependency Headers\n{file_repo_ctx}"
                if file_repo_ctx else ""
            )
            sec_knowledge = (
                f"## Repository Codebase Knowledge\n{repo_knowledge}"
                if repo_knowledge else ""
            )
            sec_change = f"## Change Specification\n{change_spec}"
            sec_instr = (
                "## Output format\n"
                "Output ONLY the complete file(s) inside ```c fences. "
                "When two files need to change, prefix each block with "
                "`### <filename>` on its own line so the names can be "
                "extracted reliably."
            )

            sys_tokens = _estimate_tokens(fix_system)
            fix_prompt = _assemble_prompt(
                [
                    ("original", sec_original, 0),
                    ("current", sec_current, 0),
                    ("header", sec_header, 0),
                    ("command", sec_cmd, 0),
                    ("errors", sec_errors, 0),
                    ("instructions", sec_instr, 0),
                    ("repo_ctx", sec_repo_ctx, 1),
                    ("knowledge", sec_knowledge, 2),
                    ("change_spec", sec_change, 3),
                ],
                max_input_tokens=MAX_INPUT_TOKENS - sys_tokens,
            )

            yield _sse({
                "type": "info",
                "stage": "compile",
                "file": fname,
                "message": (
                    f"LLM fix pass on {fname} (attempt {attempt} → "
                    f"{attempt + 1} of {max_fix_attempts})…"
                ),
            })

            try:
                fix_output = _call_llm_complete(
                    fix_system, fix_prompt,
                    max_tokens=MAX_OUTPUT_TOKENS, max_passes=3,
                )
            except Exception as e:
                log.warning("per_file_compile: LLM fix failed for %s: %s", fname, e)
                attempts_log[-1]["diff"] = f"(LLM fix raised: {e})"
                continue

            # Try to extract per-file blocks first (### <filename> headers),
            # fall back to a single fenced block (mapped to fname).
            new_files = _extract_per_file_blocks(fix_output)
            if not new_files:
                single = _extract_fenced(fix_output, "c").strip()
                if single:
                    new_files = {fname: single}

            applied: list[str] = []
            for target_name, new_code in new_files.items():
                target_path = gen_dir / target_name
                # Only touch files that we own / that the LLM was supposed to edit.
                if target_name not in (fname, companion_h.name if companion_h else None):
                    continue
                # Sanity-check completeness with the same heuristic the
                # transform step uses.
                ref = snapshots.get(target_name, "") or original_code
                if not _looks_complete_c_file(new_code, ref, target_name):
                    log.info("per_file_compile: %s LLM output looked incomplete",
                             target_name)
                    continue
                prev_text = target_path.read_text() if target_path.exists() else ""
                target_path.write_text(new_code)
                diff = _generate_diff(
                    prev_text, new_code,
                    f"{target_name} (attempt {attempt})",
                    f"{target_name} (attempt {attempt + 1})",
                )
                attempts_log[-1]["diff"] = (
                    attempts_log[-1].get("diff", "")
                    + (("\n\n" if attempts_log[-1].get("diff") else "") + diff)
                )
                applied.append(target_name)

            if not applied:
                yield _sse({
                    "type": "info",
                    "stage": "compile",
                    "file": fname,
                    "message": (
                        f"LLM fix attempt for {fname} did not yield a "
                        f"complete usable rewrite — keeping current file."
                    ),
                })
            else:
                yield _sse({
                    "type": "info",
                    "stage": "compile",
                    "file": fname,
                    "message": (
                        f"Applied LLM fix to: {', '.join(applied)}. "
                        f"Retrying compile…"
                    ),
                })

        per_file_results[fname] = result

        # Optional structural cross-check on the final accepted file.
        if has_repo and repo_dir and repo_dir.exists():
            try:
                issues = _structural_verify(
                    cs.read_text(), repo_dir, original_code, fname,
                )
                if issues:
                    log.info(
                        "per_file_compile: %s has %d post-fix structural notes",
                        fname, len(issues),
                    )
            except Exception:
                pass

    # ---- 6. Write the structured report ------------------------------------
    yield _sse({
        "type": "info",
        "stage": "compile",
        "message": "Writing compile_report.txt…",
    })

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    n_ok = sum(1 for r in per_file_results.values() if r.get("status") == "ok")
    n_fail = sum(1 for r in per_file_results.values() if r.get("status") == "failed")
    n_total = len(per_file_results)

    rep: list[str] = [
        "=" * 65,
        "PER-FILE COMPILE REPORT",
        "=" * 65,
        f"\nGenerated:          {now}",
        f"Compiler:           {sandbox_cc}",
        f"Cross-compile:      {'yes' if is_cross else 'no'}",
        f"Include dirs:       {len(include_dirs)}",
        f".c files processed: {n_total}",
        f"Compiled cleanly:   {n_ok}",
        f"Still failing:      {n_fail}",
        f"Max fix attempts:   {max_fix_attempts}",
        "",
        "-" * 65,
        "INCLUDE PATH (in order; gen_dir wins over repo)",
        "-" * 65,
    ]
    for d in include_dirs:
        rep.append(f"  -I{d}")

    for cs in c_sources:
        fname = cs.name
        result = per_file_results.get(fname, {})
        attempts = per_file_attempts.get(fname, [])
        rep.extend([
            "",
            "-" * 65,
            f"FILE: {fname}",
            "-" * 65,
            f"Status:   {result.get('status', 'unknown').upper()}",
            f"Attempts: {result.get('attempts', len(attempts))}",
        ])
        if result.get("status") == "ok":
            rep.append(f"Object:   {result.get('object')}")
        for a in attempts:
            rep.extend([
                "",
                f"  ## Attempt {a['attempt']} — {'OK' if a['success'] else 'FAIL'}",
                f"  Command: {a['command']}",
                "  Output:",
                "    " + a["output"].replace("\n", "\n    ")[:6000],
            ])
            if a.get("diff"):
                rep.append("  Diff applied after this attempt:")
                rep.append("    " + a["diff"].replace("\n", "\n    ")[:6000])

    report_path.write_text("\n".join(rep) + "\n")

    yield _sse(_artefact_summary_event())

    summary_msg = (
        f"Per-file compile: {n_ok}/{n_total} succeeded"
        + (f", {n_fail} still failing" if n_fail else "")
        + "."
    )
    yield _sse({
        "type": "compile_summary",
        "stage": "compile",
        "ok": n_ok,
        "failed": n_fail,
        "total": n_total,
        "message": summary_msg,
    })
    yield _sse({"type": "info", "stage": "compile", "message": summary_msg})
    yield _sse({"type": "stage_complete", "stage": "compile"})


_PER_FILE_HEADER_RE = re.compile(
    r"^###\s+([^\n]+?)\s*$",
    re.MULTILINE,
)


def _extract_per_file_blocks(text: str) -> dict[str, str]:
    """Parse multi-file LLM output of the form::

        ### filename.c
        ```c
        ...code...
        ```

        ### filename.h
        ```c
        ...code...
        ```

    Returns a ``{filename: code}`` dict.  Missing or unparseable input ⇒
    empty dict (caller should fall back to a single fenced block).
    """
    out: dict[str, str] = {}
    headers = list(_PER_FILE_HEADER_RE.finditer(text))
    if not headers:
        return out
    for i, m in enumerate(headers):
        name = m.group(1).strip().split()[0]
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        chunk = text[start:end]
        fence = re.search(
            r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\n(.*?)\n```",
            chunk, flags=re.DOTALL,
        )
        if fence:
            out[name] = fence.group(1).strip()
    return out
