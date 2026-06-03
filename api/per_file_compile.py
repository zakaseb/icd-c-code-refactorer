"""Per-file compilation gate.

Runs between the *verification* and *sandbox_build* stages.  Every generated
``.c`` file is compiled to a ``.o`` object file using the **same** toolchain
the sandbox build will use (cross compiler when the repo's Makefile /
CMakeLists declare one, native ``gcc`` otherwise).  The artefacts land in
``gen_dir`` so they are picked up by ``/api/download/<session>``: the user
can grab the refactored ``.c`` / ``.h`` files plus the freshly compiled
``.o`` objects **before** the long-running sandbox build finishes.

Scope: this gate compiles ONLY the newly generated + verified scripts in
``gen_dir``.  Header resolution is deliberately narrow:

  1. ``gen_dir`` — newly generated + verified headers (top priority).
  2. ``code_dir`` — the user's uploaded originals as a safe fallback.
  3. **Selectively resolved repo headers** — for each ``#include "..."``
     the generated code asks for that is NOT found under (1) / (2),
     we look it up case-insensitively in the broader repository ZIP
     and pull in *only* that one header's parent directory.  Any
     candidate sitting in a foreign cross-toolchain sysroot
     (``arm-xilinx-eabi``, ``*-eabi``, ``*-elf``, ``toolchain``,
     ``sysroot``, ``newlib``, or any directory that itself carries
     libc sentinels like ``string.h`` / ``stdio.h``) is filtered out
     so it cannot shadow the host compiler's libc.
     Case mismatches (e.g. ``#include "Timer.h"`` vs on-disk
     ``timer.h``) are normalised via a symlink/copy under
     ``session_dir/"compile_includes"`` so the include resolves with
     its requested case on case-sensitive filesystems.

The repo is never swept blindly.  Full-project header resolution with
the right sysroot is the responsibility of the sandbox build stage.

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

# Path-component markers that strongly suggest a foreign cross-toolchain
# sysroot (a directory whose `string.h` / `stdio.h` etc. are authored for
# a different compiler and therefore must NEVER end up on the host
# compiler's `-I` path, even transitively while resolving a project-local
# header request like `#include "Timer.h"`).
_TOXIC_PATH_SEGMENTS = (
    "arm-eabi", "arm-none-eabi", "arm-xilinx-eabi", "arm-linux-eabi",
    "aarch64-eabi", "aarch64-elf", "aarch64-none-elf",
    "riscv-elf", "riscv-none-elf", "riscv64-elf", "riscv32-eabi",
    "powerpc-eabi", "powerpc-elf",
    "mips-elf", "mips-eabi",
    "msp430-elf", "avr-eabi",
    "/toolchain/", "/sysroot/", "/newlib/",
    "/gcc-arm-", "/gcc-aarch64-", "/gcc-riscv-",
    "/include-fixed/",
)

# Sentinel libc headers — if any of these files exist directly inside a
# candidate directory, that directory is almost certainly a libc / sysroot
# headers dir for *some* toolchain, not a project-local include root.
_LIBC_SENTINELS = (
    "string.h", "stdio.h", "stddef.h", "stdint.h", "stdlib.h",
    "errno.h", "math.h", "limits.h", "ctype.h", "time.h",
)


def _sse(payload: dict) -> str:
    """SSE wire-format helper, duplicated locally to avoid circular import."""
    return f"data: {json.dumps(payload)}\n\n"


def _is_toxic_include_dir(dir_path: Path) -> bool:
    """True if *dir_path* looks like a foreign cross-toolchain sysroot.

    Adding such a directory to the compiler's `-I` path lets *both*
    quote- and angle-form ``#include`` directives resolve to libc /
    runtime headers authored for a different compiler.  Those headers
    typically rely on builtins (``wint_t``, ``size_t``, …) the host
    compiler does not provide and produce cascading "unknown type name"
    errors against perfectly fine generated code.  See the IMU.c
    failure pattern fixed in Test 11.

    Detection is intentionally conservative — we want zero false
    negatives on toolchain sysroots, accepting some false positives
    (the resolver will simply fall back to "header not found", which is
    a clear diagnostic the user / agentic LLM can act on).
    """
    s = str(dir_path).lower().replace("\\", "/")
    if any(seg in s for seg in _TOXIC_PATH_SEGMENTS):
        return True
    try:
        for sentinel in _LIBC_SENTINELS:
            if (dir_path / sentinel).is_file():
                return True
    except OSError:
        return False
    return False


def _index_repo_headers(repo_dir: Path) -> dict[str, list[Path]]:
    """Pre-walk *repo_dir* once and index header files by lowercase basename.

    Returned dict maps ``"timer.h" -> [Path(...), Path(...), ...]``.
    This lets the per-quote-include resolver answer
    "where in the repo is `Timer.h` (case-insensitive)?" in O(1) instead
    of re-walking the tree per lookup.
    """
    idx: dict[str, list[Path]] = {}
    if not repo_dir or not repo_dir.exists():
        return idx
    for cand in repo_dir.rglob("*"):
        try:
            if not cand.is_file():
                continue
            if cand.suffix.lower() not in _HEADER_EXTS:
                continue
        except OSError:
            continue
        idx.setdefault(cand.name.lower(), []).append(cand)
    return idx


def _resolve_quoted_includes_in_repo(
    *,
    needs: set[str],
    primary_dirs: list[Path],
    repo_index: dict[str, list[Path]],
    shim_dir: Path,
) -> tuple[list[Path], dict[str, dict]]:
    """Demand-driven, case-insensitive, toolchain-safe resolution of
    project-local ``#include "..."`` directives against *repo_index*.

    For each header in *needs* that is not already resolvable in
    *primary_dirs* (``gen_dir`` / ``code_dir`` — case-insensitive
    basename check), we look it up in *repo_index*:

      - any candidate whose parent directory looks like a foreign
        cross-toolchain sysroot is filtered out (see
        :func:`_is_toxic_include_dir`);
      - the remaining candidates are sorted shallow-first under
        ``repo_dir`` (canonical project copies tend to sit higher in
        the tree than vendored / archived copies);
      - the winner's parent directory is added to the include list
        when the on-disk filename matches the requested case **and**
        the parent dir does not itself look libc-like;
      - otherwise we symlink (or copy as a fallback) the file under
        the *requested* case into *shim_dir* and add *shim_dir* to
        the include list — this makes ``#include "Timer.h"`` resolve
        even when the actual file on disk is ``timer.h``.

    Returns ``(extra_include_dirs, resolutions)`` where *resolutions*
    is a per-header diagnostic dict the caller can surface in SSE
    messages and the compile report.
    """
    extra: list[Path] = []
    seen_extra: set[Path] = set()
    resolutions: dict[str, dict] = {}

    def _already_resolvable(rel: str) -> bool:
        """Is *rel* findable case-insensitively under any primary dir?"""
        parts = rel.replace("\\", "/").split("/")
        leaf_lower = parts[-1].lower()
        for d in primary_dirs:
            if not d or not d.exists():
                continue
            exact = d.joinpath(*parts)
            try:
                if exact.is_file():
                    return True
            except OSError:
                pass
            parent_dir = d.joinpath(*parts[:-1]) if len(parts) > 1 else d
            try:
                if parent_dir.is_dir():
                    for p in parent_dir.iterdir():
                        if p.is_file() and p.name.lower() == leaf_lower:
                            return True
            except OSError:
                continue
        return False

    for need in sorted(needs):
        rel = need.replace("\\", "/")
        parts = rel.split("/")
        leaf = parts[-1]
        leaf_lower = leaf.lower()

        if _already_resolvable(rel):
            continue

        bucket = repo_index.get(leaf_lower, []) if repo_index else []
        if not bucket:
            resolutions[need] = {"status": "missing"}
            continue

        legit: list[Path] = []
        toxic: list[Path] = []
        for cand in bucket:
            if len(parts) > 1:
                # Multi-segment includes: ensure the tail of *cand* matches
                # the requested relative path (case-insensitive).
                cand_parts = [p.lower() for p in cand.parts]
                need_parts = [p.lower() for p in parts]
                if cand_parts[-len(need_parts):] != need_parts:
                    continue
            if _is_toxic_include_dir(cand.parent):
                toxic.append(cand)
                continue
            legit.append(cand)

        if not legit:
            resolutions[need] = {
                "status": "skipped_toxic_only",
                "skipped_toxic": [str(p) for p in toxic][:8],
            }
            continue

        legit.sort(key=lambda p: (len(p.parts), str(p)))
        chosen = legit[0]

        # Compute the include root: the directory `cand` would be reached
        # from given the relative include path. For `#include "Timer.h"`
        # that's `chosen.parent`. For `#include "sub/Timer.h"` it's the
        # grandparent (one extra ".." for each path segment).
        root = chosen
        for _ in range(len(parts)):
            root = root.parent

        case_matches = (chosen.name == leaf)
        # Even if the on-disk case matches, fall back to the shim path
        # if the parent directory itself looks libc-like — this is a
        # belt-and-braces guard against a project header that happens
        # to live next to a `string.h`. (Should be vanishingly rare in
        # legit code, but we already filter aggressively above.)
        parent_safe = case_matches and not _is_toxic_include_dir(root)

        if parent_safe:
            r = root.resolve()
            if r not in seen_extra:
                seen_extra.add(r)
                extra.append(r)
            resolutions[need] = {
                "status": "found",
                "path": str(chosen),
                "include_root": str(r),
                "skipped_toxic": [str(p) for p in toxic][:8],
            }
            continue

        # Case mismatch (or parent unsafe): materialize a shim with the
        # requested case under *shim_dir* and add the shim dir to -I.
        try:
            shim_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            resolutions[need] = {
                "status": "shim_create_failed",
                "path": str(chosen),
                "skipped_toxic": [str(p) for p in toxic][:8],
            }
            continue

        shim_path = shim_dir.joinpath(*parts)
        try:
            shim_path.parent.mkdir(parents=True, exist_ok=True)
            if shim_path.is_symlink() or shim_path.exists():
                shim_path.unlink()
            try:
                shim_path.symlink_to(chosen.resolve())
            except OSError:
                # Filesystem without symlink support (e.g. some FAT/exFAT
                # mounts) — fall back to a plain copy.
                shim_path.write_bytes(chosen.read_bytes())
        except OSError as e:
            resolutions[need] = {
                "status": "shim_write_failed",
                "path": str(chosen),
                "error": str(e),
                "skipped_toxic": [str(p) for p in toxic][:8],
            }
            continue

        s = shim_dir.resolve()
        if s not in seen_extra:
            seen_extra.add(s)
            extra.append(s)
        resolutions[need] = {
            "status": "found_case_normalized" if not case_matches else "found_via_shim",
            "path": str(chosen),
            "shim": str(shim_path),
            "skipped_toxic": [str(p) for p in toxic][:8],
        }

    return extra, resolutions


def _quoted_includes_in_dir(directory: Path) -> set[str]:
    """All ``#include "..."`` directives found in any .c/.h under *directory*.

    Uses the same regex as ``app._extract_includes`` (kept local to keep
    this module free of app-level imports).
    """
    out: set[str] = set()
    if not directory.exists():
        return out
    for fp in directory.iterdir():
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in (_HEADER_EXTS | _SOURCE_EXTS):
            continue
        try:
            text = fp.read_text(errors="replace")
        except OSError:
            continue
        for m in _QUOTED_INCLUDE_RE.finditer(text):
            out.add(m.group(1).strip())
    return out


_QUOTED_INCLUDE_RE = re.compile(
    r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE,
)


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


# Patterns gcc emits that we can convert into a clear, structured punch
# list for the agentic LLM. Each pattern captures the identifier(s)
# the LLM needs to either add, declare, or otherwise resolve.
#
# IMPORTANT: gcc emits diagnostics with Unicode "smart" quotes by
# default (U+2018 `‘`, U+2019 `’`) — only ``-fno-diagnostics-color
# -fdiagnostics-format=plain`` or ``LC_ALL=C`` falls back to ASCII.
# Real user runs always show ``‘foo’``, so every quote-based regex
# below accepts BOTH ASCII single quotes and the smart-quote pair.
# Without this, the structured punch list silently extracted ZERO
# missing-member errors from real compile logs and the focus banner
# never fired (the IMU.c production failure spent four attempts on
# cosmetic .c edits because the LLM was never told what to fix).
_Q_OPEN  = r"['\u2018\u201A\u2039\u00AB`]"
_Q_CLOSE = r"['\u2019\u201B\u203A\u00BB`]"
_QUOTED  = rf"{_Q_OPEN}([^'\u2018\u2019\u201A\u201B\u2039\u203A\u00AB\u00BB`]+){_Q_CLOSE}"

_MISSING_MEMBER_RE = re.compile(
    rf"{_QUOTED}\s+has no member named\s+{_QUOTED}",
)
_UNDECLARED_RE = re.compile(
    rf"{_QUOTED}\s+undeclared(?:\s+\(first use in this function\))?",
)
_UNKNOWN_TYPE_RE = re.compile(
    rf"unknown type name\s+{_QUOTED}",
)
_IMPLICIT_DECL_RE = re.compile(
    rf"implicit declaration of function\s+{_QUOTED}",
)
_CONFLICTING_TYPES_RE = re.compile(
    rf"conflicting types for\s+{_QUOTED}",
)
_INCOMPATIBLE_PTR_RE = re.compile(
    rf"incompatible (?:pointer )?type[^'\u2018\u2019]*{_QUOTED}"
    rf"\s+(?:to|from)\s+{_QUOTED}",
)
_NO_FILE_RE = re.compile(
    r"fatal error:\s+([^:\n]+):\s+No such file or directory",
)


def _structured_compile_errors(output: str) -> dict[str, list]:
    """Parse a gcc/clang output into a structured punch list of common
    error categories. Returned dict keys (each a list of unique tuples):

      ``missing_members``       : list[(struct_name, member_name)]
      ``undeclared``            : list[symbol]
      ``unknown_types``         : list[type_name]
      ``implicit_decls``        : list[function_name]
      ``conflicting_types``     : list[symbol]
      ``incompatible_pointers`` : list[(lhs_type, rhs_type)]
      ``missing_headers``       : list[header_name]

    Empty lists are kept so callers can render a consistent layout.
    """
    def _uniq(seq: list) -> list:
        seen: set = set()
        out: list = []
        for item in seq:
            key = item if isinstance(item, str) else tuple(item)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    text = output or ""
    return {
        "missing_members": _uniq(
            [(m.group(1), m.group(2)) for m in _MISSING_MEMBER_RE.finditer(text)]
        ),
        "undeclared": _uniq(
            [m.group(1) for m in _UNDECLARED_RE.finditer(text)]
        ),
        "unknown_types": _uniq(
            [m.group(1) for m in _UNKNOWN_TYPE_RE.finditer(text)]
        ),
        "implicit_decls": _uniq(
            [m.group(1) for m in _IMPLICIT_DECL_RE.finditer(text)]
        ),
        "conflicting_types": _uniq(
            [m.group(1) for m in _CONFLICTING_TYPES_RE.finditer(text)]
        ),
        "incompatible_pointers": _uniq(
            [(m.group(1), m.group(2)) for m in _INCOMPATIBLE_PTR_RE.finditer(text)]
        ),
        "missing_headers": _uniq(
            [m.group(1).strip() for m in _NO_FILE_RE.finditer(text)]
        ),
    }


def _format_error_punchlist(struct: dict[str, list]) -> str:
    """Render the output of :func:`_structured_compile_errors` as a
    Markdown punch list suitable for inclusion in the agentic-fix LLM
    prompt. Returns an empty string when nothing actionable was found
    (i.e. callers can do ``if punchlist: prompt += punchlist``).
    """
    lines: list[str] = []
    if struct["missing_members"]:
        lines.append(
            "Missing struct/union members — the FIX is to ADD each member "
            "to the named struct in its declaring header (e.g. `### IMU.h`). "
            "DO NOT silently remove the member's usage from the .c — that "
            "would undo an ICD-mandated change:"
        )
        for s, m in struct["missing_members"]:
            lines.append(f"  - struct `{s}` is missing member `{m}`")
    if struct["unknown_types"]:
        lines.append(
            "Unknown type names — either typedef them in the appropriate "
            "header or add the right `#include`:"
        )
        for t in struct["unknown_types"]:
            lines.append(f"  - `{t}`")
    if struct["undeclared"]:
        lines.append(
            "Undeclared identifiers — declare them or add the right "
            "`#include` (these may also be the result of a missing macro "
            "or a typo to fix in the .c):"
        )
        for s in struct["undeclared"]:
            lines.append(f"  - `{s}`")
    if struct["implicit_decls"]:
        lines.append(
            "Functions used without a declaration — add a prototype to "
            "the matching header (or include the header that already "
            "provides it):"
        )
        for s in struct["implicit_decls"]:
            lines.append(f"  - `{s}()`")
    if struct["conflicting_types"]:
        lines.append(
            "Conflicting types — the symbol is declared with two "
            "different signatures. Make the declaration in the header "
            "match the new signature mandated by the ICD change:"
        )
        for s in struct["conflicting_types"]:
            lines.append(f"  - `{s}`")
    if struct["incompatible_pointers"]:
        lines.append(
            "Incompatible pointer assignments — adjust the type of "
            "either the variable or the pointed-to value so they match:"
        )
        for a, b in struct["incompatible_pointers"]:
            lines.append(f"  - `{a}` vs `{b}`")
    if struct["missing_headers"]:
        lines.append(
            "Headers that could not be resolved (drop the include OR "
            "rename it to a header that actually exists in the project):"
        )
        for h in struct["missing_headers"]:
            lines.append(f"  - `{h}`")
    if not lines:
        return ""
    return "## Structured Error Punch List\n" + "\n".join(lines)


def run_per_file_compile(
    *,
    session_dir: Path,
    gen_dir: Path,
    code_dir: Path,
    repo_dir: Path,
    has_repo: bool,
    change_spec: str,
    repo_knowledge: str,
    gitnexus_report: str = "",
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

    # ---- 3. Build include-dir list ----------------------------------------
    # SCOPE: the per-file compile gate is a fast smoke test that ONLY the
    # newly generated and verified scripts (.c/.h in gen_dir) compile to
    # .o objects with the project's toolchain. We restrict the `-I`
    # search path to a deliberately narrow set:
    #   1. gen_dir       — newly generated + verified headers (highest priority)
    #   2. code_dir      — user's uploaded originals as a safe fallback for
    #                       any header the pipeline did not regenerate
    #   3. selectively-resolved repo headers — only those project-local
    #                       quote-includes the generated code actually asks
    #                       for, picked case-insensitively and ONLY from
    #                       non-toolchain locations (no `arm-xilinx-eabi`,
    #                       `*-eabi`, `*-elf`, `toolchain`, `sysroot`,
    #                       `newlib`, or any dir that itself contains
    #                       libc sentinels like `string.h`).
    # We never sweep the broader repo blindly. See Test 11 for the
    # production failure mode (foreign `string.h` shadow) this guards
    # against, and Test 13/14 for the project-local resolution paths
    # this enables (Timer.h, case mismatch via shim).
    include_dirs = _collect_include_dirs(
        gen_dir,
        code_dir if code_dir and code_dir.exists() else None,
    )
    # Always include the gen_dir itself even if it has no .h yet (gen_dir
    # often hosts the canonical generated headers next to the .c sources).
    gd_resolved = gen_dir.resolve()
    if gd_resolved not in include_dirs:
        include_dirs.insert(0, gd_resolved)

    # ---- 3b. Selective project-local resolution from repo_dir -------------
    quote_needs = _quoted_includes_in_dir(gen_dir)
    resolutions: dict[str, dict] = {}
    extra_dirs: list[Path] = []
    shim_dir = session_dir / "compile_includes"
    if has_repo and repo_dir and repo_dir.exists() and quote_needs:
        repo_index = _index_repo_headers(repo_dir)
        primary_dirs_for_resolve: list[Path] = [
            gd_resolved,
        ]
        if code_dir and code_dir.exists():
            primary_dirs_for_resolve.append(code_dir.resolve())
        extra_dirs, resolutions = _resolve_quoted_includes_in_repo(
            needs=quote_needs,
            primary_dirs=primary_dirs_for_resolve,
            repo_index=repo_index,
            shim_dir=shim_dir,
        )
        for d in extra_dirs:
            if d not in include_dirs:
                include_dirs.append(d)

    n_found = sum(
        1 for r in resolutions.values()
        if r.get("status") in ("found", "found_via_shim", "found_case_normalized")
    )
    n_missing = sum(
        1 for r in resolutions.values()
        if r.get("status") in ("missing", "skipped_toxic_only",
                                "shim_create_failed", "shim_write_failed")
    )

    scope_msg = (
        f"Resolved {len(include_dirs)} include directories "
        f"(scoped to newly generated + verified scripts and "
        f"demand-driven project-local headers — "
        f"repository ZIP is intentionally NOT swept; only the specific "
        f"headers the generated code quote-includes are pulled in, and "
        f"only from non-toolchain locations)."
    )
    yield _sse({"type": "info", "stage": "compile", "message": scope_msg})

    if resolutions:
        # Report resolutions in a stable, human-readable form.
        found_lines = []
        norm_lines = []
        missing_lines = []
        for need in sorted(resolutions):
            st = resolutions[need].get("status", "?")
            if st == "found":
                found_lines.append(
                    f"  - {need} -> {resolutions[need].get('include_root', '?')}"
                )
            elif st in ("found_case_normalized", "found_via_shim"):
                norm_lines.append(
                    f"  - {need} -> shim "
                    f"({resolutions[need].get('path', '?')}; "
                    f"requested case differs from on-disk case)"
                )
            else:
                missing_lines.append(f"  - {need} ({st})")
        bits = []
        if found_lines:
            bits.append(
                f"Resolved {len(found_lines)} project-local header(s) "
                f"from repo via case-exact match:"
            )
            bits.extend(found_lines)
        if norm_lines:
            bits.append(
                f"Resolved {len(norm_lines)} project-local header(s) "
                f"via case-normalizing shim:"
            )
            bits.extend(norm_lines)
        if missing_lines:
            bits.append(
                f"COULD NOT resolve {len(missing_lines)} project-local "
                f"header(s) — generated code will fail to compile unless "
                f"the agentic fix loop removes or replaces these:"
            )
            bits.extend(missing_lines)
        yield _sse({
            "type": "info", "stage": "compile",
            "message": "\n".join(bits),
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
        "Your PRIMARY job is to make this single .c file (and its "
        "matching .h, if both need edits) COMPILE CLEANLY with the "
        "per-file compile command shown. The ICD change specification "
        "is REFERENCE CONTEXT for understanding intent, NOT a checklist "
        "to apply in this attempt — every ICD change visible in the "
        "current transformed code must be PRESERVED, but do NOT spend "
        "this attempt polishing comments, dates, magic numbers, or "
        "other cosmetic ICD wording. The user can see the green compile "
        "they need; cosmetic polish can come later. Every single "
        "compiler error in the output below MUST be addressed in your "
        "rewrite.\n\n"
        f"TARGET COMPILER:\n- {sandbox_cc}\n"
        f"- Cross-compile: {'yes' if is_cross else 'no'}\n"
        "- C standard: C99 with GCC extensions (-std=gnu99) — M_PI, "
        "unnamed structs/unions, `__attribute__`, and statement "
        "expressions are ACCEPTED here\n"
        "- Use <stdint.h> fixed-width types\n\n"
        "OUTPUT FORMAT (MUST follow exactly — the parser is strict):\n"
        "1. Output ONLY the COMPLETE final file(s). NEVER patches, diffs, "
        "ellipses, '...', '// unchanged', or partial code.\n"
        "2. If exactly one file needs editing, output exactly one fenced\n"
        "   block wrapped in ```c ... ``` (no header line needed).\n"
        "3. If BOTH the .c and the .h need editing, output TWO fenced "
        "blocks. Each block MUST be immediately preceded by a single-line "
        "filename header of the form:\n"
        "       ### <filename>\n"
        "   e.g. `### IMU.c` then ```c ... ``` then `### IMU.h` then "
        "```c ... ```. The filename MUST end in `.c` or `.h` and match "
        "the file you are rewriting.\n"
        "4. NEVER add any other `###` (or `##` / `####`) section headers "
        "anywhere in your reply — no `### Analysis`, `### Reasoning`, "
        "`### Summary`, `### Fix`, etc. They will be interpreted as "
        "filename markers and silently break the parser.\n"
        "5. NO prose, NO numbered lists, NO commentary outside the "
        "fenced code blocks. The only allowed content outside fences is "
        "the `### <filename>` markers themselves.\n"
        "\n"
        "FIX RULES:\n"
        "A. Fix EVERY compiler error visible in the build output.\n"
        "B. Preserve all naming conventions and `#include` paths that "
        "already work — only add or rename what the errors demand.\n"
        "C. NEVER silently delete or roll back ICD-mandated changes "
        "from the .c to make the compile pass. If the error says a "
        "struct/union member is missing, ADD that member to the struct "
        "in its declaring header. If the error says a function is "
        "undeclared, ADD a prototype to the header (or include the "
        "header that already declares it).\n"
        "D. When the error says `'X' has no member named 'Y'`, the "
        "correct fix is almost always to output BOTH the `### <name>.c` "
        "and `### <name>.h` blocks — the .c unchanged (or only the new "
        "uses preserved) and the .h with the missing fields added to "
        "struct X.\n"
        "E. The .h must keep its include guards (`#ifndef <BASE>_H` / "
        "`#define <BASE>_H` ... `#endif`). The .c must keep its "
        "`#include \"<base>.h\"` (or equivalent) if it had one.\n"
        "F. If the file declares MULTIPLE per-variation structs for the "
        "same peripheral (one per ICD variation, each with its own "
        "explanatory comment banner), PRESERVE all of them — do NOT "
        "merge, dedupe, or delete variations to make the compile pass. "
        "Unused variation structs are intentional and are NOT errors."
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
        # Tracks whether the previous attempt's punch list had
        # missing-member / undeclared / unknown-type errors but the
        # LLM's response did NOT touch the .h. Set true when the LLM
        # ignored an obvious header-side fix; surfaces a forceful
        # banner at the TOP of the next prompt.
        prev_missed_h_side: bool = False
        prev_missed_details: list[str] = []

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

            # Pre-parse the compile output into a structured "missing
            # symbols / type mismatches" punch list so the LLM has a
            # short, unambiguous action list instead of having to
            # re-derive it from raw gcc output.
            err_struct = _structured_compile_errors(output)
            sec_punchlist = _format_error_punchlist(err_struct)

            # If the PREVIOUS attempt's punch list contained
            # missing-member / undeclared / unknown-type errors that
            # should have triggered a .h-side fix and the LLM emitted
            # zero `.h` blocks, hoist a very visible banner to the
            # TOP of this attempt's prompt. The IMU.c production
            # failure was the LLM emitting four .c-only "fixes" in a
            # row (date format, magic numbers, cosmetic ICD polish)
            # while the .h that owned ``sIMU_InertialData`` was
            # never opened.
            sec_focus_banner = ""
            if prev_missed_h_side:
                sec_focus_banner = (
                    "## CRITICAL — READ FIRST\n"
                    "Your PREVIOUS rewrite attempt only touched the .c "
                    "file. The compiler errors below STILL list "
                    "header-side problems that REQUIRE editing the .h, "
                    "not the .c.\n\n"
                    "Symptoms the previous attempt failed to address:\n"
                    + "\n".join(f"  - {d}" for d in prev_missed_details)
                    + "\n\n"
                    "On this attempt you MUST output BOTH:\n"
                    "  ### "
                    + fname
                    + "\n  ```c\n  ...the FULL .c...\n  ```\n\n"
                    "  ### "
                    + (companion_h.name if companion_h else fname.replace(".c", ".h"))
                    + "\n  ```c\n  ...the FULL .h with the missing "
                    "declarations / typedefs / struct members ADDED...\n"
                    "  ```\n\n"
                    "Stop polishing dates, comments, prose, or "
                    "ICD-cosmetic magic numbers — those are zero-priority "
                    "until the compile is green.\n"
                )

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
            sec_gitnexus = (
                f"## GitNexus Codebase Understanding\n{gitnexus_report}"
                if gitnexus_report else ""
            )
            sec_change = f"## Change Specification\n{change_spec}"
            sec_instr = (
                "## Output format reminder\n"
                "Output ONLY the complete file(s) inside ```c fences. "
                "When TWO files need to change, prefix each block with "
                "`### <filename>` (e.g. `### "
                f"{fname}` then ```c ... ``` then `### "
                f"{companion_h.name if companion_h else fname.replace('.c', '.h')}`"
                " then ```c ... ```). The filename MUST end in `.c` or "
                "`.h`. Do NOT add any other `###` headings anywhere "
                "in your reply (`### Analysis`, `### Fix`, `### Notes`, "
                "etc. will be parsed as filenames and silently dropped)."
            )

            # Surface the include-resolution audit so the LLM can react to
            # quote-includes that genuinely don't exist anywhere in the
            # repo (it can drop them, rename them, or stub them out).
            sec_resolutions = ""
            if resolutions:
                resolution_lines: list[str] = []
                miss = [n for n, r in resolutions.items()
                        if r.get("status") in ("missing", "skipped_toxic_only",
                                                "shim_create_failed",
                                                "shim_write_failed")]
                norm = [n for n, r in resolutions.items()
                        if r.get("status") in ("found_case_normalized",
                                                "found_via_shim")]
                if miss:
                    resolution_lines.append(
                        "Could NOT resolve these quote-includes from the "
                        "uploaded code or the repository (they don't exist "
                        "as project-local headers anywhere safe). If your "
                        "fix still depends on them you must remove, "
                        "rename, or stub them out:"
                    )
                    for n in sorted(miss):
                        st = resolutions[n].get("status")
                        resolution_lines.append(f"  - {n}  ({st})")
                if norm:
                    resolution_lines.append(
                        "These quote-includes were resolved via a "
                        "case-normalizing shim — keep using the EXACT "
                        "case the original code used:"
                    )
                    for n in sorted(norm):
                        resolution_lines.append(
                            f"  - {n}  (on disk: "
                            f"{Path(resolutions[n].get('path', '?')).name})"
                        )
                if resolution_lines:
                    sec_resolutions = (
                        "## Quote-Include Resolution Audit\n"
                        + "\n".join(resolution_lines)
                    )

            sys_tokens = _estimate_tokens(fix_system)
            fix_prompt = _assemble_prompt(
                [
                    # The focus banner must always reach the LLM if it
                    # exists; it's tiny and disposable if there's
                    # nothing to say.
                    ("focus", sec_focus_banner, 0),
                    ("original", sec_original, 0),
                    ("current", sec_current, 0),
                    ("header", sec_header, 0),
                    ("command", sec_cmd, 0),
                    ("errors", sec_errors, 0),
                    # The structured punch list is a tiny, high-signal
                    # summary of the raw error log — keep it at top
                    # priority so it always fits in the prompt budget.
                    ("punchlist", sec_punchlist, 0),
                    ("resolutions", sec_resolutions, 0),
                    ("instructions", sec_instr, 0),
                    ("repo_ctx", sec_repo_ctx, 1),
                    ("gitnexus", sec_gitnexus, 2),
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
                attempts_log[-1]["llm_diagnostics"] = {
                    "error": f"LLM call raised: {e}",
                }
                yield _sse({
                    "type": "info",
                    "stage": "compile",
                    "file": fname,
                    "message": (
                        f"LLM fix call for {fname} failed ({type(e).__name__})"
                        f" — keeping current file."
                    ),
                })
                continue

            # ---- Parse + fall-back strategy ----
            # (1) Try strict `### <filename>.<ext>` parsing.
            # (2) If that yields no usable mapping, look at every fenced
            #     block in the output and assign each by content sniff
            #     (.h via include-guard pattern, .c via function-body
            #     pattern). This rescues the very common LLM failure
            #     mode of forgetting / mangling filename headers.
            # (3) Final fallback: single fenced block -> fname.
            allowed = {fname}
            if companion_h is not None:
                allowed.add(companion_h.name)

            parsed = _extract_per_file_blocks(fix_output)
            parsed_filtered = {k: v for k, v in parsed.items() if k in allowed}

            new_files: dict[str, str] = dict(parsed_filtered)

            if not new_files:
                # Fallback A: split every fenced block by content sniff
                fenced = _all_fenced_blocks(fix_output)
                if len(fenced) >= 2:
                    for body in fenced:
                        guess = _guess_filename_for_block(
                            body, c_name=fname,
                            h_name=(companion_h.name if companion_h else None),
                        )
                        if guess and guess not in new_files:
                            new_files[guess] = body
                # Fallback B: single fenced block -> fname
                if not new_files:
                    single = _strip_filename_marker_leakage(
                        _extract_fenced(fix_output, "c").strip()
                    )
                    if single:
                        new_files = {fname: single}

            # Defence in depth: re-strip every block right before the
            # decision loop. Even if a future code path introduces a
            # new fallback that forgets to call the stripper, a leaked
            # `### IMU.c` line can NEVER reach the on-disk file.
            new_files = {
                k: _strip_filename_marker_leakage(v)
                for k, v in new_files.items()
                if _strip_filename_marker_leakage(v)
            }

            applied: list[str] = []
            decisions: list[dict] = []  # per-target audit trail
            for target_name, new_code in new_files.items():
                # Only touch files that we own / that the LLM was supposed to edit.
                if target_name not in allowed:
                    decisions.append({
                        "target": target_name,
                        "decision": "skipped_unknown_name",
                        "reason": (
                            f"name not in {{{', '.join(sorted(allowed))}}}"
                        ),
                    })
                    continue
                # Sanity-check completeness with the same heuristic the
                # transform step uses.
                ref = snapshots.get(target_name, "") or original_code
                if not _looks_complete_c_file(new_code, ref, target_name):
                    detail = _incomplete_file_reason(new_code, ref, target_name)
                    log.info(
                        "per_file_compile: %s LLM output rejected: %s",
                        target_name, detail,
                    )
                    decisions.append({
                        "target": target_name,
                        "decision": "rejected_incomplete",
                        "reason": detail,
                    })
                    continue
                target_path = gen_dir / target_name
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
                decisions.append({
                    "target": target_name,
                    "decision": "applied",
                    "reason": "passed completeness heuristic",
                })

            # Did the punch list point at header-side problems and did
            # the LLM actually touch the .h on this attempt?
            had_header_side_errors = bool(
                err_struct["missing_members"]
                or err_struct["unknown_types"]
                or err_struct["implicit_decls"]
                or err_struct["conflicting_types"]
            )
            touched_h = any(t.endswith(".h") for t in applied)
            if had_header_side_errors and not touched_h:
                prev_missed_h_side = True
                prev_missed_details = []
                for s, m in err_struct["missing_members"]:
                    prev_missed_details.append(
                        f"struct `{s}` is missing member `{m}` "
                        f"(declared in the .h, not the .c)"
                    )
                for t in err_struct["unknown_types"]:
                    prev_missed_details.append(
                        f"type `{t}` is undeclared (typedef belongs in a .h)"
                    )
                for f_ in err_struct["implicit_decls"]:
                    prev_missed_details.append(
                        f"`{f_}()` used without a prototype "
                        f"(prototype belongs in a .h)"
                    )
                for t in err_struct["conflicting_types"]:
                    prev_missed_details.append(
                        f"`{t}` has conflicting types — header "
                        f"declaration must be updated to match"
                    )
                # Cap the list to keep the banner readable.
                prev_missed_details = prev_missed_details[:8]
            else:
                prev_missed_h_side = False
                prev_missed_details = []

            # Record LLM diagnostics for the report + post-mortem debugging.
            llm_preview = fix_output if len(fix_output) <= 1800 else (
                fix_output[:900] + "\n\n[... LLM output truncated ...]\n\n"
                + fix_output[-900:]
            )
            attempts_log[-1]["llm_diagnostics"] = {
                "length": len(fix_output),
                "parsed_names": sorted(parsed.keys()),
                "fenced_block_count": len(_all_fenced_blocks(fix_output)),
                "considered_files": sorted(new_files.keys()),
                "decisions": decisions,
                "applied": applied,
                "preview": llm_preview,
                "had_header_side_errors": had_header_side_errors,
                "touched_h": touched_h,
                "focus_banner_used_next": had_header_side_errors and not touched_h,
            }

            if not applied:
                reason = _summarise_reject_reason(
                    fix_output=fix_output,
                    parsed_names=list(parsed.keys()),
                    decisions=decisions,
                    allowed=allowed,
                )
                yield _sse({
                    "type": "info",
                    "stage": "compile",
                    "file": fname,
                    "message": (
                        f"LLM fix attempt for {fname} not applied "
                        f"({reason}) — keeping current file."
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
        f"Quote-includes:     {len(quote_needs)} parsed, "
        f"{n_found} resolved, {n_missing} unresolved"
        if quote_needs else "Quote-includes:     none",
        "",
        "-" * 65,
        "INCLUDE PATH (gen_dir + code_dir + demand-driven project-local "
        "headers from repo; toolchain sysroots filtered out)",
        "-" * 65,
    ]
    for d in include_dirs:
        rep.append(f"  -I{d}")

    if resolutions:
        rep.extend([
            "",
            "-" * 65,
            "QUOTE-INCLUDE RESOLUTIONS",
            "-" * 65,
        ])
        for need in sorted(resolutions):
            r = resolutions[need]
            st = r.get("status", "?")
            line = f"  [{st}] {need}"
            if "path" in r and r["path"]:
                line += f"\n      path: {r['path']}"
            if "include_root" in r and r["include_root"]:
                line += f"\n      -I:   {r['include_root']}"
            if "shim" in r and r["shim"]:
                line += f"\n      shim: {r['shim']}"
            if r.get("skipped_toxic"):
                line += "\n      skipped toxic candidates:"
                for s in r["skipped_toxic"]:
                    line += f"\n        - {s}"
            if "error" in r and r["error"]:
                line += f"\n      error: {r['error']}"
            rep.append(line)

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
            if a.get("llm_diagnostics"):
                diag = a["llm_diagnostics"]
                if "error" in diag:
                    rep.append(
                        f"  LLM diagnostics: ERROR — {diag['error']}"
                    )
                else:
                    rep.append("  LLM diagnostics:")
                    rep.append(
                        f"    response length     : {diag.get('length', '?')} chars"
                    )
                    rep.append(
                        f"    fenced code blocks  : {diag.get('fenced_block_count', '?')}"
                    )
                    rep.append(
                        "    parsed names        : "
                        + (", ".join(diag.get("parsed_names") or []) or "(none)")
                    )
                    rep.append(
                        "    considered files    : "
                        + (", ".join(diag.get("considered_files") or []) or "(none)")
                    )
                    rep.append(
                        "    applied             : "
                        + (", ".join(diag.get("applied") or []) or "(none)")
                    )
                    if diag.get("decisions"):
                        rep.append("    per-target decisions:")
                        for d in diag["decisions"]:
                            rep.append(
                                f"      - {d['target']}: {d['decision']} "
                                f"({d['reason']})"
                            )
                    preview = diag.get("preview") or ""
                    if preview:
                        rep.append("    LLM output preview:")
                        rep.append(
                            "      "
                            + preview.replace("\n", "\n      ")[:6000]
                        )
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
    # Match 2-to-6 hashes (covers `## file`, `### file`, etc. — some LLMs
    # over- or under-shoot when asked for `###`). The captured group is
    # the entire text after the hashes, which we then strip and validate.
    r"^#{2,6}\s+([^\n]+?)\s*$",
    re.MULTILINE,
)
# Recognises the C-family extensions we accept as filename markers. Any
# `###`-prefixed line whose payload does NOT end in one of these (after
# stripping wrappers) is treated as a markdown section header and
# IGNORED — this stops the parser from misreading `### Analysis` /
# `### Fix` / `### Reasoning` etc. as filenames and silently dropping
# the actual fenced code block.
_ALLOWED_C_EXTS_RE = re.compile(
    r"\.(?:c|h|cc|hh|cpp|hpp|cxx|hxx|inc)$",
    re.IGNORECASE,
)
# Characters LLMs often wrap filenames in: backticks, asterisks, quotes,
# angle brackets, italic underscores, trailing colons / dashes / spaces.
_FILENAME_WRAPPER_CHARS = "`*\"'<>_ \t:-—–·•"


def _looks_like_c_filename(token: str) -> Optional[str]:
    """Return the cleaned-up filename if *token* looks like a C-family
    source/header filename, else ``None``.

    Handles common LLM dressings:

      - ``### IMU.c`` -> ``"IMU.c"``
      - ``### `IMU.c```` -> ``"IMU.c"``
      - ``### **IMU.h**`` -> ``"IMU.h"``
      - ``### src/IMU.c`` -> ``"IMU.c"`` (basename)
      - ``### "IMU.c":`` -> ``"IMU.c"``
      - ``### IMU.c  (corrected)`` -> ``"IMU.c"`` (drops trailing junk)
      - ``### Analysis`` -> ``None``  (no C extension)
      - ``### Fix`` -> ``None``
      - ``### # IMU.c`` -> ``"IMU.c"`` (handles extra leading hashes)
    """
    if not token:
        return None
    # 1. Drop trailing parenthetical/inline annotations the LLM
    # sometimes adds: `### IMU.c  (corrected)` or `### IMU.c : fixed`.
    for sep in ("(", "[", ":", " - ", " — ", "  "):
        idx = token.find(sep)
        if idx != -1:
            token = token[:idx]
    # 2. Pick the first whitespace-separated token (after the strip
    # above, this is usually the entire payload).
    parts = token.split()
    if not parts:
        return None
    candidate = parts[0].strip(_FILENAME_WRAPPER_CHARS)
    # 3. Reduce any directory prefix to a bare basename.
    candidate = candidate.replace("\\", "/").rsplit("/", 1)[-1]
    if not candidate:
        return None
    if not _ALLOWED_C_EXTS_RE.search(candidate):
        return None
    return candidate


# A leading `### <name>.<c-ext>` line inside a fenced block has only
# one possible origin: the LLM mis-nested its own filename marker
# inside the code fence. C/C++ grammar can never start with three
# hashes, so it's an unambiguous parser-corruption signal. We strip
# any such leading lines (and any blanks they leave behind) to
# prevent the marker from being written verbatim to disk and
# triggering ``error: stray '##' in program`` on the very next
# compile (the IMU.c production failure).
_LEAKED_FILENAME_MARKER_RE = re.compile(
    r"^\s*#{2,6}\s+\S+\.(?:c|h|cc|hh|cpp|hpp|cxx|hxx|inc)\s*$",
    re.IGNORECASE,
)


def _strip_filename_marker_leakage(body: str) -> str:
    """Drop any leading `### foo.c` / `### foo.h` lines (and the blank
    lines that follow them) from a fenced-block body. Idempotent and
    safe to call on every extracted block. Returns the body unchanged
    when nothing leaks.
    """
    if not body:
        return body
    lines = body.splitlines()
    changed = False
    while lines and _LEAKED_FILENAME_MARKER_RE.match(lines[0]):
        lines.pop(0)
        changed = True
    # Drop one or more blank lines the marker may have left behind so
    # the resulting file doesn't start with a stray blank.
    while lines and not lines[0].strip():
        lines.pop(0)
        changed = True
    if not changed:
        return body
    return "\n".join(lines)


def _fenced_spans(text: str) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` for every triple-fenced region in
    *text*. Spans cover the opening ``` and the closing ``` so any
    position strictly inside the span is "inside a code fence".
    Used by :func:`_extract_per_file_blocks` to ignore `###` markers
    the LLM nested inside its own code fence (the IMU.c production
    failure: the LLM repeated `### IMU.c` *inside* its ``` block,
    which confused the strict parser into returning empty).
    """
    return [
        (m.start(), m.end())
        for m in re.finditer(
            r"```[^\n]*\n.*?\n```", text, flags=re.DOTALL,
        )
    ]


def _pos_in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in spans)


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

    Returns a ``{filename: code}`` dict.  Filename headers whose name
    does NOT look like a C-family source/header filename are treated
    as ordinary markdown section headers and ignored (no false-positive
    parse of `### Analysis` / `### Fix` etc.).  `###` markers that fall
    INSIDE an already-open code fence are also ignored — they are
    LLM-side mistakes that previously broke chunk slicing.  Missing or
    unparseable input ⇒ empty dict (caller should fall back to a single
    fenced block).

    Each extracted body is also stripped of any leading `### filename`
    lines the LLM may have mistakenly nested INSIDE the code fence
    (see :func:`_strip_filename_marker_leakage`).
    """
    out: dict[str, str] = {}
    spans = _fenced_spans(text)
    headers = [
        m for m in _PER_FILE_HEADER_RE.finditer(text)
        if not _pos_in_spans(m.start(), spans)
    ]
    if not headers:
        return out
    for i, m in enumerate(headers):
        cleaned = _looks_like_c_filename(m.group(1))
        if not cleaned:
            continue
        start = m.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        chunk = text[start:end]
        fence = re.search(
            r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\n(.*?)\n```",
            chunk, flags=re.DOTALL,
        )
        if fence:
            body = _strip_filename_marker_leakage(fence.group(1).strip())
            if body:
                out[cleaned] = body
    return out


def _all_fenced_blocks(text: str) -> list[str]:
    """Return every triple-fence body in *text* (in source order).

    Each body is post-processed by :func:`_strip_filename_marker_leakage`
    so a leaked `### IMU.c` at the top of a fenced block can never end
    up written to disk.
    """
    out: list[str] = []
    for m in re.finditer(
        r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\n(.*?)\n```",
        text, flags=re.DOTALL,
    ):
        body = _strip_filename_marker_leakage(m.group(1).strip())
        if body:
            out.append(body)
    return out


def _incomplete_file_reason(generated: str, original: str, filename: str) -> str:
    """Mirror :func:`api.app._looks_complete_c_file` but return a short
    human-readable label describing the FIRST failed check (or
    ``"passes"`` if the file passes all heuristics). Used to surface
    actionable diagnostics instead of the opaque
    "did not yield a complete usable rewrite" message.
    """
    text = (generated or "").strip()
    if not text:
        return "LLM output was empty"
    if "[... TRUNCATED" in text or text.endswith("..."):
        return "LLM output looked truncated (ends with ellipsis)"
    if text.count("{") != text.count("}"):
        return (
            f"unbalanced braces (got {text.count('{')} `{{` vs "
            f"{text.count('}')} `}}`)"
        )
    if text.count("/*") > text.count("*/"):
        return "unbalanced /* ... */ block comment"
    if filename.endswith(".h"):
        has_ifndef = re.search(r"^\s*#\s*ifn?def\b", text, flags=re.MULTILINE)
        has_endif = re.search(r"^\s*#\s*endif\b", text, flags=re.MULTILINE)
        if has_ifndef and not has_endif:
            return "include guard opened with `#ifndef` but no matching `#endif`"
    min_len = max(120, int(len((original or "").strip()) * 0.35))
    if len(text) < min_len:
        return (
            f"too short ({len(text)} chars < {min_len} required relative "
            f"to the original {len((original or '').strip())}-char file)"
        )
    last = text.splitlines()[-1].strip() if text.splitlines() else ""
    if not (
        last.endswith("}")
        or last.endswith(";")
        or last.endswith("*/")
        or last.startswith("#endif")
    ):
        return (
            f"last line does not look like a terminator: "
            f"{last[:60]!r}"
        )
    return "passes"


def _summarise_reject_reason(
    *,
    fix_output: str,
    parsed_names: list[str],
    decisions: list[dict],
    allowed: set,
) -> str:
    """Build a short, precise summary of WHY the LLM fix attempt
    produced no usable rewrite — for SSE clients and the report.
    """
    if not fix_output.strip():
        return "LLM returned an empty response"
    fenced = _all_fenced_blocks(fix_output)
    if not fenced and not parsed_names:
        return "no triple-fenced code blocks found in LLM output"
    if not decisions:
        # We parsed blocks but none ever entered the decision loop —
        # i.e. every parsed name was something other than the allowed
        # set AND the content-sniff/single-fence fallbacks all failed.
        return (
            "LLM output had fenced code but no block could be attributed "
            f"to {fname_summary(allowed)} "
            f"(parser saw filenames {parsed_names or 'none'})"
        )
    rejected_incomplete = [
        d for d in decisions if d["decision"] == "rejected_incomplete"
    ]
    skipped_unknown = [
        d for d in decisions if d["decision"] == "skipped_unknown_name"
    ]
    if rejected_incomplete:
        first = rejected_incomplete[0]
        return (
            f"rewrite of {first['target']} failed completeness check "
            f"({first['reason']})"
        )
    if skipped_unknown:
        names = ", ".join(sorted({d["target"] for d in skipped_unknown}))
        return (
            f"LLM rewrote {names!r} but neither is in "
            f"{fname_summary(allowed)}"
        )
    return "no applicable rewrite produced"


def fname_summary(allowed: set) -> str:
    """Render a small set of allowed filenames as ``{IMU.c, IMU.h}``."""
    return "{" + ", ".join(sorted(allowed)) + "}"


def _guess_filename_for_block(
    body: str,
    *,
    c_name: str,
    h_name: Optional[str],
) -> Optional[str]:
    """Decide whether a fenced block looks like the .c or the .h file.

    Used as a last-resort fallback when the LLM emitted code blocks but
    forgot to mark them with `### <filename>` headers.  Heuristics:

      - A typical .h has an include guard (``#ifndef <BASE>_H`` /
        ``#define <BASE>_H`` ... ``#endif``).
      - A typical .c does ``#include "<base>.h"`` and contains
        function bodies (``int foo(...) {`` patterns).
    """
    if not body or not c_name:
        return None
    base = Path(c_name).stem.upper()
    h_guard = re.search(
        rf"#\s*ifn?def\s+{re.escape(base)}_H\b", body, flags=re.IGNORECASE,
    )
    h_endif = re.search(r"#\s*endif\b", body)
    looks_like_h = bool(h_guard and h_endif) and h_name is not None
    if looks_like_h:
        return h_name
    # Cheap "this looks like a .c" check: at least one function body
    # opener with `{` on a non-comment line.
    looks_like_c = bool(
        re.search(r"^\s*[A-Za-z_][\w\s\*]*\([^;]*\)\s*\{", body, re.MULTILINE)
    )
    if looks_like_c:
        return c_name
    return None
