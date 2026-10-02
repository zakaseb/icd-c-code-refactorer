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
to repair the offending ``.c`` and the project headers it includes (not
only the same-stem ``.h``) using the same prompting strategy as
``_sandbox_build_iterate``:

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
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Generator, Optional

try:
    from api.hex_sdk import (  # noqa: F401  (re-exported for tests)
        HexSdkContext,
        discover as _discover_hex_sdk_raw,
        find_sdk_header as _find_sdk_header,
        render_generated_header as _render_generated_header,
        upload_needs_hex_sdk as _upload_needs_hex_sdk,
    )
except ImportError:  # flat layout (uvicorn app:app)
    from hex_sdk import (  # type: ignore[no-redef]
        HexSdkContext,
        discover as _discover_hex_sdk_raw,
        find_sdk_header as _find_sdk_header,
        render_generated_header as _render_generated_header,
        upload_needs_hex_sdk as _upload_needs_hex_sdk,
    )


def _discover_hex_sdk(
    *, repo_dir: Optional[Path], cross_hint: Optional[str],
) -> Optional["HexSdkContext"]:
    """Thin wrapper so tests can monkey-patch discovery on this module.

    The compile gate probes for an SDK using the uploaded repository as
    the search root first (users often bundle the SDK), then falls back
    to sibling directories / OS install locations handled by
    :func:`api.hex_sdk.discover`.
    """
    extras: list[Path] = []
    if repo_dir and repo_dir.exists():
        extras.append(repo_dir)
    ctx = _discover_hex_sdk_raw(
        repo_root=repo_dir,
        extra_search_dirs=tuple(extras),
        cross_hint=cross_hint,
    )
    # A SDK installed for the whole machine must not retarget an
    # unrelated native compile. HEX sources mention L1_api.h / the
    # Zynq BSP; everything else keeps the previous toolchain.
    if ctx is not None and repo_dir is not None and not _upload_needs_hex_sdk(repo_dir):
        return None
    return ctx

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
_ANGLE_INCLUDE_RE = re.compile(
    r'^\s*#\s*include\s*<([^>]+)>', re.MULTILINE,
)
_LIBC_ANGLE_SKIP = {
    "assert.h", "ctype.h", "errno.h", "float.h", "limits.h", "math.h",
    "setjmp.h", "signal.h", "stdarg.h", "stddef.h", "stdint.h", "stdio.h",
    "stdlib.h", "string.h", "time.h", "wchar.h", "wctype.h", "stdbool.h",
    "windows.h",
}

# Identifiers the ICD is not allowed to respell just because the peripheral
# number changed (IMU20 -> IMU15). Includes are handled separately.
_SCRIPT_SYMBOL_PREFIXES = ("EV_", "FIFO_", "HUB_", "HI_", "PERFT_")
_INCLUDE_LINE_RE = re.compile(
    r"^([ \t]*#[ \t]*include[ \t]*)([<\"])([^>\"]+)([>\"])",
    re.MULTILINE,
)
_C_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_IMPACT_MANIFEST_LINE_RE = re.compile(
    r"^[ \t]*IMPACT[-_ ]?MANIFEST\b[^\n]*\n?",
    re.MULTILINE | re.IGNORECASE,
)


def _peripheral_name_key(text: str) -> str:
    """Compare script and symbol names with peripheral digits removed.

    ``plImu15Msg.h`` and ``plImu20msg.h`` share the key ``plimumsg.h``.
    ``EV_IMU15_RDY`` and ``EV_IMU_RDY`` share ``ev_imu_rdy``.
    """
    leaf = Path(str(text).replace("\\", "/")).name
    return re.sub(r"\d+", "", leaf).casefold()


def _code_identifiers(text: str) -> set[str]:
    """Identifiers that appear outside comments and string literals."""
    found: set[str] = set()
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        if text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end + 1
            continue
        ch = text[i]
        if ch in "\"'":
            i += 1
            while i < n and text[i] != ch:
                if text[i] == "\\":
                    i += 2
                else:
                    i += 1
            i += 1
            continue
        match = _C_IDENT_RE.match(text, i)
        if match:
            found.add(match.group(0))
            i = match.end()
            continue
        i += 1
    return found


def _drop_code_lines_using(text: str, victims: set[str]) -> str:
    """Remove lines whose code (not comments or strings) uses *victims*."""
    kept: list[str] = []
    in_block = False
    for line in text.splitlines(keepends=True):
        code = line
        if in_block:
            end = code.find("*/")
            if end < 0:
                kept.append(line)
                continue
            code = code[end + 2:]
            in_block = False
        stripped = []
        i = 0
        while i < len(code):
            if code.startswith("/*", i):
                end = code.find("*/", i + 2)
                if end < 0:
                    in_block = True
                    break
                i = end + 2
                continue
            if code.startswith("//", i):
                break
            if code[i] in "\"'":
                quote = code[i]
                i += 1
                while i < len(code) and code[i] != quote:
                    i += 2 if code[i] == "\\" else 1
                i += 1
                continue
            stripped.append(code[i])
            i += 1
        code_text = "".join(stripped)
        idents = set(_C_IDENT_RE.findall(code_text))
        if idents & victims:
            continue
        kept.append(line)
    return "".join(kept)


def preserve_original_script_names(original: str, generated: str) -> str:
    """Put original script names back into a transformed file.

    The transform is allowed to update ICD fields and comments. It is not
    allowed to rename the script those fields live in. An include that
    changed only by a peripheral number (``plImu20msg.h`` -> ``plImu15Msg.h``)
    is restored to the original spelling, and the same rule applies to
    ``EV_`` / ``FIFO_`` / ``HUB_`` / ``HI_`` / ``PERFT_`` identifiers.
    """
    generated = _IMPACT_MANIFEST_LINE_RE.sub("", generated)
    if generated and not generated.endswith("\n"):
        generated += "\n"
    if not original or not generated or original == generated:
        return generated

    orig_paths: list[str] = []
    by_key: dict[str, list[str]] = {}
    for match in _INCLUDE_LINE_RE.finditer(original):
        path = match.group(3).strip()
        if path not in orig_paths:
            orig_paths.append(path)
        by_key.setdefault(_peripheral_name_key(path), [])
        if path not in by_key[_peripheral_name_key(path)]:
            by_key[_peripheral_name_key(path)].append(path)

    def _restore_include(match: re.Match) -> str:
        path = match.group(3).strip()
        if path in orig_paths:
            return match.group(0)
        candidates = by_key.get(_peripheral_name_key(path), [])
        if len(candidates) == 1 and candidates[0] != path:
            return f"{match.group(1)}{match.group(2)}{candidates[0]}{match.group(4)}"
        for candidate in candidates:
            if candidate.casefold() == path.casefold() and candidate != path:
                return (
                    f"{match.group(1)}{match.group(2)}{candidate}{match.group(4)}"
                )
        return match.group(0)

    text = _INCLUDE_LINE_RE.sub(_restore_include, generated)

    unique: dict[str, str] = {}
    grouped: dict[str, set[str]] = {}
    for token in _C_IDENT_RE.findall(original):
        if not token.startswith(_SCRIPT_SYMBOL_PREFIXES):
            continue
        grouped.setdefault(_peripheral_name_key(token), set()).add(token)
    for key, tokens in grouped.items():
        if len(tokens) == 1:
            unique[key] = next(iter(tokens))

    restored: set[str] = set()

    def _restore_symbol(match: re.Match) -> str:
        token = match.group(0)
        if not token.startswith(_SCRIPT_SYMBOL_PREFIXES):
            return token
        original_token = unique.get(_peripheral_name_key(token))
        if original_token and original_token != token:
            restored.add(original_token)
            return original_token
        return token

    text = _C_IDENT_RE.sub(_restore_symbol, text)
    if not restored:
        return text
    # A comment-only name (EV_IMU_RDY in the banner) must not be promoted
    # into executable code under the new peripheral's spelling. Drop those
    # code lines; comments keep the original name.
    original_code_idents = _code_identifiers(original)
    comment_only = restored - original_code_idents
    if not comment_only:
        return text
    return _drop_code_lines_using(text, comment_only)


def _attribute_block_to_original_script(
    name: str, allowed: set[str],
) -> Optional[str]:
    """Map a model-chosen filename back onto the script being repaired.

    The fix loop asks for ``prxyImu.c``. A reply labeled ``### prxyImu.h``
    or ``### plImu15Msg.h`` (when the original header is ``plImu20Msg.h``)
    is attributed to that original name instead of being discarded.
    """
    if name in allowed:
        return name
    stem = Path(name).stem.casefold()
    suffix = Path(name).suffix.lower()
    # A header block is never the .c of the same stem (and vice versa): a
    # `### prxyImu.h` reply to a missing prxyImu.h used to land on prxyImu.c,
    # where it replaced the source or was rejected as a truncated .c.
    is_header = suffix in _HEADER_EXTS
    same_stem = [
        candidate for candidate in allowed
        if Path(candidate).stem.casefold() == stem
        and (Path(candidate).suffix.lower() in _HEADER_EXTS) == is_header
    ]
    if len(same_stem) == 1:
        return same_stem[0]
    key = _peripheral_name_key(name)
    same_key = [
        candidate for candidate in allowed
        if _peripheral_name_key(candidate) == key
        and Path(candidate).suffix.lower() == suffix
    ]
    if len(same_key) == 1:
        return same_key[0]
    return None


def _restore_script_names_in_gen(
    gen_dir: Path, code_dir: Optional[Path],
) -> list[str]:
    """Rewrite generated scripts so they keep the uploaded filenames.

    A generated ``plImu15Msg.h`` is renamed to ``plImu20Msg.h`` when that
    is the uploaded header and the generated name is only a peripheral-number
    change. Includes and event/FIFO identifiers inside each generated file
    are restored the same way.
    """
    notes: list[str] = []
    if code_dir is None or not code_dir.is_dir():
        return notes
    exts = _SOURCE_EXTS | _HEADER_EXTS
    originals = {
        path.name: path
        for path in code_dir.iterdir()
        if path.is_file() and path.suffix.lower() in exts
    }
    for generated_path in list(gen_dir.iterdir()):
        if (
            not generated_path.is_file()
            or generated_path.suffix.lower() not in exts
            or generated_path.name in originals
        ):
            continue
        matches = [
            original_name for original_name in originals
            if _peripheral_name_key(original_name) == _peripheral_name_key(generated_path.name)
            and Path(original_name).suffix.lower() == generated_path.suffix.lower()
        ]
        if len(matches) != 1:
            continue
        dest = gen_dir / matches[0]
        if dest.exists():
            continue
        generated_path.rename(dest)
        notes.append(f"{generated_path.name} -> {matches[0]}")
    for original_name, original_path in originals.items():
        generated_path = gen_dir / original_name
        if not generated_path.is_file():
            continue
        try:
            original_text = original_path.read_text(errors="replace")
            generated_text = generated_path.read_text(errors="replace")
        except OSError:
            continue
        fixed = preserve_original_script_names(original_text, generated_text)
        if fixed != generated_text:
            generated_path.write_text(fixed)
            notes.append(original_name)
    return notes


def _angle_includes_in_tree(directory: Optional[Path], *, max_files: int = 2500) -> set[str]:
    """Non-libc ``#include <...>`` paths under *directory*."""
    out: set[str] = set()
    if directory is None or not directory.exists():
        return out
    seen = 0
    try:
        for fp in directory.rglob("*"):
            if not fp.is_file() or fp.suffix.lower() not in (_HEADER_EXTS | _SOURCE_EXTS):
                continue
            seen += 1
            if seen > max_files:
                break
            try:
                text = fp.read_text(errors="replace")
            except OSError:
                continue
            for m in _ANGLE_INCLUDE_RE.finditer(text):
                rel = m.group(1).strip().replace("\\", "/")
                if not rel or Path(rel).name.lower() in _LIBC_ANGLE_SKIP:
                    continue
                out.add(rel)
    except OSError:
        return out
    return out


def _find_local_header(rel: str, dirs: list[Path]) -> Path | None:
    """Find a project header by basename, preferring an exact-case match.

    ``#include <plImu20msg.h>`` resolves to on-disk ``plImu20Msg.h``.
    Search order follows *dirs* (generated, then uploaded, then repo).
    """
    leaf = Path(rel.replace("\\", "/")).name
    if not leaf or leaf.lower() in _LIBC_ANGLE_SKIP:
        return None
    leaf_l = leaf.lower()
    folded: Path | None = None
    for directory in dirs:
        if directory is None or not directory.is_dir():
            continue
        exact = directory / leaf
        try:
            if exact.is_file():
                return exact
        except OSError:
            continue
        if folded is not None:
            continue
        try:
            for path in directory.iterdir():
                if path.is_file() and path.name.lower() == leaf_l:
                    folded = path
                    break
        except OSError:
            continue
    return folded


def _header_already_on_path(rel: str, include_dirs: list[Path]) -> bool:
    parts = rel.split("/")
    for d in include_dirs:
        cand = d.joinpath(*parts)
        try:
            if cand.is_file():
                return True
        except OSError:
            continue
    return False


def _materialise_header_shim(shim_dir: Path, rel: str, target: Path) -> Path | None:
    """Symlink (or copy) *target* into *shim_dir* under the requested name."""
    try:
        shim_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    shim_path = shim_dir.joinpath(*rel.split("/"))
    try:
        shim_path.parent.mkdir(parents=True, exist_ok=True)
        if shim_path.is_symlink() or shim_path.exists():
            shim_path.unlink()
        try:
            shim_path.symlink_to(target.resolve())
        except OSError:
            shim_path.write_bytes(target.read_bytes())
    except OSError:
        return None
    return shim_path


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
                     include_dirs: list[Path],
                     extra_flags: list[str] | None = None) -> str:
    """Build the per-file compile shell command (single .c → .o).

    *extra_flags* is appended AFTER the include list so any ``-D`` or
    ``-U`` supplied by the HEX SDK (or another integration point) can
    influence preprocessor conditionals without being shadowed by an
    earlier project-local flag.
    """
    inc_args = " ".join(f'"-I{d}"' for d in include_dirs)
    extra = " ".join(_quote_flag(f) for f in (extra_flags or []))
    return f'{cc} {inc_args} {extra} -c -o "{obj}" "{src}" 2>&1'


def _quote_flag(flag: str) -> str:
    """Shell-safe rendering of a single compiler flag.

    Uses double quotes so path-carrying flags such as ``-I/some/dir``
    survive spaces / parentheses. Simple ``-D`` and ``-U`` tokens fall
    back to no quoting so ``-DL1_LOCAL_PTR_SIZE=32`` reaches the
    compiler with the ``=`` intact.
    """
    if not flag:
        return ""
    if flag.startswith(("-I", "-L", "-isystem")) or "/" in flag:
        return f'"{flag}"'
    return flag


def _run_compile(cc: str, src: Path, obj: Path,
                 include_dirs: list[Path], timeout: int,
                 cwd: Path,
                 extra_flags: list[str] | None = None) -> tuple[bool, str, str]:
    """Run one compile.  Returns ``(success, command, combined_output)``.

    Removes any stale ``obj`` first so a missing ``.o`` after the run is an
    unambiguous failure signal.
    """
    try:
        if obj.exists():
            obj.unlink()
    except OSError:
        pass
    cmd = _compile_command(cc, src, obj, include_dirs, extra_flags=extra_flags)
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
_READONLY_RE = re.compile(
    rf"assignment of member\s+{_QUOTED}\s+in read-only object",
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
      ``readonly_assigns``    : list[member_name]

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
        "readonly_assigns": _uniq(
            [m.group(1) for m in _READONLY_RE.finditer(text)]
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
    if struct.get("readonly_assigns"):
        lines.append(
            "Assignment to a const object — remove `const` from that "
            "object's definition in the .c, or stop assigning to it. "
            "A unified diff of the .c is enough; do not rewrite the "
            "whole file:"
        )
        for member in struct["readonly_assigns"]:
            lines.append(f"  - member `{member}`")
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


_MAX_FIX_HEADERS = 8
_FIX_HEADER_CHARS = 20000
# A complete project header plus the few members the compiler asked
# for. Larger than the global reply cap so the fix is not split across
# a continuation that reopens a code fence.
_COMPILE_FIX_OUTPUT_TOKENS = 12288


def _header_mentions(text: str, names: list[str]) -> bool:
    """True when ``text`` contains any compiler-named type as a whole word."""
    for name in names:
        if not name or len(name) < 2:
            continue
        if re.search(rf"\b{re.escape(name)}\b", text):
            return True
    return False


def _collect_fix_headers(
    c_text: str,
    err_struct: dict,
    gen_dir: Path,
    code_dir: Optional[Path],
    companion: Optional[Path],
) -> list[Path]:
    """Project headers the compile-fix agent may edit for this ``.c``.

    The same-stem companion is included when it exists. So is every
    project header the ``.c`` includes, and any project header that
    already mentions a type or struct named in the compiler errors.
    The generated-tree copy is the one returned, because that is the
    file the compiler reads first. Libc and SDK headers are not
    candidates: they are not in the generated or uploaded tree.
    """
    declare_names: list[str] = []
    declare_names.extend(err_struct.get("unknown_types") or [])
    declare_names.extend(
        struct_name for struct_name, _member in (err_struct.get("missing_members") or [])
    )

    included: set[str] = set()
    for rx in (_QUOTED_INCLUDE_RE, _ANGLE_INCLUDE_RE):
        for match in rx.finditer(c_text or ""):
            base = Path(match.group(1).strip()).name.lower()
            if base in _LIBC_ANGLE_SKIP:
                continue
            included.add(base)

    search_dirs: list[Path] = []
    if gen_dir is not None and gen_dir.is_dir():
        search_dirs.append(gen_dir)
    if (
        code_dir is not None
        and code_dir.is_dir()
        and code_dir.resolve() != gen_dir.resolve()
    ):
        search_dirs.append(code_dir)

    # lower-case basename -> (score, path). A higher score wins. An
    # equal score prefers the generated tree.
    scored: dict[str, tuple[int, Path]] = {}

    def consider(path: Path, score: int) -> None:
        if score <= 0 or not path.is_file():
            return
        key = path.name.lower()
        prev = scored.get(key)
        in_gen = path.parent.resolve() == gen_dir.resolve()
        if prev is None:
            scored[key] = (score, path)
            return
        prev_score, prev_path = prev
        prev_in_gen = prev_path.parent.resolve() == gen_dir.resolve()
        if score > prev_score or (score == prev_score and in_gen and not prev_in_gen):
            scored[key] = (score, path)

    if companion is not None:
        consider(companion, 5)

    for directory in search_dirs:
        for header in directory.iterdir():
            if not header.is_file() or header.suffix.lower() not in _HEADER_EXTS:
                continue
            score = 5 if companion is not None and header.name == companion.name else 0
            if header.name.lower() in included:
                score += 3
            try:
                text = header.read_text(errors="replace")
            except OSError:
                text = ""
            if declare_names and _header_mentions(text, declare_names):
                score += 10
            consider(header, score)

    ranked = sorted(
        scored.values(),
        key=lambda item: (-item[0], item[1].name.lower()),
    )[:_MAX_FIX_HEADERS]

    chosen: list[Path] = []
    for _score, path in ranked:
        dest = gen_dir / path.name
        if path.parent.resolve() != gen_dir.resolve():
            if not dest.exists():
                try:
                    dest.write_text(path.read_text(errors="replace"))
                except OSError:
                    continue
            path = dest
        chosen.append(path)
    return chosen


def _brace_span(text: str, open_at: int) -> Optional[tuple[int, int]]:
    """Return ``(open, close)`` indexes for the brace at ``open_at``."""
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return open_at, i
    return None


def _typedef_struct_spans(text: str) -> list[tuple[str, int, int]]:
    """``(name, body_start, close_brace)`` for ``typedef struct { } Name;``."""
    found: list[tuple[str, int, int]] = []
    for match in re.finditer(r"typedef\s+struct\b[^{;]*\{", text):
        span = _brace_span(text, match.end() - 1)
        if span is None:
            continue
        after = text[span[1] + 1:span[1] + 120]
        name = re.match(r"\s*(\w+)\s*;", after)
        if name:
            found.append((name.group(1), span[0] + 1, span[1]))
    return found


def _split_params(param_str: str) -> list[str]:
    if not param_str or param_str.strip() in ("", "void"):
        return []
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in param_str:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _type_of_param(param: str) -> str:
    """Drop the parameter name from ``IMUD_Timing * pTiming``."""
    cleaned = re.sub(r"/\*.*?\*/", "", param).strip()
    cleaned = cleaned.replace("*", " * ")
    tokens = cleaned.split()
    if not tokens:
        return "int"
    if len(tokens) >= 2 and re.match(r"[A-Za-z_]\w*$", tokens[-1]):
        tokens = tokens[:-1]
    return " ".join(tokens).strip() or "int"


def _pointee(type_name: str) -> str:
    stripped = type_name.strip()
    if stripped.endswith("*"):
        return stripped[:-1].strip() or "int"
    return stripped


_SKIP_CALLS = {"if", "for", "while", "switch", "return", "sizeof"}


def _iter_calls(text: str):
    """Yield ``(function_name, [arg, ...])`` for C calls in ``text``."""
    i = 0
    n = len(text)
    while i < n:
        match = re.search(r"\b([A-Za-z_]\w*)\s*\(", text[i:])
        if not match:
            return
        name = match.group(1)
        paren = i + match.end() - 1
        if name in _SKIP_CALLS:
            i = paren + 1
            continue
        depth = 0
        j = paren
        while j < n:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
                if depth == 0:
                    yield name, _split_params(text[paren + 1:j])
                    i = j + 1
                    break
            j += 1
        else:
            return


# A declaration has a return type before the name. A call, and a
# ``sizeof(...)`` that has no semicolon of its own, do not. Nested
# parentheses are excluded so one declarator cannot swallow the next.
_PROTO_RE = re.compile(
    r"(?:^|[;{}])\s*(?:[A-Za-z_]\w*\s+)+([A-Za-z_]\w*)\s*\(([^;{}()]*)\)\s*;",
    re.M,
)


def _looks_like_type(type_name: str) -> bool:
    return bool(re.fullmatch(
        r"[A-Za-z_][\w\s\*]*(\[\s*\d+\s*\])?",
        (type_name or "").strip(),
    ))


def _index_prototypes(include_dirs: list[Path], extra_texts: list[str]) -> dict[str, list[str]]:
    """Map a function name to the type of each parameter."""
    protos: dict[str, list[str]] = {}

    def absorb(text: str) -> None:
        for match in _PROTO_RE.finditer(_without_comments(text)):
            name = match.group(1)
            if name in _SKIP_CALLS or name in protos:
                continue
            protos[name] = [
                _type_of_param(part) for part in _split_params(match.group(2))
            ]

    for directory in include_dirs or []:
        if directory is None or not Path(directory).is_dir():
            continue
        for header in Path(directory).iterdir():
            if not header.is_file() or header.suffix.lower() not in _HEADER_EXTS:
                continue
            try:
                absorb(header.read_text(errors="replace"))
            except OSError:
                continue
    for text in extra_texts:
        absorb(text)
    return protos


def _member_passed_as(arg: str, member: str) -> Optional[bool]:
    """True when ``arg`` is ``&....member``, False when it is ``....member``.

    None when this argument is not a use of ``member``.
    """
    cleaned = re.sub(r"^\s*\([^)]*\)\s*", "", arg.strip())
    addressed = cleaned.startswith("&")
    if addressed:
        cleaned = cleaned[1:].strip()
    if re.search(rf"(?:\.|->)\s*{re.escape(member)}\s*$", cleaned):
        return addressed
    return None


def _rhs_type(rhs: str, style: str, known: dict[str, str]) -> str:
    """A C type for an assignment's right-hand side."""
    if re.search(r"\b\d+\.\d+f\b", rhs):
        base = "float"
    elif re.search(r"\b\d+\.\d+\b", rhs):
        base = "double"
    elif re.fullmatch(r"-?\d+[uUlL]*", rhs.strip()):
        base = "int"
    else:
        base = "int"
        for ident in re.findall(r"\b[A-Za-z_]\w*\b", rhs):
            existing = known.get(ident)
            if existing and re.search(r"float|double", existing, re.I):
                base = existing
                break
    if base == "float" and "float32_t" in style:
        return "float32_t"
    return base


def _without_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def _infer_member_type(
    c_text: str,
    member: str,
    style: str,
    protos: dict[str, list[str]],
    known: dict[str, str],
    owners: Optional[list[str]] = None,
) -> str:
    """Choose a type for ``member`` from how the ``.c`` uses it.

    ``owners`` limits the search to variables of the struct being fixed,
    so a field named ``status`` on two different structs does not inherit
    the other struct's type.
    """
    owner_alt = "|".join(re.escape(name) for name in (owners or []))
    if owner_alt:
        access = rf"\b(?:{owner_alt})\s*(?:\.|->)\s*{re.escape(member)}"
    else:
        access = rf"\b{re.escape(member)}"
    indexes = [
        int(n) for n in re.findall(rf"{access}\s*\[\s*(\d+)\s*\]", c_text)
    ]
    assigned = re.search(
        rf"{access}\s*(?:\[\s*\d+\s*\])?\s*=\s*([^;]+)",
        c_text,
    )
    element = _rhs_type(assigned.group(1), style, known) if assigned else "int"
    if indexes:
        return f"{element}[{max(indexes) + 1}]"
    for func, args in _iter_calls(c_text):
        params = protos.get(func)
        if not params:
            continue
        for index, arg in enumerate(args):
            if index >= len(params):
                break
            if owners and not any(
                re.search(rf"\b{re.escape(owner)}\b", arg) for owner in owners
            ):
                continue
            addressed = _member_passed_as(arg, member)
            if addressed is None:
                continue
            param_type = params[index]
            chosen = _pointee(param_type) if addressed else param_type
            if _looks_like_type(chosen):
                return chosen
    if assigned:
        return element
    return "int"


def _vars_of_type(text: str, type_name: str) -> list[str]:
    return re.findall(
        rf"\b{re.escape(type_name)}\s*\*?\s*(\w+)", text,
    )


def _members_used_on(text: str, var: str) -> list[str]:
    found = re.findall(
        rf"\b{re.escape(var)}\s*(?:\.|->)\s*(\w+)", text,
    )
    seen: list[str] = []
    for name in found:
        if name not in seen:
            seen.append(name)
    return seen


def _insert_member(text: str, struct_name: str, member: str, type_name: str) -> tuple[str, bool]:
    """Add ``member`` inside the existing ``typedef struct`` named ``struct_name``."""
    for name, body_start, close_at in _typedef_struct_spans(text):
        if name != struct_name:
            continue
        body = text[body_start:close_at]
        if re.search(rf"\b{re.escape(member)}\b", _without_comments(body)):
            return text, False
        # ``float[3] name`` is not valid; the array bound belongs on the name.
        if type_name.endswith("]") and "[" in type_name:
            element, bound = type_name.split("[", 1)
            decl = f"    {element.strip()} {member}[{bound}"
            if not decl.endswith(";"):
                decl = decl if decl.endswith("]") else decl
            if not decl.rstrip().endswith(";"):
                decl = decl + ";"
        else:
            decl = f"    {type_name} {member};"
        if not decl.endswith("\n"):
            decl += "\n"
        return text[:close_at] + decl + text[close_at:], True
    return text, False


def _append_typedef(text: str, block: str) -> str:
    last = None
    for last in re.finditer(r"^[ \t]*#\s*endif\b.*$", text, re.M):
        pass
    insertion = block if block.endswith("\n") else block + "\n"
    if last is not None:
        return text[:last.start()] + insertion + text[last.start():]
    if text and not text.endswith("\n"):
        text += "\n"
    return text + insertion


def _drop_retypedefs(text: str) -> str:
    """Remove a later ``typedef Other Name;`` when ``Name`` is already a struct."""
    defined = {name for name, _start, _close in _typedef_struct_spans(text)}
    if not defined:
        return text

    def repl(match: re.Match) -> str:
        return "" if match.group(1) in defined else match.group(0)

    return re.sub(
        r"^[ \t]*typedef\s+(?!struct\b).+\s+(\w+)\s*;[ \t]*\n?",
        repl,
        text,
        flags=re.M,
    )


def _drop_const_on(text: str, object_name: str) -> tuple[str, bool]:
    lines = text.splitlines(keepends=True)
    pattern = re.compile(rf"\b{re.escape(object_name)}\b")
    for index, line in enumerate(lines):
        if "const" not in line or not pattern.search(line):
            continue
        if "=" not in line and ";" not in line:
            continue
        lines[index] = re.sub(r"\bconst\b\s*", "", line, count=1)
        return "".join(lines), True
    return text, False


_READONLY_AT_RE = re.compile(
    rf":(\d+):\d+:\s*(?:error|warning):\s*assignment of member\s+{_QUOTED}\s+in read-only object",
)
_SUGGEST_RE = re.compile(
    rf"{_QUOTED}\s+undeclared\b[^\n]*\bdid you mean\s+{_QUOTED}",
    re.IGNORECASE,
)


def _known_field_types(texts: list[str]) -> dict[str, str]:
    known: dict[str, str] = {}
    for text in texts:
        for _name, body_start, close_at in _typedef_struct_spans(text):
            body = text[body_start:close_at]
            for decl in re.findall(r"([^;{}]+);", body):
                cleaned = re.sub(r"/\*.*?\*/", "", decl).strip()
                if not cleaned or cleaned.startswith("#"):
                    continue
                first, *rest = [part.strip() for part in cleaned.split(",")]
                tokens = first.replace("*", " * ").split()
                if len(tokens) >= 2 and re.match(r"[A-Za-z_]\w*$", tokens[-1]):
                    type_name = " ".join(tokens[:-1])
                    fields = [tokens[-1]]
                else:
                    continue
                for extra in rest:
                    field = extra.replace("*", " ").split()
                    if field:
                        fields.append(field[-1])
                for field in fields:
                    field = field.split("[", 1)[0]
                    if field:
                        known.setdefault(field, type_name)
    return known


def realize_compiler_fixes(
    *,
    c_path: Path,
    headers: list[Path],
    err_struct: dict,
    compiler_output: str,
    include_dirs: list[Path],
) -> list[str]:
    """Finish a header edit the compile-fix agent left short of the compiler.

    The agent is asked to add a missing member to the struct the compiler
    named. When it instead declares a new struct, or leaves a const object
    and an unknown type untouched, this puts the member on the existing
    struct, typedefs an unknown type from how the ``.c`` uses it, aliases
    an undeclared name to the compiler's own suggestion, and drops ``const``
    from the object the assignment is writing.
    """
    notes: list[str] = []
    if not c_path.is_file():
        return notes
    c_text = c_path.read_text(errors="replace")
    header_texts = {
        path: path.read_text(errors="replace")
        for path in headers
        if path.is_file()
    }
    texts = [c_text, *header_texts.values()]
    known = _known_field_types(texts)
    protos = _index_prototypes(include_dirs, texts)
    style = "\n".join(texts)

    def host_for(struct_name: str) -> Optional[Path]:
        if struct_name in {n for n, _s, _c in _typedef_struct_spans(c_text)}:
            return c_path
        for path, text in header_texts.items():
            if struct_name in {n for n, _s, _c in _typedef_struct_spans(text)}:
                return path
        return None

    def write_back(path: Path, text: str) -> None:
        nonlocal c_text
        if path == c_path:
            c_text = text
            c_path.write_text(text)
        elif path in header_texts:
            header_texts[path] = text
            path.write_text(text)

    # Const removal uses the compiler's line numbers, so it has to happen
    # before any insertion shifts the file.
    c_lines = c_text.splitlines()
    objects: set[str] = set()
    for match in _READONLY_AT_RE.finditer(compiler_output or ""):
        line_no = int(match.group(1))
        member = match.group(2)
        if 1 <= line_no <= len(c_lines):
            found = re.search(
                rf"\b(\w+)\s*(?:\.|->)\s*{re.escape(member)}\b",
                c_lines[line_no - 1],
            )
            if found:
                objects.add(found.group(1))
    for obj in sorted(objects):
        updated, changed = _drop_const_on(c_text, obj)
        if changed:
            write_back(c_path, updated)
            notes.append(f"removed const from `{obj}`")

    for struct_name, member in err_struct.get("missing_members") or []:
        path = host_for(struct_name)
        if path is None:
            continue
        current = c_text if path == c_path else header_texts[path]
        owners = _vars_of_type(c_text, struct_name)
        type_name = _infer_member_type(
            c_text, member, style, protos, known, owners,
        )
        # Keep the array bound on the declarator. _infer returns ``double[3]``.
        updated, changed = _insert_member(current, struct_name, member, type_name)
        if changed:
            updated = _drop_retypedefs(updated)
            write_back(path, updated)
            notes.append(
                f"added `{member}` to existing struct `{struct_name}` in {path.name}"
            )

    dest = next(iter(header_texts), None)
    for type_name in err_struct.get("unknown_types") or []:
        if any(type_name in {n for n, _s, _c in _typedef_struct_spans(t)} for t in texts):
            continue
        owners = _vars_of_type(c_text, type_name)
        members = []
        for var in owners:
            for member in _members_used_on(c_text, var):
                if member not in members:
                    members.append(member)
        fields = []
        for member in members:
            field_type = _infer_member_type(
                c_text, member, style, protos, known, owners,
            )
            if field_type.endswith("]") and "[" in field_type:
                element, bound = field_type.split("[", 1)
                fields.append(f"    {element.strip()} {member}[{bound};")
            else:
                fields.append(f"    {field_type} {member};")
        if not fields:
            fields.append("    unsigned char _unused;")
        block = (
            f"typedef struct {{\n"
            + "\n".join(fields)
            + f"\n}} {type_name};\n"
        )
        target = dest if dest is not None else c_path
        current = header_texts[target] if target in header_texts else c_text
        write_back(target, _append_typedef(current, block))
        texts.append(block)
        notes.append(f"typedef'd unknown type `{type_name}` in {target.name}")

    suggested = {
        match.group(1): match.group(2)
        for match in _SUGGEST_RE.finditer(compiler_output or "")
    }
    macros: list[str] = []
    for name in err_struct.get("undeclared") or []:
        if not re.match(r"[A-Za-z_]\w*$", name):
            continue
        if name in suggested and re.match(r"[A-Za-z_]\w*$", suggested[name]):
            macros.append(f"#define {name} {suggested[name]}")
        elif name.isupper() or "_" in name and name.upper() == name:
            macros.append(f"#define {name} 0")
    if macros:
        target = dest if dest is not None else c_path
        current = header_texts[target] if target in header_texts else c_text
        fresh = [
            line for line in macros
            if not re.search(rf"^\s*#\s*define\s+{re.escape(line.split()[1])}\b", current, re.M)
        ]
        if fresh:
            write_back(target, _append_typedef(current, "\n".join(fresh) + "\n"))
            notes.append(
                "declared " + ", ".join(line.split()[1] for line in fresh)
            )

    return notes


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
    max_fix_attempts: int = 10,
    compile_timeout: int = 60,
    cc_override: Optional[str] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Generator[str, None, None]:
    """Run the per-file compile gate, yielding SSE strings.

    Side effects on success: writes ``<base>.o`` files into ``gen_dir`` and
    writes ``gen_dir / "compile_report.txt"``.  On *partial* failure (some
    files still won't compile after the fix budget is spent), the report
    explicitly records the failures and the pipeline is allowed to proceed.

    ``should_stop`` is polled before every compile. Once it returns True no
    further fix pass runs: the file being fixed keeps its current state, the
    files not reached yet get one plain compile, and the report is written.

    Resume semantics: if ``is_resume`` and ``"compile"`` is already in
    ``completed_stages`` and the report file exists, the stage emits a
    short "reusing existing artefacts" sequence and returns.
    """
    # Lazy imports — avoid circular dependency with api/app.py.
    from app import (
        _read_text_safe,
        SANDBOX_CC_NATIVE,
        MAX_REPO_CONTEXT_CHARS,
        MAX_INPUT_TOKENS,
        MAX_OUTPUT_TOKENS,
        COMPILE_FIX_LEAN_PROMPT,
        COMPILE_FIX_MAX_INPUT_TOKENS,
        COMPILE_INCLUDE_SWEEP_REPO,
        COMPILE_INCLUDE_SWEEP_MAX_DIRS,
        _detect_build_system,
        _detect_cross_compiler,
        _build_file_repo_context,
        _build_repo_context,
        _extract_fenced,
        _looks_complete_c_file,
        _looks_complete_header,
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

    # The transform sometimes renames a script to the new ICD peripheral
    # (plImu20msg.h -> plImu15Msg.h, EV_IMU_RDY -> EV_IMU15_RDY). Put the
    # uploaded names back before include resolution, so the compiler looks
    # up the header that actually exists.
    restored = _restore_script_names_in_gen(
        gen_dir, code_dir if code_dir and code_dir.exists() else None,
    )
    if restored:
        yield _sse({
            "type": "info",
            "stage": "compile",
            "message": (
                "Kept original script names (a new peripheral name does not "
                "rename the file): " + ", ".join(restored)
            ),
        })
        c_sources = sorted(
            p for p in gen_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _SOURCE_EXTS
        )

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

    # ---- 2b. HEX / VirtuosoNext SDK discovery ------------------------------
    # If a HEX SDK is available (env HEX_SDK_DIR, sibling folder, or an
    # install under ~/HEX2), pull in its public include dirs and RTOS
    # preprocessor defines so user code that #includes <L1_api.h> et al.
    # can actually compile in the per-file gate.  Without this every
    # generated file that touches the RTOS API fails on the very first
    # header lookup and the fix loop can never make forward progress.
    hex_sdk = _discover_hex_sdk(repo_dir=repo_dir, cross_hint=sandbox_cc.split()[0])
    if hex_sdk is not None:
        yield _sse({
            "type": "info",
            "stage": "compile",
            "message": (
                "HEX SDK detected: "
                + hex_sdk.sdk_root.name
                + f" (platform={hex_sdk.platform}, variant={hex_sdk.variant}, "
                + f"CO={hex_sdk.compiler_opts})."
            ),
        })
        # If the toolchain the user's build files nominated doesn't match
        # the SDK's platform (e.g. the repo has no Makefile so we defaulted
        # to native gcc but the SDK only ships arm-cortex-a9 libs), swap
        # in the SDK's cross-compiler hint provided it is actually on PATH.
        if hex_sdk.is_cross and not is_cross and hex_sdk.cc_hint:
            preflight = subprocess.run(
                ["which", hex_sdk.cc_hint], capture_output=True, text=True,
                timeout=5,
            )
            if preflight.returncode == 0 and preflight.stdout.strip():
                sandbox_cc = f"{hex_sdk.cc_hint} -std=gnu99"
                is_cross = True
                yield _sse({
                    "type": "info",
                    "stage": "compile",
                    "message": (
                        f"Switched compiler to {sandbox_cc} to match "
                        f"the HEX SDK platform '{hex_sdk.platform}'."
                    ),
                })

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

    # ---- 3c. Optional full repo sweep -------------------------------------
    # COMPILE_INCLUDE_SWEEP_REPO=1: add every header-bearing directory under
    # the uploaded ZIP. Appended AFTER the demand-driven dirs above so
    # gen_dir / code_dir / resolved shims keep priority and a repo copy of a
    # regenerated header cannot shadow the newly generated one.
    swept_dirs: list[Path] = []
    swept_toxic: list[Path] = []
    swept_capped = 0
    if COMPILE_INCLUDE_SWEEP_REPO and has_repo and repo_dir and repo_dir.exists():
        for d in _collect_include_dirs(repo_dir):
            if d in include_dirs or d in swept_dirs:
                continue
            # The sysroot guard stays on: without it a vendored
            # arm-*-eabi/include on the -I path shadows libc string.h
            # and every file fails with unrelated type errors.
            if _is_toxic_include_dir(d):
                swept_toxic.append(d)
                continue
            swept_dirs.append(d)

        if (COMPILE_INCLUDE_SWEEP_MAX_DIRS > 0
                and len(swept_dirs) > COMPILE_INCLUDE_SWEEP_MAX_DIRS):
            swept_capped = len(swept_dirs) - COMPILE_INCLUDE_SWEEP_MAX_DIRS
            # _collect_include_dirs already sorts shallow-first, so the
            # truncation drops the deepest (least likely) dirs.
            swept_dirs = swept_dirs[:COMPILE_INCLUDE_SWEEP_MAX_DIRS]

        include_dirs.extend(swept_dirs)
        log.info(
            "per_file_compile: repo sweep added %d include dirs "
            "(%d toxic skipped, %d dropped by cap)",
            len(swept_dirs), len(swept_toxic), swept_capped,
        )

    # ---- 3d. HEX SDK include roots + defines -----------------------------
    # Add the SDK's public ``include/`` on -I so `<L1_api.h>` resolves,
    # and every -D from the SDK's RTOS.cmake so preprocessor
    # conditionals (VIRTUOSO_NEXT, L1_LOCAL_PTR_SIZE, ARM_CORTEX_A9 etc.)
    # evaluate correctly.  Appended AFTER the project-local dirs so a
    # user-supplied header of the same name can still shadow the SDK.
    hex_extra_flags: list[str] = []
    if hex_sdk is not None:
        for d in hex_sdk.include_dirs:
            if d.resolve() not in include_dirs:
                include_dirs.append(d.resolve())
        hex_extra_flags = list(hex_sdk.defines)
        # Angle-bracket headers that are not at the SDK include root
        # (board/MCP_P3L/core_0/L1_CoreSP.h) or that differ in case
        # (L1_soCSP.h vs L1_SoCSP.h) still fail with "No such file"
        # even after the root -I is added. Resolve them against the
        # SDK tree and, for the two Visual Designer outputs that are
        # not in the SDK at all, drop a stand-in into the shim dir.
        angle_needs = _angle_includes_in_tree(gen_dir)
        if has_repo and repo_dir and repo_dir.exists():
            angle_needs |= _angle_includes_in_tree(repo_dir)
        for rel in sorted(angle_needs):
            if _header_already_on_path(rel, include_dirs):
                continue
            found = _find_sdk_header(hex_sdk, rel)
            if found is not None and found.name == Path(rel).name:
                root = found
                for _seg in rel.split("/"):
                    root = root.parent
                resolved = root.resolve()
                if resolved not in include_dirs:
                    include_dirs.append(resolved)
                continue
            if found is not None:
                shimmed = _materialise_header_shim(shim_dir, rel, found)
                if shimmed is not None:
                    s = shim_dir.resolve()
                    if s not in include_dirs:
                        include_dirs.append(s)
                continue
            local_dirs = [gen_dir]
            if code_dir is not None and code_dir.exists():
                local_dirs.append(code_dir)
            # gen_dir holds plImu20Msg.h; the original include spells
            # plImu20msg.h. Match that case here, then fall back to the repo.
            local = _find_local_header(rel, local_dirs)
            if local is None and has_repo and repo_dir is not None and repo_dir.exists():
                leaf_l = Path(rel).name.lower()
                try:
                    for path in repo_dir.rglob("*"):
                        if path.is_file() and path.name.lower() == leaf_l:
                            local = path
                            break
                except OSError:
                    local = None
            if local is not None:
                if local.name == Path(rel).name:
                    root = local.parent.resolve()
                    if root not in include_dirs:
                        include_dirs.append(root)
                else:
                    shimmed = _materialise_header_shim(shim_dir, rel, local)
                    if shimmed is not None:
                        s = shim_dir.resolve()
                        if s not in include_dirs:
                            include_dirs.append(s)
                continue
            body = _render_generated_header(
                rel, [p for p in (repo_dir, gen_dir) if p is not None],
            )
            if not body:
                continue
            try:
                shim_dir.mkdir(parents=True, exist_ok=True)
                dest = shim_dir / Path(rel).name
                dest.write_text(body)
            except OSError:
                continue
            s = shim_dir.resolve()
            if s not in include_dirs:
                include_dirs.append(s)

    n_found = sum(
        1 for r in resolutions.values()
        if r.get("status") in ("found", "found_via_shim", "found_case_normalized")
    )
    n_missing = sum(
        1 for r in resolutions.values()
        if r.get("status") in ("missing", "skipped_toxic_only",
                                "shim_create_failed", "shim_write_failed")
    )

    if COMPILE_INCLUDE_SWEEP_REPO:
        scope_msg = (
            f"Resolved {len(include_dirs)} include directories "
            f"(FULL REPO SWEEP enabled: {len(swept_dirs)} header-bearing "
            f"directories from the uploaded ZIP added after gen_dir / "
            f"original_code / demand-resolved shims"
            + (f", {len(swept_toxic)} toolchain-sysroot dirs excluded"
               if swept_toxic else "")
            + (f", {swept_capped} dropped by the "
               f"{COMPILE_INCLUDE_SWEEP_MAX_DIRS}-dir cap"
               if swept_capped else "")
            + ")."
        )
    else:
        scope_msg = (
            f"Resolved {len(include_dirs)} include directories "
            f"(scoped to newly generated + verified scripts and "
            f"demand-driven project-local headers — "
            f"repository ZIP is NOT swept because "
            f"COMPILE_INCLUDE_SWEEP_REPO=0; only the specific headers the "
            f"generated code quote-includes are pulled in, and only from "
            f"non-toolchain locations). Unset it to restore the default "
            f"full-repo sweep."
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

    # A raw brace count stays balanced when both sides of an #ifdef open a
    # block and a later '}' is extra on the branch GCC actually compiles.
    # Strip that closer before the first compile, and snapshot the repaired
    # file so a stalled fix attempt does not put the '}' back.
    brace_defines = _defines_from_flags(hex_extra_flags)
    brace_fixed: list[str] = []
    for path in list(gen_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in (_SOURCE_EXTS | _HEADER_EXTS):
            continue
        try:
            current = path.read_text(errors="replace")
        except OSError:
            continue
        fixed, fence_notes = strip_markdown_fence_lines(current)
        braced, notes = repair_stray_closing_braces(fixed, brace_defines)
        notes = fence_notes + notes
        if braced != current and notes:
            path.write_text(braced)
            brace_fixed.append(f"{path.name}: {notes[0]}")
    if brace_fixed:
        yield _sse({
            "type": "info",
            "stage": "compile",
            "message": "Removed stray file-scope braces and markdown fences before compile — "
            + "; ".join(brace_fixed),
        })

    # ---- 4. Snapshot the pre-compile-gate generated files (for stall reset)
    snapshots: dict[str, str] = {}
    for cs in c_sources:
        snapshots[cs.name] = _read_text_safe(cs)
    for hp in gen_dir.iterdir():
        if hp.is_file() and hp.suffix.lower() in _HEADER_EXTS:
            snapshots[hp.name] = _read_text_safe(hp)

    # ---- 5. Per-file compile + agentic fix loop ----------------------------
    uploaded_names = {p.name for p in code_dir.iterdir() if p.is_file()} if code_dir.exists() else set()
    terminated = False
    per_file_results: dict[str, dict] = {}
    per_file_attempts: dict[str, list[dict]] = {}

    fix_system = (
        "You are an expert C programmer. The transformed C file failed "
        "per-file compilation (one .c → .o pass with the project's "
        "toolchain).\n\n"
        "You will receive:\n"

        "- The CURRENT transformed code that failed to compile\n"
        "- The project headers the .c includes or that declare a type "
        "named in the errors (they may not share the .c file's name)\n"
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
        "1. A header must be the COMPLETE file. A long .c may be a "
        "unified diff of only the lines that must change, inside a "
        "```diff fence. Do not use ellipses, '...', or '// unchanged' "
        "inside a full file.\n"
        "2. If you edit only the .c, output exactly one fenced block "
        "wrapped in ```c ... ```.\n"
        "3. If you edit a header — even as the only file — or more than "
        "one file, precede each block with a single-line filename header:\n"
        "       ### <filename>\n"
        "   e.g. `### IMU.h` then ```c ... ```, or `### IMU.c` then "
        "```c ... ``` then `### proto.h` then ```c ... ```. The filename "
        "MUST end in `.c` or `.h` and match a file you were given.\n"
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
        "already work — only add or rename what the errors demand. The "
        "project headers were transformed to the Target ICD first: keep their "
        "Target-ICD layouts and values. Never fix an error by bringing back a "
        "Source-ICD field, value or message.\n"
        "C. NEVER silently delete or roll back ICD-mandated changes "
        "from the .c to make the compile pass. If the error says a "
        "struct/union member is missing, ADD that member to the struct "
        "in its declaring header. If the error says a function is "
        "undeclared, ADD a prototype to the header (or include the "
        "header that already declares it).\n"
        "D. When the error says `'X' has no member named 'Y'`, or reports "
        "an unknown type, ADD the member or typedef to the declaration "
        "that already exists. Do not declare that name a second time "
        "and do not alias a new `*_ext` struct back to it. That header "
        "is named in the prompt and may not share the .c file's name. "
        "Output the COMPLETE header. If the error is in a struct defined "
        "in the .c, or an assignment to a const object, change the .c "
        "with a unified diff when the .c is long. Leave the .c out of "
        "the reply when it does not need to change.\n"
        "E. The .h must keep its include guards (`#ifndef <BASE>_H` / "
        "`#define <BASE>_H` ... `#endif`). The .c must keep its "
        "`#include \"<base>.h\"` (or equivalent) if it had one.\n"
        "F. Peripheral variations are intentional. If a .c file declares "
        "MULTIPLE per-variation structs for the same peripheral (each with "
        "its own banner comment), PRESERVE all of them inside the original "
        "filename. Do NOT rename the script or an `#include` to the new "
        "peripheral (keep `plImu20msg.h`, do not emit `plImu15Msg.h`). Do "
        "NOT rename EV_ / FIFO_ / HUB_ identifiers to insert that peripheral "
        "number. The `### <filename>` marker, when you emit one, MUST be a "
        "file you were given: the .c under repair, its same-stem header, or "
        "another project header listed in the prompt. Do not invent a new "
        "peripheral filename."
    )

    for cs in c_sources:
        fname = cs.name
        base = cs.stem
        obj_path = gen_dir / f"{base}.o"
        original_code = ""
        orig_repo_match = (code_dir / fname)
        if orig_repo_match.exists():
            try:
                # NB: UnicodeDecodeError is a ValueError, not an OSError, so
                # the enclosing `except OSError` would not catch a cp1252 file.
                original_code = _read_text_safe(orig_repo_match)
            except OSError:
                original_code = ""
        elif has_repo and repo_dir and repo_dir.exists():
            for m in repo_dir.rglob(fname):
                if m.is_file():
                    try:
                        original_code = _read_text_safe(m)
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
        # Headers this file's fixer is allowed to edit. Starts with the
        # same-stem companion; grows once compiler errors name a type
        # that lives in a different project header.
        editable_header_names: set[str] = set()
        if companion_h is not None:
            editable_header_names.add(companion_h.name)

        yield _sse({
            "type": "info",
            "stage": "compile",
            "file": fname,
            "message": f"Compiling {fname}…",
        })

        # Project headers gcc cannot find anywhere, which the fixer may create.
        creatable_headers: set[str] = set()

        for attempt in range(1, max_fix_attempts + 1):
            if not terminated and should_stop is not None and should_stop():
                terminated = True
                yield _sse({
                    "type": "info",
                    "stage": "compile",
                    "message": (
                        "Stop requested — no further fix passes; remaining "
                        "files get one compile and the report is written."
                    ),
                })
            ok, cmd, output = _run_compile(
                sandbox_cc, cs, obj_path, include_dirs, compile_timeout,
                cwd=gen_dir,
                extra_flags=hex_extra_flags or None,
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

            if attempt == max_fix_attempts or terminated:
                yield _sse({
                    "type": "compile_file_result",
                    "stage": "compile",
                    "file": fname,
                    "success": False,
                    "attempt": attempt,
                    "message": (
                        f"{fname} still fails to compile — stopped by the user."
                        if terminated else
                        f"{fname} still fails to compile after "
                        f"{max_fix_attempts} agentic fix attempts."
                    ),
                })
                result = {
                    "status": "failed",
                    "attempts": attempt,
                    "last_output": output[-2000:],
                }
                if terminated:
                    result["reason"] = "stopped by the user"
                break

            # Stall reset: rewind to the pre-gate snapshot before re-prompting,
            # so the LLM doesn't compound its prior misguided edits.
            if stall >= 1:
                restored = []
                if fname in snapshots:
                    cs.write_text(snapshots[fname])
                    restored.append(fname)
                for header_name in sorted(editable_header_names):
                    if header_name in snapshots:
                        (gen_dir / header_name).write_text(snapshots[header_name])
                        restored.append(header_name)
                stall = 0
                yield _sse({
                    "type": "info",
                    "stage": "compile",
                    "file": fname,
                    "message": (
                        "Same compile errors persisted — reset "
                        + ", ".join(restored)
                        + " to the pre-compile-gate snapshot before re-prompting."
                    ),
                })

            # ---- Agentic fix prompt --------------------------------------
            pre_fix_c = _read_text_safe(cs)

            file_repo_ctx = ""
            if (
                not COMPILE_FIX_LEAN_PROMPT
                and has_repo and repo_dir and repo_dir.exists()
            ):
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
            declaration_errors = bool(
                err_struct["missing_members"]
                or err_struct["unknown_types"]
                or err_struct["implicit_decls"]
                or err_struct["conflicting_types"]
            )
            fix_headers = _collect_fix_headers(
                pre_fix_c, err_struct, gen_dir, code_dir, companion_h,
            )
            # A project header gcc cannot find on ANY -I path (not libc) can
            # only be fixed by writing it.
            for missing in err_struct.get("missing_headers") or []:
                base = Path(missing.strip()).name
                if (
                    Path(base).suffix.lower() in _HEADER_EXTS
                    and base.lower() not in _LIBC_ANGLE_SKIP
                    and not (gen_dir / base).exists()
                ):
                    creatable_headers.add(base)
            for header_path in fix_headers:
                editable_header_names.add(header_path.name)
                if header_path.name not in snapshots:
                    snapshots[header_path.name] = _read_text_safe(header_path)
            header_names = [header_path.name for header_path in fix_headers]

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
                banner_headers = header_names or [
                    companion_h.name if companion_h else fname.replace(".c", ".h")
                ]
                header_blocks = "\n\n".join(
                    f"  ### {header_name}\n"
                    "  ```c\n"
                    "  ...the FULL header with the missing declarations"
                    " / typedefs / struct members ADDED...\n"
                    "  ```"
                    for header_name in banner_headers
                )
                sec_focus_banner = (
                    "## CRITICAL — READ FIRST\n"
                    "Your PREVIOUS rewrite attempt did not edit a "
                    "declaring header. The compiler errors below STILL "
                    "list header-side problems that REQUIRE editing the .h, "
                    "not a rewrite of the .c.\n\n"
                    "Symptoms the previous attempt failed to address:\n"
                    + "\n".join(f"  - {d}" for d in prev_missed_details)
                    + "\n\n"
                    "On this attempt you MUST output the COMPLETE header"
                    "(s) below. A header-only reply is valid. Do not "
                    "rewrite the .c unless a compiler error is a mistake "
                    "in the .c itself — a truncated .c is rejected.\n\n"
                    + header_blocks
                    + "\n\n"
                    "Stop polishing dates, comments, prose, or "
                    "ICD-cosmetic magic numbers — those are zero-priority "
                    "until the compile is green.\n"
                )

            # The pre-ICD code is deliberately NOT shown: it is exactly what a
            # fixer copies back in to get a clean compile, which puts Source-ICD
            # fields and names back into the Target-ICD deliverable.
            sec_original = ""
            sec_current = (
                f"## Current Transformed Code ({fname})\n"
                f"This version failed the per-file compile:\n"
                f"```c\n{pre_fix_c}\n```"
            )
            header_chunks: list[str] = []
            for header_path in fix_headers:
                body = _read_text_safe(header_path)
                if len(body) > _FIX_HEADER_CHARS:
                    body = (
                        body[:_FIX_HEADER_CHARS]
                        + "\n/* ... header truncated for the fix prompt ... */\n"
                    )
                header_chunks.append(
                    f"## Project header you may edit ({header_path.name})\n"
                    f"```c\n{body}\n```"
                )
            sec_header = "\n\n".join(header_chunks)
            pending_create = sorted(
                h for h in creatable_headers if not (gen_dir / h).exists()
            )
            if pending_create:
                sec_header = (
                    "## Missing project header(s) you may CREATE\n"
                    "The compiler cannot find these anywhere on the include "
                    "path. Write each as a new COMPLETE header, preceded by its "
                    "`### <filename>` marker, with an include guard and only "
                    f"the declarations `{fname}` needs from it. Keep the "
                    "`#include` in the .c as it is.\n"
                    + "\n".join(f"  - ### {h}" for h in pending_create)
                    + ("\n\n" + sec_header if sec_header else "")
                )
            if declaration_errors and header_names:
                listed = "\n".join(f"  - ### {name}" for name in header_names)
                sec_header = (
                    "## Headers the compiler errors belong to\n"
                    "These project headers are included by the .c or "
                    "already mention a type named in the errors. They "
                    "may not share the .c file's name. Add the missing "
                    "member, typedef, or prototype there. Output each "
                    "COMPLETE header preceded by its `###` filename. A "
                    "header-only reply is valid. Copy the header and add "
                    "only the types, members, and macros named in the "
                    "compiler errors. Do not invent extra structs. Do not "
                    "rewrite `"
                    + fname
                    + "` unless an error is a mistake in that .c file.\n"
                    + listed
                    + ("\n\n" + sec_header if sec_header else "")
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
                if repo_knowledge and not COMPILE_FIX_LEAN_PROMPT else ""
            )
            sec_change = f"## Change Specification\n{change_spec}"
            # HEX SDK context (only emitted when discovered). Kept
            # top-priority-but-optional: it always fits when present
            # because :func:`render_llm_context` self-caps at 6000 chars.
            sec_hex_sdk = ""
            if hex_sdk is not None:
                try:
                    from api.hex_sdk import render_llm_context  # type: ignore
                except ImportError:
                    from hex_sdk import render_llm_context  # type: ignore
                sec_hex_sdk = render_llm_context(hex_sdk, max_chars=6000)
            example_header = (
                header_names[0] if header_names
                else (companion_h.name if companion_h else fname.replace(".c", ".h"))
            )
            sec_instr = (
                "## Output format reminder\n"
                "Output ONLY the complete file(s) inside ```c fences. "
                "A header — even when it is the only file you edit — "
                "must be prefixed with `### <filename>` (e.g. `### "
                f"{example_header}` then ```c ... ```). The .c, when "
                f"you also rewrite it, is `### {fname}`. The filename "
                "MUST end in `.c` or `.h` and MUST be one of the files "
                "above. Do NOT add any other `###` headings anywhere "
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
                    # Declaring headers go before the full .c so a long
                    # source file cannot push them out of the budget.
                    ("header", sec_header, 0),
                    ("original", sec_original, 0),
                    ("current", sec_current, 0),
                    ("command", sec_cmd, 0),
                    ("errors", sec_errors, 0),
                    # The structured punch list is a tiny, high-signal
                    # summary of the raw error log — keep it at top
                    # priority so it always fits in the prompt budget.
                    ("punchlist", sec_punchlist, 0),
                    ("resolutions", sec_resolutions, 0),
                    ("hex_sdk", sec_hex_sdk, 0),
                    ("instructions", sec_instr, 0),
                    ("repo_ctx", sec_repo_ctx, 1),
                    ("knowledge", sec_knowledge, 2),
                    # change_spec is small and high-signal; with repo_ctx and
                    # knowledge suppressed it now comfortably fits, so it is
                    # promoted ahead of them rather than being dropped last.
                    ("change_spec", sec_change, 1 if COMPILE_FIX_LEAN_PROMPT else 3),
                ],
                # Cap against COMPILE_FIX_MAX_INPUT_TOKENS, which already
                # reserves a full MAX_OUTPUT_TOKENS reply plus the safety
                # margin, so _clamp_chat_request never has to shrink
                # max_tokens and truncate the fix mid-fence.
                max_input_tokens=min(
                    MAX_INPUT_TOKENS, COMPILE_FIX_MAX_INPUT_TOKENS,
                ) - sys_tokens,
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
                # 4096 tokens cuts a real header mid-comment. The continuation
                # pass then reopens a markdown fence inside that comment
                # and the rewrite is rejected. Ask for a larger reply; the
                # context clamp shrinks it when the prompt is already full.
                fix_output = _call_llm_complete(
                    fix_system, fix_prompt,
                    max_tokens=max(MAX_OUTPUT_TOKENS, _COMPILE_FIX_OUTPUT_TOKENS),
                    max_passes=3,
                )
                # Persist the raw reply so a discarded fix can be diagnosed
                # (truncated mid-fence vs. genuinely malformed vs. refused).
                # Without this the output is lost and the only signal is
                # "no triple-fenced code blocks found".
                try:
                    _raw_dir = session_dir / "compile_fix_raw"
                    _raw_dir.mkdir(exist_ok=True)
                    (_raw_dir / f"{fname}.attempt{attempt}.txt").write_text(
                        fix_output, encoding="utf-8",
                    )
                except Exception:
                    pass
                log.info(
                    "per_file_compile: fix output for %s attempt %d: "
                    "%d chars, %d fenced block(s)",
                    fname, attempt, len(fix_output), fix_output.count("```") // 2,
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
            allowed.update(editable_header_names)
            allowed.update(creatable_headers)

            parsed = _extract_per_file_blocks(fix_output)
            attributed: dict[str, str] = {}
            for parsed_name, parsed_body in parsed.items():
                target = _attribute_block_to_original_script(parsed_name, allowed)
                if target is None:
                    attributed.setdefault(parsed_name, parsed_body)
                    continue
                # An exact filename wins over a peripheral-renamed label
                # for the same original script.
                if target not in attributed or parsed_name == target:
                    attributed[target] = parsed_body
            parsed = attributed
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
                            header_names=sorted(editable_header_names),
                        )
                        if guess and guess not in new_files:
                            new_files[guess] = body
                # Fallback B: a single unmarked block is the .c, unless
                # it is a header (include guard). A header must not
                # overwrite the .c.
                if not new_files:
                    single = _strip_filename_marker_leakage(
                        _extract_fenced(fix_output, "c").strip()
                    )
                    if single:
                        guess = _guess_filename_for_block(
                            single, c_name=fname,
                            h_name=(companion_h.name if companion_h else None),
                            header_names=sorted(editable_header_names),
                        )
                        if guess:
                            new_files = {guess: single}
                        elif not re.search(r"#\s*ifndef\b", single):
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
                creating = (
                    target_name in creatable_headers
                    and not (gen_dir / target_name).exists()
                )
                name_anchor = ""
                if code_dir is not None:
                    anchor_path = code_dir / target_name
                    if anchor_path.is_file():
                        name_anchor = _read_text_safe(anchor_path)
                if not name_anchor and target_name == fname:
                    name_anchor = original_code
                if _looks_like_unified_diff(new_code):
                    base_text = (
                        _read_text_safe(gen_dir / target_name)
                        or snapshots.get(target_name, "")
                    )
                    patched = _apply_unified_diff(base_text, new_code)
                    if patched is None:
                        decisions.append({
                            "target": target_name,
                            "decision": "rejected_incomplete",
                            "reason": "unified diff did not apply",
                        })
                        continue
                    new_code = patched
                if name_anchor:
                    new_code = preserve_original_script_names(name_anchor, new_code)
                new_code, _fence_notes = strip_markdown_fence_lines(new_code)
                new_code, _brace_notes = repair_stray_closing_braces(
                    new_code, _defines_from_flags(hex_extra_flags),
                )
                complete = (
                    _looks_complete_header(new_code) if creating
                    else _looks_complete_c_file(new_code, ref, target_name)
                )
                if not complete:
                    detail = (
                        "new header is not a complete header file" if creating
                        else _incomplete_file_reason(new_code, ref, target_name)
                    )
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
                prev_text = _read_text_safe(target_path) if target_path.exists() else ""
                target_path.write_text(new_code)
                if creating:
                    editable_header_names.add(target_name)
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
            if touched_h and (
                err_struct.get("missing_members")
                or err_struct.get("unknown_types")
                or err_struct.get("undeclared")
                or err_struct.get("readonly_assigns")
            ):
                repair_notes = realize_compiler_fixes(
                    c_path=cs,
                    headers=fix_headers,
                    err_struct=err_struct,
                    compiler_output=output,
                    include_dirs=include_dirs,
                )
                if repair_notes:
                    for repaired in [cs, *fix_headers]:
                        if repaired.is_file():
                            snapshots[repaired.name] = _read_text_safe(repaired)
                    yield _sse({
                        "type": "info",
                        "stage": "compile",
                        "file": fname,
                        "message": (
                            "Placed compiler-named fields on the existing "
                            "declarations — " + "; ".join(repair_notes)
                        ),
                    })
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
                    _read_text_safe(cs), repo_dir, original_code, fname,
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
        *(["Stopped:            yes — compile loop stopped by the user"]
          if terminated else []),
        f"Quote-includes:     {len(quote_needs)} parsed, "
        f"{n_found} resolved, {n_missing} unresolved"
        if quote_needs else "Quote-includes:     none",
    ]
    if hex_sdk is not None:
        rep.extend([
            "",
            "-" * 65,
            "HEX / VirtuosoNext SDK",
            "-" * 65,
        ])
        rep.extend(f"  {line}" for line in hex_sdk.summary_lines())
        rep.append(f"  extra flags : {' '.join(hex_extra_flags)}")
    rep.extend([
        "",
        "-" * 65,
        "INCLUDE PATH (gen_dir + code_dir + demand-driven project-local "
        "headers from repo; toolchain sysroots filtered out"
        + (
            "; HEX SDK include roots appended"
            if hex_sdk is not None else ""
        )
        + ")",
        "-" * 65,
    ])
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
            f"Status:   {result.get('status', 'unknown').upper()}"
            + (f" ({result['reason']})" if result.get("reason") else ""),
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
    # ```c / ### file.c / ```c / <code> leaves the inner opener behind
    # once the filename line is gone. That opener is not C.
    while lines and re.match(r"^```[ \t]*[A-Za-z0-9_+-]*[ \t]*$", lines[0]):
        lines.pop(0)
        changed = True
        while lines and not lines[0].strip():
            lines.pop(0)
    if not changed:
        return body
    return "\n".join(lines)


_DIRECTIVE_RE = re.compile(r"^#\s*([A-Za-z]+)\b\s*(.*?)\s*$")
_DEFINED_RE = re.compile(
    r"^(!)?\s*defined\s*(?:\(\s*([A-Za-z_]\w*)\s*\)|([A-Za-z_]\w*))\s*$"
)
_IDENT_COND_RE = re.compile(r"^([A-Za-z_]\w*)$")


def _defines_from_flags(flags: list[str] | None) -> dict[str, str]:
    """Turn compiler ``-DNAME`` / ``-DNAME=value`` flags into a macro map."""
    out: dict[str, str] = {}
    for flag in flags or []:
        if not flag.startswith("-D") or len(flag) < 3:
            continue
        body = flag[2:]
        if "=" in body:
            name, value = body.split("=", 1)
            out[name] = value
        else:
            out[body] = "1"
    return out


def _condition_is_true(expr: str, defines: dict[str, str]) -> Optional[bool]:
    """Evaluate a simple ``#if`` expression. ``None`` means "leave it alone"."""
    expr = (expr or "").strip()
    if expr in {"0", "0L", "0U", "0UL"}:
        return False
    if expr in {"1", "1L", "1U", "1UL"}:
        return True
    defined = _DEFINED_RE.match(expr)
    if defined:
        name = defined.group(2) or defined.group(3)
        truth = name in defines
        return (not truth) if defined.group(1) else truth
    ident = _IDENT_COND_RE.match(expr)
    if ident:
        name = ident.group(1)
        if name not in defines:
            return False
        return defines[name] not in {"", "0"}
    return None


_FENCE_ONLY_LINE = re.compile(r"^\s*```[A-Za-z0-9_+-]*\s*$")


def strip_markdown_fence_lines(text: str) -> tuple[str, list[str]]:
    """Drop markdown fence lines an LLM left inside a C translation unit.

    A line whose only text is ````` `` or `````c`` is not C; GCC reports
    it as ``stray '\\`' in program``. When the fence splits a statement,
    the line above it is often a truncated copy of the line below
    (``foo.bar`` then the fence then ``foo.bar = 1;``). That truncated
    line is dropped as well. A fence that shares its line with other
    tokens is left alone.
    """
    if not text or "```" not in text:
        return text, []
    ends_nl = text.endswith("\n")
    raw = text.splitlines()
    out: list[str] = []
    dropped_fences = 0
    dropped_prefixes = 0
    i = 0
    while i < len(raw):
        if not _FENCE_ONLY_LINE.match(raw[i]):
            out.append(raw[i])
            i += 1
            continue
        dropped_fences += 1
        i += 1
        while i < len(raw) and _FENCE_ONLY_LINE.match(raw[i]):
            dropped_fences += 1
            i += 1
        if i < len(raw) and out:
            prev = out[-1].strip()
            nxt = raw[i].strip()
            if (
                prev
                and nxt.startswith(prev)
                and len(nxt) > len(prev)
                and not prev.endswith((";", "{", "}", ",", "\\"))
            ):
                out.pop()
                dropped_prefixes += 1
    if not dropped_fences:
        return text, []
    body = "\n".join(out)
    if ends_nl:
        body += "\n"
    notes = [f"removed {dropped_fences} markdown fence line(s)"]
    if dropped_prefixes:
        notes.append(
            f"removed {dropped_prefixes} statement(s) split by a fence"
        )
    return body, notes


def repair_stray_closing_braces(
    text: str,
    predefined: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    """Remove a file-scope ``}`` that only balances a brace in an inactive branch.

    A raw ``count('{') == count('}')`` check sees both sides of
    ``#ifdef`` / ``#else``. When each branch opens a block, the extra
    closer at the end of the function looks balanced and every stage
    before the compiler accepts the file. GCC then reports
    ``expected identifier or '(' before '}' token``.

    This walk follows the active branch (``#define`` in the file, plus
    any ``-D`` macros). A line whose only code is ``}`` while the active
    branch is already at file scope is dropped. Anything else — a
    missing brace, or a condition this walker cannot evaluate — is left
    untouched and described in the returned notes.
    """
    if not text:
        return text, []
    defines = dict(predefined or {})
    lines = text.splitlines(keepends=True)
    depth = 0
    in_block = False
    # Each frame: active for this branch, whether any branch was taken,
    # whether the parent region was active.
    stack: list[tuple[bool, bool, bool]] = []
    uncertain = False
    drop: list[int] = []
    hard_error = False

    def active() -> bool:
        return stack[-1][0] if stack else True

    for idx, line in enumerate(lines):
        logical = line[:-1] if line.endswith("\n") else line
        if logical.endswith("\r"):
            logical = logical[:-1]
        directive = None if in_block else _DIRECTIVE_RE.match(logical.strip())
        if directive:
            kind = directive.group(1)
            rest = directive.group(2).split("//", 1)[0].strip()
            parent_active = stack[-1][0] if stack else True
            if kind in {"ifdef", "ifndef"}:
                name = rest.split()[0] if rest.split() else ""
                cond = (name in defines) if kind == "ifdef" else (name not in defines)
                if not name:
                    uncertain = True
                    cond = True
                taken = bool(parent_active and cond)
                stack.append((taken, taken, parent_active))
                continue
            if kind == "if":
                cond = _condition_is_true(rest, defines)
                if cond is None:
                    uncertain = True
                    cond = True
                taken = bool(parent_active and cond)
                stack.append((taken, taken, parent_active))
                continue
            if kind == "elif" and stack:
                branch_active, taken, parent = stack[-1]
                cond = _condition_is_true(rest, defines)
                if cond is None:
                    uncertain = True
                    cond = False
                now = bool(parent and not taken and cond)
                stack[-1] = (now, taken or now, parent)
                continue
            if kind == "else" and stack:
                _branch_active, taken, parent = stack[-1]
                now = bool(parent and not taken)
                stack[-1] = (now, True, parent)
                continue
            if kind == "endif" and stack:
                stack.pop()
                continue
            if active() and kind == "define":
                name = re.match(r"([A-Za-z_]\w*)", rest)
                if name:
                    after = rest[name.end():].lstrip()
                    if after.startswith("("):
                        defines[name.group(1)] = "1"
                    else:
                        defines[name.group(1)] = after.split("/*", 1)[0].strip()
                continue
            if active() and kind == "undef":
                name = rest.split()[0] if rest.split() else ""
                defines.pop(name, None)
                continue
            continue

        if not active():
            # Still track block comments so a comment opened in a skipped
            # branch does not hide the code that follows #endif.
            i = 0
            while i < len(logical):
                if in_block:
                    end = logical.find("*/", i)
                    if end < 0:
                        break
                    in_block = False
                    i = end + 2
                    continue
                if logical.startswith("/*", i):
                    end = logical.find("*/", i + 2)
                    if end < 0:
                        in_block = True
                        break
                    i = end + 2
                    continue
                i += 1
            continue

        saw_other = False
        saw_open = False
        stray = False
        i = 0
        while i < len(logical):
            if in_block:
                end = logical.find("*/", i)
                if end < 0:
                    break
                in_block = False
                i = end + 2
                continue
            if logical.startswith("//", i):
                break
            if logical.startswith("/*", i):
                end = logical.find("*/", i + 2)
                if end < 0:
                    in_block = True
                    break
                i = end + 2
                continue
            ch = logical[i]
            if ch in "\"'":
                quote = ch
                i += 1
                while i < len(logical) and logical[i] != quote:
                    i += 2 if logical[i] == "\\" else 1
                saw_other = True
                i += 1
                continue
            if ch == "{":
                depth += 1
                saw_open = True
            elif ch == "}":
                if depth == 0 and not saw_other and not saw_open:
                    stray = True
                elif depth == 0:
                    hard_error = True
                else:
                    depth -= 1
            elif not ch.isspace():
                saw_other = True
            i += 1
        if stray and not saw_other and not saw_open:
            drop.append(idx)
        if hard_error:
            return text, [
                "closing brace at file scope is not alone on its line; "
                "left unchanged for the compile-fix pass"
            ]

    if uncertain and drop:
        return text, [
            "preprocessor condition could not be evaluated; "
            "stray '}' was not removed"
        ]
    if depth != 0:
        return text, [
            f"active-branch braces are unbalanced (depth {depth}); "
            "file was not rewritten"
        ]
    if not drop:
        return text, []
    kept = [line for i, line in enumerate(lines) if i not in set(drop)]
    repaired = "".join(kept)
    if repaired and not repaired.endswith("\n"):
        repaired += "\n"
    return repaired, [
        f"removed {len(drop)} stray file-scope '}}' "
        "(balanced only by a brace in an inactive #if/#else branch)"
    ]


# A closing fence is a line of backticks only. ```c is an opening fence;
# treating it as a closer swallows the .c body when the model writes
# ```c / ### boImuIn.c / ```c / <code> / ```.
_FENCE_RE = re.compile(
    r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\n(.*?)\n```[ \t]*(?:\n|\Z)",
    flags=re.DOTALL,
)


def _fenced_spans(text: str) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` for every triple-fenced region in
    *text*. Spans cover the opening ``` and the closing ``` so any
    position strictly inside the span is "inside a code fence".
    Used by :func:`_extract_per_file_blocks` to ignore `###` markers
    the LLM nested inside its own code fence (the IMU.c production
    failure: the LLM repeated `### IMU.c` *inside* its ``` block,
    which confused the strict parser into returning empty).
    """
    return [(m.start(), m.end()) for m in _FENCE_RE.finditer(text)]


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
        fence = _FENCE_RE.search(chunk)
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
    for m in _FENCE_RE.finditer(text):
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


def _looks_like_unified_diff(text: str) -> bool:
    """True when a fenced block is a unified diff rather than a full file."""
    lines = [line for line in (text or "").splitlines() if line.strip()][:12]
    if not lines:
        return False
    if any(line.startswith("@@") for line in lines):
        return True
    return any(line.startswith("--- ") for line in lines) and any(
        line.startswith("+++ ") for line in lines
    )


def _apply_unified_diff(original: str, diff_text: str) -> Optional[str]:
    """Apply a unified diff to ``original``. None when ``patch`` rejects it."""
    diff = (diff_text or "").strip() + "\n"
    if "@@" in diff and not diff.startswith("---"):
        diff = "--- a/file\n+++ b/file\n" + diff
    original_text = original if original.endswith("\n") or not original else original + "\n"
    with tempfile.TemporaryDirectory(prefix="icd_diff_") as tmp:
        src = Path(tmp) / "file"
        out = Path(tmp) / "out"
        patch_file = Path(tmp) / "change.diff"
        src.write_text(original_text)
        patch_file.write_text(diff)
        result = subprocess.run(
            [
                "patch", "--forward", "--batch",
                "-o", str(out), str(src), str(patch_file),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not out.is_file():
            return None
        return out.read_text()


def _guard_matches_header(body: str, header_name: str) -> bool:
    """True when an include guard in ``body`` spells ``header_name``.

    ``PL_IMU20_MSG_H`` and ``plImu20Msg.h`` match once punctuation is
    stripped. A same-stem companion (``IMU_H`` / ``IMU.h``) matches too.
    """
    stem_alnum = re.sub(r"[^A-Za-z0-9]", "", Path(header_name).stem).upper()
    if not stem_alnum:
        return False
    for guard in re.findall(r"#\s*ifn?def\s+(\w+)", body):
        guard_alnum = re.sub(r"[^A-Za-z0-9]", "", guard).upper()
        if guard_alnum.endswith("H") and len(guard_alnum) > len(stem_alnum):
            guard_alnum = guard_alnum[:-1]
        if guard_alnum == stem_alnum:
            return True
    return False


def _guess_filename_for_block(
    body: str,
    *,
    c_name: str,
    h_name: Optional[str],
    header_names: Optional[list[str]] = None,
) -> Optional[str]:
    """Decide whether a fenced block looks like the .c or a header.

    Used as a last-resort fallback when the LLM emitted code blocks but
    forgot to mark them with `### <filename>` headers.  Heuristics:

      - A typical .h has an include guard (``#ifndef <BASE>_H`` /
        ``#define <BASE>_H`` ... ``#endif``). The guard may belong to
        the same-stem companion or to another project header the .c
        includes.
      - A typical .c contains function bodies (``int foo(...) {``).
    """
    if not body or not c_name:
        return None
    candidates: list[str] = []
    if h_name:
        candidates.append(h_name)
    for name in header_names or []:
        if name not in candidates and name.lower().endswith(".h"):
            candidates.append(name)
    has_endif = re.search(r"#\s*endif\b", body) is not None
    if has_endif and candidates:
        matched = [
            name for name in candidates if _guard_matches_header(body, name)
        ]
        if len(matched) == 1:
            return matched[0]
    # Cheap "this looks like a .c" check: at least one function body
    # opener with `{` on a non-comment line.
    looks_like_c = bool(
        re.search(r"^\s*[A-Za-z_][\w\s\*]*\([^;]*\)\s*\{", body, re.MULTILINE)
    )
    if looks_like_c:
        return c_name
    return None
