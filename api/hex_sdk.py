"""HALCON HEX / VirtuosoNext RTOS SDK integration.

Auto-discovers a locally installed ``VisualDesigner-HEX-*`` SDK tree and
exposes the include roots, preprocessor defines, cross-compiler hints and
static-library link arguments the rest of the pipeline needs to compile and
link user code that targets the HEX RTOS.

Why this module exists
----------------------

The user's C sources typically look like this::

    #include <L1_api.h>
    #include <kernel/L1_kernel_api.h>
    #include <StdioHostService/L1_stdio_host_service_api.h>

Those angle-bracket includes are ONLY resolvable via the HEX SDK's
``targets/<platform>/include`` directory.  Without that directory on the
compiler's ``-I`` path every generated file fails with

    fatal error: L1_api.h: No such file or directory

before the LLM ever has a chance to reason about the ICD-mandated
changes.  Similarly the SDK's static libraries
(``libHEX_SP_CO0_Kernel.a`` etc.) must be on the linker path for the
final sandbox build to actually link.

Runtime discovery, not vendoring
--------------------------------

The SDK is ~415 MB (mostly ``.a`` archives) and cannot live inside the
git repository.  This module therefore *discovers* it at runtime from
one of these locations, in priority order:

    1. Explicit ``HEX_SDK_DIR`` environment variable.
    2. A ``VisualDesigner-HEX-*`` / ``VirtuosoNext*`` / ``HEX-*``
       sibling directory next to the repository root (default developer
       layout).
    3. A ``VisualDesigner-HEX-*`` folder inside the current uploaded
       repository (for users who bundle the SDK with their code).
    4. When invoked without ``repo_root`` (typical CLI one-liner):
       the current working directory, any SDK folder inside it, and
       any SDK folder in its parent.
    5. A user-installed copy under
       ``~/HEX2/VirtuosoNext`` /
       ``~/VirtuosoNext`` / ``/opt/hex-sdk`` / ``/opt/VirtuosoNext``.

Set ``HEX_SDK_DISABLE=1`` to skip discovery entirely.

Platform / variant selection
----------------------------

The SDK ships prebuilt static libraries for many combinations
(SP/MP × CO0/CO3/COs × D1/D2 × PLNONE/PLSPACE).  We expose a small,
opinionated selector via env variables:

    HEX_SDK_PLATFORM     e.g. arm-cortex-a9, win64  (auto: pick a
                         platform based on the detected cross-compiler
                         and available ``targets/*`` subdirectories)
    HEX_SDK_VARIANT      SP (default) | MP
    HEX_SDK_COMPILER     CO0 | CO3 | COs (default: COs, matches
                         RTOS_Application's default in RTOS.cmake)
    HEX_SDK_DEBUG        "" | D1 | D2 (default: unset)
    HEX_SDK_PROTECTION   "" | PLNONE | PLSPACE (default: unset)

The chosen library suffix drives the link line: with the defaults the
sandbox build will link against ``libHEX_SP_COs_Kernel.a`` and friends.
Missing archives are silently dropped from the link line — the sandbox
build's compile-only fallback still passes when only the linker cannot
resolve BSP-specific symbols.

Public API
----------

``discover(...)``
    Locate and describe the SDK.  Returns a :class:`HexSdkContext` (or
    ``None`` when disabled / missing).

``HexSdkContext``
    A frozen dataclass with everything the compile-gate and sandbox
    build need: include dirs, preprocessor defines, cross-compiler
    binary hint, lib dir, and the shortlisted ``.a`` archives to link.

``format_summary(ctx)``
    A short human-readable summary suitable for embedding into LLM
    prompts so the model knows what SDK surface is available.

Idempotent + side-effect-free: nothing here touches the filesystem
except read-only ``stat`` / directory listing.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

log = logging.getLogger(__name__)

_SDK_DIRNAME_PATTERNS = (
    r"^VisualDesigner-HEX-",   # HALCON's shipped layout
    r"^VirtuosoNext",           # user-installed layout
    r"^HEX-",                   # alt naming
    r"^hex-sdk$",               # /opt-style install
)

# Filename markers that identify a directory as an SDK root (must all
# exist under `<root>/targets/`).
_SDK_MARKERS = ("RTOS.cmake",)

# Filenames the compile gate should NEVER let leak onto ``-I`` because
# they would shadow the host compiler's libc headers. The SDK's public
# ``include/`` roots are exempt because they do not carry any of these
# at the top level (they only have subfolders like ``string/`` that host
# the HEX ``L1_string.h`` API).
_LIBC_SENTINELS_TOPLEVEL = (
    "string.h", "stdio.h", "stddef.h", "stdint.h", "stdlib.h",
    "errno.h", "math.h", "limits.h", "ctype.h", "time.h",
)


# Platform-specific base defines (extracted from RTOS.cmake in the
# HALCON SDK — see targets/<platform>/CMakeScripts/RTOS.cmake).
# These are the minimum set required to preprocess the SDK public
# headers without ``#error`` / missing-symbol failures.
_PLATFORM_DEFINES: dict[str, tuple[str, ...]] = {
    "arm-cortex-a9": (
        "-DARM_CORTEX_A9",
        "-DL1_MPU_USING_PLATFORM",
        "-DVN_NEW_TICKER",
        "-DLITTLE_ENDIAN_CPU",
        "-DL1_LOCAL_PTR_SIZE=32",
        "-DVN_SMP_TARGET",
        "-DL1_WLMONIT",
    ),
    "arm-cortex-a8": (
        "-DARM_CORTEX_A8",
        "-DL1_MPU_USING_PLATFORM",
        "-DVN_NEW_TICKER",
        "-DLITTLE_ENDIAN_CPU",
        "-DL1_LOCAL_PTR_SIZE=32",
    ),
    "arm-cortex-m0plus": (
        "-DARM_CORTEX_M0PLUS",
        "-DVN_NEW_TICKER",
        "-DLITTLE_ENDIAN_CPU",
        "-DL1_LOCAL_PTR_SIZE=32",
        "-DL1_WLMONIT",
    ),
    "arm-cortex-m3": (
        "-DARM_CORTEX_M3",
        "-DL1_MPU_USING_PLATFORM",
        "-DVN_NEW_TICKER",
        "-DLITTLE_ENDIAN_CPU",
        "-DL1_LOCAL_PTR_SIZE=32",
        "-DL1_WLMONIT",
    ),
    "arm-cortex-m33": (
        "-DARM_CORTEX_M33",
        "-DVN_NEW_TICKER",
        "-DLITTLE_ENDIAN_CPU",
        "-DL1_LOCAL_PTR_SIZE=32",
        "-DL1_WLMONIT",
    ),
    "arm-cortex-m4f": (
        "-DARM_CORTEX_M4F",
        "-DVN_NEW_TICKER",
        "-DLITTLE_ENDIAN_CPU",
        "-DL1_LOCAL_PTR_SIZE=32",
        "-DL1_WLMONIT",
    ),
    "arm-cortex-m7": (
        "-DARM_CORTEX_M7",
        "-DVN_NEW_TICKER",
        "-DLITTLE_ENDIAN_CPU",
        "-DL1_LOCAL_PTR_SIZE=32",
        "-DL1_WLMONIT",
    ),
    "arm-cortex-r4": (
        "-DARM_CORTEX_R4",
        "-DVN_NEW_TICKER",
        "-DLITTLE_ENDIAN_CPU",
        "-DL1_LOCAL_PTR_SIZE=32",
    ),
    "leon3": (
        "-DLEON3",
        "-DLITTLE_ENDIAN_CPU",
        "-DL1_LOCAL_PTR_SIZE=32",
    ),
    "posix32": (
        "-DPOSIX32",
        "-DVN_NEW_TICKER",
        "-DLITTLE_ENDIAN_CPU",
        "-DL1_LOCAL_PTR_SIZE=32",
    ),
    "powerpc_e600": (
        "-DPOWERPC_E600",
        "-DBIG_ENDIAN_CPU",
        "-DL1_LOCAL_PTR_SIZE=32",
    ),
    "powerpc_e6500": (
        "-DPOWERPC_E6500",
        "-DBIG_ENDIAN_CPU",
        "-DL1_LOCAL_PTR_SIZE=64",
        "-DVN_NEW_TICKER",
        "-DL1_MPU_USING_PLATFORM",
        "-DL1_FINEGRAIN_SPACE_PARTITIONING",
        "-DVN_NEW_RETURN_CODES",
        "-DVN_HUB_COPY_SEMANTICS",
        "-DVN_SMP_TARGET",
    ),
    "tidsp_c6000": (
        "-DTIDSP_C6000",
        "-DLITTLE_ENDIAN_CPU",
        "-DL1_LOCAL_PTR_SIZE=32",
    ),
    "win32": (
        "-DWIN32",
        "-DVN_NEW_TICKER",
        "-DLITTLE_ENDIAN_CPU",
        "-DL1_LOCAL_PTR_SIZE=32",
        "-DVN_NEW_RETURN_CODES",
        "-DVN_HUB_COPY_SEMANTICS",
        "-m32",
    ),
    "win64": (
        "-DWIN64",
        "-DLITTLE_ENDIAN_CPU",
        "-DL1_LOCAL_PTR_SIZE=64",
        "-DVN_NEW_RETURN_CODES",
        "-DVN_HUB_COPY_SEMANTICS",
        "-UVN_WITH_ABSOLUTE_TIMEOUTS",
    ),
    "arc600": (
        "-DARC600",
        "-DVN_NEW_TICKER",
        "-DLITTLE_ENDIAN_CPU",
        "-DL1_LOCAL_PTR_SIZE=32",
    ),
}

# Shared defines applied unconditionally by RTOS.cmake to every target
# (see the top of RTOS.cmake) plus the ones ``RTOS_Variant`` adds.
_COMMON_DEFINES = (
    "-DVIRTUOSO_NEXT",
    "-DVN_WITH_ABSOLUTE_TIMEOUTS",
    "-DL1_GLOBAL_PTR_SIZE=64",
    "-DVN_UART_SERVICE_SP",
    "-DASYNC_SERVICES",
    "-DL1_WIDEIDS",
    "-DL1_PRIO_INHERITANCE",
    "-DC99_STRUCTURE_INIT",
    "-Dgcc",
)

# When variant == MP, RTOS_Variant additionally adds these.
_VARIANT_MP_DEFINES = (
    "-DMP",
    "-DPENDING_REQUESTS_QUEUE",
)

# Per-platform preferred cross-compiler binary hint. Used when the
# pipeline detects HEX code targeting a specific SDK platform but no
# Makefile / CMakeLists in the user's ZIP explicitly names a CC.
_PLATFORM_CC_HINT: dict[str, str] = {
    "arm-cortex-a9":       "arm-none-eabi-gcc",
    "arm-cortex-a8":       "arm-none-eabi-gcc",
    "arm-cortex-r4":       "arm-none-eabi-gcc",
    "arm-cortex-m0plus":   "arm-none-eabi-gcc",
    "arm-cortex-m3":       "arm-none-eabi-gcc",
    "arm-cortex-m33":      "arm-none-eabi-gcc",
    "arm-cortex-m4f":      "arm-none-eabi-gcc",
    "arm-cortex-m7":       "arm-none-eabi-gcc",
    "leon3":               "sparc-elf-gcc",
    "posix32":             "gcc",
    "powerpc_e600":        "powerpc-eabi-gcc",
    "powerpc_e6500":       "powerpc-linux-gnu-gcc",
    "tidsp_c6000":         "cl6x",
    "win32":               "gcc",
    "win64":               "gcc",
    "arc600":              "mcc",
}

# Names of the SDK library archives that always contribute to a kernel
# link. Additional families (``GraphicalHostServer``, ``StdioHostServer``,
# ``sockf``, ``mcp_driver``, ``zynq_driver``, ``zynq_lwip``, ``lists``,
# ``ringBuffers``, ``prng``, ``postMortem``, ``string``, ``Cpp``,
# ``GraphicalHostClient``, ``StdioHostClient``, ``MCP*``, ``IOP*``,
# ``ZC702*``) are added opportunistically — see :func:`_shortlist_libs`.
_CORE_LIB_FAMILIES = (
    "HEX",  # matches libHEX_<VARIANT>_C<CO>[_D<DBG>][_PL<PROT>]_{Kernel,Driver,PlatformKernel,PlatformDriver}.a
)

# Companion "runtime helper" families to include when their archive
# exists for the chosen build key. Order matters for the link line —
# these come *after* the core libs and before the toolchain libraries.
_HELPER_LIB_FAMILIES = (
    "GraphicalHostServer",
    "StdioHostServer",
    "sockf",
    "Cpp",
    "lists",
    "postMortem",
    "prng",
    "ringBuffers",
    "string",
)


@dataclass(frozen=True)
class HexSdkContext:
    """A resolved snapshot of the HEX SDK for one target platform."""
    sdk_root: Path
    version: str
    platform: str
    variant: str            # "SP" or "MP"
    compiler_opts: str      # "CO0" | "CO3" | "COs" | ""
    debug_opts: str         # "" | "D1" | "D2"
    protection: str         # "" | "PLNONE" | "PLSPACE"
    platform_dir: Path      # <sdk>/targets/<platform>
    include_dirs: tuple[Path, ...]
    lib_dir: Path
    defines: tuple[str, ...]
    cc_hint: str            # e.g. "arm-none-eabi-gcc"
    is_cross: bool          # False for win64/posix32/win32, True otherwise
    lib_archives: tuple[str, ...]      # filenames only, in link order
    link_flags: tuple[str, ...]        # -L<lib_dir> -l<name> ...
    top_headers: tuple[Path, ...]      # a few public API headers used by
                                       #   :func:`format_summary`

    # ---------------------------------------------------------------
    def as_compile_flags(self) -> list[str]:
        """Return the -I / -D flags to append to a per-file compile.

        The include dirs are added FIRST so a project-local override
        (``gen_dir`` / ``code_dir`` in ``per_file_compile``) can still
        shadow the SDK copy for the same header name if the user
        intentionally provides one.
        """
        out: list[str] = []
        for d in self.include_dirs:
            out.append(f'-I{d}')
        out.extend(self.defines)
        return out

    def as_link_flags(self) -> list[str]:
        """Return the -L / -l flags for a final link.

        Kept separate from :meth:`as_compile_flags` because the
        per-file compile gate never links.
        """
        return list(self.link_flags)

    def summary_lines(self) -> list[str]:
        """Short bulleted description used in SSE messages and reports."""
        core_libs = [n for n in self.lib_archives if "HEX" in n or "Platform" in n]
        return [
            f"HEX SDK: {self.sdk_root.name} (v{self.version})",
            f"  platform    : {self.platform} ({'cross' if self.is_cross else 'native'})",
            f"  variant     : {self.variant}",
            f"  compiler CO : {self.compiler_opts or '(default)'}",
            f"  debug       : {self.debug_opts or '(none)'}",
            f"  protection  : {self.protection or '(none)'}",
            f"  cc hint     : {self.cc_hint}",
            f"  include dirs: {len(self.include_dirs)} "
            f"(root: {self.platform_dir / 'include'})",
            f"  lib dir     : {self.lib_dir}",
            f"  core libs   : {', '.join(core_libs) or '(none matching)'}",
        ]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _matches_sdk_dirname(name: str) -> bool:
    return any(re.match(p, name) for p in _SDK_DIRNAME_PATTERNS)


def _looks_like_sdk_root(path: Path) -> bool:
    """True if *path* has the ``targets/<platform>/CMakeScripts/RTOS.cmake``
    marker file the HALCON HEX SDK always ships.
    """
    try:
        targets = path / "targets"
        if not targets.is_dir():
            return False
        for plat_dir in targets.iterdir():
            if not plat_dir.is_dir():
                continue
            for marker in _SDK_MARKERS:
                if (plat_dir / "CMakeScripts" / marker).is_file():
                    return True
        return False
    except OSError:
        return False


def _candidate_roots(
    *,
    repo_root: Path | None,
    extra_search_dirs: Sequence[Path] = (),
) -> list[Path]:
    """Return an ordered list of directories to probe for an SDK root."""
    seen: set[Path] = set()
    out: list[Path] = []

    def _push(p: Path | None) -> None:
        if p is None:
            return
        try:
            r = p.resolve()
        except OSError:
            return
        if r in seen:
            return
        seen.add(r)
        out.append(r)

    env = os.environ.get("HEX_SDK_DIR", "").strip()
    if env:
        _push(Path(env))

    if repo_root:
        # In-repo checkout (developer default: the folder sits next to
        # ``api/`` etc.).
        try:
            for child in repo_root.iterdir():
                if child.is_dir() and _matches_sdk_dirname(child.name):
                    _push(child)
        except OSError:
            pass
        # One level up (SDK installed next to the workspace).
        try:
            for child in repo_root.parent.iterdir():
                if child.is_dir() and _matches_sdk_dirname(child.name):
                    _push(child)
        except OSError:
            pass

    for extra in extra_search_dirs:
        try:
            if extra.is_dir() and _matches_sdk_dirname(extra.name):
                _push(extra)
            elif extra.is_dir():
                for child in extra.iterdir():
                    if child.is_dir() and _matches_sdk_dirname(child.name):
                        _push(child)
        except OSError:
            continue

    # Ad-hoc CLI convenience: when the caller did not specify a
    # ``repo_root`` (typical for one-liner smoke-tests such as
    # ``python -c "import hex_sdk; hex_sdk.discover()"``), probe the
    # current working directory too — matches the "auto-discover"
    # intuition of running the tool from inside the project.  ``app.py``
    # always supplies an explicit ``repo_dir`` so this branch never
    # affects the FastAPI code path.
    if repo_root is None:
        try:
            cwd = Path.cwd()
        except OSError:
            cwd = None
        if cwd is not None:
            # cwd itself, in case the user cd'd into an SDK checkout.
            if _matches_sdk_dirname(cwd.name):
                _push(cwd)
            # Any SDK folder sitting directly inside cwd.
            try:
                for child in cwd.iterdir():
                    if child.is_dir() and _matches_sdk_dirname(child.name):
                        _push(child)
            except OSError:
                pass
            # One level up (SDK installed next to the project the CLI
            # was invoked from).
            try:
                for child in cwd.parent.iterdir():
                    if child.is_dir() and _matches_sdk_dirname(child.name):
                        _push(child)
            except OSError:
                pass

    # OS-level install locations.
    home = Path.home()
    for guess in (
        home / "HEX2" / "VirtuosoNext",
        home / "VirtuosoNext",
        home / "hex-sdk",
        Path("/opt/VirtuosoNext"),
        Path("/opt/hex-sdk"),
        Path("/usr/local/VirtuosoNext"),
    ):
        _push(guess)

    return out


def _extract_version_from_dirname(name: str) -> str:
    """Return e.g. ``"1.2.18.8"`` from ``"VisualDesigner-HEX-1.2.18.8"``."""
    m = re.search(r"(\d+(?:\.\d+){1,4})$", name)
    return m.group(1) if m else "unknown"


def _list_platforms(sdk_root: Path) -> list[str]:
    """Return the set of platform names that ship prebuilt libs."""
    targets = sdk_root / "targets"
    if not targets.is_dir():
        return []
    out: list[str] = []
    for p in sorted(targets.iterdir()):
        if not p.is_dir():
            continue
        # Filter out shared "share/" / "bin/" companion dirs.
        if p.name in {"share", "bin"}:
            continue
        if (p / "include").is_dir() and (p / "lib").is_dir():
            out.append(p.name)
    return out


def _pick_platform(
    requested: str | None,
    available: Sequence[str],
    *,
    cross_hint: str | None,
) -> str | None:
    """Choose a platform key from *available*.

    Priority: explicit request → cross-compiler hint match → win64 (if
    available) → arm-cortex-a9 (if available) → the first available.
    """
    if requested:
        for a in available:
            if a.lower() == requested.lower():
                return a
    if cross_hint:
        low = cross_hint.lower()
        # ARM cross-compilers with cortex-a suffixes are unusual —
        # instead detect by "arm-none-eabi" / "arm-xilinx-eabi" /
        # "aarch64-*" and default to arm-cortex-a9 when the SDK ships
        # that platform, since that is what the shipped examples use.
        if ("arm-none-eabi" in low
                or "arm-xilinx-eabi" in low
                or "aarch64" in low):
            for a in available:
                if a in ("arm-cortex-a9", "arm-cortex-a8"):
                    return a
        if "mb-gcc" in low or "microblaze" in low:
            for a in available:
                if a == "arm-cortex-a9":  # ZC702 examples use this platform
                    return a
        if "powerpc" in low:
            for a in available:
                if a.startswith("powerpc"):
                    return a
    for pref in ("win64", "arm-cortex-a9", "arm-cortex-a8", "posix32"):
        if pref in available:
            return pref
    return available[0] if available else None


def _build_key(
    variant: str, co: str, dbg: str, prot: str,
) -> str:
    """Return e.g. ``"SP_COs_D1_PLSPACE"`` — the suffix the shipped
    ``.a`` archives are keyed by.  Trailing empty parts are omitted.

    *co* is the raw compiler-opts suffix (``"0"``, ``"3"``, ``"s"``);
    the ``CO`` prefix that appears in library filenames is added here.
    *dbg* / *prot* accept either the short (``"1"`` / ``"SPACE"``) or
    full (``"D1"`` / ``"PLSPACE"``) form.
    """
    parts = [variant]
    if co:
        parts.append(f"CO{co}")
    if dbg:
        parts.append(dbg if dbg.startswith("D") else f"D{dbg}")
    if prot:
        parts.append(prot if prot.startswith("PL") else f"PL{prot}")
    return "_".join(p for p in parts if p)


def _shortlist_libs(
    lib_dir: Path,
    *,
    variant: str,
    co: str,
    dbg: str,
    prot: str,
) -> list[str]:
    """Pick the archives that match the current build key.

    We deliberately keep it deterministic: the "core" HEX archives
    (Kernel / Driver / PlatformKernel / PlatformDriver) come first —
    that mirrors :func:`RTOS_Application` in ``gnu.cmake``.  Then any
    helper family archives that also match the key.

    Missing files are skipped silently so the caller can still emit
    a `-l<name>` line without hard-failing the compile.
    """
    if not lib_dir.is_dir():
        return []

    key = _build_key(variant, co, dbg, prot)
    key_no_prot = _build_key(variant, co, dbg, "")

    # Fallback keys used only when the caller did not request a
    # specific protection level and the primary key produced no
    # matches.  ARM SDK layouts *always* include ``_PLNONE`` or
    # ``_PLSPACE`` in the archive filename (there is no
    # protection-less variant on cross targets), so we probe them
    # in turn.  win64 layouts have both — ``libHEX_SP_COs_Kernel.a``
    # (no PL suffix) and the ``_PLNONE`` variant coexist.
    fallback_keys: list[str] = []
    if not prot:
        for p in ("PLNONE", "PLSPACE"):
            fallback_keys.append(_build_key(variant, co, dbg, p))

    out: list[str] = []
    seen: set[str] = set()

    def _push(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        out.append(name)

    def _try_key(k: str) -> bool:
        found = False
        for family in _CORE_LIB_FAMILIES:
            for suf in (
                f"_{k}_Kernel.a",
                f"_{k}_Driver.a",
                f"_{k}_PlatformKernel.a",
                f"_{k}_PlatformDriver.a",
            ):
                cand = lib_dir / f"lib{family}{suf}"
                if cand.is_file():
                    _push(cand.name)
                    found = True
        return found

    # Primary attempt: the exact user-requested build key.
    _try_key(key)
    if not any(x.startswith("libHEX_") for x in out):
        # Fall back to the protection-less name (win64 layout).
        _try_key(key_no_prot)
    if not any(x.startswith("libHEX_") for x in out):
        # ARM cross targets: probe PLNONE then PLSPACE.
        for k in fallback_keys:
            if _try_key(k):
                break

    for family in _HELPER_LIB_FAMILIES:
        for suf_key in (key, key_no_prot):
            cand = lib_dir / f"lib{family}_{suf_key}.a"
            if cand.is_file():
                _push(cand.name)
                break

    return out


def _lib_link_flags(lib_dir: Path, archives: Iterable[str]) -> list[str]:
    """Turn a set of archive filenames into ``-L<dir> -l<name>`` args.

    The ``.a`` filenames are of the form ``lib<name>.a`` — strip both
    the leading ``lib`` and the trailing ``.a`` to get the ``-l<name>``
    argument.  Wrapped in ``--start-group`` / ``--end-group`` so the
    linker resolves the circular reference between Kernel and Driver
    without requiring a specific link order.
    """
    if not archives:
        return []
    args: list[str] = [f"-L{lib_dir}"]
    args.append("-Wl,--start-group")
    for name in archives:
        stem = name[3:] if name.startswith("lib") else name
        if stem.endswith(".a"):
            stem = stem[:-2]
        args.append(f"-l{stem}")
    args.append("-Wl,--end-group")
    return args


def _include_dirs(platform_dir: Path) -> list[Path]:
    """Return the ordered ``-I`` list.

    Order matters: the platform's top-level ``include/`` is added first
    so ``#include <L1_api.h>`` and ``#include <kernel/L1_kernel_api.h>``
    resolve unambiguously.  Subfolders like ``kernel/hubs/`` do NOT need
    to be on the -I path individually — GCC resolves the nested path
    from the parent include root.  We keep the walk shallow (depth 1)
    so the header resolver doesn't blow past ``ARG_MAX``.
    """
    include = platform_dir / "include"
    if not include.is_dir():
        return []
    out = [include]
    return out


def _read_platform_defines(platform: str) -> list[str]:
    return list(_PLATFORM_DEFINES.get(platform, ()))


def _select_top_headers(platform_dir: Path) -> list[Path]:
    inc = platform_dir / "include"
    if not inc.is_dir():
        return []
    picks: list[Path] = []
    for name in ("L1_api.h", "L1_types.h", "L1_hal.h"):
        p = inc / name
        if p.is_file():
            picks.append(p)
    return picks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def discover(
    *,
    repo_root: Path | None = None,
    extra_search_dirs: Sequence[Path] = (),
    platform_override: str | None = None,
    cross_hint: str | None = None,
    variant_override: str | None = None,
    compiler_override: str | None = None,
    debug_override: str | None = None,
    protection_override: str | None = None,
    disabled: bool | None = None,
) -> HexSdkContext | None:
    """Locate the HEX SDK and return a :class:`HexSdkContext`.

    Returns ``None`` when discovery is disabled (``HEX_SDK_DISABLE=1``)
    or no candidate root satisfies :func:`_looks_like_sdk_root`.
    """
    if disabled is None:
        disabled = os.environ.get("HEX_SDK_DISABLE", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
    if disabled:
        log.debug("hex_sdk: discovery disabled via HEX_SDK_DISABLE.")
        return None

    for candidate in _candidate_roots(
        repo_root=repo_root,
        extra_search_dirs=extra_search_dirs,
    ):
        if not _looks_like_sdk_root(candidate):
            continue
        available = _list_platforms(candidate)
        if not available:
            continue

        req_plat = (
            platform_override
            or os.environ.get("HEX_SDK_PLATFORM", "").strip()
            or None
        )
        platform = _pick_platform(
            req_plat, available, cross_hint=cross_hint,
        )
        if not platform:
            continue
        platform_dir = candidate / "targets" / platform
        if not (platform_dir / "include").is_dir():
            continue

        variant = (
            variant_override
            or os.environ.get("HEX_SDK_VARIANT", "").strip()
            or "SP"
        ).upper()
        if variant not in {"SP", "MP"}:
            variant = "SP"

        co = (
            compiler_override
            or os.environ.get("HEX_SDK_COMPILER", "").strip()
            or "Os"
        )
        # Normalise "0" / "3" / "s" / "O0" / "Os" / "COs" -> "COs" etc.
        # The SDK archives are named `..._COs_...` so we always pass
        # through the full ``CO<x>`` token.
        _stripped = co
        if _stripped.startswith("CO"):
            _stripped = _stripped[2:]
        if _stripped and _stripped[0] in "Oo":
            _stripped = _stripped[1:]
        co_norm = f"CO{_stripped}" if _stripped else ""

        dbg_raw = (
            debug_override
            or os.environ.get("HEX_SDK_DEBUG", "").strip()
            or ""
        )
        # Accept "1" / "2" / "D1" / "D2" — normalise to "D1"/"D2".
        if dbg_raw:
            dbg = dbg_raw if dbg_raw.startswith("D") else f"D{dbg_raw}"
            if dbg not in {"D1", "D2"}:
                dbg = ""
        else:
            dbg = ""

        prot_raw = (
            protection_override
            or os.environ.get("HEX_SDK_PROTECTION", "").strip()
            or ""
        )
        if prot_raw:
            prot = prot_raw if prot_raw.startswith("PL") else f"PL{prot_raw}"
            if prot not in {"PLNONE", "PLSPACE"}:
                prot = ""
        else:
            prot = ""

        include_dirs = tuple(_include_dirs(platform_dir))
        lib_dir = platform_dir / "lib"
        defines_l = list(_COMMON_DEFINES) + _read_platform_defines(platform)
        if variant == "MP":
            defines_l.extend(_VARIANT_MP_DEFINES)
        # The uppercase PLATFORM_DEFINE that ``RTOS_Platform`` synthesises
        # (e.g. ``ARM_CORTEX_A9``) is already present in _PLATFORM_DEFINES.
        defines = tuple(dict.fromkeys(defines_l))

        cc_hint = _PLATFORM_CC_HINT.get(platform, "gcc")
        is_cross = cc_hint != "gcc"

        archives = _shortlist_libs(
            lib_dir,
            variant=variant,
            co=co_norm[2:] if co_norm.startswith("CO") else co_norm,
            dbg=dbg,
            prot=prot,
        )
        link_flags = tuple(_lib_link_flags(lib_dir, archives))

        top_headers = tuple(_select_top_headers(platform_dir))

        ctx = HexSdkContext(
            sdk_root=candidate,
            version=_extract_version_from_dirname(candidate.name),
            platform=platform,
            variant=variant,
            compiler_opts=co_norm,
            debug_opts=dbg,
            protection=prot,
            platform_dir=platform_dir,
            include_dirs=include_dirs,
            lib_dir=lib_dir,
            defines=defines,
            cc_hint=cc_hint,
            is_cross=is_cross,
            lib_archives=tuple(archives),
            link_flags=link_flags,
            top_headers=top_headers,
        )
        log.info(
            "hex_sdk: discovered %s at %s -> platform=%s variant=%s co=%s "
            "libs=%d cc_hint=%s",
            ctx.version, ctx.sdk_root, ctx.platform, ctx.variant,
            ctx.compiler_opts, len(ctx.lib_archives), ctx.cc_hint,
        )
        return ctx

    log.debug("hex_sdk: no SDK root found in candidate list.")
    return None


# ---------------------------------------------------------------------------
# Cached discovery for hot paths
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def discover_cached(
    repo_root_str: str | None,
    cross_hint: str | None,
) -> HexSdkContext | None:
    """Wrapper around :func:`discover` with an LRU cache keyed by the
    two arguments that vary at runtime (repo root and cross-compiler
    hint).  Env overrides intentionally do NOT invalidate the cache —
    they are read once per process, which matches how uvicorn works
    for us.  Call :meth:`cache_clear` after mutating ``os.environ`` in
    tests.
    """
    return discover(
        repo_root=Path(repo_root_str) if repo_root_str else None,
        cross_hint=cross_hint,
    )


def clear_cache() -> None:
    discover_cached.cache_clear()


# ---------------------------------------------------------------------------
# Helpers used by other modules
# ---------------------------------------------------------------------------


def format_summary(ctx: HexSdkContext | None) -> str:
    """Multi-line human summary for SSE messages and compile_report."""
    if ctx is None:
        return "HEX SDK: not detected (set HEX_SDK_DIR to enable RTOS build support)."
    return "\n".join(ctx.summary_lines())


def render_llm_context(ctx: HexSdkContext | None, *, max_chars: int = 6000) -> str:
    """Return a compact markdown block describing the SDK for the LLM.

    Kept small so it does not crowd out the ICD change spec: only the
    top-level API headers get their full body embedded; deeper headers
    are listed by relative path only.
    """
    if ctx is None:
        return ""
    lines: list[str] = [
        "## HEX / VirtuosoNext RTOS SDK",
        (
            "The target is the HALCON HEX / VirtuosoNext RTOS. Use ONLY the "
            "APIs the SDK provides — do not invent function names, types, "
            "or macros.  Angle-bracket includes like `#include <L1_api.h>` "
            "resolve to the SDK header tree listed below."
        ),
        "",
        f"- SDK version : {ctx.version}",
        f"- Platform    : {ctx.platform} ({'cross-compiled' if ctx.is_cross else 'native'})",
        f"- Variant     : {ctx.variant} (Single-Processor or Multi-Processor kernel)",
        f"- Compiler set: {ctx.compiler_opts}",
        f"- Include root: {ctx.platform_dir / 'include'}",
    ]
    if ctx.lib_archives:
        lines.append(
            "- Link libs   : "
            + ", ".join(ctx.lib_archives[:8])
            + (" ..." if len(ctx.lib_archives) > 8 else "")
        )
    if ctx.defines:
        lines.append(
            "- Preprocessor defines active during build:"
        )
        for d in ctx.defines:
            lines.append(f"  * `{d}`")
    include = ctx.platform_dir / "include"
    top_files = sorted(
        p.name for p in include.iterdir() if p.suffix == ".h"
    ) if include.is_dir() else []
    if top_files:
        lines.append("")
        lines.append(f"### Public API headers under `{include}`:")
        for name in top_files:
            lines.append(f"- `{name}`")
    # Embed L1_api.h up to a soft cap so the model sees actual signatures.
    remaining = max_chars - len("\n".join(lines))
    if remaining > 1000 and ctx.top_headers:
        for hp in ctx.top_headers:
            try:
                body = hp.read_text(errors="replace")
            except OSError:
                continue
            snippet = body[: remaining - 400]
            if len(snippet) < len(body):
                snippet += "\n/* ... truncated (LLM context budget) ... */"
            lines.append(f"\n### `{hp.name}` (public API — abridged)")
            lines.append("```c")
            lines.append(snippet)
            lines.append("```")
            remaining -= len(snippet) + 200
            if remaining <= 500:
                break
    return "\n".join(lines)


def augment_toxic_allowlist(ctx: HexSdkContext | None) -> tuple[Path, ...]:
    """Return include directories that the compile-gate's
    :func:`per_file_compile._is_toxic_include_dir` should always
    consider safe, even if some sibling directory contains a
    ``string.h`` sentinel.  The HEX SDK's public ``include/`` tree
    never carries top-level libc headers so it is always safe.
    """
    if ctx is None:
        return ()
    return tuple(ctx.include_dirs)
