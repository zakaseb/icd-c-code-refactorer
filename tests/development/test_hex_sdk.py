"""Unit tests for api/hex_sdk.py — HEX / VirtuosoNext RTOS SDK integration.

Exercises the auto-discovery, platform / variant selection, library
shortlisting, and LLM-context rendering paths of `api/hex_sdk.py`, plus
the wiring into `api/per_file_compile.py` (compile-gate `-I`/`-D`
injection) and `api/app.py` (cross-compiler detection + sandbox flag
propagation).

The tests use a *fake* SDK layout built inside a temporary directory —
we never depend on the real 415 MB VisualDesigner-HEX-* tarball being
installed on the developer's box, and we don't need the real ARM cross
toolchain either.

    Test 1   `discover()` returns None when disabled via HEX_SDK_DISABLE.
    Test 2   `discover()` locates a fake SDK next to the repo root and
             fills in include_dirs / lib_dir / defines / cc_hint.
    Test 3   HEX_SDK_DIR env var takes highest priority.
    Test 3b  REGRESSION: `discover()` with no args probes the current
             working directory (fixes the Layer-4 verification one-liner
             that returned None even when the SDK sat right beside the
             caller).
    Test 4   Platform / variant / compiler overrides via env vars steer
             both the picked target and the library shortlist.
    Test 5   `_build_key` / `_shortlist_libs` pick the right archives
             for the requested (variant, CO, D, PL) tuple.
    Test 6   `as_compile_flags` produces `-I<inc>` first then `-D`; the
             link flags wrap `-L<libdir>` in --start-group/--end-group.
    Test 7   `render_llm_context` emits the SDK version, platform,
             defines and inlines the top-level L1_api.h snippet.
    Test 8   `augment_toxic_allowlist` returns the SDK include roots.
    Test 9   `_pick_platform` prefers an arm-cortex platform when the
             detected cross-compiler is `arm-none-eabi-gcc`.
    Test 10  `format_summary` never crashes on a None context and lists
             all key fields when a context is provided.
    Test 11  Integration: `per_file_compile._compile_command` accepts
             extra_flags and quotes them safely (space / quote embedded).
    Test 12  Integration: `app._discover_hex_sdk` finds a SDK sitting
             at repo_root/../VisualDesigner-HEX-1.2.3.4 via
             _candidate_roots' parent scan.
    Test 13  Integration: `app._detect_cross_compiler` recognises a
             HEX environment.mk / PROJECT_GEN cross hint.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure the `api/` module dir is importable both as a flat package
# and via `api.hex_sdk`.
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT))

# WORKSPACE_DIR must be set before app is imported so it doesn't
# pollute the real workspace/ tree.
os.environ["WORKSPACE_DIR"] = tempfile.mkdtemp(prefix="icd_hex_sdk_test_")

# Discovery cache is per-process — we mutate env vars aggressively so
# always clear the cache between test cases.
import hex_sdk  # noqa: E402
from hex_sdk import (  # noqa: E402
    HexSdkContext,
    _build_key,
    _shortlist_libs,
    _lib_link_flags,
    _pick_platform,
    augment_toxic_allowlist,
    discover,
    format_summary,
    render_llm_context,
)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} — {detail}")


def _reset_env():
    """Wipe every HEX_SDK_* env var so tests are hermetic."""
    for k in list(os.environ.keys()):
        if k.startswith("HEX_SDK_"):
            del os.environ[k]
    hex_sdk.clear_cache()


def _make_fake_sdk(
    tmp: Path,
    *,
    version: str = "1.2.18.8",
    platforms: tuple[str, ...] = ("arm-cortex-a9", "win64"),
    include_headers: tuple[str, ...] = (
        "L1_api.h", "L1_hal.h", "L1_types.h",
    ),
    variant: str = "SP",
    co: str = "s",   # bare compiler-opts token — the "CO" prefix is added
                     # in the archive filenames below, matching how the real
                     # SDK names them (libHEX_SP_COs_Kernel.a etc.).
) -> Path:
    """Build a minimal HEX SDK layout the discovery code accepts.

    Structure::

        <tmp>/VisualDesigner-HEX-1.2.18.8/
            targets/
              arm-cortex-a9/
                CMakeScripts/RTOS.cmake     # marker
                include/L1_api.h            # top header
                include/kernel/...
                lib/libHEX_SP_COs_Kernel.a  # etc.
              win64/
                CMakeScripts/RTOS.cmake
                include/L1_api.h
                lib/libHEX_SP_COs_Kernel.a
    """
    sdk = tmp / f"VisualDesigner-HEX-{version}"
    sdk.mkdir(parents=True)
    for plat in platforms:
        plat_dir = sdk / "targets" / plat
        (plat_dir / "CMakeScripts").mkdir(parents=True)
        (plat_dir / "CMakeScripts" / "RTOS.cmake").write_text(
            "# fake RTOS.cmake marker for hex_sdk._looks_like_sdk_root\n"
        )
        include = plat_dir / "include"
        include.mkdir()
        for h in include_headers:
            (include / h).write_text(
                f"/* fake {h} for hex_sdk tests */\n"
                f"typedef int L1_Status_t;\n"
                f"L1_Status_t L1_ping(void);\n"
            )
        # Nested subfolder used by <kernel/L1_kernel_api.h> lookups.
        (include / "kernel").mkdir()
        (include / "kernel" / "L1_kernel_api.h").write_text(
            "/* fake kernel header */\n"
        )
        lib = plat_dir / "lib"
        lib.mkdir()
        # The four "core" archives that _shortlist_libs looks for. For
        # the ARM layout add the PLSPACE / PLNONE variants too so the
        # protection-level tests have something to match.
        core_names = [
            f"libHEX_{variant}_CO{co}_Kernel.a",
            f"libHEX_{variant}_CO{co}_Driver.a",
            f"libHEX_{variant}_CO{co}_PlatformKernel.a",
            f"libHEX_{variant}_CO{co}_PlatformDriver.a",
        ]
        if plat == "arm-cortex-a9":
            core_names += [
                f"libHEX_{variant}_CO{co}_D1_PLSPACE_Kernel.a",
                f"libHEX_{variant}_CO{co}_D1_PLSPACE_Driver.a",
                f"libHEX_{variant}_CO{co}_PLNONE_Kernel.a",
            ]
        for n in core_names:
            (lib / n).write_bytes(b"!<arch>\n")   # ar magic; enough for is_file()

        # Helper family archive.
        helper = lib / f"libstring_{variant}_CO{co}.a"
        helper.write_bytes(b"!<arch>\n")
    return sdk


# ---------------------------------------------------------------
print("\n=== Test 1: discover() honours HEX_SDK_DISABLE=1 ===")
with tempfile.TemporaryDirectory() as _td:
    _reset_env()
    os.environ["HEX_SDK_DISABLE"] = "1"
    fake = _make_fake_sdk(Path(_td))
    ctx = discover(repo_root=fake.parent)
    check("discover() returns None when disabled", ctx is None,
          f"got {ctx!r}")

# ---------------------------------------------------------------
print("\n=== Test 2: sibling SDK discovery ===")
with tempfile.TemporaryDirectory() as _td:
    _reset_env()
    tmp = Path(_td)
    repo = tmp / "myrepo"
    repo.mkdir()
    fake = _make_fake_sdk(tmp)   # sits at tmp/VisualDesigner-HEX-1.2.18.8
    ctx = discover(repo_root=repo)
    check("sibling SDK is found", ctx is not None,
          "no context returned")
    if ctx is not None:
        check("sdk_root points at fake tree",
              ctx.sdk_root == fake.resolve(),
              f"expected {fake} got {ctx.sdk_root}")
        check("version parsed from dirname",
              ctx.version == "1.2.18.8", ctx.version)
        check("include dir on platform_dir/include",
              (ctx.platform_dir / "include") in ctx.include_dirs)
        check("VIRTUOSO_NEXT define present",
              "-DVIRTUOSO_NEXT" in ctx.defines)
        check("cc_hint set for chosen platform",
              bool(ctx.cc_hint), ctx.cc_hint)

# ---------------------------------------------------------------
print("\n=== Test 3: HEX_SDK_DIR env override wins ===")
with tempfile.TemporaryDirectory() as _td:
    _reset_env()
    tmp = Path(_td)
    other_repo = tmp / "unrelated_repo"
    other_repo.mkdir()
    fake = _make_fake_sdk(tmp / "elsewhere" / "install_root")
    os.environ["HEX_SDK_DIR"] = str(fake)
    ctx = discover(repo_root=other_repo)
    check("HEX_SDK_DIR resolves to fake root", ctx is not None,
          "no ctx")
    if ctx is not None:
        check("HEX_SDK_DIR resolved to fake root",
              ctx.sdk_root == fake.resolve())

# ---------------------------------------------------------------
print("\n=== Test 3b: discover() with no args probes the CWD ===")
# REGRESSION: previously `hex_sdk.discover()` (no repo_root argument)
# skipped the sibling/child scan entirely and only checked HEX_SDK_DIR
# + OS-level install paths, so the shipped Layer-4 verification
# one-liner ``python -c "import hex_sdk; hex_sdk.discover()"`` returned
# None even when the SDK sat right next to the caller.  The CWD probe
# now covers that ad-hoc use.
with tempfile.TemporaryDirectory() as _td:
    _reset_env()
    tmp = Path(_td)
    fake = _make_fake_sdk(tmp)
    prev_cwd = Path.cwd()
    try:
        os.chdir(tmp)
        # No repo_root, no HEX_SDK_DIR — must still find fake via CWD.
        ctx = discover()
        check("discover() with no args finds SDK in CWD",
              ctx is not None and ctx.sdk_root == fake.resolve(),
              f"ctx={ctx}")

        # Also probe from a *sibling* of the SDK so parent-of-cwd scan
        # is exercised.
        sibling = tmp / "sibling_project"
        sibling.mkdir()
        os.chdir(sibling)
        hex_sdk.clear_cache()
        ctx2 = discover()
        check("discover() with no args finds SDK in CWD parent",
              ctx2 is not None and ctx2.sdk_root == fake.resolve(),
              f"ctx2={ctx2}")

        # And when CWD *is* the SDK root itself.
        os.chdir(fake)
        hex_sdk.clear_cache()
        ctx3 = discover()
        check("discover() with no args identifies CWD as SDK root",
              ctx3 is not None and ctx3.sdk_root == fake.resolve(),
              f"ctx3={ctx3}")
    finally:
        os.chdir(prev_cwd)
        hex_sdk.clear_cache()

# ---------------------------------------------------------------
print("\n=== Test 4: platform / variant / compiler overrides ===")
with tempfile.TemporaryDirectory() as _td:
    _reset_env()
    tmp = Path(_td)
    fake = _make_fake_sdk(tmp, variant="MP", co="s")
    # Rewrite the SDK to actually carry MP archives (the helper wrote SP).
    for plat in ("arm-cortex-a9", "win64"):
        libdir = fake / "targets" / plat / "lib"
        for suf in ("_Kernel", "_Driver", "_PlatformKernel", "_PlatformDriver"):
            (libdir / f"libHEX_MP_COs{suf}.a").write_bytes(b"!<arch>\n")
    os.environ["HEX_SDK_DIR"] = str(fake)
    os.environ["HEX_SDK_PLATFORM"] = "win64"
    os.environ["HEX_SDK_VARIANT"] = "MP"
    os.environ["HEX_SDK_COMPILER"] = "COs"
    ctx = discover()
    check("override picks win64", ctx and ctx.platform == "win64",
          f"platform={ctx.platform if ctx else None}")
    if ctx is not None:
        check("variant honoured",  ctx.variant == "MP",
              f"variant={ctx.variant}")
        check("MP defines added",  "-DMP" in ctx.defines,
              f"defines={ctx.defines}")
        check("core lib shortlisted",
              any(a.startswith("libHEX_MP_COs_") for a in ctx.lib_archives),
              f"libs={ctx.lib_archives}")

# ---------------------------------------------------------------
print("\n=== Test 5: _build_key + _shortlist_libs ===")
with tempfile.TemporaryDirectory() as _td:
    _reset_env()
    tmp = Path(_td)
    fake = _make_fake_sdk(tmp)
    lib = fake / "targets" / "arm-cortex-a9" / "lib"
    key = _build_key("SP", "s", "1", "SPACE")
    check("_build_key composes SP_COs_D1_PLSPACE",
          key == "SP_COs_D1_PLSPACE", f"got {key}")
    picks = _shortlist_libs(lib, variant="SP", co="s", dbg="D1", prot="PLSPACE")
    check("D1 PLSPACE match found",
          any("SP_COs_D1_PLSPACE_Kernel.a" in n for n in picks),
          f"picks={picks}")
    picks2 = _shortlist_libs(lib, variant="SP", co="s", dbg="", prot="")
    check("no-D no-PL match falls back to bare key",
          any(n == "libHEX_SP_COs_Kernel.a" for n in picks2),
          f"picks2={picks2}")

# ---------------------------------------------------------------
print("\n=== Test 6: as_compile_flags + as_link_flags order ===")
with tempfile.TemporaryDirectory() as _td:
    _reset_env()
    tmp = Path(_td)
    fake = _make_fake_sdk(tmp)
    os.environ["HEX_SDK_DIR"] = str(fake)
    os.environ["HEX_SDK_PLATFORM"] = "arm-cortex-a9"
    ctx = discover()
    if ctx is not None:
        cflags = ctx.as_compile_flags()
        first_D = next((i for i, f in enumerate(cflags) if f.startswith("-D")), -1)
        last_I  = max((i for i, f in enumerate(cflags) if f.startswith("-I")), default=-1)
        check("includes precede defines in compile flags",
              last_I >= 0 and first_D >= 0 and last_I < first_D,
              f"first_D={first_D} last_I={last_I} flags={cflags}")
        lflags = ctx.as_link_flags()
        if lflags:
            check("link flags start with -L", lflags[0].startswith("-L"),
                  f"lflags[0]={lflags[0]}")
            check("--start-group / --end-group present",
                  "-Wl,--start-group" in lflags and "-Wl,--end-group" in lflags,
                  f"lflags={lflags}")

# ---------------------------------------------------------------
print("\n=== Test 7: render_llm_context inlines API header ===")
with tempfile.TemporaryDirectory() as _td:
    _reset_env()
    tmp = Path(_td)
    fake = _make_fake_sdk(tmp)
    os.environ["HEX_SDK_DIR"] = str(fake)
    ctx = discover()
    llm = render_llm_context(ctx, max_chars=4000)
    check("llm block mentions SDK version", "1.2.18.8" in llm,
          "no version in output")
    check("llm block mentions VIRTUOSO_NEXT", "VIRTUOSO_NEXT" in llm,
          "define not in output")
    check("llm block inlines L1_api.h snippet", "L1_ping" in llm,
          "header body missing")
    check("llm block on None context is empty",
          render_llm_context(None) == "", "should be empty")

# ---------------------------------------------------------------
print("\n=== Test 8: augment_toxic_allowlist ===")
with tempfile.TemporaryDirectory() as _td:
    _reset_env()
    fake = _make_fake_sdk(Path(_td))
    os.environ["HEX_SDK_DIR"] = str(fake)
    ctx = discover()
    allow = augment_toxic_allowlist(ctx)
    check("allowlist non-empty when SDK found", len(allow) > 0,
          "allowlist empty")
    check("allowlist contains SDK include dir",
          any("include" in str(p) for p in allow),
          f"allow={allow}")
    check("allowlist empty for None ctx",
          augment_toxic_allowlist(None) == (),
          "None -> non-empty tuple")

# ---------------------------------------------------------------
print("\n=== Test 9: _pick_platform prefers ARM cross-compiler ===")
_reset_env()
plat = _pick_platform(
    None,
    ("arm-cortex-a9", "win64", "posix32"),
    cross_hint="arm-none-eabi-gcc",
)
check("arm cross hint picks arm-cortex-a9", plat == "arm-cortex-a9",
      f"got {plat}")

plat = _pick_platform(
    None,
    ("win64", "posix32"),
    cross_hint=None,
    host_system="linux",
)
check("linux host prefers posix32 over win64 (win64 needs windows.h)",
      plat == "posix32", f"got {plat}")

plat = _pick_platform(
    None,
    ("win64", "arm-cortex-a9"),
    cross_hint=None,
    host_system="linux",
)
check("linux host prefers arm-cortex-a9 over win64",
      plat == "arm-cortex-a9", f"got {plat}")

plat = _pick_platform(
    None,
    ("win64", "arm-cortex-a9"),
    cross_hint=None,
    host_system="win32",
)
check("windows host prefers win64", plat == "win64", f"got {plat}")

plat = _pick_platform(
    "posix32",
    ("win64", "posix32"),
    cross_hint="arm-none-eabi-gcc",
)
check("explicit request wins over hint", plat == "posix32", f"got {plat}")

# ---------------------------------------------------------------
print("\n=== Test 10: format_summary robustness ===")
_reset_env()
s_none = format_summary(None)
check("format_summary(None) mentions 'not detected'",
      "not detected" in s_none.lower(), s_none)

with tempfile.TemporaryDirectory() as _td:
    fake = _make_fake_sdk(Path(_td))
    os.environ["HEX_SDK_DIR"] = str(fake)
    ctx = discover()
    s = format_summary(ctx)
    check("summary lists SDK version", "1.2.18.8" in s, s)
    check("summary lists platform",    "platform" in s.lower(), s)
    check("summary lists cc hint",     "cc hint" in s.lower(), s)

# ---------------------------------------------------------------
print("\n=== Test 11: per_file_compile._compile_command extra_flags ===")
_reset_env()
import per_file_compile  # noqa: E402
cmd = per_file_compile._compile_command(
    cc="gcc",
    src=Path("/tmp/foo.c"),
    obj=Path("/tmp/foo.o"),
    include_dirs=[Path("/tmp/inc")],
    extra_flags=["-DFOO=1", "-DBAR=\"hi there\"", "-Wl,--start-group"],
)
check("extra_flags appear on the command line", "-DFOO=1" in cmd,
      f"cmd={cmd}")
check("extra_flags with spaces are quoted",
      "-DBAR=" in cmd and "'hi there'" in cmd.replace('"', "'") or
      "\"hi there\"" in cmd,
      f"cmd={cmd}")
check("-Wl group flag survives", "-Wl,--start-group" in cmd, f"cmd={cmd}")

# ---------------------------------------------------------------
print("\n=== Test 12: app._discover_hex_sdk finds sibling SDK ===")
_reset_env()
import app  # noqa: E402
with tempfile.TemporaryDirectory() as _td:
    tmp = Path(_td)
    repo = tmp / "repo_contents"
    repo.mkdir()
    fake = _make_fake_sdk(tmp)   # tmp/VisualDesigner-HEX-1.2.18.8
    # The app wrapper ignores an SDK when the upload does not look
    # like HEX code, so unrelated native projects stay on host gcc.
    (repo / "fw.h").write_text("#include <L1_api.h>\n")
    hex_sdk.clear_cache()
    ctx = app._discover_hex_sdk(repo_dir=repo, cross_hint=None)
    check("app-level discovery finds sibling SDK",
          ctx is not None and ctx.sdk_root == fake.resolve(),
          f"ctx={ctx}")

# ---------------------------------------------------------------
print("\n=== Test 14: session-nested repo still finds the sibling SDK ===")
# REGRESSION: the compile gate passes repo_root =
# workspace/sessions/<id>/repo_contents. The SDK lives next to the
# git checkout, several directories above that. Discovery used to
# look only at the session folder and returned None, so the UI
# compiled with native gcc and died on `#include <L1_api.h>`.
with tempfile.TemporaryDirectory() as _td:
    _reset_env()
    tmp = Path(_td)
    fake = _make_fake_sdk(tmp)
    nested = tmp / "workspace" / "sessions" / "abc" / "repo_contents"
    nested.mkdir(parents=True)
    (nested / "fw.h").write_text(
        '#include <bsp/zynq/gpiops/gpiops.h>\n'
        '/* Product: P3L Light MCP */\n'
    )
    ctx = discover(repo_root=nested, cross_hint="gcc")
    check("nested session repo finds ancestor SDK",
          ctx is not None and ctx.sdk_root == fake.resolve(),
          f"ctx={ctx}")
    if ctx is not None:
        check("zynq include sniffs arm-cortex-a9 (not win64)",
              ctx.platform == "arm-cortex-a9",
              f"platform={ctx.platform}")
        check("arm platform is marked cross",
              ctx.is_cross and ctx.cc_hint == "arm-none-eabi-gcc",
              f"is_cross={ctx.is_cross} cc={ctx.cc_hint}")
        check("P3L product selects the MCP_P3L board package",
              ctx.board == "MCP_P3L",
              f"board={ctx.board}")

print("\n=== Test 13: _detect_cross_compiler recognises HEX env.mk ===")
_reset_env()
with tempfile.TemporaryDirectory() as _td:
    tmp = Path(_td)
    # Simulate a project the user zipped up that carries the HEX
    # example environment.mk. `_detect_cross_compiler` should pull
    # the cross-compiler out of `PROJECT_GEN` and return the ARM
    # sandbox CC command.
    (tmp / "src").mkdir()
    (tmp / "src" / "L1_default_abort_handler.c").write_text("/* app */\n")
    makefile = tmp / "Makefile"
    makefile.write_text(
        "# top-level makefile that would otherwise look native\n"
        "all:\n\t@echo build\n"
    )
    (tmp / "environment.mk").write_text(
        "RTOS_DIR=/opt/VisualDesigner-HEX-1.2.18.8\n"
        "PROJECT_GEN=/opt/VisualDesigner-HEX-1.2.18.8/targets/arm-cortex-a9\n"
        "PROJECT_NAME=blinky\n"
    )
    build_info = {"type": "make", "path": makefile, "build_dir": tmp}
    cc_cmd = app._detect_cross_compiler(build_info)
    check("env.mk drives cross detection",
          "arm-none-eabi" in cc_cmd or cc_cmd == app.SANDBOX_CC_ARM,
          f"cc_cmd={cc_cmd}")


# ---------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
if FAIL > 0:
    sys.exit(1)
else:
    print("All hex_sdk tests passed!")
    sys.exit(0)
