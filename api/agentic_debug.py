"""
Agentic-AI inspired sandbox-build debugging pipeline.

This module replaces the iterative-rewrite loop and the ReAct-style
orchestrator with a deterministic state machine that treats debugging
as a *search problem over constrained edits*.

Architecture (high-level)
-------------------------

    BUILD  →  TRIAGE  →  ROOT_CAUSE  →  PLAN  →  PATCH  →  VERIFY  →  DECIDE
       ▲                                                              │
       └──────────────────────────────────────────────────────────────┘

Components mapped to the design brief:

* ``CodebaseIndex``         — fast symbol/file index (Codebase Context Service)
* ``BuildHarness``          — wraps ``_run_sandbox_build`` and captures raw +
                              structured logs (Build Harness)
* ``parse_build_log`` +     — Debug Intelligence Layer: tuples + classifier +
  ``classify_error`` +        cascade clustering with root-cause ranking.
  ``cluster_errors``
* ``HypothesisPlanner``     — LLM-driven planner that emits ONE typed
                              hypothesis per turn (Patch Planner)
* ``apply_hypothesis``      — minimal text/anchor-aware applier with
                              snapshots and rollback (Patch Applier)
* ``run_pre_build_checks``  — header/include and declaration/definition
                              consistency checks (Verification Suite)
* ``Playbook``              — per-session memory of successful fixes
                              keyed by error class (Memory & Learning Store)
* ``AgenticDebugLoop``      — the state machine; enforces single-hypothesis
                              per iteration, three-layer correction strategy,
                              and the documented stop conditions.

Per-attempt artifacts are written under
``<session_dir>/agentic_attempts/attempt_NN/``:
``patch.diff``, ``build.log.raw``, ``build.log.structured.json``,
``error_clusters.json``, ``hypothesis.md``, ``metrics.json``.

The loop yields the same SSE-shaped event dictionaries that the existing
orchestrator yielded, so the calling code in ``api.app`` only needs a
thin adapter to forward them.

This module is fully local: it talks to the same llama-server via the
``llm_stream`` callable injected by the caller. No data leaves the host.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DEFAULT_MAX_ATTEMPTS = 30
DEFAULT_NO_PROGRESS_LIMIT = 3      # iterations without net error reduction
DEFAULT_OSCILLATION_LIMIT = 2      # same fingerprint reappears N times
DEFAULT_EDIT_BUDGET_FILES = 60     # cumulative distinct files modified
DEFAULT_MAX_TOUCHED_FILES_PER_ATTEMPT = 3
DEFAULT_PLANNER_MAX_TOKENS = 1536
DEFAULT_PATCH_MAX_TOKENS = 4096
DEFAULT_BUILD_OUTPUT_BUDGET = 8000
DEFAULT_INDEX_FILE_LIMIT = 4000    # cap symbols/files indexed for safety


# ---------------------------------------------------------------------------
# Error schema (Debug Intelligence Layer)
# ---------------------------------------------------------------------------


class ErrorClass(str, Enum):
    """Typed classification of a build error.

    Mirrors the policy table in the design brief (section D).
    """

    MISSING_INCLUDE = "missing_include"
    UNKNOWN_TYPE = "unknown_type"
    MACRO_MISMATCH = "macro_mismatch"
    SIGNATURE_MISMATCH = "signature_mismatch"
    LINK_UNDEFINED = "link_undefined"
    QUALIFIER_MISMATCH = "qualifier_mismatch"     # const/volatile/packed
    ENUM_VALUE_DRIFT = "enum_value_drift"
    STRUCT_LAYOUT = "struct_layout"
    CALLING_CONVENTION = "calling_convention"
    OTHER = "other"


class Lane(str, Enum):
    """Independent verification lanes for mixed firmware/software stacks."""

    FIRMWARE = "firmware"
    SOFTWARE = "software"


# Deterministic repair-family ordering for Xilinx SDK 2018 convergence.
_REPAIR_FAMILY_PRIORITY: dict[ErrorClass, int] = {
    # 1) Missing/renamed headers/macros/types
    ErrorClass.MISSING_INCLUDE: 1,
    ErrorClass.UNKNOWN_TYPE: 1,
    ErrorClass.MACRO_MISMATCH: 1,
    ErrorClass.ENUM_VALUE_DRIFT: 1,
    # 2) Function prototype/signature drift
    ErrorClass.SIGNATURE_MISMATCH: 2,
    # 3) Struct/register field drift + volatile correctness
    ErrorClass.STRUCT_LAYOUT: 3,
    ErrorClass.QUALIFIER_MISMATCH: 3,
    ErrorClass.CALLING_CONVENTION: 3,
    # 4) Undefined references / linker symbol mapping
    ErrorClass.LINK_UNDEFINED: 4,
    # 5) Warning hardening / cleanup
    ErrorClass.OTHER: 5,
}


@dataclass
class BuildError:
    """One structured compiler/linker diagnostic."""

    tool: str           # "compile" or "link"
    severity: str       # "error" | "warning"
    file: str           # repo-relative path or "" if linker-global
    line: int           # 0 when unknown
    column: int         # 0 when unknown
    symbol: str         # extracted symbol/macro/type when present
    message: str
    error_class: ErrorClass = ErrorClass.OTHER
    raw: str = ""        # original line for traceability

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["error_class"] = self.error_class.value
        return d


@dataclass
class ErrorCluster:
    """A group of errors that share a root cause candidate."""

    root_cause: str           # short label, e.g. "missing typedef PeripheralX_Config"
    error_class: ErrorClass
    representative: BuildError
    members: list[BuildError]
    files: list[str]
    centrality: int           # rough score for ranking
    suggestion: str           # plain-English guidance for the planner

    def to_json(self) -> dict[str, Any]:
        return {
            "root_cause": self.root_cause,
            "error_class": self.error_class.value,
            "representative": self.representative.to_json(),
            "files": self.files,
            "centrality": self.centrality,
            "suggestion": self.suggestion,
            "member_count": len(self.members),
            "sample_messages": [e.message for e in self.members[:5]],
        }


# ---------------------------------------------------------------------------
# Build-log parsing (Debug Intelligence Layer — A)
# ---------------------------------------------------------------------------


# Most GCC/Clang diagnostics:  path:line:col: severity: message
_GCC_DIAG_RE = re.compile(
    r"^(?P<file>[^\s:][^\n:]*?):(?P<line>\d+):(?P<col>\d+):\s*"
    r"(?P<sev>error|warning|fatal error|note):\s*(?P<msg>.+)$",
    re.MULTILINE,
)
# `path:line: error: ...` (no column — older toolchains, e.g. gcc 4.x style)
_GCC_DIAG_NOCOL_RE = re.compile(
    r"^(?P<file>[^\s:][^\n:]*?):(?P<line>\d+):\s*"
    r"(?P<sev>error|warning|fatal error|note):\s*(?P<msg>.+)$",
    re.MULTILINE,
)
# Linker:  ld: undefined reference to `xil_printf'
_LD_UNDEF_RE = re.compile(
    r"undefined reference to [`']([^'`]+)['`]"
)
# Linker error of the form  file.o: in function `foo': file.c:42: undefined ...
_LD_OBJ_REF_RE = re.compile(
    r"^([^\s:][^\n:]*?\.o):\s*in function [`']([^'`]+)['`]:",
    re.MULTILINE,
)


def parse_build_log(build_output: str) -> list[BuildError]:
    """Parse compiler and linker diagnostics into a structured list.

    Both column-bearing (modern GCC/Clang) and column-less (older GCC, mb-gcc
    on Xilinx SDK 2018.x) formats are supported.
    """
    errors: list[BuildError] = []
    seen: set[tuple[str, int, int, str]] = set()

    for m in _GCC_DIAG_RE.finditer(build_output):
        file = m.group("file").strip()
        if not _looks_like_source(file):
            continue
        sev = m.group("sev").lower().replace("fatal error", "error")
        line = int(m.group("line"))
        col = int(m.group("col"))
        msg = m.group("msg").strip()
        key = (file, line, col, msg)
        if key in seen:
            continue
        seen.add(key)
        errors.append(
            BuildError(
                tool="compile",
                severity=sev,
                file=file,
                line=line,
                column=col,
                symbol=_extract_symbol(msg),
                message=msg,
                raw=m.group(0).strip(),
            )
        )

    # No-column form (some toolchains)
    for m in _GCC_DIAG_NOCOL_RE.finditer(build_output):
        file = m.group("file").strip()
        if not _looks_like_source(file):
            continue
        sev = m.group("sev").lower().replace("fatal error", "error")
        line = int(m.group("line"))
        msg = m.group("msg").strip()
        key = (file, line, 0, msg)
        if key in seen:
            continue
        seen.add(key)
        errors.append(
            BuildError(
                tool="compile",
                severity=sev,
                file=file,
                line=line,
                column=0,
                symbol=_extract_symbol(msg),
                message=msg,
                raw=m.group(0).strip(),
            )
        )

    # Linker: `undefined reference to ...`
    for m in _LD_UNDEF_RE.finditer(build_output):
        sym = m.group(1)
        key = ("", 0, 0, f"undefined reference to {sym}")
        if key in seen:
            continue
        seen.add(key)
        errors.append(
            BuildError(
                tool="link",
                severity="error",
                file="",
                line=0,
                column=0,
                symbol=sym,
                message=f"undefined reference to `{sym}'",
                raw=m.group(0).strip(),
            )
        )

    return errors


def _looks_like_source(path: str) -> bool:
    if not path:
        return False
    return path.endswith((".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".inc"))


_SYM_PATTERNS = [
    re.compile(r"['`]([A-Za-z_][\w]*)['`]"),
    re.compile(r"unknown type name\s+['`]([A-Za-z_][\w]*)['`]"),
    re.compile(r"implicit declaration of function\s+['`]([A-Za-z_][\w]*)['`]"),
    re.compile(r"no member named\s+['`]([A-Za-z_][\w]*)['`]"),
    re.compile(r"redefinition of\s+['`]([A-Za-z_][\w]*)['`]"),
    re.compile(r"['`]([A-Za-z_][\w]*)['`] (?:undeclared|not declared)"),
]


def _extract_symbol(msg: str) -> str:
    for pat in _SYM_PATTERNS:
        m = pat.search(msg)
        if m:
            return m.group(1)
    return ""


# ---------------------------------------------------------------------------
# Classification policy (D)
# ---------------------------------------------------------------------------


_CLASSIFIERS: list[tuple[ErrorClass, re.Pattern[str]]] = [
    (ErrorClass.MISSING_INCLUDE, re.compile(
        r"(no such file or directory|fatal error:.*\.h)", re.IGNORECASE,
    )),
    (ErrorClass.UNKNOWN_TYPE, re.compile(
        r"(unknown type name|incomplete type|forward declaration of "
        r"|has incomplete type|tentative definition has type)",
        re.IGNORECASE,
    )),
    (ErrorClass.MACRO_MISMATCH, re.compile(
        r"(macro .* (passed|requires) \d+ arguments?|"
        r"redefined|previous definition|"
        r"undefined macro|"
        r"used as a macro)",
        re.IGNORECASE,
    )),
    (ErrorClass.SIGNATURE_MISMATCH, re.compile(
        r"(conflicting types for|"
        r"too (many|few) arguments to function|"
        r"incompatible (pointer )?types?|"
        r"passing argument \d+ of|"
        r"expected .* but argument is of type|"
        r"makes pointer from integer)",
        re.IGNORECASE,
    )),
    (ErrorClass.QUALIFIER_MISMATCH, re.compile(
        r"(discards .* qualifiers|"
        r"const qualifier|"
        r"volatile qualifier|"
        r"alignment of|"
        r"packed)",
        re.IGNORECASE,
    )),
    (ErrorClass.ENUM_VALUE_DRIFT, re.compile(
        r"(enumeration value .* not handled|"
        r"`[A-Z_][A-Z0-9_]+'\s+undeclared|"
        r"used in comparison)",
    )),
    (ErrorClass.STRUCT_LAYOUT, re.compile(
        r"(no member named|"
        r"has no member|"
        r"`struct [^']+' has no member|"
        r"static assertion failed.*sizeof|"
        r"offsetof)",
        re.IGNORECASE,
    )),
    (ErrorClass.CALLING_CONVENTION, re.compile(
        r"(attribute.*ignored|"
        r"calling convention|"
        r"__attribute__\(\(.*\)\))",
        re.IGNORECASE,
    )),
]


def classify_error(err: BuildError) -> ErrorClass:
    if err.tool == "link":
        return ErrorClass.LINK_UNDEFINED
    msg = err.message
    for klass, pat in _CLASSIFIERS:
        if pat.search(msg):
            return klass
    return ErrorClass.OTHER


# ---------------------------------------------------------------------------
# Cascade clustering + root-cause ranking (B)
# ---------------------------------------------------------------------------


def cluster_errors(
    errors: list[BuildError],
    index: "CodebaseIndex | None" = None,
) -> list[ErrorCluster]:
    """Group errors by likely root cause and rank by centrality.

    Strategy:

    1. Errors of the *same class* sharing the *same symbol* fall into one
       cluster — one missing typedef commonly produces dozens of
       downstream errors all naming the same identifier.
    2. Otherwise group by (file, error_class).
    3. Centrality is computed from the number of files that reference the
       cluster's representative symbol (when known), plus a small bias
       toward errors that appear early in the build log (closer to
       compile-stop) because GCC frequently aborts after the first
       fatal error per translation unit.
    """
    if not errors:
        return []

    # Annotate
    for e in errors:
        if e.error_class == ErrorClass.OTHER:
            e.error_class = classify_error(e)

    buckets: dict[tuple[str, ErrorClass, str], list[BuildError]] = {}
    for idx, e in enumerate(errors):
        if e.symbol:
            key = ("sym", e.error_class, e.symbol)
        else:
            key = ("file", e.error_class, e.file)
        buckets.setdefault(key, []).append(e)
        # Stash original index so we can compute "earliness" later.
        e.__dict__.setdefault("_order", idx)

    clusters: list[ErrorCluster] = []
    for (kind, klass, key), members in buckets.items():
        rep = members[0]
        files = sorted({m.file for m in members if m.file})
        centrality = len(members)
        if index is not None and rep.symbol:
            centrality += len(index.references_to(rep.symbol)) * 2
        # Earlier-in-log bias: smaller _order ≈ earlier compile stop.
        order = min(getattr(m, "_order", 0) for m in members)
        centrality += max(0, 50 - order)

        suggestion = _suggest_fix(klass, rep, files)
        if kind == "sym":
            root_cause = f"{klass.value}: symbol '{rep.symbol}'"
        else:
            root_cause = f"{klass.value} in {rep.file or '<linker>'}"

        clusters.append(
            ErrorCluster(
                root_cause=root_cause,
                error_class=klass,
                representative=rep,
                members=members,
                files=files,
                centrality=centrality,
                suggestion=suggestion,
            )
        )

    clusters.sort(
        key=lambda c: (
            _priority_for_error_class(c.error_class),
            -c.centrality,
            c.root_cause,
        )
    )
    return clusters


def _suggest_fix(klass: ErrorClass, rep: BuildError, files: list[str]) -> str:
    """Prescriptive guidance for the planner (policy table from spec D)."""
    if klass == ErrorClass.MISSING_INCLUDE:
        return (
            "Add the missing #include or restore an alias header. Prefer a "
            "compatibility shim in a header rather than touching call sites."
        )
    if klass == ErrorClass.UNKNOWN_TYPE:
        return (
            "Declare or typedef the missing type. If the type was renamed in "
            "the new ICD, add a typedef bridge `typedef NewType OldType;` so "
            "existing call sites keep working."
        )
    if klass == ErrorClass.MACRO_MISMATCH:
        return (
            "Restore or re-define the macro with the new value/width. If "
            "argument count changed, add a wrapper macro that bridges old "
            "→ new."
        )
    if klass == ErrorClass.SIGNATURE_MISMATCH:
        return (
            "Introduce an adapter function with the OLD signature that "
            "internally calls the NEW signature. Do NOT rewrite call sites "
            "until the build is green."
        )
    if klass == ErrorClass.LINK_UNDEFINED:
        return (
            "Map undefined references via linker-safe wrappers/adapters and "
            "resolve ISR/handler symbols explicitly. Under the Xilinx profile, "
            "compile-only pass is insufficient; full link must succeed."
        )
    if klass == ErrorClass.QUALIFIER_MISMATCH:
        return (
            "Adjust const/volatile/packed qualifiers in declarations. "
            "Prefer a typedef change in the header over per-call casts."
        )
    if klass == ErrorClass.ENUM_VALUE_DRIFT:
        return (
            "Re-add the renamed enum value or provide a #define alias from "
            "old name to new name."
        )
    if klass == ErrorClass.STRUCT_LAYOUT:
        return (
            "Restore the missing struct member, or update the field name. "
            "Add static_assert(sizeof(...)==EXPECTED) to validate layout."
        )
    if klass == ErrorClass.CALLING_CONVENTION:
        return (
            "Match the function's __attribute__/calling convention to the "
            "new ICD."
        )
    return "Inspect the offending line and apply the smallest fix that "\
           "removes the diagnostic without changing behaviour."


def errors_fingerprint(errors: list[BuildError]) -> str:
    """Stable signature of the *set* of errors (paths/lines normalised)."""
    items = sorted({
        f"{e.error_class.value}|{e.symbol}|{re.sub(r'\\d+', '#', e.message)}"
        for e in errors
    })
    return "\n".join(items)


_FIRMWARE_PATH_HINTS = (
    "/bsp/", "/drivers/", "/driver/", "/hal/", "/hw/", "/mmio/", "/isr/",
    "/interrupt", "/platform", "/startup", "/boot", "/xil", "/xilinx",
    "/ps7", "/periph", "/peripheral", "/standalone",
)
_SOFTWARE_PATH_HINTS = (
    "/app/", "/application/", "/logic/", "/state/", "/workflow/", "/service/",
    "/controller/", "/module/",
)
_FIRMWARE_NAME_HINTS = (
    "xparameters", "xil_", "isr", "irq", "handler", "startup", "ps7", "bsp",
    "lscript", ".ld", ".lds",
)
_SOFTWARE_NAME_HINTS = (
    "app_", "state_", "workflow_", "logic_", "controller_",
)

_ISR_SYMBOL_RE = re.compile(r"(?:^|_)(?:isr|irq|handler)(?:_|$)", re.IGNORECASE)
_SECTION_DRIFT_RE = re.compile(
    r"(section .* will not fit|region .* overflowed|"
    r"cannot move location counter|\.text|\.data|\.bss|heap|stack)",
    re.IGNORECASE,
)
_DUP_SYMBOL_RE = re.compile(r"(multiple definition of|first defined here)", re.IGNORECASE)
_BSP_FILE_RE = re.compile(r"(xparameters.*\.h|xil_.*\.h|/bsp/|/ps7/)", re.IGNORECASE)
_BSP_BUILD_RE = re.compile(r"(bsp|standalone|libxil|xparameters)", re.IGNORECASE)

_SUPERLOOP_HEAD_RE = re.compile(r"(while\s*\(\s*1\s*\)|for\s*\(\s*;\s*;\s*\))")
_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_MODULO_RE = re.compile(r"\b([A-Za-z_]\w*)\s*%\s*(\d+)")
_BLOCKING_CALL_RE = re.compile(
    r"\b(sleep|usleep|nanosleep|delay|delay_ms|delay_us|"
    r"read|recv|accept|select|poll|sem_wait|pthread_join)\s*\(",
    re.IGNORECASE,
)

_C_KEYWORDS = {
    "if", "for", "while", "switch", "return", "sizeof", "typedef",
    "struct", "enum", "union", "do", "case",
}


@dataclass
class LaneDiagnostics:
    lane: Lane
    errors: list[BuildError]
    clusters: list[ErrorCluster]
    fingerprint: str


@dataclass
class LaneSplitResult:
    firmware: LaneDiagnostics
    software: LaneDiagnostics
    active_lane: Lane
    active_errors: list[BuildError]
    active_clusters: list[ErrorCluster]


def _priority_for_error_class(klass: ErrorClass) -> int:
    return _REPAIR_FAMILY_PRIORITY.get(klass, 99)


def _lane_for_path(path: str) -> Lane:
    pl = path.replace("\\", "/").lower()
    parts = [x for x in pl.split("/") if x]
    if any(p in {"app", "application", "logic", "state", "workflow", "service", "controller", "module"} for p in parts):
        return Lane.SOFTWARE
    if any(p in {"bsp", "drivers", "driver", "hal", "hw", "mmio", "isr", "interrupt", "platform", "startup", "boot", "xil", "xilinx", "ps7", "periph", "peripheral", "standalone"} for p in parts):
        return Lane.FIRMWARE
    if any(h in pl for h in _FIRMWARE_PATH_HINTS):
        return Lane.FIRMWARE
    if any(h in pl for h in _SOFTWARE_PATH_HINTS):
        return Lane.SOFTWARE
    name = Path(pl).name
    if any(h in name for h in _FIRMWARE_NAME_HINTS):
        return Lane.FIRMWARE
    if any(h in name for h in _SOFTWARE_NAME_HINTS):
        return Lane.SOFTWARE
    # In Xilinx superloop repos, default unknowns to firmware lane (hard gate).
    return Lane.FIRMWARE


def _lane_for_link_symbol(symbol: str, message: str = "") -> Lane:
    s = (symbol or "").lower()
    m = (message or "").lower()
    if (
        _ISR_SYMBOL_RE.search(symbol or "")
        or s.startswith(("xil_", "xscu", "xuart", "xgpio", "xspi", "xadc", "xintc"))
        or "interrupt" in s
        or "handler" in s
    ):
        return Lane.FIRMWARE
    if any(k in s for k in ("app", "state", "logic", "workflow", "dispatch")):
        return Lane.SOFTWARE
    if "undefined reference" in m:
        return Lane.FIRMWARE
    return Lane.FIRMWARE


def split_errors_by_lane(errors: list[BuildError]) -> tuple[list[BuildError], list[BuildError]]:
    firmware: list[BuildError] = []
    software: list[BuildError] = []
    for e in errors:
        if e.file:
            lane = _lane_for_path(e.file)
        else:
            lane = _lane_for_link_symbol(e.symbol, e.message)
        if lane == Lane.FIRMWARE:
            firmware.append(e)
        else:
            software.append(e)
    return firmware, software


def evaluate_lanes(errors: list[BuildError], index: "CodebaseIndex | None" = None) -> LaneSplitResult:
    firmware_errors, software_errors = split_errors_by_lane(errors)
    firmware_clusters = cluster_errors(firmware_errors, index=index)
    software_clusters = cluster_errors(software_errors, index=index)
    firmware = LaneDiagnostics(
        lane=Lane.FIRMWARE,
        errors=firmware_errors,
        clusters=firmware_clusters,
        fingerprint=errors_fingerprint(firmware_errors),
    )
    software = LaneDiagnostics(
        lane=Lane.SOFTWARE,
        errors=software_errors,
        clusters=software_clusters,
        fingerprint=errors_fingerprint(software_errors),
    )
    if firmware_errors:
        active_lane = Lane.FIRMWARE
        active_errors = firmware_errors
        active_clusters = firmware_clusters
    else:
        active_lane = Lane.SOFTWARE
        active_errors = software_errors
        active_clusters = software_clusters
    return LaneSplitResult(
        firmware=firmware,
        software=software,
        active_lane=active_lane,
        active_errors=active_errors,
        active_clusters=active_clusters,
    )


def _is_compile_only_success(build_output: str) -> bool:
    return "Compile-only PASSED" in (build_output or "")


def _linker_errors_present(errors: list[BuildError]) -> bool:
    return any(e.error_class == ErrorClass.LINK_UNDEFINED for e in errors)


def unresolved_icd_deltas(errors: list[BuildError]) -> bool:
    """Whether unresolved ICD-mapped type/macro/symbol drifts remain."""
    blockers = {
        ErrorClass.MISSING_INCLUDE,
        ErrorClass.UNKNOWN_TYPE,
        ErrorClass.MACRO_MISMATCH,
        ErrorClass.SIGNATURE_MISMATCH,
        ErrorClass.STRUCT_LAYOUT,
        ErrorClass.QUALIFIER_MISMATCH,
        ErrorClass.ENUM_VALUE_DRIFT,
    }
    return any(e.error_class in blockers for e in errors)


def run_linker_bsp_consistency_audits(
    *,
    build_output: str,
    errors: list[BuildError],
    sandbox_dir: Path,
    touched_files: Iterable[Path],
    baseline_snapshots: dict[Path, str],
) -> list[str]:
    """Per-attempt Xilinx linker/BSP consistency audits."""
    audits: list[str] = []

    unresolved_isr = sorted({
        e.symbol for e in errors
        if e.error_class == ErrorClass.LINK_UNDEFINED
        and (_ISR_SYMBOL_RE.search(e.symbol or "") or "handler" in (e.symbol or "").lower())
    })
    if unresolved_isr:
        audits.append(
            "Unresolved ISR/handler symbols: "
            + ", ".join(unresolved_isr[:12])
            + (" …" if len(unresolved_isr) > 12 else "")
        )

    sec_hits = []
    for ln in (build_output or "").splitlines():
        if _SECTION_DRIFT_RE.search(ln):
            sec_hits.append(ln.strip())
    if sec_hits:
        audits.append(
            "Section placement/size drift detected: "
            + " | ".join(sec_hits[:3])
            + (" …" if len(sec_hits) > 3 else "")
        )

    dup_hits = []
    for ln in (build_output or "").splitlines():
        if _DUP_SYMBOL_RE.search(ln):
            dup_hits.append(ln.strip())
    if dup_hits:
        audits.append(
            "Duplicate symbol conflict detected (possible wrapper collision): "
            + " | ".join(dup_hits[:2])
            + (" …" if len(dup_hits) > 2 else "")
        )

    touched = list(touched_files)
    for p in touched:
        rel = str(p.relative_to(sandbox_dir)) if p.is_absolute() else str(p)
        if not _BSP_FILE_RE.search(rel.replace("\\", "/")):
            continue
        if not p.exists() or not p.is_file():
            continue
        before = baseline_snapshots.get(p)
        if before is None:
            continue
        try:
            after = p.read_text(errors="replace")
        except OSError:
            continue
        if after != before:
            audits.append(
                f"BSP-generated header divergence candidate: {rel} "
                "(differs from baseline snapshot)"
            )

    if _BSP_BUILD_RE.search(build_output or "") and not any("BSP" in a for a in audits):
        audits.append("BSP/linker diagnostics present in build log; inspect generated headers and linker script.")
    return audits


def _find_matching_brace(text: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _extract_first_superloop_body(text: str) -> str:
    m = _SUPERLOOP_HEAD_RE.search(text or "")
    if not m:
        return ""
    open_idx = text.find("{", m.end())
    if open_idx < 0:
        return ""
    close_idx = _find_matching_brace(text, open_idx)
    if close_idx < 0:
        return ""
    return text[open_idx + 1:close_idx]


def _phase_for_call(name: str) -> str | None:
    n = name.lower()
    if any(k in n for k in ("init", "setup", "config", "start")):
        return "init"
    if any(k in n for k in ("poll", "sample", "read", "tick", "scan")):
        return "poll"
    if any(k in n for k in ("process", "update", "compute", "handle")):
        return "process"
    if any(k in n for k in ("dispatch", "send", "publish", "write", "emit", "tx")):
        return "dispatch"
    return None


def _superloop_phase_sequence(loop_body: str) -> list[str]:
    seq: list[str] = []
    for m in _CALL_RE.finditer(loop_body or ""):
        fn = m.group(1)
        if fn in _C_KEYWORDS:
            continue
        phase = _phase_for_call(fn)
        if phase and phase not in seq:
            seq.append(phase)
    return seq


def _extract_modulo_schedule(loop_body: str) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for m in _MODULO_RE.finditer(loop_body or ""):
        counter = m.group(1)
        freq = int(m.group(2))
        out.setdefault(counter, set()).add(freq)
    return out


def _extract_blocking_calls(loop_body: str) -> set[str]:
    return {m.group(1).lower() for m in _BLOCKING_CALL_RE.finditer(loop_body or "")}


def run_superloop_contract_checks(
    *,
    sandbox_dir: Path,
    patch_result: PatchResult,
) -> tuple[list[str], list[str]]:
    """Validate superloop ordering/cadence/non-blocking contracts."""
    errors: list[str] = []
    warnings: list[str] = []
    phase_idx = {"init": 0, "poll": 1, "process": 2, "dispatch": 3}

    for path in patch_result.applied:
        if path.suffix not in (".c", ".h"):
            continue
        before = patch_result.snapshot_before.get(path) or ""
        try:
            after = path.read_text(errors="replace")
        except OSError:
            continue
        before_loop = _extract_first_superloop_body(before)
        after_loop = _extract_first_superloop_body(after)
        if not after_loop and not before_loop:
            continue

        rel = str(path.relative_to(sandbox_dir))

        # Contract 1: call-ordering (init -> poll -> process -> dispatch)
        after_seq = _superloop_phase_sequence(after_loop)
        if all(p in after_seq for p in ("init", "poll", "process", "dispatch")):
            ordered = sorted(after_seq, key=lambda p: phase_idx[p])
            if after_seq != ordered:
                errors.append(
                    f"{rel}: superloop ordering changed; expected init->poll->process->dispatch, got {'->'.join(after_seq)}"
                )
        before_seq = _superloop_phase_sequence(before_loop)
        common = [p for p in before_seq if p in after_seq]
        if len(common) >= 2:
            for i in range(len(common) - 1):
                a = common[i]
                b = common[i + 1]
                if after_seq.index(a) > after_seq.index(b):
                    errors.append(
                        f"{rel}: superloop relative phase order changed ({a} now after {b})"
                    )
                    break

        # Contract 2: cadence via modulo schedules.
        before_sched = _extract_modulo_schedule(before_loop)
        after_sched = _extract_modulo_schedule(after_loop)
        for counter, old_freqs in before_sched.items():
            if counter in after_sched and after_sched[counter] != old_freqs:
                errors.append(
                    f"{rel}: scheduling cadence changed for '{counter}' "
                    f"from {sorted(old_freqs)} to {sorted(after_sched[counter])}"
                )

        # Contract 3: no new blocking behavior inside superloop.
        before_blk = _extract_blocking_calls(before_loop)
        after_blk = _extract_blocking_calls(after_loop)
        new_blk = sorted(after_blk - before_blk)
        if new_blk:
            errors.append(
                f"{rel}: new blocking call(s) introduced in superloop: {', '.join(new_blk)}"
            )

        if not before_loop and after_loop:
            warnings.append(
                f"{rel}: superloop detected for first time in touched file; review call ordering and cadence manually."
            )

    return errors, warnings


def run_touched_module_regression_checks(
    *,
    sandbox_dir: Path,
    touched_files: Iterable[Path],
) -> list[str]:
    findings: list[str] = []
    for p in touched_files:
        if not p.exists() or not p.is_file():
            continue
        try:
            txt = p.read_text(errors="replace")
        except OSError:
            continue
        rel = str(p.relative_to(sandbox_dir)) if p.is_absolute() else str(p)
        if p.suffix in (".c", ".h"):
            if txt.count("{") != txt.count("}"):
                findings.append(f"{rel}: unbalanced braces after patch")
        if p.suffix == ".h":
            if not _INCLUDE_GUARD_RE.search(txt) and "#pragma once" not in txt:
                findings.append(f"{rel}: missing include guard / #pragma once after patch")
    return findings


def _capture_checkpoint(build_root: Path) -> dict[Path, str]:
    checkpoint: dict[Path, str] = {}
    interesting = {".c", ".h", ".ld", ".lds", ".s", ".S", ".inc"}
    for p in build_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in interesting:
            continue
        try:
            checkpoint[p] = p.read_text(errors="replace")
        except OSError:
            continue
    return checkpoint


def _restore_checkpoint(checkpoint: dict[Path, str]) -> None:
    for p, content in checkpoint.items():
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        except OSError:
            log.warning("checkpoint restore: failed for %s", p)


def _is_lane_oscillation(history: list[Lane]) -> bool:
    if len(history) < 3:
        return False
    a, b, c = history[-3], history[-2], history[-1]
    return a == c and a != b


# ---------------------------------------------------------------------------
# Codebase Context Service
# ---------------------------------------------------------------------------


_DECL_RE = re.compile(
    r"\b(?:typedef|struct|enum|union)\s+([A-Za-z_]\w*)|"
    r"\b([A-Za-z_]\w*)\s*\([^;{]*\)\s*\{",
)
_DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)", re.MULTILINE)


@dataclass
class CodebaseIndex:
    sandbox_dir: Path
    file_index: dict[str, list[Path]] = field(default_factory=dict)
    symbol_to_files: dict[str, list[Path]] = field(default_factory=dict)
    file_to_symbols: dict[Path, list[str]] = field(default_factory=dict)

    @classmethod
    def build(
        cls, sandbox_dir: Path, build_root: Path,
        max_files: int = DEFAULT_INDEX_FILE_LIMIT,
    ) -> "CodebaseIndex":
        idx = cls(sandbox_dir=sandbox_dir)
        files = 0
        for p in build_root.rglob("*.[ch]"):
            if not p.is_file():
                continue
            if files >= max_files:
                log.warning(
                    "CodebaseIndex: capping at %d files; remainder ignored",
                    max_files,
                )
                break
            files += 1
            idx.file_index.setdefault(p.name, []).append(p)
            try:
                txt = p.read_text(errors="replace")
            except OSError:
                continue
            symbols: list[str] = []
            for m in _DECL_RE.finditer(txt):
                sym = m.group(1) or m.group(2) or ""
                if sym:
                    symbols.append(sym)
            for m in _DEFINE_RE.finditer(txt):
                symbols.append(m.group(1))
            symbols = list({s for s in symbols if s})
            idx.file_to_symbols[p] = symbols
            for sym in symbols:
                idx.symbol_to_files.setdefault(sym, []).append(p)
        return idx

    def references_to(self, symbol: str) -> list[Path]:
        """Naive references count via file_to_symbols+grep fallback.

        For ranking we only need an order-of-magnitude estimate, so a quick
        linear pass over the indexed text is sufficient.
        """
        if not symbol or len(symbol) < 2:
            return []
        if symbol in self.symbol_to_files:
            return list(self.symbol_to_files[symbol])
        # Fallback: grep symbol in every indexed file (cheap on small sandbox).
        hits: list[Path] = []
        pat = re.compile(rf"\b{re.escape(symbol)}\b")
        for p in self.file_to_symbols.keys():
            try:
                if pat.search(p.read_text(errors="replace")):
                    hits.append(p)
            except OSError:
                continue
        return hits


# ---------------------------------------------------------------------------
# Pre-build verification (Verification Suite — F)
# ---------------------------------------------------------------------------


_INCLUDE_GUARD_RE = re.compile(
    r"#ifndef\s+([A-Za-z_]\w*)\s*\n\s*#define\s+\1\b"
)


def run_pre_build_checks(
    sandbox_dir: Path, touched: Iterable[Path],
) -> list[str]:
    """Lightweight static checks before each rebuild.

    Returns a list of human-readable warnings (not blocking).  These are
    folded back into the planner's prompt so the next hypothesis can
    address them proactively.
    """
    warnings: list[str] = []
    for p in touched:
        if not p.is_file():
            continue
        if p.suffix != ".h":
            continue
        try:
            txt = p.read_text(errors="replace")
        except OSError:
            continue
        if not _INCLUDE_GUARD_RE.search(txt) and "#pragma once" not in txt:
            warnings.append(
                f"{p.relative_to(sandbox_dir)}: missing include-guard / #pragma once"
            )
    return warnings


# ---------------------------------------------------------------------------
# Memory store (Memory & Learning Store)
# ---------------------------------------------------------------------------


class Playbook:
    """Append-only JSONL of fix outcomes for the current session.

    A successful (cluster_signature → patch_kind) pair becomes a hint for
    later iterations and future runs.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: list[dict] = []
        if path.exists():
            try:
                for ln in path.read_text(errors="replace").splitlines():
                    if ln.strip():
                        self.entries.append(json.loads(ln))
            except (OSError, json.JSONDecodeError):
                self.entries = []

    def record(self, entry: dict) -> None:
        self.entries.append(entry)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            log.warning("Playbook: failed to persist entry to %s", self.path)

    def hints_for(self, error_class: ErrorClass, symbol: str = "") -> list[str]:
        """Return short suggestions from past successful fixes."""
        hints: list[str] = []
        for e in self.entries:
            if not e.get("success"):
                continue
            if e.get("error_class") != error_class.value:
                continue
            if symbol and e.get("symbol") and symbol != e["symbol"]:
                continue
            kind = e.get("patch_kind") or "unknown"
            hints.append(
                f"Previously fixed {error_class.value}"
                + (f" for symbol '{symbol}'" if symbol else "")
                + f" via {kind}: {e.get('summary', '')[:120]}"
            )
        return hints[-5:]


# ---------------------------------------------------------------------------
# Hypothesis schema (Patch Planner)
# ---------------------------------------------------------------------------


@dataclass
class Hypothesis:
    """One actionable fix proposal — exactly one per iteration."""

    id: str                       # short label, e.g. "H1"
    layer: str                    # "shim" | "semantic" | "cleanup"
    kind: str                     # patch_kind, e.g. "header_typedef_bridge"
    title: str
    rationale: str                # why this addresses the highest-ranked cluster
    target_files: list[str]       # repo-relative paths (existing or new)
    risk: str                     # "low" | "medium" | "high"
    expected_resolved: list[str]  # cluster root_causes expected to be removed
    notes: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Planner / Patcher prompts
# ---------------------------------------------------------------------------


_PLANNER_SYSTEM_PROMPT = """You are the **Fix Strategy Agent** of an agentic
debugging pipeline for embedded C / Xilinx-SDK 2018.x (GCC 7.3.1, newlib,
C99).

Your job is to read a structured build report and propose **exactly ONE**
hypothesis that fixes the *highest-priority root cause* with the smallest
possible blast radius.

## Strict rules

1. Output a single JSON object inside a ```json ... ``` fenced block —
   nothing else.
2. Schema (all fields required):

   ```json
   {
     "id": "H1",
     "layer": "shim" | "semantic" | "cleanup",
     "kind": "<short snake_case patch kind>",
     "title": "<one-line summary>",
     "rationale": "<why this addresses the highest-ranked cluster>",
     "target_files": ["repo/relative/path.h", ...],
     "risk": "low" | "medium" | "high",
     "expected_resolved": ["root_cause_id_1", ...],
     "notes": "<optional>"
   }
   ```

3. **One change hypothesis per turn.** Do NOT batch unrelated fixes.
4. Prefer in this order: compatibility adapters/shims → header typedef
   /macro bridges → wrapper functions → deep call-site rewrites
   (last resort).
5. Respect lane gating:
   - Lane A (firmware) is a hard gate.
   - If firmware lane still has errors, do not emit software-lane hypotheses.
6. The current correction layer is provided — you must respect it.
   - Layer **shim**: build-breaker fixes only (make it compile/link with a
     compatibility layer).
   - Layer **semantic**: replace shims with precise mappings from the
     target ICD.
   - Layer **cleanup**: remove dead aliases, tighten types.
7. If strategy mode says `second_choice`, avoid the #1 ranked cluster and
   target the second-best feasible cluster/hypothesis instead.
8. Keep touched files very small (target <=3 files) unless impossible.
9. If no further hypothesis is reasonable (build is green or you would
   make things worse), set `"kind": "stop"` and explain in `notes`.
"""


_PATCHER_SYSTEM_PROMPT = """You are the **Patch Agent** of the agentic
debugging pipeline.

You are given:

* a single Hypothesis (already approved)
* the current contents of each file in `target_files`
* the ICD change spec
* the structured error clusters
* the active correction layer (shim / semantic / cleanup)

Your job: produce **the new full contents** of every file listed in the
hypothesis' `target_files`, applying the *minimum* edit that realises the
hypothesis.  Never modify files outside `target_files`.

## Output format (STRICT)

For each file, output one fenced block with the path on its own header
line:

```
### path/to/file.c
```c
<full new file content>
```
```

Repeat for every target file.  No preamble, no postscript, no diff
markers.  Each `### path/to/file.ext` MUST exactly match an entry from
`target_files` (so the applier can safely route content).

## Constraints

* Target compiler: GCC 7.3.1 (Xilinx SDK 2018.x), arm-none-eabi or
  mb-gcc, C99, newlib (no glibc, no POSIX).
* Keep the existing architecture and naming style.
* Do NOT remove unrelated code, comments, or includes.
* Ensure `#include` paths still resolve.
* For new files, use the path the hypothesis listed and provide a sane
  include-guard.
* Preserve superloop behavior: do not reorder init/poll/process/dispatch,
  do not change modulo-scheduling cadence, and do not introduce blocking
  calls in superloop paths.
* Prefer compatibility aliases/wrappers over broad rewrites.
"""


def _planner_prompt(
    *,
    layer: str,
    attempt: int,
    max_attempts: int,
    metrics_history: list[dict],
    lane_result: LaneSplitResult,
    pre_build_warnings: list[str],
    linker_bsp_audits: list[str],
    superloop_findings: list[str],
    sandbox_cc: str,
    is_cross: bool,
    change_spec: str,
    repo_knowledge: str,
    playbook_hints: list[str],
    strategy_mode: str,
    compile_only_success_streak: int,
    gitnexus_report: str = "",
) -> str:
    clusters = lane_result.active_clusters
    cluster_blob = json.dumps(
        [c.to_json() for c in clusters[:8]],
        indent=2, ensure_ascii=False,
    )
    fw_blob = json.dumps(
        [c.to_json() for c in lane_result.firmware.clusters[:6]],
        indent=2, ensure_ascii=False,
    )
    sw_blob = json.dumps(
        [c.to_json() for c in lane_result.software.clusters[:6]],
        indent=2, ensure_ascii=False,
    )
    metrics_tail = metrics_history[-5:]
    metrics_blob = json.dumps(metrics_tail, indent=2, ensure_ascii=False)
    hints = "\n".join(f"- {h}" for h in playbook_hints) or "(none yet)"
    warnings = "\n".join(f"- {w}" for w in pre_build_warnings) or "(none)"
    audits = "\n".join(f"- {a}" for a in linker_bsp_audits) or "(none)"
    superloop = "\n".join(f"- {s}" for s in superloop_findings) or "(none)"
    cross_note = (
        "Cross-compile sandbox is in use. BSP linker errors (Xil_*, "
        "xil_printf, etc.) are EXPECTED and resolved by compile-only "
        "fallback in generic mode, but this Xilinx profile requires full "
        "build green before DONE."
        if is_cross else "Native compile."
    )
    return (
        f"## Build pipeline state\n"
        f"- Attempt: {attempt}/{max_attempts}\n"
        f"- Active correction layer: {layer}\n"
        f"- Compiler: {sandbox_cc}\n"
        f"- Active verification lane: {lane_result.active_lane.value}\n"
        f"- Lane-A firmware errors: {len(lane_result.firmware.errors)}\n"
        f"- Lane-B software errors: {len(lane_result.software.errors)}\n"
        f"- Strategy mode: {strategy_mode}\n"
        f"- Compile-only success streak: {compile_only_success_streak}\n"
        f"- {cross_note}\n\n"
        "## Deterministic repair ordering (MUST follow)\n"
        "1) Missing/renamed headers/macros/types\n"
        "2) Function prototype/signature drift\n"
        "3) Struct/register field drift + volatile correctness\n"
        "4) Undefined references / linker symbol mapping\n"
        "5) Warning hardening and cleanup\n\n"
        "## Lane gating rule\n"
        "- Solve Lane-A firmware completely before Lane-B software.\n"
        "- Do NOT propose software-lane edits while firmware lane still has errors.\n\n"
        f"## ICD change spec (truncated)\n{change_spec[:4000]}\n\n"
        f"## Repo knowledge (truncated)\n{repo_knowledge[:2000]}\n\n"
        + (
            f"## GitNexus codebase understanding (truncated)\n"
            f"Embedded-systems relationships extracted from the repo "
            f"(ISR/task wiring, drivers, RTOS/superloop, state machines, "
            f"comm stacks, memory ownership, HAL boundary, bootloader, "
            f"FW update, safety chains, cross-module deps, global var "
            f"graph, script deps). Honor these when patching.\n"
            f"{gitnexus_report[:3000]}\n\n"
            if gitnexus_report else ""
        )
        + f"## Recent metrics history\n```json\n{metrics_blob}\n```\n\n"
        "## Lane-A firmware clusters\n"
        f"```json\n{fw_blob}\n```\n\n"
        "## Lane-B software clusters\n"
        f"```json\n{sw_blob}\n```\n\n"
        "## Active-lane clusters (what you should fix now)\n"
        f"```json\n{cluster_blob}\n```\n\n"
        f"## Pre-build static warnings\n{warnings}\n\n"
        f"## Linker/BSP consistency audits\n{audits}\n\n"
        f"## Superloop validation findings\n{superloop}\n\n"
        f"## Playbook hints (memory of past successful fixes)\n{hints}\n\n"
        "## Patch priority for Xilinx SDK 2018\n"
        "1. Header-level macro/type aliases\n"
        "2. Wrapper functions preserving old signatures\n"
        "3. Isolated per-peripheral translation units\n"
        "4. Broad call-site rewrites only if compatibility is impossible\n\n"
        "## Your turn\n"
        "Produce one Hypothesis JSON now. Keep touched files minimal "
        "(target <=3 files)."
    )


def _patcher_prompt(
    *,
    hypothesis: Hypothesis,
    file_contents: dict[str, str],
    layer: str,
    clusters: list[ErrorCluster],
    change_spec: str,
    gitnexus_report: str = "",
) -> str:
    files_blob_parts: list[str] = []
    for path, content in file_contents.items():
        snippet = content if len(content) <= 12_000 else (
            content[:6000] + "\n/* ... mid-file truncated ... */\n"
            + content[-3000:]
        )
        files_blob_parts.append(f"### {path}\n```c\n{snippet}\n```")
    files_blob = "\n\n".join(files_blob_parts) or "(target file does not yet exist — create it)"

    cluster_blob = json.dumps(
        [c.to_json() for c in clusters[:6]],
        indent=2, ensure_ascii=False,
    )
    gitnexus_section = (
        f"## GitNexus codebase understanding (truncated)\n"
        f"Embedded-systems relationships across the repository. Avoid "
        f"breaking ISR/task wiring, shared globals, comm-stack callers, "
        f"state-machine dispatch tables and safety-critical paths.\n"
        f"{gitnexus_report[:3000]}\n\n"
        if gitnexus_report else ""
    )
    return (
        f"## Hypothesis\n```json\n{json.dumps(hypothesis.to_json(), indent=2)}\n```\n\n"
        f"## Active layer\n{layer}\n\n"
        f"## ICD change spec (truncated)\n{change_spec[:4000]}\n\n"
        f"{gitnexus_section}"
        f"## Top error clusters\n```json\n{cluster_blob}\n```\n\n"
        f"## Current target files\n{files_blob}\n\n"
        f"## Your turn\n"
        f"Output the full new content of EVERY file in `target_files`, "
        f"using the strict ### path / ```c block format."
    )


# ---------------------------------------------------------------------------
# Hypothesis parsing
# ---------------------------------------------------------------------------


_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n(\{.*?\})\s*\n```",
    re.DOTALL | re.IGNORECASE,
)


def parse_hypothesis(raw: str) -> Hypothesis | None:
    if not raw:
        return None
    m = _JSON_FENCE_RE.search(raw)
    blob = m.group(1) if m else raw.strip()
    # Try to recover from minor fence noise.
    if not blob.startswith("{"):
        first = blob.find("{")
        last = blob.rfind("}")
        if first == -1 or last == -1:
            return None
        blob = blob[first:last + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    try:
        return Hypothesis(
            id=str(data.get("id") or "H?"),
            layer=str(data.get("layer") or "shim").lower(),
            kind=str(data.get("kind") or "unknown"),
            title=str(data.get("title") or ""),
            rationale=str(data.get("rationale") or ""),
            target_files=[str(p) for p in (data.get("target_files") or [])],
            risk=str(data.get("risk") or "medium").lower(),
            expected_resolved=[
                str(x) for x in (data.get("expected_resolved") or [])
            ],
            notes=str(data.get("notes") or ""),
        )
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Patch parsing + applier
# ---------------------------------------------------------------------------


_PATCH_FILE_RE = re.compile(
    r"^###\s+([^\n]+?)\s*\n```[a-zA-Z]*\s*\n(.*?)\n```",
    re.DOTALL | re.MULTILINE,
)


def parse_patch(raw: str) -> dict[str, str]:
    """Map ``path -> new file content`` from a patcher response."""
    out: dict[str, str] = {}
    for m in _PATCH_FILE_RE.finditer(raw or ""):
        path = m.group(1).strip()
        content = m.group(2)
        if path:
            out[path] = content
    return out


@dataclass
class PatchResult:
    applied: list[Path]
    rejected: list[tuple[str, str]]  # (path, reason)
    diff: str
    snapshot_before: dict[Path, str | None]  # None means file did not exist


def apply_hypothesis(
    sandbox_dir: Path,
    hypothesis: Hypothesis,
    new_contents: dict[str, str],
) -> PatchResult:
    """Atomically apply a multi-file patch with snapshots for rollback.

    A path is *only* allowed when:
      * it appears in `hypothesis.target_files` (whitelist), AND
      * the resolved absolute path is inside ``sandbox_dir``.
    """
    snap: dict[Path, str | None] = {}
    applied: list[Path] = []
    rejected: list[tuple[str, str]] = []
    diff_lines: list[str] = []

    whitelist = {tf.strip() for tf in hypothesis.target_files if tf.strip()}

    for rel_path, content in new_contents.items():
        rel_norm = rel_path.lstrip("./").replace("\\", "/")
        if rel_norm not in whitelist and rel_path not in whitelist:
            rejected.append((rel_path, "not in hypothesis.target_files"))
            continue
        try:
            target = (sandbox_dir / rel_norm).resolve()
            target.relative_to(sandbox_dir.resolve())
        except (ValueError, OSError) as e:
            rejected.append((rel_path, f"path traversal: {e}"))
            continue

        before = target.read_text(errors="replace") if target.exists() else None
        snap[target] = before
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        except OSError as e:
            rejected.append((rel_path, f"write failed: {e}"))
            continue
        applied.append(target)
        diff_lines.append(_minidiff(rel_norm, before, content))

    return PatchResult(
        applied=applied,
        rejected=rejected,
        diff="\n".join(diff_lines),
        snapshot_before=snap,
    )


def rollback(snapshot_before: dict[Path, str | None]) -> None:
    """Undo a previously applied patch using its snapshot map."""
    for path, before in snapshot_before.items():
        try:
            if before is None:
                if path.exists():
                    path.unlink()
            else:
                path.write_text(before)
        except OSError:
            log.warning("Rollback: could not restore %s", path)


def _minidiff(rel_path: str, before: str | None, after: str) -> str:
    """Compact unified-diff-ish summary (line counts + first 60 lines)."""
    before_lines = (before or "").splitlines()
    after_lines = after.splitlines()
    header = (
        f"--- a/{rel_path}\n+++ b/{rel_path}\n"
        f"@@ -{len(before_lines)} +{len(after_lines)} @@"
    )
    sample: list[str] = []
    for ln in after_lines[:60]:
        sample.append("+" + ln)
    if len(after_lines) > 60:
        sample.append(f"+ ... [{len(after_lines) - 60} more lines]")
    return "\n".join([header, *sample])


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class Phase(str, Enum):
    BUILD = "build"
    TRIAGE = "triage"
    ROOT_CAUSE = "root_cause"
    PLAN = "plan"
    PATCH = "patch"
    SUPERLOOP = "superloop_validation"
    VERIFY = "verify"
    DECIDE = "decide"
    DONE = "done"


@dataclass
class AttemptMetrics:
    attempt: int
    phase: str
    errors_total: int
    error_families: int
    new_errors: int
    resolved_errors: int
    files_touched: int
    patch_size_lines: int
    duration_s: float
    layer: str
    hypothesis_id: str
    hypothesis_kind: str
    success: bool
    fingerprint: str


def _llm_complete(
    llm_stream: Callable[..., Iterator[str]],
    system: str,
    user: str,
    *,
    max_tokens: int,
    on_token: Callable[[str], None] | None = None,
) -> str:
    parts: list[str] = []
    meta: dict = {}
    for chunk in llm_stream(system, user, max_tokens=max_tokens, meta=meta):
        parts.append(chunk)
        if on_token is not None:
            on_token(chunk)
    return "".join(parts).strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_agentic_debug(
    *,
    session_dir: Path,
    sandbox_dir: Path,
    build_info: dict,
    sandbox_cc: str,
    is_cross: bool,
    gen_files: dict[str, str],            # {repo-rel-path: gen_filename}
    change_spec: str,
    repo_knowledge: str,
    gitnexus_report: str = "",
    file_index: dict[str, list[Path]],
    snapshots: dict[Path, str],
    build_runner: Callable[[], tuple[bool, str]],
    llm_stream: Callable[..., Iterator[str]],
    max_attempts: int | None = DEFAULT_MAX_ATTEMPTS,
    no_progress_limit: int = DEFAULT_NO_PROGRESS_LIMIT,
    oscillation_limit: int = DEFAULT_OSCILLATION_LIMIT,
    edit_budget_files: int = DEFAULT_EDIT_BUDGET_FILES,
    max_touched_files_per_attempt: int = DEFAULT_MAX_TOUCHED_FILES_PER_ATTEMPT,
    planner_max_tokens: int = DEFAULT_PLANNER_MAX_TOKENS,
    patch_max_tokens: int = DEFAULT_PATCH_MAX_TOKENS,
) -> Iterator[dict]:
    """Drive the agentic-debug state machine.

    ``max_attempts`` may be ``None`` to allow unlimited hypothesis attempts
    (used when the UI selects an indefinite sandbox retry budget).

    Yields events of the same shape as ``api.orchestrator.run_orchestrator``
    so the calling SSE adapter can stay (almost) unchanged:

        {"type": "step",        "step": int}
        {"type": "phase",       "step": int, "phase": str}
        {"type": "thought",     "step": int, "text": str}
        {"type": "action",      "step": int, "tool": str, "args": dict}
        {"type": "observation", "step": int, "text": str, "error": bool}
        {"type": "build",       "step": int, "success": bool, "calls": int}
        {"type": "raw_token",   "text": str}
        {"type": "warning",     "message": str}
        {"type": "done",        "success": bool, "reason": str,
                                "steps": int, "builds": int}
    """
    attempts_dir = session_dir / "agentic_attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    playbook = Playbook(session_dir / "playbook.jsonl")
    build_root = build_info.get("build_dir", sandbox_dir)
    index = CodebaseIndex.build(sandbox_dir, build_root)

    # Best-known checkpoint used by stall recovery.
    best_checkpoint = _capture_checkpoint(build_root)
    best_score: tuple[int, int, int] = (10**9, 10**9, 10**9)

    layer_order = ["shim", "semantic", "cleanup"]
    layer_idx = 0
    second_choice_mode = False
    metrics_history: list[dict] = []
    seen_fingerprints: dict[str, int] = {}
    lane_history: list[Lane] = []
    touched_recent: list[Path] = []
    no_progress = 0
    distinct_files_touched: set[Path] = set()
    compile_only_success_streak = 0
    superloop_fail_streak = 0
    last_superloop_findings: list[str] = []
    last_linker_bsp_audits: list[str] = []

    # ------------------------------------------------------------------ BUILD
    yield {"type": "phase", "step": 0, "phase": Phase.BUILD.value}
    ok, build_output = build_runner()
    build_calls = 1
    yield {"type": "build", "step": 0, "success": ok, "calls": build_calls}

    init_errors = parse_build_log(build_output)
    for e in init_errors:
        e.error_class = classify_error(e)
    init_real_errors = [e for e in init_errors if e.severity == "error"]
    init_lanes = evaluate_lanes(init_real_errors, index=index)
    init_compile_only = _is_compile_only_success(build_output)
    if init_compile_only and _linker_errors_present(init_real_errors):
        compile_only_success_streak = 1

    best_score = (
        len(init_lanes.firmware.errors),
        len(init_lanes.software.errors),
        len(init_real_errors),
    )

    full_build_ok = ok and not init_compile_only
    if (
        full_build_ok
        and not init_lanes.firmware.errors
        and not init_lanes.software.errors
        and not unresolved_icd_deltas(init_real_errors)
    ):
        _write_attempt_artifacts(
            attempts_dir / "attempt_00",
            raw_log=build_output, errors=init_real_errors, clusters=[],
            hypothesis=None, metrics=AttemptMetrics(
                attempt=0, phase=Phase.DONE.value, errors_total=0,
                error_families=0, new_errors=0, resolved_errors=0,
                files_touched=0, patch_size_lines=0, duration_s=0.0,
                layer="shim", hypothesis_id="-", hypothesis_kind="-",
                success=True, fingerprint=errors_fingerprint(init_real_errors),
            ),
            patch_diff="",
        )
        yield {
            "type": "done",
            "success": True,
            "reason": (
                "Initial build already satisfies Xilinx profile gates "
                "(full build + firmware lane + software lane)."
            ),
            "steps": 0,
            "builds": build_calls,
        }
        return

    if ok and init_compile_only:
        yield {
            "type": "observation",
            "step": 0,
            "text": (
                "Build returned compile-only success, but this Xilinx profile "
                "requires full link success before DONE."
            ),
            "error": True,
        }

    last_build_output = build_output

    attempt = 0
    while True:
        attempt += 1
        if max_attempts is not None and attempt > max_attempts:
            break
        attempt_start = time.monotonic()
        attempt_dir = attempts_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        yield {"type": "step", "step": attempt}

        # -------------------------------------------------------- TRIAGE
        yield {"type": "phase", "step": attempt, "phase": Phase.TRIAGE.value}
        errors = parse_build_log(last_build_output)
        for e in errors:
            e.error_class = classify_error(e)
        real_errors = [e for e in errors if e.severity == "error"]
        lane_result = evaluate_lanes(real_errors, index=index)
        if lane_result.active_errors:
            lane_history.append(lane_result.active_lane)

        fingerprint = errors_fingerprint(real_errors)
        seen_fingerprints[fingerprint] = seen_fingerprints.get(fingerprint, 0) + 1

        last_linker_bsp_audits = run_linker_bsp_consistency_audits(
            build_output=last_build_output,
            errors=real_errors,
            sandbox_dir=sandbox_dir,
            touched_files=touched_recent,
            baseline_snapshots=snapshots,
        )

        try:
            (attempt_dir / "build.log.raw").write_text(last_build_output)
            (attempt_dir / "build.log.structured.json").write_text(
                json.dumps([e.to_json() for e in errors], indent=2, ensure_ascii=False)
            )
            (attempt_dir / "lane_firmware_clusters.json").write_text(
                json.dumps([c.to_json() for c in lane_result.firmware.clusters], indent=2, ensure_ascii=False)
            )
            (attempt_dir / "lane_software_clusters.json").write_text(
                json.dumps([c.to_json() for c in lane_result.software.clusters], indent=2, ensure_ascii=False)
            )
            (attempt_dir / "linker_bsp_audits.txt").write_text(
                "\n".join(last_linker_bsp_audits) + ("\n" if last_linker_bsp_audits else "")
            )
        except OSError:
            pass

        active_clusters = lane_result.active_clusters
        if not active_clusters and real_errors:
            active_clusters = cluster_errors(real_errors, index=index)
        if not real_errors:
            yield {
                "type": "observation",
                "step": attempt,
                "text": (
                    "Build failed but no parseable diagnostics were produced. "
                    "Will continue with linker/BSP audits and planner hints."
                ),
                "error": True,
            }

        yield {
            "type": "observation",
            "step": attempt,
            "text": (
                "Lane triage: firmware="
                f"{len(lane_result.firmware.errors)} error(s), software="
                f"{len(lane_result.software.errors)} error(s). "
                f"Active lane: {lane_result.active_lane.value}."
            ),
            "error": False,
        }
        if last_linker_bsp_audits:
            yield {
                "type": "observation",
                "step": attempt,
                "text": "Linker/BSP audits: " + " | ".join(last_linker_bsp_audits[:3]),
                "error": False,
            }

        # -------------------------------------------------------- ROOT_CAUSE
        yield {"type": "phase", "step": attempt, "phase": Phase.ROOT_CAUSE.value}
        clusters = active_clusters
        try:
            (attempt_dir / "error_clusters.json").write_text(
                json.dumps([c.to_json() for c in clusters], indent=2, ensure_ascii=False)
            )
        except OSError:
            pass
        if not clusters:
            no_progress += 1
            if no_progress >= no_progress_limit:
                yield {
                    "type": "done",
                    "success": False,
                    "reason": "Unable to extract actionable error clusters repeatedly.",
                    "steps": attempt,
                    "builds": build_calls,
                }
                return
            continue

        # -------------------------------------------------------- PLAN
        yield {"type": "phase", "step": attempt, "phase": Phase.PLAN.value}
        layer = layer_order[layer_idx]
        playbook_hints: list[str] = []
        if clusters:
            playbook_hints = playbook.hints_for(
                clusters[0].error_class,
                clusters[0].representative.symbol,
            )
        prebuild_warnings = run_pre_build_checks(sandbox_dir, list(distinct_files_touched))
        planner_user = _planner_prompt(
            layer=layer,
            attempt=attempt,
            max_attempts=max_attempts,
            metrics_history=metrics_history,
            lane_result=lane_result,
            pre_build_warnings=prebuild_warnings,
            linker_bsp_audits=last_linker_bsp_audits,
            superloop_findings=last_superloop_findings,
            sandbox_cc=sandbox_cc,
            is_cross=is_cross,
            change_spec=change_spec,
            repo_knowledge=repo_knowledge,
            playbook_hints=playbook_hints,
            strategy_mode=("second_choice" if second_choice_mode else "primary"),
            compile_only_success_streak=compile_only_success_streak,
            gitnexus_report=gitnexus_report,
        )
        planner_raw_parts: list[str] = []

        def _on_planner_token(t: str) -> None:
            planner_raw_parts.append(t)

        try:
            planner_raw = _llm_complete(
                llm_stream,
                _PLANNER_SYSTEM_PROMPT,
                planner_user,
                max_tokens=planner_max_tokens,
                on_token=_on_planner_token,
            )
        except Exception as e:
            yield {"type": "warning", "message": f"planner LLM failed: {e}"}
            planner_raw = "".join(planner_raw_parts)

        for piece in _chunk_text(planner_raw, 1024):
            yield {"type": "raw_token", "text": piece}

        hypothesis = parse_hypothesis(planner_raw)
        if hypothesis is None:
            yield {
                "type": "observation",
                "step": attempt,
                "text": "Planner produced no parsable hypothesis JSON.",
                "error": True,
            }
            no_progress += 1
            _record_metrics(
                attempt_dir,
                metrics_history,
                AttemptMetrics(
                    attempt=attempt,
                    phase=Phase.PLAN.value,
                    errors_total=len(real_errors),
                    error_families=len(clusters),
                    new_errors=0,
                    resolved_errors=0,
                    files_touched=0,
                    patch_size_lines=0,
                    duration_s=time.monotonic() - attempt_start,
                    layer=layer,
                    hypothesis_id="-",
                    hypothesis_kind="parse_fail",
                    success=False,
                    fingerprint=fingerprint,
                ),
            )
            if no_progress >= no_progress_limit:
                yield {
                    "type": "done",
                    "success": False,
                    "reason": "Planner failed to produce hypotheses repeatedly.",
                    "steps": attempt,
                    "builds": build_calls,
                }
                return
            continue

        if hypothesis.kind == "stop":
            yield {
                "type": "done",
                "success": False,
                "reason": (
                    "Planner declined to propose more changes "
                    f"(layer={layer}, notes={hypothesis.notes[:200]})."
                ),
                "steps": attempt,
                "builds": build_calls,
            }
            return

        # Enforce lane gating (A before B).
        if (
            lane_result.active_lane == Lane.FIRMWARE
            and all(_lane_for_path(p) == Lane.SOFTWARE for p in hypothesis.target_files)
        ):
            yield {
                "type": "observation",
                "step": attempt,
                "text": (
                    "Rejected hypothesis: firmware lane still failing but target files are software-only."
                ),
                "error": True,
            }
            no_progress += 1
            continue

        yield {
            "type": "thought",
            "step": attempt,
            "text": (
                f"[plan/{layer}/{lane_result.active_lane.value}] {hypothesis.id} "
                f"kind={hypothesis.kind} risk={hypothesis.risk} "
                f"target_files={hypothesis.target_files}\n"
                f"rationale: {hypothesis.rationale[:600]}"
            ),
        }
        yield {
            "type": "action",
            "step": attempt,
            "tool": "patch",
            "args": {"hypothesis": hypothesis.to_json()},
        }
        try:
            (attempt_dir / "hypothesis.md").write_text(_hypothesis_md(hypothesis, clusters))
        except OSError:
            pass

        # -------------------------------------------------------- PATCH
        yield {"type": "phase", "step": attempt, "phase": Phase.PATCH.value}
        file_contents: dict[str, str] = {}
        for rel in hypothesis.target_files:
            p = sandbox_dir / rel.lstrip("./")
            if p.exists():
                try:
                    file_contents[rel] = p.read_text(errors="replace")
                except OSError:
                    file_contents[rel] = ""
            else:
                file_contents[rel] = ""

        patcher_user = _patcher_prompt(
            hypothesis=hypothesis,
            file_contents=file_contents,
            layer=layer,
            clusters=clusters,
            change_spec=change_spec,
            gitnexus_report=gitnexus_report,
        )
        patcher_raw_parts: list[str] = []

        def _on_patcher_token(t: str) -> None:
            patcher_raw_parts.append(t)

        try:
            patcher_raw = _llm_complete(
                llm_stream,
                _PATCHER_SYSTEM_PROMPT,
                patcher_user,
                max_tokens=patch_max_tokens,
                on_token=_on_patcher_token,
            )
        except Exception as e:
            yield {"type": "warning", "message": f"patcher LLM failed: {e}"}
            patcher_raw = "".join(patcher_raw_parts)

        for piece in _chunk_text(patcher_raw, 2048):
            yield {"type": "raw_token", "text": piece}

        new_files = parse_patch(patcher_raw)
        if not new_files:
            yield {
                "type": "observation",
                "step": attempt,
                "text": "Patch agent returned no parseable file blocks.",
                "error": True,
            }
            no_progress += 1
            if no_progress >= no_progress_limit:
                yield {
                    "type": "done",
                    "success": False,
                    "reason": "Patch agent failed to emit edits repeatedly.",
                    "steps": attempt,
                    "builds": build_calls,
                }
                return
            continue

        result = apply_hypothesis(sandbox_dir, hypothesis, new_files)
        touched_recent = list(result.applied)
        try:
            (attempt_dir / "patch.diff").write_text(result.diff or "(empty diff)")
        except OSError:
            pass
        yield {
            "type": "observation",
            "step": attempt,
            "text": (
                f"Applied {len(result.applied)} file(s). "
                + (f"Rejected: {result.rejected}" if result.rejected else "")
            ),
            "error": bool(result.rejected and not result.applied),
        }
        if not result.applied:
            no_progress += 1
            continue

        # Patch entropy guardrail per attempt.
        if len(result.applied) > max_touched_files_per_attempt:
            rollback(result.snapshot_before)
            no_progress += 1
            yield {
                "type": "observation",
                "step": attempt,
                "text": (
                    f"Rejected patch: touched {len(result.applied)} files "
                    f"(max per attempt is {max_touched_files_per_attempt})."
                ),
                "error": True,
            }
            continue

        for f in result.applied:
            distinct_files_touched.add(f)
        if len(distinct_files_touched) > edit_budget_files:
            yield {
                "type": "done",
                "success": False,
                "reason": (
                    f"Edit budget exhausted "
                    f"({len(distinct_files_touched)} > {edit_budget_files} files)."
                ),
                "steps": attempt,
                "builds": build_calls,
            }
            return

        # ------------------------------------------------ SUPERLOOP VALIDATION
        yield {"type": "phase", "step": attempt, "phase": Phase.SUPERLOOP.value}
        superloop_errors, superloop_warnings = run_superloop_contract_checks(
            sandbox_dir=sandbox_dir,
            patch_result=result,
        )
        last_superloop_findings = [*superloop_errors, *superloop_warnings]
        try:
            (attempt_dir / "superloop_checks.txt").write_text(
                "\n".join(last_superloop_findings) + ("\n" if last_superloop_findings else "")
            )
        except OSError:
            pass

        if superloop_warnings:
            yield {
                "type": "observation",
                "step": attempt,
                "text": "Superloop warnings: " + " | ".join(superloop_warnings[:3]),
                "error": False,
            }

        if superloop_errors:
            rollback(result.snapshot_before)
            superloop_fail_streak += 1
            no_progress += 1
            yield {
                "type": "observation",
                "step": attempt,
                "text": "Superloop contract failed: " + " | ".join(superloop_errors[:3]),
                "error": True,
            }
            _record_metrics(
                attempt_dir,
                metrics_history,
                AttemptMetrics(
                    attempt=attempt,
                    phase=Phase.SUPERLOOP.value,
                    errors_total=len(real_errors),
                    error_families=len(clusters),
                    new_errors=0,
                    resolved_errors=0,
                    files_touched=len(result.applied),
                    patch_size_lines=sum(c.count("\n") for c in new_files.values()),
                    duration_s=time.monotonic() - attempt_start,
                    layer=layer,
                    hypothesis_id=hypothesis.id,
                    hypothesis_kind=hypothesis.kind,
                    success=False,
                    fingerprint=fingerprint,
                ),
            )
            if superloop_fail_streak >= 2:
                _restore_checkpoint(best_checkpoint)
                second_choice_mode = True
                no_progress = 0
                yield {
                    "type": "observation",
                    "step": attempt,
                    "text": (
                        "Repeated superloop validation failures; restored best checkpoint "
                        "and switching to second-ranked hypothesis mode."
                    ),
                    "error": True,
                }
                ok_restore, restored_output = build_runner()
                build_calls += 1
                yield {"type": "build", "step": attempt, "success": ok_restore, "calls": build_calls}
                last_build_output = restored_output
                continue
            if no_progress >= no_progress_limit:
                yield {
                    "type": "done",
                    "success": False,
                    "reason": (
                        f"No net error reduction across {no_progress_limit} iterations "
                        "(superloop regression gate)."
                    ),
                    "steps": attempt,
                    "builds": build_calls,
                }
                return
            continue
        superloop_fail_streak = 0

        # -------------------------------------------------------- VERIFY (build)
        yield {"type": "phase", "step": attempt, "phase": Phase.VERIFY.value}
        ok_new, new_build_output = build_runner()
        build_calls += 1
        yield {"type": "build", "step": attempt, "success": ok_new, "calls": build_calls}

        new_errors = parse_build_log(new_build_output)
        for e in new_errors:
            e.error_class = classify_error(e)
        new_real_errors = [e for e in new_errors if e.severity == "error"]
        new_lane_result = evaluate_lanes(new_real_errors, index=index)
        compile_only_now = _is_compile_only_success(new_build_output)
        full_build_success = ok_new and not compile_only_now
        if compile_only_now and _linker_errors_present(new_real_errors):
            compile_only_success_streak += 1
        else:
            compile_only_success_streak = 0

        regression_findings = run_touched_module_regression_checks(
            sandbox_dir=sandbox_dir,
            touched_files=result.applied,
        )
        unresolved_delta = unresolved_icd_deltas(new_real_errors)

        prev_count = len(real_errors)
        new_count = len(new_real_errors)
        resolved = max(0, prev_count - new_count)
        new_intro = max(0, new_count - prev_count)

        metrics = AttemptMetrics(
            attempt=attempt,
            phase=Phase.VERIFY.value,
            errors_total=new_count,
            error_families=len({(e.error_class, e.symbol) for e in new_real_errors}),
            new_errors=new_intro,
            resolved_errors=resolved,
            files_touched=len(result.applied),
            patch_size_lines=sum(c.count("\n") for c in new_files.values()),
            duration_s=time.monotonic() - attempt_start,
            layer=layer,
            hypothesis_id=hypothesis.id,
            hypothesis_kind=hypothesis.kind,
            success=full_build_success,
            fingerprint=errors_fingerprint(new_real_errors),
        )
        _record_metrics(attempt_dir, metrics_history, metrics)

        # Acceptance criteria gate: full build + both lanes + superloop + ICD deltas + regression checks.
        if (
            full_build_success
            and not new_lane_result.firmware.errors
            and not new_lane_result.software.errors
            and not unresolved_delta
            and not regression_findings
        ):
            playbook.record({
                "error_class": (clusters[0].error_class.value if clusters else "other"),
                "symbol": (clusters[0].representative.symbol if clusters else ""),
                "patch_kind": hypothesis.kind,
                "summary": hypothesis.title,
                "success": True,
            })
            yield {
                "type": "done",
                "success": True,
                "reason": (
                    "Xilinx profile acceptance criteria met: full build green, "
                    "firmware lane pass, software lane pass, superloop contracts pass, "
                    "no unresolved ICD deltas, and touched-module regressions clear."
                ),
                "steps": attempt,
                "builds": build_calls,
            }
            return

        if ok_new and compile_only_now:
            yield {
                "type": "observation",
                "step": attempt,
                "text": (
                    "Compile-only success is not sufficient under this profile; "
                    "continuing until full link succeeds."
                ),
                "error": True,
            }
        if regression_findings:
            yield {
                "type": "observation",
                "step": attempt,
                "text": "Regression checks failed: " + " | ".join(regression_findings[:3]),
                "error": True,
            }

        # Update best checkpoint if objective improved.
        score = (
            len(new_lane_result.firmware.errors),
            len(new_lane_result.software.errors),
            len(new_real_errors),
        )
        if score < best_score:
            best_score = score
            best_checkpoint = _capture_checkpoint(build_root)

        # -------------------------------------------------------- DECIDE
        yield {"type": "phase", "step": attempt, "phase": Phase.DECIDE.value}
        stall_reasons: list[str] = []

        if metrics.fingerprint == fingerprint:
            no_progress += 1
            rollback(result.snapshot_before)
            yield {
                "type": "observation",
                "step": attempt,
                "text": "Patch did not change active error family — rolling back.",
                "error": True,
            }
        elif new_intro > 0 and resolved == 0:
            rollback(result.snapshot_before)
            no_progress += 1
            playbook.record({
                "error_class": (clusters[0].error_class.value if clusters else "other"),
                "symbol": (clusters[0].representative.symbol if clusters else ""),
                "patch_kind": hypothesis.kind,
                "summary": "regression: " + hypothesis.title,
                "success": False,
            })
            yield {
                "type": "observation",
                "step": attempt,
                "text": (
                    f"Patch introduced {new_intro} new errors with no resolved families — rolling back."
                ),
                "error": True,
            }
        else:
            no_progress = 0
            second_choice_mode = False

        if _is_lane_oscillation(lane_history):
            stall_reasons.append("A->B->A lane oscillation detected")
        if compile_only_success_streak >= 2 and _linker_errors_present(new_real_errors):
            stall_reasons.append("linker failures persist after two compile-only iterations")
        if superloop_fail_streak >= 2:
            stall_reasons.append("superloop validation repeatedly failing")
        if seen_fingerprints.get(metrics.fingerprint, 0) >= oscillation_limit:
            stall_reasons.append("error fingerprint oscillation")

        if stall_reasons:
            _restore_checkpoint(best_checkpoint)
            second_choice_mode = True
            no_progress = 0
            yield {
                "type": "observation",
                "step": attempt,
                "text": (
                    "Stall indicators triggered: " + " | ".join(stall_reasons[:3])
                    + ". Restored best checkpoint and switching to second-ranked hypothesis strategy."
                ),
                "error": True,
            }
            if layer_idx < len(layer_order) - 1 and "error fingerprint oscillation" in stall_reasons:
                layer_idx += 1
                yield {
                    "type": "observation",
                    "step": attempt,
                    "text": f"Escalated correction layer to '{layer_order[layer_idx]}' after oscillation.",
                    "error": False,
                }
            ok_restore, restored_output = build_runner()
            build_calls += 1
            yield {"type": "build", "step": attempt, "success": ok_restore, "calls": build_calls}
            last_build_output = restored_output
            continue

        if no_progress >= no_progress_limit:
            yield {
                "type": "done",
                "success": False,
                "reason": (
                    f"No net error reduction across {no_progress_limit} iterations."
                ),
                "steps": attempt,
                "builds": build_calls,
            }
            return

        last_build_output = new_build_output
        seen_fingerprints[metrics.fingerprint] = (
            seen_fingerprints.get(metrics.fingerprint, 0) + 1
        )

    yield {
        "type": "done",
        "success": False,
        "reason": (
            f"Attempt budget exhausted ({max_attempts}) before satisfying "
            "Xilinx acceptance criteria."
        ),
        "steps": max_attempts if max_attempts is not None else attempt,
        "builds": build_calls,
    }


# ---------------------------------------------------------------------------
# Helpers (artifacts + small utilities)
# ---------------------------------------------------------------------------


def _chunk_text(text: str, n: int) -> Iterator[str]:
    if not text:
        return
    for i in range(0, len(text), n):
        yield text[i:i + n]


def _hypothesis_md(h: Hypothesis, clusters: list[ErrorCluster]) -> str:
    expected_lines = (
        [f"- {x}" for x in h.expected_resolved]
        if h.expected_resolved else ["(unspecified)"]
    )
    lines = [
        f"# Hypothesis {h.id}",
        f"- **layer**: {h.layer}",
        f"- **kind**: `{h.kind}`",
        f"- **risk**: {h.risk}",
        f"- **target_files**:",
        *[f"  - `{p}`" for p in h.target_files],
        "",
        "## Rationale",
        h.rationale or "(none)",
        "",
        "## Expected to resolve",
        *expected_lines,
        "",
        "## Top error clusters at proposal time",
    ]
    for c in clusters[:5]:
        lines.append(
            f"- **{c.root_cause}** "
            f"({c.error_class.value}, members={len(c.members)}, "
            f"centrality={c.centrality})"
        )
    if h.notes:
        lines += ["", "## Notes", h.notes]
    return "\n".join(lines) + "\n"


def _record_metrics(
    attempt_dir: Path,
    history: list[dict],
    metrics: AttemptMetrics,
) -> None:
    blob = asdict(metrics)
    history.append(blob)
    try:
        (attempt_dir / "metrics.json").write_text(
            json.dumps(blob, indent=2, ensure_ascii=False)
        )
    except OSError:
        pass


def _write_attempt_artifacts(
    attempt_dir: Path,
    *,
    raw_log: str,
    errors: list[BuildError],
    clusters: list[ErrorCluster],
    hypothesis: Hypothesis | None,
    metrics: AttemptMetrics,
    patch_diff: str,
) -> None:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    try:
        (attempt_dir / "build.log.raw").write_text(raw_log or "")
        (attempt_dir / "build.log.structured.json").write_text(
            json.dumps([e.to_json() for e in errors], indent=2)
        )
        (attempt_dir / "error_clusters.json").write_text(
            json.dumps([c.to_json() for c in clusters], indent=2)
        )
        if hypothesis is not None:
            (attempt_dir / "hypothesis.md").write_text(
                _hypothesis_md(hypothesis, clusters)
            )
        (attempt_dir / "patch.diff").write_text(patch_diff or "")
        (attempt_dir / "metrics.json").write_text(
            json.dumps(asdict(metrics), indent=2)
        )
    except OSError:
        log.exception("Failed to write attempt artifacts to %s", attempt_dir)


__all__ = [
    "ErrorClass", "BuildError", "ErrorCluster", "CodebaseIndex",
    "Hypothesis", "Phase", "AttemptMetrics", "Playbook",
    "parse_build_log", "classify_error", "cluster_errors",
    "errors_fingerprint", "run_pre_build_checks",
    "parse_hypothesis", "parse_patch", "apply_hypothesis", "rollback",
    "run_agentic_debug",
    "DEFAULT_MAX_ATTEMPTS", "DEFAULT_NO_PROGRESS_LIMIT",
    "DEFAULT_OSCILLATION_LIMIT", "DEFAULT_EDIT_BUDGET_FILES",
]
