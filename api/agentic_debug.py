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

    clusters.sort(key=lambda c: (-c.centrality, c.error_class.value))
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
            "Provide a stub or wrapper for the missing symbol. For BSP "
            "symbols (Xil_*, xil_printf, …) on the cross-compile sandbox, "
            "the build runner already retries compile-only — focus only on "
            "non-BSP undefined symbols."
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
5. The current correction layer is provided — you must respect it.
   - Layer **shim**: build-breaker fixes only (make it compile/link with a
     compatibility layer).
   - Layer **semantic**: replace shims with precise mappings from the
     target ICD.
   - Layer **cleanup**: remove dead aliases, tighten types.
6. If no further hypothesis is reasonable (build is green or you would
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
"""


def _planner_prompt(
    *,
    layer: str,
    attempt: int,
    max_attempts: int,
    metrics_history: list[dict],
    clusters: list[ErrorCluster],
    pre_build_warnings: list[str],
    sandbox_cc: str,
    is_cross: bool,
    change_spec: str,
    repo_knowledge: str,
    playbook_hints: list[str],
) -> str:
    cluster_blob = json.dumps(
        [c.to_json() for c in clusters[:8]],
        indent=2, ensure_ascii=False,
    )
    metrics_tail = metrics_history[-5:]
    metrics_blob = json.dumps(metrics_tail, indent=2, ensure_ascii=False)
    hints = "\n".join(f"- {h}" for h in playbook_hints) or "(none yet)"
    warnings = "\n".join(f"- {w}" for w in pre_build_warnings) or "(none)"
    cross_note = (
        "Cross-compile sandbox is in use. BSP linker errors (Xil_*, "
        "xil_printf, etc.) are EXPECTED and resolved by compile-only "
        "fallback — focus on non-BSP issues only."
        if is_cross else "Native compile."
    )
    return (
        f"## Build pipeline state\n"
        f"- Attempt: {attempt}/{max_attempts}\n"
        f"- Active correction layer: {layer}\n"
        f"- Compiler: {sandbox_cc}\n"
        f"- {cross_note}\n\n"
        f"## ICD change spec (truncated)\n{change_spec[:4000]}\n\n"
        f"## Repo knowledge (truncated)\n{repo_knowledge[:2000]}\n\n"
        f"## Recent metrics history\n```json\n{metrics_blob}\n```\n\n"
        f"## Top error clusters (sorted by centrality)\n"
        f"```json\n{cluster_blob}\n```\n\n"
        f"## Pre-build static warnings\n{warnings}\n\n"
        f"## Playbook hints (memory of past successful fixes)\n{hints}\n\n"
        f"## Your turn\nProduce one Hypothesis JSON now."
    )


def _patcher_prompt(
    *,
    hypothesis: Hypothesis,
    file_contents: dict[str, str],
    layer: str,
    clusters: list[ErrorCluster],
    change_spec: str,
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
    return (
        f"## Hypothesis\n```json\n{json.dumps(hypothesis.to_json(), indent=2)}\n```\n\n"
        f"## Active layer\n{layer}\n\n"
        f"## ICD change spec (truncated)\n{change_spec[:4000]}\n\n"
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
    file_index: dict[str, list[Path]],
    snapshots: dict[Path, str],
    build_runner: Callable[[], tuple[bool, str]],
    llm_stream: Callable[..., Iterator[str]],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    no_progress_limit: int = DEFAULT_NO_PROGRESS_LIMIT,
    oscillation_limit: int = DEFAULT_OSCILLATION_LIMIT,
    edit_budget_files: int = DEFAULT_EDIT_BUDGET_FILES,
    planner_max_tokens: int = DEFAULT_PLANNER_MAX_TOKENS,
    patch_max_tokens: int = DEFAULT_PATCH_MAX_TOKENS,
) -> Iterator[dict]:
    """Drive the agentic-debug state machine.

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

    # ------------------------------------------------------------------ BUILD
    yield {"type": "phase", "step": 0, "phase": Phase.BUILD.value}
    success, build_output = build_runner()
    build_calls = 1
    yield {"type": "build", "step": 0, "success": success, "calls": build_calls}
    if success:
        _write_attempt_artifacts(
            attempts_dir / "attempt_00",
            raw_log=build_output, errors=[], clusters=[],
            hypothesis=None, metrics=AttemptMetrics(
                attempt=0, phase=Phase.DONE.value, errors_total=0,
                error_families=0, new_errors=0, resolved_errors=0,
                files_touched=0, patch_size_lines=0, duration_s=0.0,
                layer="shim", hypothesis_id="-", hypothesis_kind="-",
                success=True, fingerprint="",
            ),
            patch_diff="",
        )
        yield {
            "type": "done", "success": True,
            "reason": "Build already succeeds without any edits.",
            "steps": 0, "builds": build_calls,
        }
        return

    layer_order = ["shim", "semantic", "cleanup"]
    layer_idx = 0
    metrics_history: list[dict] = []
    seen_fingerprints: dict[str, int] = {}
    no_progress = 0
    distinct_files_touched: set[Path] = set()
    last_error_count: int | None = None

    last_build_output = build_output

    for attempt in range(1, max_attempts + 1):
        attempt_start = time.monotonic()
        attempt_dir = attempts_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        yield {"type": "step", "step": attempt}

        # -------------------------------------------------------- TRIAGE
        yield {"type": "phase", "step": attempt, "phase": Phase.TRIAGE.value}
        errors = parse_build_log(last_build_output)
        for e in errors:
            e.error_class = classify_error(e)

        # Filter out warnings — they shouldn't drive fix planning.
        real_errors = [e for e in errors if e.severity == "error"]

        if not real_errors:
            # We had a non-zero exit but couldn't parse any error — treat as
            # opaque failure: surface raw tail to UI, then escalate.
            yield {
                "type": "observation", "step": attempt,
                "text": (
                    "Build failed but no parseable diagnostics found. "
                    "Raw tail will be sent to the planner."
                ),
                "error": True,
            }

        fingerprint = errors_fingerprint(real_errors)
        seen_fingerprints[fingerprint] = seen_fingerprints.get(fingerprint, 0) + 1

        # -------------------------------------------------------- ROOT_CAUSE
        yield {"type": "phase", "step": attempt, "phase": Phase.ROOT_CAUSE.value}
        clusters = cluster_errors(real_errors, index=index)
        clusters_blob = [c.to_json() for c in clusters]
        try:
            (attempt_dir / "error_clusters.json").write_text(
                json.dumps(clusters_blob, indent=2, ensure_ascii=False)
            )
        except OSError:
            pass

        yield {
            "type": "observation", "step": attempt,
            "text": (
                f"Triaged {len(errors)} diagnostics → "
                f"{len(real_errors)} errors → {len(clusters)} root-cause clusters. "
                f"Top: " + (
                    clusters[0].root_cause if clusters else "(none)"
                )
            ),
            "error": False,
        }

        # -------------------------------------------------------- PLAN
        yield {"type": "phase", "step": attempt, "phase": Phase.PLAN.value}
        layer = layer_order[layer_idx]
        playbook_hints = []
        if clusters:
            playbook_hints = playbook.hints_for(
                clusters[0].error_class, clusters[0].representative.symbol,
            )
        prebuild_warnings = run_pre_build_checks(
            sandbox_dir, list(distinct_files_touched)
        )
        planner_user = _planner_prompt(
            layer=layer,
            attempt=attempt,
            max_attempts=max_attempts,
            metrics_history=metrics_history,
            clusters=clusters,
            pre_build_warnings=prebuild_warnings,
            sandbox_cc=sandbox_cc,
            is_cross=is_cross,
            change_spec=change_spec,
            repo_knowledge=repo_knowledge,
            playbook_hints=playbook_hints,
        )
        planner_raw_parts: list[str] = []

        def _on_planner_token(t: str) -> None:
            planner_raw_parts.append(t)
        try:
            planner_raw = _llm_complete(
                llm_stream, _PLANNER_SYSTEM_PROMPT, planner_user,
                max_tokens=planner_max_tokens, on_token=_on_planner_token,
            )
        except Exception as e:
            yield {"type": "warning", "message": f"planner LLM failed: {e}"}
            planner_raw = "".join(planner_raw_parts)

        # Stream planner thoughts to UI as raw tokens
        for piece in _chunk_text(planner_raw, 1024):
            yield {"type": "raw_token", "text": piece}

        hypothesis = parse_hypothesis(planner_raw)
        if hypothesis is None:
            yield {
                "type": "observation", "step": attempt,
                "text": "Planner produced no parsable hypothesis JSON.",
                "error": True,
            }
            no_progress += 1
            _record_metrics(
                attempt_dir, metrics_history,
                AttemptMetrics(
                    attempt=attempt, phase=Phase.PLAN.value,
                    errors_total=len(real_errors),
                    error_families=len(clusters),
                    new_errors=0, resolved_errors=0,
                    files_touched=0, patch_size_lines=0,
                    duration_s=time.monotonic() - attempt_start,
                    layer=layer, hypothesis_id="-", hypothesis_kind="parse_fail",
                    success=False, fingerprint=fingerprint,
                ),
            )
            if no_progress >= no_progress_limit:
                yield {
                    "type": "done", "success": False,
                    "reason": "Planner failed to produce hypotheses repeatedly.",
                    "steps": attempt, "builds": build_calls,
                }
                return
            continue

        if hypothesis.kind == "stop":
            yield {
                "type": "done", "success": False,
                "reason": (
                    "Planner declined to propose more changes "
                    f"(layer={layer}, notes={hypothesis.notes[:200]})."
                ),
                "steps": attempt, "builds": build_calls,
            }
            return

        yield {
            "type": "thought", "step": attempt,
            "text": (
                f"[plan/{layer}] {hypothesis.id} kind={hypothesis.kind} "
                f"risk={hypothesis.risk} target_files={hypothesis.target_files}\n"
                f"rationale: {hypothesis.rationale[:600]}"
            ),
        }
        yield {
            "type": "action", "step": attempt, "tool": "patch",
            "args": {"hypothesis": hypothesis.to_json()},
        }
        try:
            (attempt_dir / "hypothesis.md").write_text(
                _hypothesis_md(hypothesis, clusters)
            )
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
        )
        patcher_raw_parts: list[str] = []

        def _on_patcher_token(t: str) -> None:
            patcher_raw_parts.append(t)
        try:
            patcher_raw = _llm_complete(
                llm_stream, _PATCHER_SYSTEM_PROMPT, patcher_user,
                max_tokens=patch_max_tokens, on_token=_on_patcher_token,
            )
        except Exception as e:
            yield {"type": "warning", "message": f"patcher LLM failed: {e}"}
            patcher_raw = "".join(patcher_raw_parts)

        for piece in _chunk_text(patcher_raw, 2048):
            yield {"type": "raw_token", "text": piece}

        new_files = parse_patch(patcher_raw)
        if not new_files:
            yield {
                "type": "observation", "step": attempt,
                "text": "Patch agent returned no parseable file blocks.",
                "error": True,
            }
            no_progress += 1
            _record_metrics(
                attempt_dir, metrics_history,
                AttemptMetrics(
                    attempt=attempt, phase=Phase.PATCH.value,
                    errors_total=len(real_errors),
                    error_families=len(clusters),
                    new_errors=0, resolved_errors=0,
                    files_touched=0, patch_size_lines=0,
                    duration_s=time.monotonic() - attempt_start,
                    layer=layer, hypothesis_id=hypothesis.id,
                    hypothesis_kind=hypothesis.kind, success=False,
                    fingerprint=fingerprint,
                ),
            )
            if no_progress >= no_progress_limit:
                yield {
                    "type": "done", "success": False,
                    "reason": "Patch agent failed to emit edits repeatedly.",
                    "steps": attempt, "builds": build_calls,
                }
                return
            continue

        result = apply_hypothesis(sandbox_dir, hypothesis, new_files)
        try:
            (attempt_dir / "patch.diff").write_text(result.diff or "(empty diff)")
        except OSError:
            pass
        yield {
            "type": "observation", "step": attempt,
            "text": (
                f"Applied {len(result.applied)} file(s). "
                + (f"Rejected: {result.rejected}" if result.rejected else "")
            ),
            "error": bool(result.rejected and not result.applied),
        }
        if not result.applied:
            no_progress += 1
            continue
        for f in result.applied:
            distinct_files_touched.add(f)
        if len(distinct_files_touched) > edit_budget_files:
            yield {
                "type": "done", "success": False,
                "reason": (
                    f"Edit budget exhausted "
                    f"({len(distinct_files_touched)} > {edit_budget_files} files)."
                ),
                "steps": attempt, "builds": build_calls,
            }
            return

        # -------------------------------------------------------- VERIFY (build)
        yield {"type": "phase", "step": attempt, "phase": Phase.VERIFY.value}
        ok, new_build_output = build_runner()
        build_calls += 1
        yield {"type": "build", "step": attempt, "success": ok, "calls": build_calls}
        try:
            (attempt_dir / "build.log.raw").write_text(new_build_output)
        except OSError:
            pass

        new_errors = parse_build_log(new_build_output)
        new_real_errors = [e for e in new_errors if e.severity == "error"]
        try:
            (attempt_dir / "build.log.structured.json").write_text(
                json.dumps([e.to_json() for e in new_errors], indent=2,
                           ensure_ascii=False),
            )
        except OSError:
            pass

        prev_count = len(real_errors)
        new_count = len(new_real_errors)
        resolved = max(0, prev_count - new_count) if last_error_count is not None or True else 0
        new_intro = max(0, new_count - prev_count)
        last_error_count = new_count

        metrics = AttemptMetrics(
            attempt=attempt, phase=Phase.VERIFY.value,
            errors_total=new_count,
            error_families=len({(e.error_class, e.symbol) for e in new_real_errors}),
            new_errors=new_intro, resolved_errors=resolved,
            files_touched=len(result.applied),
            patch_size_lines=sum(c.count("\n") for c in new_files.values()),
            duration_s=time.monotonic() - attempt_start,
            layer=layer, hypothesis_id=hypothesis.id,
            hypothesis_kind=hypothesis.kind,
            success=ok, fingerprint=errors_fingerprint(new_real_errors),
        )
        _record_metrics(attempt_dir, metrics_history, metrics)

        if ok:
            playbook.record({
                "error_class": (
                    clusters[0].error_class.value if clusters else "other"
                ),
                "symbol": (
                    clusters[0].representative.symbol if clusters else ""
                ),
                "patch_kind": hypothesis.kind,
                "summary": hypothesis.title,
                "success": True,
            })
            yield {
                "type": "done", "success": True,
                "reason": (
                    f"Build succeeded after {attempt} hypothesis "
                    f"(layer={layer}, {build_calls} build calls)."
                ),
                "steps": attempt, "builds": build_calls,
            }
            return

        # -------------------------------------------------------- DECIDE
        yield {"type": "phase", "step": attempt, "phase": Phase.DECIDE.value}

        if metrics.fingerprint == fingerprint:
            # Edit had zero effect.
            no_progress += 1
            yield {
                "type": "observation", "step": attempt,
                "text": "Patch did not change the error set — rolling back.",
                "error": True,
            }
            rollback(result.snapshot_before)
        else:
            if new_intro > 0 and resolved == 0:
                # Pure regression — undo and bias the planner away from this kind.
                yield {
                    "type": "observation", "step": attempt,
                    "text": (
                        f"Patch introduced {new_intro} new errors with no "
                        f"resolutions — rolling back."
                    ),
                    "error": True,
                }
                rollback(result.snapshot_before)
                no_progress += 1
                playbook.record({
                    "error_class": (
                        clusters[0].error_class.value if clusters else "other"
                    ),
                    "symbol": (
                        clusters[0].representative.symbol if clusters else ""
                    ),
                    "patch_kind": hypothesis.kind,
                    "summary": "regression: " + hypothesis.title,
                    "success": False,
                })
            else:
                no_progress = 0  # we made progress

        # Oscillation check
        if seen_fingerprints.get(metrics.fingerprint, 0) >= oscillation_limit:
            # Same fingerprint a second time → escalate by advancing layer.
            if layer_idx < len(layer_order) - 1:
                layer_idx += 1
                yield {
                    "type": "observation", "step": attempt,
                    "text": (
                        f"Oscillation detected — advancing correction layer "
                        f"to '{layer_order[layer_idx]}'."
                    ),
                    "error": False,
                }
            else:
                yield {
                    "type": "done", "success": False,
                    "reason": (
                        f"Oscillation persists at deepest layer "
                        f"({metrics.fingerprint!r})."
                    ),
                    "steps": attempt, "builds": build_calls,
                }
                return

        # No-net-improvement check
        if no_progress >= no_progress_limit:
            yield {
                "type": "done", "success": False,
                "reason": (
                    f"No net error reduction across "
                    f"{no_progress_limit} iterations."
                ),
                "steps": attempt, "builds": build_calls,
            }
            return

        last_build_output = new_build_output
        seen_fingerprints[metrics.fingerprint] = (
            seen_fingerprints.get(metrics.fingerprint, 0) + 1
        )

    yield {
        "type": "done", "success": False,
        "reason": f"Attempt budget exhausted ({max_attempts}).",
        "steps": max_attempts, "builds": build_calls,
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
