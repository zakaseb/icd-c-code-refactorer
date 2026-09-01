"""
CodeGraph — deterministic symbol / cross-reference index for the transform stage.

Why this exists
---------------
The ICD transform stage rewrites ONE uploaded ``.c`` / ``.h`` file at a time
against the ICD change specification.  Until now it saw the rest of the
codebase only through two context builders in ``app.py``:

  * ``_build_repo_knowledge``    — a repo-wide inventory hard-capped at *the
    first* 30 structs / 40 signatures / 40 macros **in filesystem order**.
    Nothing in it is selected for relevance to the file being transformed.
  * ``_build_file_repo_context`` — the **full text** of every resolved
    ``#include``, transitively, spliced into a 15 000-character budget.  A
    single ``xparameters.h`` exhausts the budget before any project header is
    reached.

Both walk only *outgoing* edges: what the file consumes.  Neither can answer
the question that actually governs a safe refactor —

    "if I rename this struct field, what else in the codebase breaks?"

That is an *incoming*-edge question, so the model was never given the
information it needed to keep a change local.  It would rename ``msg_id`` to
``message_id`` for tidiness and the damage would only surface much later, at
the per-file compile or sandbox build, as a pile of errors the agentic debug
loop had to reverse-engineer.

What this module provides
-------------------------
A purely deterministic index — regexes and a single scan pass, no LLM, no
network, milliseconds on a real repo, fully reproducible:

  * every symbol definition (struct, union, enum, typedef, function, macro,
    global, and individual struct fields / enumerators) with the exact source
    slice that declares it;
  * the reference graph: for each symbol, which files use it and how often;
  * the include graph in **both** directions (``includes`` / ``included_by``).

From those three the module renders the two prompt sections the transform
stage actually needs: a *public surface* table telling the model which of the
symbols it owns are load-bearing elsewhere, and *declaration slices* — a
typedef body is ~8 lines, not the 2 000-line header it lives in.

Design notes
------------
* Reference counting runs ONE identifier scan per file and intersects against
  the known-symbol set.  It deliberately does not run one regex per symbol
  over every file (the approach in ``agentic_debug.CodebaseIndex`` —
  O(symbols x files), which does not scale to a Xilinx-SDK tree).
* Definitions are extracted from raw text so comments survive into the
  declaration slices; references are counted on comment-stripped text so a
  doc banner mentioning ``msg_id`` does not inflate its blast radius.
* The regex library mirrors ``gitnexus.py``, which extracts the same shapes
  for its report but is not wired into the pipeline.
* Heuristics are conservative: an over-count makes the model *more* cautious
  about renaming, which is the safe direction to fail in.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

log = logging.getLogger(__name__)

HEADER_EXTS = {".h", ".hpp", ".hh", ".hxx"}
SOURCE_EXTS = {".c", ".cpp", ".cc", ".cxx"}
CODE_EXTS = HEADER_EXTS | SOURCE_EXTS

# Ceilings that keep a pathological repository from blowing up memory or the
# wall-clock budget of the transform stage.
DEFAULT_MAX_FILES = 4_000
MAX_REF_SAMPLES = 12          # sample call sites retained per symbol
MAX_DECL_CHARS = 2_500        # a single declaration slice never exceeds this

# Identifiers that regexes below can match structurally but which are never
# user symbols. Keeping this tight matters: a false symbol pollutes the
# public-surface table and wastes prompt budget.
_C_KEYWORDS = {
    "if", "else", "while", "for", "switch", "case", "default", "do", "return",
    "break", "continue", "goto", "sizeof", "typedef", "struct", "union",
    "enum", "static", "extern", "const", "volatile", "register", "inline",
    "unsigned", "signed", "void", "char", "short", "int", "long", "float",
    "double", "_Bool", "restrict", "auto", "defined", "asm", "__asm__",
    "__attribute__", "__inline__", "__volatile__",
}

# Very common library types/macros: they are legitimate symbols but listing
# them as "dependencies to show the model" is pure noise, since the model
# already knows <stdint.h>.
_UBIQUITOUS = {
    "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t",
    "int32_t", "int64_t", "size_t", "ssize_t", "bool", "true", "false",
    "NULL", "EXIT_SUCCESS", "EXIT_FAILURE",
}

_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
# Struct-field usage: `hdr.msg_id`, `p->msg_id`, and designated initialisers
# (`.msg_id = 4`). Matching fields as bare identifiers instead would make
# every `len` / `data` / `status` in the tree look load-bearing.
_MEMBER_ACCESS_RE = re.compile(r"(?:\.|->)\s*([A-Za-z_]\w*)")

# --- definition patterns ---------------------------------------------------

_TYPEDEF_RECORD_RE = re.compile(
    r"\btypedef\s+(struct|union|enum)\s+(\w+)?\s*\{(.*?)\}\s*([A-Za-z_]\w*)\s*;",
    re.DOTALL,
)
_RECORD_RE = re.compile(
    r"\b(struct|union|enum)\s+([A-Za-z_]\w*)\s*\{(.*?)\}\s*;",
    re.DOTALL,
)
_TYPEDEF_SIMPLE_RE = re.compile(
    r"\btypedef\s+((?:[A-Za-z_]\w*|\*|\s)+?)\s+\*?([A-Za-z_]\w*)\s*;",
)
_DEFINE_RE = re.compile(
    r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)(\([^)]*\))?([^\n\\]*(?:\\\n[^\n\\]*)*)",
    re.MULTILINE,
)
_DECL_PREFIX = (
    r"(?:(?:static|extern|inline|const|volatile|unsigned|signed|"
    r"struct|enum|union|__inline__)\s+)*"
)
_FUNC_DEF_RE = re.compile(
    r"^[ \t]*(" + _DECL_PREFIX + r"[A-Za-z_]\w*(?:\s*\*)*)\s+\*?\s*"
    r"([A-Za-z_]\w*)\s*\(([^;{)]*)\)\s*\{",
    re.MULTILINE,
)
_FUNC_PROTO_RE = re.compile(
    r"^[ \t]*(" + _DECL_PREFIX + r"[A-Za-z_]\w*(?:\s*\*)*)\s+\*?\s*"
    r"([A-Za-z_]\w*)\s*\(([^;{)]*)\)\s*;",
    re.MULTILINE,
)
_GLOBAL_RE = re.compile(
    r"^[ \t]*((?:(?:static|extern|volatile|const)\s+)+[A-Za-z_]\w*(?:\s*\*)*)"
    r"\s+\*?([A-Za-z_]\w*)(\s*\[[^\]]*\])?\s*(?:=[^;]*)?;",
    re.MULTILINE,
)
_INC_QUOTE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
# `#ifndef FOO_H` / `#define FOO_H` — an include guard, not a real macro.
# Listing guards in the impact table is pure noise.
_INCLUDE_GUARD_RE = re.compile(
    r"#\s*ifndef\s+([A-Za-z_]\w*)\s*\n\s*#\s*define\s+\1\b"
)


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

@dataclass
class Symbol:
    """One definition site."""
    name: str
    kind: str                 # struct|union|enum|typedef|func|macro|global|field|enumerator
    file: Path
    line: int
    decl: str                 # exact source slice that declares it
    parent: str = ""          # owning record, for kind in {field, enumerator}

    @property
    def is_member(self) -> bool:
        return self.kind in ("field", "enumerator")


@dataclass
class Ref:
    """One sampled reference site."""
    file: Path
    line: int
    text: str


@dataclass
class Impact:
    """Blast radius of one symbol, split by whether the referencing file is
    itself part of this transform batch."""
    name: str
    kind: str
    parent: str = ""          # owning record, for fields / enumerators
    external_sites: int = 0   # references from files NOT being transformed
    external_files: int = 0
    sibling_sites: int = 0    # references from other files in the batch
    sibling_files: int = 0
    samples: list[Ref] = field(default_factory=list)

    @property
    def frozen(self) -> bool:
        """True when renaming this symbol breaks code outside the batch."""
        return self.external_sites > 0


@dataclass
class CodeGraph:
    roots: list[Path] = field(default_factory=list)
    symbols: dict[str, list[Symbol]] = field(default_factory=dict)
    # name -> {file: exact occurrence count} (definition sites excluded)
    ref_counts: dict[str, dict[Path, int]] = field(default_factory=dict)
    ref_samples: dict[str, list[Ref]] = field(default_factory=dict)
    includes: dict[Path, set[Path]] = field(default_factory=dict)
    included_by: dict[Path, set[Path]] = field(default_factory=dict)
    files: list[Path] = field(default_factory=list)

    # -- construction -------------------------------------------------------

    @classmethod
    def build(
        cls,
        *roots: Path,
        read_text: Callable[[Path], str] | None = None,
        max_files: int = DEFAULT_MAX_FILES,
    ) -> "CodeGraph":
        """Index every ``.c`` / ``.h`` file under *roots*.

        *read_text* lets the caller inject its own encoding-tolerant reader
        (``app._read_text_safe``) so there is exactly one decoding policy at
        runtime. The default never raises but is deliberately dumb.
        """
        reader = read_text or _default_read
        graph = cls(roots=[r for r in roots if r and r.exists()])

        raw: dict[Path, str] = {}
        for root in graph.roots:
            for p in sorted(root.rglob("*")):
                if not p.is_file() or p.suffix.lower() not in CODE_EXTS:
                    continue
                if len(raw) >= max_files:
                    log.warning(
                        "CodeGraph: capped at %d files; remainder ignored",
                        max_files,
                    )
                    break
                try:
                    raw[p] = reader(p)
                except Exception as exc:            # noqa: BLE001
                    log.debug("CodeGraph: cannot read %s: %s", p, exc)

        graph.files = sorted(raw)
        for path, text in raw.items():
            for sym in extract_symbols(text, path):
                graph.symbols.setdefault(sym.name, []).append(sym)

        graph._scan_references(raw)
        graph._build_include_graph(raw)
        log.info(
            "CodeGraph: %d files, %d distinct symbols, %d referenced symbols",
            len(graph.files), len(graph.symbols), len(graph.ref_counts),
        )
        return graph

    def _scan_references(self, raw: dict[Path, str]) -> None:
        """Two identifier passes per file, intersected with the symbol set.

        Struct fields are matched only through member access (``.field`` /
        ``->field`` / designated initialiser). Counting them as bare
        identifiers would be catastrophic for the common ones — every ``len``,
        ``data`` or ``status`` in the tree would land on some struct's field
        and the impact table would say everything is load-bearing.

        Both passes run on comment-stripped text so a documentation banner
        naming a field does not inflate its blast radius.
        """
        if not self.symbols:
            return
        # Names reachable only as record members, vs everything else.
        member_only = {
            name
            for name, syms in self.symbols.items()
            if all(s.kind == "field" for s in syms)
        }
        plain = set(self.symbols) - member_only

        # Precomputed once: scanning self.symbols per file would be
        # O(symbols x files), the very cost this scan exists to avoid.
        defined_by_file: dict[Path, set[str]] = defaultdict(set)
        for name, syms in self.symbols.items():
            for sym in syms:
                defined_by_file[sym.file].add(name)

        for path, text in raw.items():
            stripped = _strip_comments(text)
            defined_here = defined_by_file.get(path, set())
            line_starts = _line_start_offsets(stripped)
            per_file: dict[str, int] = defaultdict(int)

            def record(name: str, offset: int) -> None:
                per_file[name] += 1
                samples = self.ref_samples.setdefault(name, [])
                if len(samples) < MAX_REF_SAMPLES:
                    line_no = _line_of(line_starts, offset)
                    samples.append(
                        Ref(
                            file=path,
                            line=line_no,
                            text=_line_at(stripped, line_starts, line_no),
                        )
                    )

            for m in _IDENT_RE.finditer(stripped):
                name = m.group(0)
                if name in plain and name not in defined_here:
                    record(name, m.start())
            for m in _MEMBER_ACCESS_RE.finditer(stripped):
                name = m.group(1)
                if name in member_only and name not in defined_here:
                    record(name, m.start(1))

            for name, count in per_file.items():
                self.ref_counts.setdefault(name, {})[path] = count

    def _build_include_graph(self, raw: dict[Path, str]) -> None:
        by_name: dict[str, list[Path]] = defaultdict(list)
        for p in raw:
            by_name[p.name].append(p)
        for path, text in raw.items():
            outgoing: set[Path] = set()
            for inc in _INC_QUOTE_RE.findall(text):
                base = Path(inc).name
                for cand in by_name.get(base, []):
                    outgoing.add(cand)
                    self.included_by.setdefault(cand, set()).add(path)
            self.includes[path] = outgoing

    # -- queries ------------------------------------------------------------

    def defined_in(self, file: Path) -> list[Symbol]:
        """Every symbol whose definition site is *file*."""
        return [
            s
            for name in sorted(self.symbols)
            for s in self.symbols[name]
            if s.file == file
        ]

    def impact_of(self, name: str, batch: set[Path]) -> Impact:
        """Blast radius of *name*, with *batch* = files being transformed."""
        kinds = {s.kind for s in self.symbols.get(name, [])}
        imp = Impact(name=name, kind=sorted(kinds)[0] if kinds else "unknown")
        for path, count in self.ref_counts.get(name, {}).items():
            if path in batch:
                imp.sibling_sites += count
                imp.sibling_files += 1
            else:
                imp.external_sites += count
                imp.external_files += 1
        imp.samples = [
            r for r in self.ref_samples.get(name, []) if r.file not in batch
        ]
        return imp

    def public_surface(self, file: Path, batch: set[Path]) -> list[Impact]:
        """Symbols defined in *file* that anything else depends on.

        Sorted most load-bearing first so a byte budget truncates the tail,
        which is what the caller can most afford to lose.
        """
        out: list[Impact] = []
        seen: set[str] = set()
        for sym in self.defined_in(file):
            if sym.name in seen or sym.name in _UBIQUITOUS:
                continue
            seen.add(sym.name)
            imp = self.impact_of(sym.name, batch)
            imp.kind = sym.kind
            imp.parent = sym.parent
            out.append(imp)
        out.sort(
            key=lambda i: (-i.external_sites, -i.sibling_sites, i.name)
        )
        return out

    def slice_for(self, name: str, prefer: Path | None = None) -> Symbol | None:
        """Best definition site for *name*.

        Prefers a header (the declaration a consumer actually compiles
        against) over an implementation file, and skips record members —
        a field's slice is its parent record's slice.
        """
        cands = [s for s in self.symbols.get(name, []) if not s.is_member]
        if not cands:
            cands = list(self.symbols.get(name, []))
        if not cands:
            return None
        if prefer is not None:
            for s in cands:
                if s.file == prefer:
                    return s

        def rank(s: Symbol) -> tuple[int, int, str]:
            header = 0 if s.file.suffix.lower() in HEADER_EXTS else 1
            # A definition beats a bare prototype for the same name.
            proto = 1 if s.kind == "func" and s.decl.rstrip().endswith(";") else 0
            return (header, proto, str(s.file))

        return sorted(cands, key=rank)[0]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _default_read(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="replace")


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _line_start_offsets(text: str) -> list[int]:
    starts = [0]
    for m in re.finditer(r"\n", text):
        starts.append(m.end())
    return starts


def _line_of(line_starts: list[int], offset: int) -> int:
    lo, hi = 0, len(line_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if line_starts[mid] <= offset:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def _line_at(text: str, line_starts: list[int], line_no: int) -> str:
    start = line_starts[line_no - 1]
    end = text.find("\n", start)
    return text[start: end if end != -1 else len(text)].strip()


def _clip_decl(decl: str) -> str:
    decl = decl.strip()
    if len(decl) <= MAX_DECL_CHARS:
        return decl
    return decl[:MAX_DECL_CHARS] + "\n/* … declaration truncated … */"


def _record_members(body: str, kind: str) -> list[str]:
    """Member names of a struct/union/enum body.

    Conservative by design: anything it cannot parse is simply not reported,
    which costs the model a hint but never invents a symbol.
    """
    out: list[str] = []
    if kind == "enum":
        for part in body.split(","):
            part = part.split("=")[0].strip()
            if re.fullmatch(r"[A-Za-z_]\w*", part) and part not in _C_KEYWORDS:
                out.append(part)
        return out

    depth = 0
    buf: list[str] = []
    members: list[str] = []
    # Split on ';' at nesting depth 0 so an anonymous inner struct/union
    # (very common in embedded register maps) does not corrupt the split.
    for ch in body:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == ";" and depth == 0:
            members.append("".join(buf))
            buf = []
        else:
            buf.append(ch)

    for member in members:
        member = member.strip()
        if not member or member.startswith("#"):
            continue
        member = re.sub(r":\s*\d+\s*$", "", member).strip()   # bitfield width
        member = re.sub(r"\{.*?\}", " ", member, flags=re.DOTALL)
        for declarator in member.split(","):
            declarator = re.sub(r"\[[^\]]*\]", "", declarator).strip()
            idents = _IDENT_RE.findall(declarator)
            if not idents:
                continue
            name = idents[-1]
            if name in _C_KEYWORDS:
                continue
            out.append(name)
    return out


def extract_symbols(text: str, path: Path) -> list[Symbol]:
    """All definition sites in one file. Raw text in — comments are part of
    a declaration slice and are worth showing the model."""
    out: list[Symbol] = []
    starts = _line_start_offsets(text)

    def add(name: str, kind: str, offset: int, decl: str, parent: str = "") -> None:
        if not name or name in _C_KEYWORDS:
            return
        out.append(
            Symbol(
                name=name,
                kind=kind,
                file=path,
                line=_line_of(starts, offset),
                decl=_clip_decl(decl),
                parent=parent,
            )
        )

    consumed: list[tuple[int, int]] = []

    for m in _TYPEDEF_RECORD_RE.finditer(text):
        record_kind, tag, body, alias = m.group(1), m.group(2), m.group(3), m.group(4)
        decl = m.group(0)
        add(alias, "typedef", m.start(), decl)
        if tag:
            add(tag, record_kind, m.start(), decl)
        member_kind = "enumerator" if record_kind == "enum" else "field"
        for member in _record_members(body, record_kind):
            add(member, member_kind, m.start(), decl, parent=alias)
        consumed.append((m.start(), m.end()))

    for m in _RECORD_RE.finditer(text):
        if _overlaps(m.start(), consumed):
            continue
        record_kind, tag, body = m.group(1), m.group(2), m.group(3)
        decl = m.group(0)
        add(tag, record_kind, m.start(), decl)
        member_kind = "enumerator" if record_kind == "enum" else "field"
        for member in _record_members(body, record_kind):
            add(member, member_kind, m.start(), decl, parent=tag)
        consumed.append((m.start(), m.end()))

    for m in _TYPEDEF_SIMPLE_RE.finditer(text):
        if _overlaps(m.start(), consumed):
            continue
        add(m.group(2), "typedef", m.start(), m.group(0))

    guards = {m.group(1) for m in _INCLUDE_GUARD_RE.finditer(text)}
    for m in _DEFINE_RE.finditer(text):
        if m.group(1) in guards:
            continue
        add(m.group(1), "macro", m.start(), m.group(0))

    for m in _FUNC_DEF_RE.finditer(text):
        signature = f"{m.group(1).strip()} {m.group(2)}({' '.join(m.group(3).split())});"
        add(m.group(2), "func", m.start(), signature)

    for m in _FUNC_PROTO_RE.finditer(text):
        add(m.group(2), "func", m.start(), m.group(0))

    for m in _GLOBAL_RE.finditer(text):
        add(m.group(2), "global", m.start(), m.group(0))

    return out


def _overlaps(offset: int, spans: Iterable[tuple[int, int]]) -> bool:
    return any(start <= offset < end for start, end in spans)


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def render_public_surface(
    graph: CodeGraph,
    file: Path,
    batch: set[Path],
    *,
    max_rows: int = 60,
    max_usage_symbols: int = 12,
    max_usage_sites: int = 3,
) -> str:
    """The change-impact contract: what this file owns, and who leans on it."""
    surface = graph.public_surface(file, batch)
    load_bearing = [i for i in surface if i.external_sites or i.sibling_sites]
    if not load_bearing:
        return ""

    rows: list[str] = []
    for imp in surface[:max_rows]:
        if imp.frozen:
            tag = "FROZEN"
            where = (
                f"used at {imp.external_sites} site(s) in "
                f"{imp.external_files} file(s) you are NOT editing"
            )
        elif imp.sibling_sites:
            tag = "shared"
            where = (
                f"used at {imp.sibling_sites} site(s) in "
                f"{imp.sibling_files} other file(s) in this batch"
            )
        else:
            tag = "local "
            where = "not referenced outside this file"
        label = f"{imp.name} ({imp.parent})" if imp.parent else imp.name
        rows.append(f"  {tag}  {label:<34} {imp.kind:<10} {where}")

    hidden = max(0, len(surface) - max_rows)
    more = f"\n  … (+{hidden} lower-impact symbols omitted)" if hidden else ""

    # Real call sites for the most load-bearing symbols. The counts alone say
    # "do not rename this"; the sites say *how* it is consumed, which is what
    # decides whether a widening or a reordering is actually safe.
    usage: list[str] = []
    for imp in surface[:max_usage_symbols]:
        if not imp.frozen or not imp.samples:
            continue
        label = f"{imp.parent}.{imp.name}" if imp.parent else imp.name
        usage.append(f"\n**{label}** is used as:")
        for ref in imp.samples[:max_usage_sites]:
            usage.append(f"  {_display_path(graph, ref.file)}:{ref.line}: {ref.text[:110]}")
    usage_block = (
        "\nHow the FROZEN symbols are actually used by those files:\n"
        + "\n".join(usage) + "\n"
        if usage else ""
    )

    return (
        f"## Change Impact — public surface of {file.name}\n"
        "Symbols this file defines, and what depends on them TODAY.\n"
        "`FROZEN` means renaming or re-signaturing it breaks files that are "
        "not part of this transform and will not be regenerated.\n\n"
        "```\n" + "\n".join(rows) + more + "\n```\n"
        + usage_block
    )


def render_dependency_slices(
    graph: CodeGraph,
    file: Path,
    source_text: str,
    *,
    max_chars: int = 12_000,
) -> str:
    """Declaration slices for the symbols *file* actually references.

    Whole slices only: when the budget runs out the remaining slices are
    dropped intact rather than character-sliced, so the model never sees a
    struct cut in half.
    """
    stripped = _strip_comments(source_text)
    own = {s.name for s in graph.defined_in(file)}
    usage: dict[str, int] = defaultdict(int)
    for m in _IDENT_RE.finditer(stripped):
        name = m.group(0)
        if name in own or name in _UBIQUITOUS or name in _C_KEYWORDS:
            continue
        if name in graph.symbols:
            usage[name] += 1

    if not usage:
        return ""

    ranked = sorted(usage.items(), key=lambda kv: (-kv[1], kv[0]))
    emitted: set[str] = set()
    blocks: list[str] = []
    budget = max_chars
    dropped = 0

    for name, hits in ranked:
        sym = graph.slice_for(name)
        if sym is None:
            continue
        # A field and its parent record share one slice — emit it once.
        key = f"{sym.file}:{sym.line}"
        if key in emitted:
            continue
        rel = _display_path(graph, sym.file)
        block = (
            f"\n### {name} — {sym.kind}, defined in {rel}:{sym.line} "
            f"({hits} use(s) in {file.name})\n```c\n{sym.decl}\n```\n"
        )
        if len(block) > budget:
            dropped += 1
            continue
        emitted.add(key)
        blocks.append(block)
        budget -= len(block)

    if not blocks:
        return ""

    tail = (
        f"\n_({dropped} further declaration(s) omitted to fit the context "
        "budget.)_\n"
        if dropped else ""
    )
    return (
        "## Dependency Declarations (exact, from the codebase)\n"
        "The real declarations of what this file uses — ground truth for type "
        "names, field names, signatures and macro values. Use these exactly; "
        "do not guess or re-invent them.\n"
        + "".join(blocks)
        + tail
    )


def render_spec_targets(
    graph: CodeGraph,
    file: Path,
    spec_tokens: set[str],
    batch: set[Path],
    *,
    max_rows: int = 40,
) -> str:
    """Which change-spec identifiers this file OWNS vs merely CONSUMES.

    The transform runs per file; without this the model cannot tell which
    edits are its job and which belong to a sibling file in the same batch.
    """
    own = {s.name for s in graph.defined_in(file)}
    owns: list[str] = []
    consumes: list[str] = []
    for token in sorted(spec_tokens):
        if token not in graph.symbols or token in _UBIQUITOUS:
            continue
        if token in own:
            owns.append(token)
        else:
            sym = graph.slice_for(token)
            if sym is None:
                continue
            consumes.append(f"{token} (defined in {_display_path(graph, sym.file)})")

    if not owns and not consumes:
        return ""

    parts = [
        "## Change-Spec Symbols Present in This Codebase\n",
        "Identifiers named by the change specification that already exist in "
        "the code.\n\n",
    ]
    if owns:
        parts.append(
            f"**Defined in {file.name} — these edits are YOURS to make:**\n"
            + "\n".join(f"  - {n}" for n in owns[:max_rows]) + "\n\n"
        )
    if consumes:
        parts.append(
            "**Defined elsewhere — do NOT redefine them here; another file "
            "in this batch owns them:**\n"
            + "\n".join(f"  - {n}" for n in consumes[:max_rows]) + "\n"
        )
    return "".join(parts)


def _display_path(graph: CodeGraph, path: Path) -> str:
    for root in graph.roots:
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    return path.name
