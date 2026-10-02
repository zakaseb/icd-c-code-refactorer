"""
Deterministic helpers for the transform stage.

* SEARCH/REPLACE edit blocks. A large ``.c`` (tens of kB) cannot be re-emitted
  in one reply: prompt plus a full rewrite exceed the context window, the reply
  is cut off, and continuation passes only make it worse. Asking for edit
  blocks makes the output scale with the change instead of the file.
* No-op detection. A model handed a file in full often returns it unchanged;
  that is caught here instead of being accepted as a transform.

No LLM and no I/O.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_EDIT_BLOCK_RE = re.compile(
    r"<{5,9}[ \t]*SEARCH[ \t]*\r?\n(.*?)\r?\n?={5,9}[ \t]*\r?\n(.*?)\r?\n?>{5,9}[ \t]*REPLACE",
    re.DOTALL,
)
NO_MORE_EDITS = "NO MORE EDITS"

EDIT_FORMAT_INSTRUCTIONS = (
    "OUTPUT FORMAT — EDIT BLOCKS ONLY (the file is too large to re-emit):\n"
    "Reply with SEARCH/REPLACE blocks for the lines that must change:\n"
    "<<<<<<< SEARCH\n"
    "<exact lines copied from the current file>\n"
    "=======\n"
    "<the replacement lines>\n"
    ">>>>>>> REPLACE\n"
    "Rules:\n"
    "- SEARCH must copy 2-20 consecutive lines EXACTLY as they appear in the "
    "current file, enough to be unique. No `...` or elisions.\n"
    "- One block per separate change; blocks are applied top to bottom.\n"
    "- To delete lines, leave the replacement empty. To insert, include the "
    "neighbouring line in both SEARCH and REPLACE.\n"
    "- Do NOT output the whole file. No prose outside the blocks.\n"
    "- If you run out of room, stop after a complete block: you will be asked "
    "for the remaining edits against the updated file.\n"
    f"- When nothing more needs to change, reply exactly `{NO_MORE_EDITS}`.\n"
)


@dataclass
class EditResult:
    text: str
    applied: int
    failed: list[str]


def parse_edit_blocks(reply: str) -> list[tuple[str, str]]:
    """``(search, replace)`` pairs in reply order."""
    return [
        (m.group(1), m.group(2))
        for m in _EDIT_BLOCK_RE.finditer(reply or "")
        if m.group(1).strip()
    ]


def _aligned_hits(text: str, search: str) -> list[int]:
    """Exact occurrences of *search* that start and end on line boundaries."""
    hits = []
    start = text.find(search)
    while start != -1:
        end = start + len(search)
        if (start == 0 or text[start - 1] == "\n") and (
            end == len(text) or text[end] == "\n" or search.endswith("\n")
        ):
            hits.append(start)
        start = text.find(search, start + 1)
    return hits


def _trim_blank(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return lines


def apply_edit_blocks(text: str, blocks: list[tuple[str, str]]) -> EditResult:
    """Apply blocks in order: an exact line-aligned unique match first, then a
    whitespace-insensitive line match. Unmatched or ambiguous blocks are
    reported in ``failed`` and skipped."""
    applied = 0
    failed: list[str] = []
    for search, replace in blocks:
        aligned = _aligned_hits(text, search)
        if len(aligned) == 1:
            pos = aligned[0]
            text = text[:pos] + replace + text[pos + len(search):]
            applied += 1
            continue
        lines = text.split("\n")
        want = [s.strip() for s in _trim_blank(search.split("\n"))]
        have = [s.strip() for s in lines]
        hits = [
            i for i in range(len(have) - len(want) + 1)
            if want and have[i:i + len(want)] == want
        ]
        if len(hits) == 1:
            start = hits[0]
            lines[start:start + len(want)] = replace.split("\n") if replace else []
            text = "\n".join(lines)
            applied += 1
            continue
        first = next((ln.strip() for ln in search.split("\n") if ln.strip()), "")
        kind = "ambiguous" if len(hits) > 1 or len(aligned) > 1 else "not found"
        failed.append(f"{kind}: {first[:80]}")
    return EditResult(text, applied, failed)


_COMMENT_RE = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)


def code_fingerprint(text: str) -> str:
    """Code with comments and all whitespace removed."""
    return re.sub(r"\s+", "", _COMMENT_RE.sub("", text or ""))


def is_noop_transform(original: str, generated: str) -> bool:
    """True when *generated* is *original* with at most comment/whitespace
    changes."""
    return bool(original) and code_fingerprint(original) == code_fingerprint(generated)
