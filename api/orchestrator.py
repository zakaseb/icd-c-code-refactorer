"""
Sandbox-build debugging orchestrator agent.

A tool-using ReAct-style agent that drives the iterative debugging stage
of the sandbox build pipeline. It runs entirely against the local
llama-server (same model used elsewhere in the app), so all repository
contents and prompts stay on-device — no data leakage.

Why this exists
---------------
The previous fix loop did one thing per iteration: rewrite an entire
generated file (or a repo file) in a single LLM call given the latest
compile errors. For real-world Xilinx-SDK-scale repositories that has
several drawbacks:

1. **No memory.** Each iteration is independent: the model cannot see
   what it tried two iterations ago.
2. **No investigation.** It cannot peek at the offending header to
   understand a struct definition, or grep for who actually uses a
   symbol.
3. **No grouping.** Fifty errors caused by one missing struct field
   are treated as fifty unrelated bugs.
4. **Whole-file rewrites.** Every patch regenerates an entire file —
   wasting tokens and frequently introducing regressions.

The orchestrator addresses all four problems via a stateful, planning
loop with explicit tools (read_file, search, patch, build, …) and a
running transcript that the agent itself can inspect.

Action protocol
---------------
The agent emits a single block per turn:

    <think>
    short explanation, optional
    </think>
    <action>
    {"tool": "<name>", "args": {...}}
    </action>

We parse the JSON inside ``<action>`` and dispatch to the matching tool
implementation. Tool result is wrapped in an ``<observation>`` block and
appended to the conversation transcript on the next turn.

The orchestrator never edits files outside the sandbox, never executes
shell commands the user did not authorise (only the existing build
runner is exposed via the ``build`` tool), and stays on-device.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration knobs
# ---------------------------------------------------------------------------

DEFAULT_MAX_STEPS: int | None = None
DEFAULT_MAX_BUILDS = 25
DEFAULT_MAX_FILE_BYTES = 12_000          # per read_file response
DEFAULT_MAX_SEARCH_HITS = 30
DEFAULT_MAX_TRANSCRIPT_CHARS = 60_000    # rolling-window cap for prompt
DEFAULT_BUILD_OUTPUT_BUDGET = 6_000      # per build observation in transcript
DEFAULT_OBSERVATION_BUDGET = 4_000       # per non-build observation


# ---------------------------------------------------------------------------
# Action / Observation containers
# ---------------------------------------------------------------------------

@dataclass
class Action:
    tool: str
    args: dict[str, Any]
    raw: str = ""           # raw JSON the model emitted, for debugging
    thought: str = ""

    def render(self) -> str:
        body = json.dumps({"tool": self.tool, "args": self.args}, indent=2)
        if self.thought:
            return f"<think>\n{self.thought.strip()}\n</think>\n<action>\n{body}\n</action>"
        return f"<action>\n{body}\n</action>"


@dataclass
class Observation:
    text: str
    truncated: bool = False
    error: bool = False

    def render(self) -> str:
        return f"<observation>\n{self.text}\n</observation>"


@dataclass
class StepRecord:
    """One agent turn (decision + result) kept in the transcript."""
    step: int
    action: Action
    observation: Observation


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are an autonomous embedded-C debugging orchestrator working inside a \
sandboxed copy of a user's repository. The user transformed some C source \
files (the "generated files") to comply with a new ICD (Interface Control \
Document); the transformed code does not yet compile. Your job is to drive \
the project to a clean build using the tools listed below.

ENVIRONMENT
-----------
- The repository sits under /SANDBOX_DIR/ inside the container.
- Generated files were placed at known paths (listed in the brief below).
- Build is run via the `build` tool; you decide when to call it.
- Toolchain is GCC (native or arm-none-eabi cross). If the build is
  cross-compiled, linker errors about missing BSP symbols (Xil_*,
  xil_printf, …) mean the BSP libraries are not in the upload — they
  are *not* errors you can or should fix. Focus exclusively on COMPILE
  errors. The compile-only fallback already accepts that case.

TOOLS
-----
You issue ONE tool call per turn using this exact format:

    <think>
    one short paragraph of strategy / observation, optional
    </think>
    <action>
    {"tool": "<tool_name>", "args": { ... }}
    </action>

Available tools:

  read_file       args: {"path": "<rel/path>", "offset": <int?>, "limit": <int?>}
                  Read up to N bytes of a file (relative to sandbox root).

  list_dir        args: {"path": "<rel/path>", "max": <int?>}
                  List entries in a directory.

  search          args: {"pattern": "<regex>", "path": "<rel/dir?>",
                          "max_hits": <int?>, "ext": "c|h|all"}
                  Search source/header files for a regex (ripgrep-like).

  find_files      args: {"name": "<basename>"}
                  Locate files by basename.

  patch           args: {"path": "<rel/path>", "find": "<exact text>",
                         "replace": "<new text>"}
                  Surgical edit. The "find" string MUST match exactly once,
                  including whitespace. Prefer this over write_file.

  write_file      args: {"path": "<rel/path>", "content": "<full file body>"}
                  Replace a file's full content. Use sparingly — only when
                  the file needs sweeping changes.

  reset_file      args: {"path": "<rel/path>"}
                  Restore a file to the snapshot taken before the loop
                  started. Useful when a patch made things worse.

  build           args: {}
                  Run the project build. Returns success flag plus the
                  trimmed compile output (errors, if any).

  note            args: {"text": "<short remark>"}
                  Record a finding to your own scratchpad (visible in
                  later turns). No side effects.

  done            args: {"reason": "<short reason>"}
                  Stop. Use this only when the last build call succeeded.

STRATEGY
--------
1. Always read the latest build output before changing anything.
2. Prefer `patch` over `write_file`. Whole-file rewrites cause regressions.
3. Group errors by ROOT CAUSE before patching. If 30 errors point at one
   missing struct field, fix the header *first* and rebuild — do not
   patch each consumer individually.
4. Inspect headers/types with `read_file` and `search` before editing
   anything you are unsure about. Hallucinated patches waste turns.
5. Avoid repeating an action that already failed. If a patch doesn't
   apply (because "find" wasn't unique or didn't match), re-read the
   file to see the actual current text, do not retry blindly.
6. When the build error count drops to zero, call `done`. If the build
   succeeded with linker-only failures during cross-compilation, that
   is also a successful state.

OUTPUT RULES
------------
- ALWAYS output exactly one <action>...</action> block.
- The JSON inside <action> MUST be valid; no comments, no trailing commas.
- Do NOT wrap the action in markdown fences.
- Do NOT output anything after the closing </action> tag.
"""


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)


def parse_action(text: str) -> Action | None:
    """Pull the first <action>{...}</action> JSON object out of *text*."""
    m = _ACTION_RE.search(text)
    if not m:
        return None
    body = m.group(1).strip()
    body = body.strip("`")
    if body.startswith("json"):
        body = body[4:].strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        # Try to repair common mistakes: stray trailing text after the
        # JSON object, or stray markdown fences.
        end = body.rfind("}")
        if end > 0:
            try:
                data = json.loads(body[:end + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None
    tool = data.get("tool") or data.get("name")
    args = data.get("args") or data.get("arguments") or {}
    if not isinstance(tool, str) or not isinstance(args, dict):
        return None
    thought = ""
    tm = _THINK_RE.search(text)
    if tm:
        thought = tm.group(1).strip()
    return Action(tool=tool, args=args, raw=body, thought=thought)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _normalise_rel(rel: str) -> str:
    """Strip leading absolute markers and a single leading ``./``.

    We deliberately do NOT collapse ``..`` here — that's the path-traversal
    check's job in :func:`_safe_resolve`.
    """
    rel = rel.replace("\\", "/")
    while rel.startswith("/"):
        rel = rel[1:]
    if rel.startswith("./"):
        rel = rel[2:]
    return rel


def _safe_resolve(sandbox_dir: Path, rel: str) -> Path:
    """Resolve *rel* against *sandbox_dir* refusing escape attempts."""
    if not rel:
        raise ValueError("path is required")
    rel = _normalise_rel(rel)
    p = (sandbox_dir / rel).resolve()
    sb = sandbox_dir.resolve()
    try:
        p.relative_to(sb)
    except ValueError as exc:
        raise ValueError(
            f"path '{rel}' is outside the sandbox; refusing access."
        ) from exc
    return p


def _resolve_or_obs(
    ctx: "ToolContext", rel: str, tool_name: str,
) -> tuple[Path | None, Observation | None]:
    """Helper: resolve a path or return an error Observation."""
    try:
        return _safe_resolve(ctx.sandbox_dir, rel), None
    except ValueError as e:
        return None, Observation(text=f"{tool_name}: {e}", error=True)


def _truncate_observation(text: str, budget: int) -> tuple[str, bool]:
    if len(text) <= budget:
        return text, False
    head = text[: budget // 2]
    tail = text[-(budget // 2):]
    return f"{head}\n\n[… {len(text) - budget} chars omitted …]\n\n{tail}", True


@dataclass
class ToolContext:
    sandbox_dir: Path
    file_index: dict[str, list[Path]]
    snapshots: dict[Path, str]
    build_runner: Callable[[], tuple[bool, str]]
    observation_budget: int = DEFAULT_OBSERVATION_BUDGET
    build_output_budget: int = DEFAULT_BUILD_OUTPUT_BUDGET
    build_calls: int = 0
    last_build_success: bool = False
    notes: list[str] = field(default_factory=list)


def tool_read_file(ctx: ToolContext, args: dict) -> Observation:
    rel = _normalise_rel(str(args.get("path", "")))
    offset = int(args.get("offset") or 0)
    limit = int(args.get("limit") or DEFAULT_MAX_FILE_BYTES)
    limit = max(256, min(limit, DEFAULT_MAX_FILE_BYTES))
    p, err = _resolve_or_obs(ctx, rel, "read_file")
    if err:
        return err
    assert p is not None
    if not p.exists() or not p.is_file():
        return Observation(
            text=f"read_file: '{rel}' does not exist or is not a regular file.",
            error=True,
        )
    try:
        data = p.read_text(errors="replace")
    except OSError as e:
        return Observation(text=f"read_file: failed to read '{rel}': {e}", error=True)
    total = len(data)
    chunk = data[offset: offset + limit]
    truncated = len(chunk) < (total - offset) or offset > 0
    header = (
        f"read_file: '{rel}' "
        f"(showing {len(chunk)} of {total} chars, offset={offset})\n"
        f"{'-' * 60}\n"
    )
    body, body_truncated = _truncate_observation(
        header + chunk, ctx.observation_budget
    )
    return Observation(text=body, truncated=truncated or body_truncated)


def tool_list_dir(ctx: ToolContext, args: dict) -> Observation:
    rel = _normalise_rel(str(args.get("path", ""))) or "."
    max_entries = int(args.get("max") or 100)
    p, err = _resolve_or_obs(ctx, rel, "list_dir")
    if err:
        return err
    assert p is not None
    if not p.exists() or not p.is_dir():
        return Observation(text=f"list_dir: '{rel}' is not a directory.", error=True)
    entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    rendered = []
    for entry in entries[:max_entries]:
        kind = "dir " if entry.is_dir() else "file"
        rel_e = entry.relative_to(ctx.sandbox_dir)
        rendered.append(f"  {kind}  {rel_e}")
    extra = len(entries) - max_entries
    suffix = f"\n  … {extra} more entries omitted" if extra > 0 else ""
    body = f"list_dir: '{rel}' ({len(entries)} entries)\n" + "\n".join(rendered) + suffix
    return Observation(text=body)


def tool_search(ctx: ToolContext, args: dict) -> Observation:
    pattern = str(args.get("pattern", ""))
    if not pattern:
        return Observation(text="search: 'pattern' is required.", error=True)
    sub = _normalise_rel(str(args.get("path", "")))
    ext = str(args.get("ext", "all")).lower()
    max_hits = int(args.get("max_hits") or DEFAULT_MAX_SEARCH_HITS)
    max_hits = max(1, min(max_hits, 200))

    if sub:
        base, err = _resolve_or_obs(ctx, sub, "search")
        if err:
            return err
        assert base is not None
    else:
        base = ctx.sandbox_dir
    if not base.exists():
        return Observation(text=f"search: path '{sub}' not found.", error=True)

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return Observation(text=f"search: invalid regex: {e}", error=True)

    suffixes: tuple[str, ...]
    if ext == "c":
        suffixes = (".c",)
    elif ext == "h":
        suffixes = (".h",)
    else:
        suffixes = (".c", ".h")

    hits: list[str] = []
    walked = 0
    for p in base.rglob("*"):
        if not p.is_file() or p.suffix not in suffixes:
            continue
        walked += 1
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                rel_p = p.relative_to(ctx.sandbox_dir)
                hits.append(f"{rel_p}:{i}: {line.rstrip()}")
                if len(hits) >= max_hits:
                    break
        if len(hits) >= max_hits:
            break

    header = (
        f"search: pattern={pattern!r} ext={ext} "
        f"under {sub or '.'} -> {len(hits)} hits "
        f"(scanned {walked} files)\n{'-' * 60}\n"
    )
    body = header + "\n".join(hits) if hits else header + "(no matches)"
    body, _ = _truncate_observation(body, ctx.observation_budget)
    return Observation(text=body)


def tool_find_files(ctx: ToolContext, args: dict) -> Observation:
    name = str(args.get("name", "")).strip()
    if not name:
        return Observation(text="find_files: 'name' is required.", error=True)
    matches: list[Path]
    if name in ctx.file_index:
        matches = list(ctx.file_index[name])
    else:
        matches = [p for p in ctx.sandbox_dir.rglob(name) if p.is_file()]
    rels = sorted(str(m.relative_to(ctx.sandbox_dir)) for m in matches)
    body = (
        f"find_files: name={name!r} -> {len(rels)} match(es)\n"
        + "\n".join(f"  {r}" for r in rels[:50])
    )
    if len(rels) > 50:
        body += f"\n  … {len(rels) - 50} more omitted"
    return Observation(text=body)


def tool_patch(ctx: ToolContext, args: dict) -> Observation:
    rel = _normalise_rel(str(args.get("path", "")))
    find = args.get("find")
    replace = args.get("replace")
    if not rel:
        return Observation(text="patch: 'path' is required.", error=True)
    if not isinstance(find, str) or not isinstance(replace, str):
        return Observation(
            text="patch: 'find' and 'replace' must both be strings.",
            error=True,
        )
    if not find:
        return Observation(text="patch: 'find' must be non-empty.", error=True)

    p, err = _resolve_or_obs(ctx, rel, "patch")
    if err:
        return err
    assert p is not None
    if not p.exists() or not p.is_file():
        return Observation(
            text=f"patch: '{rel}' does not exist or is not a regular file.",
            error=True,
        )

    try:
        text = p.read_text(errors="replace")
    except OSError as e:
        return Observation(text=f"patch: read failed: {e}", error=True)

    count = text.count(find)
    if count == 0:
        return Observation(
            text=(
                f"patch: 'find' string did not occur in '{rel}'.\n"
                "Re-read the file (using read_file) to confirm the actual text "
                "before retrying. Whitespace and indentation must match exactly."
            ),
            error=True,
        )
    if count > 1:
        return Observation(
            text=(
                f"patch: 'find' string occurs {count} times in '{rel}'. "
                "Make it unique by including more surrounding context "
                "(neighbouring lines or qualifying tokens) before retrying."
            ),
            error=True,
        )

    new_text = text.replace(find, replace, 1)
    if new_text == text:
        return Observation(text=f"patch: no-op (replace == find) in '{rel}'.", error=True)
    try:
        p.write_text(new_text)
    except OSError as e:
        return Observation(text=f"patch: write failed: {e}", error=True)
    delta = len(new_text) - len(text)
    return Observation(
        text=(
            f"patch: '{rel}' updated "
            f"(Δ {delta:+d} chars; {len(find)}→{len(replace)})."
        ),
    )


def tool_write_file(ctx: ToolContext, args: dict) -> Observation:
    rel = _normalise_rel(str(args.get("path", "")))
    content = args.get("content")
    if not rel:
        return Observation(text="write_file: 'path' is required.", error=True)
    if not isinstance(content, str):
        return Observation(text="write_file: 'content' must be a string.", error=True)
    if len(content) < 20:
        return Observation(
            text="write_file: refusing to write a near-empty file. "
                 "Use patch for small edits.",
            error=True,
        )
    p, err = _resolve_or_obs(ctx, rel, "write_file")
    if err:
        return err
    assert p is not None
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(content)
    except OSError as e:
        return Observation(text=f"write_file: write failed: {e}", error=True)
    return Observation(text=f"write_file: '{rel}' written ({len(content)} chars).")


def tool_reset_file(ctx: ToolContext, args: dict) -> Observation:
    rel = _normalise_rel(str(args.get("path", "")))
    if not rel:
        return Observation(text="reset_file: 'path' is required.", error=True)
    p, err = _resolve_or_obs(ctx, rel, "reset_file")
    if err:
        return err
    assert p is not None
    if p not in ctx.snapshots:
        return Observation(
            text=f"reset_file: no snapshot recorded for '{rel}'.", error=True,
        )
    try:
        p.write_text(ctx.snapshots[p])
    except OSError as e:
        return Observation(text=f"reset_file: write failed: {e}", error=True)
    return Observation(text=f"reset_file: '{rel}' restored from initial snapshot.")


def tool_build(ctx: ToolContext, args: dict) -> Observation:
    ctx.build_calls += 1
    success, output = ctx.build_runner()
    ctx.last_build_success = success
    head = (
        f"build #{ctx.build_calls}: "
        f"{'SUCCESS' if success else 'FAILED'}\n{'-' * 60}\n"
    )
    body, _ = _truncate_observation(head + output, ctx.build_output_budget)
    return Observation(text=body)


def tool_note(ctx: ToolContext, args: dict) -> Observation:
    text = str(args.get("text", "")).strip()
    if not text:
        return Observation(text="note: 'text' is required.", error=True)
    ctx.notes.append(text)
    return Observation(text=f"note recorded ({len(ctx.notes)} total): {text[:200]}")


def tool_done(ctx: ToolContext, args: dict) -> Observation:
    reason = str(args.get("reason", "")).strip() or "no reason given"
    return Observation(text=f"done: {reason}")


TOOL_REGISTRY: dict[str, Callable[[ToolContext, dict], Observation]] = {
    "read_file":  tool_read_file,
    "list_dir":   tool_list_dir,
    "search":     tool_search,
    "find_files": tool_find_files,
    "patch":      tool_patch,
    "write_file": tool_write_file,
    "reset_file": tool_reset_file,
    "build":      tool_build,
    "note":       tool_note,
    "done":       tool_done,
}


# ---------------------------------------------------------------------------
# Transcript management
# ---------------------------------------------------------------------------

def render_transcript(
    history: list[StepRecord],
    max_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS,
) -> str:
    """Render the running history with a head+tail rolling-window cap.

    The first 1-2 turns set strategy; the most recent turns matter most for
    the next decision. So we keep both ends and drop the middle.
    """
    if not history:
        return "(no prior actions)"
    parts: list[str] = []
    for rec in history:
        parts.append(rec.action.render())
        parts.append(rec.observation.render())
    full = "\n".join(parts)
    if len(full) <= max_chars:
        return full

    head_budget = max_chars // 4
    tail_budget = max_chars - head_budget
    head_lines: list[str] = []
    tail_lines: list[str] = []
    used_head = used_tail = 0
    for rec in history:
        chunk = rec.action.render() + "\n" + rec.observation.render()
        if used_head + len(chunk) <= head_budget:
            head_lines.append(chunk)
            used_head += len(chunk) + 1
    for rec in reversed(history):
        chunk = rec.action.render() + "\n" + rec.observation.render()
        if used_tail + len(chunk) <= tail_budget:
            tail_lines.insert(0, chunk)
            used_tail += len(chunk) + 1
        else:
            break
    omitted = len(history) - len(head_lines) - len(tail_lines)
    sep = (
        f"\n[… {omitted} earlier turns elided to fit context window …]\n"
        if omitted > 0 else "\n"
    )
    return "\n".join(head_lines) + sep + "\n".join(tail_lines)


def build_brief(
    sandbox_dir: Path,
    build_info: dict,
    sandbox_cc: str,
    is_cross: bool,
    gen_files: dict[str, str],          # rel_path -> "generated" / "original"
    change_spec: str,
    repo_knowledge: str,
    initial_build_output: str,
    notes: list[str],
    gitnexus_report: str = "",
) -> str:
    """Compose the user message that frames the task each turn."""
    rel_build = (
        str(build_info["path"].relative_to(sandbox_dir))
        if build_info.get("path") else "(none)"
    )
    parts = [
        "## Task brief",
        f"- Sandbox root: {sandbox_dir}",
        f"- Build system: {build_info.get('type', 'none')} at {rel_build}",
        f"- Compiler: {sandbox_cc}",
        f"- Cross-compile: {'yes' if is_cross else 'no'}",
        "",
        "## Generated files (replaced inside the repo by the ICD transform)",
    ]
    for rel, _ in sorted(gen_files.items()):
        parts.append(f"  - {rel}")
    parts.extend([
        "",
        "## Latest build output (most recent attempt)",
        "```",
        _truncate(initial_build_output, 4000),
        "```",
    ])
    if notes:
        parts.extend([
            "",
            "## Your scratchpad (use `note` to add more)",
            *(f"- {n}" for n in notes[-12:]),
        ])
    parts.extend([
        "",
        "## ICD change specification",
        _truncate(change_spec, 4000),
    ])
    if repo_knowledge:
        parts.extend([
            "",
            "## Repository codebase knowledge",
            _truncate(repo_knowledge, 3000),
        ])
    if gitnexus_report:
        parts.extend([
            "",
            "## GitNexus codebase understanding",
            "Embedded-systems-specific relationships (ISR/task wiring, "
            "drivers/peripherals, RTOS or superloop, state machines, "
            "communication stacks, memory ownership, HAL boundary, "
            "bootloader/firmware-update hooks, safety chains, "
            "cross-module #include graph, global variable read/write "
            "graph, build-script deps). Honor these when patching.",
            _truncate(gitnexus_report, 4000),
        ])
    parts.extend([
        "",
        "## Reminder",
        "Emit exactly ONE <action> block with valid JSON. Do not output "
        "anything after </action>. When the build is green, call `done`.",
    ])
    return "\n".join(parts)


def _truncate(text: str, n: int) -> str:
    if not text or len(text) <= n:
        return text or ""
    return text[:n] + f"\n[… {len(text) - n} more chars …]"


# ---------------------------------------------------------------------------
# Orchestrator entry point
# ---------------------------------------------------------------------------

def run_orchestrator(
    *,
    sandbox_dir: Path,
    build_info: dict,
    sandbox_cc: str,
    is_cross: bool,
    gen_files: dict[str, str],
    change_spec: str,
    repo_knowledge: str,
    file_index: dict[str, list[Path]],
    snapshots: dict[Path, str],
    build_runner: Callable[[], tuple[bool, str]],
    llm_stream: Callable[..., Iterator[str]],
    max_steps: int | None = DEFAULT_MAX_STEPS,
    max_builds: int = DEFAULT_MAX_BUILDS,
    max_input_tokens: int = 24_000,
    max_output_tokens: int = 1024,
    gitnexus_report: str = "",
) -> Iterator[dict]:
    """Drive the debugging loop. Yields events for SSE forwarding.

    Event shape:
        {"type": "step",      "step": int}
        {"type": "thought",   "step": int, "text": str}
        {"type": "action",    "step": int, "tool": str, "args": dict}
        {"type": "observation","step": int, "text": str, "error": bool}
        {"type": "build",     "step": int, "success": bool, "calls": int}
        {"type": "done",      "success": bool, "reason": str, "steps": int,
                              "builds": int}
        {"type": "raw_token", "text": str}     # forwarded from LLM stream
        {"type": "warning",   "message": str}

    The caller is responsible for translating these into SSE messages.
    """
    ctx = ToolContext(
        sandbox_dir=sandbox_dir,
        file_index=file_index,
        snapshots=snapshots,
        build_runner=build_runner,
    )

    success, initial_build_output = build_runner()
    ctx.build_calls = 1
    ctx.last_build_success = success
    yield {"type": "build", "step": 0, "success": success, "calls": 1}
    if success:
        yield {
            "type": "done",
            "success": True,
            "reason": "Build already succeeds without any patches.",
            "steps": 0,
            "builds": ctx.build_calls,
        }
        return

    history: list[StepRecord] = []
    last_build_output = initial_build_output

    step = 1
    while True:
        if max_steps is not None and step > max_steps:
            yield {
                "type": "done",
                "success": ctx.last_build_success,
                "reason": (
                    f"step budget exhausted ({max_steps} steps, "
                    f"{ctx.build_calls} builds)."
                ),
                "steps": step - 1,
                "builds": ctx.build_calls,
            }
            return

        yield {"type": "step", "step": step}

        # --- Build prompt ---------------------------------------------------
        brief = build_brief(
            sandbox_dir=sandbox_dir,
            build_info=build_info,
            sandbox_cc=sandbox_cc,
            is_cross=is_cross,
            gen_files=gen_files,
            change_spec=change_spec,
            repo_knowledge=repo_knowledge,
            initial_build_output=last_build_output,
            notes=ctx.notes,
            gitnexus_report=gitnexus_report,
        )
        transcript = render_transcript(history)
        char_budget = max_input_tokens * 4
        prompt = (
            f"{brief}\n\n## Prior turns\n{transcript}\n\n"
            f"## Your turn (step {step}"
            + (f"/{max_steps}" if max_steps is not None else "")
            + ")\n"
            "Decide the single best next tool call."
        )
        if len(prompt) > char_budget:
            keep = char_budget - len(brief) - 200
            transcript = render_transcript(history, max_chars=max(2000, keep))
            prompt = (
                f"{brief}\n\n## Prior turns\n{transcript}\n\n"
                f"## Your turn (step {step}"
                + (f"/{max_steps}" if max_steps is not None else "")
                + ")\n"
                "Decide the single best next tool call."
            )

        # --- Call LLM (stream) ----------------------------------------------
        meta: dict = {}
        parts: list[str] = []
        try:
            for chunk in llm_stream(
                ORCHESTRATOR_SYSTEM_PROMPT, prompt,
                max_tokens=max_output_tokens, meta=meta,
            ):
                parts.append(chunk)
                yield {"type": "raw_token", "text": chunk}
        except Exception as e:
            yield {
                "type": "warning",
                "message": f"orchestrator LLM call failed at step {step}: {e}",
            }
            yield {
                "type": "done",
                "success": ctx.last_build_success,
                "reason": f"LLM error: {e}",
                "steps": step - 1,
                "builds": ctx.build_calls,
            }
            return

        raw = "".join(parts).strip()

        # If the model truncated mid-action, do one continuation pass.
        if meta.get("finish_reason") == "length" and "</action>" not in raw:
            cont_meta: dict = {}
            cont_prompt = (
                "Your previous response was cut off before </action>. "
                "Resume EXACTLY where you stopped and finish the JSON. "
                f"End of what you wrote:\n---\n{raw[-2000:]}\n---\n"
            )
            try:
                for chunk in llm_stream(
                    ORCHESTRATOR_SYSTEM_PROMPT, cont_prompt,
                    max_tokens=max_output_tokens, meta=cont_meta,
                ):
                    parts.append(chunk)
                    yield {"type": "raw_token", "text": chunk}
                raw = "".join(parts).strip()
            except Exception as e:
                yield {
                    "type": "warning",
                    "message": f"orchestrator continuation failed: {e}",
                }

        action = parse_action(raw)
        if action is None:
            obs = Observation(
                text=(
                    "no parsable <action> JSON detected in response. "
                    "Try again. Output exactly ONE <action> block with valid JSON."
                ),
                error=True,
            )
            history.append(
                StepRecord(
                    step=step,
                    action=Action(tool="(invalid)", args={}, raw=raw[:500]),
                    observation=obs,
                )
            )
            yield {
                "type": "observation",
                "step": step,
                "text": obs.text,
                "error": True,
            }
            continue

        if action.thought:
            yield {"type": "thought", "step": step, "text": action.thought}
        yield {
            "type": "action",
            "step": step,
            "tool": action.tool,
            "args": action.args,
        }

        # --- Dispatch -------------------------------------------------------
        if action.tool == "build" and ctx.build_calls >= max_builds:
            obs = Observation(
                text=(
                    f"build budget exhausted ({ctx.build_calls}/{max_builds}). "
                    "Make more patches before the next build."
                ),
                error=True,
            )
        else:
            handler = TOOL_REGISTRY.get(action.tool)
            if handler is None:
                obs = Observation(
                    text=(
                        f"unknown tool '{action.tool}'. "
                        f"Available: {', '.join(sorted(TOOL_REGISTRY))}"
                    ),
                    error=True,
                )
            else:
                try:
                    obs = handler(ctx, action.args)
                except Exception as e:
                    obs = Observation(
                        text=f"tool '{action.tool}' raised {type(e).__name__}: {e}",
                        error=True,
                    )

        history.append(StepRecord(step=step, action=action, observation=obs))
        yield {
            "type": "observation",
            "step": step,
            "text": obs.text,
            "error": obs.error,
        }

        if action.tool == "build":
            yield {
                "type": "build",
                "step": step,
                "success": ctx.last_build_success,
                "calls": ctx.build_calls,
            }
            last_build_output = obs.text
            if ctx.last_build_success:
                yield {
                    "type": "done",
                    "success": True,
                    "reason": (
                        f"Build succeeded after {step} agent step(s) "
                        f"({ctx.build_calls} build calls)."
                    ),
                    "steps": step,
                    "builds": ctx.build_calls,
                }
                return

        if action.tool == "done":
            # Trust 'done' only if the most recent build was a success.
            if ctx.last_build_success:
                yield {
                    "type": "done",
                    "success": True,
                    "reason": str(action.args.get("reason", "agent signalled done")),
                    "steps": step,
                    "builds": ctx.build_calls,
                }
                return
            # Otherwise force one more build to confirm.
            obs2 = tool_build(ctx, {})
            history[-1] = StepRecord(
                step=step,
                action=action,
                observation=Observation(
                    text=obs.text + "\n\n(forced build to verify done)\n" + obs2.text,
                ),
            )
            yield {
                "type": "build",
                "step": step,
                "success": ctx.last_build_success,
                "calls": ctx.build_calls,
            }
            last_build_output = obs2.text
            if ctx.last_build_success:
                yield {
                    "type": "done",
                    "success": True,
                    "reason": "Verified by forced build after `done`.",
                    "steps": step,
                    "builds": ctx.build_calls,
                }
                return

        step += 1
