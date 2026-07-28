"""Stage 3 — GenerationTeam (SSE stage: ``transform``).

Multi-agent system for code generation.  Headers (.h) are generated FIRST,
then implementation files (.c) — the .c agents see the freshly generated
headers so interface and implementation stay consistent.

Roster
    VariantScout           — detects when the Target ICD defines multiple
                             variants of a header's peripheral (JSON verdict)
                             and plans one output file per variant.
    InterfaceArchitect     — generates the .h files (one per variant when
                             applicable, exactly like the classic pipeline).
    ImplementationEngineer — generates the .c files against the NEW headers.
    CompletionCritic       — deterministic: completeness gates
                             (braces/guards/length) with focused
                             continuation retries.

Artifacts (identical to the classic pipeline)
    generated_code/<file>.c/.h and per-variant
    generated_code/<stem>_<Variant>.h files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ..agents import Agent
from ..blackboard import Blackboard
from ..context import PipelineContext
from .base import Team

_MAX_GEN_ATTEMPTS = 3


def _transform_system(app) -> str:
    """The proven embedded/RTOS code-generation contract.

    Kept aligned with the classic pipeline: Xilinx SDK 2018 / GCC 7.3.1 /
    gnu99 / newlib / freestanding — the constraints that make the output
    compatible with the RTOS embedded development framework.
    """
    return (
        "You are an expert C programmer specializing in embedded systems "
        "and interface implementations governed by Interface Control "
        "Documents.\n\n"
        "TARGET TOOLCHAIN:\n"
        "- Xilinx SDK 2018.x with GCC 7.3.1 (arm-none-eabi / mb-gcc)\n"
        "- C standard: C99 with GCC extensions (-std=gnu99) — "
        "GCC-specific constructs commonly used in embedded code "
        "are ACCEPTED, including unnamed structs/unions, "
        "`__attribute__`, statement expressions, and M_PI / M_E "
        "from <math.h>\n"
        "- C library: newlib (NOT glibc) — do NOT use glibc-specific "
        "functions (e.g. asprintf, getline, strdup, strndup, vasprintf)\n"
        "- Use <stdint.h> fixed-width types (uint8_t, uint16_t, uint32_t)\n"
        "- No POSIX headers (unistd.h, sys/*.h) — embedded freestanding\n"
        "- Avoid GCC extensions added after GCC 7 (no "
        "__attribute__((access)), no __builtin_expect_with_probability, "
        "etc.)\n\n"
        "RULES:\n"
        "1. Output ONLY the complete, transformed C source code\n"
        "2. Preserve the overall code architecture, style, and conventions\n"
        "3. Apply ALL changes required by the target ICD per the change "
        "specification\n"
        "4. Update data structures, function signatures, constants, enums, "
        "macros\n"
        "5. Update comments/doc-strings to reflect the new ICD version\n"
        "6. Ensure type correctness and compilability with GCC 7.3.1\n"
        "7. Keep header/source consistency across the project\n"
        "8. Do NOT add prose explanations — only output C code\n"
        "9. Wrap the entire output in ```c ... ``` fences\n"
        "10. Match naming conventions, coding style, variable naming, "
        "and communication patterns from the repository codebase\n"
        "11. Ensure #include directives reference correct repository headers\n"
        "12. Maintain compatibility with all dependent modules in the "
        "repository"
        f"{app.PERIPHERAL_VARIATION_CODEGEN_GUIDANCE}"
    )


class GenerationTeam(Team):
    stage = "transform"
    title = "Code Generation Team"

    def __init__(self):
        super().__init__()
        self.variant_scout = Agent(
            name="VariantScout",
            role="peripheral-variant detection and file planning",
        )
        self.interface_architect = Agent(
            name="InterfaceArchitect",
            role=".h generation (headers first)",
        )
        self.impl_engineer = Agent(
            name="ImplementationEngineer",
            role=".c generation against the new headers",
        )
        self.completion_critic = Agent(
            name="CompletionCritic",
            role="deterministic completeness gate + retry director",
        )
        self.agents = [
            self.variant_scout, self.interface_architect,
            self.impl_engineer, self.completion_critic,
        ]

    # ------------------------------------------------------------------
    def run(self, ctx: PipelineContext, bb: Blackboard) -> Iterator[dict]:
        a = ctx.app
        code_files = ctx.code_files
        headers = [p for p in code_files if p.name.lower().endswith(".h")]
        sources = [p for p in code_files if not p.name.lower().endswith(".h")]
        ordered = headers + sources        # .h first, then .c — by design

        yield self.evt_stage(
            f"Generating code: {len(headers)} header(s) first, then "
            f"{len(sources)} source file(s)…"
        )
        yield self.evt_roster()

        feedback = self.feedback_section(bb)
        if feedback:
            yield self.evt_info(
                "Regenerating with downstream feedback injected "
                f"({len(bb.pending_feedback(self.stage))} item(s))."
            )

        change_spec = str(bb.data.get("change_spec", ""))
        target_summary = str(bb.data.get("target_summary", ""))
        repo_knowledge = str(bb.data.get("repo_knowledge", ""))
        gitnexus_report = str(bb.data.get("gitnexus_report", ""))
        impact_map = str(bb.data.get("impact_map", ""))

        all_code_ctx = ""
        for cf in code_files:
            all_code_ctx += f"\n### File: {cf.name}\n```c\n{cf.read_text()}\n```\n"
        all_code_ctx = a._truncate_text(
            all_code_ctx, a.MAX_CODE_CONTEXT_CHARS, "all_code_ctx",
        )

        transform_system = _transform_system(a)
        generated: dict[str, str] = {}
        failed_files: list[str] = []

        for i, code_file in enumerate(ordered):
            fname = code_file.name
            yield self.evt_stage(
                f"Transforming {fname}…",
                file=fname, index=i, total=len(ordered),
            )
            original = code_file.read_text()
            base_sections = self._base_sections(
                ctx, original, change_spec, target_summary, repo_knowledge,
                gitnexus_report, all_code_ctx, impact_map, feedback,
                generated,
            )
            is_header = fname.lower().endswith(".h")

            # ---- VariantScout: per-variant header planning -------------
            if is_header:
                variant_names = a._detect_peripheral_variants(
                    original, fname, change_spec, target_summary,
                )
                if len(variant_names) >= 2:
                    emitted = yield from self._emit_variant_headers(
                        ctx, bb, fname, original, variant_names,
                        base_sections, transform_system, generated,
                    )
                    if emitted:
                        continue    # per-variant files written

            # ---- Single-file generation with CompletionCritic gate ------
            code = yield from self._generate_single(
                ctx, fname, original, base_sections, transform_system,
            )
            if code:
                (ctx.gen_dir / fname).write_text(code)
                generated[fname] = code
                yield {
                    "type": "file_complete", "file": fname, "size": len(code),
                }
            else:
                failed_files.append(fname)
                yield self.evt_info(
                    f"CompletionCritic: {fname} never reached a complete "
                    "state — flagged for mission control."
                )

        bb.data["generated_files"] = dict(generated)
        bb.resolve_feedback(self.stage)

        arts = sorted(generated)
        if failed_files:
            bb.add_feedback(
                source_stage=self.stage, target_stage=self.stage,
                severity="blocker",
                summary=(
                    f"Generation incomplete for: {', '.join(failed_files)}"
                ),
                details=(
                    "These files never passed the completeness gate after "
                    f"{_MAX_GEN_ATTEMPTS} attempts."
                ),
            )
            status = "partial"
            summary = (
                f"Generated {len(generated)}/{len(ordered)} file(s); "
                f"failed: {', '.join(failed_files)}."
            )
        else:
            status = "success"
            summary = (
                f"Generated {len(generated)} file(s) "
                "(headers first, then sources)."
            )
        bb.record_stage(self.stage, status, summary, artifacts=arts)
        report_findings = [
            f"Files generated: {len(generated)}.",
            f"Failed completeness gate: {len(failed_files)} "
            f"({', '.join(failed_files) if failed_files else 'none'}).",
            "",
            "### Generated file inventory",
            *[
                f"- `{name}` ({len(code):,} chars)"
                for name, code in sorted(generated.items())
            ],
        ]
        yield from self.emit_consolidated_report(
            ctx, bb, report_findings,
            status=status, summary=summary, artifacts=arts,
        )
        yield self.evt_stage_complete()

    # ------------------------------------------------------------------
    def _base_sections(
        self, ctx: PipelineContext, original: str, change_spec: str,
        target_summary: str, repo_knowledge: str, gitnexus_report: str,
        all_code_ctx: str, impact_map: str, feedback: str,
        generated: dict[str, str],
    ) -> list[tuple[str, str, int]]:
        a = ctx.app
        file_repo_ctx = ""
        if ctx.has_repo:
            file_repo_ctx = a._build_file_repo_context(
                ctx.repo_dir, original, ctx.uploaded_names,
                max_chars=a.MAX_REPO_CONTEXT_CHARS,
            )
            if not file_repo_ctx:
                file_repo_ctx = a._build_repo_context(
                    ctx.repo_dir, exclude_names=ctx.uploaded_names,
                )
        sections: list[tuple[str, str, int]] = [
            ("change_spec",
             f"## Change Specification (Source ICD -> Target ICD)\n\n"
             f"{change_spec}", 1),
        ]
        if file_repo_ctx:
            sections.append((
                "repo_dependencies",
                "## Repository Dependency Context\n"
                "These are the actual headers and modules this file "
                "depends on. Use the exact type names, function "
                "signatures, macros, and naming conventions from these "
                f"files.\n\n{file_repo_ctx}", 1,
            ))
        # NEW headers already generated this run: the .c implementation
        # agents must code against these, not the old interfaces.
        gen_headers = {
            n: c for n, c in generated.items() if n.lower().endswith(".h")
        }
        if gen_headers:
            hdr_ctx = "".join(
                f"\n### {n}\n```c\n{c}\n```\n" for n, c in
                sorted(gen_headers.items())
            )
            sections.append((
                "new_headers",
                "## Newly Generated Headers (POST-CHANGE interfaces)\n"
                "These header files were just regenerated for the Target "
                "ICD. Implementation files MUST be consistent with these "
                f"interfaces.\n{hdr_ctx}", 1,
            ))
        if feedback:
            sections.append(("feedback", feedback, 1))
        if impact_map:
            sections.append((
                "impact_map",
                f"## Impact Map (from the Codebase Understanding Team)\n"
                f"{impact_map}", 2,
            ))
        if gitnexus_report:
            sections.append((
                "gitnexus",
                "## GitNexus Codebase Understanding\n"
                "Embedded-systems-specific relationships extracted from "
                "the uploaded repository. Honor these relationships when "
                "modifying ISRs, drivers, shared globals, comm-stack "
                "callers, state-machine dispatch tables and "
                f"safety-critical paths.\n\n{gitnexus_report}", 2,
            ))
        if repo_knowledge:
            sections.append((
                "repo_knowledge",
                "## Repository Codebase Knowledge\n"
                "Detailed inventory of types, functions, variables, and "
                "macros from the repository. Use these as ground truth "
                f"for naming, types, and conventions.\n\n{repo_knowledge}",
                2,
            ))
        if target_summary:
            sections.append((
                "target_summary",
                f"## Target ICD Consolidated Summary\n\n{target_summary}", 3,
            ))
        if all_code_ctx:
            sections.append((
                "cross_file_ctx",
                f"## All Project Files (cross-file context)\n{all_code_ctx}",
                4,
            ))
        return sections

    # ------------------------------------------------------------------
    def _emit_variant_headers(
        self, ctx: PipelineContext, bb: Blackboard, fname: str,
        original: str, variant_names: list[str],
        base_sections: list[tuple[str, str, int]], transform_system: str,
        generated: dict[str, str],
    ) -> Iterator[dict]:
        """Generate one self-contained .h per detected peripheral variant."""
        a = ctx.app
        base_stem = fname[:-2]
        yield self.evt_info(
            f"VariantScout: {fname} has {len(variant_names)} variants "
            f"({', '.join(variant_names)}) — InterfaceArchitect will emit "
            "one header file per variant.",
            file=fname,
        )
        variant_files: dict[str, str] = {}
        for vname in variant_names:
            slug = a._variant_slug(vname)
            if not slug:
                continue
            hfn = f"{base_stem}_{slug}.h"
            yield self.evt_info(
                f"InterfaceArchitect: generating {hfn} for variant "
                f"'{vname}'…", file=hfn,
            )
            vcode = a._generate_variant_header(
                base_sections=base_sections,
                transform_system=transform_system,
                base_header_name=fname,
                original=original,
                variant_name=vname,
                all_variants=variant_names,
                header_filename=hfn,
            )
            if vcode and a._looks_complete_header(vcode):
                variant_files[hfn] = vcode
            else:
                yield self.evt_info(
                    f"CompletionCritic: variant '{vname}' header looked "
                    "incomplete; it will be skipped.", file=hfn,
                )
        if len(variant_files) >= 2:
            for hfn, vcode in sorted(variant_files.items()):
                (ctx.gen_dir / hfn).write_text(vcode)
                generated[hfn] = vcode
                yield {
                    "type": "file_complete", "file": hfn, "size": len(vcode),
                }
            yield self.evt_info(
                f"{fname}: generated {len(variant_files)} separate "
                f"per-variant header files "
                f"({', '.join(sorted(variant_files))}).", file=fname,
            )
            return True
        yield self.evt_info(
            f"{fname}: variant plan did not yield two complete headers — "
            "falling back to single-file generation.", file=fname,
        )
        return False

    # ------------------------------------------------------------------
    def _generate_single(
        self, ctx: PipelineContext, fname: str, original: str,
        base_sections: list[tuple[str, str, int]], transform_system: str,
    ) -> Iterator[dict]:
        a = ctx.app
        is_header = fname.lower().endswith(".h")
        agent = self.interface_architect if is_header else self.impl_engineer
        sec_file = (
            f"## File to Transform: {fname}\n\n```c\n{original}\n```\n\n"
            "Output the COMPLETE transformed file conforming to the Target "
            "ICD change specification. Output ONLY the code in ```c fences; "
            "do not summarise or truncate."
        )
        sections = list(base_sections) + [("file_to_transform", sec_file, 0)]
        system_tokens = a._estimate_tokens(transform_system)
        prompt = a._assemble_prompt(
            sections, max_input_tokens=a.MAX_INPUT_TOKENS - system_tokens,
        )
        clean = ""
        for attempt in range(1, _MAX_GEN_ATTEMPTS + 1):
            attempt_prompt = prompt
            if attempt > 1 and clean:
                yield self.evt_info(
                    f"CompletionCritic: {fname} incomplete after attempt "
                    f"{attempt - 1} — {agent.name} continuing from the "
                    "cut-off point.", file=fname,
                )
                attempt_prompt = (
                    f"{prompt}\n\n"
                    "The previous output was incomplete/truncated. Continue "
                    "from the exact point where it stopped, output ONLY the "
                    "missing remainder, and ensure all braces/comments are "
                    "closed"
                    + (" and the file ends with #endif" if is_header else "")
                    + ".\n\nPrevious partial output:\n```c\n"
                    f"{clean}\n```"
                )
            try:
                # The full embedded/RTOS toolchain contract is the system
                # prompt; the agent's role is reflected in events/logs.
                out = ctx.llm.complete(
                    transform_system, attempt_prompt,
                    max_tokens=4096, max_passes=4,
                )
            except Exception as e:  # noqa: BLE001
                yield self.evt_info(
                    f"{agent.name}: generation call for {fname} failed: {e}",
                    file=fname,
                )
                break
            chunk = a._extract_fenced(out, "c").strip()
            if not chunk:
                continue
            clean = chunk if attempt == 1 else (
                clean.rstrip() + "\n" + chunk.lstrip()
            ).strip()
            if a._looks_complete_c_file(clean, original, fname):
                return clean
        # Final acceptance check — partial output is never written.
        if clean and a._looks_complete_c_file(clean, original, fname):
            return clean
        return ""
