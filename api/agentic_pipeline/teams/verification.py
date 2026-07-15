"""Stage 4 — VerificationTeam (SSE stage: ``verification``).

Multi-agent system auditing the generated files against the change
specification.

Roster
    StructuralAuditor  — deterministic: brace/guard/include/function checks
                         (same ``_structural_verify`` gate as the classic
                         pipeline).
    ComplianceReviewer — LLM: checks each generated file against the change
                         specification and classifies remaining issues.
    RepairEngineer     — LLM: applies local fixes when the issue is confined
                         to the file; otherwise raises targeted feedback so
                         the MissionController can send the mission back to
                         generation (or even analysis).

Artifacts (identical to the classic pipeline)
    generated_code/verification_report.txt
"""

from __future__ import annotations

from typing import Iterator

from ..agents import Agent
from ..blackboard import Blackboard
from ..context import PipelineContext
from .base import Team

_REVIEW_FILE_CAP = 8


class VerificationTeam(Team):
    stage = "verification"
    title = "Verification Team"

    def __init__(self):
        super().__init__()
        self.structural_auditor = Agent(
            name="StructuralAuditor",
            role="deterministic structural checks",
        )
        self.compliance_reviewer = Agent(
            name="ComplianceReviewer",
            role="spec-compliance review",
            system_prompt=(
                "You are a meticulous embedded C code reviewer. You check "
                "a generated file against an ICD change specification and "
                "report ONLY genuine problems: missing/incorrect fields, "
                "types, constants, scale factors, message IDs, or dropped "
                "peripheral variations. Generated per-variant headers are "
                "INTENTIONALLY scoped to one variant — do not report other "
                "variants as missing from them. Respond with STRICT JSON: "
                '{"compliant": true|false, "issues": ["..."], '
                '"needs_spec_change": false, "reason": "..."} — '
                '"needs_spec_change" is true ONLY when the change '
                "specification itself is ambiguous or contradictory."
            ),
        )
        self.repair_engineer = Agent(
            name="RepairEngineer",
            role="local repair of verified issues",
            system_prompt=(
                "You are an expert embedded C programmer. You repair a "
                "generated C file so it satisfies the listed review issues "
                "while keeping every other aspect unchanged. Embedded "
                "constraints: GCC 7.3.1, -std=gnu99, newlib, freestanding "
                "(no POSIX). Preserve ALL per-variation structs/sections — "
                "never merge or drop variations. Output ONLY the complete "
                "corrected file in ```c fences."
            ),
        )
        self.agents = [
            self.structural_auditor, self.compliance_reviewer,
            self.repair_engineer,
        ]

    # ------------------------------------------------------------------
    def run(self, ctx: PipelineContext, bb: Blackboard) -> Iterator[dict]:
        a = ctx.app
        yield self.evt_stage("Verifying generated files against the spec…")
        yield self.evt_roster()

        change_spec = str(bb.data.get("change_spec", ""))
        gen_files = sorted(
            p for p in ctx.gen_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".c", ".h")
        ) if ctx.gen_dir.exists() else []

        if not gen_files:
            bb.record_stage(
                self.stage, "failed", "No generated files to verify.",
            )
            bb.add_feedback(
                source_stage=self.stage, target_stage="transform",
                severity="blocker",
                summary="Verification found no generated .c/.h files.",
            )
            yield self.evt_info("No generated files found to verify.")
            yield self.evt_stage_complete()
            return

        report: list[str] = ["# Verification Report (agentic pipeline)\n"]
        unresolved: list[str] = []
        spec_doubts: list[str] = []
        reviewed = 0

        for gf in gen_files:
            fname = gf.name
            generated = gf.read_text()
            original = self._original_for(ctx, fname)

            # ---- StructuralAuditor --------------------------------------
            issues = a._structural_verify(
                generated, ctx.repo_dir if ctx.has_repo else None,
                original, fname,
            )
            report.append(f"\n## {fname}")
            if issues:
                yield self.evt_info(
                    f"StructuralAuditor: {fname} has "
                    f"{len(issues)} structural issue(s)."
                )
                report.append("Structural issues:")
                report.extend(f"  - {i}" for i in issues)
            else:
                report.append("Structural checks: PASS")

            # ---- ComplianceReviewer (bounded) ----------------------------
            review_issues: list[str] = []
            should_review = bool(change_spec) and (
                bool(issues) or reviewed < _REVIEW_FILE_CAP
            )
            if should_review:
                reviewed += 1
                verdict = self._review(ctx, fname, generated, change_spec)
                review_issues = [str(x) for x in verdict.get("issues", [])]
                if verdict.get("needs_spec_change"):
                    spec_doubts.append(
                        f"{fname}: {verdict.get('reason', 'spec ambiguity')}"
                    )
                if review_issues:
                    yield self.evt_info(
                        f"ComplianceReviewer: {fname} — "
                        f"{len(review_issues)} spec-compliance issue(s)."
                    )
                    report.append("Spec-compliance issues:")
                    report.extend(f"  - {i}" for i in review_issues)
                elif verdict:
                    report.append("Spec-compliance review: PASS")

            # ---- RepairEngineer ------------------------------------------
            all_issues = issues + review_issues
            if all_issues:
                fixed = self._repair(
                    ctx, fname, generated, original, all_issues, change_spec,
                )
                if fixed:
                    residual = a._structural_verify(
                        fixed, ctx.repo_dir if ctx.has_repo else None,
                        original, fname,
                    )
                    if len(residual) <= len(issues):
                        gf.write_text(fixed)
                        gen_map = bb.data.get("generated_files")
                        if isinstance(gen_map, dict):
                            gen_map[fname] = fixed
                        report.append(
                            f"RepairEngineer: applied local fix "
                            f"({len(all_issues)} issue(s) addressed, "
                            f"{len(residual)} structural remaining)."
                        )
                        yield self.evt_info(
                            f"RepairEngineer: {fname} repaired in place."
                        )
                        if residual:
                            unresolved.append(
                                f"{fname}: {'; '.join(residual[:4])}"
                            )
                        continue
                unresolved.append(f"{fname}: {'; '.join(all_issues[:4])}")
                report.append(
                    "RepairEngineer: local repair not applied — escalating "
                    "to mission control."
                )

        (ctx.gen_dir / "verification_report.txt").write_text(
            "\n".join(report) + "\n"
        )

        # ---- Feedback routing (the non-sequential part) -------------------
        for doubt in spec_doubts:
            bb.add_feedback(
                source_stage=self.stage, target_stage="analysis",
                severity="warning",
                summary=f"Change specification ambiguity suspected: {doubt}",
            )
        if unresolved:
            bb.add_feedback(
                source_stage=self.stage, target_stage="transform",
                severity="blocker",
                summary=(
                    f"{len(unresolved)} file(s) failed verification and "
                    "could not be locally repaired."
                ),
                details="\n".join(unresolved)[:4000],
            )

        bb.resolve_feedback(self.stage)
        status = "success" if not unresolved else "partial"
        bb.record_stage(
            self.stage, status,
            f"{len(gen_files)} file(s) verified; "
            f"{len(unresolved)} unresolved; "
            f"{len(spec_doubts)} spec doubt(s).",
            artifacts=["verification_report.txt"],
        )
        yield self.evt_stage_complete()

    # ------------------------------------------------------------------
    def _original_for(self, ctx: PipelineContext, fname: str) -> str:
        exact = ctx.code_dir / fname
        if exact.exists():
            return exact.read_text()
        # Per-variant headers (<stem>_<Variant>.h) verify against the base
        # header they were derived from.
        if fname.lower().endswith(".h") and "_" in fname:
            base = fname[:fname.rindex("_")] + ".h"
            cand = ctx.code_dir / base
            if cand.exists():
                return cand.read_text()
        return ""

    def _review(
        self, ctx: PipelineContext, fname: str, generated: str,
        change_spec: str,
    ) -> dict:
        a = ctx.app
        prompt = (
            f"## Change specification\n{change_spec[:10_000]}\n\n"
            f"## Generated file: {fname}\n```c\n{generated[:14_000]}\n```\n\n"
            "Review the generated file against the specification and "
            "respond with the STRICT JSON verdict."
        )
        try:
            out = self.compliance_reviewer.act(
                ctx.llm, prompt, max_tokens=1024, max_passes=2,
            )
        except Exception:  # noqa: BLE001
            return {}
        return a._extract_json_object(out) or {}

    def _repair(
        self, ctx: PipelineContext, fname: str, generated: str,
        original: str, issues: list[str], change_spec: str,
    ) -> str:
        a = ctx.app
        prompt = (
            f"## Review issues to fix\n"
            + "\n".join(f"- {i}" for i in issues[:12])
            + f"\n\n## Change specification (context)\n"
            f"{change_spec[:6_000]}\n\n"
            f"## Current generated file: {fname}\n```c\n{generated}\n```\n\n"
            "Output the complete corrected file."
        )
        try:
            out = self.repair_engineer.act(
                ctx.llm, prompt, max_tokens=4096, max_passes=4,
            )
        except Exception:  # noqa: BLE001
            return ""
        fixed = a._extract_fenced(out, "c").strip()
        if fixed and a._looks_complete_c_file(
            fixed, original or generated, fname,
        ):
            return fixed
        return ""
