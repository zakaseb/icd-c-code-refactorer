"""Stage 1 — IngestionTeam (SSE stage: ``analysis``).

Multi-agent system for input ingestion and ICD delta analysis.

Roster
    TargetSummarizer  — condenses the Target ICD into a consolidated summary
                        (chunked map-reduce).
    DocumentAnalyst   — mines each Target ICD chunk for deltas against the
                        Source ICD, grounded in the uploaded C sources.
    SpecSynthesizer   — merges the per-chunk findings into ONE comprehensive
                        change specification (``change_spec_raw``), preserving
                        per-peripheral-variation structure.
    Distiller         — rewrites the raw spec into a shorter, fact-complete
                        ``change_spec``.
    FactAuditor       — deterministic critic: verifies the distillation kept
                        >= 80% of the technical fact tokens; forces one
                        focused retry otherwise (same guarantee as the
                        classic pipeline).

Artifacts (identical to the classic pipeline)
    change_spec.txt, change_spec_raw.txt, target_summary.txt,
    icd_analysis.txt (session + generated_code copies).
"""

from __future__ import annotations

from typing import Iterator

from ..agents import Agent
from ..blackboard import Blackboard
from ..context import PipelineContext
from .base import Team

_MAX_ANALYSIS_CHUNKS = 40
_MAX_SUMMARY_CHUNKS = 16


class IngestionTeam(Team):
    stage = "analysis"
    title = "Ingestion & Analysis Team"

    def __init__(self):
        super().__init__()
        self.summarizer = Agent(
            name="TargetSummarizer",
            role="Target ICD consolidation",
            system_prompt=(
                "You are a senior embedded-systems documentation analyst. "
                "You summarise Interface Control Document (ICD) sections "
                "into dense, technical bullet points. Preserve every "
                "concrete value: field names, types, bit widths, offsets, "
                "units, scale factors, constants, message IDs, sample "
                "rates and timing. Never editorialise."
            ),
        )
        self.analyst = Agent(
            name="DocumentAnalyst",
            role="ICD delta mining",
            system_prompt=(
                "You are an expert embedded C engineer comparing two "
                "Interface Control Documents. You identify EVERY difference "
                "that affects C code: changed/added/removed fields, structs, "
                "enums, constants, macros, message IDs, scale factors, "
                "sample rates, timing and protocol behaviour. Report "
                "OLD -> NEW mappings with concrete values and cite which "
                "uploaded .c/.h files are impacted. If a section shows no "
                "relevant delta, say 'NO DELTA'."
            ),
        )
        self.synthesizer = Agent(
            name="SpecSynthesizer",
            role="change-spec synthesis",
            system_prompt=(
                "You are an expert embedded C refactoring analyst. You "
                "merge per-section delta findings into ONE well-organised, "
                "comprehensive change specification that an engineer can "
                "refactor C code from without consulting the original "
                "documents. Merge duplicates, keep every technical fact, "
                "and organise per peripheral variation when several "
                "variations of the same peripheral exist."
            ),
        )
        self.distiller = Agent(
            name="Distiller",
            role="lossless spec compression",
            system_prompt=(
                "You are an expert embedded C refactoring analyst. You "
                "produce a SHORTER, more compact rewrite of ICD delta "
                "specifications. Your output is a distillation — it must "
                "be different from the input, removing redundancy and "
                "verbose prose, while preserving EVERY technical fact "
                "verbatim (hex constants, numeric literals, type names, "
                "field names, identifiers, units, scale factors). An "
                "engineer must be able to refactor C code from the "
                "distilled spec alone."
            ),
        )
        self.fact_auditor = Agent(
            name="FactAuditor",
            role="deterministic fact-preservation critic",
        )
        self.agents = [
            self.summarizer, self.analyst, self.synthesizer,
            self.distiller, self.fact_auditor,
        ]

    # ------------------------------------------------------------------
    def run(self, ctx: PipelineContext, bb: Blackboard) -> Iterator[dict]:
        a = ctx.app
        yield self.evt_stage("Ingesting inputs and analysing ICD delta…")
        yield self.evt_roster()

        feedback = self.feedback_section(bb)
        if feedback:
            yield self.evt_info(
                "Re-running analysis with downstream feedback injected "
                f"({len(bb.pending_feedback(self.stage))} item(s))."
            )

        code_ctx = a._build_source_scripts_context(ctx.code_dir)
        bb.data["code_source_scripts"] = code_ctx

        # ---- TargetSummarizer: consolidated Target ICD summary ---------
        yield self.evt_info("TargetSummarizer: consolidating the Target ICD…")
        target_summary = yield from self._summarize_target(ctx)
        bb.data["target_summary"] = target_summary
        (ctx.session_dir / "target_summary.txt").write_text(target_summary)

        # ---- DocumentAnalyst: chunked delta mining ----------------------
        yield self.evt_info("DocumentAnalyst: mining per-section ICD deltas…")
        findings = yield from self._mine_deltas(ctx, code_ctx, feedback)

        # ---- SpecSynthesizer: raw comprehensive spec --------------------
        yield self.evt_info(
            f"SpecSynthesizer: merging {len(findings)} finding block(s) "
            "into the comprehensive change specification…"
        )
        raw_change_spec = self._synthesize(ctx, findings, code_ctx, feedback)
        if not raw_change_spec:
            bb.record_stage(
                self.stage, "failed",
                "Analysis produced no change specification.",
            )
            bb.add_feedback(
                source_stage=self.stage, target_stage=self.stage,
                severity="blocker",
                summary=(
                    "ICD analysis produced no change specification — "
                    "the analysis needs to be re-run."
                ),
            )
            yield self.evt_info(
                "Analysis produced no change specification — mission "
                "control will decide how to proceed."
            )
            return

        # ---- Distiller + FactAuditor ------------------------------------
        yield self.evt_info(
            "Distiller: compressing the specification "
            "(FactAuditor gating fact preservation)…"
        )
        change_spec = self._distill(ctx, raw_change_spec)
        if change_spec != raw_change_spec:
            yield self.evt_info(
                f"Distilled change_spec: {len(raw_change_spec):,} -> "
                f"{len(change_spec):,} chars (fact check passed)."
            )
        else:
            yield self.evt_info(
                "Distillation did not produce a usable shorter version — "
                "change_spec mirrors the comprehensive raw spec."
            )

        bb.data["raw_change_spec"] = raw_change_spec
        bb.data["change_spec"] = change_spec

        artifacts = [
            "change_spec.txt", "change_spec_raw.txt",
            "target_summary.txt", "icd_analysis.txt",
        ]
        (ctx.session_dir / "change_spec.txt").write_text(change_spec)
        (ctx.session_dir / "change_spec_raw.txt").write_text(raw_change_spec)
        # icd_analysis.txt is enriched with repo knowledge by CodebaseTeam;
        # write the spec-only version now so the artifact always exists.
        (ctx.session_dir / "icd_analysis.txt").write_text(change_spec)
        ctx.gen_dir.mkdir(parents=True, exist_ok=True)
        (ctx.gen_dir / "icd_analysis.txt").write_text(change_spec)

        bb.resolve_feedback(self.stage)
        bb.record_stage(
            self.stage, "success",
            f"change_spec {len(change_spec):,} chars "
            f"(raw {len(raw_change_spec):,}), target_summary "
            f"{len(target_summary):,} chars.",
            artifacts=artifacts,
        )
        yield self.evt_stage_complete()

    # ------------------------------------------------------------------
    def _summarize_target(self, ctx: PipelineContext) -> Iterator[dict]:
        a = ctx.app
        chunks = a._split_text_chunks(
            ctx.target_icd, a.ICD_CHUNK_CHARS * 3,
        )[:_MAX_SUMMARY_CHUNKS]
        partials: list[str] = []
        for i, chunk in enumerate(chunks):
            out = self.summarizer.act(
                ctx.llm,
                "Summarise this Target ICD section into dense technical "
                "bullet points (preserve all concrete values):\n\n"
                f"{chunk}",
                max_tokens=1024, max_passes=2,
            ).strip()
            if out:
                partials.append(out)
            if len(chunks) > 4 and (i + 1) % 4 == 0:
                yield self.evt_info(
                    f"TargetSummarizer: {i + 1}/{len(chunks)} sections done."
                )
        merged = "\n\n".join(partials).strip()
        if len(merged) > 14_000:
            merged = self.summarizer.act(
                ctx.llm,
                "Merge these per-section Target ICD summaries into one "
                "consolidated summary. De-duplicate, keep every concrete "
                "value:\n\n" + merged[:36_000],
                max_tokens=3072, max_passes=3,
            ).strip() or merged
        return merged

    def _mine_deltas(
        self, ctx: PipelineContext, code_ctx: str, feedback: str,
    ) -> Iterator[dict]:
        a = ctx.app
        chunks = a._split_text_chunks(
            ctx.target_icd, a.ICD_CHUNK_CHARS,
        )[:_MAX_ANALYSIS_CHUNKS]
        source_excerpt = a._truncate_text(
            ctx.source_icd, 9_000, "source_icd",
        )
        code_excerpt = a._truncate_text(code_ctx, 8_000, "code_ctx")
        findings: list[str] = []
        for i, chunk in enumerate(chunks):
            prompt = (
                f"## Source ICD (pre-change, may be truncated)\n"
                f"{source_excerpt}\n\n"
                f"## Target ICD — section {i + 1}/{len(chunks)}\n{chunk}\n\n"
                f"## Uploaded C sources (pre-change ground truth)\n"
                f"{code_excerpt}\n\n"
            )
            if feedback:
                prompt += f"{feedback}\n\n"
            prompt += (
                "List every delta in THIS Target ICD section relative to "
                "the Source ICD that impacts the C code. Give OLD -> NEW "
                "mappings with concrete values and note impacted "
                "files/symbols. If none, reply exactly 'NO DELTA'."
            )
            out = self.analyst.act(
                ctx.llm, prompt, max_tokens=2048, max_passes=2,
            ).strip()
            if out and "NO DELTA" not in out[:40].upper():
                findings.append(f"### Findings — section {i + 1}\n{out}")
            if len(chunks) > 5 and (i + 1) % 5 == 0:
                yield self.evt_info(
                    f"DocumentAnalyst: {i + 1}/{len(chunks)} sections mined "
                    f"({len(findings)} with deltas)."
                )
        return findings

    def _synthesize(
        self, ctx: PipelineContext, findings: list[str],
        code_ctx: str, feedback: str,
    ) -> str:
        a = ctx.app
        joined = a._truncate_text(
            "\n\n".join(findings), 40_000, "findings",
        ) if findings else "(no per-section findings were produced)"
        prompt = (
            "Merge the per-section delta findings below into ONE "
            "comprehensive change specification (Source ICD -> Target "
            "ICD). Requirements:\n"
            "- Keep EVERY technical fact and OLD -> NEW mapping.\n"
            "- Organise by peripheral and, when the Target ICD defines "
            "multiple variations of the same peripheral, give each "
            "variation its own clearly-labelled subsection plus a SHARED "
            "section.\n"
            "- Include an 'Impact on C code' part citing the uploaded "
            "files/symbols each change touches."
            f"{a.PERIPHERAL_VARIATION_ANALYSIS_GUIDANCE}\n\n"
        )
        if feedback:
            prompt += f"{feedback}\n\n"
        prompt += (
            f"## Per-section findings\n{joined}\n\n"
            f"## Uploaded C sources (excerpt)\n"
            f"{a._truncate_text(code_ctx, 6_000, 'code_ctx')}"
        )
        try:
            return self.synthesizer.act(
                ctx.llm, prompt, max_tokens=4096, max_passes=6,
            ).strip()
        except Exception:  # noqa: BLE001
            return ""

    def _distill(self, ctx: PipelineContext, raw: str) -> str:
        """Distill with the same content-based guarantee as the classic path."""
        a = ctx.app
        base_prompt = (
            "Produce a DISTILLED version of the following ICD change "
            "specification. It MUST be shorter than the input (aim for "
            "40-60% of the original length), MUST NOT echo the input "
            "verbatim, and MUST preserve every technical fact verbatim "
            "(hex constants, numeric literals, types, field names, "
            "identifiers, units, scale factors, OLD -> NEW mappings, "
            "file/symbol citations, and the per-variation structure). "
            "Remove only duplication and filler prose.\n\n"
            f"## Original change specification\n{raw}"
        )
        try:
            distilled = self.distiller.act(
                ctx.llm, base_prompt, max_tokens=4096, max_passes=6,
            ).strip()
        except Exception:  # noqa: BLE001
            return raw
        ok, missing, kept = a._distillation_fact_check(raw, distilled)
        if not distilled or distilled == raw or not ok:
            reason = (
                "previous attempt produced no output" if not distilled
                else "previous attempt returned the input verbatim; you "
                     "MUST condense it" if distilled == raw
                else "previous attempt dropped required technical tokens — "
                     "re-include them verbatim: " + ", ".join(missing[:60])
            )
            try:
                retry = self.distiller.act(
                    ctx.llm,
                    f"{base_prompt}\n\n## Retry guidance\n{reason}.\n"
                    "Produce a NEW, shorter distillation that satisfies "
                    "every rule above.",
                    max_tokens=4096, max_passes=6,
                ).strip()
            except Exception:  # noqa: BLE001
                retry = ""
            ok2, _, kept2 = a._distillation_fact_check(raw, retry)
            if retry and retry != raw:
                first_unusable = not distilled or distilled == raw
                if first_unusable or kept2 >= kept:
                    distilled = retry
        if distilled and distilled != raw:
            return distilled
        return raw
