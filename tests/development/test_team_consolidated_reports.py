"""Consolidated per-team findings reports are written and downloadable."""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock


def _make_ctx(tmp: Path):
    from api.agentic_pipeline.context import PipelineContext

    session = tmp / "session"
    gen = session / "generated_code"
    code = session / "original_code"
    gen.mkdir(parents=True)
    code.mkdir(parents=True)
    (code / "a.c").write_text("int main(void){return 0;}\n")
    return PipelineContext(
        session_dir=session,
        gen_dir=gen,
        code_dir=code,
        repo_dir=session / "repo_contents",
        has_repo=False,
        source_icd="source",
        target_icd="target",
        llm=MagicMock(name="fake-llm"),
        app=MagicMock(name="fake-app"),
    )


def test_team_writes_consolidated_report_to_gen_and_archive():
    from api.agentic_pipeline.blackboard import Blackboard
    from api.agentic_pipeline.mission import STAGE_ORDER
    from api.agentic_pipeline.teams.base import Team

    class DummyTeam(Team):
        stage = "analysis"
        title = "Dummy Analysis Team"

        def __init__(self):
            super().__init__()
            from api.agentic_pipeline.agents import Agent
            self.agents = [Agent(name="A", role="tester")]

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ctx = _make_ctx(tmp)
        bb = Blackboard(ctx.session_dir, STAGE_ORDER)
        bb.record_stage("analysis", "success", "ok summary", artifacts=["change_spec.txt"])
        team = DummyTeam()
        events = list(
            team.emit_consolidated_report(
                ctx, bb,
                ["Finding one.", "Finding two."],
                status="success",
                summary="ok summary",
                artifacts=["change_spec.txt"],
            )
        )
        live = ctx.gen_dir / "team_report_analysis.md"
        archive = (
            ctx.session_dir / "agentic_pipeline" / "team_reports"
            / "analysis_team_report.md"
        )
        assert live.exists() and archive.exists()
        body = live.read_text()
        assert "Dummy Analysis Team" in body
        assert "Finding one." in body
        assert "change_spec.txt" in body
        # Plain findings are bulleted; structured markdown is preserved.
        assert "- Finding one." in body
        events = list(
            team.emit_consolidated_report(
                ctx, bb,
                [
                    "Simple finding.",
                    "",
                    "### Section",
                    "```",
                    "line1\nline2",
                    "```",
                ],
                status="success",
                summary="ok summary",
                artifacts=["change_spec.txt"],
            )
        )
        body2 = live.read_text()
        assert "### Section" in body2
        assert "```\nline1\nline2\n```" in body2
        assert "- ```" not in body2
        types = [e.get("type") for e in events]
        assert "info" in types
        assert "deliverables_updated" in types
        upd = next(e for e in events if e["type"] == "deliverables_updated")
        assert "team_report_analysis.md" in upd["files"]
        assert upd["synced"] == ["team_report_analysis.md"]


def test_ensure_report_is_noop_when_present():
    from api.agentic_pipeline.blackboard import Blackboard
    from api.agentic_pipeline.mission import STAGE_ORDER
    from api.agentic_pipeline.teams.base import Team

    class DummyTeam(Team):
        stage = "compile"
        title = "Compile Team"

    with tempfile.TemporaryDirectory() as d:
        ctx = _make_ctx(Path(d))
        bb = Blackboard(ctx.session_dir, STAGE_ORDER)
        team = DummyTeam()
        (ctx.gen_dir / "team_report_compile.md").write_text("# already\n")
        events = list(team.ensure_consolidated_report(ctx, bb))
        assert events == []
        assert (ctx.gen_dir / "team_report_compile.md").read_text() == "# already\n"


def test_download_zip_includes_team_reports():
    """Mirrors /api/download packaging of agentic_pipeline/team_reports/."""
    with tempfile.TemporaryDirectory() as d:
        session = Path(d) / "session"
        gen = session / "generated_code"
        reports = session / "agentic_pipeline" / "team_reports"
        gen.mkdir(parents=True)
        reports.mkdir(parents=True)
        (gen / "team_report_analysis.md").write_text("# analysis\n")
        (gen / "a.c").write_text("int x;\n")
        (reports / "analysis_team_report.md").write_text("# analysis archive\n")
        (reports / "transform_team_report.md").write_text("# transform\n")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(gen.iterdir()):
                if f.is_file():
                    zf.write(f, f.name)
            for rf in sorted(reports.rglob("*")):
                if rf.is_file():
                    arc = "agentic_pipeline/team_reports/" + str(
                        rf.relative_to(reports)
                    )
                    zf.write(rf, arc)
        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            names = set(zf.namelist())
            assert "team_report_analysis.md" in names
            assert "a.c" in names
            assert "agentic_pipeline/team_reports/analysis_team_report.md" in names
            assert "agentic_pipeline/team_reports/transform_team_report.md" in names
            assert zf.read("team_report_analysis.md").decode().startswith("# analysis")


if __name__ == "__main__":
    test_team_writes_consolidated_report_to_gen_and_archive()
    test_ensure_report_is_noop_when_present()
    test_download_zip_includes_team_reports()
    print("ALL TESTS PASSED")
