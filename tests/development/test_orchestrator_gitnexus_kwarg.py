"""Regression: sandbox callers pass gitnexus_report into orchestrator/agentic."""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "api"))

import agentic_debug as adev  # noqa: E402
import orchestrator as orch  # noqa: E402


def test_signatures_accept_gitnexus_report():
    assert "gitnexus_report" in inspect.signature(orch.run_orchestrator).parameters
    assert "gitnexus_report" in inspect.signature(orch.build_brief).parameters
    assert "gitnexus_report" in inspect.signature(adev.run_agentic_debug).parameters
    assert "gitnexus_report" in inspect.signature(adev._planner_prompt).parameters
    assert "gitnexus_report" in inspect.signature(adev._patcher_prompt).parameters


def test_run_orchestrator_accepts_gitnexus_kwarg_without_typeerror():
    """Reproduces the indefinite-retry crash from missing gitnexus_report kwarg."""
    tmp = Path(tempfile.mkdtemp())
    events = list(
        orch.run_orchestrator(
            sandbox_dir=tmp,
            build_info={"type": "make", "dir": tmp},
            sandbox_cc="gcc",
            is_cross=False,
            gen_files={},
            change_spec="change sync constant",
            repo_knowledge="",
            gitnexus_report="## ISR wiring\nfoo_isr -> TaskFoo",
            file_index={},
            snapshots={},
            build_runner=lambda: (True, "ok"),
            llm_stream=lambda *a, **k: iter(()),
            max_steps=1,
            max_builds=1,
        )
    )
    assert any(e.get("type") == "done" and e.get("success") for e in events)


def test_build_brief_includes_gitnexus_section():
    tmp = Path(tempfile.mkdtemp())
    brief = orch.build_brief(
        sandbox_dir=tmp,
        build_info={"type": "make", "path": tmp},
        sandbox_cc="gcc",
        is_cross=False,
        gen_files={"src/a.c": "generated"},
        change_spec="spec",
        repo_knowledge="knowledge",
        initial_build_output="error: foo",
        notes=[],
        gitnexus_report="ISR -> task wiring detail",
    )
    assert "## GitNexus codebase understanding" in brief
    assert "ISR -> task wiring detail" in brief


if __name__ == "__main__":
    test_signatures_accept_gitnexus_report()
    test_run_orchestrator_accepts_gitnexus_kwarg_without_typeerror()
    test_build_brief_includes_gitnexus_section()
    print("ALL TESTS PASSED")
