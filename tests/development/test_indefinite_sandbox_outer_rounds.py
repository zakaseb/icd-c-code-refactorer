"""Indefinite sandbox retries must not spin unlimited outer reset rounds."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "api"))
sys.path.insert(0, str(REPO))


def test_indefinite_plan_is_single_unlimited_campaign():
    import app as appmod

    plan = appmod._sandbox_retry_plan(None)
    assert plan["mode"] == "indefinite"
    assert plan["orch_outer"] == 1
    assert plan["orch_builds"] is None
    assert plan["agentic_outer"] == 1
    assert plan["agentic_attempts"] is None


def test_indefinite_crash_loop_terminates_without_flooding():
    """Reproduce the UI-killing ∞ outer reset loop and prove it stops.

    Previously indefinite set orch_outer=None, so a TypeError on every
    orchestrator call reset forever (80k+ rounds) and the browser stage
    disappeared under SSE flood / OOM.
    """
    import app as appmod

    sess = Path(tempfile.mkdtemp())
    gen = sess / "generated_code"
    repo = sess / "repo"
    gen.mkdir()
    repo.mkdir()
    (gen / "a.c").write_text("int main(void){return 0;}\n")
    (repo / "a.c").write_text("int main(void){return 0;}\n")
    (repo / "Makefile").write_text("all:\n\t@false\n")

    def boom(*args, **kwargs):
        raise TypeError(
            "run_orchestrator() got an unexpected keyword argument 'gitnexus_report'"
        )

    events = []
    with patch.object(appmod, "_run_orchestrator", side_effect=boom), \
         patch.object(appmod, "SANDBOX_USE_AGENTIC", False), \
         patch.object(appmod, "SANDBOX_USE_ORCHESTRATOR", True), \
         patch.object(
             appmod, "_run_sandbox_build",
             return_value=(False, "error: boom\n"),
         ):
        for evt in appmod._sandbox_build_iterate(
            session_dir=sess,
            gen_dir=gen,
            repo_dir=repo,
            change_spec="spec",
            uploaded_names={"a.c"},
            has_repo=True,
            sandbox_max_retries=None,  # indefinite
        ):
            payload = appmod._parse_sse_event(evt)
            if payload:
                events.append(payload)

    start_msgs = [
        e.get("message", "") for e in events
        if e.get("type") == "info" and "Starting debugging orchestrator" in e.get("message", "")
    ]
    # Single outer campaign — must NOT spin thousands of reset rounds.
    assert len(start_msgs) <= 3, f"flooded outer rounds: {len(start_msgs)}"
    assert any(e.get("type") == "sandbox_build_result" for e in events)
    assert any("indefinite" in e.get("message", "").lower() for e in events if e.get("type") == "info")


if __name__ == "__main__":
    test_indefinite_plan_is_single_unlimited_campaign()
    test_indefinite_crash_loop_terminates_without_flooding()
    print("ALL TESTS PASSED")
