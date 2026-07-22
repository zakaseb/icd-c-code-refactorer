"""Verify finite sandbox_retries actually terminates the orchestrator."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "api"))

import orchestrator as orch  # noqa: E402

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}  {detail}")


class ScriptedLLM:
    """Emit a fixed sequence of agent actions, then repeat the last one."""

    def __init__(self, actions: list[str]):
        self.actions = list(actions)
        self.i = 0

    def __call__(self, system_prompt, user_prompt, max_tokens=1024, meta=None):
        if meta is not None:
            meta["finish_reason"] = "stop"
        idx = min(self.i, len(self.actions) - 1)
        self.i += 1
        text = self.actions[idx]
        yield text


def _failing_runner():
    return False, "error: undefined reference to `foo'\n"


def run_with_budget(max_builds: int | None, actions: list[str]):
    tmp = Path(tempfile.mkdtemp())
    events = list(orch.run_orchestrator(
        sandbox_dir=tmp,
        build_info={"type": "make", "dir": tmp},
        sandbox_cc="gcc",
        is_cross=False,
        gen_files={},
        change_spec="change sync constant",
        repo_knowledge="",
        file_index={},
        snapshots={},
        build_runner=_failing_runner,
        llm_stream=ScriptedLLM(actions),
        max_steps=None,          # unlimited steps — the bug case
        max_builds=max_builds,
    ))
    return events


print("\n=== Finite budget must hard-stop (no endless patching) ===")
# Initial build fails (call 1). Then agent patches + builds three more times
# (calls 2,3,4). After the 4th failed build the loop must terminate even
# though max_steps is unlimited and the scripted LLM would keep going.
actions = [
    '<think>patch</think>\n<action>{"tool":"note","args":{"text":"x"}}</action>',
    '<think>rebuild</think>\n<action>{"tool":"build","args":{}}</action>',
] * 20  # plenty of fuel if the bug still lets it run

events = run_with_budget(4, actions)
dones = [e for e in events if e.get("type") == "done"]
builds = [e for e in events if e.get("type") == "build"]
steps = [e for e in events if e.get("type") == "step"]

check("exactly one done event", len(dones) == 1, str(dones))
check("done is failure", dones and dones[0].get("success") is False)
check("done reason mentions budget", dones and "budget exhausted" in dones[0].get("reason", ""))
check("build calls reported == 4", dones and dones[0].get("builds") == 4,
      str(dones[0] if dones else None))
check("no more than 4 build events", len(builds) <= 4, f"got {len(builds)}")
check("did not run hundreds of steps", len(steps) < 30, f"got {len(steps)}")

print("\n=== Budget=1 stops after initial failed build ===")
events1 = run_with_budget(1, actions)
dones1 = [e for e in events1 if e.get("type") == "done"]
steps1 = [e for e in events1 if e.get("type") == "step"]
check("budget=1 yields done failure", dones1 and dones1[0].get("success") is False)
check("budget=1 builds==1", dones1 and dones1[0].get("builds") == 1)
check("budget=1 enters no agent steps", len(steps1) == 0, f"got {len(steps1)}")

print("\n=== Indefinite (max_builds=None) still allows many builds ===")
# With unlimited builds, scripted build actions should keep going until we
# artificially stop via max_steps.
events_inf = list(orch.run_orchestrator(
    sandbox_dir=Path(tempfile.mkdtemp()),
    build_info={"type": "make", "dir": Path(".")},
    sandbox_cc="gcc",
    is_cross=False,
    gen_files={},
    change_spec="",
    repo_knowledge="",
    file_index={},
    snapshots={},
    build_runner=_failing_runner,
    llm_stream=ScriptedLLM([
        '<think>b</think>\n<action>{"tool":"build","args":{}}</action>',
    ] * 10),
    max_steps=5,
    max_builds=None,
))
dones_inf = [e for e in events_inf if e.get("type") == "done"]
builds_inf = [e for e in events_inf if e.get("type") == "build"]
check("indefinite eventually ends via step budget",
      dones_inf and "step budget" in dones_inf[0].get("reason", ""))
check("indefinite performed >1 build before step cap",
      len(builds_inf) > 1, f"got {len(builds_inf)}")

print("\n" + "=" * 60)
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
if FAIL:
    sys.exit(1)
print("All tests passed!")
