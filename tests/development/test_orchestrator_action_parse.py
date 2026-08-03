"""Orchestrator action parsing must accept common local-LLM formats."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "api"))

import orchestrator as orch  # noqa: E402


SAMPLES = {
    "canonical": '''
<think>look for wind.h</think>
<action>
{"tool": "search", "args": {"pattern": "wind\\\\.h", "max_hits": 20}}
</action>
''',
    "canonical_pretty": '''
<action>
{
  "tool": "read_file",
  "args": {
    "path": "src/IMU.h",
    "limit": 50,
    "offset": 100
  }
}
</action>
''',
    "tool_call": '''
The build failed because AircraftInterface.c includes wind.h.

<tool_call>
<function=search>
{"pattern": "wind\\\\.h", "path": "/home/developer/workspace/sessions/abc/sandbox/src", "max_hits": 20, "ext": "all"}
</function>
''',
    "tool_call_named_close": '''
<tool_call>

search
{"pattern": "wind\\\\.h", "path": "src", "max_hits": 5, "ext": "c"}
</search>
</tool_call>
''',
    "tool_call_function_eq": '''
<tool_call>
function=search
{"pattern": "wind\\\\.h", "path": "src", "max_hits": 5, "ext": "all"}
</function>
</tool_call>
''',
    "tool_call_unclosed": '''
<tool_call>

read_file
{
  "path": "src/AircraftInterface.c",
  "limit": 30,
  "offset": 15
}
''',
    "function_only": '''
<function=find_files>
{"name": "wind.h"}
</function>
''',
    "bare_json": '''
{
  "tool": "read_file",
  "args": {
    "path": "src/FPGA.h",
    "limit": 100
  }
}
''',
    "bare_tool_name_args": '''
    </think>
    search
    {"pattern": "#include.*wind\\\\.h", "path": "src", "max_hits": 5, "ext": "c"}
    </action>
''',
    "bare_build": '''
    </think>
    build
    {}
''',
    "fn_call": '''
search("FPGA_UART_IMU_BASEADDR", "src", max_hits=10, ext="h")
</action>
''',
    "unclosed_action": '''
<action>
{"tool": "note", "args": {"text": "missing header"}}
''',
}


def test_parse_all_common_formats():
    for name, text in SAMPLES.items():
        act = orch.parse_action(text)
        assert act is not None, f"{name} failed to parse:\n{text}"
        assert act.tool in orch._KNOWN_TOOL_NAMES, f"{name}: bad tool {act.tool}"
        assert isinstance(act.args, dict), f"{name}: args not dict"
    # Spot-check tools
    assert orch.parse_action(SAMPLES["tool_call"]).tool == "search"
    assert orch.parse_action(SAMPLES["bare_build"]).tool == "build"
    assert orch.parse_action(SAMPLES["fn_call"]).tool == "search"
    assert orch.parse_action(SAMPLES["fn_call"]).args.get("pattern") == (
        "FPGA_UART_IMU_BASEADDR"
    )
    assert orch.parse_action(SAMPLES["function_only"]).tool == "find_files"


def test_reject_prose_without_action():
    prose = (
        'The build is failing because AircraftInterface.c includes "wind.h". '
        "I need to determine if wind.h is actually needed."
    )
    assert orch.parse_action(prose) is None
    assert orch.parse_action("") is None
    assert orch.parse_action("search for \"wind.h\" to see if it exists") is None


def test_coerce_abs_sandbox_path():
    sb = Path("/home/developer/workspace/sessions/abc/sandbox")
    rel = orch._coerce_rel_path(
        sb,
        "/home/developer/workspace/sessions/abc/sandbox/src/AircraftInterface.c",
    )
    assert rel == "src/AircraftInterface.c"
    assert orch._coerce_rel_path(sb, "src/foo.c") == "src/foo.c"


def test_canonical_still_preferred():
    text = SAMPLES["canonical"]
    act = orch.parse_action(text)
    assert act is not None
    assert act.tool == "search"
    assert "wind" in act.args.get("pattern", "")


if __name__ == "__main__":
    test_parse_all_common_formats()
    test_reject_prose_without_action()
    test_coerce_abs_sandbox_path()
    test_canonical_still_preferred()
    print("ALL TESTS PASSED")
