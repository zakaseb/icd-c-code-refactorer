"""
Focused tests for the Xilinx SDK 2018 + superloop operational profile.

These tests cover the refactored agentic pipeline primitives:
- deterministic error-family ordering,
- firmware/software lane separation,
- superloop contract validation,
- linker/BSP consistency audits,
- end-to-end lane-gated convergence in run_agentic_debug.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.agentic_debug import (  # noqa: E402
    BuildError,
    ErrorClass,
    Hypothesis,
    Lane,
    apply_hypothesis,
    cluster_errors,
    evaluate_lanes,
    run_agentic_debug,
    run_linker_bsp_consistency_audits,
    run_superloop_contract_checks,
)


PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} — {detail}")


def test_deterministic_ordering() -> None:
    print("\n=== Test 1: Deterministic repair-family ordering ===")
    errs = [
        BuildError(
            tool="link",
            severity="error",
            file="",
            line=0,
            column=0,
            symbol="Xil_ExceptionHandler",
            message="undefined reference to `Xil_ExceptionHandler'",
            error_class=ErrorClass.LINK_UNDEFINED,
        ),
        BuildError(
            tool="compile",
            severity="error",
            file="app/state.c",
            line=10,
            column=3,
            symbol="process_state",
            message="too few arguments to function 'process_state'",
            error_class=ErrorClass.SIGNATURE_MISMATCH,
        ),
        BuildError(
            tool="compile",
            severity="error",
            file="drivers/uart.c",
            line=2,
            column=1,
            symbol="legacy_regs.h",
            message="fatal error: legacy_regs.h: No such file or directory",
            error_class=ErrorClass.MISSING_INCLUDE,
        ),
    ]
    clusters = cluster_errors(errs, index=None)
    classes = [c.error_class for c in clusters]
    check(
        "ordering: include before signature before linker",
        classes[:3] == [
            ErrorClass.MISSING_INCLUDE,
            ErrorClass.SIGNATURE_MISMATCH,
            ErrorClass.LINK_UNDEFINED,
        ],
        str(classes),
    )


def test_lane_split() -> None:
    print("\n=== Test 2: Lane split and active-lane gate ===")
    errs = [
        BuildError(
            tool="compile",
            severity="error",
            file="bsp/drivers/spi.c",
            line=7,
            column=1,
            symbol="XSpi_Config",
            message="unknown type name 'XSpi_Config'",
            error_class=ErrorClass.UNKNOWN_TYPE,
        ),
        BuildError(
            tool="compile",
            severity="error",
            file="app/state_machine.c",
            line=42,
            column=4,
            symbol="process_state",
            message="too few arguments to function 'process_state'",
            error_class=ErrorClass.SIGNATURE_MISMATCH,
        ),
        BuildError(
            tool="link",
            severity="error",
            file="",
            line=0,
            column=0,
            symbol="Timer_IRQHandler",
            message="undefined reference to `Timer_IRQHandler'",
            error_class=ErrorClass.LINK_UNDEFINED,
        ),
    ]
    lanes = evaluate_lanes(errs)
    check("firmware lane has 2 errors", len(lanes.firmware.errors) == 2, str(len(lanes.firmware.errors)))
    check("software lane has 1 error", len(lanes.software.errors) == 1, str(len(lanes.software.errors)))
    check("active lane is firmware (hard gate)", lanes.active_lane == Lane.FIRMWARE, lanes.active_lane.value)


def test_superloop_contracts() -> None:
    print("\n=== Test 3: Superloop contract checks ===")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        main_c = root / "main.c"
        before = (
            "void init_hw(void); void poll_inputs(void); void process_state(void); void dispatch_outputs(void);\n"
            "int main(void) {\n"
            "  int tick = 0;\n"
            "  while (1) {\n"
            "    init_hw();\n"
            "    poll_inputs();\n"
            "    process_state();\n"
            "    dispatch_outputs();\n"
            "    if ((tick % 10) == 0) { poll_inputs(); }\n"
            "    tick++;\n"
            "  }\n"
            "}\n"
        )
        main_c.write_text(before)
        h = Hypothesis(
            id="H1",
            layer="shim",
            kind="test_patch",
            title="break superloop ordering",
            rationale="unit test",
            target_files=["main.c"],
            risk="low",
            expected_resolved=[],
        )
        after = (
            "void init_hw(void); void poll_inputs(void); void process_state(void); void dispatch_outputs(void);\n"
            "int main(void) {\n"
            "  int tick = 0;\n"
            "  while (1) {\n"
            "    init_hw();\n"
            "    process_state();\n"
            "    poll_inputs();\n"
            "    dispatch_outputs();\n"
            "    if ((tick % 5) == 0) { poll_inputs(); }\n"
            "    sleep(1);\n"
            "    tick++;\n"
            "  }\n"
            "}\n"
        )
        patch = apply_hypothesis(root, h, {"main.c": after})
        errs, warns = run_superloop_contract_checks(
            sandbox_dir=root,
            patch_result=patch,
        )
        msg = " | ".join(errs + warns)
        check("ordering regression detected", any("ordering" in e for e in errs), msg)
        check("cadence regression detected", any("cadence" in e for e in errs), msg)
        check("blocking regression detected", any("blocking" in e for e in errs), msg)


def test_linker_bsp_audits() -> None:
    print("\n=== Test 4: Linker/BSP consistency audits ===")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        bsp = root / "bsp/include/xparameters.h"
        bsp.parent.mkdir(parents=True, exist_ok=True)
        bsp.write_text("#define XPAR_UART0_BASEADDR 0x40000000\n")
        baseline = {bsp: "#define XPAR_UART0_BASEADDR 0x40000000\n"}
        # mutate header (simulating unintended divergence)
        bsp.write_text("#define XPAR_UART0_BASEADDR 0x50000000\n")
        output = (
            "ld: section .text will not fit in region BRAM\n"
            "ld: region BRAM overflowed by 1024 bytes\n"
            "foo.o: multiple definition of `drv_init'; bar.o: first defined here\n"
        )
        errs = [
            BuildError(
                tool="link",
                severity="error",
                file="",
                line=0,
                column=0,
                symbol="Timer_IRQHandler",
                message="undefined reference to `Timer_IRQHandler'",
                error_class=ErrorClass.LINK_UNDEFINED,
            )
        ]
        audits = run_linker_bsp_consistency_audits(
            build_output=output,
            errors=errs,
            sandbox_dir=root,
            touched_files=[bsp],
            baseline_snapshots=baseline,
        )
        blob = " | ".join(audits)
        check("ISR unresolved audit present", "ISR/handler" in blob, blob)
        check("section placement audit present", "Section placement" in blob, blob)
        check("duplicate symbol audit present", "Duplicate symbol" in blob, blob)
        check("BSP divergence audit present", "BSP-generated header divergence" in blob, blob)


def test_end_to_end_lane_gated_convergence() -> None:
    print("\n=== Test 5: End-to-end lane-gated convergence ===")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        session_dir = root / "session"
        sandbox_dir = session_dir / "sandbox"
        (sandbox_dir / "drivers").mkdir(parents=True)
        (sandbox_dir / "app").mkdir(parents=True)

        (sandbox_dir / "drivers/uart.h").write_text(
            "typedef int NewRegType;\n"
        )
        (sandbox_dir / "drivers/uart.c").write_text(
            '#include "uart.h"\nOldRegType g_uart;\n'
        )
        (sandbox_dir / "app/state.c").write_text(
            "int process_state_v2(int mode);\n"
            "int app_tick(void) { return process_state_v2(); }\n"
        )

        build_calls = {"n": 0}

        def fake_runner() -> tuple[bool, str]:
            build_calls["n"] += 1
            hdr = (sandbox_dir / "drivers/uart.h").read_text()
            app = (sandbox_dir / "app/state.c").read_text()
            errs = []
            if "typedef int OldRegType;" not in hdr:
                errs.append("drivers/uart.c:3:1: error: unknown type name 'OldRegType'")
            if "process_state_v2(0)" not in app:
                errs.append("app/state.c:2:20: error: too few arguments to function 'process_state_v2'")
            if errs:
                return False, "\n".join(errs)
            return True, "Build succeeded with full link."

        planner_prompts: list[str] = []

        def fake_llm(system: str, user: str, *, max_tokens: int, meta=None):
            if "Fix Strategy Agent" in system:
                planner_prompts.append(user)
                if "Active verification lane: firmware" in user:
                    yield (
                        "```json\n"
                        "{\"id\":\"H1\",\"layer\":\"shim\",\"kind\":\"header_typedef_bridge\","
                        "\"title\":\"Add OldRegType compatibility alias\","
                        "\"rationale\":\"Firmware lane has unknown type OldRegType\","
                        "\"target_files\":[\"drivers/uart.h\"],"
                        "\"risk\":\"low\",\"expected_resolved\":[\"unknown_type: symbol 'OldRegType'\"],"
                        "\"notes\":\"\"}\n```"
                    )
                else:
                    yield (
                        "```json\n"
                        "{\"id\":\"H2\",\"layer\":\"shim\",\"kind\":\"signature_wrapper\","
                        "\"title\":\"Provide old call signature bridge\","
                        "\"rationale\":\"Software lane has signature drift\","
                        "\"target_files\":[\"app/state.c\"],"
                        "\"risk\":\"low\",\"expected_resolved\":[\"signature_mismatch\"],"
                        "\"notes\":\"\"}\n```"
                    )
            else:
                if "\"drivers/uart.h\"" in user:
                    yield (
                        "### drivers/uart.h\n```c\n"
                        "typedef int NewRegType;\n"
                        "typedef int OldRegType;\n"
                        "```\n"
                    )
                else:
                    yield (
                        "### app/state.c\n```c\n"
                        "int process_state_v2(int mode);\n"
                        "int app_tick(void) { return process_state_v2(0); }\n"
                        "```\n"
                    )
            if meta is not None:
                meta["finish_reason"] = "stop"

        snapshots = {
            sandbox_dir / "drivers/uart.h": (sandbox_dir / "drivers/uart.h").read_text(),
            sandbox_dir / "drivers/uart.c": (sandbox_dir / "drivers/uart.c").read_text(),
            sandbox_dir / "app/state.c": (sandbox_dir / "app/state.c").read_text(),
        }
        file_index = {
            "uart.h": [sandbox_dir / "drivers/uart.h"],
            "uart.c": [sandbox_dir / "drivers/uart.c"],
            "state.c": [sandbox_dir / "app/state.c"],
        }

        events = list(run_agentic_debug(
            session_dir=session_dir,
            sandbox_dir=sandbox_dir,
            build_info={"type": "make", "build_dir": sandbox_dir, "path": None},
            sandbox_cc="arm-none-eabi-gcc -std=c99",
            is_cross=True,
            gen_files={"drivers/uart.c": "uart.c"},
            change_spec="Legacy OldRegType renamed to NewRegType; process_state_v2 now takes mode.",
            repo_knowledge="",
            file_index=file_index,
            snapshots=snapshots,
            build_runner=fake_runner,
            llm_stream=fake_llm,
            max_attempts=8,
            no_progress_limit=3,
            oscillation_limit=2,
        ))
        done = [e for e in events if e.get("type") == "done"]
        check("done event emitted", bool(done), str(events[-3:]))
        check("done success true", bool(done) and done[-1]["success"] is True, str(done[-1] if done else ""))
        check(
            "planner saw firmware lane first",
            any("Active verification lane: firmware" in p for p in planner_prompts),
            json.dumps(planner_prompts, indent=2)[:500],
        )
        check(
            "planner later saw software lane",
            any("Active verification lane: software" in p for p in planner_prompts),
            json.dumps(planner_prompts, indent=2)[:500],
        )


if __name__ == "__main__":
    test_deterministic_ordering()
    test_lane_split()
    test_superloop_contracts()
    test_linker_bsp_audits()
    test_end_to_end_lane_gated_convergence()

    print(f"\n{'=' * 60}")
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} checks")
    if FAIL:
        sys.exit(1)
    print("All tests passed!")
    sys.exit(0)
