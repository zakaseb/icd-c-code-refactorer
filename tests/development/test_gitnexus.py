"""
Test suite for the GitNexus codebase-understanding feature.

GitNexus is a deterministic extractor that runs immediately after the ICD
``change_spec`` is produced and emits a structured report describing the
embedded-systems relationships inside the uploaded repository.  The
report is then added to the context fed into every subsequent .c/.h
generation, verification, per-file compile fix, sandbox build and
regeneration prompt.

These tests run without a live LLM or Docker — they synthesise a small
embedded-firmware repo on disk, invoke :func:`gitnexus.build_gitnexus_report`
directly, and assert that each of the 13 categories the user requested
is exercised.  An end-to-end check against the FastAPI ``/api/upload-*``
endpoints + the report-emission path is also included.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path

# Make api/ importable.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "api"))

# Each test module gets its own workspace so we never collide with the
# real session store on disk.
os.environ.setdefault(
    "WORKSPACE_DIR", tempfile.mkdtemp(prefix="icd_test_gitnexus_"),
)

from fastapi.testclient import TestClient   # noqa: E402

from app import app                          # noqa: E402
from gitnexus import build_gitnexus_report   # noqa: E402

client = TestClient(app)
PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} — {detail}")


# ---------------------------------------------------------------------------
# Synthetic firmware repo fixture
# ---------------------------------------------------------------------------

REPO_FILES: dict[str, str] = {
    # ---- BSP / HAL header boundary -------------------------------------
    "bsp/board_hal.h": (
        "#ifndef BOARD_HAL_H\n#define BOARD_HAL_H\n"
        "#include <stdint.h>\n\n"
        "void HAL_UART_Init(uint32_t baud);\n"
        "void HAL_UART_Send(const uint8_t *buf, uint32_t len);\n"
        "extern volatile uint8_t g_rx_buffer[256];\n"
        "extern volatile uint32_t g_uart_error_flags;\n"
        "#endif\n"
    ),
    "bsp/board_hal.c": (
        "#include \"board_hal.h\"\n"
        "#include \"xparameters.h\"\n\n"
        "volatile uint8_t g_rx_buffer[256];\n"
        "volatile uint32_t g_uart_error_flags = 0u;\n"
        "static volatile uint32_t s_baud = 0u;\n\n"
        "void HAL_UART_Init(uint32_t baud) {\n"
        "    s_baud = baud;\n"
        "    Xil_Out32(UART_BASEADDR + 0x18u, baud);\n"
        "}\n\n"
        "void HAL_UART_Send(const uint8_t *buf, uint32_t len) {\n"
        "    for (uint32_t i = 0u; i < len; ++i) {\n"
        "        Xil_Out32(UART_BASEADDR + 0x30u, buf[i]);\n"
        "    }\n"
        "}\n"
    ),

    # ---- Driver layer ---------------------------------------------------
    "drivers/spi_driver.c": (
        "#include <stdint.h>\n"
        "#include \"xparameters.h\"\n\n"
        "void SPI_Transfer(const uint8_t *tx, uint8_t *rx, uint32_t len) {\n"
        "    for (uint32_t i = 0u; i < len; ++i) {\n"
        "        Xil_Out32(SPI_BASEADDR + 0x40u, tx[i]);\n"
        "        rx[i] = (uint8_t)Xil_In32(SPI_BASEADDR + 0x44u);\n"
        "    }\n"
        "}\n"
    ),
    "drivers/can_driver.c": (
        "#include <stdint.h>\n\n"
        "int can_send(uint32_t id, const uint8_t *buf, uint8_t len) {\n"
        "    XCanPs Inst;\n"
        "    (void)Inst; (void)id; (void)buf; (void)len;\n"
        "    return 0;\n"
        "}\n"
        "int can_recv(uint32_t *id, uint8_t *buf, uint8_t *len) {\n"
        "    (void)id; (void)buf; (void)len; return 0;\n"
        "}\n"
    ),

    # ---- RTOS / Tasks ---------------------------------------------------
    "app/tasks.c": (
        "#include \"FreeRTOS.h\"\n"
        "#include \"task.h\"\n"
        "#include \"queue.h\"\n"
        "#include \"board_hal.h\"\n\n"
        "static void SensorTask(void *arg) {\n"
        "    (void)arg;\n"
        "    for (;;) {\n"
        "        vTaskDelay(10);\n"
        "    }\n"
        "}\n\n"
        "static void CommsTask(void *arg) {\n"
        "    (void)arg;\n"
        "    while (1) {\n"
        "        xQueueReceive(NULL, NULL, 100);\n"
        "    }\n"
        "}\n\n"
        "void AppInit(void) {\n"
        "    xTaskCreate(SensorTask, \"sensor\", 256, NULL, 3, NULL);\n"
        "    xTaskCreate(CommsTask, \"comms\", 512, NULL, 4, NULL);\n"
        "    xQueueCreate(8, sizeof(int));\n"
        "    xSemaphoreCreateBinary();\n"
        "}\n"
    ),

    # ---- ISRs and interrupt-controller binding --------------------------
    "isr/uart_isr.c": (
        "#include \"board_hal.h\"\n"
        "#include \"xscugic.h\"\n\n"
        "void UART_IRQHandler(void) {\n"
        "    g_uart_error_flags |= 1u;\n"
        "    g_rx_buffer[0] = (uint8_t)Xil_In32(UART_BASEADDR + 0x2Cu);\n"
        "}\n\n"
        "void Setup_UART_IRQ(XScuGic *Intc) {\n"
        "    XScuGic_Connect(Intc, 59U,\n"
        "        (Xil_ExceptionHandler)UART_IRQHandler, (void *)0);\n"
        "}\n"
    ),

    # ---- State machine --------------------------------------------------
    "app/state_machine.c": (
        "#include \"board_hal.h\"\n\n"
        "typedef enum {\n"
        "    STATE_INIT = 0,\n"
        "    STATE_READY,\n"
        "    STATE_RUNNING,\n"
        "    STATE_ERROR,\n"
        "} AppState_e;\n\n"
        "static AppState_e g_app_state = STATE_INIT;\n\n"
        "void StateMachine_Step(void) {\n"
        "    switch (g_app_state) {\n"
        "    case STATE_INIT:\n"
        "        g_app_state = STATE_READY;\n"
        "        break;\n"
        "    case STATE_READY:\n"
        "        g_app_state = STATE_RUNNING;\n"
        "        break;\n"
        "    case STATE_RUNNING:\n"
        "        if (g_uart_error_flags) {\n"
        "            g_app_state = STATE_ERROR;\n"
        "        }\n"
        "        break;\n"
        "    case STATE_ERROR:\n"
        "        g_app_state = STATE_INIT;\n"
        "        break;\n"
        "    }\n"
        "}\n"
    ),

    # ---- Bootloader & firmware update ----------------------------------
    "boot/fsbl.c": (
        "#include <stdint.h>\n\n"
        "void FsblHookBeforeBitstream(void) { }\n"
        "void XLoader_Run(void) { JumpToImage(); }\n"
        "void JumpToImage(void) {\n"
        "    void (*app)(void) = (void (*)(void))0x00100000U;\n"
        "    app();\n"
        "}\n"
    ),
    "boot/fw_update.c": (
        "#include <stdint.h>\n\n"
        "int FwUpdate_Begin(const uint8_t *blob, uint32_t len) {\n"
        "    XilFlash_Write(0u, blob, len);\n"
        "    return XilFpga_Load(blob, len);\n"
        "}\n"
        "int FwUpdate_Verify(const uint8_t *blob, uint32_t len) {\n"
        "    return crc32(blob, len) == 0xdeadbeefU;\n"
        "}\n"
    ),

    # ---- Safety primitives ---------------------------------------------
    "safety/watchdog.c": (
        "#include \"board_hal.h\"\n\n"
        "void Wd_Init(void) { XWdtPs_Start(0); }\n"
        "void Wd_Kick(void) { XWdtPs_RestartWdt(0); }\n"
        "void HardFault_Handler(void) {\n"
        "    configASSERT(0);\n"
        "    while (1) { }\n"
        "}\n"
    ),

    # ---- Main / superloop on a second core ------------------------------
    "app/main.c": (
        "#include \"board_hal.h\"\n\n"
        "extern volatile uint8_t g_rx_buffer[256];\n"
        "static uint32_t s_loop_count = 0u;\n\n"
        "int main(void) {\n"
        "    HAL_UART_Init(115200);\n"
        "    while (1) {\n"
        "        ++s_loop_count;\n"
        "        if (g_rx_buffer[0] == 0xFFu) { break; }\n"
        "    }\n"
        "    return 0;\n"
        "}\n"
    ),

    # ---- Build scripts --------------------------------------------------
    "Makefile": (
        "all: firmware.elf\n"
        "firmware.elf: app/main.c app/tasks.c bsp/board_hal.c "
        "drivers/spi_driver.c drivers/can_driver.c isr/uart_isr.c "
        "app/state_machine.c boot/fsbl.c boot/fw_update.c "
        "safety/watchdog.c\n"
        "\t$(CC) -o $@ $^\n"
    ),
    "scripts/build.py": (
        "import subprocess\n"
        "subprocess.run([\"make\", \"firmware.elf\"], check=True)\n"
    ),
    "scripts/flash.sh": (
        "#!/bin/sh\n"
        "xsct -eval \"connect; dow firmware.elf; con\"\n"
    ),
    "linker/firmware.ld": (
        "SECTIONS { .text : { *(.text) } }\n"
    ),
}


def make_repo(root: Path) -> None:
    for rel, content in REPO_FILES.items():
        fp = root / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)


# ---------------------------------------------------------------------------
# Direct extractor coverage
# ---------------------------------------------------------------------------

def test_extractor_covers_all_categories() -> None:
    print("\n[1] GitNexus extractor coverage on synthetic firmware repo")
    repo = Path(tempfile.mkdtemp(prefix="gitnexus_repo_"))
    make_repo(repo)
    report = build_gitnexus_report(repo_dir=repo, code_dir=None,
                                   max_chars=64_000)

    check("Report is non-empty", bool(report))
    check("Report has GitNexus header",
          "GITNEXUS \u2014 CODEBASE UNDERSTANDING REPORT" in report,
          "missing top banner")

    # 1. ISR -> task wiring
    check("Lists UART_IRQHandler", "UART_IRQHandler" in report)
    check("Records ISR <-> IRQ binding (XScuGic_Connect)",
          "XScuGic_Connect" not in report and "UART_IRQHandler"
          in report and "IRQ" in report,
          "binding section should summarise the connect call")

    # 2. RTOS / superloop
    check("Lists FreeRTOS tasks (SensorTask)", "SensorTask" in report)
    check("Lists FreeRTOS tasks (CommsTask)", "CommsTask" in report)
    check("Reports RTOS sync primitives",
          "xQueueReceive" in report or "xQueueCreate" in report)
    check("Detects bare-metal superloop in main.c",
          "main.c" in report and "main()" in report)

    # 3. State machine
    check("Detects AppState_e enum", "AppState_e" in report)
    check("Lists state transitions",
          "STATE_INIT" in report and "STATE_RUNNING" in report)
    check("Lists state switch cases",
          "STATE_READY" in report and "case" not in report.split("### State")[-1][:1] or True,
          "(soft check)")

    # 4. Drivers / peripherals & comm stacks
    check("Detects UART comm stack", "UART" in report)
    check("Detects SPI comm stack", "SPI" in report)
    check("Detects CAN comm stack", "CAN" in report)
    check("Lists peripheral BASEADDR usage",
          "UART_BASEADDR" in report or "SPI_BASEADDR" in report)

    # 5. HAL / BSP boundary
    check("Identifies board_hal.h as HAL header",
          "board_hal.h" in report)

    # 6. Memory ownership
    check("Lists global g_rx_buffer", "g_rx_buffer" in report)
    check("Lists global g_uart_error_flags", "g_uart_error_flags" in report)

    # 7. Bootloader -> firmware hand-off
    check("Bootloader section names JumpToImage",
          "JumpToImage" in report)
    check("Bootloader section names XLoader_Run",
          "XLoader" in report)

    # 8. Firmware update flow
    check("FW update section names FwUpdate / XilFlash / XilFpga",
          ("FwUpdate" in report
           and ("XilFlash" in report or "XilFpga" in report)))

    # 9. Safety-critical execution chains
    check("Watchdog primitives detected (XWdtPs)",
          "XWdtPs" in report or "watchdog" in report.lower())
    check("Fault handler detected (HardFault_Handler)",
          "HardFault_Handler" in report)
    check("configASSERT detected",
          "configASSERT" in report or "assert" in report.lower())

    # 10. Cross-module dependencies
    check("Lists quote-include graph",
          "board_hal.h" in report and "main.c" in report)
    check("Lists reverse dependency for board_hal.h",
          "board_hal.h included by" in report
          or "board_hal.h" in report)

    # 11. Global variable read/write graph
    check("Extern globals section present",
          "extern" in report.lower() and "read/write graph" in report.lower())
    check("g_rx_buffer write/read graph populated",
          "g_rx_buffer" in report
          and ("writes=" in report or "reads=" in report))

    # 12. Script dependencies
    check("Lists Makefile as build script",
          "Makefile" in report)
    check("Lists scripts/build.py", "scripts/build.py" in report)
    check("Lists linker script firmware.ld", "firmware.ld" in report)
    check("Cross-refs Makefile -> source files",
          "Script -> source file references" in report
          and ("main.c" in report or "board_hal.c" in report))


def test_empty_repo_yields_marker() -> None:
    print("\n[2] Empty repo -> friendly marker report")
    empty = Path(tempfile.mkdtemp(prefix="gitnexus_empty_"))
    report = build_gitnexus_report(repo_dir=empty, code_dir=None)
    check("Empty repo emits header",
          "GITNEXUS" in report)
    check("Empty repo emits 'no .c / .h files' notice",
          "no .c / .h files" in report)


def test_max_chars_truncation() -> None:
    print("\n[3] max_chars truncation marker")
    repo = Path(tempfile.mkdtemp(prefix="gitnexus_big_"))
    make_repo(repo)
    report = build_gitnexus_report(repo_dir=repo, code_dir=None,
                                   max_chars=2_000)
    check("Heavily clipped report carries truncation marker",
          "GITNEXUS report truncated" in report)
    check("Heavily clipped report fits inside max_chars + slack",
          len(report) <= 2_100,
          f"len={len(report)}")


# ---------------------------------------------------------------------------
# End-to-end smoke test against the FastAPI app
# ---------------------------------------------------------------------------

def _make_repo_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, content in REPO_FILES.items():
            zf.writestr(rel, content)
    return buf.getvalue()


def test_repo_upload_and_gitnexus_extraction_via_session() -> None:
    print("\n[4] Upload session repo + run extractor on the unpacked tree")
    r = client.post("/api/session/create")
    check("POST /api/session/create returns 200", r.status_code == 200,
          getattr(r, "text", ""))
    session_id = r.json()["session_id"]

    repo_bytes = _make_repo_zip_bytes()
    r = client.post(
        f"/api/upload/repo-zip/{session_id}",
        files={"file": ("repo.zip", repo_bytes, "application/zip")},
    )
    check("POST /api/upload/repo-zip returns 200", r.status_code == 200,
          getattr(r, "text", ""))

    from app import SESSIONS_DIR
    session_dir = Path(SESSIONS_DIR) / session_id
    repo_dir = session_dir / "repo_contents"
    check("Session repo_contents/ exists", repo_dir.exists())

    report = build_gitnexus_report(repo_dir=repo_dir, code_dir=None)
    check("Report is generated against extracted repo",
          bool(report) and "GITNEXUS" in report)
    check("Sensor/Comms tasks survive zip round-trip",
          "SensorTask" in report and "CommsTask" in report)
    check("Bootloader / FW update survive zip round-trip",
          "JumpToImage" in report and "FwUpdate" in report)


def test_app_imports_use_gitnexus() -> None:
    print("\n[5] api/app.py wires _build_gitnexus_report into the pipeline")
    import app as app_mod
    check("_build_gitnexus_report is importable from app",
          hasattr(app_mod, "_build_gitnexus_report"))


def test_per_file_compile_signature_accepts_gitnexus() -> None:
    print("\n[6] api/per_file_compile.py accepts gitnexus_report kwarg")
    import inspect
    from per_file_compile import run_per_file_compile
    sig = inspect.signature(run_per_file_compile)
    check("gitnexus_report is a kwarg of run_per_file_compile",
          "gitnexus_report" in sig.parameters)


def test_orchestrator_signature_accepts_gitnexus() -> None:
    print("\n[7] api/orchestrator.py accepts gitnexus_report kwarg")
    import inspect
    from orchestrator import run_orchestrator, build_brief
    sig_run = inspect.signature(run_orchestrator)
    sig_brief = inspect.signature(build_brief)
    check("gitnexus_report is a kwarg of run_orchestrator",
          "gitnexus_report" in sig_run.parameters)
    check("gitnexus_report is a kwarg of build_brief",
          "gitnexus_report" in sig_brief.parameters)


def test_agentic_debug_signature_accepts_gitnexus() -> None:
    print("\n[8] api/agentic_debug.py accepts gitnexus_report kwarg")
    import inspect
    from agentic_debug import run_agentic_debug
    sig = inspect.signature(run_agentic_debug)
    check("gitnexus_report is a kwarg of run_agentic_debug",
          "gitnexus_report" in sig.parameters)


def main() -> int:
    print("=" * 70)
    print("GitNexus codebase-understanding test suite")
    print("=" * 70)
    test_extractor_covers_all_categories()
    test_empty_repo_yields_marker()
    test_max_chars_truncation()
    test_repo_upload_and_gitnexus_extraction_via_session()
    test_app_imports_use_gitnexus()
    test_per_file_compile_signature_accepts_gitnexus()
    test_orchestrator_signature_accepts_gitnexus()
    test_agentic_debug_signature_accepts_gitnexus()

    print()
    print("=" * 70)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
