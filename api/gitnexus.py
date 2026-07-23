"""
GitNexus — Codebase Understanding Context for the ICD Refactorer.

GitNexus runs *after* the ICD ``change_spec`` has been generated and
extracts a structured, deterministic snapshot of *embedded-systems-specific*
relationships in the uploaded repository (and uploaded source scripts).

The output is a single human-readable Markdown report stored at
``session_dir / "gitnexus_report.txt"``.  It is **also** added to the
context bundle that every subsequent .c / .h generation, verification,
per-file-compile fix, sandbox build and regeneration prompt sees — so the
LLM rewriting the code has direct knowledge of:

  * ISR -> task wiring
  * driver -> peripheral access
  * RTOS / superloop task interactions
  * state-machine transitions
  * communication stack dependencies
  * memory ownership relationships
  * hardware abstraction (HAL / BSP) boundaries
  * bootloader -> firmware hand-off
  * firmware update flow
  * safety-critical execution chains (watchdog, CRC, asserts)
  * cross-module include dependencies
  * global-variable read/write graph
  * build / packaging script dependencies

The extraction is **purely deterministic** — it relies on regex pattern
matching across the repository so it runs in milliseconds, has no LLM
cost, and is fully reproducible.  Heuristics are intentionally
conservative: when a pattern is uncertain it surfaces the raw evidence
("file: line — symbol") rather than fabricating a high-level claim.
That keeps GitNexus useful even when the LLM consumes it as
ground-truth context downstream.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

# Source / header / build-script extensions we care about.
_HEADER_EXTS = {".h", ".hpp", ".hh", ".hxx"}
_SOURCE_EXTS = {".c", ".cpp", ".cc", ".cxx"}
_CODE_EXTS = _HEADER_EXTS | _SOURCE_EXTS
_SCRIPT_EXTS = {".sh", ".py", ".tcl", ".cmake", ".mk", ".ld", ".lds",
                ".bif", ".bat", ".ps1", ".yaml", ".yml", ".json"}
_SCRIPT_NAMES = {"makefile", "cmakelists.txt", "build.xml", "kbuild",
                 "kconfig", "rules.mk"}

# ---------------------------------------------------------------------------
# Pattern library
# ---------------------------------------------------------------------------

# ISR handlers across the common embedded toolchains: ARM CMSIS, Xilinx
# (XScuGic_Connect / XIntc_Connect), GCC __attribute__((interrupt)),
# vendor-style *_IRQHandler / *_Handler / ISR_ prefixes.
_ISR_PATTERNS = [
    re.compile(r'\bvoid\s+([A-Za-z_]\w*?_IRQHandler)\s*\(', re.MULTILINE),
    re.compile(r'\bvoid\s+(ISR_[A-Za-z_]\w*)\s*\(', re.MULTILINE),
    re.compile(r'\bvoid\s+([A-Za-z_]\w*?_Handler)\s*\(', re.MULTILINE),
    re.compile(r'__attribute__\s*\(\(\s*interrupt[^)]*\)\)\s*'
               r'(?:void\s+)?([A-Za-z_]\w*)', re.MULTILINE),
    re.compile(r'\bvoid\s+([A-Za-z_]\w*?_isr)\s*\(', re.MULTILINE),
]
# Interrupt-controller registration calls — these link an ISR symbol
# with the IRQ source it services.  We capture the call so the report
# can list "ISR <-> IRQ id" wiring.
_ISR_BIND_PATTERNS = [
    re.compile(
        r'XScuGic_Connect\s*\(\s*[^,]+,\s*([^,]+),\s*'
        r'\(\s*Xil_(?:ExceptionHandler|InterruptHandler)\s*\)\s*'
        r'([A-Za-z_]\w*)\s*,', re.MULTILINE),
    re.compile(
        r'XIntc_Connect\s*\(\s*[^,]+,\s*([^,]+),\s*'
        r'\(\s*XInterruptHandler\s*\)\s*([A-Za-z_]\w*)\s*,', re.MULTILINE),
    re.compile(
        r'NVIC_EnableIRQ\s*\(\s*([A-Z_][A-Z0-9_]+_IRQn)\s*\)',
        re.MULTILINE),
    re.compile(
        r'IRQ_CONNECT\s*\(\s*([^,]+),\s*[^,]+,\s*([A-Za-z_]\w*)',
        re.MULTILINE),
]

# RTOS API surface (FreeRTOS / CMSIS-RTOS / Zephyr / xilkernel).
_RTOS_TASK_PATTERNS = [
    re.compile(
        r'\b(xTaskCreate(?:Static|Restricted)?)\s*\(\s*'
        r'([A-Za-z_]\w*)\s*,\s*"?([^",)]+)"?', re.MULTILINE),
    re.compile(r'\b(osThreadNew)\s*\(\s*([A-Za-z_]\w*)\s*,', re.MULTILINE),
    re.compile(r'\b(K_THREAD_DEFINE)\s*\(\s*([A-Za-z_]\w*)\s*,',
               re.MULTILINE),
    re.compile(r'\b(thread_create)\s*\(\s*[^,]+,\s*[^,]+,\s*'
               r'([A-Za-z_]\w*)\s*,', re.MULTILINE),
]
_RTOS_SYNC_PATTERNS = [
    re.compile(r'\b(xQueue(?:Create|Send|Receive|SendFromISR|ReceiveFromISR))\s*\(',
               re.MULTILINE),
    re.compile(r'\b(xSemaphore(?:Create\w*|Take|Give|GiveFromISR))\s*\(',
               re.MULTILINE),
    re.compile(r'\b(xEventGroup(?:Create|SetBits|WaitBits|ClearBits))\s*\(',
               re.MULTILINE),
    re.compile(r'\b(k_(?:msgq|sem|mutex|fifo)_\w+)\s*\(', re.MULTILINE),
]
# Superloop hint — bare main loops are the dominant pattern in
# bare-metal firmware.
_SUPERLOOP_PATTERNS = [
    re.compile(r'\bint\s+main\s*\([^)]*\)\s*\{', re.MULTILINE),
    re.compile(r'\bvoid\s+main\s*\([^)]*\)\s*\{', re.MULTILINE),
]
_SUPERLOOP_BODY_PATTERNS = [
    re.compile(r'\bwhile\s*\(\s*1\s*\)', re.MULTILINE),
    re.compile(r'\bwhile\s*\(\s*true\s*\)', re.MULTILINE),
    re.compile(r'\bfor\s*\(\s*;\s*;\s*\)', re.MULTILINE),
]

# Communication stacks — UART / SPI / I2C / CAN / Ethernet / lwIP /
# FreeRTOS+TCP / Modbus.  We tag the *driver family* per file so the
# report can answer "which file talks to which transport".
_COMM_STACKS: list[tuple[str, re.Pattern]] = [
    ("UART", re.compile(r'\b(XUart(?:Ps|Lite|Ns550)|HAL_UART|usart\w*|uart\w*'
                        r'|XUartPs_(?:Recv|Send))', re.IGNORECASE)),
    ("SPI", re.compile(r'\b(XSpi(?:Ps)?|HAL_SPI|spi_transfer|spi_xfer)',
                       re.IGNORECASE)),
    ("I2C", re.compile(r'\b(XIicPs|XIic|HAL_I2C|i2c_(?:read|write|transfer))',
                       re.IGNORECASE)),
    ("CAN", re.compile(r'\b(XCanPs|XCan|HAL_CAN|can_(?:send|recv|tx|rx))',
                       re.IGNORECASE)),
    ("Ethernet", re.compile(r'\b(XEmacPs|XEmacLite|HAL_ETH|lwip_\w+|'
                            r'tcpip_\w+|netif_\w+|emac\w*)', re.IGNORECASE)),
    ("USB", re.compile(r'\b(XUsbPs|HAL_PCD|HAL_HCD|tud_\w+|usbd_\w+)',
                       re.IGNORECASE)),
    ("Modbus", re.compile(r'\b(eMB\w+|MB_\w+|modbus_\w+)')),
]

# Peripheral access — base-address macros and Xilinx low-level
# Xil_Out32 / Xil_In32 reads/writes.
_PERIPH_PATTERNS = [
    re.compile(r'\b([A-Z][A-Z0-9_]*_BASE(?:ADDR)?)\b'),
    re.compile(r'\bXil_(?:Out|In)(?:8|16|32|64)\s*\(\s*([A-Z][A-Z0-9_]*)'),
    re.compile(r'\b(GPIO[A-Z]?|USART\d+|SPI\d+|I2C\d+|TIM\d+|UART\d+)'
               r'\s*->\s*[A-Z]', re.MULTILINE),
]
_DRIVER_FILENAME_HINTS = re.compile(
    r'(?i)^(?P<base>[a-z0-9]+(?:_(?:drv|driver|hal|bsp|dev|periph))?'
    r'|x[a-z0-9_]+_(?:hw|ll|sinit|intr))(?:\.\w+)?$'
)
_HAL_HEADER_HINTS = re.compile(
    r'(?i)(?:_hal|_bsp|_ll|_hw|hal_|bsp_|board_|platform_)\.h$')

# State machine — `enum *_State*` definitions and `case STATE_X: ... = STATE_Y;`
_STATE_ENUM_RE = re.compile(
    r'\btypedef\s+enum\b[^{]*\{([^}]+)\}\s*([A-Za-z_]\w*?(?:State|state|STATE)\w*)\s*;',
    re.DOTALL)
_STATE_TRANSITION_RE = re.compile(
    r'\b([A-Za-z_]\w*?[Ss]tate[A-Za-z_]*)\s*=\s*([A-Z_][A-Z0-9_]*)\s*;')
_STATE_SWITCH_CASE_RE = re.compile(
    r'\bcase\s+(STATE_[A-Z0-9_]+|[A-Z_][A-Z0-9_]*_STATE)\b')

# Memory ownership — globals, static buffers, DMA / shared memory.
_GLOBAL_VAR_RE = re.compile(
    r'^(?P<store>(?:static|extern|volatile|const)\s+){0,3}'
    r'(?P<type>(?:unsigned|signed|long|short|struct|enum|union|'
    r'[A-Za-z_]\w*)\s*(?:\*\s*)*)'
    r'(?P<name>[A-Za-z_]\w+)'
    r'(?P<arr>\s*\[[^\]]*\])?'
    r'(?:\s*=\s*[^;]+)?\s*;',
    re.MULTILINE)
_MALLOC_RE = re.compile(r'\b(p?v?Port?Malloc|k_malloc|malloc|calloc|realloc)'
                        r'\s*\(([^)]*)\)')
_FREE_RE = re.compile(r'\b(p?v?Port?Free|k_free|free)\s*\(([^)]*)\)')
_DMA_RE = re.compile(r'\b(XAxiDma|HAL_DMA|dma_\w+|DMA_\w+|XScuGic_DistInit)',
                     re.IGNORECASE)
_SHARED_MEM_RE = re.compile(
    r'__attribute__\s*\(\(\s*section\s*\(\s*"\.([^"]+)"\s*\)\s*\)\)',
    re.MULTILINE)

# Bootloader / firmware-update hooks — Xilinx XilFpga, XilFlash, Zynq
# bootROM markers, OTA / FwUpdate naming.
_BOOTLOADER_PATTERNS = [
    re.compile(r'\b(XLoader\w*|XilLoader\w*|FsblHook\w*|psu_init|'
               r'XPm_BootApu)\b'),
    re.compile(r'\b(JumpToImage|BootMain|StartApp|RunApp|JumpToApplication)\b'),
    re.compile(r'\b(boot_(?:partition|header|image)|image_header_t)\b'),
]
_FWUPDATE_PATTERNS = [
    re.compile(r'\b(XilFlash\w*|XilFpga\w*|FwUpdate\w*|Firmware(?:Update|Image)'
               r'|OtaUpdate\w*|DfuUpdate\w*)\b'),
    re.compile(r'\b(bitstream|XCsuDma_\w+|XilSecure_\w+)\b'),
    re.compile(r'\b(crc32|sha256|sha2|rsa_verify|hash_compute)\s*\(',
               re.IGNORECASE),
]

# Safety-critical execution — watchdog, CRC, asserts, fault handlers.
_SAFETY_PATTERNS = [
    ("watchdog", re.compile(r'\b(XWdtPs|XScuWdt|IWDG_\w+|WWDG_\w+|'
                            r'wdt_(?:start|kick|feed)|Wd_\w+)\b')),
    ("crc",      re.compile(r'\b(crc(?:8|16|32)\w*|Xil_Crc\w*|HAL_CRC|'
                            r'CRC_Calculate)\b', re.IGNORECASE)),
    ("ecc",      re.compile(r'\b(ECC_\w+|Xil_Ecc\w*|ecc_correct\w*)\b')),
    ("fault",    re.compile(r'\b(HardFault_Handler|MemManage_Handler|'
                            r'BusFault_Handler|UsageFault_Handler|'
                            r'FaultISR|fault_handler)\b')),
    ("assert",   re.compile(r'\b(configASSERT|Xil_AssertVoid|Xil_AssertNonvoid|'
                            r'assert|__ASSERT)\s*\(')),
]

# Cross-module include graph — track both quote- and angle-includes so we
# can also list which 3rd-party / SDK headers are pulled in.
_INC_QUOTE_RE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
_INC_ANGLE_RE = re.compile(r'^\s*#\s*include\s*<([^>]+)>', re.MULTILINE)

# Global variable read/write graph.  We look for `extern <type> <name>;`
# declarations in headers to determine which symbols are intended to be
# shared, then scan every .c for read/write occurrences.
_EXTERN_DECL_RE = re.compile(
    r'^\s*extern\s+(?:volatile\s+|const\s+){0,2}'
    r'(?:[A-Za-z_]\w*\s+){1,4}'
    r'(\*?\s*[A-Za-z_]\w+)\s*'
    r'(?:\[[^\]]*\])?\s*;',
    re.MULTILINE)
_ASSIGN_RE_TPL = r'\b{name}\s*(?:\[[^\]]*\])?\s*(?:=(?!=)|\+=|-=|\|=|&=|\^=|\*=|/=|%=)'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_read(fp: Path) -> str:
    try:
        return fp.read_text(errors="replace")
    except Exception as exc:                       # noqa: BLE001
        log.debug("gitnexus: cannot read %s: %s", fp, exc)
        return ""


def _iter_code_files(repo_dir: Path) -> list[Path]:
    return sorted(
        p for p in repo_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _CODE_EXTS
    )


def _iter_script_files(repo_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(repo_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in _SCRIPT_EXTS:
            out.append(p)
            continue
        if p.name.lower() in _SCRIPT_NAMES:
            out.append(p)
    return out


def _rel(repo_dir: Path, fp: Path) -> str:
    try:
        return str(fp.relative_to(repo_dir))
    except ValueError:
        return fp.name


def _line_of(content: str, offset: int) -> int:
    return content.count("\n", 0, offset) + 1


def _strip_comments(text: str) -> str:
    """Strip C block and line comments so identifier scans aren't poisoned
    by commented-out code."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _trim(items, n: int, *, full: bool) -> tuple[list, int]:
    """Return ``(visible, hidden_count)`` for an iterable.

    In ``full`` mode every item is visible and ``hidden_count == 0`` so the
    caller emits no "… (+N more)" marker.  In the distilled mode the
    first ``n`` items are returned and ``hidden_count`` reflects the
    remainder for the suffix.
    """
    seq = list(items)
    if full or len(seq) <= n:
        return seq, 0
    return seq[:n], len(seq) - n


# ---------------------------------------------------------------------------
# Per-category extractors
# ---------------------------------------------------------------------------

def _extract_isr_to_task(files: list[tuple[Path, str]],
                        repo_dir: Path) -> list[str]:
    """Return human-readable lines describing ISR <-> task wiring."""
    isr_handlers: list[tuple[str, str]] = []        # (handler, "rel:line")
    bindings: list[tuple[str, str, str]] = []       # (irq_id, handler, rel)
    handler_locations: dict[str, str] = {}

    for fp, raw in files:
        text = _strip_comments(raw)
        rel = _rel(repo_dir, fp)
        for pat in _ISR_PATTERNS:
            for m in pat.finditer(text):
                handler = m.group(1)
                where = f"{rel}:{_line_of(text, m.start())}"
                isr_handlers.append((handler, where))
                handler_locations.setdefault(handler, where)
        for pat in _ISR_BIND_PATTERNS:
            for m in pat.finditer(text):
                groups = m.groups()
                irq_id = (groups[0] or "").strip() if groups else ""
                handler = (groups[1] or "").strip() if len(groups) > 1 else ""
                if not handler and irq_id:
                    handler = irq_id
                    irq_id = ""
                bindings.append((irq_id or "(unknown IRQ)", handler, rel))

    lines: list[str] = []
    if bindings:
        lines.append("### Interrupt -> ISR bindings")
        for irq_id, handler, rel in sorted(set(bindings)):
            loc = handler_locations.get(handler)
            tail = f"  [defined at {loc}]" if loc else ""
            lines.append(f"  - IRQ {irq_id} -> {handler}() in {rel}{tail}")
    if isr_handlers:
        lines.append("\n### ISR handler functions discovered")
        seen = set()
        for handler, where in sorted(isr_handlers):
            if handler in seen:
                continue
            seen.add(handler)
            lines.append(f"  - {handler}()  ({where})")
    if not lines:
        lines.append("  (no ISR handlers detected)")
    return lines


def _extract_rtos_tasks(files: list[tuple[Path, str]],
                       repo_dir: Path) -> list[str]:
    tasks: list[tuple[str, str, str, str]] = []  # (api, task_fn, name, rel)
    sync_calls: dict[str, set[str]] = defaultdict(set)
    superloops: list[str] = []

    for fp, raw in files:
        text = _strip_comments(raw)
        rel = _rel(repo_dir, fp)
        for pat in _RTOS_TASK_PATTERNS:
            for m in pat.finditer(text):
                grps = m.groups()
                api = grps[0]
                task_fn = grps[1] if len(grps) > 1 else ""
                task_name = grps[2] if len(grps) > 2 else task_fn
                tasks.append((api, task_fn, task_name or task_fn, rel))
        for pat in _RTOS_SYNC_PATTERNS:
            for m in pat.finditer(text):
                sync_calls[rel].add(m.group(1))

        main_pos = next((m.start() for p in _SUPERLOOP_PATTERNS
                         for m in p.finditer(text)), None)
        if main_pos is not None:
            body_window = text[main_pos:main_pos + 4096]
            if any(p.search(body_window) for p in _SUPERLOOP_BODY_PATTERNS):
                superloops.append(rel)

    lines: list[str] = []
    if tasks:
        lines.append("### RTOS tasks / threads")
        for api, fn, name, rel in sorted(set(tasks)):
            lines.append(
                f"  - {api}: entry={fn}() name=\"{name}\" in {rel}"
            )
    if sync_calls:
        lines.append("\n### RTOS sync primitives used (per file)")
        for rel in sorted(sync_calls):
            apis = sorted(sync_calls[rel])
            lines.append(f"  - {rel}: " + ", ".join(apis))
    if superloops:
        lines.append("\n### Bare-metal superloops detected")
        for rel in sorted(set(superloops)):
            lines.append(f"  - {rel}: main() with infinite loop body")
    if not lines:
        lines.append("  (no RTOS tasks or superloops detected)")
    return lines


def _extract_state_machines(files: list[tuple[Path, str]],
                           repo_dir: Path, *, full: bool = False) -> list[str]:
    enums: list[tuple[str, list[str], str]] = []        # (typename, members, rel)
    transitions: dict[str, set[str]] = defaultdict(set) # state-var -> assignments
    cases_per_file: dict[str, set[str]] = defaultdict(set)

    for fp, raw in files:
        text = _strip_comments(raw)
        rel = _rel(repo_dir, fp)
        for m in _STATE_ENUM_RE.finditer(text):
            members = [
                v.strip().split("=")[0].strip()
                for v in m.group(1).split(",")
                if v.strip()
            ]
            enums.append((m.group(2), members, rel))
        for m in _STATE_TRANSITION_RE.finditer(text):
            var = m.group(1)
            nxt = m.group(2)
            transitions[var].add(nxt)
        for m in _STATE_SWITCH_CASE_RE.finditer(text):
            cases_per_file[rel].add(m.group(1))

    lines: list[str] = []
    if enums:
        lines.append("### State enums")
        for typename, members, rel in enums:
            preview, hidden = _trim(members, 8, full=full)
            preview_str = ", ".join(preview)
            more = f", … (+{hidden} more)" if hidden else ""
            lines.append(f"  - {typename} in {rel}: {{ {preview_str}{more} }}")
    if transitions:
        lines.append("\n### State variable assignments (transitions)")
        for var in sorted(transitions):
            tgts_sorted = sorted(transitions[var])
            preview, hidden = _trim(tgts_sorted, 10, full=full)
            preview_str = ", ".join(preview)
            more = f", … (+{hidden} more)" if hidden else ""
            lines.append(f"  - {var} := {preview_str}{more}")
    if cases_per_file:
        lines.append("\n### State switch dispatch (cases per file)")
        for rel in sorted(cases_per_file):
            cases_sorted = sorted(cases_per_file[rel])
            preview, hidden = _trim(cases_sorted, 10, full=full)
            preview_str = ", ".join(preview)
            more = f", … (+{hidden} more)" if hidden else ""
            lines.append(f"  - {rel}: {preview_str}{more}")
    if not lines:
        lines.append("  (no state machines detected)")
    return lines


def _extract_drivers_peripherals(files: list[tuple[Path, str]],
                                repo_dir: Path, *,
                                full: bool = False) -> tuple[list[str], list[str]]:
    """Return (drivers_section_lines, hal_boundary_lines)."""
    stack_hits: dict[str, set[str]] = defaultdict(set)        # stack -> files
    periph_hits: dict[str, set[str]] = defaultdict(set)       # file -> bases
    hal_headers: set[str] = set()
    driver_files: set[str] = set()

    for fp, raw in files:
        text = _strip_comments(raw)
        rel = _rel(repo_dir, fp)
        for stack_name, pat in _COMM_STACKS:
            if pat.search(text):
                stack_hits[stack_name].add(rel)
        for pat in _PERIPH_PATTERNS:
            for m in pat.finditer(text):
                token = m.group(1) if m.groups() else m.group(0)
                periph_hits[rel].add(token)
        if fp.suffix.lower() in _HEADER_EXTS and _HAL_HEADER_HINTS.search(fp.name):
            hal_headers.add(rel)
        if _DRIVER_FILENAME_HINTS.match(fp.name):
            driver_files.add(rel)

    drv_lines: list[str] = []
    if stack_hits:
        drv_lines.append("### Communication-stack usage (per stack)")
        for stack in sorted(stack_hits):
            rels_sorted = sorted(stack_hits[stack])
            preview, hidden = _trim(rels_sorted, 8, full=full)
            preview_str = ", ".join(preview)
            more = f" (+{hidden} more)" if hidden else ""
            drv_lines.append(f"  - {stack}: {preview_str}{more}")
    if periph_hits:
        drv_lines.append("\n### Peripheral / register access (per file)")
        for rel in sorted(periph_hits):
            bases_sorted = sorted(periph_hits[rel])
            preview, hidden = _trim(bases_sorted, 8, full=full)
            preview_str = ", ".join(preview)
            more = f", … (+{hidden} more)" if hidden else ""
            drv_lines.append(f"  - {rel}: {preview_str}{more}")
    if not drv_lines:
        drv_lines.append("  (no driver/peripheral patterns detected)")

    hal_lines: list[str] = []
    if hal_headers:
        hal_lines.append("### HAL / BSP header boundary")
        for rel in sorted(hal_headers):
            hal_lines.append(f"  - {rel}")
    if driver_files:
        hal_lines.append("\n### Files matching driver naming conventions")
        for rel in sorted(driver_files):
            hal_lines.append(f"  - {rel}")
    if not hal_lines:
        hal_lines.append("  (no obvious HAL/BSP boundary detected — flat code layout)")
    return drv_lines, hal_lines


def _extract_memory_ownership(files: list[tuple[Path, str]],
                             repo_dir: Path, *,
                             full: bool = False) -> list[str]:
    globals_by_file: dict[str, list[str]] = defaultdict(list)
    alloc_hits: dict[str, set[str]] = defaultdict(set)
    free_hits: dict[str, set[str]] = defaultdict(set)
    dma_hits: set[str] = set()
    sections: dict[str, set[str]] = defaultdict(set)        # section -> files

    for fp, raw in files:
        text = _strip_comments(raw)
        rel = _rel(repo_dir, fp)
        if fp.suffix.lower() in _SOURCE_EXTS:
            for m in _GLOBAL_VAR_RE.finditer(text):
                store = (m.group("store") or "").strip()
                vname = m.group("name")
                vtype = " ".join((m.group("type") or "").split())
                arr = (m.group("arr") or "").strip()
                if vname in {"if", "while", "for", "switch", "return",
                              "sizeof", "else", "case", "do"}:
                    continue
                # crude scope filter: globals only when not indented inside
                # a function body
                line_start = text.rfind("\n", 0, m.start()) + 1
                indent = m.start() - line_start
                if indent > 0:
                    continue
                qualifier = (store + " ").strip()
                desc = f"{qualifier} {vtype} {vname}{arr}".strip()
                globals_by_file[rel].append(desc)
        for m in _MALLOC_RE.finditer(text):
            alloc_hits[rel].add(m.group(1))
        for m in _FREE_RE.finditer(text):
            free_hits[rel].add(m.group(1))
        if _DMA_RE.search(text):
            dma_hits.add(rel)
        for m in _SHARED_MEM_RE.finditer(text):
            sections[m.group(1)].add(rel)

    lines: list[str] = []
    if globals_by_file:
        lines.append("### Global / static variable definitions")
        for rel in sorted(globals_by_file):
            vars_ = globals_by_file[rel]
            preview, hidden = _trim(vars_, 10, full=full)
            for v in preview:
                lines.append(f"  - {rel}: {v}")
            if hidden:
                lines.append(f"  - {rel}: … (+{hidden} more globals)")
    if alloc_hits or free_hits:
        lines.append("\n### Heap allocation usage")
        all_files_ = set(alloc_hits) | set(free_hits)
        for rel in sorted(all_files_):
            a = sorted(alloc_hits.get(rel, set()))
            f = sorted(free_hits.get(rel, set()))
            parts: list[str] = []
            if a:
                parts.append("alloc=" + ",".join(a))
            if f:
                parts.append("free=" + ",".join(f))
            lines.append(f"  - {rel}: " + "; ".join(parts))
    if dma_hits:
        lines.append("\n### DMA / shared bus usage")
        for rel in sorted(dma_hits):
            lines.append(f"  - {rel}")
    if sections:
        lines.append("\n### Named memory sections (linker placement)")
        for sec in sorted(sections):
            rels_sorted = sorted(sections[sec])
            preview, hidden = _trim(rels_sorted, 6, full=full)
            preview_str = ", ".join(preview)
            more = f" (+{hidden} more)" if hidden else ""
            lines.append(f"  - .{sec}: {preview_str}{more}")
    if not lines:
        lines.append("  (no global / heap / DMA / section hooks detected)")
    return lines


def _extract_boot_and_update(files: list[tuple[Path, str]],
                            repo_dir: Path, *,
                            full: bool = False) -> tuple[list[str], list[str]]:
    boot_hits: dict[str, set[str]] = defaultdict(set)
    upd_hits: dict[str, set[str]] = defaultdict(set)
    for fp, raw in files:
        text = _strip_comments(raw)
        rel = _rel(repo_dir, fp)
        for pat in _BOOTLOADER_PATTERNS:
            for m in pat.finditer(text):
                boot_hits[rel].add(m.group(0))
        for pat in _FWUPDATE_PATTERNS:
            for m in pat.finditer(text):
                upd_hits[rel].add(m.group(0))

    boot_lines: list[str] = []
    if boot_hits:
        boot_lines.append("### Bootloader / firmware hand-off markers")
        for rel in sorted(boot_hits):
            toks_sorted = sorted(boot_hits[rel])
            preview, hidden = _trim(toks_sorted, 8, full=full)
            preview_str = ", ".join(preview)
            more = f", … (+{hidden} more)" if hidden else ""
            boot_lines.append(f"  - {rel}: {preview_str}{more}")
    else:
        boot_lines.append("  (no bootloader / hand-off symbols detected)")

    upd_lines: list[str] = []
    if upd_hits:
        upd_lines.append("### Firmware-update / OTA / bitstream markers")
        for rel in sorted(upd_hits):
            toks_sorted = sorted(upd_hits[rel])
            preview, hidden = _trim(toks_sorted, 8, full=full)
            preview_str = ", ".join(preview)
            more = f", … (+{hidden} more)" if hidden else ""
            upd_lines.append(f"  - {rel}: {preview_str}{more}")
    else:
        upd_lines.append("  (no firmware-update flow detected)")
    return boot_lines, upd_lines


def _extract_safety(files: list[tuple[Path, str]],
                   repo_dir: Path, *,
                   full: bool = False) -> list[str]:
    findings: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for fp, raw in files:
        text = _strip_comments(raw)
        rel = _rel(repo_dir, fp)
        for kind, pat in _SAFETY_PATTERNS:
            for m in pat.finditer(text):
                findings[rel][kind].add(m.group(0))

    if not findings:
        return ["  (no safety-critical primitives detected)"]
    lines: list[str] = []
    by_kind: dict[str, list[str]] = defaultdict(list)
    for rel, per_kind in findings.items():
        for kind, toks in per_kind.items():
            tok_preview, _ = _trim(sorted(toks), 5, full=full)
            by_kind[kind].append(f"  - {rel}: " + ", ".join(tok_preview))
    for kind in sorted(by_kind):
        lines.append(f"### Safety primitive: {kind}")
        files_for_kind = sorted(by_kind[kind])
        visible, hidden = _trim(files_for_kind, 20, full=full)
        lines.extend(visible)
        if hidden:
            lines.append(f"  - … (+{hidden} more files)")
        lines.append("")
    return lines


def _extract_include_graph(files: list[tuple[Path, str]],
                          repo_dir: Path, *,
                          full: bool = False) -> tuple[list[str], dict[str, set[str]]]:
    """Return the cross-module include graph lines plus the
    "file -> {quote includes}" map (used by script-deps section)."""
    includes_by_file: dict[str, set[str]] = {}
    reverse_quote: dict[str, set[str]] = defaultdict(set)
    angle_hits: dict[str, int] = defaultdict(int)

    for fp, raw in files:
        rel = _rel(repo_dir, fp)
        quotes = set(_INC_QUOTE_RE.findall(raw))
        angles = _INC_ANGLE_RE.findall(raw)
        includes_by_file[rel] = quotes
        for inc in quotes:
            base = Path(inc).name
            reverse_quote[base].add(rel)
        for ang in angles:
            angle_hits[ang] += 1

    lines: list[str] = []
    if includes_by_file:
        lines.append("### Quote includes per file (project-local dependencies)")
        for rel in sorted(includes_by_file):
            quotes_sorted = sorted(includes_by_file[rel])
            if not quotes_sorted:
                continue
            preview, hidden = _trim(quotes_sorted, 8, full=full)
            preview_str = ", ".join(preview)
            more = f", … (+{hidden} more)" if hidden else ""
            lines.append(f"  - {rel}: {preview_str}{more}")
    if reverse_quote:
        lines.append("\n### Reverse dependency (header -> consumers)")
        for hdr in sorted(reverse_quote):
            consumers = sorted(reverse_quote[hdr])
            if not full and len(consumers) < 2:
                continue                       # uninteresting (single consumer)
            preview, hidden = _trim(consumers, 6, full=full)
            preview_str = ", ".join(preview)
            more = f", … (+{hidden} more)" if hidden else ""
            lines.append(f"  - {hdr} included by: {preview_str}{more}")
    if angle_hits:
        lines.append("\n### Most-used angle includes (SDK / libc)")
        ordered = sorted(angle_hits.items(), key=lambda kv: (-kv[1], kv[0]))
        top, hidden = _trim(ordered, 15, full=full)
        for hdr, n in top:
            lines.append(f"  - <{hdr}>: {n} files")
        if hidden:
            lines.append(f"  - … (+{hidden} more angle includes)")
    if not lines:
        lines.append("  (no #include directives discovered)")
    return lines, includes_by_file


def _extract_global_var_graph(files: list[tuple[Path, str]],
                             repo_dir: Path, *,
                             full: bool = False) -> list[str]:
    """Resolve every `extern <type> <name>;` in headers and list the
    `.c` files that read or write each shared symbol."""
    externs: list[tuple[str, str]] = []         # (name, header_rel)
    file_text: dict[str, str] = {}

    for fp, raw in files:
        rel = _rel(repo_dir, fp)
        stripped = _strip_comments(raw)
        file_text[rel] = stripped
        if fp.suffix.lower() in _HEADER_EXTS:
            for m in _EXTERN_DECL_RE.finditer(stripped):
                name_raw = m.group(1).strip().lstrip("*").strip()
                if not name_raw or not re.match(r"[A-Za-z_]\w*$", name_raw):
                    continue
                externs.append((name_raw, rel))

    if not externs:
        return ["  (no `extern` shared symbols discovered in headers)"]

    name_to_header: dict[str, str] = {}
    for name, header in externs:
        name_to_header.setdefault(name, header)

    writers: dict[str, set[str]] = defaultdict(set)
    readers: dict[str, set[str]] = defaultdict(set)

    for name, header in externs:
        assign_re = re.compile(_ASSIGN_RE_TPL.format(name=re.escape(name)))
        read_re = re.compile(rf"\b{re.escape(name)}\b")
        for rel, text in file_text.items():
            if Path(rel).suffix.lower() not in _SOURCE_EXTS:
                continue
            if assign_re.search(text):
                writers[name].add(rel)
            elif read_re.search(text):
                readers[name].add(rel)

    lines: list[str] = ["### Shared (extern) variables — read/write graph"]
    seen_names: set[str] = set()
    for name in sorted(name_to_header):
        if name in seen_names:
            continue
        seen_names.add(name)
        header = name_to_header[name]
        w = sorted(writers.get(name, set()))
        r = sorted(readers.get(name, set()))
        if not w and not r:
            lines.append(f"  - {name} (declared in {header}): no usages found")
            continue
        wpart = "writes=" + (", ".join(w) if w else "(none)")
        rpart = "reads=" + (", ".join(r) if r else "(none)")
        lines.append(f"  - {name} (declared in {header}): {wpart}; {rpart}")
        if not full and len(lines) > 60:
            lines.append("  - … (extern read/write graph truncated)")
            break
    return lines


def _extract_script_dependencies(repo_dir: Path,
                                quote_includes: dict[str, set[str]], *,
                                full: bool = False) -> list[str]:
    scripts = _iter_script_files(repo_dir)
    if not scripts and not quote_includes:
        return ["  (no build / packaging scripts detected)"]

    lines: list[str] = []
    if scripts:
        lines.append("### Build / packaging scripts in repository")
        visible, hidden = _trim(scripts, 40, full=full)
        for fp in visible:
            try:
                size = fp.stat().st_size
            except OSError:
                size = -1
            lines.append(f"  - {_rel(repo_dir, fp)} ({size} bytes)")
        if hidden:
            lines.append(f"  - … (+{hidden} more)")

    # Cross-references: which scripts mention which source files?
    if scripts:
        refs: dict[str, set[str]] = defaultdict(set)
        code_basenames = {Path(rel).name
                          for rel in quote_includes
                          for _ in [None]}
        # also collect .c basenames found in repo
        for cf in _iter_code_files(repo_dir):
            code_basenames.add(cf.name)
        if code_basenames:
            for sp in scripts:
                text = _safe_read(sp)
                if not text:
                    continue
                for base in code_basenames:
                    if base and base in text:
                        refs[_rel(repo_dir, sp)].add(base)
        if refs:
            lines.append("\n### Script -> source file references")
            for sp in sorted(refs):
                bases_sorted = sorted(refs[sp])
                preview, hidden = _trim(bases_sorted, 8, full=full)
                preview_str = ", ".join(preview)
                more = f", … (+{hidden} more)" if hidden else ""
                lines.append(f"  - {sp}: {preview_str}{more}")
    return lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_gitnexus_report(
    *,
    repo_dir: Path | None,
    code_dir: Path | None = None,
    max_chars: int = 16_000,
    full: bool = False,
) -> str:
    """Produce the GitNexus report for *repo_dir* (and any uploaded code).

    Both directories are optional — when neither contains code the report
    is a short notice that GitNexus had nothing to chew on.

    Two flavours are supported:

      * ``full=False`` (default) — the **distilled** view used by the
        downstream LLM prompts.  Per-section lists are capped (e.g. 8/10
        entries with a ``(+N more)`` suffix) and the final string is
        clipped to ``max_chars`` with an explicit truncation marker.
        This keeps the prompt budget under control.

      * ``full=True`` — the **untruncated** view written to
        ``gitnexus_report.txt`` and exposed in the downloadable ZIP for
        QA purposes.  Every extractor emits the complete set of findings
        with no per-section caps, and ``max_chars`` is intentionally
        ignored so reviewers see the entire evidence trail.
    """
    pairs: list[tuple[Path, str]] = []
    if repo_dir is not None and repo_dir.exists():
        for fp in _iter_code_files(repo_dir):
            pairs.append((fp, _safe_read(fp)))
    if code_dir is not None and code_dir.exists():
        for fp in sorted(code_dir.rglob("*")):
            if fp.is_file() and fp.suffix.lower() in _CODE_EXTS:
                # Avoid double-counting files that are also present
                # inside the repo by basename: code_dir entries always
                # take priority since they are the *target* of the
                # refactor.
                rel = fp.name
                pairs = [(p, t) for p, t in pairs
                         if _rel(repo_dir or fp.parent, p) != rel]
                pairs.append((fp, _safe_read(fp)))

    flavour = "FULL (untruncated, QA download)" if full else "DISTILLED (LLM context)"
    header = [
        "=" * 65,
        f"GITNEXUS — CODEBASE UNDERSTANDING REPORT [{flavour}]",
        "=" * 65,
        "",
        "This report is auto-generated immediately after the ICD",
        "change_spec.  It captures embedded-systems-specific relationships",
        "extracted directly from the uploaded repository and source",
        "scripts via deterministic pattern matching.  The downstream",
        "code-generation, verification, per-file compile, sandbox build",
        "and regeneration prompts receive the DISTILLED flavour of this",
        "report as additional ground-truth context alongside the ICD",
        "change_spec and repo_knowledge.  The FULL flavour is kept on",
        "disk (gitnexus_report.txt) and shipped in the downloadable ZIP",
        "so QA can audit the complete extraction without prompt-budget",
        "clipping.",
        "",
    ]

    if not pairs:
        return "\n".join(header + [
            "(no .c / .h files were available to GitNexus)",
            "",
        ])

    anchor_repo = repo_dir if (repo_dir and repo_dir.exists()) else (
        code_dir if code_dir else Path(".")
    )

    isr_lines = _extract_isr_to_task(pairs, anchor_repo)
    rtos_lines = _extract_rtos_tasks(pairs, anchor_repo)
    state_lines = _extract_state_machines(pairs, anchor_repo, full=full)
    drv_lines, hal_lines = _extract_drivers_peripherals(
        pairs, anchor_repo, full=full,
    )
    mem_lines = _extract_memory_ownership(pairs, anchor_repo, full=full)
    boot_lines, upd_lines = _extract_boot_and_update(
        pairs, anchor_repo, full=full,
    )
    safety_lines = _extract_safety(pairs, anchor_repo, full=full)
    inc_lines, quote_includes = _extract_include_graph(
        pairs, anchor_repo, full=full,
    )
    extern_lines = _extract_global_var_graph(pairs, anchor_repo, full=full)
    script_lines = _extract_script_dependencies(
        anchor_repo, quote_includes, full=full,
    ) if (repo_dir and repo_dir.exists()) else [
        "  (no repository scripts uploaded — GitNexus saw uploaded sources only)"
    ]

    sections: list[tuple[str, Iterable[str]]] = [
        ("1. ISR -> task wiring",                       isr_lines),
        ("2. RTOS / superloop task interactions",       rtos_lines),
        ("3. State machine transitions",                state_lines),
        ("4. Drivers / peripherals & comm stacks",      drv_lines),
        ("5. Hardware abstraction (HAL / BSP) boundary", hal_lines),
        ("6. Memory ownership relationships",           mem_lines),
        ("7. Bootloader -> firmware hand-off",          boot_lines),
        ("8. Firmware update / OTA flow",               upd_lines),
        ("9. Safety-critical execution chains",         safety_lines),
        ("10. Cross-module dependencies (#include graph)", inc_lines),
        ("11. Global variable read/write graph",        extern_lines),
        ("12. Script dependencies",                     script_lines),
    ]

    parts: list[str] = list(header)
    parts.append(f"## Files inspected: {len(pairs)} C/C++ source/header files\n")
    for title, body in sections:
        parts.append("-" * 65)
        parts.append(title)
        parts.append("-" * 65)
        for line in body:
            parts.append(line)
        parts.append("")

    result = "\n".join(parts).rstrip() + "\n"
    # ``full`` reports are never length-clipped — that flavour is the
    # canonical record for the downloadable QA report.  Only the
    # distilled flavour is bounded by ``max_chars``.
    if not full and len(result) > max_chars:
        clip = max_chars - 200
        result = (
            result[:clip]
            + "\n\n[... GITNEXUS report truncated — full report saved to "
              "gitnexus_report.txt ...]\n"
        )
    return result


__all__ = ["build_gitnexus_report"]
