#!/usr/bin/env python3
"""Generate dummy ICD PDFs and repository ZIP for testing the refactorer."""

import zipfile
import os
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not installed locally, trying fpdf2 fallback...")
    fitz = None

SCRIPT_DIR = Path(__file__).parent
REPO_STAGING = SCRIPT_DIR / "_repo_staging"


def create_pdf_pymupdf(path: Path, text: str):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    tw = fitz.TextWriter(page.rect)
    font = fitz.Font("helv")
    y = 72
    for line in text.split("\n"):
        if y > 720:
            tw.write_text(page)
            page = doc.new_page(width=612, height=792)
            tw = fitz.TextWriter(page.rect)
            y = 72
        tw.append((72, y), line, font=font, fontsize=10)
        y += 14
    tw.write_text(page)
    doc.save(str(path))
    doc.close()


def create_pdf_fpdf(path: Path, text: str):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    for line in text.split("\n"):
        pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


SOURCE_ICD_TEXT = """INTERFACE CONTROL DOCUMENT
System Communication Protocol
Version 1.0

1. SCOPE
This ICD defines the message formats and communication protocol
between subsystems for the embedded platform (Version 1.0).

2. MESSAGE FORMAT
All messages use the following structure:
  - msg_id     : uint8   (message identifier)
  - seq_num    : uint8   (sequence counter 0-255)
  - payload_len: uint16  (length of payload in bytes)
  - payload    : uint8[] (up to 64 bytes)
  - checksum   : uint16  (sum of all preceding fields)

3. MESSAGE IDENTIFIERS
  MSG_ID_HEARTBEAT = 0x01  Periodic heartbeat
  MSG_ID_STATUS    = 0x02  System status report
  MSG_ID_COMMAND   = 0x03  Command message

4. STATUS REPORT PAYLOAD
The status report contains:
  - system_id   : uint8       (originating system)
  - state       : enum        (IDLE=0, ACTIVE=1, ERROR=2)
  - uptime_sec  : uint16      (seconds since boot)
  - temperature : int8        (degrees Celsius)

5. CHECKSUM
Simple additive checksum: sum of msg_id + seq_num + payload_len
  + all payload bytes, truncated to uint16.

6. TIMING
  Heartbeat interval: 1000 ms
  Status report interval: 5000 ms
  Command timeout: 2000 ms
"""

TARGET_ICD_TEXT = """INTERFACE CONTROL DOCUMENT
System Communication Protocol
Version 2.0

1. SCOPE
This ICD defines the message formats and communication protocol
between subsystems for the embedded platform (Version 2.0).
This version adds priority levels, CRC-16 checksums, and extended
status reporting.

2. MESSAGE FORMAT
All messages use the following structure:
  - msg_id     : uint8   (message identifier)
  - seq_num    : uint16  (CHANGED: sequence counter 0-65535)
  - priority   : uint8   (NEW: 0=low, 1=normal, 2=high, 3=critical)
  - payload_len: uint16  (length of payload in bytes)
  - payload    : uint8[] (up to 128 bytes, CHANGED from 64)
  - crc16      : uint16  (CHANGED: CRC-16/CCITT replaces additive checksum)

3. MESSAGE IDENTIFIERS
  MSG_ID_HEARTBEAT  = 0x01  Periodic heartbeat
  MSG_ID_STATUS     = 0x02  System status report
  MSG_ID_COMMAND    = 0x03  Command message
  MSG_ID_ACK        = 0x04  (NEW) Acknowledgment message
  MSG_ID_DIAGNOSTIC = 0x05  (NEW) Diagnostic data dump

4. STATUS REPORT PAYLOAD
The status report contains:
  - system_id    : uint8       (originating system)
  - state        : enum        (IDLE=0, ACTIVE=1, ERROR=2, MAINTENANCE=3)
                                (CHANGED: added MAINTENANCE state)
  - uptime_sec   : uint32      (CHANGED: uint32 to support longer uptime)
  - temperature  : int16       (CHANGED: int16 for 0.1 degree resolution)
  - voltage_mv   : uint16      (NEW: supply voltage in millivolts)
  - error_count  : uint16      (NEW: cumulative error counter)

5. CHECKSUM
CRC-16/CCITT polynomial 0x1021, initial value 0xFFFF.
Computed over: msg_id, seq_num (2 bytes), priority, payload_len,
  and all payload bytes.

6. TIMING
  Heartbeat interval: 500 ms   (CHANGED from 1000 ms)
  Status report interval: 2000 ms  (CHANGED from 5000 ms)
  Command timeout: 1000 ms     (CHANGED from 2000 ms)
  ACK timeout: 500 ms          (NEW)

7. ACKNOWLEDGMENTS (NEW)
All commands (MSG_ID_COMMAND) must be acknowledged with MSG_ID_ACK
within the ACK timeout. The ACK payload contains the original
msg_id and seq_num of the command being acknowledged.
"""


def create_repo_zip(out_path: Path):
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(REPO_STAGING):
            for fname in files:
                fpath = Path(root) / fname
                arcname = str(fpath.relative_to(REPO_STAGING))
                zf.write(fpath, arcname)
    print(f"Created {out_path}  ({out_path.stat().st_size} bytes)")


def main():
    create_pdf = create_pdf_pymupdf if fitz else create_pdf_fpdf

    src_pdf = SCRIPT_DIR / "source_icd_v1.pdf"
    tgt_pdf = SCRIPT_DIR / "target_icd_v2.pdf"
    repo_zip = SCRIPT_DIR / "repo.zip"

    create_pdf(src_pdf, SOURCE_ICD_TEXT)
    print(f"Created {src_pdf}  ({src_pdf.stat().st_size} bytes)")

    create_pdf(tgt_pdf, TARGET_ICD_TEXT)
    print(f"Created {tgt_pdf}  ({tgt_pdf.stat().st_size} bytes)")

    create_repo_zip(repo_zip)

    print("\nDummy inputs ready in", SCRIPT_DIR)
    print("  Source code:  comms.c, comms.h")
    print("  Source ICD:   source_icd_v1.pdf")
    print("  Target ICD:   target_icd_v2.pdf")
    print("  Repository:   repo.zip  (Makefile + src/comms.c + src/comms.h + src/main.c)")


if __name__ == "__main__":
    main()
