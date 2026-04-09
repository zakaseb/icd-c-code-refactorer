# Dummy Inputs for Testing

Minimal test inputs that exercise the full pipeline including sandbox build.

## Files

| File | Upload Zone | Description |
|---|---|---|
| `comms.c`, `comms.h` | Source Code | ICD v1.0 communication module |
| `source_icd_v1.pdf` | Source ICD | 1-page ICD v1.0 spec (structs, enums, message format) |
| `target_icd_v2.pdf` | Target ICD | 2-page ICD v2.0 spec (adds priority, CRC-16, wider fields) |
| `repo.zip` | Repository ZIP | Buildable repo: `Makefile` + `src/comms.c` + `src/comms.h` + `src/main.c` |

## Key differences between v1.0 and v2.0

- `seq_num` widens from `uint8` to `uint16`
- New `priority` field added to message header
- `MAX_PAYLOAD_LEN` increases from 64 to 128
- Checksum changes from additive to CRC-16/CCITT
- `StatusReport.uptime_sec` widens from `uint16` to `uint32`
- `StatusReport.temperature` widens from `int8` to `int16`
- New fields: `voltage_mv`, `error_count`
- New `MAINTENANCE` state added to `SystemState` enum
- New message IDs: `MSG_ID_ACK` (0x04), `MSG_ID_DIAGNOSTIC` (0x05)
- Timing intervals changed

## Regenerating

If you modify the ICD text or repo sources, re-run:

```bash
python3 dummy_input/generate_inputs.py
```

Requires PyMuPDF (`pip install PyMuPDF`).
