#!/bin/bash
# One-time patch for root-owned scripts/serve_web.sh: pass agentic defaults
# into the container so host overrides work (AGENTIC_PIPELINE=0 ./scripts/serve_web.sh).
# The Docker image also bakes in AGENTIC_PIPELINE=1 after rebuild; this patch
# is only needed for explicit -e overrides from the host.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$SCRIPT_DIR/serve_web.sh"
if [ ! -f "$TARGET" ]; then
  echo "ERROR: $TARGET not found" >&2
  exit 1
fi
if grep -q 'AGENTIC_PIPELINE=' "$TARGET"; then
  echo "serve_web.sh already includes AGENTIC_PIPELINE — nothing to do."
  exit 0
fi
python3 <<PY
from pathlib import Path
p = Path("$TARGET")
text = p.read_text()
needle = "  -e LLAMA_ARG_N_PREDICT=32768 \\\n"
insert = (
    "  -e LLAMA_ARG_N_PREDICT=32768 \\\n"
    "  -e AGENTIC_PIPELINE=\"\${AGENTIC_PIPELINE:-1}\" \\\n"
    "  -e AGENTIC_LLM_BACKEND=\"\${AGENTIC_LLM_BACKEND:-auto}\" \\\n"
)
if needle not in text:
    raise SystemExit("Could not find insertion point in serve_web.sh")
p.write_text(text.replace(needle, insert, 1))
print("Patched $TARGET")
PY
