#!/usr/bin/env bash
# Log GPU and system memory while exercising the web UI (run in a second terminal).
# On unified-memory (e.g. DGX Spark), heavy GPU + browser can contend for the same
# physical pool — use this to correlate tab lag/crashes with memory pressure.
set -euo pipefail
LOG="${1:-/tmp/icd_refactorer_ui_monitor.log}"
INTERVAL="${2:-3}"
echo "Appending to $LOG every ${INTERVAL}s (Ctrl+C to stop)"
{
  echo "==== start $(date -Is) host=$(hostname) ===="
  uname -a
} >>"$LOG"
while true; do
  {
    echo "==== $(date -Is) ===="
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || echo "nvidia-smi: not available"
    command -v tegrastats >/dev/null 2>&1 && timeout 2 tegrastats --interval 1000 2>/dev/null | tail -1 || true
    free -h | head -4
  } >>"$LOG" 2>&1
  sleep "$INTERVAL"
done
