#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$SCRIPT_DIR/models" ]; then
  echo "ERROR: models/ directory not found at $SCRIPT_DIR/models"
  echo "Create a symlink to your model cache, e.g.:"
  echo "  ln -s /path/to/qwen-claude-code-sw2/models $SCRIPT_DIR/models"
  exit 1
fi

mkdir -p "$SCRIPT_DIR/workspace"

WEB_PORT=8081
if ss -tlnp 2>/dev/null | grep -q ":$WEB_PORT" || netstat -tlnp 2>/dev/null | grep -q ":$WEB_PORT"; then
  echo "ERROR: Port $WEB_PORT is already in use. Stop the conflicting process or change WEB_PORT in this script."
  exit 1
fi

echo "Starting container (models: $SCRIPT_DIR/models)."
echo "Once ready, open http://localhost:$WEB_PORT in your browser."

# Reserve CPU cores for the OS and other apps (llama still uses the GPU for inference).
HOST_CPU=$(nproc 2>/dev/null || echo 4)
if [ "$HOST_CPU" -gt 2 ]; then
  LLAMA_THREADS=$((HOST_CPU - 2))
else
  LLAMA_THREADS=1
fi
echo "llama-server CPU threads: $LLAMA_THREADS (host logical CPUs: $HOST_CPU)"

# GPU preflight: fail fast if CUDA device initialization is unavailable in containers.
echo "Running GPU preflight (llama-server --list-devices)..."
GPU_PREFLIGHT_OUT=$(docker run --rm --gpus all --entrypoint /app/llama-server \
  icd-c-code-refactorer:llama.cpp --list-devices 2>&1 || true)
if printf '%s\n' "$GPU_PREFLIGHT_OUT" | python3 -c '
import re
import sys
txt = sys.stdin.read()
# Device lines generally look like:
#   0: NVIDIA RTX ... (CUDA)
device_lines = re.findall(r"^\s*\d+\s*:\s+.+$", txt, flags=re.MULTILINE)
sys.exit(0 if device_lines else 1)
'
then
  echo "GPU preflight passed."
else
  echo "ERROR: GPU preflight failed. llama.cpp could not initialize a usable CUDA device."
  echo "Preflight output:"
  echo "$GPU_PREFLIGHT_OUT"
  echo
  echo "Fix host GPU container runtime first (nvidia-container-toolkit / driver / docker integration),"
  echo "then rerun ./run_web.sh. Starting anyway would cause UI LLM 'connection refused' errors."
  exit 1
fi

docker run -ti --rm --name icd-c-code-refactorer --network=host --gpus all \
  -e LLAMA_ARG_THREADS="$LLAMA_THREADS" \
  -e LLAMA_ARG_CTX_SIZE=32768 \
  -e LLAMA_ARG_N_PREDICT=32768 \
  -v "$SCRIPT_DIR/models":/home/developer/models \
  -v "$SCRIPT_DIR/workspace":/home/developer/workspace \
  icd-c-code-refactorer:llama.cpp
