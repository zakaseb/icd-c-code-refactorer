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
# Retry up to 3 times with increasing delays — CUDA can fail transiently after a
# previous container exits without cleaning up the driver state.
GPU_OK=0
MAX_GPU_ATTEMPTS=3
for gpu_attempt in $(seq 1 $MAX_GPU_ATTEMPTS); do
  echo "Running GPU preflight (attempt $gpu_attempt/$MAX_GPU_ATTEMPTS)..."
  GPU_PREFLIGHT_OUT=$(docker run --rm --gpus all --entrypoint /app/llama-server \
    icd-c-code-refactorer:llama.cpp --list-devices 2>&1 || true)
  if printf '%s\n' "$GPU_PREFLIGHT_OUT" | python3 -c '
import re, sys
txt = sys.stdin.read()
device_lines = re.findall(r"^\s*(?:\d+|CUDA\d+|Device\s+\d+)\s*:\s+.+(?:NVIDIA|AMD|GPU).+$", txt, flags=re.MULTILINE | re.IGNORECASE)
sys.exit(0 if device_lines else 1)
'
  then
    GPU_OK=1
    echo "GPU preflight passed."
    break
  fi

  if [ "$gpu_attempt" -lt "$MAX_GPU_ATTEMPTS" ]; then
    WAIT=$((gpu_attempt * 3))
    echo "GPU preflight attempt $gpu_attempt failed. Trying nvidia-smi reset and waiting ${WAIT}s..."
    nvidia-smi 2>/dev/null || true
    sleep "$WAIT"
  fi
done

if [ "$GPU_OK" -ne 1 ]; then
  echo "WARNING: GPU preflight failed after $MAX_GPU_ATTEMPTS attempts."
  echo "Preflight output:"
  echo "$GPU_PREFLIGHT_OUT"
  echo
  echo "Possible fixes:"
  echo "  1. Run: sudo nvidia-smi --gpu-reset"
  echo "  2. Restart nvidia-persistenced: sudo systemctl restart nvidia-persistenced"
  echo "  3. Check: nvidia-container-toolkit / driver / docker integration"
  echo
  read -r -p "Continue anyway in CPU-only mode? (y/N) " REPLY
  if [ "$REPLY" != "y" ] && [ "$REPLY" != "Y" ]; then
    echo "Aborting. Fix GPU access, then rerun ./run_web.sh."
    exit 1
  fi
  echo "Continuing without GPU acceleration (inference will be slow)."
fi

docker run -ti --rm --name icd-c-code-refactorer --network=host --gpus all \
  -e LLAMA_ARG_THREADS="$LLAMA_THREADS" \
  -e LLAMA_ARG_CTX_SIZE=32768 \
  -e LLAMA_ARG_N_PREDICT=32768 \
  -v "$SCRIPT_DIR/models":/home/developer/models \
  -v "$SCRIPT_DIR/workspace":/home/developer/workspace \
  icd-c-code-refactorer:llama.cpp
