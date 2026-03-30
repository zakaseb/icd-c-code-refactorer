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

# Detect hard GPU fault state early. In this state CUDA init returns
# "unknown error" until a full host reboot.
GPU_RECOVERY_ACTION=$(nvidia-smi -q 2>/dev/null | awk -F': ' '/GPU Recovery Action/ {print $2; exit}')
if [ "$GPU_RECOVERY_ACTION" = "Reboot" ]; then
  echo "ERROR: NVIDIA driver reports 'GPU Recovery Action: Reboot'."
  echo "CUDA initialization will fail (host + container) until the machine is rebooted."
  echo "Action required: reboot the host, then rerun ./run_web.sh."
  exit 1
fi

# GPU preflight: fail fast if CUDA device initialization is unavailable in containers.
# Retry up to 3 times with increasing delays — CUDA can fail transiently after a
# previous container exits without cleaning up the driver state.
GPU_OK=0
MAX_GPU_ATTEMPTS=3
for gpu_attempt in $(seq 1 $MAX_GPU_ATTEMPTS); do
  echo "Running GPU preflight (attempt $gpu_attempt/$MAX_GPU_ATTEMPTS)..."
  GPU_PREFLIGHT_OUT=$(docker run --rm --gpus all --entrypoint /bin/bash \
    icd-c-code-refactorer:llama.cpp -lc 'export LD_LIBRARY_PATH="/app:${LD_LIBRARY_PATH}"; /app/llama-server --list-devices' 2>&1 || true)
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
  echo "Host diagnostic:"
  python3 - <<'PY'
import ctypes
try:
    c = ctypes.CDLL("libcuda.so.1")
    print("  host cuInit rc =", c.cuInit(0))
except Exception as e:
    print("  host cuInit probe failed:", e)
PY
  echo "  nvidia-container-cli version: $(nvidia-container-cli -V 2>/dev/null | awk '/cli-version:/ {print $2}')"
  echo
  echo "Possible fixes:"
  echo "  1. If 'GPU Recovery Action: Reboot' appears in 'nvidia-smi -q', reboot the host."
  echo "  2. Update nvidia-container-toolkit to a recent version (>= 1.16 recommended)."
  echo "  3. Restart docker and nvidia-persistenced:"
  echo "       sudo systemctl restart nvidia-persistenced docker"
  echo "  4. Re-test with:"
  echo "       docker run --rm --gpus all --entrypoint /bin/bash nvidia/cuda:12.6.3-devel-ubuntu22.04 -lc 'cat > /tmp/t.cu <<\"EOF\""
  echo "       #include <cstdio>"
  echo "       #include <cuda_runtime.h>"
  echo "       int main(){int n=0; auto e=cudaGetDeviceCount(&n); std::printf(\"%d %s\\n\", n, cudaGetErrorString(e)); return e!=cudaSuccess;}"
  echo "       EOF"
  echo "       nvcc /tmp/t.cu -o /tmp/t && /tmp/t'"
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
