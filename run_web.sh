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
# --gpus all: expose every NVIDIA GPU. LLAMA_ARG_N_GPU_LAYERS=all is set in the Dockerfile
# for full layer offload; add -e LLAMA_ARG_N_GPU_LAYERS=N here only if you must cap VRAM use.
docker run -ti --rm --name icd-c-code-refactorer --network=host --gpus all \
  -e LLAMA_ARG_CTX_SIZE=32768 \
  -e LLAMA_ARG_N_PREDICT=32768 \
  -v "$SCRIPT_DIR/models":/home/developer/models \
  -v "$SCRIPT_DIR/workspace":/home/developer/workspace \
  icd-c-code-refactorer:llama.cpp
