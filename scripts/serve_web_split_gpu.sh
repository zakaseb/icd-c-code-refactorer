#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE="icd-c-code-refactorer:llama.cpp"
LLAMA_CONTAINER="icd-c-code-refactorer-llama"
WEB_CONTAINER="icd-c-code-refactorer"

if [ ! -d "$PROJECT_ROOT/models" ]; then
  echo "ERROR: models/ directory not found at $PROJECT_ROOT/models"
  echo "Create a symlink to your model cache, e.g.:"
  echo "  ln -s /path/to/your/models $PROJECT_ROOT/models"
  exit 1
fi

mkdir -p "$PROJECT_ROOT/workspace"

WEB_PORT=8081
LITELLM_PORT=4000
LLAMA_PORT=24000
for port in "$WEB_PORT" "$LITELLM_PORT" "$LLAMA_PORT"; do
  if ss -tlnp 2>/dev/null | grep -q ":$port" || netstat -tlnp 2>/dev/null | grep -q ":$port"; then
    echo "ERROR: Port $port is already in use."
    exit 1
  fi
done

HOST_CPU=$(nproc 2>/dev/null || echo 4)
if [ "$HOST_CPU" -gt 2 ]; then
  LLAMA_THREADS=$((HOST_CPU - 2))
else
  LLAMA_THREADS=1
fi

cleanup() {
  docker rm -f "$WEB_CONTAINER" >/dev/null 2>&1 || true
  docker rm -f "$LLAMA_CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Starting GPU container ($LLAMA_CONTAINER) for llama-server..."
docker rm -f "$LLAMA_CONTAINER" >/dev/null 2>&1 || true
docker run -d --rm --name "$LLAMA_CONTAINER" --network=host --gpus "device=0" \
  -e LLAMA_ARG_THREADS="$LLAMA_THREADS" \
  -e LLAMA_ARG_CTX_SIZE=32768 \
  -e LLAMA_ARG_N_PREDICT=32768 \
  -v "$PROJECT_ROOT/models":/home/developer/models \
  -v "$PROJECT_ROOT/workspace":/home/developer/workspace \
  --entrypoint /bin/bash \
  "$IMAGE" -lc '
set -e
cd "$HOME"
source /opt/venv/bin/activate
python3 hf_download.py
cd /app
exec ./llama-server --prio 0 \
  --n-gpu-layers "${LLAMA_ARG_N_GPU_LAYERS:-auto}" \
  --split-mode "${LLAMA_ARG_SPLIT_MODE:-none}" \
  --temp "${LLAMA_SAMPLING_TEMPERATURE:-0.7}" \
  --min-p "${LLAMA_SAMPLING_MIN_P:-0}" \
  --top-p "${LLAMA_SAMPLING_TOP_P:-0.80}" \
  --top-k "${LLAMA_SAMPLING_TOP_K:-20}" \
  --repeat-penalty "${LLAMA_SAMPLING_REPETITION_PENALTY:-1.05}" \
  --chat-template-file "${HF_CHAT_TEMPLATE}"
'

echo "Waiting for llama-server to become ready on :$LLAMA_PORT ..."
READY=0
for i in $(seq 1 150); do
  if curl -s -H "Authorization: Bearer ${OPENAI_API_KEY:-sk-1234-miaw}" "http://127.0.0.1:${LLAMA_PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done
if [ "$READY" -ne 1 ]; then
  echo "ERROR: llama-server did not become ready in time."
  echo "Last logs from $LLAMA_CONTAINER:"
  docker logs --tail 200 "$LLAMA_CONTAINER" || true
  exit 1
fi

echo "Starting CPU-only container ($WEB_CONTAINER) for LiteLLM + web UI..."
echo "Once ready, open http://localhost:$WEB_PORT in your browser."
docker rm -f "$WEB_CONTAINER" >/dev/null 2>&1 || true

# Explicitly hide GPUs from this container so its processes never create CUDA contexts.
docker run -ti --rm --name "$WEB_CONTAINER" --network=host \
  -e CUDA_VISIBLE_DEVICES= \
  -e NVIDIA_VISIBLE_DEVICES=void \
  -e NVIDIA_DRIVER_CAPABILITIES= \
  -e AGENTIC_PIPELINE="${AGENTIC_PIPELINE:-1}" \
  -e AGENTIC_LLM_BACKEND="${AGENTIC_LLM_BACKEND:-auto}" \
  -v "$PROJECT_ROOT/workspace":/home/developer/workspace \
  --entrypoint /bin/bash \
  "$IMAGE" -lc '
set -e
source /opt/venv/bin/activate
tmux new -s icd -d
tmux rename-window -t icd web
tmux send-keys -t icd "litellm --model $ANTHROPIC_MODEL --temperature ${LLAMA_SAMPLING_TEMPERATURE:-0.7} --drop_params" C-m
tmux split-window -h -t icd
tmux send-keys -t icd "cd /home/developer/webapp && python3 -m uvicorn app:app --host 0.0.0.0 --port 8081 --app-dir /home/developer/webapp" C-m
tmux select-layout tiled
tmux attach -t icd
'
