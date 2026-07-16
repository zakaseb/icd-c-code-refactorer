SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

HOST_CPU=$(nproc 2>/dev/null || echo 4)
if [ "$HOST_CPU" -gt 2 ]; then
  LLAMA_THREADS=$((HOST_CPU - 2))
else
  LLAMA_THREADS=1
fi

docker run -ti --rm --name icd-c-code-refactorer --network=host --gpus "device=0" \
  -e LLAMA_ARG_THREADS="$LLAMA_THREADS" \
  -e LLAMA_ARG_CTX_SIZE=32768 \
  -e LLAMA_ARG_N_PREDICT=32768 \
  -e AGENTIC_PIPELINE="${AGENTIC_PIPELINE:-1}" \
  -e AGENTIC_LLM_BACKEND="${AGENTIC_LLM_BACKEND:-auto}" \
  -v "$PROJECT_ROOT/models":/home/developer/models \
  -v "$PROJECT_ROOT/workspace":/home/developer/workspace \
  icd-c-code-refactorer:llama.cpp
