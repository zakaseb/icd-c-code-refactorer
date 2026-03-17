SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker run -ti --rm --name icd-c-code-refactorer --network=host --gpus all \
  -e LLAMA_ARG_N_GPU_LAYERS=30 \
  -e LLAMA_ARG_CTX_SIZE=32768 \
  -e LLAMA_ARG_N_PREDICT=32768 \
  -v "$SCRIPT_DIR/models":/home/developer/models \
  -v "$SCRIPT_DIR/workspace":/home/developer/workspace \
  icd-c-code-refactorer:llama.cpp
