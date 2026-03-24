#!/bin/bash

set -e

cd $HOME

python3 hf_download.py

echo set-option -g default-shell /bin/bash >> .tmux.conf
tmux new -s llama-server -d
tmux rename-window -t llama-server $HF_MODEL
tmux send-keys -t llama-server 'cd /app; ./llama-server --prio 0 --n-gpu-layers ${LLAMA_ARG_N_GPU_LAYERS:-9999} --split-mode ${LLAMA_ARG_SPLIT_MODE:-none} --temp $LLAMA_SAMPLING_TEMPERATURE --min-p $LLAMA_SAMPLING_MIN_P --top-p $LLAMA_SAMPLING_TOP_P --top-k $LLAMA_SAMPLING_TOP_K --repeat-penalty $LLAMA_SAMPLING_REPETITION_PENALTY --chat-template-file $HF_CHAT_TEMPLATE'  C-m
tmux split-window -h -t llama-server
tmux send-keys -t llama-server 'litellm --model $ANTHROPIC_MODEL --temperature $LLAMA_SAMPLING_TEMPERATURE --drop_params' C-m
tmux split-window -v -t llama-server
tmux send-keys -t llama-server 'cd /home/developer/webapp && python3 -m uvicorn app:app --host 0.0.0.0 --port 8081' C-m
tmux select-layout tiled
echo 'Loading model (waiting for llama-server health)...'
READY=0
for i in $(seq 1 60); do
  if curl -s -H "Authorization: Bearer $OPENAI_API_KEY" "http://${LLAMA_ARG_HOST}:${LLAMA_ARG_PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done

if [ "$READY" -ne 1 ]; then
  echo 'ERROR: llama-server did not become ready in time.'
  echo 'Last llama-server pane output:'
  tmux capture-pane -t llama-server:0.0 -p -S -120 || true
  exit 1
fi

/bin/bash

tmux send-keys -t llama-server C-c
echo 'Shutting down llama-server ...'
sleep 2
tmux kill-session -t llama-server
