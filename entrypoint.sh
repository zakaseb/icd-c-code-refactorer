#!/bin/bash

set -e

cd $HOME

python3 hf_download.py

echo set-option -g default-shell /bin/bash >> .tmux.conf
tmux new -s llama-server -d
tmux rename-window -t llama-server $HF_MODEL
tmux send-keys -t llama-server 'cd /app; ./llama-server --prio 3 --temp $LLAMA_SAMPLING_TEMPERATURE --min-p $LLAMA_SAMPLING_MIN_P --top-p $LLAMA_SAMPLING_TOP_P --top-k $LLAMA_SAMPLING_TOP_K --repeat-penalty $LLAMA_SAMPLING_REPETITION_PENALTY --chat-template-file $HF_CHAT_TEMPLATE'  C-m
tmux split-window -h -t llama-server
tmux send-keys -t llama-server 'litellm --model $ANTHROPIC_MODEL --temperature $LLAMA_SAMPLING_TEMPERATURE --drop_params' C-m
tmux split-window -v -t llama-server
tmux send-keys -t llama-server 'cd /home/developer/webapp && python3 -m uvicorn app:app --host 0.0.0.0 --port 8080' C-m
tmux select-layout tiled
echo 'Loading model (wait 15s for llama-server to be ready)...'
sleep 15

/bin/bash

tmux send-keys -t llama-server C-c
echo 'Shutting down llama-server ...'
sleep 2
tmux kill-session -t llama-server
