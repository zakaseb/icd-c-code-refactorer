FROM ghcr.io/ggml-org/llama.cpp:server-cuda

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       vim git curl tmux python3 python3-pip python3-venv sudo \
       build-essential cmake \
       gcc-arm-none-eabi libnewlib-arm-none-eabi libnewlib-dev \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

# PEP 668: Ubuntu's system Python is externally managed — install app deps
# into a venv and put it first on PATH so entrypoint/hf_download/uvicorn see
# huggingface_hub and the rest without needing an explicit activate.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -U pip \
    && /opt/venv/bin/pip install --no-cache-dir -U huggingface_hub hf_transfer 'litellm[proxy]' \
       fastapi 'uvicorn[standard]' httpx PyMuPDF python-multipart

ENV PATH="/opt/venv/bin:${PATH}"

RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

RUN if getent group $GROUP_ID > /dev/null; then \
        EXISTING_GROUP=$(getent group $GROUP_ID | cut -d: -f1); \
        if [ "$EXISTING_GROUP" != "developer" ]; then \
            groupmod -n developer $EXISTING_GROUP; \
        fi; \
    else \
        groupadd -g $GROUP_ID developer; \
    fi \
    && useradd -u $USER_ID -g developer -m developer || (EXISTING_USER=$(getent passwd $USER_ID | cut -d: -f1) && usermod -l developer -d /home/developer -m $EXISTING_USER && usermod -g developer developer) \
    && echo developer ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/developer \
    && chmod 0440 /etc/sudoers.d/developer \
    && usermod -a -G video developer

RUN chown -R developer:developer /opt/venv

ENV HF_CHAT_TEMPLATE='/app/Qwen3-Coder-Next-UD-Q4_K_XL.jinja'
ADD --chown=developer:developer https://huggingface.co/unsloth/Qwen3-Coder-Next/raw/main/chat_template.jinja $HF_CHAT_TEMPLATE

USER developer

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_REPO_ID="unsloth/Qwen3-Coder-Next-GGUF"
ENV HF_MODEL="Qwen3-Coder-Next-UD-Q4_K_XL.gguf"

ENV LLAMA_ARG_HOST=0.0.0.0
ENV LLAMA_ARG_PORT=24000
ENV LLAMA_ARG_N_PARALLEL=1
ENV LLAMA_API_KEY=sk-1234-miaw
ENV LLAMA_ARG_JINJA=1
ENV LLAMA_ARG_THREADS=-1
ENV LLAMA_LOG_COLORS=1
ENV LLAMA_LOG_PREFIX=1

ENV LLAMA_ARG_MODEL="/home/developer/models/$HF_REPO_ID/$HF_MODEL"
ENV LLAMA_ARG_CTX_SIZE=98274
ENV LLAMA_ARG_N_PREDICT=$LLAMA_ARG_CTX_SIZE
# GPU offload policy:
# - "auto" lets llama.cpp fit offload against current free VRAM + context/KV footprint.
# - This avoids startup OOMs on laptops where free VRAM fluctuates.
ENV LLAMA_ARG_N_GPU_LAYERS=auto
ENV LLAMA_ARG_NO_CONTEXT_SHIFT=1
# Prefer GPU flash-attention when the backend supports it (CUDA).
ENV LLAMA_ARG_FLASH_ATTN=on
ENV LLAMA_ARG_CACHE_TYPE_K=q8_0
ENV LLAMA_ARG_CACHE_TYPE_V=q8_0
# Single-GPU default: 'none' uses one device. 'row' is for multi-GPU tensor parallel and can
# prevent offload when only one GPU is present.
ENV LLAMA_ARG_SPLIT_MODE=none
ENV LLAMA_SAMPLING_TEMPERATURE=0.7
ENV LLAMA_SAMPLING_MIN_P=0
ENV LLAMA_SAMPLING_TOP_P=0.80
ENV LLAMA_SAMPLING_TOP_K=20
ENV LLAMA_SAMPLING_REPETITION_PENALTY=1.05

ENV OPENAI_API_KEY="$LLAMA_API_KEY"
ENV OPENAI_BASE_URL="http://${LLAMA_ARG_HOST}:${LLAMA_ARG_PORT}"

ENV ANTHROPIC_BASE_URL="http://${LLAMA_ARG_HOST}:4000"
ENV ANTHROPIC_AUTH_TOKEN="$OPENAI_API_KEY"
ENV ANTHROPIC_MODEL="openai/$HF_MODEL"
ENV ANTHROPIC_SMALL_FAST_MODEL="openai/$HF_MODEL"
ENV CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

# Sandbox build/debug: agentic state-machine loop by default (api/agentic_debug.py).
# Set SANDBOX_USE_AGENTIC=0 to use the ReAct orchestrator instead.
ENV SANDBOX_USE_AGENTIC=1

COPY --chown=developer:developer deployments/docker/entrypoint.sh /home/developer/entrypoint.sh
COPY --chown=developer:developer src/utils/hf_download.py /home/developer/hf_download.py
COPY --chown=developer:developer api /home/developer/webapp

ENTRYPOINT [ "/home/developer/entrypoint.sh" ]
