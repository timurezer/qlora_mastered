#!/bin/bash
# Serve a DeepSeek-V4 w8a8 checkpoint as model "ds".
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/data/models/DeepSeek-V4-Flash-Base-w8a8}
PORT=${PORT:-8061}
BATCH_SIZE=${BATCH_SIZE:-64}
# -1 = auto-fit max_model_len to the available KV cache (vllm >= 0.30);
# a fixed 32768 does not fit in this build (needs 17.25 GiB KV, ~16.4 available).
MAX_MODEL_LEN=${MAX_MODEL_LEN:--1}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
# FULL_DECODE_ONLY graph capture needs the MoE-LoRA patched vllm-ascend;
# without it capture dies with "stream is captured". Use NONE to fall back to eager.
COMPILATION_CONFIG=${COMPILATION_CONFIG:-'{"cudagraph_mode":"FULL_DECODE_ONLY"}'}
ADDITIONAL_CONFIG=${ADDITIONAL_CONFIG:-'{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":"true","multistream_overlap_shared_expert":false}'}

# Env from the reference LoRA server (run_ds_lora.sh); VLLM_ASCEND_LORA_MOE_OVERLAP
# is required for MoE + LoRA with graph capture.
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_LORA_MOE_OVERLAP=1
export VLLM_DISABLE_COMPILE_CACHE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024

vllm serve "$MODEL_PATH" \
    --served-model-name ds \
    --safetensors-load-strategy prefetch \
    --tensor-parallel-size 8 \
    --data-parallel-size 1 \
    --enable-expert-parallel \
    --quantization ascend \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$BATCH_SIZE" \
    --gpu-memory-utilization 0.92 \
    --block-size 32 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --port "$PORT" \
    --tokenizer-mode deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --async-scheduling \
    --compilation-config "$COMPILATION_CONFIG" \
    --additional-config "$ADDITIONAL_CONFIG" \
    --default-chat-template-kwargs '{"thinking": false}' \
    "$@"
