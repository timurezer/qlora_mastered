#!/bin/bash
# Serve Base-w8a8 with a QuaRot-rotated LoRA adapter as "lora-adapter".
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LORA_PATH=${LORA_PATH:-$SCRIPT_DIR/../models_ds/deepseek4_flash_base_bf16_teacher_hf_quarot}

exec bash "$SCRIPT_DIR/serve_ds_w8a8.sh" \
    --enable-lora \
    --max-loras 1 \
    --max-lora-rank 16 \
    --lora-modules "lora-adapter=$LORA_PATH"
