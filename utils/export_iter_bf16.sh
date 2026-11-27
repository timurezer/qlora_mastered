#!/bin/bash
# Export one mcore QLoRA iteration to an unrotated HF/PEFT adapter.
# The adapter must be QuaRot-rotated before serving on Base-w8a8.
# CPU only; run inside th_va30_qlora_train from any directory:
#   RUN=<mcore run name> IT=221 OUT=<adapter name> bash utils/export_iter_bf16.sh
set -euo pipefail

if [ -z "${RUN:-}" ] || [ -z "${IT:-}" ] || [ -z "${OUT:-}" ]; then
    echo "Usage: RUN=<mcore run name> IT=<iteration> OUT=<adapter name> bash $0"
    exit 1
fi


ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODELS_DIR=$ROOT/models_ds
# Recorded as base_model_name_or_path in adapter_config.json (a name only, the
# converter never reads tensors from it).
BASE=${BASE:-models_ds/DeepSeek-V4-Flash-Base-bf16}
SRC_DIR=$MODELS_DIR/$RUN
OUT_DIR=$MODELS_DIR/$OUT
CONVERTER=$ROOT/LoraAdapterTransformer/examples/mindspeed-dsv4-flash/convert_mindspeed_dsv4_lora_to_vllm.py

export PYTHONPATH="${PYTHONPATH:-}:$ROOT/MindSpeed-LLM"

python "$CONVERTER" \
    --input-dir "$SRC_DIR" \
    --iteration "$IT" \
    --output-dir "$OUT_DIR" \
    --base-model "$BASE" \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --expert-parallel-size 4 \
    --num-experts 256 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --dtype bfloat16

echo "Exported unrotated adapter: $OUT_DIR"
echo "Rotate it with scripts_qlora/rotate_adapter.sh before serving."
