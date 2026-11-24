#!/bin/bash
# Export a trained mcore QLoRA adapter to a PEFT (vLLM-format) HF adapter, CPU only.
# Uses the LoraAdapterTransformer converter from the qlora_clear tree — master's own
# --save-lora-to-hf rejects grouped-GEMM experts (convert_mg2hf.py:1435) and its _merge_lora
# only merges into base weights, which is meaningless for the packed nf4 checkpoint.
# Sizes (TP/PP/EP, rank, experts, layers) are inferred from the checkpoint args.
# Examples (inside th_va30_qlora_train):
#   bash scripts_qlora/export_adapter_mcore2hf.sh \
#     RUN=deepseek4_flash_base_3layer_ep4_pp1_qlora_smoke IT=5 \
#     OUT=deepseek4_flash_base_3layer_ep4_pp1_qlora_smoke_hf \
#     BASE_NAME=models_ds/DeepSeek-V4-Flash-base-bf16-3layer
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

RUN=${RUN:?mcore adapter dir name under models_ds}
IT=${IT:?iteration to export}
OUT=${OUT:?output adapter dir name under models_ds}
BASE_NAME=${BASE_NAME:?value for base_model_name_or_path in adapter_config.json}
CONVERTER=${CONVERTER:-/data/lab-shared/timur/qlora_tests/qlora_clear/LoraAdapterTransformer/examples/mindspeed-dsv4-flash/convert_mindspeed_dsv4_lora_to_vllm.py}

SRC_DIR=$ROOT/models_ds/$RUN
OUT_DIR=$ROOT/models_ds/$OUT
if [ -e "$OUT_DIR" ]; then echo "ABORT: $OUT_DIR exists"; exit 1; fi

export PYTHONPATH="$ROOT/MindSpeed-LLM:${PYTHONPATH:-}"

python "$CONVERTER" \
    --input-dir "$SRC_DIR" \
    --iteration "$IT" \
    --output-dir "$OUT_DIR" \
    --base-model "$BASE_NAME" \
    --dtype bfloat16

echo "Exported adapter: $OUT_DIR"
ls -la "$OUT_DIR"
