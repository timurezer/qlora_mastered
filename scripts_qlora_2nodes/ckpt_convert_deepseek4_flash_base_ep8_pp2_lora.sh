# DeepSeek-V4-Flash-Base -> mcore bf16 (ordinary LoRA), EP8 / PP2 / TP1 — target layout
# for the 2-node (16-rank) LoRA runs. Identical to the _qlora.sh conversion but WITHOUT
# --qlora-nf4 (stock bf16 path); CPU-only, no NPU needed. Produces 16 rank files.
# NOTE: the output is ~550 GB; check disk space first.
# Run inside the training container, detached:
#   nohup bash scripts_qlora_2nodes/ckpt_convert_deepseek4_flash_base_ep8_pp2_lora.sh > scripts_qlora_2nodes/logs/log_convert_full_ep8pp2_bf16.txt 2>&1 &
set -e
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$ROOT/scripts_qlora_2nodes/logs"

export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}

LOAD_DIR=${LOAD_DIR:-$ROOT/models_ds/DeepSeek-V4-Flash-Base-bf16}
SAVE_DIR=${SAVE_DIR:-$ROOT/models_ds/deepseek4_flash_base_full_ep8_pp2_bf16_mcore_comp}
if [ -e "$SAVE_DIR" ]; then echo "ABORT: $SAVE_DIR exists"; exit 1; fi

python "$ROOT/MindSpeed-LLM/convert_ckpt_v2.py" \
  --load-model-type hf \
  --save-model-type mg \
  --model-type-hf deepseek4 \
  --load-dir $LOAD_DIR \
  --save-dir $SAVE_DIR/ \
  --target-tensor-parallel-size 1 \
  --target-pipeline-parallel-size 2 \
  --target-expert-parallel-size 8 \
  --moe-grouped-gemm \
  --noop-layers 43
echo "convert exit $?"

# bf16 mode: compressor / indexer keys present, NO uint8 (nf4) tensors.
python "$ROOT/utils/check_ckpt_comp.py" "$SAVE_DIR" --bf16
