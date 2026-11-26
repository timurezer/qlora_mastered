# DeepSeek-V4-Flash-Base -> mcore nf4 (QLoRA), EP8 / PP2 / TP1 — target layout for the
# 2-node (16-rank) runs. Single-process tool: runs on ONE node (no 16 GPUs needed; the
# bnb NPU quantizer uses the devices in $DEVICES), produces 16 rank files.
# Input is the bf16 dequant of Base (scripts_qlora/dequant_base_fp8_to_bf16.sh); its
# config.json already carries the real compress_ratios, so no config swap is needed.
# Run inside the training container, detached:
#   nohup bash scripts_qlora_2nodes/ckpt_convert_deepseek4_flash_base_ep8_pp2_qlora.sh > scripts_qlora_2nodes/logs/log_convert_full_ep8pp2_nf4.txt 2>&1 &
set -e
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$ROOT/scripts_qlora_2nodes/logs"
DEVICES=${DEVICES:-0,1,2,3}

export ASCEND_RT_VISIBLE_DEVICES=$DEVICES
export NPU_VISIBLE_DEVICES=$DEVICES
# The bnb NPU quantizer keeps large transients reserved; expandable segments return them
# between tensors (without it the EP2 3-layer conversion OOMed at 52 GB reserved).
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}

LOAD_DIR=${LOAD_DIR:-$ROOT/models_ds/DeepSeek-V4-Flash-Base-bf16}
SAVE_DIR=${SAVE_DIR:-$ROOT/models_ds/deepseek4_flash_base_full_ep8_pp2_nf4_mcore_comp}
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
  --noop-layers 43 \
  --qlora-nf4
echo "convert exit $?"

# Expect 16 rank files; compressor / indexer / uint8 (nf4) keys must be present.
python "$ROOT/utils/check_ckpt_comp.py" "$SAVE_DIR"
