# DeepSeek-V4-Flash-Base 3-layer -> mcore nf4 (QLoRA), EP4 / PP1 / TP1, real compress_ratios.
# PP1 variant for the 4-card smoke run: all hash layers (num_hash_layers=3) must live on one stage.
# Run inside th_va30_qlora_train:
#   nohup bash scripts_qlora/ckpt_convert_deepseek4_flash_base_ep4_pp1_qlora_3layer.sh > scripts_qlora/logs/log_convert_3layer_ep4pp1.txt 2>&1 &
set -e
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$ROOT/scripts_qlora/logs"

# The bnb NPU quantizer keeps large transients reserved; expandable segments return them
# between tensors (EP2 doubles the stacked expert tensor vs EP4 and OOMed without this).
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}

LOAD_DIR=${LOAD_DIR:-$ROOT/models_ds/DeepSeek-V4-Flash-base-bf16-3layer}
SAVE_DIR=${SAVE_DIR:-$ROOT/models_ds/deepseek4_flash_base_3layer_ep4_pp1_nf4_mcore_comp}
if [ -e "$SAVE_DIR" ]; then echo "ABORT: $SAVE_DIR exists"; exit 1; fi

python "$ROOT/MindSpeed-LLM/convert_ckpt_v2.py" \
  --load-model-type hf \
  --save-model-type mg \
  --model-type-hf deepseek4 \
  --load-dir $LOAD_DIR \
  --save-dir $SAVE_DIR/ \
  --target-tensor-parallel-size 1 \
  --target-pipeline-parallel-size 1 \
  --target-expert-parallel-size 4 \
  --moe-grouped-gemm \
  --noop-layers 3 \
  --qlora-nf4
echo "convert exit $?"

python "$ROOT/utils/check_ckpt_comp.py" "$SAVE_DIR"
