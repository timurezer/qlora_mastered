# DeepSeek-V4-Flash-Base -> mcore nf4 (QLoRA), EP4 / PP2 / TP1, real compress_ratios.
# Same as ../ckpt_convert_deepseek4_flash_0731_ep4_pp2_full_qlora_comp.sh, except:
#   * input is the bf16 dequant of Base (dequant_base_fp8_to_bf16.sh); its config.json already carries the real
#     compress_ratios (44 = 43 + 1 mtp, identical to 0731's config_comp.json), so no config swap in /data/models;
#   * key check at the end: compressor / indexer weights must be present.
# Run inside lab-th_va25_qlora:
#   nohup bash ckpt_convert_deepseek4_flash_base_ep4_pp2_full_qlora_comp.sh > logs/log_convert.txt 2>&1 &
set -e
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEVICES=${DEVICES:-4,5,6,7}

export ASCEND_RT_VISIBLE_DEVICES=$DEVICES
export NPU_VISIBLE_DEVICES=$DEVICES

LOAD_DIR=${LOAD_DIR:-$ROOT/models_ds/DeepSeek-V4-Flash-base-bf16-3layer}
SAVE_DIR=${SAVE_DIR:-$ROOT/models_ds/deepseek4_flash_base_3layer_full_ep4_pp2_nf4_mcore_comp}
if [ -e "$SAVE_DIR" ]; then echo "ABORT: $SAVE_DIR exists"; exit 1; fi

python "$ROOT/MindSpeed-LLM/convert_ckpt_v2.py" \
  --load-model-type hf \
  --save-model-type mg \
  --model-type-hf deepseek4 \
  --load-dir $LOAD_DIR \
  --save-dir $SAVE_DIR/ \
  --target-tensor-parallel-size 1 \
  --target-pipeline-parallel-size 2 \
  --target-expert-parallel-size 4 \
  --moe-grouped-gemm \
  --noop-layers 3 \
  --qlora-nf4
echo "convert exit $?"

python "$ROOT/utils/check_ckpt_comp.py" "$SAVE_DIR"
