#!/bin/bash
# Full run-D-style pipeline, inside th_va30_qlora_train on node1:
# data prep -> FP8 Base dequant -> HF->mcore NF4 conversion -> 1-epoch QLoRA
# training -> mcore->HF adapter export -> QuaRot rotation for vLLM serving.
# Each step refuses to overwrite existing output; move directories aside to rerun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# data and checkpoint preparation
bash ../utils/prep_shuffled_distill_ds.sh
bash dequant_base_fp8_to_bf16.sh
bash ckpt_convert_deepseek4_flash_base_ep4_pp2_qlora.sh

# training, export to HF, QuaRot rotation for vLLM serving
bash qlora_finetune_deepseek4_flash_base_distill_1epoch.sh
bash export_mcore2hf.sh
bash rotate_adapter.sh
