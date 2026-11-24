#!/bin/bash
# Prepare a fresh inference container (va30 image): eval dependencies plus the
# MoE-LoRA patched vllm-ascend required for graph capture with LoRA.
# Run INSIDE the inference container (STAGE=infer):
#   bash installation/prepare_eval_container.sh
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# pyarrow<26 is required with the image's numpy 1.26.x.
pip install openpyxl
pip install pyarrow==25.0.1
pip install evalscope

# Switch vllm-ascend to the branch with the MoE-LoRA patch.
cd /vllm-workspace/vllm-ascend
git remote add dg_upstream https://github.com/dghuawei/vllm-ascend.git
git fetch dg_upstream
git checkout dg_upstream/v30_lora_v2

# Rebuild the custom kernels (add_lora ops); takes ~10 minutes.
bash "$SCRIPT_DIR/rebuild_va.sh"

