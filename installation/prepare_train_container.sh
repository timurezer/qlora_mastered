#!/bin/bash
# Install the dependencies of the run-D pipeline.
# Clone MindSpeed-LLM (ds_qlora_support) into the project root before running
# this script; LoraAdapterTransformer, MindSpeed, Megatron-LM and ops-transformer
# are cloned below, each skipped if its directory already exists.
# Run INSIDE the training container, with the CANN environment loaded
# (source /usr/local/Ascend/ascend-toolkit/set_env.sh, or use bash -lc):
#   bash installation/prepare_train_container.sh
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

# TianChen library for the mcore-to-HF LoRA conversion (not in the git repo;
# only its examples/mindspeed-dsv4-flash converter is used).
if [ ! -d LoraAdapterTransformer ]; then
  git clone https://github.com/SkychenLee/LoraAdapterTransformer.git
  git -C LoraAdapterTransformer checkout af1bef0
fi

# MindSpeed
if [ ! -d MindSpeed ]; then
  git clone https://gitcode.com/Ascend/MindSpeed.git
  git -C MindSpeed checkout a15490cd
fi
pip install -r MindSpeed/requirements.txt
pip install -e ./MindSpeed

# Megatron core for MindSpeed-LLM (megatron/ is git-ignored there), plus the
# variable-length pipeline patch required by NOPAD=1.
if [ ! -d MindSpeed-LLM/megatron ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git
  git -C Megatron-LM checkout core_v0.12.1
  cp -r Megatron-LM/megatron MindSpeed-LLM/
  rm -rf Megatron-LM
fi
python MindSpeed-LLM/tools/megatron_patches/p2p_variable_shapes_mhc/apply_p2p_variable_shapes_mhc.py

# MindSpeed-LLM Python requirements (includes bitsandbytes-npu-beta).
pip install -r MindSpeed-LLM/requirements.txt
# pyarrow 26 (pulled by datasets 5.x) requires numpy>=2 and breaks parquet on
# numpy 1.26; pin the pair proven on the reference containers.
pip install pyarrow==25.0.1

# Other packages. torchao is required by the FP8 dequantization;
# msmodelslim and openpyxl are only used by the optional utils/quantization tooling.
pip install torchao
pip install msmodelslim
pip install openpyxl

# very important otherwise in newer versions only shared experts are trained
pip install peft==0.20.0

echo
echo "Install OK. Verify with: bash scripts_qlora/check_training_setup.sh"
