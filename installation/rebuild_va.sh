set -euo pipefail

VLLM_PATH=/vllm-workspace/vllm/vllm
VLLM_ASCEND_PATH=/vllm-workspace/vllm-ascend/vllm_ascend

cd $VLLM_ASCEND_PATH/..
rm -rf build
COMPILE_CUSTOM_KERNELS=1 SOC_VERSION="${SOC_VERSION:-ascend910b3}" MAX_JOBS="${MAX_JOBS:-32}" \
  python setup.py build_ext --inplace
python -c "
import torch, torch_npu, vllm_ascend.vllm_ascend_C
for op in ('add_lora', 'add_lora_shrink', 'add_lora_expand'):
    print(f'{op:18s} present:', hasattr(torch.ops._C_ascend, op))
"