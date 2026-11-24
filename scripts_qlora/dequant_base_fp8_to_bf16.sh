# DeepSeek-V4-Flash-Base is FP8 (e4m3 + 128x128 ue8m0 block scales, experts too); convert_ckpt_v2.py needs bf16.
# Official tool (MindSpeed-LLM tests/tools/ckpt_dequant/deepseekv4_ckpt_dequant.py), CPU only.
# utils/torchao_stub satisfies an import the bfloat16 path never calls; do NOT pip install torchao in the
# shared container. MindSpeed-LLM goes first on PYTHONPATH because site-packages has a "tests" package
# that shadows tests.tools.ckpt_dequant.
# Run inside th_va30_qlora_train:
#   nohup bash scripts_qlora/dequant_base_fp8_to_bf16.sh > scripts_qlora/logs/log_dequant.txt 2>&1 &
# Output ~570 GB. The Base config.json already has the real compress_ratios (44 = 43 layers + 1 mtp, equal to
# 0731's config_comp.json), so no config swapping is needed.

set -e
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FP8_BASE=${FP8_BASE:-/data/models/DeepSeek-V4-Flash-Base}
OUT=${OUT:-$ROOT/models_ds/DeepSeek-V4-Flash-Base-bf16}
if [ -e "$OUT" ]; then echo "ABORT: $OUT exists"; exit 1; fi
mkdir -p $ROOT/models_ds
mkdir -p $OUT

python -c "import torchao" 2>/dev/null || pip install torchao

cd "$ROOT/MindSpeed-LLM"
python tests/tools/ckpt_dequant/deepseekv4_ckpt_dequant.py \
    --input_fp8_hf_path "$FP8_BASE" \
    --output_hf_path "$OUT" \
    --quant_type bfloat16
echo "dequant exit $?"
ls "$OUT" | grep -c safetensors
