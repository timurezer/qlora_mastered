# Build the 3-layer bf16 HF test model for DeepSeek-V4-Flash-Base.
# node1 has no bf16 Base (the Oct full dequant lives on node2, which is down), so:
#   1. slice the FP8 Base to 3 layers (CPU only, utils/create_small_ds_sharded.py)
#   2. dequantize FP8 -> bf16 (CPU only, MindSpeed-LLM tests/tools/ckpt_dequant)
# The torchao_stub on PYTHONPATH satisfies an import that the bfloat16 path never calls.
# Run from anywhere; paths are rooted at the repo:
#   nohup bash scripts_qlora/prepare_3layer_base_hf.sh > scripts_qlora/logs/log_prepare_3layer.txt 2>&1 &
set -e
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$ROOT/scripts_qlora/logs" "$ROOT/models_ds"

FP8_BASE=${FP8_BASE:-/data/models/DeepSeek-V4-Flash-Base}
FP8_SMALL=${FP8_SMALL:-$ROOT/models_ds/DeepSeek-V4-Flash-Base-fp8-3layer}
OUT=${OUT:-$ROOT/models_ds/DeepSeek-V4-Flash-base-bf16-3layer}

if [ ! -e "$FP8_SMALL" ]; then
  python "$ROOT/utils/create_small_ds_sharded.py" \
      --input_dir "$FP8_BASE" \
      --n_layers 3 \
      --output_dir "$FP8_SMALL" \
      --use_comp_ratios
else
  echo "skip slice: $FP8_SMALL exists"
fi

if [ -e "$OUT" ]; then echo "ABORT: $OUT exists"; exit 1; fi
# MindSpeed-LLM first: site-packages contains a "tests" package that would shadow tests.tools.ckpt_dequant.
export PYTHONPATH="$ROOT/MindSpeed-LLM:$ROOT/utils/torchao_stub:${PYTHONPATH:-}"
python "$ROOT/MindSpeed-LLM/tests/tools/ckpt_dequant/deepseekv4_ckpt_dequant.py" \
    --input_fp8_hf_path "$FP8_SMALL" \
    --output_hf_path "$OUT" \
    --quant_type bfloat16

python - "$OUT" <<'EOF'
import json, sys
c = json.load(open(f"{sys.argv[1]}/config.json"))
print("num_hidden_layers:", c["num_hidden_layers"])
print("compress_ratios:", c["compress_ratios"])
assert c["num_hidden_layers"] == 3
assert c["compress_ratios"][:3] == [0, 0, 4], "compressor/indexer layer missing"
print("OK")
EOF
echo "prepare_3layer_base_hf done"
